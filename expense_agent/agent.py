"""The BudgetAgent: a model-based, utility-driven, learning agent.

Agent loop (once per day)
-------------------------
 1. PERCEIVE   receive the day's raw transactions (timestamp, text, amount)
 2. INTERPRET  categorise each one (Naive Bayes) and score it for anomalies
 3. UPDATE     internal state: spend per category, daily series, recent txns
 4. DECIDE     per transaction: fraud action by expected utility
               per category:    budget action by expected utility
 5. ACT        emit Actions (messages, holds, reallocations)
 6. LEARN      - p_act (does this user respond to warnings?)  Beta posterior,
                 updated by *observing* spend before vs after each warning
               - fatigue scale from explicit dismiss / helpful feedback
               - categoriser from category corrections (online NB update)
               - anomaly detector from "this merchant is legitimate" feedback

Three policies share this loop so the evaluation isolates the decision layer:
    policy="utility"  expected-utility policies (ours)
    policy="rule"     fixed-threshold rules (typical budgeting app)
    policy="none"     passive tracker (no actions)
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

from .anomaly import AnomalyDetector, merchant_key
from .categorizer import NaiveBayesCategorizer
from .decision import (Action, RuleBudgetPolicy, ThresholdFraudPolicy, UtilityBudgetPolicy,
                       UtilityFraudPolicy, overspend_stats)
from .forecaster import SpendForecaster


class BudgetAgent:
    def __init__(self, budgets: dict[str, float], categorizer: NaiveBayesCategorizer,
                 detector: AnomalyDetector, forecaster: SpendForecaster, policy: str = "utility",
                 uncertain_threshold: float = 0.60, budget_policy=None, fraud_policy=None):
        self.budgets = dict(budgets)
        self.categorizer = categorizer
        self.detector = detector
        self.forecaster = forecaster
        self.policy = policy
        self.uncertain_threshold = uncertain_threshold
        if policy == "utility":
            self.budget_policy = budget_policy or UtilityBudgetPolicy()
            self.fraud_policy = fraud_policy or UtilityFraudPolicy()
        elif policy == "rule":
            self.budget_policy = budget_policy or RuleBudgetPolicy()
            self.fraud_policy = fraud_policy or ThresholdFraudPolicy()
        else:
            self.budget_policy = None
            self.fraud_policy = None
        # learned user model
        self.p_act_alpha, self.p_act_beta = 3.0, 2.0      # Beta prior, mean 0.6
        self.fatigue_scale = 1.0
        self.records: list[dict] = []                     # all interpreted transactions
        self.learning_log: list[dict] = []
        self.start_month(2026, 1)

    # ---------------------------------------------------------------- state
    @property
    def p_act(self) -> float:
        return self.p_act_alpha / (self.p_act_alpha + self.p_act_beta)

    def start_month(self, year: int, month: int) -> None:
        self.year, self.month = year, month
        self.days_in_month = calendar.monthrange(year, month)[1]
        self.month_budgets = dict(self.budgets)
        self.spent = {c: 0.0 for c in self.budgets}
        self.daily = {c: [0.0] * (self.days_in_month + 1) for c in self.budgets}
        self.recent: list[dict] = []
        self.alerts: list[Action] = []
        self.last_alert: dict[str, tuple[int, float]] = {}
        self.pending_evals: list[dict] = []
        self.month_records: list[dict] = []
        self.held: dict[int, dict] = {}                   # flagged txns awaiting the user's answer
        self.realloc_done: set[str] = set()               # categories re-planned this month
        self.n_asks = 0                                   # category questions asked this month

    def _recent_rate(self, cat: str, day: int, window: int = 7) -> float:
        lo = max(1, day - window + 1)
        return sum(self.daily[cat][lo:day + 1]) / (day - lo + 1)

    def _donors(self, cat: str, day: int) -> list[tuple[str, float]]:
        """Categories almost surely under budget, with their slack."""
        donors = []
        for c in self.budgets:
            if c == cat or c in ("Rent", "Bills") or c in self.realloc_done:
                continue
            mean, sd = self.forecaster.project(c, self.spent[c], day, self.days_in_month, self._recent_rate(c, day))
            p_over, _ = overspend_stats(mean, sd, self.month_budgets[c])
            slack = self.month_budgets[c] - (mean + 1.0 * sd)
            if p_over < 0.10 and slack > 100:
                donors.append((c, slack))
        return donors

    # ---------------------------------------------------------------- loop
    def process_day(self, d: date, transactions: list[dict]) -> list[Action]:
        day = d.day
        actions: list[Action] = []
        cutoff = d - timedelta(days=1)
        self.recent = [r for r in self.recent if r["ts"].date() >= cutoff]

        for t in sorted(transactions, key=lambda x: x["ts"]):
            cat, conf = self.categorizer.predict(t["description"], t["amount"], t["ts"].hour)
            if cat not in self.spent:
                self.spent[cat] = 0.0
                self.daily[cat] = [0.0] * (self.days_in_month + 1)
                self.month_budgets.setdefault(cat, 0.0)
            score, evidence = self.detector.score(t, cat, self.recent)
            rec = {"txn_id": t["txn_id"], "ts": t["ts"], "description": t["description"], "amount": t["amount"],
                   "key": merchant_key(t["description"]), "pred_category": cat, "confidence": conf,
                   "anomaly_score": score, "evidence": evidence, "day": day}
            self.recent.append(rec)
            self.records.append(rec)
            self.month_records.append(rec)
            fa = None
            if self.fraud_policy is not None:
                fa = self.fraud_policy.choose(t, score, AnomalyDetector.explain(evidence), day)
            if fa is not None and fa.kind != "NONE":
                # hold the amount: it only counts as spending once the user confirms it is genuine
                fa.category = cat
                actions.append(fa)
                self.held[t["txn_id"]] = rec
            else:
                self.spent[cat] += t["amount"]
                self.daily[cat][day] += t["amount"]
            if conf < self.uncertain_threshold and self.policy != "none":
                self.n_asks += 1
                actions.append(Action("ASK_CATEGORY", cat,
                                      f"Not sure about '{t['description']}' (Rs {t['amount']:,.0f}): guessed {cat} ({conf:.0%}).",
                                      0.0, {"confidence": conf}, day, t["txn_id"]))

        if self.budget_policy is not None:
            for cat in sorted(self.budgets):
                budget = self.month_budgets[cat]
                mean, sd = self.forecaster.project(cat, self.spent[cat], day, self.days_in_month, self._recent_rate(cat, day))
                recent_alerts = sum(1 for a in self.alerts if a.kind in ("WARN", "INFO") and day - a.day <= 7)
                donors = self._donors(cat, day) if self.policy == "utility" else []
                act = self.budget_policy.choose(cat, budget, self.spent[cat], mean, sd, day, self.days_in_month,
                                                self.p_act, recent_alerts, self.last_alert.get(cat), donors,
                                                self.fatigue_scale, realloc_allowed=cat not in self.realloc_done)
                if act.kind == "NONE":
                    continue
                actions.append(act)
                self.alerts.append(act)
                if act.kind in ("WARN", "INFO"):
                    self.last_alert[cat] = (day, act.details.get("p_over", 1.0))
                    self.pending_evals.append({"category": cat, "day": day, "rate_before": self._recent_rate(cat, day),
                                               "kind": act.kind})
                elif act.kind == "REALLOCATE":
                    for donor, amt in act.details["transfer_plan"]:
                        self.month_budgets[donor] -= amt
                        self.month_budgets[cat] += amt
                        self.realloc_done.add(donor)
                    self.realloc_done.add(cat)

        self._learn_from_outcomes(day)
        return actions

    # ---------------------------------------------------------------- learning
    def _learn_from_outcomes(self, day: int) -> None:
        """Seven days after a warning, compare the spend rate after it with a
        reference rate.  Warnings are triggered by unusually high spending, which
        tends to fall back on its own (regression to the mean), so the reference
        is the geometric mean of the pre-warning rate and the user's normal rate
        rather than the pre-warning rate alone."""
        still = []
        for p in self.pending_evals:
            if day - p["day"] >= 7:
                after = self._recent_rate(p["category"], day)
                normal = self.forecaster.mu0.get(p["category"], 0.0) / self.days_in_month
                reference = (p["rate_before"] * normal) ** 0.5 if p["rate_before"] > 0 and normal > 0 else p["rate_before"]
                responded = after < 0.85 * reference if reference > 0 else False
                if p["kind"] == "WARN":
                    if responded:
                        self.p_act_alpha += 1.0
                    else:
                        self.p_act_beta += 1.0
                    self.learning_log.append({"month": (self.year, self.month), "day": day, "category": p["category"],
                                              "responded": responded, "p_act": round(self.p_act, 3)})
            else:
                still.append(p)
        self.pending_evals = still

    def feedback(self, action: Action, response: str) -> None:
        """response: dismiss | helpful | confirm_fraud | legit | reject_realloc | category:<Name>"""
        if response == "dismiss":
            self.fatigue_scale = min(3.0, self.fatigue_scale * 1.15)
        elif response == "helpful":
            self.fatigue_scale = max(0.5, self.fatigue_scale * 0.92)
        elif response == "legit":
            rec = self.held.pop(action.txn_id, None) or self._record(action.txn_id)
            if rec is not None:
                self.detector.mark_legit(rec["description"])
                self.spent[rec["pred_category"]] += rec["amount"]
                self.daily[rec["pred_category"]][rec["day"]] += rec["amount"]
        elif response == "confirm_fraud":
            # a fraudulent payment is not the user's spending: it stays out of the totals
            rec = self.held.pop(action.txn_id, None)
            if rec is not None:
                rec["confirmed_fraud"] = True
        elif response == "reject_realloc":
            for donor, amt in action.details.get("transfer_plan", []):
                self.month_budgets[donor] += amt
                self.month_budgets[action.category] -= amt
        elif response.startswith("category:"):
            true_cat = response.split(":", 1)[1]
            rec = self._record(action.txn_id)
            if rec is not None:
                self.correct_category(rec, true_cat)

    def correct_category(self, rec: dict, true_cat: str) -> None:
        old = rec["pred_category"]
        if rec["txn_id"] in self.held or rec.get("confirmed_fraud"):
            rec["pred_category"] = true_cat   # not in the totals: only relabel and learn
        elif old != true_cat:
            self.spent[old] -= rec["amount"]
            self.daily[old][rec["day"]] -= rec["amount"]
            self.spent.setdefault(true_cat, 0.0)
            self.daily.setdefault(true_cat, [0.0] * (self.days_in_month + 1))
            self.spent[true_cat] += rec["amount"]
            self.daily[true_cat][rec["day"]] += rec["amount"]
            rec["pred_category"] = true_cat
        # online learning with extra weight: a human label is worth more than one noisy example
        self.categorizer.update(rec["description"], rec["amount"], rec["ts"].hour, true_cat, weight=3.0)

    def _record(self, txn_id):
        for r in reversed(self.records):
            if r["txn_id"] == txn_id:
                return r
        return None

    # ---------------------------------------------------------------- reporting
    def month_summary(self) -> dict:
        """Overspend is always measured against the ORIGINAL budgets, so a
        reallocation cannot flatter the agent; the adjusted-budget figure and the
        net total are reported alongside."""
        rows = {}
        for c in self.budgets:
            b0 = self.budgets[c]
            b1 = self.month_budgets.get(c, b0)
            s = self.spent.get(c, 0.0)
            rows[c] = {"budget": b0, "adjusted_budget": b1, "spent": round(s, 2),
                       "overspend": round(max(s - b0, 0.0), 2), "overspend_adjusted": round(max(s - b1, 0.0), 2)}
        total_spent = sum(r["spent"] for r in rows.values())
        total_budget = sum(self.budgets.values())
        return {"year": self.year, "month": self.month, "categories": rows,
                "total_overspend": round(sum(r["overspend"] for r in rows.values()), 2),
                "total_overspend_adjusted": round(sum(r["overspend_adjusted"] for r in rows.values()), 2),
                "net_total_overspend": round(max(total_spent - total_budget, 0.0), 2),
                "n_alerts": sum(1 for a in self.alerts if a.kind in ("WARN", "INFO")),
                "n_realloc": sum(1 for a in self.alerts if a.kind == "REALLOCATE"),
                "n_asks": self.n_asks,
                "p_act": round(self.p_act, 3), "fatigue_scale": round(self.fatigue_scale, 2)}
