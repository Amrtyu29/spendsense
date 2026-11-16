"""Simulated user and month runner used for evaluation and the demo.

The simulated user has hidden traits the agent must learn about:
* p_respond   probability of cutting spending after a WARN (INFO is weaker)
* annoyance   after more than `annoyance_threshold` alerts in 7 days the user
              mostly ignores alerts and starts dismissing them
* accepts budget reallocations most of the time
* answers fraud verifications truthfully
* answers "which category?" questions most of the time, and occasionally
  notices and corrects a mis-categorised transaction on their own
"""
from __future__ import annotations

import calendar
from datetime import date

import numpy as np

from .agent import BudgetAgent
from .data_gen import TransactionSimulator, public_view
from .decision import Action


class SimulatedUser:
    def __init__(self, seed: int = 0, p_respond: float = 0.65, annoyance_threshold: int = 3,
                 p_accept_realloc: float = 0.8, p_answer_category: float = 0.7, p_spot_error: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.p_respond = p_respond
        self.annoyance_threshold = annoyance_threshold
        self.p_accept_realloc = p_accept_realloc
        self.p_answer_category = p_answer_category
        self.p_spot_error = p_spot_error
        self.active: dict[str, tuple[float, int]] = {}   # category -> (multiplier, until_day)
        self.alert_days: list[int] = []
        self.stats = {"dismissed": 0, "responded": 0, "ignored": 0, "realloc_accepted": 0,
                      "realloc_rejected": 0, "fraud_confirmed": 0, "false_flags": 0, "category_answers": 0}

    def start_month(self) -> None:
        self.active = {}
        self.alert_days = []

    def multipliers(self, day: int) -> dict[str, float]:
        return {c: m for c, (m, until) in self.active.items() if day <= until}

    def react(self, actions: list[Action], day: int, agent: BudgetAgent, truth: dict[int, dict]) -> None:
        for a in actions:
            if a.kind in ("WARN", "INFO"):
                self.alert_days.append(day)
                recent = sum(1 for d in self.alert_days if day - d <= 7)
                annoyed = recent > self.annoyance_threshold
                p = self.p_respond * (0.3 if annoyed else 1.0) * (1.0 if a.kind == "WARN" else 0.45)
                if self.rng.random() < p:
                    self.active[a.category] = (0.70 if a.kind == "WARN" else 0.85, day + 7)
                    self.stats["responded"] += 1
                    if self.rng.random() < 0.5:
                        agent.feedback(a, "helpful")
                else:
                    self.stats["ignored"] += 1
                    if annoyed or self.rng.random() < 0.25:
                        self.stats["dismissed"] += 1
                        agent.feedback(a, "dismiss")
            elif a.kind == "REALLOCATE":
                if self.rng.random() < self.p_accept_realloc:
                    self.stats["realloc_accepted"] += 1
                else:
                    self.stats["realloc_rejected"] += 1
                    agent.feedback(a, "reject_realloc")
            elif a.kind in ("VERIFY", "BLOCK"):
                t = truth.get(a.txn_id)
                if t is not None and t["is_anomaly"]:
                    self.stats["fraud_confirmed"] += 1
                    agent.feedback(a, "confirm_fraud")
                else:
                    self.stats["false_flags"] += 1
                    agent.feedback(a, "legit")
            elif a.kind == "ASK_CATEGORY":
                t = truth.get(a.txn_id)
                if t is not None and self.rng.random() < self.p_answer_category:
                    self.stats["category_answers"] += 1
                    agent.feedback(a, f"category:{t['category']}")
        # occasionally the user browses the app and fixes a wrong label
        for rec in agent.month_records[-15:]:
            t = truth.get(rec["txn_id"])
            if t is not None and rec["pred_category"] != t["category"] and self.rng.random() < self.p_spot_error:
                agent.correct_category(rec, t["category"])


def run_month(sim: TransactionSimulator, agent: BudgetAgent, user: SimulatedUser | None, year: int, month: int,
              month_idx: int, verbose: bool = False, log=None) -> dict:
    """Run one month of the agent loop.  Returns the month summary with extra
    evaluation fields (counterfactual spend, anomaly detection counts)."""
    agent.start_month(year, month)
    if user is not None:
        user.start_month()
    truth: dict[int, dict] = {}
    base_spent = {c: 0.0 for c in agent.budgets}
    flagged_ids: set[int] = set()
    n_anomalous = 0
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        d = date(year, month, day)
        mult = user.multipliers(day) if user is not None else {}
        txns = sim.generate_day(d, mult, month_idx)
        for t in sim.generate_day(d, {}, month_idx):          # counterfactual: no behaviour change
            if not t["is_anomaly"]:
                base_spent[t["category"]] += t["amount"]
        for t in txns:
            truth[t["txn_id"]] = t
            n_anomalous += int(t["is_anomaly"])
        actions = agent.process_day(d, [public_view(t) for t in txns])
        for a in actions:
            if a.kind in ("VERIFY", "BLOCK"):
                flagged_ids.add(a.txn_id)
        if verbose:
            _print_day(d, txns, agent, actions, log)
        if user is not None:
            user.react(actions, day, agent, truth)
    summary = agent.month_summary()
    tp = sum(1 for i in flagged_ids if truth[i]["is_anomaly"])
    summary.update({
        "counterfactual_overspend": round(sum(max(base_spent[c] - agent.budgets[c], 0.0) for c in agent.budgets), 2),
        "counterfactual_net_total_overspend": round(max(sum(base_spent.values()) - sum(agent.budgets.values()), 0.0), 2),
        "counterfactual_spent": {c: round(v, 2) for c, v in base_spent.items()},
        "anomalies_present": n_anomalous, "anomalies_flagged_tp": tp, "flags_fp": len(flagged_ids) - tp,
        "true_spent": {c: round(sum(t["amount"] for t in truth.values() if t["category"] == c and not t["is_anomaly"]), 2)
                       for c in agent.budgets},
    })
    return summary


def _emit(log, text: str) -> None:
    print(text)
    if log is not None:
        log.write(text + "\n")


def _print_day(d: date, txns: list[dict], agent: BudgetAgent, actions: list[Action], log) -> None:
    _emit(log, f"\n--- {d:%a %d %b %Y} ---")
    recs = {r["txn_id"]: r for r in agent.month_records}
    for t in txns:
        r = recs[t["txn_id"]]
        flag = ""
        if r["anomaly_score"] >= 0.3:
            flag = f"  <risk {r['anomaly_score']:.0%}>"
        mark = "" if r["pred_category"] == t["category"] else f"  (true: {t['category']})"
        _emit(log, f"  {t['ts']:%H:%M}  Rs {t['amount']:>7,.0f}  {t['description'][:38]:<38} -> {r['pred_category']:<14}"
                   f"{r['confidence']:>4.0%}{flag}{mark}")
    for a in actions:
        _emit(log, f"  >> {a}")
        if a.kind in ("WARN", "INFO", "REALLOCATE") and "utilities" in a.details:
            u = a.details["utilities"]
            _emit(log, f"     utilities: " + ", ".join(f"{k}={v}" for k, v in u.items()) + f"  (p_act={a.details['p_act']})")
        if a.kind in ("VERIFY", "BLOCK"):
            u = a.details["utilities"]
            _emit(log, f"     utilities: " + ", ".join(f"{k}={v}" for k, v in u.items()))
