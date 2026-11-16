"""Decision making under uncertainty (expected utility).

Budget alerts
-------------
Given the projection N(mean, sd) for a category and its budget B:

    P(over)      = 1 - Phi((B - mean) / sd)
    E[overspend] = sd * phi(z) + (mean - B) * (1 - Phi(z)),   z = (B - mean)/sd

A warning only helps if the user acts (probability p_act, *learned* online) and
only on the money not yet spent:

    avoidable     = min(E[overspend], effectiveness * (mean - spent))
    U(WARN)       = p_act * avoidable                 - fatigue_cost * (1 + 0.5 * recent_alerts)
    U(INFO)       = 0.35 * p_act * avoidable          - 0.20 * fatigue_cost * (1 + 0.5 * recent_alerts)
    U(REALLOCATE) = 0.8 * min(E[overspend], slack)    - realloc_cost
    U(NONE)       = 0

REALLOCATE is a *re-planning* action for budgets that turned out unrealistic:
it is only on the table when P(over) >= 0.7, the projection exceeds the budget
by 20% or more, enough days remain, donors can spare a meaningful amount, and
the category has not been re-planned this month.  A warning cannot fix an
unrealistic budget; moving slack from categories that are safely under budget
can, at the cost of one decision for the user.

The action with the highest utility is chosen if it beats NONE by a margin.
A cooldown suppresses repeated warnings for the same category unless the risk
rose noticeably.

Fraud handling
--------------
With anomaly score s, calibrated to a fraud probability p = s^gamma (gamma = 2:
weak evidence below 0.5 is mostly noise and must compound), amount a, and a
follow-on-loss constant L for fraud that goes unnoticed:

    U(NONE)   = - p * (a + L)
    U(VERIFY) = - verify_cost - 0.10 * p * (a + L)
    U(BLOCK)  = - (1 - p) * (false_block_cost + 0.5 * a)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def overspend_stats(mean: float, sd: float, budget: float) -> tuple[float, float]:
    """Returns (P(spend > budget), E[max(spend - budget, 0)])."""
    sd = max(sd, 1e-6)
    z = (budget - mean) / sd
    p_over = 1.0 - norm_cdf(z)
    e_over = sd * norm_pdf(z) + (mean - budget) * p_over
    return p_over, max(e_over, 0.0)


@dataclass
class Action:
    kind: str                       # NONE | INFO | WARN | REALLOCATE | VERIFY | BLOCK | ASK_CATEGORY
    category: str | None
    message: str
    utility: float = 0.0
    details: dict = field(default_factory=dict)
    day: int = 0
    txn_id: int | None = None

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


class UtilityBudgetPolicy:
    def __init__(self, fatigue_cost: float = 120.0, effectiveness: float = 0.30,
                 realloc_cost: float = 100.0, cooldown_days: int = 4, min_days_left: int = 2,
                 min_day: int = 5, min_gain: float = 25.0, realloc_p_over: float = 0.70,
                 realloc_ratio: float = 1.20, realloc_min_days_left: int = 5):
        self.fatigue_cost = fatigue_cost
        self.effectiveness = effectiveness
        self.realloc_cost = realloc_cost
        self.cooldown_days = cooldown_days
        self.min_days_left = min_days_left
        self.min_day = min_day          # no budget alerts before this day: too little evidence
        self.min_gain = min_gain        # act only if the best action beats doing nothing by this margin
        self.realloc_p_over = realloc_p_over
        self.realloc_ratio = realloc_ratio
        self.realloc_min_days_left = realloc_min_days_left

    def choose(self, category: str, budget: float, spent: float, proj_mean: float, proj_sd: float,
               day: int, days_in_month: int, p_act: float, recent_alerts: int,
               last_alert: tuple[int, float] | None, donors: list[tuple[str, float]],
               fatigue_scale: float = 1.0, realloc_allowed: bool = True) -> Action:
        days_left = days_in_month - day
        p_over, e_over = overspend_stats(proj_mean, proj_sd, budget)
        base = {"p_over": round(p_over, 3), "e_over": round(e_over, 0), "proj_mean": round(proj_mean),
                "proj_sd": round(proj_sd), "spent": round(spent), "budget": budget, "p_act": round(p_act, 2)}
        if day < self.min_day or days_left < self.min_days_left or e_over < 0.02 * budget:
            return Action("NONE", category, "", 0.0, base, day)

        remaining = max(proj_mean - spent, 0.0)
        avoidable = min(e_over, self.effectiveness * remaining)
        fatigue = self.fatigue_cost * fatigue_scale * (1.0 + 0.5 * recent_alerts)

        u_warn = p_act * avoidable - fatigue
        u_info = 0.35 * p_act * avoidable - 0.20 * fatigue
        slack = sum(s for _, s in donors)
        transfer = min(max(proj_mean - budget, 0.0), slack)
        realloc_ok = (realloc_allowed and slack > 0 and p_over >= self.realloc_p_over
                      and proj_mean >= self.realloc_ratio * budget and days_left >= self.realloc_min_days_left
                      and transfer >= max(200.0, 0.10 * budget))
        u_realloc = 0.8 * transfer - self.realloc_cost if realloc_ok else -math.inf

        # cooldown: do not nag about the same category unless risk clearly rose
        if last_alert is not None and (day - last_alert[0]) < self.cooldown_days and p_over < last_alert[1] + 0.2:
            u_warn, u_info = -math.inf, -math.inf

        options = {"NONE": 0.0, "WARN": u_warn, "INFO": u_info, "REALLOCATE": u_realloc}
        kind = max(options, key=options.get)
        base["utilities"] = {k: (round(v, 1) if math.isfinite(v) else None) for k, v in options.items()}
        if options[kind] < self.min_gain:
            kind = "NONE"

        if kind == "NONE":
            return Action("NONE", category, "", 0.0, base, day)
        if kind == "REALLOCATE":
            plan, left = [], transfer
            for donor, s in sorted(donors, key=lambda d: -d[1]):
                take = min(s, left)
                if take >= 50:
                    plan.append((donor, round(take / 50) * 50))
                    left -= take
                if left <= 0:
                    break
            base["transfer_plan"] = plan
            desc = ", ".join(f"Rs {amt:.0f} from {d}" for d, amt in plan)
            msg = (f"{category}: projected Rs {proj_mean:,.0f} vs budget Rs {budget:,.0f} "
                   f"(P(over) = {p_over:.0%}). Other categories are under budget; suggest moving {desc} to {category}.")
            return Action("REALLOCATE", category, msg, u_realloc, base, day)
        pct = spent / budget if budget else 0.0
        msg = (f"{category}: Rs {spent:,.0f} spent by day {day} ({pct:.0%} of Rs {budget:,.0f}). "
               f"Projected Rs {proj_mean:,.0f} +/- {proj_sd:,.0f}; {p_over:.0%} chance of overspending by about Rs {e_over:,.0f}.")
        if kind == "WARN":
            cap = max(budget - spent, 0.0) / max(days_left, 1)
            msg += f" Keep {category} under Rs {cap:,.0f}/day for the remaining {days_left} days."
        return Action(kind, category, msg, options[kind], base, day)


class RuleBudgetPolicy:
    """Baseline: warn once per category when 80% of budget is spent before day 25."""

    def __init__(self, threshold: float = 0.80, last_day: int = 25):
        self.threshold = threshold
        self.last_day = last_day

    def choose(self, category, budget, spent, proj_mean, proj_sd, day, days_in_month, p_act,
               recent_alerts, last_alert, donors, fatigue_scale=1.0, realloc_allowed=True) -> Action:
        if last_alert is None and day <= self.last_day and budget > 0 and spent >= self.threshold * budget:
            msg = f"{category}: {spent / budget:.0%} of Rs {budget:,.0f} budget used by day {day}."
            return Action("WARN", category, msg, 0.0, {"spent": spent, "budget": budget}, day)
        return Action("NONE", category, "", 0.0, {}, day)


class UtilityFraudPolicy:
    def __init__(self, verify_cost: float = 80.0, false_block_cost: float = 150.0, follow_on_loss: float = 150.0,
                 gamma: float = 2.0):
        self.verify_cost = verify_cost
        self.false_block_cost = false_block_cost
        self.follow_on_loss = follow_on_loss
        self.gamma = gamma

    def choose(self, txn: dict, score: float, explanation: str, day: int) -> Action:
        a = txn["amount"]
        p = score ** self.gamma
        loss = a + self.follow_on_loss
        u = {"NONE": -p * loss,
             "VERIFY": -self.verify_cost - 0.10 * p * loss,
             "BLOCK": -(1 - p) * (self.false_block_cost + 0.5 * a)}
        kind = max(u, key=u.get)
        details = {"score": round(score, 3), "p_fraud": round(p, 3), "utilities": {k: round(v, 1) for k, v in u.items()}}
        if kind == "NONE":
            return Action("NONE", None, "", 0.0, details, day, txn["txn_id"])
        verb = "Hold and ask user to confirm" if kind == "VERIFY" else "Block and alert user"
        msg = f"{verb}: Rs {a:,.0f} '{txn['description']}' at {txn['ts']:%d %b %H:%M} (risk {score:.0%}: {explanation})."
        return Action(kind, None, msg, u[kind], details, day, txn["txn_id"])


class ThresholdFraudPolicy:
    """Baseline: flag when score crosses a fixed threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def choose(self, txn, score, explanation, day) -> Action:
        if score >= self.threshold:
            return Action("VERIFY", None, f"Flagged Rs {txn['amount']:,.0f} '{txn['description']}'",
                          0.0, {"score": round(score, 3)}, day, txn["txn_id"])
        return Action("NONE", None, "", 0.0, {}, day, txn["txn_id"])
