"""Replay engine behind the dashboard (app.py).

The dashboard is stateless between clicks: every interaction replays the agent
from day 1 up to the chosen day and applies the person's feedback on the day of
the transaction or alert it refers to.  Replays are deterministic, so the same
inputs always give the same screen, and a month can be scrubbed day by day.

Two data sources:
* "demo"       a simulated month (the same one `python main.py demo` uses),
               optionally with the simulated user who reacts to alerts
* "statement"  a bank / UPI statement export (CSV: date, description, amount,
               optional time)
"""
from __future__ import annotations

import calendar
import copy
import functools
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .anomaly import AnomalyDetector
from .data_gen import public_view
from .decision import UtilityBudgetPolicy, overspend_stats
from .demo import pick_interesting_month
from .evaluate import build_trained_agent
from .simulation import SimulatedUser


@dataclass
class Feedback:
    """Everything the person has told the agent so far."""
    payments: dict = field(default_factory=dict)     # txn_id -> "legit" | "confirm_fraud"
    categories: dict = field(default_factory=dict)   # txn_id -> correct category
    alerts: dict = field(default_factory=dict)       # (day, category) -> "helpful" | "dismiss" | "reject_realloc"

    def count(self) -> int:
        return len(self.payments) + len(self.categories) + len(self.alerts)


@functools.lru_cache(maxsize=4)
def _base(seed: int):
    """Train once per seed: simulator, trained agent, the demo month."""
    sim, hist, agent = build_trained_agent(seed, "utility")
    ym = pick_interesting_month(sim, agent)
    return sim, agent, ym, len(hist)


def _renumber(day: int, txns: list[dict]) -> list[dict]:
    """Stable ids (independent of how often the simulator was called)."""
    for i, t in enumerate(txns):
        t["txn_id"] = day * 1000 + i
    return txns


@functools.lru_cache(maxsize=4)
def _counterfactual(seed: int) -> dict:
    """Daily spend per category for the demo month with no agent at all."""
    sim, agent, (y, m, idx), _ = _base(seed)
    days = calendar.monthrange(y, m)[1]
    out = {c: [0.0] * (days + 1) for c in agent.budgets}
    for d in range(1, days + 1):
        for t in sim.generate_day(date(y, m, d), {}, idx):
            if not t["is_anomaly"]:
                out[t["category"]][d] += t["amount"]
    return out


def default_budgets(seed: int = 7) -> dict:
    return dict(_base(seed)[1].budgets)


def demo_month(seed: int = 7) -> tuple[int, int]:
    y, m, _ = _base(seed)[2]
    return y, m


def load_statement(file) -> pd.DataFrame:
    """Read a statement export. Needs columns date, description, amount; time is optional."""
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in ("date", "description", "amount") if c not in cols]
    if missing:
        raise ValueError(f"The file needs the columns date, description and amount. Missing: {', '.join(missing)}.")
    if "time" in cols:
        ts = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str), errors="coerce")
    else:
        ts = pd.to_datetime(df[cols["date"]], errors="coerce")
    out = pd.DataFrame({"ts": ts, "description": df[cols["description"]].astype(str),
                        "amount": pd.to_numeric(df[cols["amount"]], errors="coerce").abs()})
    out = out.dropna()
    out = out[out["amount"] > 0].sort_values("ts").reset_index(drop=True)
    if out.empty:
        raise ValueError("No usable rows: check the date and amount columns.")
    out["txn_id"] = range(1, len(out) + 1)
    return out


def statement_months(df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted({(int(t.year), int(t.month)) for t in df["ts"]})


def _apply_feedback(agent, actions, day: int, fb: Feedback) -> None:
    for a in actions:
        if a.kind in ("VERIFY", "BLOCK") and a.txn_id in fb.payments:
            agent.feedback(a, fb.payments[a.txn_id])
        elif a.kind in ("WARN", "INFO", "REALLOCATE") and (day, a.category) in fb.alerts:
            agent.feedback(a, fb.alerts[(day, a.category)])
    for rec in agent.month_records:
        if rec["day"] == day and rec["txn_id"] in fb.categories:
            agent.correct_category(rec, fb.categories[rec["txn_id"]])


def replay(source: str = "demo", upto_day: int | None = None, feedback: Feedback | None = None, seed: int = 7,
           simulate_user: bool = True, fatigue_cost: float = 120.0, budgets: dict | None = None,
           statement: pd.DataFrame | None = None, ym: tuple[int, int] | None = None) -> dict:
    fb = feedback or Feedback()
    sim, base_agent, (dy, dm, didx), n_hist = _base(seed if source == "demo" else 0)
    agent = copy.deepcopy(base_agent)
    if budgets:
        agent.budgets = {c: float(v) for c, v in budgets.items()}
    agent.budget_policy = UtilityBudgetPolicy(fatigue_cost=fatigue_cost)

    if source == "demo":
        year, month = dy, dm
    else:
        if statement is None:
            raise ValueError("No statement loaded.")
        year, month = ym or statement_months(statement)[-1]
    days = calendar.monthrange(year, month)[1]
    day_limit = min(upto_day or days, days)

    agent.start_month(year, month)
    user = SimulatedUser(seed) if (source == "demo" and simulate_user) else None
    if user is not None:
        user.start_month()
    truth: dict[int, dict] = {}
    feed: list[dict] = []

    for day in range(1, day_limit + 1):
        d = date(year, month, day)
        if source == "demo":
            mult = user.multipliers(day) if user is not None else {}
            raw = _renumber(day, sim.generate_day(d, mult, didx))
            for t in raw:
                truth[t["txn_id"]] = t
            txns = [public_view(t) for t in raw]
        else:
            g = statement[(statement["ts"].dt.year == year) & (statement["ts"].dt.month == month) & (statement["ts"].dt.day == day)]
            txns = [{"txn_id": int(r.txn_id), "ts": r.ts.to_pydatetime(), "description": r.description, "amount": float(r.amount)}
                    for r in g.itertuples()]
        actions = agent.process_day(d, txns)
        for a in actions:
            feed.append({"day": day, "kind": a.kind, "category": a.category, "message": a.message,
                         "utilities": a.details.get("utilities"), "p_act": a.details.get("p_act"),
                         "score": a.details.get("score"), "txn_id": a.txn_id})
        if user is not None:
            user.react(actions, day, agent, truth)
        else:
            _apply_feedback(agent, actions, day, fb)

    # ---- transactions table
    rows = []
    for r in agent.month_records:
        status = "held for review" if r["txn_id"] in agent.held else "fraud (excluded)" if r.get("confirmed_fraud") else "counted"
        row = {"txn_id": r["txn_id"], "when": r["ts"], "description": r["description"], "amount": r["amount"],
               "category": r["pred_category"], "confidence": round(r["confidence"], 2),
               "risk": round(r["anomaly_score"], 2), "status": status,
               "evidence": AnomalyDetector.explain(r["evidence"])}
        if truth:
            row["true category"] = truth[r["txn_id"]]["category"]
            row["injected anomaly"] = truth[r["txn_id"]]["anomaly_type"] or ""
        rows.append(row)
    transactions = pd.DataFrame(rows)

    # ---- per-category state
    cats = []
    for c in sorted(agent.budgets):
        spent = agent.spent.get(c, 0.0)
        mean, sd = agent.forecaster.project(c, spent, day_limit, days, agent._recent_rate(c, day_limit))
        budget = agent.month_budgets.get(c, agent.budgets[c])
        p_over, e_over = overspend_stats(mean, sd, budget)
        cats.append({"category": c, "budget": agent.budgets[c], "adjusted budget": budget, "spent": round(spent),
                     "used": spent / budget if budget else 0.0, "projected": round(mean), "plus/minus": round(sd),
                     "risk of overspending": p_over, "expected overspend": round(e_over)})
    categories = pd.DataFrame(cats)

    series = {c: [sum(agent.daily[c][1:d + 1]) for d in range(day_limit + 1)] for c in agent.budgets}
    counter = None
    if source == "demo":
        cf = _counterfactual(seed)
        counter = {c: [sum(cf[c][1:d + 1]) for d in range(days + 1)] for c in cf}

    return {
        "label": f"{calendar.month_name[month]} {year}", "year": year, "month": month, "days": days, "day": day_limit,
        "agent": agent, "feed": feed, "transactions": transactions, "categories": categories,
        "series": series, "counterfactual": counter, "truth": truth, "history_size": n_hist,
        "user_stats": dict(user.stats) if user is not None else None,
        "learned": {"p_act": agent.p_act, "fatigue_scale": agent.fatigue_scale,
                    "categoriser_updates": agent.categorizer.n_updates, "vocabulary": len(agent.categorizer.vocab)},
    }
