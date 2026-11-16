"""Demonstration: one month of the agent, day by day, with inputs and outputs.

`python main.py demo`           synthetic month with a simulated user
`python main.py csv --path f`   run the agent over your own statement export
                                (CSV with columns: date, description, amount
                                and optionally time)
"""
from __future__ import annotations

import calendar
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data_gen import add_months, public_view
from .evaluate import HISTORY_MONTHS, OUT, START, build_trained_agent
from .simulation import SimulatedUser, _emit, run_month


def _budget_table(agent, log) -> None:
    _emit(log, "\nBudgets proposed by the agent (110% of the historical monthly mean, user can edit):")
    _emit(log, f"  {'category':<14}{'hist. mean':>12}{'budget':>10}   type")
    for c in sorted(agent.budgets):
        kind = "fixed" if agent.forecaster.fixed[c] else ("run-rate (sparse)" if agent.forecaster.sparse[c] else "run-rate")
        _emit(log, f"  {c:<14}{agent.forecaster.mu0[c]:>12,.0f}{agent.budgets[c]:>10,.0f}   {kind}")


def pick_interesting_month(sim, agent, n_candidates: int = 12) -> tuple[int, int, int]:
    """Choose the first month after deployment whose no-agent outcome overspends and
    which contains at least one injected anomaly, so the demo shows everything."""
    y, m = add_months(START[0], START[1], HISTORY_MONTHS)
    best = (y, m, HISTORY_MONTHS)
    for i in range(n_candidates):
        idx = HISTORY_MONTHS + i
        txns = sim.generate_month(y, m, idx)
        spent = {c: 0.0 for c in agent.budgets}
        for t in txns:
            if not t["is_anomaly"]:
                spent[t["category"]] += t["amount"]
        over = sum(max(spent[c] - agent.budgets[c], 0) for c in spent)
        n_anom = sum(t["is_anomaly"] for t in txns)
        if over > 300 and n_anom >= 2:
            return y, m, idx
        y, m = add_months(y, m, 1)
    return best


def timeline_plot(sim, agent, summary, year, month, month_idx, path: Path) -> None:
    """Cumulative spend vs budget for the category that received the strongest alert."""
    alert_cats = [a.category for a in agent.alerts if a.kind in ("WARN", "INFO", "REALLOCATE")]
    if not alert_cats:
        return
    cat = max(set(alert_cats), key=alert_cats.count)
    days = calendar.monthrange(year, month)[1]
    cf = [0.0] * (days + 1)
    for d in range(1, days + 1):
        for t in sim.generate_day(date(year, month, d), {}, month_idx):
            if t["category"] == cat and not t["is_anomaly"]:
                cf[d] += t["amount"]
    actual = agent.daily[cat]
    cum_cf = [sum(cf[:d + 1]) for d in range(days + 1)]
    cum_act = [sum(actual[:d + 1]) for d in range(days + 1)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(days + 1), cum_cf, "--", color="#9e9e9e", label="without agent (counterfactual)")
    ax.plot(range(days + 1), cum_act, color="#1f77b4", lw=2, label="with agent (user reacts to alerts)")
    ax.axhline(agent.budgets[cat], color="#d62728", ls=":", label=f"budget Rs {agent.budgets[cat]:,.0f}")
    for a in agent.alerts:
        if a.category == cat and a.kind in ("WARN", "INFO", "REALLOCATE"):
            ax.axvline(a.day, color="#f0a202", alpha=0.7)
            ax.text(a.day + 0.2, agent.budgets[cat] * 0.15, a.kind, rotation=90, fontsize=8, color="#f0a202")
    ax.set_xlabel(f"day of {calendar.month_name[month]} {year}"); ax.set_ylabel("cumulative spend (Rs)")
    ax.set_title(f"{cat}: cumulative spend vs budget")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_demo(seed: int = 7, out_path: Path | None = None) -> dict:
    OUT.mkdir(exist_ok=True)
    out_path = out_path or OUT / "demo_log.txt"
    with open(out_path, "w", encoding="utf-8") as log:
        _emit(log, "=" * 78)
        _emit(log, " SpendSense - Smart Expense & Budget Agent : demonstration")
        _emit(log, "=" * 78)
        sim, hist, agent = build_trained_agent(seed, policy="utility")
        _emit(log, f"\nTraining history: {len(hist)} transactions over {HISTORY_MONTHS} months "
                   f"({hist['is_anomaly'].sum()} flagged anomalies excluded from the detector's statistics).")
        _emit(log, f"Categoriser vocabulary: {len(agent.categorizer.vocab)} features; prior P(user acts on warning) = {agent.p_act:.2f}")
        _budget_table(agent, log)

        year, month, idx = pick_interesting_month(sim, agent)
        user = SimulatedUser(seed)
        _emit(log, f"\nRunning the agent for {calendar.month_name[month]} {year}. Legend: predicted category + confidence, "
                   f"<risk> = anomaly score >= 30%, (true: X) marks a mis-categorisation.")
        summary = run_month(sim, agent, user, year, month, idx, verbose=True, log=log)

        _emit(log, "\n" + "=" * 78)
        _emit(log, f" Month summary: {calendar.month_name[month]} {year}")
        _emit(log, "=" * 78)
        _emit(log, f"  {'category':<14}{'budget':>9}{'adjusted':>10}{'spent':>10}{'no-agent':>10}   status")
        for c, r in sorted(summary["categories"].items()):
            cf = summary["counterfactual_spent"][c]
            status = f"OVER by Rs {r['overspend']:,.0f}" if r["overspend"] > 0 else "ok"
            if cf > agent.budgets[c] and r["overspend"] == 0:
                status = f"ok (would have been over by Rs {cf - agent.budgets[c]:,.0f} without agent)"
            adj = f"{r['adjusted_budget']:>10,.0f}" if r["adjusted_budget"] != r["budget"] else f"{'-':>10}"
            _emit(log, f"  {c:<14}{r['budget']:>9,.0f}{adj}{r['spent']:>10,.0f}{cf:>10,.0f}   {status}")
        _emit(log, f"\n  Category overspend vs original budgets: with agent Rs {summary['total_overspend']:,.0f}, "
                   f"without agent Rs {summary['counterfactual_overspend']:,.0f}")
        _emit(log, f"  Net overspend of the whole monthly budget: with agent Rs {summary['net_total_overspend']:,.0f}, "
                   f"without agent Rs {summary['counterfactual_net_total_overspend']:,.0f}")
        _emit(log, f"  Alerts: {summary['n_alerts']} budget alerts, {summary['n_realloc']} reallocations; "
                   f"user responded {user.stats['responded']}x, dismissed {user.stats['dismissed']}x")
        _emit(log, f"  Anomalies: {summary['anomalies_present']} present, {summary['anomalies_flagged_tp']} caught, "
                   f"{summary['flags_fp']} false flags")
        _emit(log, f"  Learned user model: P(acts on warning) = {summary['p_act']:.2f}, fatigue scale = {summary['fatigue_scale']}; "
                   f"categoriser received {user.stats['category_answers']} answers to its questions")
        timeline_plot(sim, agent, summary, year, month, idx, OUT / "demo_timeline.png")
        _emit(log, f"\nLog saved to {out_path}; timeline plot saved to {OUT / 'demo_timeline.png'}")
    return summary


def run_on_csv(path: str, seed: int = 0) -> None:
    """Run the agent over a user-provided statement.  The models are bootstrapped
    on the synthetic labelled history (in a real deployment: the user's own
    labelled months), so categories for unfamiliar merchants may be uncertain and
    the agent will ask."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "time" in cols:
        df["ts"] = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str))
    else:
        df["ts"] = pd.to_datetime(df[cols["date"]])
    df["amount"] = df[cols["amount"]].astype(float)
    df["description"] = df[cols["description"]].astype(str)
    df = df.sort_values("ts").reset_index(drop=True)
    df["txn_id"] = range(1, len(df) + 1)
    _, _, agent = build_trained_agent(seed, policy="utility")
    print(f"Loaded {len(df)} transactions from {path}")
    for (y, m), g in df.groupby([df["ts"].dt.year, df["ts"].dt.month]):
        agent.start_month(int(y), int(m))
        print(f"\n##### {calendar.month_name[int(m)]} {y}")
        for d, gd in g.groupby(g["ts"].dt.date):
            txns = [{"txn_id": int(r.txn_id), "ts": r.ts.to_pydatetime(), "description": r.description, "amount": float(r.amount)}
                    for r in gd.itertuples()]
            actions = agent.process_day(d, txns)
            for t in txns:
                rec = agent._record(t["txn_id"])
                print(f"  {t['ts']:%d %b %H:%M}  Rs {t['amount']:>8,.0f}  {t['description'][:40]:<40} -> {rec['pred_category']} ({rec['confidence']:.0%})")
            for a in actions:
                print(f"  >> {a}")
        s = agent.month_summary()
        print(f"  month total overspend: Rs {s['total_overspend']:,.0f}")
