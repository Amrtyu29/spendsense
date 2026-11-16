"""Ready-made demo statements for the dashboard.

`python main.py demodata` writes three one-month statements to data/demo/, each
chosen to show a different situation, plus a template for typing your own:

    1_normal_month.csv          ordinary month: ends within budget, nothing suspicious
    2_overspending_month.csv    spending drifts over budget: warnings and re-plans
    3_suspicious_payments.csv   three kinds of suspicious payment to review
    my_statement_template.csv   the column format, with a few example rows
    answers/                    the same three months with the true category and
                                the injected anomaly type, for checking the agent

The months are not hand-written: the generator searches simulated months from
the same environment the agent was trained on and picks the first one that
matches each description, so the files are reproducible.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd

from .data_gen import TransactionSimulator, add_months
from .evaluate import HISTORY_MONTHS, START, build_trained_agent

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"

SCENARIOS = {
    "1_normal_month": "Ordinary month: ends within budget, nothing suspicious, a few cautionary alerts",
    "2_overspending_month": "Spending drifts over budget: warnings and budget re-plans",
    "3_suspicious_payments": "Three kinds of suspicious payment to review, all caught",
}


def _month_stats(txns: list[dict], budgets: dict) -> dict:
    spent = {c: 0.0 for c in budgets}
    types: dict[str, int] = {}
    for t in txns:
        if t["is_anomaly"]:
            types[t["anomaly_type"]] = types.get(t["anomaly_type"], 0) + 1
        else:
            spent[t["category"]] += t["amount"]
    over = {c: max(spent[c] - budgets[c], 0.0) for c in budgets}
    return {"overspend": sum(over.values()), "over_categories": sum(v > 0 for v in over.values()),
            "anomaly_types": types, "n_anomalous": sum(types.values())}


def _agent_behaviour(txns: list[dict], y: int, m: int) -> dict:
    """Run the dashboard's agent over the month and summarise what it does."""
    from .app_engine import replay
    df = pd.DataFrame(txns).sort_values("ts", kind="stable").reset_index(drop=True)
    statement = pd.DataFrame({"ts": df["ts"], "description": df["description"], "amount": df["amount"],
                              "txn_id": range(1, len(df) + 1)})
    suspicious = dict(zip(statement["txn_id"], df["is_anomaly"]))
    r = replay("statement", statement=statement, ym=(y, m))
    held = [f["txn_id"] for f in r["feed"] if f["kind"] in ("VERIFY", "BLOCK")]
    return {"budget_alerts": sum(f["kind"] in ("WARN", "INFO", "REALLOCATE") for f in r["feed"]),
            "warnings": sum(f["kind"] == "WARN" for f in r["feed"]),
            "held": len(held), "held_true": sum(bool(suspicious[t]) for t in held)}


def find_scenarios(max_seed: int = 120) -> dict:
    """Return {scenario: (seed, year, month, month_idx, stats)}: the best match of each.

    A month must fit its description twice: in the raw data (what really
    happened) and in how the dashboard's agent behaves on it.  Candidates are
    filtered on the data, then ranked on the agent's behaviour:

    normal       no anomalies, no overspend, no payment held  -> fewest budget alerts
    overspending no anomalies, >= Rs 2,500 over in >= 2 categories, >= 2 warnings,
                 no payment held                              -> alert count closest to 6
    suspicious   >= 3 kinds of anomaly, every held payment truly suspicious
                                                              -> most caught, then fewest budget alerts
    """
    budgets = build_trained_agent(0, "utility")[2].budgets
    best: dict = {}

    def offer(name, score, entry):
        if name not in best or score < best[name][0]:
            best[name] = (score, entry)

    for seed in range(1, max_seed):
        sim = TransactionSimulator(seed)
        for i in range(HISTORY_MONTHS, HISTORY_MONTHS + 12):
            y, m = add_months(START[0], START[1], i)
            txns = sim.generate_month(y, m, i)
            s = _month_stats(txns, budgets)
            if s["n_anomalous"] == 0 and s["overspend"] == 0:
                b = _agent_behaviour(txns, y, m)
                if b["held"] == 0:
                    offer("1_normal_month", (b["budget_alerts"],), (seed, y, m, i, {**s, **b}))
            elif s["n_anomalous"] == 0 and s["overspend"] >= 2500 and s["over_categories"] >= 2:
                b = _agent_behaviour(txns, y, m)
                if b["warnings"] >= 2 and b["held"] == 0:
                    offer("2_overspending_month", (abs(b["budget_alerts"] - 6),), (seed, y, m, i, {**s, **b}))
            elif len(s["anomaly_types"]) >= 3 and s["overspend"] < 800:
                b = _agent_behaviour(txns, y, m)
                if b["held"] == b["held_true"] and b["held_true"] > 0:
                    offer("3_suspicious_payments", (-b["held_true"] / s["n_anomalous"], b["budget_alerts"]),
                          (seed, y, m, i, {**s, **b}))
    return {name: entry for name, (_, entry) in best.items()}


def _frames(txns: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(txns).sort_values("ts", kind="stable")
    pub = pd.DataFrame({"date": df["ts"].dt.strftime("%Y-%m-%d"), "time": df["ts"].dt.strftime("%H:%M"),
                        "description": df["description"], "amount": df["amount"].round(2)})
    ans = pub.assign(true_category=df["category"].values, suspicious=df["is_anomaly"].values,
                     anomaly_type=df["anomaly_type"].values)
    return pub, ans


def write_demo_data() -> list[dict]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "answers").mkdir(exist_ok=True)
    summary = []
    for name, (seed, y, m, idx, s) in sorted(find_scenarios().items()):
        txns = TransactionSimulator(seed).generate_month(y, m, idx)
        pub, ans = _frames(txns)
        pub.to_csv(DEMO_DIR / f"{name}.csv", index=False)
        ans.to_csv(DEMO_DIR / "answers" / f"{name}_answers.csv", index=False)
        summary.append({"file": f"{name}.csv", "what it shows": SCENARIOS[name], "month": f"{calendar.month_name[m]} {y}",
                        "transactions": len(pub), "overspend without agent (Rs)": round(s["overspend"]),
                        "suspicious payments": s["n_anomalous"], "kinds": ", ".join(sorted(s["anomaly_types"])) or "-",
                        "agent: budget alerts": s["budget_alerts"], "agent: payments held": s["held"],
                        "held that are truly suspicious": s["held_true"]})
    template = pd.DataFrame({
        "date": ["2026-11-01", "2026-11-01", "2026-11-02"],
        "time": ["08:15", "13:05", "20:40"],
        "description": ["UPI/CAMPUS CANTEEN/1234567890", "POS DMART COIMBATORE", "UPI-SWIGGY-9876543210@okaxis"],
        "amount": [85, 640, 220],
    })
    template.to_csv(DEMO_DIR / "my_statement_template.csv", index=False)
    return summary
