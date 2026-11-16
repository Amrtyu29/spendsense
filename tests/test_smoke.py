"""Smoke tests: run with  python tests/test_smoke.py  (no pytest needed)."""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expense_agent.anomaly import AnomalyDetector, merchant_key
from expense_agent.categorizer import KeywordRuleCategorizer, NaiveBayesCategorizer
from expense_agent.data_gen import MERCHANTS, TransactionSimulator
from expense_agent.decision import UtilityBudgetPolicy, UtilityFraudPolicy, overspend_stats
from expense_agent.evaluate import build_trained_agent
from expense_agent.forecaster import SpendForecaster
from expense_agent.simulation import SimulatedUser, run_month


def test_simulator_is_deterministic_and_counterfactual_comparable():
    sim = TransactionSimulator(3)
    a = sim.generate_day(date(2026, 1, 10), {}, 6)
    b = TransactionSimulator(3).generate_day(date(2026, 1, 10), {}, 6)
    assert [t["description"] for t in a] == [t["description"] for t in b]
    c = TransactionSimulator(3).generate_day(date(2026, 1, 10), {"Food": 0.5}, 6)
    assert len(a) == len(c)                       # same draws, only amounts scaled
    for x, z in zip(a, c):
        if x["category"] == "Food" and not x["is_anomaly"]:
            assert z["amount"] <= x["amount"]


def test_categorizer_learns_and_updates_online():
    sim = TransactionSimulator(1)
    hist = sim.generate_history(2025, 5, 3, 0)
    nb = NaiveBayesCategorizer(rules=KeywordRuleCategorizer.from_brands(MERCHANTS)).fit(hist.to_dict("records"))
    label, conf = nb.predict("UPI/SWIGGY/123", 180.0, 20)
    assert label == "Food" and conf > 0.8
    label, _ = nb.predict("UPI/BUNDL TECHNOLOGIES/123", 180.0, 20)   # rare legal-entity alias
    assert label == "Food"
    before, _ = nb.predict("NEW CAMPUS JUICE BAR", 60.0, 16)
    for _ in range(3):
        nb.update("NEW CAMPUS JUICE BAR", 60.0, 16, "Food", weight=3.0)
    after, conf = nb.predict("NEW CAMPUS JUICE BAR", 60.0, 16)
    assert after == "Food" and conf > 0.9


def test_anomaly_detector_flags_burst_and_large_amount():
    sim = TransactionSimulator(2)
    hist = sim.generate_history(2025, 5, 4, 0)
    det = AnomalyDetector().fit([r for r in hist.to_dict("records") if not r["is_anomaly"]])
    t0 = datetime(2026, 1, 5, 13, 0)
    recent = []
    scores = []
    for k in range(5):
        txn = {"ts": t0 + timedelta(minutes=3 * k), "description": "UPI/SWIGGY/111", "amount": 120.0}
        s, _ = det.score(txn, "Food", recent)
        scores.append(s)
        recent.append({"ts": txn["ts"], "key": merchant_key(txn["description"]), "amount": txn["amount"]})
    assert scores[0] < 0.2 and scores[-1] > 0.8
    s, ev = det.score({"ts": t0, "description": "UPI/SWIGGY/222", "amount": 4000.0}, "Food", [])
    assert s > 0.9 and ev["amount_z"] > 3


def test_decision_math():
    p, e = overspend_stats(mean=1000, sd=100, budget=1000)
    assert abs(p - 0.5) < 1e-6 and abs(e - 100 * 0.3989) < 1.0
    pol = UtilityBudgetPolicy()
    a = pol.choose("Food", 5000, 3000, 6000, 400, 15, 30, 0.6, 0, None, [])
    assert a.kind in ("WARN", "INFO")
    a = pol.choose("Food", 5000, 1000, 3000, 400, 15, 30, 0.6, 0, None, [])
    assert a.kind == "NONE"
    fp = UtilityFraudPolicy()
    assert fp.choose({"txn_id": 1, "ts": datetime(2026, 1, 1, 3), "description": "X", "amount": 2000}, 0.95, "", 1).kind != "NONE"
    assert fp.choose({"txn_id": 2, "ts": datetime(2026, 1, 1, 13), "description": "X", "amount": 150}, 0.15, "", 1).kind == "NONE"


def test_agent_month_runs_and_reports():
    sim, hist, agent = build_trained_agent(0, "utility")
    s = run_month(sim, agent, SimulatedUser(0), 2025, 11, 6)
    assert set(s["categories"]) == set(agent.budgets)
    assert s["total_overspend"] >= 0 and s["anomalies_flagged_tp"] <= s["anomalies_present"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} tests passed")
