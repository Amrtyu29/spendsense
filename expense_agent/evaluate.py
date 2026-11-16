"""Evaluation suite: three experiments with baselines, plus a sensitivity sweep.

E1  Categorisation   Naive Bayes vs keyword rules; seen vs unseen merchants;
                     ablation without amount/time features; online learning
                     from user corrections.
E2  Anomaly detection ours vs fixed threshold vs global z-score (P/R/F1,
                     PR curves, recall per anomaly type).
E3  Agent policy     utility agent vs rule-based agent vs passive tracker over
                     12 months x several seeds: overspend, alerts, alert
                     precision, dismissals, fraud catches, learned p_act.
E4  Sensitivity      fatigue cost sweep: overspend vs number of alerts.

Outputs go to ./outputs (PNG plots, results.json, RESULTS.md).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import BudgetAgent
from .anomaly import AnomalyDetector, FixedThresholdDetector, GlobalZDetector, merchant_key
from .categorizer import KeywordRuleCategorizer, NaiveBayesCategorizer
from .data_gen import CATEGORIES, MERCHANTS, TransactionSimulator, add_months
from .decision import UtilityBudgetPolicy
from .forecaster import SpendForecaster
from .simulation import SimulatedUser, run_month

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
START = (2025, 5)            # history starts May 2025; agent goes live Nov 2025
HISTORY_MONTHS = 6
LATE_BRANDS = {b for items in MERCHANTS.values() for b, _, late in items if late}
POLICY_LABELS = {"none": "Passive tracker", "rule": "Rule-based agent", "utility": "Utility agent (ours)"}


def build_trained_agent(seed: int, policy: str = "utility", budget_factor: float = 1.10,
                        budget_policy=None):
    sim = TransactionSimulator(seed)
    hist = sim.generate_history(START[0], START[1], HISTORY_MONTHS, 0)
    rows = hist.to_dict("records")
    clean = [r for r in rows if not r["is_anomaly"]]
    categorizer = NaiveBayesCategorizer(rules=KeywordRuleCategorizer.from_brands(MERCHANTS)).fit(rows)
    detector = AnomalyDetector().fit(clean)
    forecaster = SpendForecaster().fit(hist)
    budgets = forecaster.suggest_budgets(budget_factor)
    agent = BudgetAgent(budgets, categorizer, detector, forecaster, policy=policy, budget_policy=budget_policy)
    return sim, hist, agent


# =============================================================== E1
def experiment_categorizer(seed: int = 0) -> dict:
    sim = TransactionSimulator(seed)
    df = sim.generate_history(START[0], START[1], 12, 0)
    train = df[df.month_idx < HISTORY_MONTHS].to_dict("records")
    test = df[df.month_idx >= HISTORY_MONTHS].sort_values("ts").to_dict("records")

    kw = KeywordRuleCategorizer.from_brands(MERCHANTS)
    nb = NaiveBayesCategorizer(rules=kw).fit(train)
    nb_norules = NaiveBayesCategorizer().fit(train)
    nb_text = NaiveBayesCategorizer(use_context=False).fit(train)

    def acc(model, rows):
        return float(np.mean([model.predict(r["description"], r["amount"], r["hour"])[0] == r["category"] for r in rows]))

    seen = [r for r in test if r["merchant"] not in LATE_BRANDS]
    unseen = [r for r in test if r["merchant"] in LATE_BRANDS]
    res = {"n_train": len(train), "n_test": len(test), "n_test_unseen_merchant": len(unseen), "models": {}}
    for name, model in [("naive_bayes (ours: text + amount/time + brand-dictionary hint)", nb),
                        ("naive_bayes without brand-dictionary hint", nb_norules),
                        ("naive_bayes text only", nb_text), ("keyword_rules (brand dictionary alone)", kw)]:
        res["models"][name] = {"accuracy_all": acc(model, test), "accuracy_seen_merchants": acc(model, seen),
                               "accuracy_unseen_merchants": acc(model, unseen)}

    # confusion matrix for NB
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    cm = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for r in test:
        p, _ = nb.predict(r["description"], r["amount"], r["hour"])
        cm[idx[r["category"]], idx[p]] += 1
    per_class = {}
    for c, i in idx.items():
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "support": int(cm[i].sum())}
    res["naive_bayes_per_class"] = per_class

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    nb_acc = res["models"]["naive_bayes (ours: text + amount/time + brand-dictionary hint)"]["accuracy_all"]
    ax.set_title(f"Naive Bayes confusion matrix (test months 7-12, acc {nb_acc:.1%})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "e1_confusion_matrix.png", dpi=150); plt.close(fig)

    # online learning curve: stream the test months, user corrects a fraction of errors
    curves = {}
    for label, p_correct in [("no feedback", 0.0), ("30% of errors corrected", 0.3), ("all errors corrected", 1.0)]:
        model = NaiveBayesCategorizer(rules=kw).fit(train)
        rng = np.random.default_rng(seed)
        hits = []
        for r in test:
            p, _ = model.predict(r["description"], r["amount"], r["hour"])
            hits.append(p == r["category"])
            if p != r["category"] and rng.random() < p_correct:
                model.update(r["description"], r["amount"], r["hour"], r["category"], weight=3.0)
        s = pd.Series(hits, dtype=float).rolling(150, min_periods=30).mean()
        curves[label] = s
        res.setdefault("online_learning_final_accuracy_last300", {})[label] = float(np.mean(hits[-300:]))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, s in curves.items():
        ax.plot(s.values, label=label)
    ax.set_xlabel("transactions seen after deployment"); ax.set_ylabel("rolling accuracy (150 txns)")
    ax.set_title("Online learning from user corrections (new merchants appear from month 7)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.6, 1.0)
    fig.tight_layout(); fig.savefig(OUT / "e1_online_learning.png", dpi=150); plt.close(fig)
    return res


# =============================================================== E2
def experiment_anomaly(seeds=(0, 1, 2, 3, 4), test_months: int = 12) -> dict:
    scores = {"ours": [], "fixed_threshold": [], "global_zscore": []}
    labels, types = [], []
    for seed in seeds:
        sim = TransactionSimulator(seed)
        hist = sim.generate_history(START[0], START[1], HISTORY_MONTHS, 0)
        rows = hist.to_dict("records")
        clean = [r for r in rows if not r["is_anomaly"]]
        nb = NaiveBayesCategorizer(rules=KeywordRuleCategorizer.from_brands(MERCHANTS)).fit(rows)
        dets = {"ours": AnomalyDetector().fit(clean), "fixed_threshold": FixedThresholdDetector().fit(clean),
                "global_zscore": GlobalZDetector().fit(clean)}
        y0, m0 = add_months(START[0], START[1], HISTORY_MONTHS)
        test = sim.generate_history(y0, m0, test_months, HISTORY_MONTHS).sort_values("ts").to_dict("records")
        recent: list[dict] = []
        for r in test:
            cat, _ = nb.predict(r["description"], r["amount"], r["hour"])
            recent = [x for x in recent if (r["ts"] - x["ts"]).total_seconds() <= 2 * 86400]
            for name, det in dets.items():
                s, _ = det.score(r, cat, recent)
                scores[name].append(s)
            recent.append({"ts": r["ts"], "key": merchant_key(r["description"]), "amount": r["amount"]})
            labels.append(bool(r["is_anomaly"]))
            types.append(r["anomaly_type"])
    y = np.array(labels)
    res = {"n_transactions": int(len(y)), "n_anomalies": int(y.sum()), "detectors": {}}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in scores.items():
        s = np.array(s)
        pred = s >= 0.5
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        entry = {"precision@0.5": round(prec, 3), "recall@0.5": round(rec, 3), "f1@0.5": round(f1, 3),
                 "flags": tp + fp, "tp": tp, "fp": fp, "fn": fn}
        if name == "ours":
            entry["recall_by_type"] = {}
            for t in sorted(set(types) - {""}):
                mask = np.array([tt == t for tt in types])
                entry["recall_by_type"][t] = round(float((pred & mask).sum() / mask.sum()), 3)
        # PR curve and average precision (standard step-wise definition:
        # AP = sum_k (R_k - R_{k-1}) * P_k over thresholds from high to low)
        P, R = [], []
        for th in np.linspace(0.99, 0.02, 80):
            p = s >= th
            tp_ = (p & y).sum(); fp_ = (p & ~y).sum(); fn_ = (~p & y).sum()
            if tp_ + fp_ == 0:
                continue
            P.append(tp_ / (tp_ + fp_)); R.append(tp_ / (tp_ + fn_))
        ap, prev_r = 0.0, 0.0
        for p_k, r_k in zip(P, R):
            ap += (r_k - prev_r) * p_k
            prev_r = r_k
        entry["average_precision"] = round(float(ap), 3)
        res["detectors"][name] = entry
        ax.plot(R, P, marker="o", ms=3, label=f"{name} (AP {entry['average_precision']:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Anomaly detection: precision-recall")
    ax.grid(alpha=0.3); ax.legend(loc="lower left"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "e2_anomaly_pr_curve.png", dpi=150); plt.close(fig)
    return res


# =============================================================== E3
def run_policy(seed: int, policy: str, n_months: int = 12, budget_policy=None, p_respond: float = 0.65) -> dict:
    sim, hist, agent = build_trained_agent(seed, policy=policy, budget_policy=budget_policy)
    user = SimulatedUser(seed, p_respond=p_respond)
    y, m = add_months(START[0], START[1], HISTORY_MONTHS)
    months, useful, total_alerts = [], 0, 0
    cat_acc_by_month = []
    p_act_trace = []
    for i in range(n_months):
        idx = HISTORY_MONTHS + i
        s = run_month(sim, agent, user, y, m, idx)
        # alert precision: was the category genuinely heading over budget (counterfactual)?
        for a in agent.alerts:
            if a.kind in ("WARN", "INFO"):
                total_alerts += 1
                if s["counterfactual_spent"][a.category] > agent.budgets[a.category] * 0.97:
                    useful += 1
        months.append(s)
        p_act_trace.append(agent.p_act)
        y, m = add_months(y, m, 1)
    out = {
        "policy": policy, "seed": seed,
        "total_overspend": round(sum(s["total_overspend"] for s in months), 0),
        "net_total_overspend": round(sum(s["net_total_overspend"] for s in months), 0),
        "counterfactual_overspend": round(sum(s["counterfactual_overspend"] for s in months), 0),
        "counterfactual_net_total_overspend": round(sum(s["counterfactual_net_total_overspend"] for s in months), 0),
        "months_with_overspend": sum(1 for s in months if s["total_overspend"] > 0),
        "alerts": total_alerts, "useful_alerts": useful,
        "alert_precision": round(useful / total_alerts, 3) if total_alerts else None,
        "reallocations": sum(s["n_realloc"] for s in months),
        "anomalies_present": sum(s["anomalies_present"] for s in months),
        "fraud_tp": sum(s["anomalies_flagged_tp"] for s in months),
        "fraud_fp": sum(s["flags_fp"] for s in months),
        "user": dict(user.stats), "final_p_act": round(agent.p_act, 3), "final_fatigue_scale": round(agent.fatigue_scale, 2),
        "p_act_trace": [round(v, 3) for v in p_act_trace],
        "categorizer_updates": agent.categorizer.n_updates,
        "monthly_overspend": [s["total_overspend"] for s in months],
        "monthly_counterfactual": [s["counterfactual_overspend"] for s in months],
        "monthly_asks": [s["n_asks"] for s in months],
    }
    return out


def experiment_policies(seeds=(0, 1, 2, 3, 4), n_months: int = 12) -> dict:
    runs = {p: [run_policy(s, p, n_months) for s in seeds] for p in ("none", "rule", "utility")}
    summary = {}
    for p, rs in runs.items():
        agg = {}
        for key in ("total_overspend", "net_total_overspend", "counterfactual_overspend", "months_with_overspend", "alerts", "useful_alerts",
                    "reallocations", "fraud_tp", "fraud_fp", "anomalies_present"):
            agg[key] = round(float(np.mean([r[key] for r in rs])), 1)
        agg["alert_precision"] = round(float(np.mean([r["alert_precision"] for r in rs if r["alert_precision"] is not None])), 3) \
            if any(r["alert_precision"] is not None for r in rs) else None
        agg["dismissed"] = round(float(np.mean([r["user"]["dismissed"] for r in rs])), 1)
        agg["responded"] = round(float(np.mean([r["user"]["responded"] for r in rs])), 1)
        agg["false_flags_shown_to_user"] = round(float(np.mean([r["user"]["false_flags"] for r in rs])), 1)
        agg["final_p_act"] = round(float(np.mean([r["final_p_act"] for r in rs])), 3)
        agg["asks_month_1"] = round(float(np.mean([r["monthly_asks"][0] for r in rs])), 1)
        agg["asks_month_12"] = round(float(np.mean([r["monthly_asks"][-1] for r in rs])), 1)
        agg["overspend_reduction_vs_passive"] = None
        summary[p] = agg
    base = summary["none"]["total_overspend"]
    for p in summary:
        summary[p]["overspend_reduction_vs_passive"] = round(1 - summary[p]["total_overspend"] / base, 3) if base else None

    # plot: overspend and alerts per policy
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = [POLICY_LABELS[p] for p in summary]
    colors = ["#9e9e9e", "#f0a202", "#1f77b4"]
    axes[0].bar(names, [summary[p]["total_overspend"] for p in summary], color=colors)
    axes[0].set_title("Total overspend over 12 months (Rs, mean of seeds)")
    axes[1].bar(names, [summary[p]["alerts"] for p in summary], color=colors)
    axes[1].set_title("Budget alerts sent to user")
    axes[2].bar(names, [summary[p]["dismissed"] for p in summary], color=colors)
    axes[2].set_title("Alerts dismissed by user (annoyance)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "e3_policy_comparison.png", dpi=150); plt.close(fig)

    # plot: learning of p_act (true value 0.65)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs["utility"]:
        ax.plot(range(1, len(r["p_act_trace"]) + 1), r["p_act_trace"], alpha=0.7, marker="o", ms=3, label=f"seed {r['seed']}")
    ax.axhline(0.65, color="k", ls="--", label="true user responsiveness (0.65)")
    ax.set_xlabel("month after deployment"); ax.set_ylabel("agent's estimate of P(user acts on warning)")
    ax.set_title("Agent learns the user model online (Beta posterior mean)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "e3_user_model_learning.png", dpi=150); plt.close(fig)

    # plot: monthly overspend trajectories (seed 0)
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, c in zip(summary, colors):
        ax.plot(range(1, n_months + 1), runs[p][0]["monthly_overspend"], marker="o", ms=4, color=c, label=POLICY_LABELS[p])
    ax.set_xlabel("month"); ax.set_ylabel("overspend in month (Rs)"); ax.set_title("Monthly overspend, seed 0")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "e3_monthly_overspend.png", dpi=150); plt.close(fig)
    return {"summary": summary, "runs": runs}


# =============================================================== E4
def experiment_sensitivity(seeds=(0, 1, 2), n_months: int = 12, costs=(30, 60, 120, 200, 320)) -> dict:
    points = []
    for c in costs:
        rs = [run_policy(s, "utility", n_months, budget_policy=UtilityBudgetPolicy(fatigue_cost=c)) for s in seeds]
        points.append({"fatigue_cost": c, "overspend": round(float(np.mean([r["total_overspend"] for r in rs])), 0),
                       "alerts": round(float(np.mean([r["alerts"] for r in rs])), 1),
                       "dismissed": round(float(np.mean([r["user"]["dismissed"] for r in rs])), 1)})
    rule = [run_policy(s, "rule", n_months) for s in seeds]
    rule_pt = {"overspend": float(np.mean([r["total_overspend"] for r in rule])), "alerts": float(np.mean([r["alerts"] for r in rule]))}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([p["alerts"] for p in points], [p["overspend"] for p in points], marker="o", color="#1f77b4", label="utility agent (fatigue cost swept)")
    for p in points:
        ax.annotate(f"c={p['fatigue_cost']}", (p["alerts"], p["overspend"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([rule_pt["alerts"]], [rule_pt["overspend"]], color="#f0a202", s=70, zorder=5, label="rule-based agent")
    ax.set_xlabel("alerts per year"); ax.set_ylabel("overspend per year (Rs)")
    ax.set_title("Trade-off: nagging vs savings (one knob = user's annoyance cost)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "e4_sensitivity.png", dpi=150); plt.close(fig)
    return {"points": points, "rule_based": rule_pt}


# =============================================================== report
def write_results_md(res: dict) -> None:
    e1, e2, e3, e4 = res["E1_categorization"], res["E2_anomaly"], res["E3_policies"]["summary"], res["E4_sensitivity"]
    L = ["# Evaluation results (auto-generated by `python main.py evaluate`)", ""]
    L += ["## E1 Transaction categorisation", "",
          f"Train: {e1['n_train']} transactions (months 1-6). Test: {e1['n_test']} transactions (months 7-12), "
          f"of which {e1['n_test_unseen_merchant']} are from merchants never seen in training.", "",
          "| Model | Accuracy (all) | Seen merchants | Unseen merchants |", "|---|---|---|---|"]
    for name, m in e1["models"].items():
        L.append(f"| {name} | {m['accuracy_all']:.1%} | {m['accuracy_seen_merchants']:.1%} | {m['accuracy_unseen_merchants']:.1%} |")
    L += ["", "Online learning (accuracy on the last 300 test transactions):", ""]
    for k, v in e1["online_learning_final_accuracy_last300"].items():
        L.append(f"- {k}: {v:.1%}")
    L += ["", "![](e1_confusion_matrix.png)", "![](e1_online_learning.png)", ""]
    L += ["## E2 Anomaly / fraud detection", "",
          f"{e2['n_transactions']} test transactions over 5 seeds x 12 months, {e2['n_anomalies']} anomalous.", "",
          "| Detector | Precision@0.5 | Recall@0.5 | F1@0.5 | Avg precision | Flags raised |", "|---|---|---|---|---|---|"]
    for name, m in e2["detectors"].items():
        L.append(f"| {name} | {m['precision@0.5']:.2f} | {m['recall@0.5']:.2f} | {m['f1@0.5']:.2f} | {m['average_precision']:.2f} | {m['flags']} |")
    L += ["", "Recall of our detector by anomaly type: " +
          ", ".join(f"{k} {v:.0%}" for k, v in e2["detectors"]["ours"]["recall_by_type"].items()), "",
          "![](e2_anomaly_pr_curve.png)", ""]
    L += ["## E3 Agent policy comparison (12 months x 5 seeds, means)", "",
          "| Metric | Passive tracker | Rule-based agent | Utility agent (ours) |", "|---|---|---|---|"]
    for key, label in [("total_overspend", "Category overspend vs original budgets (Rs)"), ("overspend_reduction_vs_passive", "Overspend reduction vs passive"),
                       ("net_total_overspend", "Net overspend of the whole monthly budget (Rs)"),
                       ("months_with_overspend", "Months with any overspend"), ("alerts", "Budget alerts"),
                       ("alert_precision", "Alert precision (category truly heading over)"), ("dismissed", "Alerts dismissed (annoyance)"),
                       ("responded", "Alerts the user acted on"), ("reallocations", "Budget reallocations"),
                       ("fraud_tp", "Anomalies caught"), ("fraud_fp", "False fraud flags"), ("anomalies_present", "Anomalies present"),
                       ("final_p_act", "Learned P(user acts) [true 0.65]"),
                       ("asks_month_1", "Category questions asked, month 1"), ("asks_month_12", "Category questions asked, month 12")]:
        def fmt(v):
            if v is None:
                return "-"
            return f"{v:.1%}" if key in ("overspend_reduction_vs_passive", "alert_precision") else f"{v:g}"
        L.append(f"| {label} | {fmt(e3['none'][key])} | {fmt(e3['rule'][key])} | {fmt(e3['utility'][key])} |")
    L += ["", "![](e3_policy_comparison.png)", "![](e3_monthly_overspend.png)",
          "![](e3_user_model_learning.png)", ""]
    L += ["## E4 Sensitivity to the annoyance cost", "", "| fatigue cost | alerts / year | overspend / year (Rs) | dismissed |", "|---|---|---|---|"]
    for p in e4["points"]:
        L.append(f"| {p['fatigue_cost']} | {p['alerts']} | {p['overspend']:.0f} | {p['dismissed']} |")
    L += [f"| rule-based | {e4['rule_based']['alerts']:.1f} | {e4['rule_based']['overspend']:.0f} | - |", "",
          "![](e4_sensitivity.png)", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def main(quick: bool = False) -> dict:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    seeds = (0, 1) if quick else (0, 1, 2, 3, 4)
    res = {}
    print("E1 categorisation ..."); res["E1_categorization"] = experiment_categorizer(0)
    print("E2 anomaly detection ..."); res["E2_anomaly"] = experiment_anomaly(seeds)
    print("E3 policy comparison ..."); res["E3_policies"] = experiment_policies(seeds)
    print("E4 sensitivity ..."); res["E4_sensitivity"] = experiment_sensitivity(seeds[:3] if not quick else seeds)
    res["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_results_md(res)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))
    print(f"done in {res['runtime_seconds']} s -> {OUT}")
    return res
