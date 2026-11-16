"""SpendSense - Smart Expense & Budget Agent.

Usage
-----
python main.py app                    open the dashboard in your browser
python main.py demo [--seed 7]        one month, day by day, with explanations
python main.py evaluate [--quick]     all experiments -> outputs/RESULTS.md + plots
python main.py generate [--months 12] write data/sample_transactions.csv
python main.py csv --path FILE        run the agent on your own statement export
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_generate(months: int) -> None:
    from expense_agent.data_gen import TransactionSimulator
    (ROOT / "data").mkdir(exist_ok=True)
    df = TransactionSimulator(seed=0).generate_history(2025, 5, months, 0)
    out = ROOT / "data" / "sample_transactions.csv"
    cols = ["txn_id", "ts", "description", "merchant", "amount", "category", "is_anomaly", "anomaly_type", "month_idx"]
    df[cols].to_csv(out, index=False)
    # also an unlabelled version in the format a bank export would have
    pub = df[["ts", "description", "amount"]].copy()
    pub["date"] = pub["ts"].dt.strftime("%Y-%m-%d")
    pub["time"] = pub["ts"].dt.strftime("%H:%M")
    pub[["date", "time", "description", "amount"]].to_csv(ROOT / "data" / "statement_unlabelled.csv", index=False)
    print(f"wrote {len(df)} transactions to {out} (labelled) and data/statement_unlabelled.csv (bank-export format)")
    print(df.groupby("category")["amount"].agg(["count", "sum"]).round(0))
    print(f"anomalies: {int(df.is_anomaly.sum())} ({df.is_anomaly.mean():.1%}) by type:\n{df[df.is_anomaly].anomaly_type.value_counts()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo"); p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("evaluate"); p.add_argument("--quick", action="store_true")
    p = sub.add_parser("generate"); p.add_argument("--months", type=int, default=12)
    p = sub.add_parser("csv"); p.add_argument("--path", required=True)
    sub.add_parser("app")
    args = ap.parse_args()
    if args.cmd == "app":
        import subprocess
        import sys
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                                          "--browser.gatherUsageStats", "false"]))
    if args.cmd == "demo":
        from expense_agent.demo import run_demo
        run_demo(args.seed)
    elif args.cmd == "evaluate":
        from expense_agent.evaluate import main as evaluate_main
        evaluate_main(quick=args.quick)
    elif args.cmd == "generate":
        cmd_generate(args.months)
    elif args.cmd == "csv":
        from expense_agent.demo import run_on_csv
        run_on_csv(args.path)


if __name__ == "__main__":
    main()
