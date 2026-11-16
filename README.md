# SpendSense: a Smart Expense & Budget Agent

An AI agent that watches a person's bank / UPI transactions, categorises them,
forecasts where each budget category will end the month, detects suspicious
payments, and decides *whether, when and how* to intervene, weighing the money
it could save against the annoyance of nagging the user. It keeps learning
from the user: how responsive they are, which merchants are genuine, and how to
categorise new shops.

Everything is implemented from scratch in Python (numpy, pandas, matplotlib
only): Multinomial Naive Bayes, robust anomaly scoring, Bayesian shrinkage
forecasting, expected-utility decision making and Beta-Bernoulli learning.

| Document | What it contains |
|---|---|
| [REPORT.md](REPORT.md) | Problem definition, agent architecture, algorithms, decision process, demonstration, evaluation |
| [outputs/RESULTS.md](outputs/RESULTS.md) | Auto-generated evaluation tables and figures |
| [outputs/demo_log.txt](outputs/demo_log.txt) | One full month of the agent, day by day, with its reasoning |
| [VIVA_NOTES.md](VIVA_NOTES.md) | Likely viva questions with answers |
| [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md) | Slide-by-slide outline for the presentation |

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py demo
```
Runs one month of the agent on a simulated user and prints every transaction,
the predicted category with confidence, anomaly evidence, the actions chosen
and the utilities behind them. Saves `outputs/demo_log.txt` and
`outputs/demo_timeline.png`.

```bash
python main.py evaluate
```
Runs the four experiments (about 15 seconds) and writes `outputs/RESULTS.md`,
`outputs/results.json` and the figures.

```bash
python main.py generate
```
Writes `data/sample_transactions.csv` (labelled) and
`data/statement_unlabelled.csv` (bank-export format).

```bash
python main.py csv --path data/statement_unlabelled.csv
```
Runs the agent over a statement export of your own (columns `date`,
`description`, `amount`, optional `time`).

```bash
python tests/test_smoke.py
```

## Project layout

```
main.py                      command-line entry point
expense_agent/
  data_gen.py                synthetic UPI/bank environment with injected anomalies
  categorizer.py             Multinomial Naive Bayes (online) + keyword-rule baseline
  anomaly.py                 noisy-OR anomaly scorer + two baselines
  forecaster.py              end-of-month projection with Bayesian shrinkage
  decision.py                expected-utility policies (budget, fraud) + rule baselines
  agent.py                   the agent loop: perceive, interpret, update, decide, act, learn
  simulation.py              simulated user and month runner
  evaluate.py                experiments E1-E4, plots, RESULTS.md
  demo.py                    day-by-day demonstration, CSV mode
  diagram.py                 draws outputs/architecture.png
tests/test_smoke.py          smoke tests
outputs/                     figures, logs and results (generated)
data/                        sample datasets (generated)
```

## Headline results (12 months x 5 seeds, see REPORT.md for details)

| | Passive tracker | Rule-based agent | **Utility agent (ours)** |
|---|---|---|---|
| Overspend vs category budgets (Rs / year) | 67,214 | 35,292 | **30,039** (-55%) |
| Net overspend of the whole monthly budget (Rs / year) | 27,629 | 6,405 | **3,771** |
| Budget alerts per year | 0 | 59.8 | 60.4 |
| Alert precision | - | 57% | **71%** |
| Anomalies caught (of 25) / false flags | 0 | 21 / 5 | **22** / 7.4 |

Categorisation accuracy 87.9% on six unseen months (54% on never-seen
merchants, vs 25% without the brand-dictionary hint); anomaly detection
F1 0.81 vs 0.40 for the best baseline.
