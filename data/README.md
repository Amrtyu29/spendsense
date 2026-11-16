# Data in this folder

All of it is simulated. No real person's transactions are included.

## Demo statements (`demo/`)

Three one-month statements, ready to use in the app. In the sidebar choose
**My statement (CSV)**, then pick one from the **Statement** list.

| File | Month | What it shows | What the agent does |
|---|---|---|---|
| `1_normal_month.csv` | July 2026 | An ordinary month: 96 payments, ends within every budget, nothing suspicious | Three cautionary warnings and one budget re-plan; no payment held. A good example that a warning is a forecast, not a verdict |
| `2_overspending_month.csv` | February 2026 | Spending drifts about Rs 3,170 over budget across several categories | Warnings on Transport and Groceries from day 5, two budget re-plans, a late warning with a daily cap |
| `3_suspicious_payments.csv` | October 2026 | Three suspicious payments of three different kinds among 87 | Holds exactly those three: an unusually large payment, a duplicate charge, and a night-time payment to an unknown merchant. No false flags |

`demo/answers/` has the same three months with two extra columns, the true
category and whether the payment was injected as suspicious. Use them to check
the agent's guesses, or to show the examiner the ground truth.

In all three the agent meets merchants it has never seen, so it asks about
some categories. Correct a few in the **Transactions** tab to show it learning.

## Your own statement

`demo/my_statement_template.csv` shows the format. The app needs three columns
and accepts a fourth:

| Column | Required | Example |
|---|---|---|
| `date` | yes | `2026-11-01` |
| `description` | yes | `UPI/CAMPUS CANTEEN/1234567890` |
| `amount` | yes | `85` |
| `time` | no | `08:15` |

Export a statement from your bank or UPI app as CSV, rename the columns to
match, and choose **Upload my own file** in the app. Everything stays on your
computer.

## Other files

| File | What it is |
|---|---|
| `statement_unlabelled.csv` | Twelve months in bank-export format (the "Sample: twelve months" entry in the app) |
| `sample_transactions.csv` | The same twelve months with true categories and anomaly labels |

## Regenerating

```bash
python main.py demodata
```

```bash
python main.py generate
```

The demo months are not hand-written. The generator scans simulated months from
the environment the agent was trained on and keeps the best match for each
description, judged both on the data and on how the agent behaves, so the files
are reproducible.
