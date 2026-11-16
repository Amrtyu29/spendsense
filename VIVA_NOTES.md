# Viva preparation notes

Short, direct answers you can say out loud. Numbers are from `outputs/RESULTS.md`.

## The problem and the agent

**Q. What problem does the agent solve?**
People overspend without noticing until the month is over. Bank apps only show
a list of transactions. The agent reads that stream, categorises it, forecasts
where each category will end the month, catches suspicious payments, and
decides when a nudge is worth the interruption. It also learns about the user.

**Q. Why is this an *agent* and not just a program?**
It sits in a perceive-decide-act loop with an environment (the transaction
stream and the user), it keeps internal state (spend, projections, a user
model), it chooses actions by maximising expected utility, and it changes its
behaviour over time from feedback. In Russell & Norvig's taxonomy it is a
model-based, utility-based, learning agent.

**Q. Give the PEAS description.**
Performance: money kept within budget, fraud losses avoided, minus user
annoyance. Environment: the user's transactions and the user, partially
observable (we never see intent or future spending), stochastic, sequential,
dynamic. Actuators: messages (info, warn), holds on payments, questions,
budget reallocation. Sensors: timestamp, statement text, amount, and the
user's feedback.

**Q. How would the agent get real data? Can it read Google Pay or PhonePe?**
Not directly: UPI apps expose no data to third-party software. Three realistic
routes. First, a statement export, which every bank and UPI app provides as CSV
or PDF; this project reads one with `python main.py csv`. Second, bank SMS
parsing: every payment triggers an SMS such as "Rs 250 debited to SWIGGY via
UPI", and an Android app with SMS permission can parse it in real time; Walnut,
Axio and CRED launched this way. Third, the Account Aggregator framework, the
RBI-regulated consent-based API run by Sahamati with aggregators such as Finvu
and OneMoney; the user consents once and the bank sends transactions over a
standard encrypted channel. The agent is independent of the source: it only
needs timestamp, description and amount, and all processing stays on the
device.

**Q. Why not just rules like "warn at 80% of budget"?**
That is our baseline. It ignores how much of the month is left, how uncertain
the projection is, whether the user ever responds, and how many alerts were
sent recently. Our utility agent reduces overspend by 55% against 47% for the
rule and its alerts are right 71% of the time against 57%, with the same
number of alerts.

## Algorithms

**Q. Explain Naive Bayes as you use it.**
Multinomial Naive Bayes over features of a transaction: words and bigrams of
the cleaned description, an amount bucket (floor of log2 amount), a
time-of-day token, and the guess of a brand dictionary as one more token.
P(c | x) is proportional to P(c) times the product of P(feature | c), with
Laplace smoothing alpha = 0.5. We keep running counts, so one user correction
is a constant-time update, weighted 3 because a human label is worth more than
one noisy example.

**Q. Why the brand-dictionary hint inside Naive Bayes instead of using rules?**
Rules alone score 54% because statements show legal entity names (BUNDL
TECHNOLOGIES for Swiggy). Learning alone scores 25% on merchants never seen.
Adding the rule's guess as a feature lets the model learn how far to trust it:
88% overall and 54% on never-seen merchants.

**Q. How does the anomaly detector work?**
Four evidence sources combined with a noisy-OR: a robust z-score of log amount
inside the predicted category (median and MAD, so one old outlier does not
inflate the scale), a burst count of payments to the same merchant in 30
minutes, a duplicate check (same merchant, same amount within 5 minutes), and
novelty of the merchant, worse at night. Noisy-OR means independent pieces of
evidence compound: score = 1 - product of (1 - s_i).

**Q. Why median and MAD instead of mean and standard deviation?**
Spending amounts are heavy-tailed. One large legitimate purchase would inflate
the standard deviation and hide the next anomaly. Median and MAD have a 50%
breakdown point.

**Q. How do you forecast end-of-month spend?**
A shrinkage estimate. Early in the month we trust the historical mean, later
the observed run-rate. remaining = w * rate-based + (1 - w) * prior, with w =
frac^0.8. The uncertainty shrinks as the month progresses. Fixed categories
like rent are handled as max(spent, historical mean).

**Q. Explain the decision rule for a warning.**
From the projection N(mean, sd) and budget B we compute P(over) and the
expected overspend E[max(X - B, 0)] in closed form. A warning only saves money
the user has not spent yet and only if they act, so benefit = p_act x
min(E[over], 0.3 x remaining). Cost = fatigue cost x (1 + 0.5 x alerts in the
last 7 days), scaled by a learned factor. We also score INFO (weaker, cheaper)
and REALLOCATE, and pick the maximum, acting only if it beats doing nothing by
a margin and the cooldown allows.

**Q. Why is REALLOCATE gated so strictly?**
Moving budget does not change behaviour, so it only makes sense when the
budget was unrealistic: P(over) at least 0.7, projection at least 20% over,
enough days left, and donors that are safely under budget. Otherwise it would
fire constantly and hide real overspend. We also measure overspend against the
original budgets so reallocation cannot flatter the agent.

**Q. How is fraud handled?**
Expected cost of three options. NONE costs p x (amount + follow-on loss),
VERIFY costs a fixed interruption plus a small residual, BLOCK costs (1 - p) x
(false-block cost + half the amount). p = score squared, a calibration so weak
evidence must compound. Flagged payments are held out of the totals until the
user confirms them.

**Q. What does the agent learn online?**
Four things. p_act, the probability the user acts on a warning, as a Beta
posterior updated by comparing spend rate after a warning with a reference
rate. A fatigue scale from dismiss/helpful feedback. The categoriser from
corrections. A merchant whitelist from "this is genuine" answers.

**Q. Why a Beta distribution for p_act?**
It is the conjugate prior of a Bernoulli outcome (acted / did not act). Prior
Beta(3, 2) encodes a mean of 0.6 with low confidence; each observation adds
one to alpha or beta; the posterior mean is alpha / (alpha + beta).

## Evaluation

**Q. Where does the data come from?**
A simulator of a Kerala student's UPI and bank transactions: ten categories,
realistic merchant aliases, weekend effects, month-level variation and shock
months, plus four kinds of injected anomalies. Every day is generated from its
own seed, so the same month can be replayed with and without the agent. That
gives a clean counterfactual, which real data cannot.

**Q. Is synthetic data a weakness?**
Yes, and we say so. Real statements are messier and users are less
predictable. The design is ready for real data: the CSV mode runs the agent
on a bank export. The simulator was needed to measure counterfactual
overspend and to have ground-truth anomaly labels.

**Q. What are the baselines?**
Categorisation: keyword rules and ablations of our own model. Anomalies: fixed
threshold on amount, and a global z-score without category or context. Policy:
a passive tracker and a rule-based agent that warns at 80% of budget and flags
at a fixed score, sharing the same perception so only the decision layer
differs.

**Q. Your learned p_act is 0.77 but the true value is 0.65. Why?**
The estimate is observational. A warning is triggered by unusually high
spending, which tends to fall back on its own, so some "responses" are
regression to the mean. We partly correct for it by comparing against the
geometric mean of the pre-warning rate and the user's normal rate. A proper
fix is to randomly withhold a small fraction of warnings and compare, which is
an A/B test; that is listed as future work.

**Q. Burst recall is only 65%. Why?**
By construction the first two payments of a burst look normal; the detector
flags from the third onwards. Per burst event, detection is near 100%; per
transaction it is 65%.

**Q. Why does the sensitivity curve matter?**
The fatigue cost is the one knob that encodes the user's tolerance. Sweeping it
traces a frontier from 33 alerts and Rs 31k overspend to 113 alerts and Rs 27k.
The rule-based agent sits above that frontier: more overspend for the same
number of alerts. The user can pick their point on the curve.

## Limitations and future work (say these before they are asked)

- Synthetic environment; next step is a study on real anonymised statements.
- Causal estimate of responsiveness is confounded; use randomised holdouts.
- Naive Bayes assumes independent features; a small linear model or a
  transformer embedding of the description would improve new-merchant accuracy.
- Budgets of tiny categories (Education, Rs 200) are often breached by one
  purchase; a floor on budgets or a shared "misc" envelope would help.
- Privacy: everything runs locally on the user's device by design.
