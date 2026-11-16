# Presentation outline (8 slides, about 10 minutes)

This matches `SpendSense_Presentation.pptx`. Every slide has speaker notes in the
file; this page is the short version to rehearse from. Aim for a little over a
minute per slide and keep two minutes for the live demo.

1. **Title.** SpendSense: An Intelligent AI Financial Assistant. Point at the
   dashboard on the right: it previews what the agent produces, a budget
   projection, categorised payments, and risk decisions with evidence.

2. **Problem and solution.** Four problems a bank app does not solve, each paired
   with the component that addresses it: cryptic statements and Naive Bayes,
   noisy spending and the Bayesian forecast, hidden fraud and the noisy-OR score,
   alert fatigue and expected-utility decisions. Stress the last pair: deciding
   when to speak matters as much as what to say.

3. **Agent architecture.** The six-step loop: perceive, interpret, update
   beliefs, decide, act, learn, with feedback closing the loop. Then PEAS and the
   agent type: model-based, utility-based, learning. Mention that three policies
   share this loop, so the evaluation isolates the decision layer.

4. **Core AI algorithms.** Five cards, one formula and one result each: Naive
   Bayes (87.9% on six unseen months), noisy-OR (F1 0.81 vs 0.40), Bayesian
   forecast, expected utility (71% alert precision), online learning. Say why
   median and MAD replace mean and standard deviation.

5. **How SpendSense decides.** Two lanes. Per payment: score, calibrate p = s²,
   compare the expected cost of NONE, VERIFY and BLOCK. Per category: projection,
   risk, utilities of INFO, WARN and REALLOCATE against doing nothing. Read the
   real Food warning on the right and its utilities.

6. **Live demonstration.** Four moments from November 2025: a blocked Rs 3,375
   cinema charge, a held night-time payment to an unknown merchant, a budget
   re-plan, and a late overspending warning. KPI strip: 5 of 5 anomalies caught,
   Rs 386 overspend with the agent against Rs 1,196 without.

7. **Performance and evaluation.** 12 months × 5 seeds against a passive tracker
   and a rule-based agent. Headlines: 55.3% less overspend than passive tracking,
   70.6% alert precision against 57.4%, 22 of 25 anomalies caught, ₹3,771 net
   monthly-budget overspend per year against ₹27,629. Explain the counterfactual:
   every day has its own seed, so a month is replayed without the agent.

8. **Limitations, future scope and conclusion.** Say the limitations before
   anyone asks: synthetic environment, responsiveness estimate biased upward,
   54% on never-seen merchants. Future: randomised holdout warnings, real data
   through the Account Aggregator framework, text embeddings. Close on the
   quote and invite questions.

## Live demo plan (2 minutes)

Run `python main.py demo`, scroll to one BLOCK and one WARN, then to the month
summary. Keep `outputs/demo_log.txt` open as a backup in case the terminal is
slow.

## If asked for more detail

The figures in `outputs/` are not on the slides but are in the report: the
confusion matrix, the online-learning curve, the precision-recall curve, the
user-model learning plot and the sensitivity trade-off. `VIVA_NOTES.md` has
spoken answers to the likely questions.
