# Presentation outline (12 slides, about 10 minutes)

Figures are in `outputs/`. Keep text on slides short; say the rest.

1. **Title.** SpendSense: a Smart Expense & Budget Agent. Name, roll number, course.

2. **The problem.** People overspend without noticing. Bank apps show a list, not a
   decision. Three things a human money-coach does: categorise, forecast, intervene
   at the right moment. Fraud catches on top. One line on scope: personal budgets,
   local on-device.

3. **PEAS and agent type.** Performance, Environment, Actuators, Sensors table.
   Model-based, utility-based, learning agent. Partially observable, stochastic,
   sequential environment.

4. **Architecture.** `outputs/architecture.png`. Walk the six stages:
   perceive, interpret, state, decide, act, learn.

5. **Perception and categorisation.** Features (tokens, bigrams, amount bucket,
   time of day, brand hint). Multinomial Naive Bayes formula with smoothing.
   Online update from corrections. Table: 88% ours, 80% without hint, 54% rules.
   `outputs/e1_online_learning.png`.

6. **Anomaly scoring.** Four evidence sources and the noisy-OR formula. Why
   median/MAD. `outputs/e2_anomaly_pr_curve.png`: F1 0.81 vs 0.40.

7. **Forecasting.** Shrinkage between prior and run-rate; uncertainty shrinks with
   the month. One equation and a sketch.

8. **Decision making.** P(over), E[overspend] in closed form. Utilities of NONE,
   INFO, WARN, REALLOCATE. Fatigue cost, cooldown, margin. Fraud: NONE / VERIFY /
   BLOCK expected cost. Show one real utility line from the demo log.

9. **Learning.** Beta posterior for p_act, fatigue scale, categoriser updates,
   merchant whitelist. `outputs/e3_user_model_learning.png`. Mention the
   regression-to-the-mean caveat yourself.

10. **Demonstration.** 6 to 8 lines from `outputs/demo_log.txt`: a BLOCK with its
    evidence, a WARN with utilities, the month summary table.
    `outputs/demo_timeline.png` (with vs without agent).

11. **Evaluation.** `outputs/e3_policy_comparison.png` and `outputs/e4_sensitivity.png`.
    Headline: 55% less overspend vs 47% for rules, alert precision 71% vs 57%,
    net overspend Rs 3.8k vs 6.4k vs 27.6k. Say what the baselines are and how
    the counterfactual is obtained.

12. **Limitations, future work, conclusion.** Synthetic data, confounded
    responsiveness estimate, Naive Bayes independence, tiny budgets. Future:
    randomised holdout warnings, real statements, richer text model, notification
    timing as a bandit.

Live demo plan (2 minutes): run `python main.py demo`, scroll to one BLOCK and one
WARN, then to the summary. Have the log file open as a backup in case the
terminal is slow.
