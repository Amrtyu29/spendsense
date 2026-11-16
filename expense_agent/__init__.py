"""SpendSense: a smart expense and budget agent.

Modules
-------
data_gen     synthetic UPI / bank transaction environment (with injected anomalies)
categorizer  Multinomial Naive Bayes text categoriser (from scratch, online learning)
anomaly      robust per-category anomaly detector (noisy-OR of evidence sources)
forecaster   end-of-month spend projection with Bayesian shrinkage
decision     expected-utility decision policies (budget alerts, fraud handling)
agent        the agent loop: perceive -> categorise -> update belief -> decide -> act -> learn
simulation   simulated user + environment for evaluation
evaluate     experiments, baselines and plots
"""

__version__ = "1.0.0"
