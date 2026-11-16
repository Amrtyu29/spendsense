"""End-of-month spend projection.

For each category the agent keeps a prior from the user's labelled history:
mu0 = mean monthly total, sd0 = *robust* spread of monthly totals
(1.4826 x median absolute deviation, so one unusual month does not inflate it).

During the month the remaining spend is a shrinkage estimate between what the
prior says is still to come and what the current run-rate says:

    frac            = day / days_in_month
    prior_remaining = mu0 * (1 - frac)
    rate_remaining  = 0.5 * (spent / frac - spent)  +  0.5 * recent_daily_rate * days_left
    w               = frac ** 0.8          (halved for sparse categories, <= 8 txns / month)
    remaining       = w * rate_remaining + (1 - w) * prior_remaining
    mean            = spent + remaining
    sd              = sqrt( (sd0 * (1 - frac))^2 + (0.15 * remaining)^2 ) + 1

Fixed categories (rent, bills: few payments, tiny month-to-month variation)
cannot be forecast from a run-rate; their projection is max(spent, mu0).
"""
from __future__ import annotations

import math
import statistics

import pandas as pd


class SpendForecaster:
    def __init__(self):
        self.mu0: dict[str, float] = {}
        self.sd0: dict[str, float] = {}
        self.txn_rate: dict[str, float] = {}
        self.fixed: dict[str, bool] = {}
        self.sparse: dict[str, bool] = {}

    def fit(self, history: pd.DataFrame) -> "SpendForecaster":
        """history: labelled months with columns category, amount, month_idx, is_anomaly."""
        normal = history[~history["is_anomaly"]] if "is_anomaly" in history else history
        months = sorted(normal["month_idx"].unique())
        for cat, g in normal.groupby("category"):
            totals = [float(g[g["month_idx"] == m]["amount"].sum()) for m in months]
            mu = statistics.mean(totals)
            med = statistics.median(totals)
            mad = statistics.median(abs(t - med) for t in totals)
            sd = 1.4826 * mad if mad > 0 else (statistics.stdev(totals) if len(totals) > 1 else 0.2 * mu)
            self.mu0[cat] = mu
            self.sd0[cat] = min(max(sd, 0.08 * mu), 0.40 * mu)
            per_month = len(g) / len(months)
            self.txn_rate[cat] = per_month
            self.fixed[cat] = per_month <= 4 and (self.sd0[cat] / mu) < 0.15
            self.sparse[cat] = per_month <= 8
        return self

    def project(self, category: str, spent: float, day: int, days_in_month: int,
                recent_daily_rate: float | None = None) -> tuple[float, float]:
        mu0 = self.mu0.get(category, max(spent, 1.0))
        sd0 = self.sd0.get(category, 0.25 * mu0)
        frac = day / days_in_month
        days_left = days_in_month - day
        if self.fixed.get(category, False):
            return max(spent, mu0), max(sd0 * (1 - frac), 0.02 * mu0 + 1.0)
        prior_remaining = mu0 * (1 - frac)
        if day < 3:
            remaining = prior_remaining
        else:
            rate = recent_daily_rate if recent_daily_rate is not None else spent / day
            rate_remaining = 0.5 * (spent / frac - spent) + 0.5 * rate * days_left
            w = (frac ** 0.8) * (0.5 if self.sparse.get(category, False) else 1.0)
            remaining = w * rate_remaining + (1 - w) * prior_remaining
        remaining = max(remaining, 0.0)
        mean = spent + remaining
        sd = math.sqrt((sd0 * (1 - frac)) ** 2 + (0.15 * remaining) ** 2) + 1.0
        return mean, sd

    def suggest_budgets(self, factor: float = 1.10, round_to: int = 50) -> dict[str, float]:
        return {c: float(round(self.mu0[c] * factor / round_to) * round_to) for c in self.mu0}
