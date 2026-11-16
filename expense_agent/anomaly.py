"""Anomaly / fraud evidence scoring.

AnomalyDetector combines four independent evidence sources with a noisy-OR:

    score = 1 - (1 - s_amount)(1 - s_burst)(1 - s_duplicate)(1 - s_novelty)

* s_amount    robust z-score of log(amount) within the predicted category,
              using median and MAD (median absolute deviation), so a few large
              historical outliers do not inflate the scale.
* s_burst     several payments to the same merchant inside 30 minutes
              (card-testing pattern).
* s_duplicate the same merchant and the same amount inside 5 minutes.
* s_novelty   a merchant whose tokens were never seen in history, worse when
              the payment happens at night (00:00-05:59).

Two baselines are provided for the evaluation section.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime

from .categorizer import clean_tokens


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def merchant_key(description: str) -> frozenset:
    return frozenset(clean_tokens(description))


class AnomalyDetector:
    def __init__(self, amount_z0: float = 3.2, amount_k: float = 2.5):
        self.amount_z0 = amount_z0
        self.amount_k = amount_k
        self.stats: dict[str, tuple[float, float]] = {}
        self.global_stats: tuple[float, float] = (math.log(100), 0.7)
        self.known_tokens: set[str] = set()

    def fit(self, rows) -> "AnomalyDetector":
        by_cat: dict[str, list[float]] = {}
        all_logs: list[float] = []
        for r in rows:
            la = math.log(max(r["amount"], 1.0))
            by_cat.setdefault(r["category"], []).append(la)
            all_logs.append(la)
            self.known_tokens.update(clean_tokens(r["description"]))
        for c, vals in by_cat.items():
            med = statistics.median(vals)
            mad = statistics.median(abs(v - med) for v in vals) if len(vals) > 1 else 0.3
            self.stats[c] = (med, max(mad, 0.05))
        med = statistics.median(all_logs)
        mad = statistics.median(abs(v - med) for v in all_logs)
        self.global_stats = (med, max(mad, 0.05))
        return self

    def mark_legit(self, description: str) -> None:
        """User confirmed a flagged merchant is genuine: learn it."""
        self.known_tokens.update(clean_tokens(description))

    def score(self, txn: dict, category: str, recent: list[dict]) -> tuple[float, dict]:
        """txn: dict(ts, description, amount); recent: earlier txns today/yesterday
        each with keys ts, key, amount.  Returns (score in [0,1], evidence)."""
        amount = txn["amount"]
        ts: datetime = txn["ts"]
        la = math.log(max(amount, 1.0))
        med, mad = self.stats.get(category, self.global_stats)
        z = (la - med) / (1.4826 * mad)
        s_amount = _sigmoid(self.amount_k * (z - self.amount_z0)) if z > 0 else 0.0

        key = merchant_key(txn["description"])
        same = [r for r in recent if r["key"] == key and 0 <= (ts - r["ts"]).total_seconds() <= 1800]
        n = len(same) + 1
        s_burst = _sigmoid(2.0 * (n - 3)) if n >= 3 else 0.0

        dup = any(r["key"] == key and abs(r["amount"] - amount) < 0.01
                  and 0 <= (ts - r["ts"]).total_seconds() <= 300 for r in recent)
        s_dup = 0.85 if dup else 0.0

        # novel if at least half of the merchant tokens were never seen in history
        novel = len(key) > 0 and len(key - self.known_tokens) * 2 >= len(key)
        night = ts.hour < 6
        # a new merchant alone is common (people try new shops); new + night is not
        s_nov = 0.70 if (novel and night) else 0.12 if novel else 0.08 if night else 0.0

        score = 1.0 - (1 - s_amount) * (1 - s_burst) * (1 - s_dup) * (1 - s_nov)
        evidence = {"amount_z": round(z, 2), "s_amount": round(s_amount, 2), "burst_n": n,
                    "s_burst": round(s_burst, 2), "duplicate": dup, "novel_merchant": novel,
                    "night": night, "s_novelty": round(s_nov, 2)}
        return score, evidence

    @staticmethod
    def explain(evidence: dict) -> str:
        reasons = []
        if evidence["s_amount"] >= 0.3:
            reasons.append(f"amount is {evidence['amount_z']:.1f} robust-SD above normal for this category")
        if evidence["s_burst"] >= 0.3:
            reasons.append(f"{evidence['burst_n']} payments to the same merchant within 30 min")
        if evidence["duplicate"]:
            reasons.append("same amount to same merchant within 5 min (possible double charge)")
        if evidence["novel_merchant"]:
            reasons.append("merchant never seen before")
        if not reasons and evidence["s_amount"] > 0.05:
            reasons.append(f"amount somewhat high for this category ({evidence['amount_z']:.1f} robust-SD)")
        if evidence["night"]:
            reasons.append("payment between midnight and 6 am")
        return "; ".join(reasons) if reasons else "no strong evidence"


class FixedThresholdDetector:
    """Baseline 1: flag if amount > k * overall mean amount."""

    def __init__(self, k: float = 3.0):
        self.k = k
        self.mean = 100.0

    def fit(self, rows):
        amounts = [r["amount"] for r in rows]
        self.mean = sum(amounts) / len(amounts)
        return self

    def score(self, txn, category=None, recent=None):
        s = min(1.0, txn["amount"] / (self.k * self.mean))
        return (s if s >= 1.0 else s * 0.49), {}


class GlobalZDetector:
    """Baseline 2: same robust z-score logic but ignoring category and context."""

    def __init__(self, z0: float = 2.5, k: float = 1.5):
        self.z0, self.k = z0, k
        self.med, self.mad = math.log(100), 0.7

    def fit(self, rows):
        logs = [math.log(max(r["amount"], 1.0)) for r in rows]
        self.med = statistics.median(logs)
        self.mad = max(statistics.median(abs(v - self.med) for v in logs), 0.05)
        return self

    def score(self, txn, category=None, recent=None):
        z = (math.log(max(txn["amount"], 1.0)) - self.med) / (1.4826 * self.mad)
        return (_sigmoid(self.k * (z - self.z0)) if z > 0 else 0.0), {"amount_z": round(z, 2)}
