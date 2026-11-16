"""Transaction categorisation.

NaiveBayesCategorizer  Multinomial Naive Bayes implemented from scratch with
                       Laplace smoothing.  Supports *online* updates so the
                       agent keeps learning from user corrections.
KeywordRuleCategorizer A hand-written keyword dictionary (what a typical
                       budgeting app ships).  Used as a baseline.

Features extracted from a transaction
-------------------------------------
* word unigrams and bigrams of the cleaned statement description
* an amount bucket token  (amt_<floor(log2 amount)>)
* a time-of-day token     (hr_morning / hr_afternoon / hr_evening / hr_night)
The amount and time tokens let the model separate e.g. a Rs.6000 "transfer"
(rent) from a Rs.400 one (friend), and give it something to work with when the
merchant text is entirely new.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

GENERIC_TOKENS = {
    "upi", "pos", "payment", "to", "neft", "imps", "card", "ref", "okaxis", "ltd",
    "pvt", "inc", "india", "online", "pmt", "txn", "the", "and",
    # location words carry no merchant information
    "palakkad", "coimbatore", "kochi", "bangalore", "thrissur", "chennai", "mumbai", "delhi",
}
_NON_ALPHA = re.compile(r"[^a-z]+")


def clean_tokens(description: str) -> list[str]:
    text = _NON_ALPHA.sub(" ", description.lower())
    return [t for t in text.split() if len(t) >= 2 and t not in GENERIC_TOKENS]


def hour_bucket(hour: int) -> str:
    if hour < 6:
        return "hr_night"
    if hour < 12:
        return "hr_morning"
    if hour < 17:
        return "hr_afternoon"
    return "hr_evening"


def featurize(description: str, amount: float | None = None, hour: int | None = None,
              context_weight: float = 0.5) -> list[tuple[str, float]]:
    """Returns (feature, weight) pairs.  Text features weigh 1; the amount and
    time-of-day tokens weigh `context_weight` so that one strong merchant word
    is not overruled by an unusual amount."""
    toks = clean_tokens(description)
    feats = [(t, 1.0) for t in toks]
    feats += [(a + "_" + b, 1.0) for a, b in zip(toks, toks[1:])]
    if amount is not None:
        feats.append((f"amt_{int(math.log2(max(amount, 1.0)))}", context_weight))
    if hour is not None:
        feats.append((hour_bucket(hour), context_weight))
    return feats


class NaiveBayesCategorizer:
    """Multinomial Naive Bayes.

    P(c | x) proportional to P(c) * prod_t P(t | c)^count(t)
    with P(t | c) = (N_tc + alpha) / (N_c + alpha * |V|)
    All sums are kept as running counts so update() is O(#features).
    """

    def __init__(self, alpha: float = 0.5, use_context: bool = True, context_weight: float = 0.5,
                 rules: "KeywordRuleCategorizer | None" = None):
        self.alpha = alpha
        self.use_context = use_context   # False = text only (ablation)
        self.context_weight = context_weight
        # Optional prior knowledge: a brand dictionary.  Its guess is added as one
        # more feature ("rule_Food"), so the model *learns* how far to trust it and
        # can still categorise brands the user has never paid before.
        self.rules = rules
        self.class_counts: Counter = Counter()
        self.token_counts: dict[str, Counter] = defaultdict(Counter)
        self.class_token_totals: Counter = Counter()
        self.vocab: set[str] = set()
        self.n_docs = 0
        self.n_updates = 0

    def _features(self, description: str, amount: float | None, hour: int | None) -> list[tuple[str, float]]:
        if not self.use_context:
            amount, hour = None, None
        feats = featurize(description, amount, hour, self.context_weight)
        if self.rules is not None:
            hint, _ = self.rules.predict(description)
            if hint != "Uncategorized":
                feats.append(("rule_" + hint, 1.0))
        return feats

    # ----------------------------------------------------------- learning
    def update(self, description: str, amount: float, hour: int, label: str, weight: float = 1.0) -> None:
        feats = self._features(description, amount, hour)
        self.class_counts[label] += weight
        self.n_docs += weight
        for f, w in feats:
            self.token_counts[label][f] += weight * w
            self.class_token_totals[label] += weight * w
            self.vocab.add(f)
        self.n_updates += 1

    def fit(self, rows) -> "NaiveBayesCategorizer":
        """rows: iterable of dicts with description, amount, hour, category."""
        for r in rows:
            self.update(r["description"], r["amount"], r["hour"], r["category"])
        return self

    # ----------------------------------------------------------- inference
    @property
    def classes(self) -> list[str]:
        return sorted(self.class_counts)

    def log_scores(self, description: str, amount: float | None, hour: int | None) -> dict[str, float]:
        feats = self._features(description, amount, hour)
        V = len(self.vocab) + 1
        K = len(self.class_counts)
        scores = {}
        for c in self.classes:
            lp = math.log((self.class_counts[c] + self.alpha) / (self.n_docs + self.alpha * K))
            denom = self.class_token_totals[c] + self.alpha * V
            tc = self.token_counts[c]
            for f, w in feats:
                lp += w * math.log((tc.get(f, 0.0) + self.alpha) / denom)
            scores[c] = lp
        return scores

    def predict_proba(self, description: str, amount: float | None = None, hour: int | None = None) -> dict[str, float]:
        scores = self.log_scores(description, amount, hour)
        m = max(scores.values())
        exp = {c: math.exp(s - m) for c, s in scores.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    def predict(self, description: str, amount: float | None = None, hour: int | None = None) -> tuple[str, float]:
        proba = self.predict_proba(description, amount, hour)
        label = max(proba, key=proba.get)
        return label, proba[label]


class KeywordRuleCategorizer:
    """Baseline: longest-keyword-wins dictionary lookup on brand names."""

    def __init__(self, keyword_to_category: dict[str, str]):
        # longest keywords first so "amazon kindle" beats "amazon"
        self.rules = sorted(((k.lower(), v) for k, v in keyword_to_category.items()),
                            key=lambda kv: -len(kv[0]))

    @classmethod
    def from_brands(cls, merchants: dict) -> "KeywordRuleCategorizer":
        mapping = {}
        for cat, items in merchants.items():
            for brand, _aliases, _late in items:
                mapping[brand] = cat
        return cls(mapping)

    def predict(self, description: str, amount: float | None = None, hour: int | None = None) -> tuple[str, float]:
        d = description.lower()
        for kw, cat in self.rules:
            if kw in d:
                return cat, 1.0
        return "Uncategorized", 0.0
