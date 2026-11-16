"""Synthetic transaction environment.

The environment produces a stream of bank / UPI transactions for a student in
Kerala.  Every transaction carries a *true* category and a *true* anomaly flag
that the agent never sees; they are used only for evaluation.

Design choices that matter for evaluation
-----------------------------------------
* Each calendar day is generated from its own random seed derived from
  (simulator seed, day).  Behaviour changes caused by the agent (the user
  spending less after a warning) only *scale* amounts, so the same month can be
  replayed with and without the agent to obtain a fair counterfactual.
* Merchants appear in statements under several *aliases* (legal entity names,
  abbreviations).  Some merchants are "late": they only appear after
  LATE_MERCHANT_MONTH, which tests how the categoriser handles unseen shops.
* Anomalies of four kinds are injected: unusually large amounts, bursts of
  rapid small payments (card-testing fraud), unknown merchants at night, and
  duplicate charges.
"""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = [
    "Food", "Groceries", "Transport", "Shopping", "Entertainment",
    "Bills", "Health", "Education", "Rent", "Transfer",
]
DISCRETIONARY = ["Food", "Groceries", "Transport", "Shopping", "Entertainment",
                 "Health", "Education", "Transfer"]
LATE_MERCHANT_MONTH = 6

# category -> list of (brand, [statement aliases], appears_late)
MERCHANTS = {
    "Food": [
        ("Swiggy", ["SWIGGY", "BUNDL TECHNOLOGIES", "SWIGGY FOOD ORDER"], False),
        ("Zomato", ["ZOMATO", "ZOMATO MEDIA PVT LTD", "ZOMATO ONLINE"], False),
        ("Campus Canteen", ["CAMPUS CANTEEN", "IITPKD CANTEEN", "CANTEEN COUNTER"], False),
        ("Annapoorna Mess", ["ANNAPOORNA MESS", "ANNAPURNA HOTEL"], False),
        ("Chai Point", ["CHAI POINT", "MOUNTAIN TRAIL FOODS"], False),
        ("Dominos", ["DOMINOS PIZZA", "JUBILANT FOODWORKS"], False),
        ("KFC", ["KFC", "YUM RESTAURANTS INDIA"], True),
        ("Paragon Restaurant", ["PARAGON RESTAURANT", "PARAGON BIRYANI"], True),
    ],
    "Groceries": [
        ("BigBasket", ["BIGBASKET", "INNOVATIVE RETAIL CONCEPTS", "BB DAILY"], False),
        ("DMart", ["DMART", "AVENUE SUPERMARTS"], False),
        ("Blinkit", ["BLINKIT", "BLINK COMMERCE PVT", "GROFERS"], False),
        ("Reliance Fresh", ["RELIANCE FRESH", "RELIANCE RETAIL LTD"], False),
        ("Kirana Store", ["SREE KRISHNA STORES", "KIRANA STORE", "RAJU PROVISION STORE"], False),
        ("Zepto", ["ZEPTO", "KIRANAKART TECHNOLOGIES"], True),
        ("More Supermarket", ["MORE SUPERMARKET", "MORE RETAIL LTD"], True),
    ],
    "Transport": [
        ("Ola", ["OLA CABS", "ANI TECHNOLOGIES", "OLA MONEY"], False),
        ("Uber", ["UBER", "UBER INDIA SYSTEMS", "UBER TRIP"], False),
        ("Rapido", ["RAPIDO", "ROPPEN TRANSPORTATION"], False),
        ("IRCTC", ["IRCTC", "IRCTC RAIL TICKET", "INDIAN RAILWAYS"], False),
        ("KSRTC", ["KSRTC BUS", "KERALA RTC"], False),
        ("HP Petrol Pump", ["HP PETROL PUMP", "HPCL FUEL STATION", "HINDUSTAN PETROLEUM"], False),
        ("RedBus", ["REDBUS", "IBIBO GROUP"], True),
        ("Indian Oil", ["INDIAN OIL", "IOCL PETROL BUNK"], True),
    ],
    "Shopping": [
        ("Amazon", ["AMAZON", "AMAZON PAY INDIA", "AMAZON SELLER SERVICES"], False),
        ("Flipkart", ["FLIPKART", "FLIPKART INTERNET PVT"], False),
        ("Myntra", ["MYNTRA", "MYNTRA DESIGNS"], False),
        ("Decathlon", ["DECATHLON", "DECATHLON SPORTS INDIA"], False),
        ("Meesho", ["MEESHO", "FASHNEAR TECHNOLOGIES"], True),
        ("Croma", ["CROMA", "INFINITI RETAIL"], True),
    ],
    "Entertainment": [
        ("Netflix", ["NETFLIX", "NETFLIX ENTERTAINMENT SERVICES"], False),
        ("Spotify", ["SPOTIFY", "SPOTIFY INDIA"], False),
        ("BookMyShow", ["BOOKMYSHOW", "BIGTREE ENTERTAINMENT"], False),
        ("PVR Cinemas", ["PVR CINEMAS", "PVR INOX LTD"], False),
        ("Hotstar", ["HOTSTAR", "NOVI DIGITAL ENTERTAINMENT"], True),
        ("Steam", ["STEAM GAMES", "VALVE CORPORATION"], True),
    ],
    "Bills": [
        ("Airtel", ["AIRTEL", "BHARTI AIRTEL LTD", "AIRTEL PREPAID RECHARGE"], False),
        ("Jio", ["JIO", "RELIANCE JIO INFOCOMM", "JIO RECHARGE"], False),
        ("KSEB", ["KSEB", "KSEB ELECTRICITY BILL", "KERALA STATE ELECTRICITY"], False),
        ("ACT Fibernet", ["ACT FIBERNET", "ATRIA CONVERGENCE"], False),
        ("Water Authority", ["KERALA WATER AUTHORITY", "KWA WATER BILL"], True),
    ],
    "Health": [
        ("Apollo Pharmacy", ["APOLLO PHARMACY", "APOLLO HEALTHCO"], False),
        ("MedPlus", ["MEDPLUS", "MEDPLUS HEALTH SERVICES"], False),
        ("City Hospital", ["CITY HOSPITAL", "DISTRICT HOSPITAL PALAKKAD"], False),
        ("Practo", ["PRACTO", "PRACTO TECHNOLOGIES"], False),
        ("1mg", ["TATA 1MG", "1MG TECHNOLOGIES"], True),
    ],
    "Education": [
        ("Coursera", ["COURSERA", "COURSERA INC"], False),
        ("Xerox Shop", ["XEROX SHOP", "CAMPUS XEROX", "PRINT POINT"], False),
        ("Stationery Mart", ["STATIONERY MART", "BOOK AND PEN STORE"], False),
        ("Udemy", ["UDEMY", "UDEMY INC"], False),
        ("Kindle", ["AMAZON KINDLE", "KINDLE EBOOK"], True),
    ],
    "Rent": [
        ("Hostel Rent", ["HOSTEL RENT TRANSFER", "PG RENT", "HOUSE RENT"], False),
    ],
    "Transfer": [
        ("Friend Transfer", ["UPI TRANSFER TO RAHUL", "UPI TRANSFER TO ANJALI", "SPLIT BILL TRANSFER"], False),
        ("Family Transfer", ["TRANSFER TO AMMA", "FAMILY TRANSFER"], False),
        ("Wallet Load", ["PAYTM WALLET LOAD", "PHONEPE WALLET"], True),
    ],
}

SUSPICIOUS_ALIASES = [
    "WWW XYZ TRADING LLC", "INTL PMT GLOBALPAY", "QUICKCASH SERVICES",
    "LUCKYSPIN GAMING", "CRYPTOX EXCHANGE", "FASTLOAN APP", "OVERSEAS MERCHANT 7731",
]
CITIES = ["PALAKKAD", "COIMBATORE", "KOCHI", "BANGALORE", "ONLINE", "THRISSUR"]


@dataclass
class CategoryProfile:
    daily_rate: float          # mean transactions per weekday
    weekend_mult: float        # multiplier on rate at weekends
    median_amount: float       # lognormal median (rupees)
    sigma: float               # lognormal sigma
    peak_hours: tuple          # typical hours of day
    scheduled: tuple = ()      # fixed payments: (day_of_month, brand, amount)


PROFILES = {
    "Food": CategoryProfile(1.5, 1.3, 90, 0.55, (8, 13, 20)),
    "Groceries": CategoryProfile(0.35, 1.4, 220, 0.6, (10, 18)),
    "Transport": CategoryProfile(0.5, 1.4, 80, 0.75, (9, 18)),
    "Shopping": CategoryProfile(0.12, 1.6, 650, 0.8, (12, 21)),
    "Entertainment": CategoryProfile(0.08, 2.2, 220, 0.6, (19, 21),
                                     scheduled=((3, "Netflix", 199),)),
    "Bills": CategoryProfile(0.0, 1.0, 400, 0.4, (10,),
                             scheduled=((5, "Airtel", 299), (10, "KSEB", 650), (15, "ACT Fibernet", 799))),
    "Health": CategoryProfile(0.07, 1.0, 280, 0.7, (11, 17)),
    "Education": CategoryProfile(0.1, 0.8, 120, 1.0, (11, 15)),
    "Rent": CategoryProfile(0.0, 1.0, 6000, 0.0, (9,), scheduled=((2, "Hostel Rent", 6000),)),
    "Transfer": CategoryProfile(0.12, 1.2, 400, 0.8, (12, 20)),
}


def _round_amount(x: float) -> float:
    x = max(10.0, x)
    return float(round(x / 5) * 5) if x > 50 else float(round(x))


def _describe(rng: np.random.Generator, alias: str) -> str:
    ref = str(rng.integers(10**9, 10**10))
    city = CITIES[rng.integers(len(CITIES))]
    last4 = str(rng.integers(1000, 9999))
    templates = [
        f"UPI/{alias}/{ref}",
        f"UPI-{alias}-{ref}@okaxis",
        f"POS {alias} {city}",
        f"{alias} PAYMENT {ref}",
        f"PAYMENT TO {alias}",
        f"{alias}",
        f"CARD {last4} {alias} {city}",
        f"IMPS/{ref}/{alias}",
    ]
    return templates[rng.integers(len(templates))]


class TransactionSimulator:
    """Generates transactions day by day.  Deterministic per (seed, day)."""

    def __init__(self, seed: int = 0, anomaly_day_prob: float = 0.05):
        self.seed = seed
        self.anomaly_day_prob = anomaly_day_prob
        self._txn_counter = 0

    # ------------------------------------------------------------------ helpers
    def _month_multipliers(self, year: int, month: int) -> dict:
        """Hidden month-level variation (festivals, exams, medical shocks)."""
        rng = np.random.default_rng([self.seed, 777, year, month])
        mult = {c: float(rng.lognormal(0.0, 0.12)) for c in CATEGORIES}
        mult["Rent"] = 1.0
        if rng.random() < 0.15:  # shock month for one discretionary category
            c = DISCRETIONARY[rng.integers(len(DISCRETIONARY))]
            mult[c] *= 1.6
        return mult

    def _merchant(self, rng, category: str, month_idx: int):
        options = [m for m in MERCHANTS[category] if (not m[2]) or month_idx >= LATE_MERCHANT_MONTH]
        brand, aliases, _ = options[rng.integers(len(options))]
        return brand, aliases[rng.integers(len(aliases))]

    def _hour(self, rng, profile: CategoryProfile) -> int:
        h = profile.peak_hours[rng.integers(len(profile.peak_hours))] + rng.normal(0, 1.5)
        return int(min(23, max(6, round(h))))

    def _new_txn(self, ts, description, brand, amount, category, anomaly_type=None) -> dict:
        self._txn_counter += 1
        return {
            "txn_id": self._txn_counter,
            "ts": ts,
            "description": description,
            "merchant": brand,
            "amount": float(amount),
            "category": category,
            "is_anomaly": anomaly_type is not None,
            "anomaly_type": anomaly_type or "",
        }

    # ------------------------------------------------------------------ public
    def generate_day(self, d: date, user_multipliers: dict | None = None, month_idx: int = 0) -> list[dict]:
        """Generate all transactions of one day.

        user_multipliers scales *discretionary* spending (how the simulated user
        reacts to the agent).  They never change which random numbers are drawn,
        so replays with different multipliers are directly comparable.
        """
        rng = np.random.default_rng([self.seed, d.toordinal()])
        month_mult = self._month_multipliers(d.year, d.month)
        user_multipliers = user_multipliers or {}
        weekend = d.weekday() >= 5
        txns: list[dict] = []

        for cat, prof in PROFILES.items():
            scale = month_mult[cat] * user_multipliers.get(cat, 1.0)
            # scheduled fixed payments
            for dom, brand, amt in prof.scheduled:
                if dom == d.day:
                    aliases = next(m[1] for m in MERCHANTS[cat] if m[0] == brand)
                    alias = aliases[rng.integers(len(aliases))]
                    jitter = 1.0 if cat == "Rent" else float(rng.uniform(0.9, 1.1))
                    amount = _round_amount(amt * jitter * (month_mult[cat] if cat == "Bills" else 1.0))
                    ts = datetime(d.year, d.month, d.day, int(rng.integers(8, 20)), int(rng.integers(60)))
                    txns.append(self._new_txn(ts, _describe(rng, alias), brand, amount, cat))
            # random purchases
            rate = prof.daily_rate * (prof.weekend_mult if weekend else 1.0)
            n = rng.poisson(rate)
            for _ in range(n):
                brand, alias = self._merchant(rng, cat, month_idx)
                amount = _round_amount(prof.median_amount * math.exp(prof.sigma * rng.normal()) * scale)
                ts = datetime(d.year, d.month, d.day, self._hour(rng, prof), int(rng.integers(60)))
                txns.append(self._new_txn(ts, _describe(rng, alias), brand, amount, cat))

        # ---- anomaly injection (never scaled by user behaviour)
        if rng.random() < self.anomaly_day_prob:
            kind = rng.choice(["large_amount", "burst", "unknown_night", "duplicate"],
                              p=[0.45, 0.12, 0.28, 0.15])
            if kind == "large_amount":
                cat = DISCRETIONARY[rng.integers(len(DISCRETIONARY))]
                prof = PROFILES[cat]
                brand, alias = self._merchant(rng, cat, month_idx)
                amount = _round_amount(prof.median_amount * math.exp(prof.sigma * 2.5) * rng.uniform(2.0, 4.0))
                ts = datetime(d.year, d.month, d.day, self._hour(rng, prof), int(rng.integers(60)))
                txns.append(self._new_txn(ts, _describe(rng, alias), brand, amount, cat, "large_amount"))
            elif kind == "burst":
                cat = ["Food", "Transport", "Shopping"][rng.integers(3)]
                prof = PROFILES[cat]
                brand, alias = self._merchant(rng, cat, month_idx)
                desc = _describe(rng, alias)
                start = datetime(d.year, d.month, d.day, self._hour(rng, prof), int(rng.integers(50)))
                for k in range(int(rng.integers(4, 7))):
                    amount = _round_amount(prof.median_amount * math.exp(prof.sigma * rng.normal()))
                    ts = start + timedelta(minutes=int(rng.integers(1, 6)) * (k + 1))
                    txns.append(self._new_txn(ts, desc, brand, amount, cat, "burst"))
            elif kind == "unknown_night":
                alias = SUSPICIOUS_ALIASES[rng.integers(len(SUSPICIOUS_ALIASES))]
                amount = _round_amount(1800 * math.exp(0.5 * rng.normal()))
                ts = datetime(d.year, d.month, d.day, int(rng.integers(1, 5)), int(rng.integers(60)))
                txns.append(self._new_txn(ts, _describe(rng, alias), alias.title(), amount, "Transfer", "unknown_night"))
            elif kind == "duplicate" and txns:
                src = txns[rng.integers(len(txns))]
                ts = src["ts"] + timedelta(minutes=int(rng.integers(1, 4)))
                txns.append(self._new_txn(ts, src["description"], src["merchant"], src["amount"],
                                          src["category"], "duplicate"))

        txns.sort(key=lambda t: t["ts"])
        return txns

    def generate_month(self, year: int, month: int, month_idx: int = 0,
                       user_multipliers: dict | None = None) -> list[dict]:
        out = []
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            out.extend(self.generate_day(date(year, month, day), user_multipliers, month_idx))
        return out

    def generate_history(self, start_year: int, start_month: int, n_months: int,
                         first_month_idx: int = 0) -> pd.DataFrame:
        rows = []
        y, m = start_year, start_month
        for i in range(n_months):
            for t in self.generate_month(y, m, month_idx=first_month_idx + i):
                t = dict(t)
                t["month_idx"] = first_month_idx + i
                rows.append(t)
            m += 1
            if m > 12:
                m, y = 1, y + 1
        df = pd.DataFrame(rows)
        df["date"] = df["ts"].dt.date
        df["hour"] = df["ts"].dt.hour
        return df


def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    m = month - 1 + k
    return year + m // 12, m % 12 + 1


def public_view(txn: dict) -> dict:
    """What the agent is allowed to see (no labels)."""
    return {"txn_id": txn["txn_id"], "ts": txn["ts"], "description": txn["description"],
            "amount": txn["amount"]}
