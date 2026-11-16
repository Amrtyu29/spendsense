"""Draws the agent architecture diagram to outputs/architecture.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "outputs"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.32, "\n".join(lines), ha="center", va="top", fontsize=8.2, linespacing=1.35)


def arrow(ax, p, q, label=None, color="#333", rad=0.0, dx=0.0, dy=0.05):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy
        ax.text(mx, my, label, ha="center", fontsize=7.8, color=color,
                bbox=dict(fc="white", ec="none", pad=1))


def draw(path: Path | None = None) -> Path:
    path = path or OUT / "architecture.png"
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.0); ax.axis("off")

    # environment
    box(ax, 0.3, 2.3, 2.3, 2.6, "ENVIRONMENT", ["bank / UPI transaction", "stream", "", "the user",
                                               "(spends, answers,", "dismisses, corrects)"], "#f3e5ab")
    # agent frame
    ax.add_patch(FancyBboxPatch((3.1, 0.6), 9.6, 6.1, boxstyle="round,pad=0.02", fc="#f7f9fc", ec="#1f77b4", lw=1.8, ls="--"))
    ax.text(3.3, 6.55, "BUDGET AGENT  (model-based, utility-driven, learning)", fontsize=11, weight="bold", color="#1f77b4", va="top")

    # row 1
    box(ax, 3.4, 4.3, 2.5, 1.8, "1. PERCEIVE", ["timestamp, description,", "amount", "features: tokens,",
                                               "bigrams, amount bucket,", "time of day"], "#dbe9f6")
    box(ax, 6.2, 4.3, 3.0, 1.8, "2. INTERPRET", ["Naive Bayes categoriser", "(online, + brand hint)",
                                                 "anomaly score: noisy-OR of", "robust z, burst, duplicate,", "novelty"], "#dbe9f6")
    box(ax, 9.5, 4.3, 3.0, 1.8, "3. STATE / BELIEF", ["spend per category, daily series",
                                                      "end-of-month projection", "N(mean, sd) by shrinkage",
                                                      "budgets, held payments", "user model: Beta(p_act), fatigue"], "#e3f1e1")
    # row 2
    box(ax, 3.4, 1.7, 2.5, 2.1, "6. LEARN", ["p_act: Beta update from", "spend before / after a warning",
                                             "fatigue: dismiss / helpful", "categoriser: corrections",
                                             "whitelist: 'this is genuine'"], "#ece0f5")
    box(ax, 6.2, 1.7, 3.0, 2.1, "4. DECIDE (expected utility)", ["per txn:  NONE / VERIFY / BLOCK",
                                                                 "per category:  NONE / INFO /", "WARN / REALLOCATE",
                                                                 "P(over), E[overspend], p_act,", "fatigue cost, cooldown, margin"], "#fde2cf")
    box(ax, 9.5, 1.7, 3.0, 2.1, "5. ACT", ["explainable messages", "(evidence + utilities)",
                                           "hold a payment, ask user,", "move budget between", "categories"], "#fde2cf")

    arrow(ax, (2.6, 4.3), (3.4, 5.2), "transactions (daily)", rad=-0.2, dx=-0.9, dy=0.75)
    arrow(ax, (5.9, 5.2), (6.2, 5.2))
    arrow(ax, (9.2, 5.2), (9.5, 5.2))
    arrow(ax, (10.6, 4.3), (8.4, 3.8), rad=0.25)
    arrow(ax, (9.2, 2.75), (9.5, 2.75))
    arrow(ax, (11.0, 1.7), (1.6, 2.3), "actions, questions", color="#b03a2e", rad=-0.3, dy=-0.55)
    arrow(ax, (2.6, 3.0), (3.4, 2.9), "feedback /\noutcomes", color="#6a3d9a", dy=0.35)
    arrow(ax, (5.9, 2.75), (6.2, 2.75), color="#6a3d9a")
    arrow(ax, (4.65, 3.8), (6.2, 4.6), color="#6a3d9a", rad=0.2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw())
