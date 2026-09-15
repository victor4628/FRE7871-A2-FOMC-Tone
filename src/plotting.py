"""Publication-quality tone chart with readable time labels and full-width panels."""
from __future__ import annotations

import os

from .config import INTERIM_DIR, OUTPUT_DIR, WARSH_START

os.environ.setdefault("MPLCONFIGDIR", str(INTERIM_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

GREEN = "#254C45"
ORANGE = "#A65F1A"
BLUE = "#426B8A"
GRID = "#DEE7E4"

DOCUMENT_STYLES = {
    "Statement": {"color": GREEN, "marker": "o"},
    "Minutes": {"color": BLUE, "marker": "s"},
    "Chair communication": {"color": ORANGE, "marker": "^"},
}


def plot_tone_trends(documents: pd.DataFrame):
    frame = documents.copy()
    frame["release_date"] = pd.to_datetime(frame.release_date)
    metrics = [
        ("wordlist_score", "Published dictionary (+ hawkish)"),
        ("roberta_score", "RoBERTa reproduction (+ hawkish)"),
        ("finbert_sentiment", "FinBERT sentiment (+ positive)"),
    ]
    missing = [metric for metric, _ in metrics if metric not in frame]
    if missing:
        raise ValueError(f"Cannot create final Figure 1; missing {missing}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    warsh_start = pd.Timestamp(WARSH_START)
    for panel, (metric, title) in enumerate(metrics):
        ax = axes[panel]
        for doc_type, style in DOCUMENT_STYLES.items():
            group = frame[frame.doc_type == doc_type].sort_values("release_date")
            ax.scatter(
                group.release_date,
                group[metric],
                s=13,
                alpha=0.18,
                color=style["color"],
                marker=style["marker"],
                edgecolors="none",
                zorder=2,
            )
            smooth = group.set_index("release_date")[metric].rolling(5, min_periods=2).mean()
            ax.plot(
                smooth.index,
                smooth,
                color=style["color"],
                linewidth=1.9,
                label=doc_type,
                zorder=3,
            )
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.axvline(warsh_start, color="#555555", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=GREEN, loc="left")
        ax.set_ylabel("Score", fontsize=9)

    axes[0].legend(loc="upper left", ncols=3, frameon=False, fontsize=8.5)
    axes[0].annotate(
        "Warsh starts",
        xy=(warsh_start, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-4, -4),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Release date", fontsize=9)
    fig.suptitle(
        "Figure 1. Communication scores over time by release type",
        fontsize=14,
        fontweight="bold",
        color=GREEN,
    )
    dest = OUTPUT_DIR / "figure1_tone_trends_v2.png"
    fig.savefig(dest, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest
