"""Publication-quality charts for Assignment 2."""

from __future__ import annotations

import os

from .config import INTERIM_DIR, OUTPUT_DIR, WARSH_START

os.environ.setdefault("MPLCONFIGDIR", str(INTERIM_DIR / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"wordlist_z": "#254C45", "finbert_z": "#A65F1A"}
REFERENCE_COLOR = "#5F6664"


def plot_tone_trends(documents: pd.DataFrame):
    frame = documents.copy()
    frame["release_date"] = pd.to_datetime(frame.release_date)
    types = ["Statement", "Minutes", "Chair communication"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    for axis, doc_type in zip(axes, types):
        group = frame[frame.doc_type == doc_type].set_index("release_date")
        monthly = group[["wordlist_z", "finbert_z"]].resample("MS").mean()
        smooth = monthly.rolling(4, min_periods=1, center=True).mean()
        for score, label in [("wordlist_z", "Policy phrase list"), ("finbert_z", "FinBERT anchors")]:
            axis.scatter(group.index, group[score], s=9, alpha=0.18, color=COLORS[score])
            axis.plot(smooth.index, smooth[score], lw=2.0, color=COLORS[score], label=label)
        axis.axhline(0, color="#777777", lw=0.8)
        axis.axvline(pd.Timestamp(WARSH_START), color=REFERENCE_COLOR, lw=1.4, ls="--")
        axis.set_ylabel("Hawkishness\n(Powell-period SD)")
        axis.set_title(doc_type, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#E2E8E5", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Release date")
    fig.suptitle("Figure 1. Hawkish/dovish tone by document type", y=1.025, fontsize=14, fontweight="bold")
    axes[0].text(
        pd.Timestamp(WARSH_START), 3.55, " Warsh begins", rotation=90,
        color=REFERENCE_COLOR, ha="right", va="top", fontsize=8,
    )
    destination = OUTPUT_DIR / "figure1_tone_trends.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination
