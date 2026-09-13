"""Reproduce the Assignment 2 data, scores, event study, forecast, and figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "interim" / "matplotlib"))

from src.analysis import run_analysis
from src.collect_fed import collect_fed_documents
from src.collect_market import collect_market_data
from src.plotting import plot_tone_trends
from src.tone import score_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="redownload public source data")
    parser.add_argument("--rescore", action="store_true", help="recompute FinBERT scores")
    args = parser.parse_args()

    documents = collect_fed_documents(refresh=args.refresh)
    market = collect_market_data(refresh=args.refresh)
    scores = score_documents(documents, refresh=args.rescore)
    results = run_analysis(scores, market)
    figure = plot_tone_trends(scores)
    print(results["counts"])
    print(results["forecast"])
    print(f"Saved {figure}")


if __name__ == "__main__":
    main()

