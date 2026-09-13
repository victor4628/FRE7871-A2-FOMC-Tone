"""Event study, regressions, and September 2026 forecast."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import AS_OF_DATE, INTERIM_DIR, OUTPUT_DIR, WARSH_START


OUTCOMES = {
    "dxy_change_pct": "DXY (%)",
    "spread_change_bp": "10s2s (bp)",
    "dgs1_change_bp": "1-year yield (bp)",
    "growth_value_pp": "Growth - value (pp)",
}
TONES = {"wordlist_z": "Word list", "finbert_z": "FinBERT anchors"}


def attach_market_reactions(documents: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.sort_index().dropna(subset=["dxy", "iwf", "iwn"])
    dates = market.index
    records = []
    for _, row in documents.iterrows():
        event_date = pd.Timestamp(row.release_date)
        hour, minute = map(int, str(row.release_time_et).split(":"))
        after_close = (hour, minute) > (16, 0)
        eligible = dates[dates > event_date] if after_close else dates[dates >= event_date]
        if len(eligible) == 0:
            continue
        close_date = eligible[0]
        position = dates.get_loc(close_date)
        if not isinstance(position, (int, np.integer)) or position == 0:
            continue
        prior_date = dates[position - 1]
        before, after = market.loc[prior_date], market.loc[close_date]
        item = row.to_dict()
        item.update(
            {
                "window_start": prior_date.date().isoformat(),
                "window_end": close_date.date().isoformat(),
                "dxy_change_pct": 100 * (after.dxy / before.dxy - 1),
                "spread_change_bp": 100 * (after.t10y2y - before.t10y2y),
                "dgs1_change_bp": 100 * (after.dgs1 - before.dgs1),
                "growth_value_pp": 100
                * ((after.iwf / before.iwf - 1) - (after.iwn / before.iwn - 1)),
                "dgs3mo_change_bp": 100 * (after.dgs3mo - before.dgs3mo),
            }
        )
        records.append(item)
    result = pd.DataFrame(records)
    result.to_csv(INTERIM_DIR / "event_reactions.csv", index=False)
    return result


def document_counts(documents: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "statement": "FOMC statements",
        "minutes": "FOMC minutes",
        "speech": "Chair speeches",
        "testimony": "Chair testimony",
        "press_conference": "Press-conference transcripts",
    }
    table = (
        documents.assign(Type=documents.subtype.map(labels))
        .pivot_table(index="Type", columns="chair", values="document_id", aggfunc="count", fill_value=0)
        .reindex(labels.values())
        .fillna(0)
        .astype(int)
    )
    for chair in ("Powell", "Warsh"):
        if chair not in table:
            table[chair] = 0
    table["Total"] = table["Powell"] + table["Warsh"]
    table.loc["Total"] = table.sum(axis=0)
    return table[["Powell", "Warsh", "Total"]]


def regression_table(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    rows = []
    models: dict[tuple[str, str], object] = {}
    type_dummies = pd.get_dummies(events.doc_type, prefix="type", drop_first=True, dtype=float)
    for outcome, outcome_label in OUTCOMES.items():
        for tone, tone_label in TONES.items():
            sample = events[[outcome, tone, "dgs3mo_change_bp"]].join(type_dummies).dropna()
            x = sm.add_constant(sample.drop(columns=[outcome]), has_constant="add").astype(float)
            y = sample[outcome].astype(float)
            fitted = sm.OLS(y, x).fit(cov_type="HC3")
            models[(outcome, tone)] = fitted
            rows.append(
                {
                    "Indicator": outcome_label,
                    "Tone method": tone_label,
                    "Tone coefficient": fitted.params[tone],
                    "Tone SE": fitted.bse[tone],
                    "Tone p-value": fitted.pvalues[tone],
                    "DGS3MO coefficient": fitted.params["dgs3mo_change_bp"],
                    "DGS3MO SE": fitted.bse["dgs3mo_change_bp"],
                    "Adjusted R2": fitted.rsquared_adj,
                    "N": int(fitted.nobs),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "table3_regressions.csv", index=False)
    return table, models


def _decision(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"(?:lower|lowered|reduce|reduced) the target range", lowered):
        return "Cut"
    if re.search(r"(?:raise|raised|increase|increased) the target range", lowered):
        return "Hike"
    if re.search(r"(?:maintain|maintained|keep|kept) the target range", lowered):
        return "Hold"
    return None


def _market_before(market: pd.DataFrame, event_date: str, lookback: int = 20) -> tuple[float, float]:
    eligible = market.loc[market.index < pd.Timestamp(event_date)].dropna(subset=["dgs1", "t10y2y"])
    if len(eligible) <= lookback:
        return 0.0, 0.0
    return (
        100 * (eligible.dgs1.iloc[-1] - eligible.dgs1.iloc[-lookback - 1]),
        100 * (eligible.t10y2y.iloc[-1] - eligible.t10y2y.iloc[-lookback - 1]),
    )


def _meeting_features(documents: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    statements = documents[documents.subtype == "statement"].sort_values("release_date").copy()
    statements["decision"] = statements.text.map(_decision)
    statements["combined_tone"] = statements[["wordlist_z", "finbert_z"]].mean(axis=1)
    rows = []
    for index in range(1, len(statements)):
        previous = statements.iloc[index - 1]
        current = statements.iloc[index]
        interim = documents[
            (documents.release_date > previous.release_date)
            & (documents.release_date < current.release_date)
            & (documents.subtype != "statement")
        ]
        dgs1_20, curve_20 = _market_before(market, current.release_date)
        rows.append(
            {
                "date": current.release_date,
                "prior_tone": previous.combined_tone,
                "interim_tone": interim[["wordlist_z", "finbert_z"]].mean(axis=1).mean()
                if len(interim) else previous.combined_tone,
                "dgs1_20bp": dgs1_20,
                "curve_20bp": curve_20,
                "decision": current.decision,
                "more_hawkish": int(current.combined_tone > previous.combined_tone),
                "current_tone": current.combined_tone,
                "wordlist_delta": current.wordlist_z - previous.wordlist_z,
                "finbert_delta": current.finbert_z - previous.finbert_z,
            }
        )
    return pd.DataFrame(rows).dropna(subset=["decision"]), statements


def _rounded_percentages(values: dict[str, float]) -> dict[str, int]:
    raw = {key: 100 * value / sum(values.values()) for key, value in values.items()}
    floors = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = 100 - sum(floors.values())
    order = sorted(raw, key=lambda key: raw[key] - floors[key], reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def build_forecast(
    documents: pd.DataFrame,
    market: pd.DataFrame,
    regression_models: dict[tuple[str, str], object],
) -> dict:
    meetings, statements = _meeting_features(documents, market)
    latest = statements.iloc[-1]
    interim = documents[
        (documents.release_date > latest.release_date)
        & (documents.release_date <= AS_OF_DATE)
        & (documents.subtype != "statement")
    ]
    dgs1_20, curve_20 = _market_before(market, "2026-09-16")
    current_x = pd.DataFrame(
        [{
            "prior_tone": latest.combined_tone,
            "interim_tone": interim[["wordlist_z", "finbert_z"]].mean(axis=1).mean()
            if len(interim) else latest.combined_tone,
            "dgs1_20bp": dgs1_20,
            "curve_20bp": curve_20,
        }]
    )
    features = ["prior_tone", "interim_tone", "dgs1_20bp", "curve_20bp"]

    valid = meetings.dropna(subset=features + ["decision"])
    rate_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.35, max_iter=2000, random_state=7871)
    )
    rate_model.fit(valid[features], valid.decision)
    model_probs = dict(zip(rate_model.classes_, rate_model.predict_proba(current_x[features])[0]))
    counts = valid.decision.value_counts()
    prior = {label: (counts.get(label, 0) + 1) / (len(valid) + 3) for label in ("Cut", "Hold", "Hike")}
    blended = {label: 0.8 * model_probs.get(label, 0.0) + 0.2 * prior[label] for label in prior}
    rate_probabilities = _rounded_percentages(blended)

    tone_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.35, max_iter=2000, random_state=7871)
    )
    tone_model.fit(valid[features], valid.more_hawkish)
    p_model = float(tone_model.predict_proba(current_x[features])[0, 1])
    p_base = float(valid.more_hawkish.mean())
    p_hawk = 0.8 * p_model + 0.2 * p_base

    expected_control = 25 * (blended["Hike"] - blended["Cut"])
    expected_tones = {}
    for tone, delta in [("wordlist_z", "wordlist_delta"), ("finbert_z", "finbert_delta")]:
        positive = valid.loc[valid.more_hawkish.eq(1), delta].median()
        nonpositive = valid.loc[valid.more_hawkish.eq(0), delta].median()
        expected_delta = p_hawk * positive + (1 - p_hawk) * nonpositive
        expected_tones[tone] = float(latest[tone] + expected_delta)

    market_forecast = {}
    for outcome, label in OUTCOMES.items():
        predictions = []
        variances = []
        for tone in TONES:
            fitted = regression_models[(outcome, tone)]
            row = {name: 0.0 for name in fitted.params.index}
            row.update(
                {
                    "const": 1.0,
                    tone: expected_tones[tone],
                    "dgs3mo_change_bp": expected_control,
                    "type_Statement": 1.0,
                }
            )
            x = np.array([row[name] for name in fitted.params.index], dtype=float)
            predictions.append(float(x @ fitted.params.values))
            variances.append(float(fitted.mse_resid))
        expected = float(np.mean(predictions))
        sigma = float(np.sqrt(np.mean(variances) + np.var(predictions)))
        market_forecast[label] = {
            "probability_rise": int(round(100 * norm.cdf(expected / sigma))),
            "expected_change": expected,
            "unit": "bp" if "(bp)" in label else ("pp" if "(pp)" in label else "%"),
        }

    strongest = max(market_forecast, key=lambda key: abs(market_forecast[key]["probability_rise"] - 50))
    forecast = {
        "as_of": AS_OF_DATE,
        "meeting_date": "2026-09-16",
        "rate_probabilities": rate_probabilities,
        "more_hawkish_probability": int(round(100 * p_hawk)),
        "expected_statement_tones": expected_tones,
        "market_reaction": market_forecast,
        "strongest_signal": strongest,
        "training_meetings": int(len(valid)),
        "latest_statement_date": latest.release_date,
        "interim_documents": int(len(interim)),
    }
    (OUTPUT_DIR / "forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    return forecast


def run_analysis(scores: pd.DataFrame, market: pd.DataFrame) -> dict:
    events = attach_market_reactions(scores, market)
    counts = document_counts(scores)
    counts.to_csv(OUTPUT_DIR / "table1_document_counts.csv")
    warsh = events[events.chair == "Warsh"].copy()
    warsh.to_csv(OUTPUT_DIR / "table2_warsh_releases.csv", index=False)
    regressions, models = regression_table(events)
    forecast = build_forecast(scores, market, models)
    return {
        "events": events,
        "counts": counts,
        "warsh": warsh,
        "regressions": regressions,
        "models": models,
        "forecast": forecast,
    }

