"""Leak-free pre-meeting features and internally generated policy probabilities."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POLICY_SCORES = ("wordlist_score", "roberta_score")
DECISION_CLASSES = ("Cut", "Hold", "Hike")
MIN_TRAIN = 20
FEATURES = tuple(
    f"{source}_{score}"
    for source in ("previous_statement", "latest_minutes", "intermeeting_chair")
    for score in POLICY_SCORES
)


def decision(text: str) -> str | None:
    """Read the implemented rate decision, excluding forward guidance and non-meetings."""
    value = str(text).lower()
    actions = {
        "Cut": r"(?:committee|fomc)\s+(?:also\s+)?decided (?:today )?to (?:lower|reduce) the target range",
        "Hike": r"(?:committee|fomc)\s+(?:also\s+)?decided (?:today )?to (?:raise|increase) the target range",
        "Hold": r"(?:committee|fomc)\s+(?:also\s+)?decided (?:today )?to (?:maintain|keep) the target range",
    }
    for label, pattern in actions.items():
        if re.search(pattern, value):
            return label
    return None


def build_rows(documents: pd.DataFrame, future_date: str | None = None) -> pd.DataFrame:
    """Create one row per meeting from information released strictly beforehand."""
    docs = documents.copy()
    docs["release_date"] = pd.to_datetime(docs.release_date)
    statements = docs[docs.subtype.eq("statement")].copy()
    statements["decision"] = statements.selected_text.map(decision)
    meetings = statements.dropna(subset=["decision"]).sort_values("release_date")
    dates = list(meetings.release_date)
    if future_date is not None:
        dates.append(pd.Timestamp(future_date))

    rows = []
    for date in dates:
        earlier = meetings[meetings.release_date < date]
        if earlier.empty:
            continue
        previous = earlier.iloc[-1]
        minutes = docs[(docs.doc_type.eq("Minutes")) & (docs.release_date < date)].sort_values("release_date")
        latest_minutes = minutes.iloc[-1] if len(minutes) else None
        chair = docs[
            docs.doc_type.eq("Chair communication")
            & (docs.release_date > previous.release_date)
            & (docs.release_date < date)
        ]
        if chair.empty:
            chair = docs[
                docs.doc_type.eq("Chair communication") & (docs.release_date < date)
            ].sort_values("release_date").tail(3)

        current = meetings[meetings.release_date.eq(date)]
        row = {
            "date": date,
            "decision": current.iloc[0].decision if len(current) else None,
            "previous_decision": previous.decision,
            "chair_documents": len(chair),
        }
        for score in POLICY_SCORES:
            row[f"previous_statement_{score}"] = previous[score]
            row[f"latest_minutes_{score}"] = (
                latest_minutes[score] if latest_minutes is not None else np.nan
            )
            row[f"intermeeting_chair_{score}"] = chair[score].mean() if len(chair) else np.nan
            row[f"current_{score}"] = current.iloc[0][score] if len(current) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pipeline(features: tuple[str, ...]) -> Pipeline:
    transformers = []
    if features:
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("tone", numeric, list(features)))
    transformers.append(
        ("previous_decision", OneHotEncoder(handle_unknown="ignore"), ["previous_decision"])
    )
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("model", LogisticRegression(C=0.25, max_iter=5000, random_state=5768)),
        ]
    )


def _probability_dict(model: Pipeline, frame: pd.DataFrame, features: tuple[str, ...]) -> dict[str, float]:
    columns = list(features) + ["previous_decision"]
    values = model.predict_proba(frame[columns])[0]
    return dict(zip(model.named_steps["model"].classes_, map(float, values)))


def _metrics(actual: list[str], probabilities: list[dict[str, float]]) -> dict[str, float | int]:
    matrix = np.array([[p.get(label, 0.0) for label in DECISION_CLASSES] for p in probabilities])
    indexes = np.array([DECISION_CLASSES.index(label) for label in actual])
    predicted = np.array(DECISION_CLASSES)[matrix.argmax(axis=1)]
    return {
        "n": len(actual),
        "accuracy": float(np.mean(predicted == np.array(actual))),
        "log_loss": float(-np.mean(np.log(matrix[np.arange(len(matrix)), indexes]))),
        "brier": float(np.mean(np.sum((matrix - np.eye(3)[indexes]) ** 2, axis=1))),
    }


def expanding_backtest(rows: pd.DataFrame, features: tuple[str, ...]) -> dict[str, float | int]:
    actual, probabilities = [], []
    for end in range(MIN_TRAIN, len(rows)):
        train, test = rows.iloc[:end], rows.iloc[[end]]
        model = _pipeline(features)
        columns = list(features) + ["previous_decision"]
        model.fit(train[columns], train.decision)
        actual.append(test.decision.iloc[0])
        probabilities.append(_probability_dict(model, test, features))
    return _metrics(actual, probabilities)


def rounded_probabilities(probabilities: dict[str, float]) -> dict[str, int]:
    scaled = {key: 100 * probabilities.get(key, 0.0) for key in DECISION_CLASSES}
    result = {key: int(np.floor(value)) for key, value in scaled.items()}
    for key in sorted(DECISION_CLASSES, key=lambda x: scaled[x] - result[x], reverse=True)[: 100 - sum(result.values())]:
        result[key] += 1
    return result


def premeeting_forecast(documents: pd.DataFrame, meeting_date: str) -> dict:
    training = build_rows(documents)
    future = build_rows(documents, meeting_date).tail(1)
    columns = list(FEATURES) + ["previous_decision"]
    model = _pipeline(FEATURES)
    model.fit(training[columns], training.decision)
    raw = _probability_dict(model, future, FEATURES)

    baseline = _pipeline(())
    baseline.fit(training[["previous_decision"]], training.decision)
    baseline_raw = _probability_dict(baseline, future, ())

    # Define a more-hawkish statement by the equal-weight change in the two
    # policy-specific document scores. Current-statement values are outcomes only.
    labeled = training.copy()
    deltas = [labeled[f"current_{score}"] - labeled[f"previous_statement_{score}"] for score in POLICY_SCORES]
    labeled["more_hawkish"] = (sum(deltas) / len(deltas) > 0).astype(int)
    smoothed = {
        label: float((group.more_hawkish.sum() + 1) / (len(group) + 2))
        for label, group in labeled.groupby("decision")
    }
    tone_probability = float(sum(raw[label] * smoothed[label] for label in raw))

    return {
        "rate_probabilities": rounded_probabilities(raw),
        "raw_rate_probabilities": raw,
        "previous_decision_baseline": rounded_probabilities(baseline_raw),
        "backtest": expanding_backtest(training, FEATURES),
        "baseline_backtest": expanding_backtest(training, ()),
        "training_meetings": len(training),
        "backtest_start_after_meetings": MIN_TRAIN,
        "predictors": list(FEATURES) + ["previous_decision"],
        "current_features": future.iloc[0][columns].to_dict(),
        "more_hawkish_probability": int(round(100 * tone_probability)),
        "raw_more_hawkish_probability": tone_probability,
        "tone_rate_by_decision": smoothed,
        "method": "L2-regularized multinomial logit; expanding-window validation; pre-meeting documents only",
        "external_market_probability_used": False,
    }
