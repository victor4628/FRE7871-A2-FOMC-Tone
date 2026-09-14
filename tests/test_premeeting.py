import pandas as pd

from src.premeeting import build_rows, decision, rounded_probabilities


def test_decision_uses_implemented_action_not_forward_guidance():
    text = (
        "The Committee decided to keep the target range at 0 to 1/4 percent. "
        "The Committee expects it will soon be appropriate to raise the target range."
    )
    assert decision(text) == "Hold"
    assert decision("The Committee decided today to lower the target range.") == "Cut"
    assert decision("The Committee decided to raise the target range.") == "Hike"
    assert decision("The Committee approved additional Treasury purchases.") is None


def test_future_features_use_only_prior_releases():
    rows = [
        {"release_date": "2026-01-01", "doc_type": "Statement", "subtype": "statement",
         "selected_text": "The Committee decided to maintain the target range.",
         "wordlist_score": 0.1, "roberta_score": 0.2},
        {"release_date": "2026-01-20", "doc_type": "Minutes", "subtype": "minutes",
         "selected_text": "minutes", "wordlist_score": 0.3, "roberta_score": 0.4},
        {"release_date": "2026-02-01", "doc_type": "Chair communication", "subtype": "speech",
         "selected_text": "speech", "wordlist_score": 0.5, "roberta_score": 0.6},
        {"release_date": "2026-03-01", "doc_type": "Chair communication", "subtype": "speech",
         "selected_text": "future speech", "wordlist_score": 0.9, "roberta_score": 0.9},
    ]
    future = build_rows(pd.DataFrame(rows), "2026-02-15").iloc[-1]
    assert future.previous_decision == "Hold"
    assert future.latest_minutes_wordlist_score == 0.3
    assert future.intermeeting_chair_wordlist_score == 0.5


def test_probability_rounding_sums_to_one_hundred():
    result = rounded_probabilities({"Cut": 0.0648, "Hold": 0.7256, "Hike": 0.2096})
    assert result == {"Cut": 6, "Hold": 73, "Hike": 21}
    assert sum(result.values()) == 100
