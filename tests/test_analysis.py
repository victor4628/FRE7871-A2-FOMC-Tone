import pandas as pd

from src.analysis import _decision, _rounded_percentages, attach_market_reactions
from src.tone import wordlist_score


def test_policy_phrase_list_orients_hawkish_and_dovish_language():
    hawkish = wordlist_score("Inflation remains elevated. The Committee raised the target range.")
    dovish = wordlist_score("Inflation has eased. The Committee lowered the target range.")
    assert hawkish["wordlist_score"] > 0
    assert dovish["wordlist_score"] < 0


def test_decision_classifier():
    assert _decision("The Committee decided to maintain the target range.") == "Hold"
    assert _decision("The Committee decided to raise the target range.") == "Hike"
    assert _decision("The Committee decided to lower the target range.") == "Cut"


def test_event_window_rolls_weekend_to_next_close():
    market = pd.DataFrame(
        {
            "dxy": [100, 101], "iwf": [100, 102], "iwn": [100, 101],
            "t10y2y": [0.4, 0.5], "dgs1": [4.0, 4.1], "dgs3mo": [3.8, 3.8],
        },
        index=pd.to_datetime(["2026-09-11", "2026-09-14"]),
    )
    documents = pd.DataFrame(
        [{"document_id": "x", "release_date": "2026-09-12", "release_time_et": "10:00"}]
    )
    result = attach_market_reactions(documents, market).iloc[0]
    assert result.window_start == "2026-09-11"
    assert result.window_end == "2026-09-14"
    assert round(result.dxy_change_pct, 6) == 1.0


def test_rounded_probabilities_sum_to_100():
    result = _rounded_percentages({"Cut": 0.111, "Hold": 0.721, "Hike": 0.168})
    assert sum(result.values()) == 100

