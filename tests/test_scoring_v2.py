import pandas as pd
import pytest
from src.scoring import rule_sentence, wordlist_score
from src.preprocess import selected_segments, SPLITTER

def test_published_rule_precedence_and_negation():
    assert rule_sentence('Interest rates will rise.')['rule_label']==1
    assert rule_sentence('Interest rates will not rise.')['rule_label']==0
    assert rule_sentence('Interest rates will fall.')['rule_label']==0
    x=rule_sentence('Interest rates will rise or fall.')
    assert x['rule_conflict'] and x['rule_label']==0

def test_document_denominator_and_no_policy_content():
    assert wordlist_score('Interest rates will rise. Interest rates will fall. Inflation is discussed.')['wordlist_score']==0
    assert pd.isna(wordlist_score('Thank you for joining us.')['wordlist_score'])

def test_reporter_excluded_and_chair_answers_kept():
    segments=selected_segments('CHAIR POWELL. Interest rates will rise. STEVE LIESMAN. Will you cut rates? CHAIR POWELL. We do not expect that.', 'press_conference')
    assert [s for s,_ in segments]==['prepared_remarks','chair_answer']
    assert 'Will you cut' not in ' '.join(t for _,t in segments)
    assert 'We do not expect' in segments[1][1]

def test_minutes_exclude_staff_views():
    segments=selected_segments("Staff Economic Outlook Staff expect a recession. Participants’ Views on Current Conditions Participants expect expansion. Committee Policy Action Rates will rise. Voting for this action were X.",'minutes')
    text=segments[0][1]
    assert 'recession' not in text and 'Rates will rise' in text and 'Voting' not in text

def test_sentence_boundaries_keep_decimal_and_abbreviation():
    assert len(SPLITTER.tokenize('The U.S. economy grew by 2.5 percent. Inflation rose.'))==2
