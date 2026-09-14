"""Auditable speaker/section selection; raw text is retained separately."""
from __future__ import annotations
import hashlib
import re
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

VERSION = 'selection-v2-20260913'
params = PunktParameters()
params.abbrev_types = {'u.s', 'mr', 'mrs', 'ms', 'dr', 'st', 'a.m', 'p.m', 'e.g', 'i.e'}
SPLITTER = PunktSentenceTokenizer(params)
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
SPEAKER = re.compile(r'(?<![A-Za-z])([A-Z][A-Z\-\u2019\' ]{2,55})\.\s+')

def clean_prose(text: str) -> str:
    text = text.replace('\u00ad', '').replace('\u00a0', ' ')
    text = re.sub(MONTH + r'\s+\d{1,2},\s+\d{4}\s+(?:Chairman|Chair)\s+(?:Powell|Warsh)[’\']s Press Conference\s+(?:FINAL|PRELIMINARY)(?:\s+Page\s+\d+\s+of\s+\d+)?', ' ', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', ' ', text)
    text = re.sub(r'\[(?:Laughter|inaudible|cross talk|Applause)\]', ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()

def selected_segments(text: str, subtype: str) -> list[tuple[str, str]]:
    text = clean_prose(text)
    if subtype == 'press_conference':
        marks = list(SPEAKER.finditer(text))
        chair_seen = False
        segments = []
        for i, match in enumerate(marks):
            if re.fullmatch(r'CHAIR(?:MAN)? (?:POWELL|WARSH)', match.group(1)):
                body = text[match.end():marks[i+1].start() if i+1 < len(marks) else len(text)]
                segments.append(('prepared_remarks' if not chair_seen else 'chair_answer', body))
                chair_seen = True
        if not chair_seen:
            raise ValueError('No Chair speaker labels; inspect transcript before scoring')
        return segments
    if subtype == 'minutes':
        start = re.search(r'Participants[’\']? Views on (?:Current Conditions|the Economic)', text, re.I)
        if not start:
            start = re.search(r'Committee (?:Policy Action|Discussion)', text, re.I)
        if not start:
            raise ValueError('No participants/policy section; inspect minutes before scoring')
        text = text[start.start():]
        text = re.split(r'Voting (?:for|against)|Votes for this action|Attendance|List of (?:Attendees|Participants)', text, maxsplit=1, flags=re.I)[0]
        return [('committee_views_and_policy', text)]
    if subtype == 'statement':
        text = re.split(r'Voting (?:\(by notation\) )?(?:for|against)|For media inquiries', text, maxsplit=1, flags=re.I)[0]
    else:
        text = re.split(r'\bReferences\b|\bReturn to text\b|\bFor media inquiries\b', text, maxsplit=1)[0]
        text = re.split(r'\s1\.\s', text, maxsplit=1)[0]
    return [('body', text)]

def prepare_documents(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs, sentences = [], []
    for row in raw.to_dict('records'):
        if row['subtype'] == 'statement' and row['release_date'] in {'2020-03-31', '2020-08-27'}:
            continue  # Facility announcement / strategy review, not meeting statements.
        row['raw_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
        row['selection_version'] = VERSION
        match = re.search(r'(?:fomcminutes|monetary|FOMCpresconf)(\d{8})', row['source_url'])
        row['meeting_date'] = pd.to_datetime(match.group(1)).date().isoformat() if match else ''
        row['meeting_chair'] = ('Yellen' if row['meeting_date'] and row['meeting_date'] < '2018-02-05' else row['chair'])
        if 'default' in row['time_source']:
            row['release_time_et'] = ''
            row['time_source'] = 'unknown; old imputed time removed'
        selected = []
        for section, prose in selected_segments(row['text'], row['subtype']):
            for sentence in SPLITTER.tokenize(prose):
                sentence = sentence.strip()
                if len(re.findall(r'[A-Za-z]+', sentence)) < 4:
                    continue
                selected.append(sentence)
                sentences.append({'document_id': row['document_id'], 'sentence_id': len(selected)-1,
                                  'section': section, 'sentence': sentence})
        row['eligible_sentences'] = len(selected)
        row['selected_text'] = ' '.join(selected)
        row['selected_sha256'] = hashlib.sha256(row['selected_text'].encode()).hexdigest()
        docs.append(row)
    return pd.DataFrame(docs), pd.DataFrame(sentences)
