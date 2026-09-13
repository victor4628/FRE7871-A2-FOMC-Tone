"""Hawkish/dovish scoring with a monetary-policy lexicon and FinBERT anchors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINBERT_MODEL, INTERIM_DIR, WARSH_START


HAWKISH_PATTERNS = [
    r"inflation (?:remains|is|has remained) (?:too )?(?:high|elevated)",
    r"(?:persistent|persistence of|persistently) (?:high )?inflation",
    r"inflation(?:ary)? pressures?",
    r"upside risks? to inflation",
    r"inflation (?:rose|increased|accelerated)",
    r"prices? (?:rose|increased) (?:rapidly|sharply|faster)",
    r"above (?:the )?(?:committee'?s )?(?:two|2)[ -]?percent (?:objective|goal|target)",
    r"longer[- ]term inflation expectations? (?:rose|increased|moved up|are elevated)",
    r"labor market (?:remains |is )?(?:tight|strong|robust)",
    r"unemployment (?:remains |is )?(?:low|below)",
    r"job gains (?:remain |are )?(?:strong|robust|solid)",
    r"wage (?:growth|pressures?) (?:remain |is |are )?(?:strong|elevated|high)",
    r"economic activity (?:has been |is |continues to be )?(?:strong|solid|robust)",
    r"economy (?:is |remains )?(?:strong|robust|resilient)",
    r"demand (?:remains |is )?(?:strong|robust)",
    r"(?:raise|raised|increase|increased) the target range",
    r"further (?:rate )?increases?",
    r"additional (?:policy )?(?:firming|tightening|restraint)",
    r"more restrictive",
    r"maintain (?:a )?restrictive (?:stance|policy)",
    r"higher for longer",
    r"not (?:yet )?appropriate to (?:cut|reduce|lower)",
    r"premature to (?:cut|reduce|lower|ease)",
    r"reduce (?:the federal reserve'?s )?securities holdings",
    r"balance[- ]sheet runoff",
    r"restore price stability",
    r"resolute commitment to (?:restoring|restore) price stability",
    r"no tolerance for persistently elevated inflation",
]

DOVISH_PATTERNS = [
    r"inflation (?:has )?(?:eased|declined|decreased|moderated|slowed)",
    r"inflation pressures? (?:have )?(?:eased|diminished|moderated)",
    r"disinflation(?:ary)? (?:process|trend|progress)",
    r"progress (?:on|toward|towards) (?:lowering )?inflation",
    r"downside risks? to (?:employment|the labor market|growth)",
    r"labor market (?:has )?(?:cooled|softened|weakened|slowed)",
    r"labor market conditions (?:have )?(?:eased|softened|cooled)",
    r"unemployment (?:has )?(?:risen|increased|moved up)",
    r"job gains (?:have )?(?:slowed|moderated|weakened)",
    r"economic activity (?:has )?(?:slowed|weakened|moderated)",
    r"growth (?:has )?(?:slowed|weakened|moderated)",
    r"demand (?:has )?(?:slowed|weakened|softened)",
    r"below[- ]trend growth",
    r"(?:lower|lowered|reduce|reduced) the target range",
    r"(?:rate )?cuts? (?:may be|would be|are) appropriate",
    r"policy (?:easing|accommodation)",
    r"less restrictive",
    r"remove policy restraint",
    r"support (?:the committee'?s )?maximum[- ]employment goal",
    r"risks? (?:to achieving|to) (?:the committee'?s )?goals? (?:are|have moved) (?:roughly )?in balance",
    r"risks? (?:are|have become) balanced",
    r"pause (?:rate )?increases",
    r"slow(?:ing)? the pace of (?:rate )?increases",
    r"weakness in (?:the )?labor market",
    r"subdued inflation",
    r"below (?:the )?(?:committee'?s )?(?:two|2)[ -]?percent (?:objective|goal|target)",
]

HAWKISH_ANCHORS = [
    "Interest rates will rise and monetary policy will become more restrictive.",
    "Inflation is too high and persistent inflation requires additional tightening.",
    "The labor market is tight, demand is strong, and price pressures remain elevated.",
    "The Committee should raise the target range for the federal funds rate.",
    "Policy must remain restrictive for longer to restore price stability.",
    "Upside risks to inflation outweigh downside risks to employment.",
]

DOVISH_ANCHORS = [
    "Interest rates will fall and monetary policy will become less restrictive.",
    "Inflation has eased and continued disinflation permits policy accommodation.",
    "The labor market has cooled, demand has weakened, and unemployment is rising.",
    "The Committee should lower the target range for the federal funds rate.",
    "Policy restraint can be reduced to support maximum employment.",
    "Downside risks to employment outweigh upside risks to inflation.",
]


def wordlist_score(text: str) -> dict[str, float]:
    lowered = text.lower()
    hawkish = sum(len(re.findall(pattern, lowered)) for pattern in HAWKISH_PATTERNS)
    dovish = sum(len(re.findall(pattern, lowered)) for pattern in DOVISH_PATTERNS)
    words = max(len(re.findall(r"\b\w+\b", lowered)), 1)
    return {
        "hawk_hits": hawkish,
        "dove_hits": dovish,
        "wordlist_score": 1000.0 * (hawkish - dovish) / words,
    }


def _chunks(text: str, max_words: int = 80, max_chunks: int = 20) -> list[str]:
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if len(s.split()) >= 4
    ]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    if not chunks:
        chunks = [" ".join(text.split()[:max_words])]
    if len(chunks) > max_chunks:
        indices = np.unique(np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int))
        chunks = [chunks[index] for index in indices]
    return chunks


class FinBERTAnchorScorer:
    def __init__(self, model_name: str = FINBERT_MODEL, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.hawk = self._encode(HAWKISH_ANCHORS)
        self.dove = self._encode(DOVISH_ANCHORS)

    def _encode(self, texts: list[str]):
        torch = self.torch
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(batches, dim=0)

    def score(self, text: str) -> float:
        vectors = self._encode(_chunks(text))
        hawk_similarity = (vectors @ self.hawk.T).mean(dim=1)
        dove_similarity = (vectors @ self.dove.T).mean(dim=1)
        return float((hawk_similarity - dove_similarity).mean().item() * 100.0)


def score_documents(documents: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    destination = INTERIM_DIR / "document_scores.csv"
    existing = pd.DataFrame()
    if destination.exists() and not refresh:
        existing = pd.read_csv(destination)
        if set(documents.document_id) == set(existing.document_id):
            score_columns = [
                "document_id", "hawk_hits", "dove_hits", "wordlist_score",
                "finbert_anchor_score", "wordlist_z", "finbert_z",
            ]
            merged = documents.merge(existing[score_columns], on="document_id", how="left")
            merged.to_csv(destination, index=False)
            return merged

    lexical = pd.DataFrame([wordlist_score(text) for text in documents.text])
    result = pd.concat([documents.reset_index(drop=True), lexical], axis=1)

    cached: dict[str, float] = {}
    if destination.exists():
        prior = pd.read_csv(destination)
        if {"document_id", "finbert_anchor_score"}.issubset(prior.columns):
            cached = dict(zip(prior.document_id, prior.finbert_anchor_score))
    scorer = FinBERTAnchorScorer()
    scores: list[float] = []
    for index, row in result.iterrows():
        document_id = row.document_id
        value = cached.get(document_id)
        if value is None or not np.isfinite(value):
            value = scorer.score(row.text)
        scores.append(float(value))
        if (index + 1) % 20 == 0:
            print(f"FinBERT: {index + 1}/{len(result)} documents", flush=True)
            partial = result.iloc[: index + 1].copy()
            partial["finbert_anchor_score"] = scores
            partial.to_csv(destination, index=False)
    result["finbert_anchor_score"] = scores

    for raw, standardized in [
        ("wordlist_score", "wordlist_z"),
        ("finbert_anchor_score", "finbert_z"),
    ]:
        result[standardized] = np.nan
        for doc_type, group in result.groupby("doc_type"):
            baseline = group[group.chair == "Powell"][raw]
            mean = baseline.mean()
            std = baseline.std(ddof=1)
            result.loc[group.index, standardized] = ((group[raw] - mean) / std).clip(-4, 4)
    result.to_csv(destination, index=False)
    return result


if __name__ == "__main__":
    from .collect_fed import collect_fed_documents

    frame = score_documents(collect_fed_documents(), refresh=False)
    print(frame.groupby(["chair", "doc_type"])[["wordlist_z", "finbert_z"]].mean().round(3))
