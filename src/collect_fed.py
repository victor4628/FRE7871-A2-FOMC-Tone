"""Collect FOMC statements, minutes, and Chair communications from federalreserve.gov."""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import (
    AS_OF_DATE,
    FED_BASE,
    FOMC_CALENDAR_URL,
    POWELL_START,
    RAW_DIR,
    START_DATE,
    WARSH_START,
)


USER_AGENT = "FRE-GY-7871A-course-project/1.0 (public academic data collection)"
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


@dataclass
class FedDocument:
    document_id: str
    release_date: str
    release_time_et: str
    time_source: str
    doc_type: str
    subtype: str
    chair: str
    title: str
    source_url: str
    text: str
    word_count: int


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get(session: requests.Session, url: str, attempts: int = 4) -> requests.Response:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to download {url}") from error


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chair_for_date(value: str) -> str:
    return "Warsh" if value >= WARSH_START else "Powell"


def _parse_time(value: str | None, default: str = "12:00") -> str:
    if not value:
        return default
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?", value, re.I)
    if not match:
        return default
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if match.group(3).lower() == "p" and hour != 12:
        hour += 12
    if match.group(3).lower() == "a" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _article_text(session: requests.Session, url: str) -> tuple[str, str, str | None]:
    soup = BeautifulSoup(_get(session, url).content, "lxml")
    article = soup.select_one("#article")
    if article is None:
        raise ValueError(f"No article body found at {url}")
    title_node = article.select_one("h2, h1, .article__title")
    title = _clean_text(title_node.get_text(" ", strip=True)) if title_node else ""
    release = soup.select_one(".releaseTime")
    release_text = release.get_text(" ", strip=True) if release else None

    # Speech, testimony, and press-release pages wrap the prose in an inner column.
    candidates = article.select(".col-xs-12.col-sm-8.col-md-8")
    body = max(candidates, key=lambda node: len(node.get_text(" ", strip=True))) if candidates else article
    for node in body.select("script, style, nav, .share, .article__share, #lastUpdate"):
        node.decompose()
    text = _clean_text(body.get_text(" ", strip=True))
    return title, text, release_text


def _pdf_text(session: requests.Session, url: str) -> str:
    reader = PdfReader(io.BytesIO(_get(session, url).content))
    return _clean_text(" ".join(page.extract_text() or "" for page in reader.pages))


def _month_events(session: requests.Session, year: int, month: int) -> list[dict[str, str]]:
    url = f"{FED_BASE}/newsevents/{year}-{MONTHS[month - 1]}.htm"
    response = session.get(url, timeout=45)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")
    events: list[dict[str, str]] = []
    for row in soup.select(".panel-body > .row"):
        cols = row.find_all("div", recursive=False)
        if len(cols) < 3:
            continue
        day_match = re.search(r"\b(\d{1,2})\b", cols[-1].get_text(" ", strip=True))
        if not day_match:
            continue
        title_node = cols[1].select_one(".calendar__title")
        events.append(
            {
                "day": day_match.group(1),
                "time": _parse_time(cols[0].get_text(" ", strip=True)),
                "description": _clean_text(cols[1].get_text(" ", strip=True)),
                "title": _clean_text(title_node.get_text(" ", strip=True)) if title_node else "",
            }
        )
    return events


def _calendar_time(
    session: requests.Session,
    cache: dict[tuple[int, int], list[dict[str, str]]],
    release_date: str,
    kind: str,
    surname: str = "",
    title: str = "",
    default: str = "12:00",
) -> tuple[str, str]:
    stamp = pd.Timestamp(release_date)
    key = (stamp.year, stamp.month)
    if key not in cache:
        cache[key] = _month_events(session, *key)
    candidates = [e for e in cache[key] if int(e["day"]) == stamp.day]
    lowered_kind = kind.lower()
    # Chair index pages label some speeches as panels, discussions, or ceremonies.
    # Date, speaker, and title are more reliable identifiers for those items.
    if lowered_kind not in {"speech", "testimony"}:
        candidates = [e for e in candidates if lowered_kind in e["description"].lower()]
    if surname:
        surname_matches = [e for e in candidates if surname.lower() in e["description"].lower()]
        candidates = surname_matches or candidates
    if title and len(candidates) > 1:
        title_words = set(re.findall(r"[a-z]{4,}", title.lower()))
        candidates.sort(
            key=lambda e: len(title_words & set(re.findall(r"[a-z]{4,}", e["title"].lower()))),
            reverse=True,
        )
    if candidates:
        return candidates[0]["time"], "Federal Reserve events calendar"
    return default, "document-type default; calendar time unavailable"


def _record(
    *, release_date: str, release_time: str, time_source: str, doc_type: str,
    subtype: str, title: str, url: str, text: str,
) -> FedDocument:
    key = re.sub(r"\W+", "", url.rsplit("/", 1)[-1].split(".")[0])
    return FedDocument(
        document_id=f"{release_date}_{subtype}_{key}",
        release_date=release_date,
        release_time_et=release_time,
        time_source=time_source,
        doc_type=doc_type,
        subtype=subtype,
        chair=_chair_for_date(release_date),
        title=title,
        source_url=url,
        text=text,
        word_count=len(re.findall(r"\b\w+\b", text)),
    )


def _collect_meeting_documents(
    session: requests.Session, calendar_cache: dict[tuple[int, int], list[dict[str, str]]]
) -> list[FedDocument]:
    calendar_urls = [FOMC_CALENDAR_URL] + [
        f"{FED_BASE}/monetarypolicy/fomchistorical{year}.htm" for year in (2018, 2019, 2020)
    ]
    soups = [BeautifulSoup(_get(session, url).content, "lxml") for url in calendar_urls]
    documents: list[FedDocument] = []

    statement_pattern = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm$")
    for soup in soups:
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            match = statement_pattern.search(href)
            if not match:
                continue
            release_date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
            if not (START_DATE <= release_date <= AS_OF_DATE):
                continue
            url = urljoin(FED_BASE, href)
            title, text, release_text = _article_text(session, url)
            documents.append(
                _record(
                    release_date=release_date,
                    release_time=_parse_time(release_text, "14:00"),
                    time_source="article release line" if release_text else "FOMC standard release time",
                    doc_type="Statement",
                    subtype="statement",
                    title=title or "FOMC statement",
                    url=url,
                    text=text,
                )
            )

    minutes_pattern = re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm$")
    for soup in soups:
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            match = minutes_pattern.search(href)
            if not match:
                continue
            container = anchor.find_parent(class_="fomc-meeting__minutes") or anchor.parent
            released = re.search(
                r"Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
                container.get_text(" ", strip=True) if container else "",
            )
            if not released:
                continue
            release_date = datetime.strptime(released.group(1), "%B %d, %Y").date().isoformat()
            if not (START_DATE <= release_date <= AS_OF_DATE):
                continue
            url = urljoin(FED_BASE, href)
            title, text, _ = _article_text(session, url)
            release_time, source = _calendar_time(
                session, calendar_cache, release_date, "FOMC Minutes", default="14:00"
            )
            documents.append(
                _record(
                    release_date=release_date,
                    release_time=release_time,
                    time_source=source,
                    doc_type="Minutes",
                    subtype="minutes",
                    title=title or f"FOMC minutes for meeting ending {match.group(1)}",
                    url=url,
                    text=text,
                )
            )

    conference_pattern = re.compile(r"/monetarypolicy/fomcpres+conf(\d{8})\.htm$")
    for soup in soups:
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            match = conference_pattern.search(href)
            if not match:
                continue
            release_date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
            if not (START_DATE <= release_date <= AS_OF_DATE):
                continue
            landing_url = urljoin(FED_BASE, href)
            landing = BeautifulSoup(_get(session, landing_url).content, "lxml")
            transcript = landing.find("a", href=re.compile(r"FOMCpresconf\d{8}\.pdf$", re.I))
            if transcript is None:
                continue
            transcript_url = urljoin(FED_BASE, transcript.get("href", ""))
            text = _pdf_text(session, transcript_url)
            chair = _chair_for_date(release_date)
            documents.append(
                _record(
                    release_date=release_date,
                    release_time="14:30",
                    time_source="FOMC press-conference schedule",
                    doc_type="Chair communication",
                    subtype="press_conference",
                    title=f"Chair {chair} FOMC press conference",
                    url=transcript_url,
                    text=text,
                )
            )
    return documents


def _collect_chair_indexes(
    session: requests.Session, calendar_cache: dict[tuple[int, int], list[dict[str, str]]]
) -> list[FedDocument]:
    documents: list[FedDocument] = []
    for year in range(pd.Timestamp(START_DATE).year, pd.Timestamp(AS_OF_DATE).year + 1):
        for subtype, plural, doc_type in [
            ("speech", "speeches", "Chair communication"),
            ("testimony", "testimony", "Chair communication"),
        ]:
            index_url = f"{FED_BASE}/newsevents/{subtype}/{year}-{plural}.htm"
            soup = BeautifulSoup(_get(session, index_url).content, "lxml")
            for row in soup.select(".row"):
                time_node = row.select_one(".eventlist__time time")
                event_node = row.select_one(".eventlist__event")
                if time_node is None or event_node is None:
                    continue
                speaker_node = event_node.select_one(".news__speaker")
                speaker = _clean_text(speaker_node.get_text(" ", strip=True)) if speaker_node else ""
                if not re.search(r"\b(Chair(?:man)?|Chair Pro Tempore)\b", speaker, re.I):
                    continue
                if not re.search(r"\b(Powell|Warsh)\b", speaker, re.I):
                    continue
                anchor = event_node.find("a", href=re.compile(rf"/{subtype}/(?:powell|warsh)\d{{8}}", re.I))
                if anchor is None:
                    continue
                release_date = pd.to_datetime(time_node.get_text(strip=True)).date().isoformat()
                if not (START_DATE <= release_date <= AS_OF_DATE):
                    continue
                expected = _chair_for_date(release_date)
                if expected.lower() not in speaker.lower():
                    continue
                title = _clean_text(anchor.get_text(" ", strip=True))
                url = urljoin(FED_BASE, anchor.get("href", ""))
                page_title, text, release_text = _article_text(session, url)
                release_time, source = _calendar_time(
                    session,
                    calendar_cache,
                    release_date,
                    subtype,
                    surname=expected,
                    title=title,
                    default="12:00",
                )
                if release_text:
                    release_time, source = _parse_time(release_text), "article release line"
                documents.append(
                    _record(
                        release_date=release_date,
                        release_time=release_time,
                        time_source=source,
                        doc_type=doc_type,
                        subtype=subtype,
                        title=page_title or title,
                        url=url,
                        text=text,
                    )
                )
    return documents


def collect_fed_documents(refresh: bool = False) -> pd.DataFrame:
    """Return the complete document corpus and cache it under ignored data/."""
    destination = RAW_DIR / "fed_documents.jsonl"
    if destination.exists() and not refresh:
        return pd.read_json(destination, lines=True, convert_dates=False)
    session = _session()
    calendar_cache: dict[tuple[int, int], list[dict[str, str]]] = {}
    records = _collect_meeting_documents(session, calendar_cache)
    records.extend(_collect_chair_indexes(session, calendar_cache))

    unique = {record.source_url: record for record in records}
    ordered = sorted(unique.values(), key=lambda item: (item.release_date, item.release_time_et, item.subtype))
    if not ordered:
        raise RuntimeError("Federal Reserve collector returned no documents")
    if min(item.word_count for item in ordered) < 40:
        short = [(item.source_url, item.word_count) for item in ordered if item.word_count < 40]
        raise ValueError(f"Suspiciously short documents: {short}")
    with destination.open("w", encoding="utf-8") as stream:
        for item in ordered:
            stream.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    return pd.DataFrame(asdict(item) for item in ordered)


if __name__ == "__main__":
    frame = collect_fed_documents(refresh=True)
    print(frame.groupby(["doc_type", "chair"]).size().unstack(fill_value=0))
    print(f"Collected {len(frame)} documents from {frame.release_date.min()} through {frame.release_date.max()}.")
