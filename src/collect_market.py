"""Download and align the four required indicators and the DGS3MO control."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .config import AS_OF_DATE, RAW_DIR, START_DATE


H15_TREASURY_CSV = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?filetype=csv&from="
    "&label=include&lastobs=&layout=seriescolumn&rel=H15"
    "&series=bf17364827e38702b42a58cf8eaa3f78&to=&type=package"
)


def _h15_treasuries(start: str, end: str) -> pd.DataFrame:
    """Fetch the Board's H.15 package underlying the named FRED series."""
    response = requests.get(
        H15_TREASURY_CSV,
        timeout=60,
        headers={"User-Agent": "FRE-GY-7871A-course-project/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content), skiprows=5, na_values=["ND"])
    frame = frame.rename(
        columns={
            "Time Period": "date",
            "RIFLGFCM03_N.B": "dgs3mo",
            "RIFLGFCY01_N.B": "dgs1",
            "RIFLGFCY02_N.B": "dgs2",
            "RIFLGFCY10_N.B": "dgs10",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date")
    for column in ("dgs3mo", "dgs1", "dgs2", "dgs10"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["t10y2y"] = frame["dgs10"] - frame["dgs2"]
    return frame.loc[pd.Timestamp(start):pd.Timestamp(end), ["t10y2y", "dgs1", "dgs3mo"]]


def _prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ["DX-Y.NYB", "IWF", "IWN"],
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=2)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={"DX-Y.NYB": "dxy", "IWF": "iwf", "IWN": "iwn"})
    required = {"dxy", "iwf", "iwn"}
    if not required.issubset(close.columns):
        raise ValueError(f"Missing Yahoo Finance columns: {required - set(close.columns)}")
    return close[list(sorted(required))].sort_index()


def collect_market_data(refresh: bool = False) -> pd.DataFrame:
    destination = RAW_DIR / "market_daily_verified.csv"
    if destination.exists() and not refresh:
        return pd.read_csv(destination, parse_dates=["date"]).set_index("date")
    padded_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=40)).date().isoformat()
    old = RAW_DIR / 'market_daily.csv'
    prices = (pd.read_csv(old,parse_dates=['date']).set_index('date')[['dxy','iwf','iwn']]
              if old.exists() and not refresh else _prices(padded_start, AS_OF_DATE))
    panel = prices.copy()
    url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y,DGS1,DGS3MO'
    response=requests.get(url,timeout=60)
    response.raise_for_status()
    (RAW_DIR/'fred_original.csv').write_bytes(response.content)
    frame=pd.read_csv(io.BytesIO(response.content),na_values=['.'])
    frame=frame.rename(columns={'observation_date':'date','DATE':'date','T10Y2Y':'t10y2y','DGS1':'dgs1','DGS3MO':'dgs3mo'})
    frame['date']=pd.to_datetime(frame.date)
    frame=frame.set_index('date').loc[padded_start:AS_OF_DATE]
    for column in ['t10y2y','dgs1','dgs3mo']:frame[column]=pd.to_numeric(frame[column],errors='coerce')
    panel=panel.join(frame,how='outer')
    sources=[{'series':column,'url':url,'last_observation':str(frame[column].last_valid_index().date())}
             for column in ['t10y2y','dgs1','dgs3mo']]
    panel=panel.loc[padded_start:AS_OF_DATE].sort_index()
    panel.index.name = "date"
    if panel[["dxy", "iwf", "iwn", "t10y2y", "dgs1", "dgs3mo"]].dropna().empty:
        raise RuntimeError("Aligned market panel has no complete observations")
    panel.to_csv(destination)
    (RAW_DIR/'market_provenance.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'fred':sources,'prices':'Reused Yahoo adjusted-close cache' if old.exists() and not refresh else 'Yahoo adjusted-close download',
        'missing_values':'Retained; no forward filling'},indent=2))
    return panel


if __name__ == "__main__":
    data = collect_market_data(refresh=True)
    print(data.tail())
    print(f"Market observations: {len(data)}")
