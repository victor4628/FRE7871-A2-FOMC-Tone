"""Download and align the four required indicators and the DGS3MO control."""

from __future__ import annotations

import io

import pandas as pd
import requests
import yfinance as yf

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
    destination = RAW_DIR / "market_daily.csv"
    if destination.exists() and not refresh:
        return pd.read_csv(destination, parse_dates=["date"]).set_index("date")
    padded_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=40)).date().isoformat()
    prices = _prices(padded_start, AS_OF_DATE)
    panel = prices.copy()
    treasuries = _h15_treasuries(padded_start, AS_OF_DATE)
    for series in ("t10y2y", "dgs1", "dgs3mo"):
        panel[series] = treasuries[series].reindex(panel.index).ffill()
    panel.index.name = "date"
    if panel[["dxy", "iwf", "iwn", "t10y2y", "dgs1", "dgs3mo"]].dropna().empty:
        raise RuntimeError("Aligned market panel has no complete observations")
    panel.to_csv(destination)
    return panel


if __name__ == "__main__":
    data = collect_market_data(refresh=True)
    print(data.tail())
    print(f"Market observations: {len(data)}")
