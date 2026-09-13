# Assignment 2: FOMC Communication and Asset Prices

**Victor Chen (yc8027)** - FRE-GY 7871 A, Fall 2026

This project collects 315 Federal Reserve communications from February 2018
through September 13, 2026, compares Powell- and Warsh-era tone with a contextual
monetary-policy phrase list and FinBERT anchors, tests daily market reactions, and
forecasts the September 2026 FOMC meeting.

## Main submission files

- `assignment2.ipynb` - viewable notebook with all outputs saved
- `output/pdf/assignment2_report.pdf` - short report for Brightspace
- `AI_USE.md` - complete AI-use disclosure
- `METHODOLOGY.md` - detailed definitions and modeling choices
- `src/` - collectors, scoring, regressions, forecasting, and plotting
- `tests/` - numerical and timing checks

## Reproduce

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py --refresh --rescore
python scripts/build_notebook.py
python scripts/build_report.py
```

`ProsusAI/finbert` is downloaded from Hugging Face on the first scoring run. Raw
and intermediate data are intentionally excluded from Git. See `RUNNING_zh.md`
for Chinese setup notes.

## Data sources

- [Federal Reserve FOMC calendars and documents](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [Federal Reserve H.15 Treasury data](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15)
- Yahoo Finance: `DX-Y.NYB`, `IWF`, and `IWN`
- Hugging Face: `ProsusAI/finbert`

The forecast is an academic exercise and is not investment advice.
