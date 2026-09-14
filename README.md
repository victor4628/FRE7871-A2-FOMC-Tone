# Assignment 2: FOMC Communication and Asset Prices

**Victor Chen (yc8027)** - FRE-GY 7871 A, Fall 2026
Information cutoff: **September 13, 2026**

This repository analyzes 310 Federal Reserve communications released from Jerome
Powell's first day as Chair through the last pre-meeting information date. It
compares the Powell and Kevin Warsh periods using three deliberately distinct
text measures:

1. the published monetary-policy rule in Shah, Paturi, and Chava (ACL 2023);
2. a RoBERTa-large reproduction trained on their public 1996-2019 labels; and
3. `ProsusAI/finbert` financial sentiment, kept as sentiment rather than relabeled
   as hawkishness.

Daily DXY, 10s2s, 1-year Treasury, and IWF-minus-IWN changes are tested with the
DGS3MO change as a control. The event study preserves missing observations and
uses source-specific closing conventions. Results are descriptive daily
associations, not high-frequency causal communication shocks.

The meeting forecast is generated internally from pre-meeting text. A regularized
multinomial model combines the previous decision with the latest statement,
minutes, and intermeeting Chair scores. No CME or other market-implied decision
probability enters the forecast.

## Submission files

- `assignment2.ipynb` - viewable notebook with saved outputs
- `output/pdf/assignment2_report.pdf` - short report for Brightspace
- `AI_USE.md` - AI-use disclosure
- `METHODOLOGY.md` - definitions, provenance, and limitations
- `src/` - collection, preprocessing, scoring, event-study, and forecast code
- `tests/` - targeted score and event-window checks

Raw, intermediate, model, and derived data are excluded from Git as required.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python -m src.train_temporal_roberta
$env:FOMC_MODEL_PATH = "data/models/fomc-roberta-temporal"
python scripts/run_v2.py --refresh-documents --refresh-market --rescore
python scripts/build_notebook_v2.py
python scripts/build_report_v2.py
```

The official FOMC-RoBERTa checkpoint is gated. First request access at
<https://huggingface.co/gtfintechlab/FOMC-RoBERTa>, then run `hf auth login` with
a read-only token. The current saved results use the disclosed temporal
reproduction because the official access request was still awaiting author review.
After approval, remove `FOMC_MODEL_PATH`; the scoring code will use the official
checkpoint directly.
See `RUNNING_zh.md` for Windows instructions.

## Primary sources

- [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [FRED T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y),
  [DGS1](https://fred.stlouisfed.org/series/DGS1), and
  [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)
- Yahoo Finance tickers `DX-Y.NYB`, `IWF`, and `IWN`
- [Shah, Paturi, and Chava (2023)](https://aclanthology.org/2023.acl-long.368/)
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

The forecast and position are an academic exercise.
