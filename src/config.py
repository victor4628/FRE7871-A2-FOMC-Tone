from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = ROOT / "outputs"
PDF_DIR = ROOT / "output" / "pdf"

START_DATE = "2018-02-05"
AS_OF_DATE = "2026-09-13"
POWELL_START = "2018-02-05"
WARSH_START = "2026-05-22"

FED_BASE = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FINBERT_MODEL = "ProsusAI/finbert"

for directory in (RAW_DIR, INTERIM_DIR, OUTPUT_DIR, PDF_DIR):
    directory.mkdir(parents=True, exist_ok=True)
