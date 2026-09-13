"""Build and execute the submission notebook with outputs embedded."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "interim" / "matplotlib"))


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# Assignment 2: FOMC Communication and Asset Prices

**Victor Chen (yc8027)**  
FRE-GY 7871 A - NLP and the Investment Process - Fall 2026  
Information set: **September 13, 2026**

This notebook studies FOMC statements, minutes, and Chair communications from Jerome Powell's first day as Chair through the latest pre-meeting information. It uses a monetary-policy phrase list and a FinBERT anchor-similarity score, then relates tone to daily market moves while controlling for the 3-month Treasury yield.
"""),
    md("""
## Reproduction and research design

From `assignment2/`, create an environment and install `requirements.txt`, then run:

```bash
python scripts/run_all.py --refresh --rescore
python scripts/build_notebook.py
```

Raw/intermediate data are downloaded from public sources into ignored `data/` directories and are not committed. The saved outputs below make the notebook viewable without rerunning the downloads. Release times are matched to the Federal Reserve event calendar; pre-close releases use prior-close to same-close changes, while after-close/non-trading-day releases roll to the next close.
"""),
    code("""
from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if ROOT.name != 'assignment2':
    ROOT = ROOT / 'assignment2'
sys.path.insert(0, str(ROOT))

from src.analysis import document_counts

scores = pd.read_csv(ROOT/'data/interim/document_scores.csv')
events = pd.read_csv(ROOT/'data/interim/event_reactions.csv')
regressions = pd.read_csv(ROOT/'outputs/table3_regressions.csv')
forecast = json.loads((ROOT/'outputs/forecast.json').read_text(encoding='utf-8'))
print(f"Documents: {len(scores)} | Market-linked events: {len(events)}")
print(f"Sample: {scores.release_date.min()} to {scores.release_date.max()}")
"""),
    md("""
## Methods

The phrase-list score is the number of hawkish minus dovish monetary-policy phrase matches per 1,000 words. It includes contextual phrases (for example, “inflation remains elevated” and “inflation has eased”) rather than treating isolated words such as *higher* as hawkish. The language-model score embeds stratified document chunks with `ProsusAI/finbert` and subtracts average cosine similarity to six dovish anchors from similarity to six hawkish anchors. Both measures are standardized within document type using the Powell period, so zero is the Powell-type mean and positive values are more hawkish.

This adapts the factor-similarity, phrase-list, and FinBERT ideas in *Parsing the Fed*. It also follows Doh, Kim, and Yang (2021) in separating qualitative language from the rate action, but the daily window is coarser than their intraday design.
"""),
    md("## Table 1. Documents collected, by type and Chair"),
    code("""
table1 = pd.read_csv(ROOT/'outputs/table1_document_counts.csv', index_col=0)
display(table1)
"""),
    md("""
The sample contains 307 Powell-era and eight Warsh-era documents. Press conferences are a subtype of Chair communication, so the five displayed rows sum to the three assignment categories without double-counting.
"""),
    md("## Figure 1. Hawkish/dovish tone over time by document type"),
    code("display(Image(filename=str(ROOT/'outputs/figure1_tone_trends.png'), width=1000))"),
    code("""
warsh_means = scores.groupby(['chair','doc_type'])[['wordlist_z','finbert_z']].mean().round(2)
display(warsh_means)
"""),
    md("""
Warsh-era statements are more hawkish than the Powell statement baseline by **1.18 SD** using the phrase list and **0.47 SD** using FinBERT anchors. Chair communications are 0.26 and 0.55 SD more hawkish, respectively. Minutes are close to the Powell baseline on average. The methods disagree on the July statement and August minutes, an important indication of model uncertainty rather than a reason to select one score ex post.
"""),
    md("## Table 2. Warsh-era releases and one-day market reactions"),
    code("""
cols = ['release_date','subtype','wordlist_z','finbert_z','dxy_change_pct','spread_change_bp','dgs1_change_bp','growth_value_pp']
table2 = pd.read_csv(ROOT/'outputs/table2_warsh_releases.csv')[cols].copy()
table2.columns = ['Date','Release','Word-list z','FinBERT z','DXY %','10s2s bp','1Y bp','G-V pp']
display(table2.round(2))
"""),
    md("""
Market reactions are not uniform. The June statement/press conference coincided with a 14 bp rise in the one-year yield and a 9 bp flattening, whereas the July statement/press conference coincided with a 5 bp one-year yield decline and a 10 bp steepening. Same-day statement and press-conference rows share the daily window; this is transparent but limits causal separation at daily frequency.
"""),
    md("## Table 3. Market-change regressions with the 3-month bill control"),
    code("""
view = regressions.copy()
view['Tone (HC3 SE)'] = view.apply(lambda r: f"{r['Tone coefficient']:.3f} ({r['Tone SE']:.3f})", axis=1)
view['DGS3MO (HC3 SE)'] = view.apply(lambda r: f"{r['DGS3MO coefficient']:.3f} ({r['DGS3MO SE']:.3f})", axis=1)
display(view[['Indicator','Tone method','Tone (HC3 SE)','DGS3MO (HC3 SE)','Adjusted R2','N']].round({'Adjusted R2':3}))
"""),
    md("""
Each row is a separate OLS regression with HC3 standard errors and document-type fixed effects. No tone coefficient is significant at 10%. The DGS3MO control is the dominant term for DXY and the Treasury indicators; adjusted R-squared ranges from approximately 0% for growth minus value to 32% for the one-year yield. This is weaker tone evidence than the readings, consistent with our much wider daily window, mixed document types, and the absence of a direct surprise/novelty measure.
"""),
    md("## Forecast for the September 15-16, 2026 FOMC meeting"),
    code("""
rate = pd.Series(forecast['rate_probabilities'], name='Probability (%)')
market_fc = pd.DataFrame(forecast['market_reaction']).T
display(rate.to_frame())
print(f"Probability statement is more hawkish than July: {forecast['more_hawkish_probability']}%")
display(market_fc)
"""),
    md("""
The regularized multinomial model uses 68 prior meetings, the previous statement tone, interim communications, and 20-trading-day changes in the one-year yield and 10s2s spread; its probabilities are blended 80/20 with Laplace-smoothed sample frequencies. The statement-tone model uses the same predictors. Market sizes and sign probabilities average predictions from the two tone regressions and include residual uncertainty.

**Recommendation.** Put on a small DV01-neutral 2s10s flattener (receive fixed in 10-year swaps and pay fixed in 2-year swaps). The model assigns a 65% probability that the 10s2s spread falls and expects about a 1.6 bp decline, the strongest directional signal among the four indicators. The position is deliberately small because the tone coefficients are imprecise. A meeting-day steepening of more than 5 bp, especially alongside a cut and a materially more dovish statement, would reject the thesis.

This is an academic forecast, not investment advice.
"""),
    md("""
## Comparison with the readings

- **Doh, Kim, and Yang (2021):** Their qualitative statement measure can move bond prices even without a rate action. Our DGS3MO-controlled daily results preserve that identification goal but find no statistically reliable tone coefficient, suggesting daily noise and mixed document types dilute the signal.
- **Doh, Song, and Yang (2020/2025):** They place official statements between staff-written hawkish and dovish alternatives and emphasize tone, novelty, expectations, and short windows. Our public-anchor method avoids the alternatives' five-year publication delay, but it cannot identify surprises as cleanly.
- **Parsing the Fed (2021):** We reproduce its phrase-list and FinBERT/factor-similarity families. Like the presentation, the approaches disagree in places and have indicator-specific explanatory power. Our phrase list is explicitly monetary-policy contextual, while the balanced FinBERT anchors avoid equating generic positive financial sentiment with hawkishness.

Sources: [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), [Federal Reserve speeches](https://www.federalreserve.gov/newsevents/speech/2026-speeches.htm), [Doh, Kim, and Yang](https://www.kansascityfed.org/research/economic-review/how-you-say-it-matters-text-analysis-of-fomc-statements-using-natural-language-processing/), and [Doh, Song, and Yang](https://www.kansascityfed.org/research/research-working-papers/deciphering-federal-reserve-communication-via-text-analysis/).
"""),
    md("## Numerical checks"),
    code("""
import subprocess
r = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=ROOT, capture_output=True, text=True)
print(r.stdout)
assert r.returncode == 0, r.stderr
assert sum(forecast['rate_probabilities'].values()) == 100
assert len(scores[scores.chair.eq('Warsh')]) == 8
print('Forecast probabilities and Warsh sample checks passed.')
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
destination = ROOT / "assignment2.ipynb"
client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(notebook, destination)
print(f"Wrote executed notebook to {destination}")

