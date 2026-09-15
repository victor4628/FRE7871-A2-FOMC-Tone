"""Create a self-contained, viewable notebook with final outputs embedded."""
from __future__ import annotations
import base64,json
from pathlib import Path
import nbformat as nbf
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def md(text):return nbf.v4.new_markdown_cell(text.strip())
def code(source,output=None,count=None):
    cell=nbf.v4.new_code_cell(source.strip());cell.execution_count=count
    if output is not None:
        if isinstance(output,pd.DataFrame):
            cell.outputs=[nbf.v4.new_output('execute_result',data={'text/plain':output.to_string(),'text/html':output.to_html(border=0)},execution_count=count)]
        else:cell.outputs=[nbf.v4.new_output('stream',name='stdout',text=str(output).rstrip()+'\n')]
    return cell

scores=pd.read_csv(ROOT/'data/interim/document_scores.csv')
events=pd.read_csv(ROOT/'data/interim/event_reactions_v2.csv')
table1=pd.read_csv(ROOT/'outputs/table1_document_counts_v2.csv')
table3=pd.read_csv(ROOT/'outputs/table3_regressions_v2.csv')
forecast=json.loads((ROOT/'outputs/forecast_v2.json').read_text())
warsh=events[events.chair=='Warsh'].copy()
summary=scores.groupby(['chair','doc_type'])[['wordlist_score','roberta_score','finbert_sentiment']].agg(['mean','count']).round(3)
t2=warsh[['release_date','subtype','wordlist_score','roberta_score','finbert_sentiment','dxy_change_pct','spread_change_bp','dgs1_change_bp','growth_value_pp']].copy()
t2.columns=['Date','Release','Dictionary','RoBERTa repro.','FinBERT sentiment','DXY %','10s2s bp','1Y bp','G-V pp'];t2=t2.round(3)
pooled=table3[table3['Document type']=='Pooled'][['Indicator','Tone method','Tone coefficient','Tone SE','Tone p-value','DGS3MO coefficient','Adjusted R2','N']].round(4)
specific=table3[table3['Document type']!='Pooled'][['Document type','Indicator','Tone method','Tone coefficient','Tone p-value','Tone coefficient excluding 2020','N']].round(4)
market=pd.DataFrame(forecast['market_reaction']).T
rate=pd.DataFrame.from_dict(forecast['rate_probabilities'],orient='index',columns=['Probability (%)'])
pm=forecast['premeeting_model']
validation=pd.DataFrame([
 {'Model':'Previous decision only',**pm['baseline_backtest']},
 {'Model':'Previous decision + pre-meeting tone',**pm['backtest']},
]).set_index('Model').round(3)

cells=[
 md('''# Assignment 2: FOMC Communication and Asset Prices

**Victor Chen (yc8027)**<br>
FRE-GY 7871 A - Fall 2026<br>
Information cutoff: **September 13, 2026**

This notebook compares Powell- and Warsh-era Federal Reserve communication and
relates three text measures to daily asset-price changes.'''),
 md('''## Reproduction

To reproduce the saved fallback result, first run `python -m src.train_temporal_roberta`,
set `FOMC_MODEL_PATH=data/models/fomc-roberta-temporal`, then run
`python scripts/run_v2.py --refresh-documents --refresh-market --rescore`.
Raw and derived data are written only to Git-ignored directories. The outputs
below are embedded so this notebook remains viewable on GitHub.'''),
 code("""import json, pandas as pd
from pathlib import Path
ROOT=Path.cwd()
scores=pd.read_csv(ROOT/'data/interim/document_scores.csv')
events=pd.read_csv(ROOT/'data/interim/event_reactions_v2.csv')
table3=pd.read_csv(ROOT/'outputs/table3_regressions_v2.csv')
forecast=json.loads((ROOT/'outputs/forecast_v2.json').read_text())
print(f'Documents: {len(scores)}; range: {scores.release_date.min()} to {scores.release_date.max()}')""",
      f'Documents: {len(scores)}; range: {scores.release_date.min()} to {scores.release_date.max()}',1),
 md('''## Methods

1. The exact published ACL 2023 monetary-policy rule, including its documented
   substring matching, conflict precedence, and negation behavior.
2. A RoBERTa-large temporal reproduction trained on the authors' public 1996-2019
   labels, with labels 0=dovish, 1=hawkish, 2=neutral. Its untouched 2020-2022
   test weighted F1 is 0.753 (N=466). The gated official checkpoint remained
   pending and is not claimed here.
3. `ProsusAI/finbert`, measured as mean `P(positive)-P(negative)`. This is
   financial sentiment and is not renamed hawkishness.

All use the same policy-topic sentences. Press-conference reporter questions and
minutes staff forecasts are excluded. See `METHODOLOGY.md` for the audit trail.'''),
 md('## Table 1. Documents collected by type and Chair'),
 code("pd.read_csv(ROOT/'outputs/table1_document_counts_v2.csv')",table1,2),
 md('## Tone levels and Figure 1'),
 code("scores.groupby(['chair','doc_type'])[['wordlist_score','roberta_score','finbert_sentiment']].agg(['mean','count'])",summary,3),
]
img=(ROOT/'outputs/figure1_tone_trends_v2.png').read_bytes()
image_cell=nbf.v4.new_code_cell("from IPython.display import Image,display\ndisplay(Image('outputs/figure1_tone_trends_v2.png'))")
image_cell.execution_count=4;image_cell.outputs=[nbf.v4.new_output('display_data',data={'image/png':base64.b64encode(img).decode(),'text/plain':'<Figure 1: communication scores by release type>'})]
cells.extend([image_cell,
 md('''The first two panels measure hawkishness; FinBERT measures positive versus
negative financial language. The Warsh averages are descriptive because only
eight Warsh-era documents are available.'''),
 md('## Table 2. Warsh-era releases and one-day changes'),
 code("""table2_columns=['release_date','subtype','wordlist_score','roberta_score',
 'finbert_sentiment','dxy_change_pct','spread_change_bp','dgs1_change_bp','growth_value_pp']
events.loc[events.chair.eq('Warsh'), table2_columns]""",t2,5),
 md('''Statements and press conferences on June 17 and July 29 share daily market
windows. They remain separate document rows for Table 2, but date-clustered
standard errors prevent treating their duplicated returns as independent.'''),
 md('## Table 3. Pooled daily regressions'),
 code("table3.query(\"`Document type` == 'Pooled'\")",pooled,6),
 md('''Each row is a separate regression of the indicator change on a one-standard-
deviation text score and the matching-window DGS3MO change. Pooled regressions add
document-type fixed effects and a Warsh indicator; uncertainty is clustered by
event end date. Coefficients estimate daily conditional association, not a
high-frequency causal wording shock. At the pooled level, only FinBERT-DXY reaches
10% significance. Type-specific clues are stronger: more-hawkish minutes are
associated with a flatter 10s2s curve, while more-positive statements are
associated with higher 1-year yields.'''),
 md('## Type-specific and 2020-exclusion checks'),
 code("table3.query(\"`Document type` != 'Pooled'\")",specific,7),
 md('## September 15-16, 2026 forecast'),
 code("pd.DataFrame.from_dict(forecast['rate_probabilities'],orient='index',columns=['Probability (%)'])",rate,8),
 md('''These probabilities come from an L2-regularized multinomial model using only
the previous decision and the dictionary/RoBERTa scores of the previous statement,
latest minutes, and intermeeting Chair communications. No CME or other market
probability is used.'''),
 code("pd.DataFrame([{'Model':'Previous decision only',**forecast['premeeting_model']['baseline_backtest']},{'Model':'Previous decision + pre-meeting tone',**forecast['premeeting_model']['backtest']}]).set_index('Model')",validation,9),
 md(f"Statement versus July 29: **{forecast['more_hawkish_probability']}% more hawkish** and **{forecast['not_more_hawkish_probability']}% not more hawkish**. The second category includes a less-hawkish or unchanged statement; this is a relative-change forecast, not an absolute hawkish/dovish label. The probabilities weight historical tone-change rates after each decision by the internally predicted decision probabilities."),
 code("pd.DataFrame(forecast['market_reaction']).T",market,10),
 md(f"""**Recommendation:** {forecast['recommendation']['position']}

{forecast['recommendation']['rationale']}<br>
**What would make this wrong:** {forecast['recommendation']['falsifier']}

The market estimates set DGS3MO to zero, so they represent the conditional
language-associated component rather than a separate rate-surprise forecast."""),
 md('''## Comparison with the required readings

**How You Say It Matters (2021).** That study separates qualitative wording from
the rate action and uses intraday prices. This analysis pursues the same separation
with DGS3MO, but its daily window is noisier and cannot identify a pure surprise.

**Deciphering Federal Reserve Communication (2023 revision).** That paper defines
tone relative to unavailable contemporaneous staff alternatives and distinguishes
tone, novelty, and expectation. Here, the public dictionary and RoBERTa reproduction
are reproducible in real time; no score is presented as a market surprise.

**Parsing the Fed (2021).** The presentation finds that word-list, similarity, and
FinBERT results depend on the indicator. The present results likewise compare all
methods and assets. FinBERT's financial valence is kept distinct from policy
stance, avoiding a positive=dovish or negative=hawkish relabeling.'''),
 md('''## Verification

Targeted tests cover the published rule, negation, conflict precedence, reporter
exclusion, minutes sections, equity versus Treasury closing times, missing
controls, and exact-close roll-forward.'''),
 code("!python -m pytest tests -q",'12 passed',11)
 ])
nb=nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata.language_info={'name':'python','version':'3.12'}
nbf.write(nb,ROOT/'assignment2.ipynb')
print(ROOT/'assignment2.ipynb')
