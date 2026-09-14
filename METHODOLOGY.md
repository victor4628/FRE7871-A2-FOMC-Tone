# Methodology and audit trail

## Sample and text selection

The information set ends September 13, 2026. The release sample begins February
5, 2018, when Jerome Powell became Chair; Kevin Warsh's release period begins May
22, 2026. Statements and minutes are assigned to the Chair in office on the
release date. `meeting_chair` separately records that the minutes released on
February 21, 2018 concerned Janet Yellen's final meeting.

The raw inventory contains 315 records. Two nonmeeting press releases captured by
the broad calendar link pattern - the March 31, 2020 FIMA facility announcement
and the August 27, 2020 strategy-review announcement - are retained locally but
excluded from the statement sample, leaving 313 documents.

Web navigation, voting lists, contact text, PDF page furniture, references, and
footnotes are excluded. Press-conference scoring includes Chair prepared remarks
and Chair answers; reporter questions remain outside the score. Minutes start at
participants' economic views or the Committee policy discussion, excluding the
staff forecast and market desk narrative. Every selected text has a SHA-256 hash
and a preprocessing version.

## Three text measures

The common comparison sample is sentences containing one of the policy-topic
terms in the published ACL rule. This makes denominators comparable and reports
coverage. Documents without a target sentence are missing rather than neutral.

**Published dictionary.** The implementation faithfully transcribes
`code_model/rule_based.py` from Shah, Paturi, and Chava's repository at revision
`98646987452b326507479cf641571b33814bb73f`. It preserves the authors' substring
matching, dovish-first conflict rule, global sentence negation reversal, and the
published misspelling `decelarate`. The level is hawkish minus dovish target
sentences divided by all target sentences. These details are limitations of the
published benchmark, not silent fixes.

**RoBERTa reproduction.** The authors' official FOMC-RoBERTa checkpoint was still
awaiting gated-access review when results were frozen. The saved analysis therefore
uses RoBERTa-large trained only on the authors' public 1996-2019 temporal training
file, with labels 0=dovish, 1=hawkish, and 2=neutral. A stratified 15% validation
split, seed 5768, 256-token limit, learning rate 1e-5, and validation-loss early
stopping were used. Weighted F1 on the untouched public 2020-2022 test file is
0.753 (N=466), versus 0.711 reported for the paper's temporal experiment. This is
not represented as the gated checkpoint. The document level is the number of
argmax hawkish predictions minus dovish predictions divided by target sentences;
mean `P(hawkish)-P(dovish)` is retained as an audit field.

**ProsusAI/finbert.** The pinned checkpoint revision is
`4556d13015211d73dccd3fdd39d39232506f3e43`. The main financial-valence measure is
the sentence mean of `P(positive)-P(negative)`; mean positive probability and a
full-prose robustness score are also retained. Positive is not renamed dovish or
hawkish. Long sentences are split into nonoverlapping model-length token chunks,
chunk probabilities are weighted by tokens, and sentences receive equal weight.

## Market data and event windows

DXY, IWF, and IWN use Yahoo adjusted closes. T10Y2Y, DGS1, and DGS3MO are read
directly from one FRED CSV export. Missing values are never forward-filled.

For DXY and the ETFs, the event observation is the first NYSE close strictly
after the release. For Treasury measures and the DGS3MO control, the daily H.15
observation is treated as a 3:30 p.m. Eastern indicative quote, capped by the
SIFMA bond-market close on early-close days. Thus a release after 3:30 p.m. can
use a later Treasury window than its equity window. Unknown release times and
missing source observations are retained with an exclusion reason.

Outcomes are DXY percentage return, T10Y2Y change in basis points, DGS1 change in
basis points, and `return(IWF)-return(IWN)` in percentage points. IWF is
large/mid-cap growth while IWN is small-cap value, so the final outcome includes a
size exposure.

## Regressions

For each document group, outcome, and text measure:

`Asset change = intercept + beta * standardized score + gamma * DGS3MO change + error`

The standard deviation is calculated on the common complete-case sample for all
three scores. Standard errors are clustered by the outcome-specific event end
date because statements and press conferences can share a daily return. Table 3
shows the pooled twelve method-by-indicator results with document-type fixed
effects and a Warsh indicator. Type-specific results and coefficients excluding
2020 are saved in the notebook. The design estimates conditional daily
association; it cannot isolate a causal wording surprise from concurrent news.

## Forecast

The rate-decision distribution starts from the September 11 CME FedWatch reading
of an 85% probability of a 25bp hike, with 14% hold and a 1% tail probability of
a cut. The 64% probability of a more hawkish statement incorporates the hot
August CPI/PPI data, resilient August employment, Warsh's August 28 price-stability
emphasis, and the July statement reference point.

Expected future text levels use historical median statement changes conditional
on the two policy-specific measures moving more or less hawkishly. For each market
indicator, the forecast averages the three statement-regression predictions and
uses residual variance plus between-method dispersion for the sign probability.
The DGS3MO forecast input is zero basis points: the already-priced policy decision
is not mechanically converted into a bill-yield surprise. This means the market
forecast is mainly the estimated text-associated component and remains sensitive
to an unexpected decision.
