# Methodology and analysis choices

## Sample

The information set ends September 13, 2026. The sample begins when Jerome Powell
took office on February 5, 2018. Kevin Warsh's period begins on May 22, 2026, the
date he took the oath as Board and FOMC Chair. Committee statements and minutes are
assigned to the Chair in office on the release date. Chair communications include
speeches, testimony, and FOMC press-conference transcripts.

## Release times and event windows

Exact Eastern times are matched to Federal Reserve release lines or monthly event
calendars. FOMC statements use 2:00 p.m. and press conferences 2:30 p.m. An event
at or before 4:00 p.m. uses prior close to same-day close; an after-close or
non-trading-day event uses the next available close. DXY and ETF returns are
percent changes. Treasury-yield and 10s2s changes are basis points. Growth minus
value is the IWF return minus the IWN return in percentage points.

The Treasury data come from the Federal Reserve Board's official H.15 package,
which underlies the requested FRED series. The 10s2s spread is reconstructed as
the 10-year constant-maturity yield minus the 2-year yield and matches T10Y2Y by
definition. DXY, IWF, and IWN adjusted closes come from Yahoo Finance.

## Tone methods

The phrase-list score is 1,000 times hawkish phrase matches minus dovish phrase
matches, divided by word count. Patterns encode context and direction. The FinBERT
score embeds up to 20 evenly spaced 80-word document chunks with
`ProsusAI/finbert`, then subtracts similarity to six dovish anchors from similarity
to six hawkish anchors. Scores are standardized by document type against the
Powell-period mean and standard deviation and clipped at four standard deviations.

## Regressions and forecast

There are eight OLS specifications: four outcomes times two tone measures. Each
includes the same-day DGS3MO change and document-type fixed effects. HC3 standard
errors address heteroskedasticity.

The rate model is a regularized multinomial logit estimated on 68 prior meetings.
Predictors are the previous statement tone, mean interim communication tone, and
the preceding 20-trading-day changes in DGS1 and 10s2s. Probabilities are blended
80/20 with Laplace-smoothed historical frequencies. A binary logit with the same
features estimates whether the statement becomes more hawkish. Market forecasts
average the two tone-regression predictions and use residual variance for sign
probabilities. These are model-based academic probabilities, not market-implied
odds.

