# AI use disclosure

**Tool used:** OpenAI Codex.

**Scope of use.** I used Codex extensively to read and compare the assignment and
the supplied literature, inspect the earlier repository, design the research
plan, revise the Federal Reserve and market-data collectors, implement text
selection and three scoring methods, write tests, run regressions and forecasts,
create the saved-output notebook, and draft and format the report.

**Human inputs and decisions.** I supplied the assignment and reference folder,
asked that existing published word lists be preferred to a newly invented list,
required a sensible use of FinBERT, reviewed the proposed methods, authorized the
implementation, and completed the Hugging Face login and gated-model request.
Because author approval remained pending, Codex trained the clearly disclosed
temporal RoBERTa reproduction from the authors' public pre-2020 labels. I remain
responsible for reviewing the results and submission.

**Verification and limitations.** The dictionary is traced to a fixed author-code
revision and its known quirks are preserved and disclosed. Model revisions and
label mappings are checked programmatically. Tests cover negation, rule conflict
precedence, speaker and minutes selection, market closing times, missing controls,
and exact-close roll-forward. The report discloses the short Warsh sample, shared
statement/press-conference daily windows, noncausal daily regressions, the distinct
meaning of FinBERT sentiment, and dependence of the forecast on market pricing.
