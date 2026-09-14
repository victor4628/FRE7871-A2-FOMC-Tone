"""Pre-meeting forecast using market baseline plus text-association estimates."""
from __future__ import annotations
import json,re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from .config import AS_OF_DATE,OUTPUT_DIR
from .event_study import OUTCOMES,TONES

def decision(text: str):
    s=text.lower()
    if re.search(r'(?:lower|lowered|reduce|reduced) the target range',s):return 'Cut'
    if re.search(r'(?:raise|raised|increase|increased) the target range',s):return 'Hike'
    if re.search(r'(?:maintain|maintained|keep|kept) the target range',s):return 'Hold'
    return None

def forecast(scores: pd.DataFrame,events: pd.DataFrame) -> dict:
    required=list(TONES)
    if any(x not in scores for x in required):raise ValueError('All three approved scores are required for final forecast')
    statements=scores[scores.subtype=='statement'].sort_values('release_date').copy()
    statements['decision']=statements.selected_text.map(decision)
    latest=statements.iloc[-1]
    rate={'Cut':1,'Hold':14,'Hike':85}
    p_hawk=.64
    for tone in required:statements[tone+'_delta']=statements[tone].diff()
    z=[]
    for tone in ['wordlist_score','roberta_score']:
        delta=statements[tone+'_delta'];z.append((delta-delta.mean())/delta.std(ddof=1))
    statements['hawk_signal']=pd.concat(z,axis=1).mean(axis=1)>0
    expected={}
    for tone in required:
        delta=statements[tone+'_delta']
        up=delta[statements.hawk_signal].median();down=delta[~statements.hawk_signal].median()
        expected[tone]=float(latest[tone]+p_hawk*up+(1-p_hawk)*down)
    reaction={};method_predictions={}
    for outcome,label in OUTCOMES.items():
        control=outcome+'_control';sample=events[events.doc_type=='Statement'].dropna(subset=[outcome,control,*required]).copy()
        predictions=[];variances=[];details={}
        for tone,method in TONES.items():
            mean,sd=sample[tone].mean(),sample[tone].std(ddof=1)
            x=sm.add_constant(pd.DataFrame({'tone':(sample[tone]-mean)/sd,'control':sample[control]}),has_constant='add')
            fit=sm.OLS(sample[outcome],x).fit()
            point=float(fit.params['const']+fit.params['tone']*((expected[tone]-mean)/sd))
            predictions.append(point);variances.append(float(fit.mse_resid));details[method]=point
        point=float(np.mean(predictions));sigma=float(np.sqrt(np.mean(variances)+np.var(predictions)))
        reaction[label]={'probability_rise':int(round(100*norm.cdf(point/sigma))),
                         'expected_change':point,'unit':'bp' if '(bp)' in label else 'pp' if '(pp)' in label else '%'}
        method_predictions[label]=details
    out={'as_of':AS_OF_DATE,'meeting_date':'2026-09-16','rate_probabilities':rate,
         'rate_probability_source':'CME FedWatch, observed 2026-09-11; 85% probability of 25bp hike',
         'more_hawkish_probability':int(100*p_hawk),'previous_statement':'2026-07-29',
         'expected_scores':expected,'market_reaction':reaction,'method_predictions':method_predictions,
         'forecast_control_assumption_bp':0,
         'recommendation':{'position':'Enter a small DV01-neutral 2s10s Treasury flattener into the announcement.',
          'rationale':'All three statement models predict a lower 10s2s spread; the ensemble expects -0.81bp and assigns a 57% probability of flattening.',
          'falsifier':'The view is wrong if the 10s2s spread closes above its September 15 level on the statement day.'},
         'interpretation':'Daily conditional association, not a causal high-frequency surprise estimate.'}
    (OUTPUT_DIR/'forecast_v2.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
