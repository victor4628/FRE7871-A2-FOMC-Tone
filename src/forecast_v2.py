"""Pre-meeting forecast using market baseline plus text-association estimates."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from .config import AS_OF_DATE,OUTPUT_DIR
from .event_study import OUTCOMES,TONES
from .premeeting import POLICY_SCORES, premeeting_forecast

def forecast(scores: pd.DataFrame,events: pd.DataFrame) -> dict:
    required=list(TONES)
    if any(x not in scores for x in required):raise ValueError('All three approved scores are required for final forecast')
    statements=scores[scores.subtype=='statement'].sort_values('release_date').copy()
    latest=statements.iloc[-1]
    premeeting=premeeting_forecast(scores,'2026-09-16')
    rate=premeeting['rate_probabilities']
    p_hawk=premeeting['raw_more_hawkish_probability']
    for tone in required:statements[tone+'_delta']=statements[tone].diff()
    z=[]
    for tone in POLICY_SCORES:z.append(statements[tone+'_delta'])
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
        critical=float(norm.ppf(.90))
        reaction[label]={'probability_rise':int(round(100*norm.cdf(point/sigma))),
                         'expected_change':point,'unit':'bp' if '(bp)' in label else 'pp' if '(pp)' in label else '%',
                         'prediction_sd':sigma,
                         'prediction_interval_80':[point-critical*sigma,point+critical*sigma]}
        method_predictions[label]=details
    p_hawk_percent=int(round(100*p_hawk))
    sign_probabilities=[value['probability_rise'] for value in reaction.values()]
    curve_interval=reaction['10s2s (bp)']['prediction_interval_80']
    out={'as_of':AS_OF_DATE,'meeting_date':'2026-09-16','rate_probabilities':rate,
         'rate_probability_source':'Internal pre-meeting tone model; no CME or other market-implied probability used',
         'premeeting_model':premeeting,
         'more_hawkish_probability':p_hawk_percent,
         'not_more_hawkish_probability':100-p_hawk_percent,
         'previous_statement':'2026-07-29',
         'expected_scores':expected,'market_reaction':reaction,'method_predictions':method_predictions,
         'forecast_control_assumption_bp':0,
         'recommendation':{
          'position':'Stay neutral and take no directional pre-meeting position.',
          'rationale':f'The four market-direction probabilities span only {min(sign_probabilities)}% to {max(sign_probabilities)}%, while the expected changes are small relative to residual uncertainty.',
          'falsifier':f'The no-trade recommendation would be wrong if the statement-day 10s2s change fell outside its model-implied 80% interval of {curve_interval[0]:+.1f} to {curve_interval[1]:+.1f} bp, revealing materially more event risk than forecast.'},
         'interpretation':'Daily conditional association, not a causal high-frequency surprise estimate.'}
    (OUTPUT_DIR/'forecast_v2.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
