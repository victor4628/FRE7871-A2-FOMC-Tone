"""Daily associations with per-asset observation windows and clustered uncertainty."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas_market_calendars as calendars
from .config import INTERIM_DIR, OUTPUT_DIR

OUTCOMES = {'dxy_change_pct':'DXY (%)', 'spread_change_bp':'10s2s (bp)',
            'dgs1_change_bp':'1Y yield (bp)', 'growth_value_pp':'Growth - value (pp)'}
TONES = {'wordlist_score':'Published dictionary','roberta_score':'RoBERTa reproduction','finbert_sentiment':'FinBERT sentiment'}

def observation_closes(market: pd.DataFrame) -> dict:
    start,end=market.index.min(),market.index.max()
    equity=calendars.get_calendar('NYSE').schedule(start,end)
    bond=calendars.get_calendar('SIFMA_US').schedule(start,end)
    result={}
    for kind,schedule in [('equity',equity),('bond',bond)]:
        for date,row in schedule.iterrows():
            close=row.market_close.tz_convert('America/New_York')
            if kind=='bond':
                # H.15 Treasury indicative quotes are taken around 3:30 p.m. ET.
                close=min(close,pd.Timestamp(str(date.date())+' 15:30',tz='America/New_York'))
            result[(kind,date)]=close
    return result

def attach_market_reactions(documents: pd.DataFrame, market: pd.DataFrame, save=True) -> pd.DataFrame:
    market=market.sort_index();closes=observation_closes(market);records=[]
    specs={'dxy_change_pct':(['dxy'],'equity'), 'growth_value_pp':(['iwf','iwn'],'equity'),
           'spread_change_bp':(['t10y2y'],'bond'), 'dgs1_change_bp':(['dgs1'],'bond')}
    for _,doc in documents.iterrows():
        row=doc.to_dict(); valid_time=isinstance(doc.release_time_et,str) and len(doc.release_time_et)==5
        stamp=pd.Timestamp(doc.release_date+' '+doc.release_time_et,tz='America/New_York') if valid_time else None
        for outcome,(columns,kind) in specs.items():
            row[outcome]=np.nan;row[outcome+'_control']=np.nan
            row[outcome+'_start']='';row[outcome+'_end']=''
            row[outcome+'_status']='unknown release time' if stamp is None else 'no observed closing window'
            if stamp is None:continue
            dates=market.dropna(subset=columns).index
            eligible=[date for date in dates if (kind,date) in closes and closes[(kind,date)]>stamp]
            if not eligible:continue
            end=eligible[0];position=dates.get_loc(end)
            if position==0:continue
            start=dates[position-1];before,after=market.loc[start],market.loc[end]
            # A release on a trading day with no source observation is retained as missing.
            expected=[d for (k,d),c in closes.items() if k==kind and c>stamp]
            if expected and end>min(expected):
                row[outcome+'_status']='source observation missing for expected session';continue
            row[outcome+'_start']=str(start.date());row[outcome+'_end']=str(end.date())
            if outcome=='growth_value_pp':value=100*(after.iwf/before.iwf-after.iwn/before.iwn)
            elif outcome=='dxy_change_pct':value=100*(after.dxy/before.dxy-1)
            else:value=100*(after[columns[0]]-before[columns[0]])
            row[outcome]=value
            row[outcome+'_control']=100*(after.dgs3mo-before.dgs3mo)
            row[outcome+'_status']='ok' if pd.notna(row[outcome+'_control']) else 'asset observed; control missing'
        records.append(row)
    result=pd.DataFrame(records)
    for outcome in OUTCOMES:
        result[outcome+'_same_window_n']=result.groupby(outcome+'_end')[outcome+'_end'].transform('size')
    if save:result.to_csv(INTERIM_DIR/'event_reactions_v2.csv',index=False)
    return result

def document_counts(documents: pd.DataFrame) -> pd.DataFrame:
    table=documents.pivot_table(index=['doc_type','subtype'],columns='chair',values='document_id',aggfunc='count',fill_value=0)
    table=table.reindex(columns=['Powell','Warsh'],fill_value=0).astype(int)
    table['Total']=table.sum(axis=1)
    return table

def regression_table(events: pd.DataFrame, allow_partial=False) -> pd.DataFrame:
    tones={k:v for k,v in TONES.items() if k in events}
    events=events.copy()
    for tone in tones: events[tone]=pd.to_numeric(events[tone],errors='coerce')
    if len(tones)!=3 and not allow_partial:raise ValueError('All three approved scores are required')
    rows=[]
    for group in ['Statement','Minutes','Chair communication','Pooled']:
        candidates=events if group=='Pooled' else events[events.doc_type==group]
        for outcome,label in OUTCOMES.items():
            control=outcome+'_control';end=outcome+'_end'
            common=candidates.dropna(subset=[outcome,control,*tones]).copy()
            if len(common)<12:continue
            for tone,method in tones.items():
                sd=common[tone].std(ddof=1)
                if not sd>0:continue
                x=pd.DataFrame({'tone':(common[tone]-common[tone].mean())/sd,'control':common[control]})
                if group=='Pooled':
                    x=x.join(pd.get_dummies(common.doc_type,prefix='type',drop_first=True,dtype=float))
                    x['warsh']=(common.chair=='Warsh').astype(float)
                x=sm.add_constant(x,has_constant='add').astype(float)
                fit=sm.OLS(common[outcome],x).fit(cov_type='cluster',cov_kwds={'groups':common[end],'use_correction':True},use_t=True)
                no_crisis=common.release_date.str[:4]!='2020'
                sensitivity=sm.OLS(common.loc[no_crisis,outcome],x.loc[no_crisis]).fit()
                rows.append({'Document type':group,'Indicator':label,'Tone method':method,
                    'Tone coefficient':float(fit.params['tone']),'Tone SE':float(fit.bse['tone']),
                    'Tone p-value':float(fit.pvalues['tone']),'DGS3MO coefficient':float(fit.params['control']),
                    'Adjusted R2':float(fit.rsquared_adj),'R2':float(fit.rsquared),'N':int(fit.nobs),
                    'Unique dates':int(common[end].nunique()),'Score SD':float(sd),
                    'Tone coefficient excluding 2020':float(sensitivity.params['tone'])})
    result=pd.DataFrame(rows)
    suffix='_partial' if len(tones)!=3 else ''
    result.to_csv(OUTPUT_DIR/f'table3_regressions_v2{suffix}.csv',index=False)
    return result
