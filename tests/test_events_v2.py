import numpy as np
import pandas as pd
from src.event_study import attach_market_reactions

def panel():
    dates=pd.to_datetime(['2026-09-10','2026-09-11','2026-09-14'])
    return pd.DataFrame({'dxy':[100,101,102],'iwf':[100,102,104],'iwn':[100,101,101],
        't10y2y':[.3,.4,.35],'dgs1':[4,4.1,4.2],'dgs3mo':[3.8,3.85,3.9]},index=dates)

def document(day,time):
    return pd.DataFrame([{'document_id':'test','release_date':day,'release_time_et':time}])

def test_bond_quote_before_equity_close():
    e=attach_market_reactions(document('2026-09-11','15:45'),panel(),save=False).iloc[0]
    assert e.dxy_change_pct_end=='2026-09-11'
    assert e.dgs1_change_bp_end=='2026-09-14'
    assert np.isclose(e.dgs1_change_bp,10)

def test_unknown_timestamp_preserved_as_missing():
    e=attach_market_reactions(document('2026-09-11',''),panel(),save=False).iloc[0]
    assert pd.isna(e.dxy_change_pct)
    assert e.dxy_change_pct_status=='unknown release time'

def test_close_exactly_at_release_uses_next_session():
    e=attach_market_reactions(document('2026-09-11','16:00'),panel(),save=False).iloc[0]
    assert e.dxy_change_pct_end=='2026-09-14'

def test_missing_control_not_forward_filled():
    p=panel();p.loc[pd.Timestamp('2026-09-11'),'dgs3mo']=np.nan
    e=attach_market_reactions(document('2026-09-11','14:00'),p,save=False).iloc[0]
    assert pd.notna(e.dxy_change_pct) and pd.isna(e.dxy_change_pct_control)
