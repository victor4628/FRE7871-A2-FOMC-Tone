"""Run the literature-led Assignment 2 specification."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/interim/matplotlib'))
from src.collect_fed import collect_fed_documents
from src.collect_market import collect_market_data
from src.scoring import score_documents
from src.event_study import attach_market_reactions,document_counts,regression_table
from src.plotting import plot_tone_trends
from src.forecast_v2 import forecast

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-documents',action='store_true');parser.add_argument('--refresh-market',action='store_true');parser.add_argument('--rescore',action='store_true');args=parser.parse_args()
    raw=collect_fed_documents(args.refresh_documents);market=collect_market_data(args.refresh_market)
    scores=score_documents(raw,args.rescore);events=attach_market_reactions(scores,market)
    counts=document_counts(scores);counts.to_csv(ROOT/'outputs/table1_document_counts_v2.csv')
    events[events.chair=='Warsh'].to_csv(ROOT/'outputs/table2_warsh_releases_v2.csv',index=False)
    regressions=regression_table(events);plot_tone_trends(scores);prediction=forecast(scores,events)
    print(counts);print(regressions);print(json.dumps(prediction,indent=2))
if __name__=='__main__':main()
