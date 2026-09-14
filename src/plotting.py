"""Publication-quality tone-level chart with separate metric interpretations."""
from __future__ import annotations
import os
from .config import INTERIM_DIR, OUTPUT_DIR, WARSH_START
os.environ.setdefault('MPLCONFIGDIR',str(INTERIM_DIR/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

GREEN='#254C45';ORANGE='#A65F1A';BLUE='#426B8A';GRID='#DEE7E4'

def plot_tone_trends(documents: pd.DataFrame):
    frame=documents.copy();frame['release_date']=pd.to_datetime(frame.release_date)
    rows=['Statement','Minutes','Chair communication']
    metrics=[('wordlist_score','Published dictionary\n(+ hawkish)'),
             ('roberta_score','RoBERTa reproduction\n(+ hawkish)'),
             ('finbert_sentiment','FinBERT sentiment\n(+ positive)')]
    missing=[m for m,_ in metrics if m not in frame]
    if missing:raise ValueError(f'Cannot create final Figure 1; missing {missing}')
    fig,axes=plt.subplots(3,3,figsize=(11,8.1),sharex=True,constrained_layout=True)
    for i,doc_type in enumerate(rows):
        group=frame[frame.doc_type==doc_type].sort_values('release_date')
        for j,(metric,title) in enumerate(metrics):
            ax=axes[i,j]; color=[GREEN,BLUE,ORANGE][j]
            ax.scatter(group.release_date,group[metric],s=11,alpha=.28,color=color,zorder=2)
            smooth=group.set_index('release_date')[metric].rolling(5,min_periods=2).mean()
            ax.plot(smooth.index,smooth,lw=1.8,color=color,zorder=3)
            ax.axhline(0,color='#777',lw=.7);ax.axvline(pd.Timestamp(WARSH_START),color='#555',ls='--',lw=1)
            ax.grid(axis='y',color=GRID,lw=.6);ax.spines[['top','right']].set_visible(False)
            if i==0:ax.set_title(title,fontsize=10,fontweight='bold')
            if j==0:ax.set_ylabel(doc_type,fontsize=9,fontweight='bold')
            if i==2:ax.set_xlabel('Release date')
    axes[0,0].text(pd.Timestamp(WARSH_START),axes[0,0].get_ylim()[1],' Warsh starts',rotation=90,
                   va='top',ha='right',fontsize=7,color='#555')
    fig.suptitle('Figure 1. Communication scores by release type',fontsize=14,fontweight='bold',color=GREEN)
    dest=OUTPUT_DIR/'figure1_tone_trends_v2.png'
    fig.savefig(dest,dpi=240,bbox_inches='tight',facecolor='white');plt.close(fig)
    return dest
