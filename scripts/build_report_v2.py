"""Generate the final short Assignment 2 PDF from the version-2 outputs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT,TA_CENTER,TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/pdf/assignment2_report.pdf'
GREEN=colors.HexColor('#254C45');PALE=colors.HexColor('#EDF3F1');GRID=colors.HexColor('#C9D6D2')
DARK=colors.HexColor('#202524');GRAY=colors.HexColor('#606865')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=20,leading=23,textColor=GREEN,alignment=TA_LEFT,spaceAfter=8))
styles.add(ParagraphStyle(name='Sub',parent=styles['Normal'],fontSize=9.5,leading=13,textColor=GRAY,spaceAfter=14))
styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=13.5,leading=16,textColor=GREEN,spaceBefore=3,spaceAfter=7))
styles.add(ParagraphStyle(name='H3x',parent=styles['Heading3'],fontName='Helvetica-Bold',fontSize=10.2,leading=12.5,textColor=DARK,spaceBefore=5,spaceAfter=3))
styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontName='Helvetica',fontSize=8.8,leading=12,textColor=DARK,spaceAfter=6))
styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontName='Helvetica',fontSize=6.8,leading=8.5,textColor=GRAY,spaceAfter=4))
styles.add(ParagraphStyle(name='Call',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=10,leading=13,textColor=GREEN,backColor=PALE,borderPadding=8,leftIndent=6,rightIndent=6,spaceAfter=10))
styles.add(ParagraphStyle(name='Cell',parent=styles['BodyText'],fontName='Helvetica',fontSize=6.7,leading=8,textColor=DARK))
styles.add(ParagraphStyle(name='CellS',parent=styles['BodyText'],fontName='Helvetica',fontSize=5.8,leading=6.8,textColor=DARK))
styles.add(ParagraphStyle(name='Head',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=6.4,leading=7.5,textColor=colors.white))

def P(text,style='Bodyx'):return Paragraph(str(text),styles[style])
def fmt(value,digits=2):return 'NA' if pd.isna(value) else f'{float(value):.{digits}f}'
def star(p):return '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
def tbl(rows,widths,small=False,align_right=()):
    style=styles['CellS' if small else 'Cell'];data=[]
    for i,row in enumerate(rows):data.append([P(x,'Head' if i==0 else ('CellS' if small else 'Cell')) for x in row])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    ts=[('BACKGROUND',(0,0),(-1,0),GREEN),('GRID',(0,0),(-1,-1),.35,GRID),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
    for col in align_right:ts.append(('ALIGN',(col,1),(col,-1),'RIGHT'))
    for row in range(2,len(data),2):ts.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#F5F8F7')))
    t.setStyle(TableStyle(ts));return t
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(GREEN);canvas.setLineWidth(.6);canvas.line(doc.leftMargin,.46*inch,letter[0]-doc.rightMargin,.46*inch)
    canvas.setFont('Helvetica',7);canvas.setFillColor(GRAY);canvas.drawString(doc.leftMargin,.30*inch,'FRE-GY 7871 A | Assignment 2 | Victor Chen')
    canvas.drawRightString(letter[0]-doc.rightMargin,.30*inch,f'{doc.page}');canvas.restoreState()

def build():
    scores=pd.read_csv(ROOT/'data/interim/document_scores.csv');events=pd.read_csv(ROOT/'data/interim/event_reactions_v2.csv')
    counts=pd.read_csv(ROOT/'outputs/table1_document_counts_v2.csv');regs=pd.read_csv(ROOT/'outputs/table3_regressions_v2.csv')
    forecast=json.loads((ROOT/'outputs/forecast_v2.json').read_text())
    warsh=events[events.chair=='Warsh'].sort_values(['release_date','subtype'])
    doc=SimpleDocTemplate(str(DEST),pagesize=letter,rightMargin=.52*inch,leftMargin=.52*inch,topMargin=.46*inch,bottomMargin=.57*inch,title='FOMC Communication and Asset Prices',author='Victor Chen')
    story=[P('FOMC Communication and Asset Prices','Title2'),P('Assignment 2 | Victor Chen (yc8027) | September 13, 2026 information cutoff','Sub'),
           P('Research question and result','H2x'),P('How did Federal Reserve communication change from Jerome Powell to Kevin Warsh, and are document scores associated with daily DXY, Treasury-curve, 1-year-yield, and growth-minus-value moves? Warsh-era policy language is more hawkish on average in the published dictionary and temporal RoBERTa reproduction, while FinBERT records more positive financial sentiment. The eight-document Warsh sample is descriptive. Market associations vary by method and indicator; DGS3MO remains an important control.'),
           P(f"SEPTEMBER CALL: {forecast['rate_probabilities']['Cut']}% cut | {forecast['rate_probabilities']['Hold']}% hold | {forecast['rate_probabilities']['Hike']}% hike. Probability of a more hawkish statement: {forecast['more_hawkish_probability']}%.",'Call'),
           P('Table 1. Documents collected, by type and Chair','H3x')]
    rows=[['Document type','Subtype','Powell','Warsh','Total']]
    for _,r in counts.iterrows():rows.append([r['doc_type'],r['subtype'],int(r.Powell),int(r.Warsh),int(r.Total)])
    story+=[tbl(rows,[1.65*inch,1.45*inch,.65*inch,.65*inch,.65*inch],align_right=(2,3,4)),Spacer(1,6),
            P('The raw inventory has 315 records. Two calendar-linked announcements that were not meeting statements are retained in the local audit data but excluded, leaving 313 scored documents. Chair communications comprise speeches, testimony, and press-conference transcripts. One February 2018 minutes release concerned Yellen\'s final meeting and is flagged separately in the data.','Smallx'),
            P('Measurement','H3x'),P('<b>Published dictionary.</b> The exact Shah, Paturi, and Chava (2023) rule classifies policy-topic sentences using topic and direction terms. The source implementation\'s substring matching, dovish-first conflict precedence, global negation reversal, and misspelling are preserved and disclosed. The level is (hawkish - dovish) / target sentences.'),
            P('<b>RoBERTa reproduction.</b> Because the authors\' gated checkpoint remained under review, RoBERTa-large was trained on their public 1996-2019 temporal file only. Its untouched 2020-2022 test weighted F1 is 0.753 (N=466), versus 0.711 reported in the paper. It maps 0=dovish, 1=hawkish, 2=neutral; it is not claimed to be the gated checkpoint. <b>ProsusAI/finbert</b> uses mean P(positive)-P(negative); it is financial sentiment, not renamed policy stance.'),
            P('Prepared remarks and Chair answers are scored in press conferences; reporter questions are excluded. Minutes begin at participants\' views or Committee policy discussion, excluding the staff forecast. Raw and selected-text hashes plus model revisions make each score auditable.'),PageBreak(),
            P('Tone over time','H2x'),Image(str(ROOT/'outputs/figure1_tone_trends_v2.png'),width=7.25*inch,height=5.34*inch),
            P('Figure 1. Points are document levels; solid lines are five-release rolling means. The dashed line marks Warsh taking office on May 22, 2026. Positive means more hawkish in the first two columns and more positive financial sentiment in the third. The separate labels matter because good economic news can raise both positivity and the case for tightening.','Smallx')]
    means=scores.groupby(['chair','doc_type'])[['wordlist_score','roberta_score','finbert_sentiment']].mean().reset_index()
    mr=[['Chair','Type','Dictionary','RoBERTa repro.','FinBERT sentiment']]
    for _,r in means.iterrows():mr.append([r.chair,r.doc_type,fmt(r.wordlist_score,3),fmt(r.roberta_score,3),fmt(r.finbert_sentiment,3)])
    story+=[P('Powell-Warsh comparison','H3x'),tbl(mr,[.65*inch,1.55*inch,1.0*inch,1.15*inch,1.2*inch],align_right=(2,3,4)),Spacer(1,6),
            P('The two policy-specific measures provide the main Chair comparison. FinBERT is a diagnostic comparison: ACL 2023 reports weak hawkish/dovish accuracy when generic sentiment labels are directly relabeled, so this report does not make that conversion. Divergence among measures is evidence about construct uncertainty, not a reason to select a preferred result after observing markets.'),PageBreak(),
            P('Warsh-era releases and market reactions','H2x'),P('Table 2 reports every release under Warsh. Equity variables use the first NYSE close strictly after release. Treasury variables use the first H.15 observation after a 3:30 p.m. Eastern indicative quote, adjusted for SIFMA early closes. Missing values are retained; no rate or price is forward-filled.')]
    r2=[['Date','Release','Dict.','RoBERTa','FinBERT','DXY %','10s2s bp','1Y bp','G-V pp']]
    labels={'press_conference':'Press conf.','statement':'Statement','minutes':'Minutes','testimony':'Testimony','speech':'Speech'}
    for _,r in warsh.iterrows():r2.append([pd.Timestamp(r.release_date).strftime('%b %d'),labels[r.subtype],fmt(r.wordlist_score),fmt(r.roberta_score),fmt(r.finbert_sentiment),fmt(r.dxy_change_pct),fmt(r.spread_change_bp,1),fmt(r.dgs1_change_bp,1),fmt(r.growth_value_pp)])
    story+=[tbl(r2,[.52*inch,.68*inch,.55*inch,.62*inch,.62*inch,.55*inch,.65*inch,.55*inch,.55*inch],small=True,align_right=(2,3,4,5,6,7,8)),Spacer(1,6),
            P('The June 17 and July 29 statement/press-conference pairs share their respective daily outcomes. Both documents appear because the assignment requests each release; regression uncertainty is clustered by event end date so those duplicated market windows are not treated as independent. At daily frequency, their separate language effects cannot be identified.'),
            P('Market-variable caveats','H3x'),P('DXY is a percent return; T10Y2Y and DGS1 are basis-point changes; growth minus value is return(IWF)-return(IWN) in percentage points. IWF tracks Russell 1000 Growth and IWN tracks Russell 2000 Value, so the last measure mixes style with firm size. The DGS3MO control uses the matching Treasury window rather than an equity-window fill.'),PageBreak(),
            P('Table 3. Daily market-change regressions','H2x'),P('Each row is a separate pooled regression. Tone beta is the change associated with a one-standard-deviation score increase. Models control for matching-window DGS3MO, document-type fixed effects, and a Warsh indicator. Parentheses contain standard errors clustered by event end date; stars are 10%, 5%, and 1% significance.')]
    pooled=regs[regs['Document type']=='Pooled']
    r3=[['Indicator','Tone method','Tone beta (SE)','p','DGS3MO beta','Adj. R2','N','Dates']]
    for _,r in pooled.iterrows():r3.append([r.Indicator,r['Tone method'],f"{r['Tone coefficient']:.3f}{star(r['Tone p-value'])} ({r['Tone SE']:.3f})",f"{r['Tone p-value']:.3f}",f"{r['DGS3MO coefficient']:.3f}",f"{r['Adjusted R2']:.3f}",int(r.N),int(r['Unique dates'])])
    story+=[tbl(r3,[.9*inch,1.1*inch,1.08*inch,.45*inch,.82*inch,.58*inch,.35*inch,.42*inch],small=True,align_right=(2,3,4,5,6,7)),Spacer(1,6),
            P('Interpretation','H3x'),P('The table reports all twelve required method-by-indicator combinations, including weak and contrary estimates. Coefficients are conditional daily associations and should not be interpreted as pure causal wording shocks: the window contains other macro and market news, and the control is an imperfect proxy for the policy-rate surprise. Type-specific estimates and coefficients excluding 2020 are preserved in the notebook.'),
            P('Comparison to earlier empirical results','H3x'),P('<i>Parsing the Fed</i> reports indicator-specific fit across factor similarity, a word list, and several FinBERT variants. The present design similarly finds no universal winner, but uses a published monetary-policy rule and leaves FinBERT as financial valence. Differences in sample, weighting, controls, and event windows prevent direct R2 ranking. The daily window here is also far wider than the intraday bond window in <i>How You Say It Matters</i>.'),PageBreak(),
            P('September 15-16, 2026 forecast','H2x'),P('The rate probabilities use the September 11 CME FedWatch 85% hike reading as a transparent market prior. This is consistent with Warsh\'s August 28 focus on above-target inflation, August payroll growth of 162,000 with 4.1% unemployment, August CPI inflation of 3.4% year-over-year, and August PPI final demand inflation of 5.4% year-over-year.')]
    rate=forecast['rate_probabilities'];story.append(tbl([['Decision','Cut','Hold','Hike'],['Probability',f"{rate['Cut']}%",f"{rate['Hold']}%",f"{rate['Hike']}%"]],[1.8*inch,1.0*inch,1.0*inch,1.0*inch],align_right=(1,2,3)))
    story+=[Spacer(1,7),P(f"Statement tone: <b>{forecast['more_hawkish_probability']}%</b> probability that the September statement is more hawkish than July 29. July was already terse and hawkish, limiting the probability despite the incoming inflation data.",'Call'),P('Expected meeting-day market reaction','H3x')]
    rm=[['Indicator','Probability rise','Expected change']]
    for label,v in forecast['market_reaction'].items():rm.append([label.replace(' (bp)','').replace(' (pp)','').replace(' (%)',''),f"{v['probability_rise']}%",f"{v['expected_change']:+.2f} {v['unit']}"])
    story+=[tbl(rm,[2.2*inch,1.4*inch,1.4*inch],align_right=(1,2)),Spacer(1,7),
            P('Forecast construction and recommendation','H3x'),P('Expected text levels combine the July score with historical median statement changes under more- and less-hawkish policy-specific signals. For each asset, the forecast averages the three statement-regression predictions and uses residual variance plus between-method dispersion for the sign probability. DGS3MO is set to zero basis points; an already-priced 25bp hike is not mechanically treated as a 25bp surprise.'),
            P(f"<b>Position:</b> {forecast['recommendation']['position']} {forecast['recommendation']['rationale']} <b>Falsifier:</b> {forecast['recommendation']['falsifier']}",'Call'),
            P('The position is deliberately small: a heavily priced hike leaves the statement and press conference as the main sources of surprise, while the regressions are noisy. The forecast is an academic pre-meeting position, not investment advice.'),PageBreak(),
            P('Relation to the required readings','H2x'),P('Doh, Kim, and Yang (2021)','H3x'),P('Their alternative-statement design shows that qualitative descriptions can affect bond prices apart from the rate action. This project follows that separation with a DGS3MO control, but daily rather than intraday data weaken identification. Their delayed staff alternatives also cannot supply a real-time September 2026 score.'),
            P('Doh, Song, and Yang (2020; revised 2023)','H3x'),P('Their semantic approach places the official statement between staff-written hawkish and dovish alternatives and separates tone, novelty, and market expectations. This project uses a public rule and an FOMC-specific classifier available at the forecast date. It reports levels and changes separately and does not call either one a surprise.'),
            P('Parsing the Fed (2021)','H3x'),P('The presentation compares factor similarity, a hand-built word list, and FinBERT, finding method- and asset-dependent explanatory power. This project keeps that comparative structure but does not reconstruct an unpublished list. ProsusAI/finbert uses sentence probabilities, while the two stance measures come from the published ACL rule and FOMC annotations.'),
            P('Limitations','H3x'),P('Warsh has only eight releases. The daily window cannot separate same-day statements from press conferences and contains other news. Release times missing from official sources are excluded rather than guessed. The IWF-IWN outcome contains a size tilt. The RoBERTa reproduction may differ from the gated checkpoint, and the paper reports weaker performance on press conferences than on the combined corpus. The forecast relies on an 85% market prior and remains vulnerable to an unpriced decision.'),
            P('References and sources','H3x'),P('Doh, T., Kim, S., and Yang, S. (2021), <i>How You Say It Matters</i>, Federal Reserve Bank of Kansas City. Doh, T., Song, D., and Yang, S. (2020; revised 2023), <i>Deciphering Federal Reserve Communication via Text Analysis of Alternative FOMC Statements</i>, RWP 20-14. <i>Parsing the Fed</i> (2021), course presentation. Shah, A., Paturi, S., and Chava, S. (2023), <i>Trillion Dollar Words</i>, ACL, pp. 6664-6679.','Smallx'),
            P('Data: Federal Reserve document and event calendars; FRED T10Y2Y, DGS1, and DGS3MO; Yahoo Finance DX-Y.NYB, IWF, and IWN; BLS August 2026 Employment Situation, CPI, and PPI; CME FedWatch. Models: a disclosed RoBERTa-large temporal reproduction and ProsusAI/finbert. Full links and provenance appear in README.md and METHODOLOGY.md.','Smallx')]
    doc.build(story,onFirstPage=footer,onLaterPages=footer);return DEST
if __name__=='__main__':print(build())
