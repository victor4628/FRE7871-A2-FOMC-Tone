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
           P('Research question and result','H2x'),P('How did Federal Reserve communication change from Jerome Powell to Kevin Warsh, and are document scores associated with daily DXY, Treasury-curve, 1-year-yield, and growth-minus-value moves? Warsh-era policy language is more hawkish on average in the published dictionary and temporal RoBERTa reproduction, while FinBERT records more positive financial sentiment. The eight-document Warsh sample is descriptive. A leak-free pre-meeting model turns these scores into decision probabilities and improves on a previous-decision-only baseline.'),
           P(f"SEPTEMBER CALL: {forecast['rate_probabilities']['Cut']}% cut | {forecast['rate_probabilities']['Hold']}% hold | {forecast['rate_probabilities']['Hike']}% hike. Statement versus July 29: {forecast['more_hawkish_probability']}% more hawkish | {forecast['not_more_hawkish_probability']}% not more hawkish.",'Call'),
           P('Table 1. Documents collected, by type and Chair','H3x')]
    rows=[['Document type','Subtype','Powell','Warsh','Total']]
    for _,r in counts.iterrows():rows.append([r['doc_type'],r['subtype'],int(r.Powell),int(r.Warsh),int(r.Total)])
    story+=[tbl(rows,[1.65*inch,1.45*inch,.65*inch,.65*inch,.65*inch],align_right=(2,3,4)),Spacer(1,6),
            P('The raw inventory has 315 records. Five calendar-linked monetary-policy announcements that were not post-meeting statements are retained in the local audit data but excluded, leaving 310 scored documents. Chair communications comprise speeches, testimony, and press-conference transcripts. One February 2018 minutes release concerned Yellen\'s final meeting and is flagged separately in the data.','Smallx'),
            P('Measurement','H3x'),P('<b>Published dictionary.</b> The exact Shah, Paturi, and Chava (2023) rule classifies policy-topic sentences using topic and direction terms. The source implementation\'s substring matching, dovish-first conflict precedence, global negation reversal, and misspelling are preserved and disclosed. The level is (hawkish - dovish) / target sentences.'),
            P('<b>RoBERTa reproduction.</b> Because the authors\' gated checkpoint remained under review, RoBERTa-large was trained on their public 1996-2019 temporal file only. Its untouched 2020-2022 test weighted F1 is 0.753 (N=466), versus 0.711 reported in the paper. It maps 0=dovish, 1=hawkish, 2=neutral; it is not claimed to be the gated checkpoint.'),
            P('<b>FinBERT sentiment.</b> ProsusAI/finbert is applied sentence by sentence; the document score is mean P(positive) minus mean P(negative). This measures financial valence rather than monetary-policy stance, so it is retained as a diagnostic third measure and is not relabeled hawkish or dovish.'),
            P('Prepared remarks and Chair answers are scored in press conferences; reporter questions are excluded. Minutes begin at participants\' views or Committee policy discussion, excluding the staff forecast. Raw and selected-text hashes plus model revisions make each score auditable.'),PageBreak(),
            P('Tone over time','H2x'),Image(str(ROOT/'outputs/figure1_tone_trends_v2.png'),width=7.25*inch,height=4.75*inch),
            P('Figure 1. Each full-width panel is one measurement method; colors and markers distinguish statements, minutes, and Chair communications. Points are document levels and solid lines are five-release rolling means. The dashed line marks Warsh taking office on May 22, 2026. Positive means more hawkish in the first two panels and more positive financial sentiment in the third.','Smallx')]
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
            P('Interpretation','H3x'),P('At the pooled level, only FinBERT-DXY reaches 10% significance: a one-SD increase in financial positivity is associated with about a 0.044% dollar rise. Type-specific results add useful clues: more-hawkish minutes are associated with a flatter 10s2s curve, while more-positive statements are associated with higher 1-year yields. These are exploratory conditional daily associations, not pure causal wording shocks; all type-specific results and 2020 exclusions remain in the notebook.'),
            P('Comparison to earlier empirical results','H3x'),P('<i>Parsing the Fed</i> reports indicator-specific fit across factor similarity, a word list, and several FinBERT variants. The present design similarly finds no universal winner, but uses a published monetary-policy rule and leaves FinBERT as financial valence. Differences in sample, weighting, controls, and event windows prevent direct R2 ranking. The daily window here is also far wider than the intraday bond window in <i>How You Say It Matters</i>.'),PageBreak(),
            P('September 15-16, 2026 forecast','H2x'),P('The decision forecast uses no market-implied probability. A regularized three-class model combines the previous decision with the two policy-specific scores from the previous statement, latest minutes, and intermeeting Chair communications. Every historical feature was public before its meeting.')]
    rate=forecast['rate_probabilities'];story.append(tbl([['Decision','Cut','Hold','Hike'],['Probability',f"{rate['Cut']}%",f"{rate['Hold']}%",f"{rate['Hike']}%"]],[1.8*inch,1.0*inch,1.0*inch,1.0*inch],align_right=(1,2,3)))
    pm=forecast['premeeting_model']
    validation=[['Expanding-window model','Accuracy','Log loss','Brier'],
                ['Previous decision only',f"{pm['baseline_backtest']['accuracy']:.1%}",f"{pm['baseline_backtest']['log_loss']:.3f}",f"{pm['baseline_backtest']['brier']:.3f}"],
                ['Previous decision + pre-meeting tone',f"{pm['backtest']['accuracy']:.1%}",f"{pm['backtest']['log_loss']:.3f}",f"{pm['backtest']['brier']:.3f}"]]
    story+=[Spacer(1,6),tbl(validation,[2.45*inch,.8*inch,.8*inch,.8*inch],small=True,align_right=(1,2,3)),Spacer(1,7),P(f"Statement versus July 29: <b>{forecast['more_hawkish_probability']}% more hawkish</b> | <b>{forecast['not_more_hawkish_probability']}% not more hawkish</b>. The second category includes a less-hawkish or unchanged statement; the model does not estimate an absolute hawkish/dovish label. Probabilities weight historical tone-change frequencies after cuts, holds, and hikes by the internally predicted decision probabilities.",'Call'),P('Expected meeting-day market reaction','H3x')]
    rm=[['Indicator','Probability rise','Expected change']]
    for label,v in forecast['market_reaction'].items():rm.append([label.replace(' (bp)','').replace(' (pp)','').replace(' (%)',''),f"{v['probability_rise']}%",f"{v['expected_change']:+.2f} {v['unit']}"])
    story+=[tbl(rm,[2.2*inch,1.4*inch,1.4*inch],align_right=(1,2)),Spacer(1,7),
            P('Forecast construction and uncertainty','H3x'),P(f"The expanding-window test contains {pm['backtest']['n']} out-of-sample meetings after an initial 20-meeting training window. Adding text raises accuracy from {pm['baseline_backtest']['accuracy']:.1%} to {pm['backtest']['accuracy']:.1%} and lowers both probability-error measures. Expected text levels combine the July score with historical median changes under more- and less-hawkish outcomes. Each asset forecast averages the three statement regressions and uses their residual and model dispersion. DGS3MO is fixed at zero, so the table isolates the language-associated component."),
            P('The four market-direction probabilities remain close to 50%, and the expected changes are small relative to residual uncertainty. These are low-confidence model outputs for academic analysis only.','Call'),
            P('Personal recommendation','H3x'),P('In a hawkish hold scenario, language that keeps a near-term hike firmly in play could raise two-year yields more than ten-year yields and produce a bear flattening of the 10s2s curve. In a hike scenario, the curve could steepen after its initial front-end reaction. The hike may reduce near-term inflation concerns and the perceived need for additional tightening, while persistent fiscal-deficit concerns and a higher term premium continue to place upward pressure on long-term Treasury yields. A wider U.S.-Japan interest-rate differential could also increase the probability that the Japanese authorities intervene by buying yen and selling dollars. Treasury\'s expanded long-end buybacks may partially offset the rise in long-term yields, but they are not assumed to be the dominant force.'),PageBreak(),
            P('Relation to the required readings','H2x'),P('Doh, Kim, and Yang (2021)','H3x'),P('Their alternative-statement design shows that qualitative descriptions can affect bond prices apart from the rate action. This project follows that separation with a DGS3MO control, but daily rather than intraday data weaken identification. Their delayed staff alternatives also cannot supply a real-time September 2026 score.'),
            P('Doh, Song, and Yang (2020; revised 2023)','H3x'),P('Their semantic approach places the official statement between staff-written hawkish and dovish alternatives and separates tone, novelty, and market expectations. This project uses a public rule and an FOMC-specific classifier available at the forecast date. It reports levels and changes separately and does not call either one a surprise.'),
            P('Parsing the Fed (2021)','H3x'),P('The presentation compares factor similarity, a hand-built word list, and FinBERT, finding method- and asset-dependent explanatory power. This project keeps that comparative structure but does not reconstruct an unpublished list. ProsusAI/finbert uses sentence probabilities, while the two stance measures come from the published ACL rule and FOMC annotations.'),
            P('Limitations','H3x'),P('Warsh has only eight releases. The daily window cannot separate same-day statements from press conferences and contains other news. Release times missing from official sources are excluded rather than guessed. The IWF-IWN outcome contains a size tilt. The RoBERTa reproduction may differ from the gated checkpoint. The decision model has only 68 usable historical rows; its time-ordered improvement is encouraging but not an investment-grade validation. The separate statement-tone probability did not improve on an expanding historical-frequency benchmark, and the asset probabilities remain low-confidence conditional estimates.'),
            P('References and sources','H3x'),P('Doh, T., Kim, S., and Yang, S. (2021), <i>How You Say It Matters</i>, Federal Reserve Bank of Kansas City. Doh, T., Song, D., and Yang, S. (2020; revised 2023), <i>Deciphering Federal Reserve Communication via Text Analysis of Alternative FOMC Statements</i>, RWP 20-14. <i>Parsing the Fed</i> (2021), course presentation. Shah, A., Paturi, S., and Chava, S. (2023), <i>Trillion Dollar Words</i>, ACL, pp. 6664-6679.','Smallx'),
            P('Data: Federal Reserve document and event calendars; FRED T10Y2Y, DGS1, and DGS3MO; Yahoo Finance DX-Y.NYB, IWF, and IWN. Models: a disclosed RoBERTa-large temporal reproduction, ProsusAI/finbert, and an internally validated pre-meeting multinomial logit. Scenario context: U.S. Treasury long-end buyback announcement (August 19, 2026); Bank of Japan policy materials; and Japan Ministry of Finance statements on yen-buying intervention. No market-implied decision probability is used. Full links and provenance appear in README.md and METHODOLOGY.md.','Smallx')]
    doc.build(story,onFirstPage=footer,onLaterPages=footer);return DEST
if __name__=='__main__':print(build())
