"""Generate the short Assignment 2 report as a polished PDF."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "assignment2_report.pdf"
BRAND_GREEN = colors.HexColor("#254C45")
PALE_GREEN = colors.HexColor("#EDF3F1")
LIGHT_GRAY = colors.HexColor("#F4F6F5")
MID_GRAY = colors.HexColor("#666666")
DARK = colors.HexColor("#202124")
TABLE_GRID = colors.HexColor("#CBD6D3")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=21, leading=24, textColor=BRAND_GREEN, alignment=TA_LEFT, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=10.5, leading=14, textColor=MID_GRAY, spaceAfter=16,
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=14, leading=17, textColor=BRAND_GREEN, spaceBefore=4, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Subsection", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=11, leading=14, textColor=DARK, spaceBefore=6, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Body2", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.2, leading=12.8, textColor=DARK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.4, leading=9.5, textColor=MID_GRAY, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10.5, leading=14, textColor=BRAND_GREEN, leftIndent=10, rightIndent=10,
    borderPadding=10, backColor=PALE_GREEN, spaceBefore=4, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="Cell", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.2, leading=8.7, textColor=DARK,
))
styles.add(ParagraphStyle(
    name="CellSmall", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=6.3, leading=7.5, textColor=DARK,
))
styles.add(ParagraphStyle(
    name="HeaderCell", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=7.2, leading=8.7, textColor=colors.white,
))
styles.add(ParagraphStyle(
    name="HeaderCellSmall", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=6.3, leading=7.5, textColor=colors.white,
))


def P(text: str, style: str = "Body2") -> Paragraph:
    return Paragraph(text, styles[style])


def cell(value, small: bool = False, header: bool = False) -> Paragraph:
    if header:
        return P(str(value), "HeaderCellSmall" if small else "HeaderCell")
    return P(str(value), "CellSmall" if small else "Cell")


def styled_table(data, widths, header_rows=1, small=False, alignments=None):
    wrapped = [
        [cell(value, small=small, header=row_index < header_rows) for value in row]
        for row_index, row in enumerate(data)
    ]
    table = Table(wrapped, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), BRAND_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, TABLE_GRID),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, PALE_GREEN]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if alignments:
        for column, alignment in enumerate(alignments):
            commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def header_footer(canvas, document):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D3DDD9"))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, height - 0.38 * inch, width - document.rightMargin, height - 0.38 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(document.leftMargin, height - 0.29 * inch, "FRE-GY 7871 A | Assignment 2 | Victor Chen (yc8027)")
    canvas.drawRightString(width - document.rightMargin, 0.3 * inch, f"Page {document.page}")
    canvas.restoreState()


def significance(value: float) -> str:
    if value < 0.01:
        return "***"
    if value < 0.05:
        return "**"
    if value < 0.10:
        return "*"
    return ""


def build() -> Path:
    counts = pd.read_csv(ROOT / "outputs" / "table1_document_counts.csv", index_col=0)
    warsh = pd.read_csv(ROOT / "outputs" / "table2_warsh_releases.csv")
    regressions = pd.read_csv(ROOT / "outputs" / "table3_regressions.csv")
    scores = pd.read_csv(ROOT / "data" / "interim" / "document_scores.csv")
    forecast = json.loads((ROOT / "outputs" / "forecast.json").read_text(encoding="utf-8"))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch,
        topMargin=0.55 * inch, bottomMargin=0.52 * inch,
        title="Assignment 2: Evaluating the Impact of FOMC Communications on Asset Prices",
        author="Victor Chen (yc8027)",
    )
    story = []

    # Page 1
    story.extend([
        Spacer(1, 0.16 * inch),
        P("Evaluating the Impact of FOMC Communications on Asset Prices", "ReportTitle"),
        P("Victor Chen (yc8027) | FRE-GY 7871 A | Fall 2026 | Information set: September 13, 2026", "Subtitle"),
        P("Executive summary", "Section"),
        P(
            "Kevin Warsh's first two FOMC statements are more hawkish than the Powell-period statement baseline by "
            "1.18 standard deviations (SD) under the contextual phrase list and 0.47 SD under FinBERT anchors. "
            "Warsh-era Chair communications are also more hawkish on average (0.26 and 0.55 SD), while the two "
            "minutes are near the Powell baseline. The models disagree on some individual releases, especially the "
            "July statement and August minutes, so the forecast retains both signals.",
        ),
        P(
            "Across 315 events, no tone coefficient is significant at the 10 percent level after controlling for the "
            "same-day change in the 3-month Treasury yield and document type. The control dominates DXY and Treasury "
            "regressions. This is evidence of weak daily-window identification, not evidence that language never matters.",
        ),
        P(
            f"SEPTEMBER CALL: {forecast['rate_probabilities']['Cut']}% cut | "
            f"{forecast['rate_probabilities']['Hold']}% hold | {forecast['rate_probabilities']['Hike']}% hike. "
            f"The probability of a more hawkish statement is {forecast['more_hawkish_probability']}%.",
            "Callout",
        ),
        P("Table 1. Documents collected, by type and Chair", "Subsection"),
    ])
    table1_data = [["Document type", "Powell", "Warsh", "Total"]]
    for label, row in counts.iterrows():
        table1_data.append([label, int(row.Powell), int(row.Warsh), int(row.Total)])
    story.append(styled_table(table1_data, [3.5 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch], alignments=["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    story.extend([
        Spacer(1, 6),
        P(
            "The five displayed rows map to the assignment's three categories: statements, minutes, and Chair "
            "communications (speeches, testimony, and press conferences). No document is double-counted. The "
            "information set ends before the September 15-16 meeting.",
            "Small",
        ),
        P("Question and contribution", "Subsection"),
        P(
            "The empirical question is whether communication tone shifted with the Chair and whether that tone "
            "predicts one-day moves in DXY, the 10s2s Treasury spread, the 1-year Treasury yield, and Russell growth "
            "minus value. The design extends course methods to all three document types while explicitly recording "
            "release time and separating language from the contemporaneous short-rate decision.",
        ),
        PageBreak(),
    ])

    # Page 2
    story.extend([
        P("Tone over time", "Section"),
        Image(str(ROOT / "outputs" / "figure1_tone_trends.png"), width=7.25 * inch, height=5.70 * inch),
        P(
            "Figure 1. Points are document-level scores; lines are centered four-month averages of monthly means. "
            "Scores are standardized within document type using Powell-period means and SDs. Positive means more "
            "hawkish. The vertical line marks Warsh's first day as Chair, May 22, 2026.",
            "Small",
        ),
        P(
            "Statements show the clearest post-transition increase. The phrase list flags both Warsh statements as "
            "hawkish, while FinBERT treats June as hawkish and July as slightly dovish relative to Powell-era "
            "statements. The August minutes reverse the June minutes in both methods. Chair-level averages are "
            "positive, driven by the July testimony and August Jackson Hole speech rather than the long press "
            "conference transcripts.",
        ),
        P(
            "This disagreement is informative. Phrase counts are transparent and policy-specific but sparse; "
            "FinBERT captures semantic context but remains sensitive to anchors and chunk selection. Accordingly, "
            "all regressions and forecasts are estimated separately by method and combined only at the final "
            "prediction step.",
        ),
        PageBreak(),
    ])

    # Page 3
    story.extend([
        P("Warsh-era releases and market reactions", "Section"),
        P(
            "Table 2 reports the eight documents released under Warsh. An event at or before the 4:00 p.m. equity "
            "close uses the prior close to the same-day close; non-trading-day and after-close events roll forward. "
            "DXY is a percent change, yields and 10s2s are basis points, and growth minus value is percentage points.",
        ),
    ])
    labels = {
        "statement": "Statement", "minutes": "Minutes", "speech": "Speech",
        "testimony": "Testimony", "press_conference": "Press conf.",
    }
    table2_data = [["Date", "Release", "Phrase z", "FinBERT z", "DXY %", "10s2s bp", "1Y bp", "G-V pp"]]
    for _, row in warsh.iterrows():
        table2_data.append([
            pd.Timestamp(row.release_date).strftime("%b %d"), labels[row.subtype],
            f"{row.wordlist_z:.2f}", f"{row.finbert_z:.2f}", f"{row.dxy_change_pct:.2f}",
            f"{row.spread_change_bp:.0f}", f"{row.dgs1_change_bp:.0f}", f"{row.growth_value_pp:.2f}",
        ])
    story.append(styled_table(
        table2_data,
        [0.63 * inch, 0.82 * inch, 0.72 * inch, 0.75 * inch, 0.65 * inch, 0.69 * inch, 0.61 * inch, 0.66 * inch],
        small=False,
        alignments=["LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
    ))
    story.extend([
        Spacer(1, 8),
        P("What the event rows show", "Subsection"),
        P(
            "The June statement and press conference coincided with a 14 bp increase in the 1-year yield, a 9 bp "
            "flattening, and a 0.55% DXY rise. The July pair moved in the opposite Treasury direction: the 1-year "
            "yield fell 5 bp and 10s2s steepened 10 bp. July testimony was scored hawkish by both methods, yet the "
            "1-year yield fell 10 bp while growth outperformed value by 1.43 percentage points. These reversals "
            "motivate a full-sample regression rather than a narrative based on eight observations.",
        ),
        P(
            "Statements and 2:30 p.m. press conferences share the same daily return. Table 2 shows both because the "
            "assignment treats each communication as a release, but daily data cannot attribute the day's move "
            "between them. This is the most important event-study limitation.",
        ),
        P("Tone construction", "Subsection"),
        P(
            "The phrase-list measure counts contextual hawkish phrases minus dovish phrases per 1,000 words. The "
            "FinBERT measure embeds up to 20 evenly spaced chunks and computes similarity to six balanced hawkish "
            "and six dovish policy anchors. Both are standardized by document type on the Powell sample and clipped "
            "at four SDs to limit outlier influence.",
        ),
        PageBreak(),
    ])

    # Page 4
    story.extend([
        P("Do tone scores explain market moves?", "Section"),
        P(
            "Each row in Table 3 is a separate OLS regression with the indicated tone score, the one-day DGS3MO "
            "change, and document-type fixed effects. Parentheses contain HC3 standard errors. Stars denote "
            "10%, 5%, and 1% significance.",
        ),
    ])
    table3_data = [["Indicator", "Tone method", "Tone beta (SE)", "DGS3MO beta (SE)", "Adj. R2", "N"]]
    for _, row in regressions.iterrows():
        table3_data.append([
            row["Indicator"], row["Tone method"],
            f"{row['Tone coefficient']:.3f}{significance(row['Tone p-value'])}\n({row['Tone SE']:.3f})",
            f"{row['DGS3MO coefficient']:.3f}\n({row['DGS3MO SE']:.3f})",
            f"{row['Adjusted R2']:.3f}", int(row.N),
        ])
    story.append(styled_table(
        table3_data,
        [1.18 * inch, 1.05 * inch, 1.25 * inch, 1.35 * inch, 0.65 * inch, 0.45 * inch],
        small=False,
        alignments=["LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
    ))
    story.extend([
        Spacer(1, 8),
        P("Regression interpretation", "Subsection"),
        P(
            "None of the eight tone coefficients is statistically significant at 10%. The two methods even imply "
            "opposite DXY and growth-value signs. For 10s2s and the 1-year yield, both methods have small negative "
            "coefficients. By contrast, the DGS3MO coefficient is economically large: a 1 bp increase is associated "
            "with about a 0.039% DXY increase, a 0.44 bp 10s2s decline, and a 0.96 bp 1-year yield increase. Adjusted "
            "R2 is about 8.5% for DXY, 9.1% for 10s2s, 32.2% for the 1-year yield, and approximately zero for "
            "growth minus value.",
        ),
        P(
            "The result supports the assignment's reason for including DGS3MO: without that control, rate decisions "
            "could be misattributed to language. It also warns against overstating the remaining daily-frequency tone "
            "effect. Intraday windows, direct expectations, and a tone-novelty interaction are plausible extensions.",
        ),
        P("Robustness and auditability", "Subsection"),
        P(
            "All source URLs, release dates, Eastern times, raw text, token counts, phrase hits, and event-window "
            "dates are retained in ignored intermediate files. Automated tests verify policy-score orientation, rate "
            "decision labels, weekend roll-forward, and probability totals. Raw data are excluded from Git as required.",
        ),
        PageBreak(),
    ])

    # Page 5
    story.extend([
        P("September 2026 forecast", "Section"),
        P(
            f"The rate model uses {forecast['training_meetings']} prior meetings and blends regularized multinomial-logit "
            "probabilities 80/20 with Laplace-smoothed historical frequencies. Inputs are the July statement tone, "
            f"{forecast['interim_documents']} interim communications, and 20-trading-day changes in DGS1 and 10s2s.",
        ),
    ])
    rate = forecast["rate_probabilities"]
    rate_table = [
        ["Rate decision", "Cut", "Hold", "Hike"],
        ["Probability", f"{rate['Cut']}%", f"{rate['Hold']}%", f"{rate['Hike']}%"],
    ]
    story.append(styled_table(rate_table, [2.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch], alignments=["LEFT", "CENTER", "CENTER", "CENTER"]))
    story.extend([
        Spacer(1, 8),
        P(
            f"Statement tone: <b>{forecast['more_hawkish_probability']}%</b> probability that the statement is more "
            "hawkish than July. The figure is below 50% because the interim minutes were dovish and the two tone "
            "methods disagree on the July reference point.",
        ),
        P("Expected meeting-day market reaction", "Subsection"),
    ])
    market_data = [["Indicator", "Probability it rises", "Expected change"]]
    for label, values in forecast["market_reaction"].items():
        precision = 1 if values["unit"] == "bp" else 2
        market_data.append([
            label.replace(" (bp)", "").replace(" (pp)", "").replace(" (%)", ""),
            f"{values['probability_rise']}%",
            f"{values['expected_change']:+.{precision}f} {values['unit']}",
        ])
    story.append(styled_table(market_data, [2.7 * inch, 1.65 * inch, 1.55 * inch], alignments=["LEFT", "RIGHT", "RIGHT"]))
    story.extend([
        Spacer(1, 8),
        P("Recommendation", "Subsection"),
        P(
            "Enter a <b>small DV01-neutral 2s10s flattener</b>: receive fixed in 10-year swaps and pay fixed in "
            "2-year swaps. The model assigns a 65% probability that 10s2s falls and expects a 1.6 bp decline, the "
            "strongest directional signal among the four required indicators. Both tone specifications have negative "
            "10s2s coefficients, and the rate mixture puts slightly more mass on a hike than a cut.",
        ),
        P(
            "Position size should be small because the tone betas are imprecise and the expected move is modest. A "
            "meeting-day steepening greater than 5 bp, especially alongside a cut and a clearly more dovish statement, "
            "would falsify the thesis. The trade should then be closed rather than rationalized as a delayed reaction.",
            "Callout",
        ),
        Spacer(1, 5),
        P("Forecast uncertainty", "Subsection"),
        P(
            "These are model-assisted academic probabilities, not CME-implied odds. Market sign probabilities use "
            "the mean prediction across both tone regressions and the residual variance. They therefore remain close "
            "to 50% except for the curve, appropriately reflecting the weak in-sample tone evidence.",
        ),
        PageBreak(),
    ])

    # Page 6
    story.extend([
        P("Comparison with the readings", "Section"),
        P("Doh, Kim, and Yang (2021)", "Subsection"),
        P(
            "Their NLP measure shows that qualitative descriptions of economic conditions and risks can move bond "
            "prices even without a policy-rate change. This project follows the same identification principle by "
            "controlling for DGS3MO, but its tone coefficients are weaker. The difference is consistent with daily "
            "rather than high-frequency windows, inclusion of minutes and Chair remarks, and a later sample.",
        ),
        P("Doh, Song, and Yang (2020; updated 2023)", "Subsection"),
        P(
            "Their preferred stance measure locates the official statement between staff-written hawkish and dovish "
            "alternatives and separates tone, novelty, and expectations. The FinBERT anchor score here keeps their "
            "semantic-boundary idea while using public anchors, avoiding the alternatives' publication lag. The cost "
            "is weaker identification of novelty and market surprise.",
        ),
        P("Parsing the Fed (2021)", "Subsection"),
        P(
            "The course presentation compares factor similarity, a phrase lexicon, and FinBERT sentiment; it finds "
            "method- and asset-specific explanatory power and no uniformly dominant model. This project reproduces "
            "the phrase-list and FinBERT/factor-similarity families. It improves policy interpretation by using "
            "balanced hawkish/dovish FinBERT anchors instead of equating generic positive sentiment with hawkishness. "
            "The observed score disagreements and low incremental daily R2 echo the presentation's cautions.",
        ),
        P("Limitations", "Subsection"),
        P(
            "Warsh has only eight releases, so Chair comparisons are descriptive. Daily windows overlap for statements "
            "and press conferences, Treasury closing conventions do not perfectly match equity closes, and FinBERT was "
            "trained for financial sentiment rather than monetary-policy stance. The rate model is also a small-sample "
            "forecast with structural-break risk. These limits justify probability forecasts and a modest trade size.",
        ),
        P("References and data", "Subsection"),
        P(
            "Doh, T., Kim, S., and Yang, S. (2021), <i>How You Say It Matters</i>, Federal Reserve Bank of Kansas City. "
            "<link href='https://www.kansascityfed.org/research/economic-review/how-you-say-it-matters-text-analysis-of-fomc-statements-using-natural-language-processing/'>kansascityfed.org/research/economic-review</link>",
            "Small",
        ),
        P(
            "Doh, T., Song, D., and Yang, S. (2020; updated 2023), <i>Deciphering Federal Reserve Communication via "
            "Text Analysis of Alternative FOMC Statements</i>, RWP 20-14. "
            "<link href='https://www.kansascityfed.org/research/research-working-papers/deciphering-federal-reserve-communication-via-text-analysis/'>kansascityfed.org/research/research-working-papers</link>",
            "Small",
        ),
        P(
            "<i>Parsing the Fed</i> (2021), course presentation. Federal Reserve documents: "
            "<link href='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'>federalreserve.gov/monetarypolicy/fomccalendars.htm</link>. "
            "Treasuries: Federal Reserve H.15 (series corresponding to FRED DGS1, DGS3MO, and T10Y2Y). "
            "DXY and ETFs: Yahoo Finance (DX-Y.NYB, IWF, IWN). Model: ProsusAI/finbert.",
            "Small",
        ),
        P("Academic analysis only; not investment advice.", "Small"),
    ])

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
