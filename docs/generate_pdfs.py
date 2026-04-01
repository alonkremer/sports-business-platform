"""Generate PDF documents for SD FC Pricing Intelligence presentation."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Brand colors
NAVY = HexColor("#002F6C")
RED = HexColor("#C8102E")
DARK_GRAY = HexColor("#1f2937")
MED_GRAY = HexColor("#4b5563")
LIGHT_GRAY = HexColor("#f3f4f6")
BLUE = HexColor("#2563eb")
GREEN = HexColor("#059669")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=20, textColor=NAVY, spaceAfter=2, alignment=TA_LEFT,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=MED_GRAY, spaceAfter=10, alignment=TA_LEFT,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=3,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=DARK_GRAY, spaceAfter=4,
        alignment=TA_JUSTIFY, fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "BodyBold", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=DARK_GRAY, spaceAfter=4,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "BulletCustom", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=DARK_GRAY, spaceAfter=2,
        leftIndent=18, bulletIndent=6, fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "SmallGray", parent=styles["Normal"],
        fontSize=8, textColor=MED_GRAY, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        "KPIValue", parent=styles["Normal"],
        fontSize=15, textColor=NAVY, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=1
    ))
    styles.add(ParagraphStyle(
        "KPILabel", parent=styles["Normal"],
        fontSize=7, textColor=MED_GRAY, alignment=TA_CENTER,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "ViewTitle", parent=styles["Heading3"],
        fontSize=11, textColor=RED, spaceBefore=12, spaceAfter=4,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "InsightBox", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#1e3a5f"),
        backColor=HexColor("#eff6ff"), borderColor=BLUE,
        borderWidth=0.5, borderPadding=6, spaceAfter=8,
        leftIndent=6, rightIndent=6, fontName="Helvetica-Oblique"
    ))
    styles.add(ParagraphStyle(
        "ActionBox", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#065f46"),
        backColor=HexColor("#ecfdf5"), borderColor=GREEN,
        borderWidth=0.5, borderPadding=6, spaceAfter=10,
        leftIndent=6, rightIndent=6, fontName="Helvetica-Bold"
    ))
    return styles


def make_kpi_row(kpis, styles):
    """Create a row of KPI cards."""
    cells = []
    for value, label in kpis:
        cells.append([
            Paragraph(value, styles["KPIValue"]),
            Paragraph(label, styles["KPILabel"])
        ])
    t = Table([cells], colWidths=[1.8 * inch] * len(kpis))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def make_hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#e5e7eb"),
                       spaceBefore=4, spaceAfter=4)


# ---------------------------------------------------------------------------
# DOCUMENT 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
def build_exec_summary():
    path = OUT_DIR / "SD_FC_Executive_Summary.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        topMargin=0.45 * inch, bottomMargin=0.4 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch
    )
    s = get_styles()
    story = []

    # Title block
    story.append(Paragraph("SD FC Pricing Intelligence Platform", s["DocTitle"]))
    story.append(Paragraph("Executive Summary  |  VP of Data Analytics & Strategy", s["DocSubtitle"]))
    story.append(make_hr())

    # KPI strip
    story.append(make_kpi_row([
        ("28,064", "2025 Avg Attendance"),
        ("8-14%", "Forecast Error (MAPE)"),
        ("~$207K", "2025 Revenue Left on Table"),
        ("9-14%", "Projected Uplift (Balanced)"),
    ], s))
    story.append(Spacer(1, 4))

    # The Problem
    story.append(Paragraph("The Problem", s["SectionHead"]))
    story.append(Paragraph(
        "San Diego FC completed its inaugural 2025 season averaging 28,064 fans per game in a "
        "35,000-seat Snapdragon Stadium. Ticket prices were set once before the season using gut feel "
        "and competitor benchmarks. The secondary market told a different story: Baja Cup matches against "
        "Club Tijuana cleared at 40-58% above face value, while midweek games against weaker opponents "
        "sat below face. An estimated $207,000 in revenue was left on the table in Year 1 "
        "and that number grows as the market matures.",
        s["Body"]
    ))

    # The Solution
    story.append(Paragraph("The Solution", s["SectionHead"]))
    story.append(Paragraph(
        "A machine learning platform that recommends optimal ticket prices for every section and every game, "
        "grounded in what fans are actually willing to pay. It combines demand forecasting (XGBoost, 8-14% error rate), "
        "causal price elasticity (Double Machine Learning), and airline-style yield optimization (EMSR-b) "
        "calibrated against verified 2025 attendance data.",
        s["Body"]
    ))
    story.append(Paragraph(
        "The platform outputs three pricing scenarios per section per game:",
        s["Body"]
    ))
    story.append(Paragraph(
        "Conservative: minimal price movement, targets 90%+ fill rate, lowest risk",
        s["BulletCustom"], bulletText="\u2022"
    ))
    story.append(Paragraph(
        "Balanced (recommended): optimizes revenue while protecting season ticket holder resale margins",
        s["BulletCustom"], bulletText="\u2022"
    ))
    story.append(Paragraph(
        "Aggressive: maximum revenue extraction for confirmed high-demand games only",
        s["BulletCustom"], bulletText="\u2022"
    ))
    story.append(Paragraph(
        'Every recommendation includes a plain-English explanation: "Section 111 is $8 below optimal. '
        'Drivers: rivalry game (+$8), high secondary demand (+$5), Saturday night (+$3)."',
        s["Body"]
    ))

    # What Makes It Different
    story.append(Paragraph("What Makes It Different", s["SectionHead"]))
    diffs = [
        ("Causal, not correlated", "isolates the true effect of price on demand, controlling for opponent quality, weather, and timing"),
        ("Explainable", "every recommendation comes with reasons your ticketing team can understand and challenge"),
        ("Guardrailed", "STH resale margins protected by design (primary never exceeds secondary minus 10%); any increase above 25% flagged for manual review"),
        ("Three choices, not one", "management picks the risk level, the model handles the math"),
    ]
    for title, desc in diffs:
        story.append(Paragraph(f"<b>{title}</b> -- {desc}", s["BulletCustom"], bulletText="\u2022"))

    # Implementation Path
    story.append(Paragraph("Implementation Path", s["SectionHead"]))
    phases = [
        ["Phase", "Timeline", "What Happens"],
        ["Phase 1", "Now", "Offline recommendations. Run model before each pricing window, present scenarios, manually update Archtics."],
        ["Phase 2", "3-6 months", "API integration with Ticketmaster Host for automated price updates with human approval gate."],
        ["Phase 3", "6-12 months", "Real-time feedback loop with live sell-through tracking and mid-season recalibration."],
    ]
    phase_table = Table(phases, colWidths=[0.7 * inch, 0.8 * inch, 5.3 * inch])
    phase_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#d1d5db")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(phase_table)

    # 90-Day Plan
    story.append(Paragraph("First 90 Days", s["SectionHead"]))
    story.append(Paragraph(
        "<b>Days 1-30:</b> Replace synthetic data with real Archtics/CRM feeds. Validate model accuracy on real transaction data.",
        s["Body"]
    ))
    story.append(Paragraph(
        "<b>Days 31-60:</b> First live pricing recommendations for early-season home games. A/B test: control vs model-priced section groups. Dashboard live for ticketing team.",
        s["Body"]
    ))
    story.append(Paragraph(
        "<b>Days 61-90:</b> First measurable results. Revenue uplift report. STH health validation. Phase 2 roadmap presented.",
        s["Body"]
    ))

    # Footer
    story.append(Spacer(1, 6))
    story.append(make_hr())
    story.append(Paragraph(
        "Built with Python, XGBoost, DuckDB, Streamlit, FastAPI, SHAP  |  "
        "Data: FBref, SeatGeek, StubHub, Vivid Seats, OddsPortal, MLS Official, Ticketmaster",
        s["SmallGray"]
    ))

    doc.build(story)
    print(f"Executive summary: {path}")
    return path


# ---------------------------------------------------------------------------
# DOCUMENT 2: USER GUIDE
# ---------------------------------------------------------------------------
def build_user_guide():
    path = OUT_DIR / "SD_FC_User_Guide.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch
    )
    s = get_styles()
    story = []

    # Title
    story.append(Paragraph("SD FC Pricing Intelligence", s["DocTitle"]))
    story.append(Paragraph("User Guide  |  How to Read the Dashboard and Draw Insights", s["DocSubtitle"]))
    story.append(make_hr())

    # Getting Started
    story.append(Paragraph("Getting Started", s["SectionHead"]))
    story.append(Paragraph(
        "Open the dashboard at localhost:8501. The sidebar on the left has four views and a Strategic Mode selector. "
        "Start with the default: Balanced scenario, Revenue Optimization mode. The stadium seat map is your home screen.",
        s["Body"]
    ))

    # --- View 1: Seat Map ---
    story.append(Paragraph('View 1: Seat Map -- "Where are the opportunities right now?"', s["ViewTitle"]))
    story.append(Paragraph(
        "This is your starting point for any game. Select a game from the dropdown. The stadium map lights up with color-coded sections:",
        s["Body"]
    ))
    colors_data = [
        ["Color", "Meaning", "What To Do"],
        ["Red", "Underpriced -- raise price (secondary well above face)", "Consider Balanced or Aggressive pricing"],
        ["Light Red", "Moderately underpriced (room to move)", "Balanced scenario recommended"],
        ["Gray", "Priced about right", "No action needed"],
        ["Blue", "Overpriced or low demand -- consider lowering", "Consider Conservative (may lower price)"],
    ]
    ct = Table(colors_data, colWidths=[0.7 * inch, 2.8 * inch, 3.2 * inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#d1d5db")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(ct)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The Game Impact Score (1-10) at the top tells you the overall demand level for that match.",
        s["Body"]
    ))

    story.append(Paragraph(
        'EXAMPLE: You select "Sep 27 vs Club Tijuana (Baja Cup)". The map is mostly red and orange. '
        "Game Impact Score: 9.2. Lower bowl midfield (sections 111-115) shows face price $75, optimal $91, "
        "gap of $16/seat. This is your highest-opportunity game. Switch to Aggressive scenario using the "
        "radio buttons at the top. Field club sections show $180 face vs $245 secondary -- even aggressive "
        "pricing at $220 keeps STH margins healthy.",
        s["InsightBox"]
    ))
    story.append(Paragraph(
        "ACTION: Flag this game for aggressive pricing. Present the three scenarios side by side to the VP of Ticketing.",
        s["ActionBox"]
    ))

    # --- View 2: Pricing Workshop ---
    story.append(Paragraph('View 2: Pricing Workshop -- "What happens if I change this price?"', s["ViewTitle"]))
    story.append(Paragraph(
        "Select a game and section. The slider lets you drag the price up or down. Four metrics update in real time:",
        s["Body"]
    ))
    story.append(Paragraph("Projected Attendance for that section", s["BulletCustom"], bulletText="\u2022"))
    story.append(Paragraph("Section Revenue (price x expected seats sold)", s["BulletCustom"], bulletText="\u2022"))
    story.append(Paragraph("STH Resale Margin (does a season ticket holder profit if they resell?)", s["BulletCustom"], bulletText="\u2022"))
    story.append(Paragraph("Sell-through percentage", s["BulletCustom"], bulletText="\u2022"))
    story.append(Paragraph(
        "The demand curve chart shows your proposed price as a red dashed line against the revenue curve. "
        "The STH safe ceiling is a blue dashed line -- go above it and you hurt your season ticket holders.",
        s["Body"]
    ))

    story.append(Paragraph(
        'EXAMPLE: You pick "Jun 21 vs Nashville" (midweek, cold market). Current face: $55 for upper bowl. '
        "You slide to $48. Projected sell-through jumps from 68% to 79%. Revenue actually increases because "
        "you fill 385 more seats. The STH margin card stays green. This is a case where lowering the price "
        "makes more money -- the Conservative scenario catches this automatically.",
        s["InsightBox"]
    ))
    story.append(Paragraph(
        "ACTION: For cold-market games, compare Conservative vs Balanced. Conservative often recommends a small "
        "price reduction that fills more seats and generates more total revenue.",
        s["ActionBox"]
    ))

    # --- View 3: STH Value Dashboard ---
    story.append(Paragraph('View 3: STH Value Dashboard -- "Can I prove season tickets are worth it?"', s["ViewTitle"]))
    story.append(Paragraph(
        "This view powers your renewal campaign. It shows the average resale profit a season ticket holder "
        "can expect per game, broken down by section tier. The bar chart shows margin by game -- tall green bars "
        "mean high resale value (hot games), short or red bars mean the STH breaks even or loses (cold games).",
        s["Body"]
    ))

    story.append(Paragraph(
        'EXAMPLE: Filter to "lower_bowl_midfield". The dashboard shows average season resale value $142/seat '
        "above face across 17 games, average margin 11.2%. The blockquote at the bottom gives you "
        'copy-paste renewal language: "Your season tickets in the lower bowl consistently trade at '
        '10-15% above face value on the secondary market." But you notice 3 midweek games show negative '
        "margins -- the Balanced scenario already prices those slightly below face to protect STH value.",
        s["InsightBox"]
    ))
    story.append(Paragraph(
        "ACTION: Use the STH Value Statement in renewal emails. Filter by tier to customize messaging for each section group.",
        s["ActionBox"]
    ))

    # --- View 4: Performance Report ---
    story.append(Paragraph('View 4: Performance Report -- "Is this actually working?"', s["ViewTitle"]))
    story.append(Paragraph(
        "Your accountability view. It shows model accuracy (MAPE), revenue uplift vs flat pricing, "
        "market health distribution, and the 2025 retrospective -- money left on the table last season, "
        "broken down by game.",
        s["Body"]
    ))

    story.append(Paragraph(
        "EXAMPLE: The 2025 retrospective bar chart shows the Club Tijuana home match had the largest "
        "revenue opportunity at ~$38,000 for that single game. The LAFC rivalry match was second at ~$29,000. "
        "These are the games where aggressive pricing in 2026 has the strongest evidence. The midweek Pachuca "
        "match shows near-zero opportunity -- Conservative pricing is the right call for similar 2026 games.",
        s["InsightBox"]
    ))
    story.append(Paragraph(
        'ACTION: Use the retrospective to justify the pricing strategy. "$207K left on the table in Year 1" is the headline.',
        s["ActionBox"]
    ))

    # --- Strategic Modes ---
    story.append(Paragraph("Strategic Modes (Sidebar Selector)", s["SectionHead"]))
    modes_data = [
        ["Mode", "Behavior", "When To Use"],
        ["Revenue Optimization", "Pure revenue maximization within STH guardrails", "Default mode for most games"],
        ["Fan Acquisition", "Caps all increases at +10%, prioritizes filling seats", "Early seasons, building the market"],
        ["Atmosphere (Sellout)", "Targets 90%+ sell-through, prices down on cold games", "Marquee events, TV games, atmosphere priority"],
    ]
    mt = Table(modes_data, colWidths=[1.4 * inch, 2.6 * inch, 2.7 * inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#d1d5db")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(mt)

    # --- Daily Workflow ---
    story.append(Paragraph("Quick Reference: Daily Workflow", s["SectionHead"]))
    steps = [
        "Open the Seat Map. Check which upcoming games have red/orange sections.",
        "For hot games: open the Pricing Workshop, experiment with Balanced and Aggressive prices, confirm STH margins stay healthy.",
        "For cold games: check if Conservative recommends a price reduction that improves sell-through.",
        "Before renewal season: pull up the STH Value Dashboard, export the resale margin data by tier.",
        "Monthly: review the Performance Report to track model accuracy and cumulative revenue uplift.",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b>  {step}", s["Body"]))

    # Footer
    story.append(Spacer(1, 12))
    story.append(make_hr())
    story.append(Paragraph(
        "SD FC Pricing Intelligence  |  Questions? Contact VP of Data Analytics & Strategy",
        s["SmallGray"]
    ))

    doc.build(story)
    print(f"User guide: {path}")
    return path


if __name__ == "__main__":
    build_exec_summary()
    build_user_guide()
    print("Done.")
