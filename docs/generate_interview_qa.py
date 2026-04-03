"""Generate COO Interview Prep Q&A PDF for SD FC Pricing Intelligence Platform."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak
)
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
AMBER = HexColor("#d97706")

Q_BG = HexColor("#eff6ff")
A_BG = HexColor("#f9fafb")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=18, textColor=NAVY, spaceAfter=2, alignment=TA_LEFT,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=MED_GRAY, spaceAfter=10, alignment=TA_LEFT,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=12, textColor=NAVY, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "CategoryHead", parent=styles["Heading3"],
        fontSize=10, textColor=RED, spaceBefore=8, spaceAfter=3,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "Question", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#1e3a5f"),
        fontName="Helvetica-Bold", spaceAfter=2,
        leftIndent=6, rightIndent=6,
    ))
    styles.add(ParagraphStyle(
        "Answer", parent=styles["Normal"],
        fontSize=8.5, leading=11.5, textColor=DARK_GRAY, spaceAfter=2,
        alignment=TA_JUSTIFY, fontName="Helvetica",
        leftIndent=6, rightIndent=6,
    ))
    styles.add(ParagraphStyle(
        "AnswerBold", parent=styles["Normal"],
        fontSize=8.5, leading=11.5, textColor=DARK_GRAY, spaceAfter=2,
        fontName="Helvetica-Bold",
        leftIndent=6, rightIndent=6,
    ))
    styles.add(ParagraphStyle(
        "BulletCustom", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=DARK_GRAY, spaceAfter=2,
        leftIndent=22, bulletIndent=10, fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "TipBox", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=HexColor("#92400e"),
        backColor=HexColor("#fffbeb"), borderColor=AMBER,
        borderWidth=0.5, borderPadding=5, spaceAfter=6,
        leftIndent=6, rightIndent=6, fontName="Helvetica-Oblique"
    ))
    styles.add(ParagraphStyle(
        "SmallGray", parent=styles["Normal"],
        fontSize=7.5, textColor=MED_GRAY, alignment=TA_CENTER
    ))
    return styles


def make_hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#e5e7eb"),
                       spaceBefore=3, spaceAfter=3)


def qa_block(q_text, a_text, s, tip=None):
    """Create a Q&A block with optional tip."""
    elements = []
    # Question row with background
    q_table = Table(
        [[Paragraph(f"Q: {q_text}", s["Question"])]],
        colWidths=[6.8 * inch]
    )
    q_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), Q_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4, 4, 0, 0]),
    ]))
    elements.append(q_table)

    # Answer row
    a_table = Table(
        [[Paragraph(a_text, s["Answer"])]],
        colWidths=[6.8 * inch]
    )
    a_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), A_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [0, 0, 4, 4]),
    ]))
    elements.append(a_table)

    if tip:
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(f"Interview tip: {tip}", s["TipBox"]))

    elements.append(Spacer(1, 6))
    return KeepTogether(elements)


def build_interview_qa():
    path = OUT_DIR / "SD_FC_Interview_QA.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        topMargin=0.45 * inch, bottomMargin=0.4 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch
    )
    s = get_styles()
    story = []

    # Title
    story.append(Paragraph("SD FC Pricing Intelligence Platform", s["DocTitle"]))
    story.append(Paragraph(
        "COO Interview Prep  |  VP of Data Analytics &amp; Strategy  |  Q&amp;A Reference",
        s["DocSubtitle"]
    ))
    story.append(make_hr())

    story.append(Paragraph(
        "This document anticipates technical, business, and strategic questions a COO would ask "
        "about the pricing intelligence platform. Each answer is framed from the perspective of "
        "the VP candidate presenting this as their work product.",
        s["Answer"]
    ))
    story.append(Spacer(1, 6))

    # =====================================================================
    # SECTION 1: MODEL & METHODOLOGY
    # =====================================================================
    story.append(Paragraph("1. Model &amp; Methodology", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "Walk me through the model. How does it actually decide on a price?",
        "It's a three-layer system. <b>Layer 1</b> is an XGBoost demand model that forecasts "
        "expected attendance (0\u2013100%) for every section \u00d7 game combination, using 60+ features "
        "across 14 groups \u2014 schedule, opponent strength, weather, secondary-market signals, inventory "
        "velocity, social sentiment, and cross-border demand. "
        "<b>Layer 2</b> is a causal elasticity model using Double Machine Learning (DML) that isolates "
        "the true price-to-demand relationship per seating tier, controlling for confounders like "
        "rivalry status and weather. This tells us <i>how much</i> demand drops per dollar of price increase. "
        "<b>Layer 3</b> is an EMSR-b revenue optimizer (the same algorithm airlines use for seat pricing) "
        "that finds the revenue-maximizing price for each section, then applies strategic guardrails "
        "\u2014 STH resale protection, fan-equity caps, backlash risk flags \u2014 to produce three scenarios: "
        "Conservative (0.5\u00d7), Balanced (1.0\u00d7, recommended), and Aggressive (1.5\u00d7).",
        s,
        tip="Emphasize the guardrails layer \u2014 COOs care that the model won't embarrass the club."
    ))

    story.append(qa_block(
        "What are the most important features driving the model's predictions?",
        "The top drivers (by SHAP feature importance) are: "
        "<b>(1) Secondary market premium</b> \u2014 the ratio of resale price to face price is the single strongest signal. "
        "If StubHub transactions are clearing at 40% above face, the market is telling us we're underpriced. "
        "<b>(2) Opponent tier and rivalry status</b> \u2014 Baja Cup matches against Club Tijuana drive 2\u20133\u00d7 the demand "
        "of a midweek match against a bottom-table team. "
        "<b>(3) Sell-through velocity at T-7</b> \u2014 how fast seats are selling in the final week is a strong "
        "real-time demand signal. "
        "<b>(4) Day of week and season progress</b> \u2014 Saturday games significantly outperform Wednesday games; "
        "late-season games with playoff implications spike. "
        "<b>(5) Cross-border index</b> \u2014 unique to San Diego: ~6,000 regular fans cross from Tijuana, scaling to "
        "2.5\u00d7 for Baja Cup games. This is a feature no other MLS club needs.",
        s,
        tip="Name the San Diego-specific features (cross-border, military market) \u2014 it shows you built this FOR this club, not as a generic tool."
    ))

    story.append(qa_block(
        "How do you know the model is accurate? What's your error rate?",
        "We target a Mean Absolute Percentage Error (MAPE) under 15% on demand forecasts, validated "
        "using time-series cross-validation with a 14-day purge gap between folds \u2014 this prevents "
        "data leakage from sequential games. The MAPE tracks at around 8\u201312% in testing. "
        "But more importantly for business decisions, I built a <b>sensitivity analysis</b> that asks the "
        "real COO question: <i>what if the model underestimates how much demand drops when we raise prices?</i> "
        "Even if actual price elasticity is 50% steeper than our estimate, the Balanced scenario "
        "still delivers +2.6% uplift. The model's elasticity would need to be <b>2\u00d7 wrong</b> before flat "
        "pricing wins. The model doesn't need to be perfect \u2014 it just needs to be directionally right.",
        s,
        tip="Lead with the sensitivity analysis, not MAPE. COOs don't think in MAPE; they think in 'what if you're wrong?'"
    ))

    story.append(qa_block(
        "The sensitivity table \u2014 how confident can we be in the uplift numbers?",
        "The sensitivity table stress-tests the model's key assumption: <b>price elasticity</b>. "
        "Our model estimates how much demand drops per dollar of price increase (e.g., \u22120.58 for lower "
        "bowl midfield). The table asks: what if fans are actually <i>more</i> price-sensitive than we think? "
        "Results: at 50% elasticity error, Balanced still delivers +2.6% uplift. At 2\u00d7 elasticity error "
        "(model is 100% wrong about price sensitivity), we break even with flat pricing. "
        "This is a <b>very wide margin of safety</b> \u2014 academic elasticity estimates for sports tickets "
        "typically have standard errors of 15\u201325%, not 100%. "
        "The Conservative scenario is even safer: it barely moves prices, so even extreme elasticity error "
        "doesn't hurt. The Aggressive scenario gives the most upside (+7.8%) but is the first to lose "
        "if elasticity is badly wrong. That's why we recommend Balanced as the default.",
        s,
        tip="Frame it as: 'The model needs to be directionally right, not precisely right. Even if we're 50% off on elasticity, we still win.'"
    ))

    story.append(qa_block(
        "Why did you choose XGBoost over deep learning or a simpler model?",
        "Three reasons. <b>(1) Interpretability</b> \u2014 XGBoost produces SHAP explanations for every prediction, "
        "so when pricing ops asks 'why is the model recommending +$12 for this section?', we can show the "
        "exact feature contributions. A neural net is a black box. <b>(2) Sample size</b> \u2014 with ~238 section\u00d7game "
        "rows per season, we don't have millions of records. XGBoost handles small data well with proper "
        "regularization (max_depth=5, min_child_weight=3, gamma=0.1). A deep model would overfit. "
        "<b>(3) Production simplicity</b> \u2014 XGBoost runs in milliseconds, serializes to a small file, "
        "and doesn't require GPU infrastructure. For a first-year MLS club, operational simplicity matters.",
        s
    ))

    story.append(qa_block(
        "What is Double Machine Learning and why use it for elasticity?",
        "Standard regression would give us a <i>correlation</i> between price and demand, which is biased: "
        "we raise prices for popular games (rivalry, marquee), so naive regression would show higher prices "
        "= higher demand. DML solves this by <b>partialing out confounders</b>. "
        "Step 1: predict price from confounders (opponent, weather, significance) \u2014 take the residual. "
        "Step 2: predict demand from the same confounders \u2014 take the residual. "
        "Step 3: regress demand residual on price residual \u2014 this gives the <b>causal</b> price effect. "
        "The elasticities range from \u22120.35 (field club, very inelastic \u2014 committed premium buyers) to "
        "\u22121.10 (upper concourse, very elastic \u2014 price-sensitive fans). These match academic sports "
        "economics literature and our tier priors, which gives us confidence the estimates are real.",
        s,
        tip="If they push on 'is it really causal?', be honest: it's the best we can do with observational data. A true RCT would require randomizing ticket prices, which creates legal and PR risk. DML is the industry gold standard for this constraint."
    ))

    # =====================================================================
    # SECTION 2: DATA & ACCURACY
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Data &amp; Accuracy", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "What data sources does this use? Is any of it real-time?",
        "Currently the platform ingests 11 data sources: Ticketmaster (face prices), StubHub/SeatGeek/VividSeats "
        "(secondary market transactions and listings), OddsPortal (betting odds as a proxy for expected match "
        "quality), FBRef (team performance metrics like xG), weather forecasts, and our internal inventory "
        "snapshots. "
        "Right now, the pipeline runs in <b>batch mode</b> \u2014 features are computed and the model re-scores "
        "weekly. It is not real-time. The architecture supports a shift to near-real-time (API endpoints "
        "already exist), but for Year 1, batch is the right trade-off: it keeps the system simple, "
        "lets pricing ops review recommendations before they go live, and avoids the operational risk "
        "of fully automated price changes during a club's first season.",
        s,
        tip="Be upfront that it's batch, then frame it as a deliberate Year 1 choice. The roadmap to real-time is clear."
    ))

    story.append(qa_block(
        "You're using synthetic data. How can you trust the results?",
        "Good question \u2014 this is the bootstrap problem every new franchise faces: you can't build a model without "
        "data, and you can't collect data without a season. Here's how we handled it: "
        "<b>(1)</b> The synthetic data is calibrated against real MLS benchmarks \u2014 league-wide attendance distributions, "
        "published secondary market premiums from StubHub, actual weather data for San Diego, and real 2025 betting "
        "odds from OddsPortal. <b>(2)</b> We anchored face prices to Ticketmaster's published tier structure. "
        "<b>(3)</b> The secondary market premiums follow established patterns from comparable expansion clubs (Charlotte FC, "
        "St. Louis CITY). <b>(4)</b> Most importantly, the model architecture is designed to <b>retrain on real data</b> "
        "as soon as the 2025 season provides actual transactions. The synthetic data is scaffolding \u2014 the value "
        "is in the pipeline, the feature engineering, and the optimizer logic, which all transfer directly to real data.",
        s,
        tip="Acknowledge it openly, then redirect to the pipeline value. The synthetic data is ~80% of the questions about trustworthiness."
    ))

    story.append(qa_block(
        "How would you validate this with real data once the season starts?",
        "Four-stage validation plan: "
        "<b>(1) Holdout backtesting</b> \u2014 after 5\u20136 home games (~70 section\u00d7game observations), retrain the demand "
        "model on games 1\u20134 and predict game 5\u20136. Compare predicted vs actual attendance by section. "
        "<b>(2) A/B price testing</b> \u2014 pick 2\u20133 comparable sections and apply dynamic pricing to one, keep the "
        "other at flat pricing. Measure revenue lift after 4\u20135 games. "
        "<b>(3) Secondary market tracking</b> \u2014 monitor whether the gap between face and resale narrows as we "
        "implement recommendations, which would confirm we're capturing the right price points. "
        "<b>(4) STH satisfaction survey</b> \u2014 quarterly check that season ticket holders can still resell at a profit, "
        "because if STH margins go negative, renewal rates collapse and the long-term economics fail.",
        s
    ))

    story.append(qa_block(
        "What happens if the model is wrong on a specific game?",
        "The system has multiple safety nets. <b>First</b>, pricing ops always has final approval \u2014 no price "
        "goes live without a human reviewing the recommendation. <b>Second</b>, the guardrail layer caps maximum "
        "price increases at 25% before flagging for manual review (backlash risk threshold). <b>Third</b>, "
        "floor and ceiling prices per tier ensure we never go below cost or above market-breaking levels "
        "(e.g., supporters GA is capped at $18\u2013$45). <b>Fourth</b>, the STH resale guarantee ensures primary "
        "price never exceeds 90% of the secondary sold average, so season ticket holders are always protected. "
        "If all of that fails on a specific game, the blast radius is one section on one night \u2014 it's not a "
        "systemic risk.",
        s,
        tip="COOs think in risk management. List the guardrails as defense-in-depth layers."
    ))

    # =====================================================================
    # SECTION 3: BUSINESS IMPACT & REVENUE
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Business Impact &amp; Revenue", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "Where does the $796K revenue opportunity come from?",
        "It comes from the price gap analysis on the 2025 season. For every section \u00d7 game, we compare "
        "the face price to the optimal face price implied by secondary market clearing prices. "
        "Formula: optimal_face = sold_price_avg / 1.15. The gap between that and the actual face price, "
        "multiplied by remaining seats, gives the revenue left on the table. "
        "Summing across all 238 section\u00d7game combinations for 2025: <b>$796K</b>. "
        "The bulk comes from high-demand games \u2014 Club Tijuana alone accounts for ~$109K because secondary "
        "prices cleared at 40\u201358% above face. The opportunity isn't everywhere equally \u2014 about 53% of "
        "section\u00d7game combinations are 'healthy' or 'hot' (demand at or above face), and that's where "
        "the money is.",
        s
    ))

    story.append(qa_block(
        "What does +5.3% revenue uplift actually mean in dollars?",
        "For the 2025 season, total baseline revenue (face price \u00d7 capacity \u00d7 demand) across all "
        "section\u00d7game combinations is approximately $35.2M. A 5.3% uplift on that is roughly "
        "<b>$1.9M in additional revenue</b> per season. For context, that's the salary equivalent of "
        "2\u20133 quality MLS players, or the operating budget for an entire analytics department for 5+ years. "
        "And this accounts for demand elasticity \u2014 some fans will buy less when prices go up, and the model "
        "factors that in. The Aggressive scenario projects +7.8%, though we recommend Balanced because it "
        "protects STH margins and minimizes backlash risk.",
        s
    ))

    story.append(qa_block(
        "How does this compare to what other MLS clubs are doing?",
        "Most MLS clubs use rudimentary dynamic pricing \u2014 typically 3\u20134 manual price tiers set by a "
        "revenue manager based on perceived demand. A few clubs (Atlanta United, LAFC, Seattle) have "
        "invested in more sophisticated systems, but the industry is still early. "
        "What differentiates this platform: <b>(1)</b> Section-level granularity (most clubs price at "
        "tier level only, missing intra-tier variation). <b>(2)</b> Causal elasticity estimation (most "
        "clubs use rules of thumb, not econometric models). <b>(3)</b> STH health as a first-class "
        "constraint (most systems optimize revenue without protecting the resale market). "
        "<b>(4)</b> San Diego-specific features like cross-border demand and military market scoring "
        "that no off-the-shelf vendor provides.",
        s,
        tip="Position as 'best-in-class for a Year 1 club' \u2014 don't oversell vs clubs with 10 years of real data."
    ))

    story.append(qa_block(
        "What's the cost to implement this? What's the ROI?",
        "Infrastructure cost is minimal: the entire stack runs on open-source tools (Python, XGBoost, "
        "DuckDB, Streamlit). Cloud hosting is ~$50\u2013200/month. The primary cost is people \u2014 one senior "
        "analytics hire (the VP role) plus one pricing analyst to review and action recommendations. "
        "All-in, call it $300\u2013400K/year in fully loaded headcount + tools. Against a projected $1.9M "
        "revenue uplift in Year 1 (Balanced scenario, elasticity-adjusted), that's a <b>5:1 ROI in Year 1</b>, "
        "growing as the model improves with real data. By Year 2\u20133 with real data, the uplift should increase to "
        "8\u201312% as the demand model tightens and we add real-time secondary market feeds.",
        s
    ))

    # =====================================================================
    # SECTION 4: STAKEHOLDER MANAGEMENT & RISK
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Stakeholder Management &amp; Risk", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "How do you avoid negative PR from dynamic pricing? Fans hate being charged more.",
        "This is the #1 risk and I built the system around it. Three strategies: "
        "<b>(1) Never call it 'dynamic pricing' externally.</b> The language matters \u2014 we frame it as "
        "'market-based pricing' or 'demand-based seating tiers.' Airlines and hotels normalized this; "
        "sports clubs haven't yet, so the framing has to be careful. "
        "<b>(2) The backlash risk flag.</b> Any section\u00d7game where the recommended increase exceeds 25% "
        "gets flagged for manual review. We don't blindly push a 40% increase on a rivalry game \u2014 "
        "that's a social media crisis waiting to happen. "
        "<b>(3) Protect the STH first.</b> Season ticket holders are the economic backbone. The model "
        "guarantees they can resell at a net profit after marketplace fees. If a price increase would "
        "make an STH underwater on resale, the model won't recommend it. Happy STHs don't complain "
        "publicly and their renewal rate stays high.",
        s,
        tip="This is likely the COO's top concern. Lead with empathy for the fan experience, then show the safeguards."
    ))

    story.append(qa_block(
        "How do you get buy-in from internal stakeholders \u2014 ticket ops, marketing, the head coach's staff?",
        "The dashboard is designed as a <b>collaboration tool, not a dictation tool</b>. "
        "<b>Ticket Ops</b>: The Pricing Workshop view lets them test any price point with a slider and "
        "immediately see the impact on revenue, attendance, and STH margins. They keep final say. "
        "<b>Marketing</b>: The STH Value Dashboard shows which games have healthy resale margins (good for "
        "renewals) and which are cold (opportunities for targeted promos). They can use it for campaign planning. "
        "<b>Finance/FP&amp;A</b>: The Performance Report gives them sensitivity analysis and revenue projections "
        "they can put into board decks. "
        "The key principle: <b>transparency builds trust</b>. Every recommendation comes with a SHAP-based "
        "plain-English explanation ('this price increase is driven by Club Tijuana rivalry + Saturday + "
        "85% sell-through at T-7'). Nobody has to take the model's word for it.",
        s
    ))

    story.append(qa_block(
        "What if season ticket holders complain that dynamic pricing hurts their resale value?",
        "The system is explicitly designed to <b>protect STH resale economics</b>. The STH resale guarantee "
        "guardrail ensures that primary prices never exceed 90% of the average secondary market clearing "
        "price (after accounting for ~10% marketplace fees). This means an STH can always resell at a net "
        "profit on hot games. "
        "The dashboard tracks this in real time \u2014 the STH Value Dashboard shows net margin per seat, "
        "per game, after fees. For 2025, the average STH net resale value is $204/seat across the season, "
        "with only 4 of 17 games showing negative margins (cold midweek games \u2014 exactly the ones STHs "
        "would attend themselves rather than resell). "
        "If anything, smart dynamic pricing <b>helps</b> STHs: by pricing hot games closer to market clearing, "
        "we reduce the gap that scalpers exploit, which means more of the value stays in the STH ecosystem.",
        s,
        tip="'We're protecting STHs FROM the scalpers, not competing with them' is a powerful reframe."
    ))

    story.append(qa_block(
        "What's the worst-case scenario if this goes wrong?",
        "The worst case is a visible pricing error on a high-profile game \u2014 say, dropping supporters "
        "section prices to $18 for a rivalry game (model underpredicts demand) or pushing premium seats "
        "to $400 on a Wednesday game nobody wants. "
        "Mitigation: <b>(1)</b> Human review before any price goes live. <b>(2)</b> Hard floor and ceiling "
        "prices per tier ($18\u2013$45 for supporters, $140\u2013$400 for field club). <b>(3)</b> The 25% backlash "
        "threshold. <b>(4)</b> We start with the Balanced scenario, not Aggressive, keeping adjustments moderate. "
        "<b>(5)</b> We can roll back to flat pricing on any game within minutes \u2014 the system is advisory, "
        "not automated. "
        "Financial worst case: if the model is consistently wrong (MAPE >25%), we're back to flat pricing "
        "with no uplift \u2014 but we haven't lost money, just the opportunity cost. The downside is bounded; "
        "the upside is not.",
        s
    ))

    story.append(qa_block(
        "Are there legal or compliance considerations with dynamic pricing?",
        "Yes, a few. <b>(1) State consumer protection laws</b> \u2014 California doesn't prohibit dynamic pricing "
        "for events (it's legal), but we need clear disclosure at point of sale that prices are demand-based. "
        "<b>(2) ADA accessibility</b> \u2014 accessible seating must be available at the lowest price offered "
        "for the section; we enforce this in the guardrail layer. "
        "<b>(3) Sponsor/partner contractual pricing</b> \u2014 some sections may have fixed-price allocations "
        "for sponsors; the model excludes those from optimization. "
        "<b>(4) MLS league policies</b> \u2014 the league encourages revenue optimization but has soft "
        "guidance against price gouging on rivalry games. Our backlash risk flag handles this.",
        s
    ))

    # =====================================================================
    # SECTION 5: TESTING & IMPLEMENTATION
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Real-World Testing &amp; Implementation", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "How would you test this in real life before fully rolling it out?",
        "<b>Phase 1 (Games 1\u20134): Shadow mode.</b> Run the model on every game but don't change any prices. "
        "Compare what the model would have recommended vs what actually happened. Track prediction "
        "accuracy and build confidence. "
        "<b>Phase 2 (Games 5\u20138): A/B test on 2\u20133 sections.</b> Pick comparable sections (e.g., two upper "
        "bowl sections with similar demand profiles). Apply dynamic pricing to one, keep the other flat. "
        "Measure revenue, attendance, and STH feedback. "
        "<b>Phase 3 (Games 9+): Gradual rollout.</b> If the A/B shows positive lift with no backlash, "
        "expand to all sections using the Conservative scenario first, then graduate to Balanced. "
        "<b>Phase 4 (Season 2): Full deployment.</b> By now we have 17 games of real data to retrain on. "
        "Move to Balanced scenario across the board with weekly model refreshes.",
        s,
        tip="A phased rollout shows operational maturity. COOs love 'crawl, walk, run' language."
    ))

    story.append(qa_block(
        "What KPIs would you track to know if this is working?",
        "Five primary metrics: "
        "<b>(1) Revenue per game vs flat-price baseline</b> \u2014 the headline number. Target: +5\u201310% in Year 1. "
        "<b>(2) Demand forecast MAPE</b> \u2014 model accuracy. Target: &lt;15%, ideally &lt;10%. "
        "<b>(3) STH net resale margin per game</b> \u2014 must stay positive on 80%+ of games. If this drops, "
        "we're over-pricing and hurting the renewal pipeline. "
        "<b>(4) Fan sentiment score</b> \u2014 social listening + CSAT surveys. If dynamic pricing discussions "
        "go negative, we dial back. "
        "<b>(5) Attendance rate vs optimal capacity</b> \u2014 are we filling seats or just raising price "
        "on empty sections? Revenue optimization with 70% attendance is worse than moderate pricing at 95%. "
        "Secondary metrics: secondary market gap closure, price change acceptance rate by section, "
        "and guardrail trigger frequency (if >30% of recommendations hit guardrails, the model needs recalibration).",
        s
    ))

    story.append(qa_block(
        "Is this system real-time, or does someone have to run it manually?",
        "Currently it runs in <b>batch mode</b> \u2014 the full pipeline (feature engineering \u2192 model scoring \u2192 "
        "price recommendations) executes on-demand or on a weekly schedule. The results feed into the "
        "Streamlit dashboard where pricing ops reviews and approves changes. "
        "This is a <b>deliberate Year 1 choice</b>. Real-time pricing requires: (a) live inventory APIs, "
        "(b) automated price push to Ticketmaster, (c) incident response for bad recommendations. "
        "For a club's first season, adding that operational complexity creates more risk than value. "
        "The <b>roadmap to real-time</b> is: Year 2 \u2014 near-real-time secondary market feeds with "
        "4-hour refresh; Year 3 \u2014 automated Ticketmaster API integration with human-in-the-loop "
        "approval workflow; Year 4+ \u2014 fully automated with exception-based human review only.",
        s
    ))

    # =====================================================================
    # SECTION 6: STRATEGIC VISION & GROWTH
    # =====================================================================
    story.append(Paragraph("6. Strategic Vision &amp; Growth", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "Where does this go after Year 1? What's the bigger vision?",
        "This ticket pricing module is <b>Phase 1 of a broader sports business intelligence platform</b>. "
        "The roadmap: "
        "<b>Phase 2 \u2014 Sponsorship valuation:</b> Use the same demand and attendance models to price "
        "sponsorship assets (LED boards, jersey patches, in-venue activations) based on projected "
        "eyeballs per game instead of flat annual rates. "
        "<b>Phase 3 \u2014 Concessions &amp; merchandise optimization:</b> Dynamic pricing and inventory "
        "optimization for F&amp;B based on expected attendance and dwell time. "
        "<b>Phase 4 \u2014 Fan lifetime value model:</b> Predict which single-game buyers will convert to "
        "STH, what price sensitivity they have, and optimize the conversion funnel. "
        "<b>Phase 5 \u2014 Multi-club platform:</b> Package the pipeline as a product for other MLS/USL clubs. "
        "The feature engineering and optimizer are parameterized \u2014 swap in a new club's data and "
        "the system works out of the box.",
        s,
        tip="Show you're thinking beyond one module. VP of Strategy should think in platforms, not projects."
    ))

    story.append(qa_block(
        "How would you upgrade the model to be more accurate over time?",
        "Four concrete upgrades as real data flows in: "
        "<b>(1) Real transaction data retraining</b> \u2014 after each game, actual attendance and secondary "
        "market transactions feed back into the model. XGBoost retrains weekly with expanding window. "
        "MAPE should drop from ~10\u201312% to 6\u20138% by mid-season. "
        "<b>(2) Live secondary market ingestion</b> \u2014 replace batch StubHub snapshots with streaming "
        "SeatGeek/VividSeats APIs. Price signals at T-48hrs and T-24hrs are the most predictive. "
        "<b>(3) Weather API integration</b> \u2014 currently using historical San Diego weather patterns. "
        "Switching to 10-day NWS forecasts for game-specific rain probability would improve the weather "
        "demand impact feature. "
        "<b>(4) Social sentiment from real sources</b> \u2014 currently estimated. Integrating Reddit, "
        "Twitter/X, and team Discord activity as a feature (social NLP scores) gives a 2\u20133% MAPE improvement "
        "based on academic benchmarks.",
        s
    ))

    story.append(qa_block(
        "Can this work for other sports or other clubs?",
        "Yes. The architecture is intentionally modular: "
        "<b>What transfers directly:</b> the three-layer pipeline (demand model \u2192 elasticity \u2192 optimizer), "
        "the guardrail framework, and the dashboard. These are sport-agnostic. "
        "<b>What needs recalibration per club:</b> feature engineering (different sports have different demand "
        "drivers \u2014 NFL weather matters more, NBA back-to-backs matter, etc.), elasticity priors "
        "(premium NBA sections may have different elasticity than MLS), and guardrail thresholds. "
        "<b>What's San Diego-specific:</b> the cross-border features (Tijuana market), military market "
        "scoring, and Baja Cup variables. These would be replaced with local market features for "
        "another city. "
        "The total integration time for a new MLS club would be ~4\u20136 weeks if they have Ticketmaster "
        "and secondary market data accessible.",
        s
    ))

    story.append(qa_block(
        "What team would you need to build around this function?",
        "Lean and scalable: "
        "<b>Year 1 (2 people):</b> VP of Strategy &amp; Analytics (this role \u2014 owns the model, strategy, "
        "and executive communication) + one Pricing Analyst (daily ops, reviews recommendations, "
        "coordinates with ticket ops, monitors model health). "
        "<b>Year 2 (3\u20134 people):</b> Add a Data Engineer (build real-time pipelines, maintain infrastructure) "
        "and a Business Intelligence Analyst (expand dashboard to sponsorship and concessions). "
        "<b>Year 3+ (5\u20136 people):</b> Add a second ML Engineer for real-time model serving and a "
        "Fan Insights Analyst for LTV modeling and conversion optimization. "
        "The philosophy: start small, prove ROI, then invest. Don't build a 10-person team "
        "before you've proven the $1.9M uplift is real.",
        s,
        tip="COOs appreciate hiring discipline. Show you won't ask for a big team on day one."
    ))

    # =====================================================================
    # SECTION 7: PERSONAL / CANDIDATE FIT
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Your Background &amp; Fit", s["SectionHead"]))
    story.append(make_hr())

    story.append(qa_block(
        "Why are you the right person for this role?",
        "Three things make me uniquely suited: "
        "<b>(1) I've been a club CEO.</b> I don't just build models \u2014 I understand the P&amp;L, the "
        "stakeholder dynamics, the fan politics, and the board-level conversations that determine whether "
        "a pricing initiative lives or dies. Most data scientists can build the model; very few can walk "
        "into a board meeting and explain why it won't cause a PR crisis. "
        "<b>(2) Technical depth with CS + MBA.</b> I can go from writing the XGBoost feature pipeline "
        "to presenting the ROI case to investors in the same meeting. This role sits at the intersection "
        "of strategy and analytics \u2014 you need someone who's native in both languages. "
        "<b>(3) This platform is my proof of work.</b> I didn't build a slide deck about what I <i>would</i> do. "
        "I built the entire system \u2014 data pipeline, ML models, causal elasticity estimation, revenue "
        "optimizer, interactive dashboard, and the business case \u2014 end to end. For San Diego FC specifically, "
        "not as a generic exercise. That's the level of ownership I bring.",
        s,
        tip="Be specific about why club CEO experience matters in a VP analytics role \u2014 it's your biggest differentiator."
    ))

    story.append(qa_block(
        "What's the first thing you'd do in the first 90 days?",
        "<b>Days 1\u201330: Listen and validate.</b> Meet with ticket ops, marketing, finance, and the coaching "
        "staff to understand their current process, pain points, and political dynamics. Run the model "
        "in shadow mode on the first 2\u20133 home games. Build relationships before pushing changes. "
        "<b>Days 31\u201360: Quick wins.</b> Present the shadow-mode results \u2014 'here's what the model would "
        "have recommended, and here's the revenue we left on the table.' Get buy-in for an A/B test "
        "on 2\u20133 sections. Stand up the dashboard for ticket ops to play with. "
        "<b>Days 61\u201390: First results.</b> Run the A/B test, measure results, present to leadership. "
        "If positive, propose the Phase 3 rollout plan. Start scoping Phase 2 (sponsorship valuation) "
        "to show the bigger platform vision.",
        s
    ))

    story.append(qa_block(
        "How do you handle disagreements with ticket ops or other departments?",
        "With data, transparency, and respect for their expertise. "
        "If ticket ops says 'the model is wrong on this game \u2014 trust me, this rival fills the house,' "
        "my response is: '<i>Show me what I'm missing.</i>' Maybe there's a local factor the model doesn't "
        "capture yet. That becomes a new feature. "
        "The model is a <b>tool for better decisions, not a replacement for human judgment</b>. If a "
        "20-year ticket ops veteran has a strong conviction that contradicts the model on a specific game, "
        "we go with their call and track the outcome. Over time, the model earns trust by being right "
        "more often than not \u2014 but it never overrides institutional knowledge without a conversation. "
        "The worst thing an analytics leader can do is come in and say 'the model says you're wrong.' "
        "That's how you get fired and how analytics departments get disbanded.",
        s,
        tip="Show emotional intelligence, not just analytical capability. COOs manage people, and they want a VP who can too."
    ))

    # Footer
    story.append(Spacer(1, 12))
    story.append(make_hr())
    story.append(Paragraph(
        "SD FC Pricing Intelligence Platform  |  Interview Prep Document  |  Confidential",
        s["SmallGray"]
    ))

    doc.build(story)
    print(f"Interview Q&A: {path}")
    return path


if __name__ == "__main__":
    build_interview_qa()
