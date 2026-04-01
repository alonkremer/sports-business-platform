# SD FC Pricing Intelligence Platform
## Executive Summary — One Pager

---

**The Problem**

San Diego FC completed its inaugural 2025 season averaging 28,064 fans per game in a 35,000-seat Snapdragon Stadium. Ticket prices were set once before the season using gut feel and competitor benchmarks. The secondary market told a different story: Baja Cup matches against Club Tijuana cleared at 40-58% above face value, while midweek games against weaker opponents sat below face. An estimated $207,000 in revenue was left on the table in Year 1 — and that number grows as the market matures.

**The Solution**

A machine learning platform that recommends optimal ticket prices for every section and every game, grounded in what fans are actually willing to pay. It combines demand forecasting (XGBoost, 8-14% error rate), causal price elasticity (Double Machine Learning), and airline-style yield optimization (EMSR-b) — all calibrated against verified 2025 attendance data.

**How It Works**

The platform ingests data from 9 sources (secondary market platforms, betting odds, weather, attendance records) and outputs three pricing scenarios per section per game:

- Conservative — minimal price movement, targets 90%+ fill rate, lowest risk
- Balanced (recommended) — optimizes revenue while protecting season ticket holder resale margins
- Aggressive — maximum revenue extraction for confirmed high-demand games only

Every recommendation includes a plain-English explanation: "Section 111 is $8 below optimal. Drivers: rivalry game (+$8), high secondary demand (+$5), Saturday night (+$3)."

**What Makes It Different**

1. Causal, not correlated — isolates the true effect of price on demand, controlling for opponent quality, weather, and timing
2. Explainable — every recommendation comes with reasons your ticketing team can understand and challenge
3. Guardrailed — STH resale margins are protected by design (primary price never exceeds secondary minus 10%), and any increase above 25% is flagged for manual review
4. Three choices, not one — management picks the risk level, the model handles the math

**Key Numbers**

- 2025 verified season average: 28,064 attendance (calibration anchor)
- Demand forecast accuracy: 8-14% MAPE (target: under 15%)
- 2025 retrospective revenue opportunity: ~$207,000
- Balanced scenario projected uplift: 9-14% vs flat pricing
- STH resale margin maintained at 10-15% (healthy zone)

**Implementation Path**

Phase 1 (now): Offline recommendations. Run the model before each pricing window, present scenarios to VP of Ticketing, manually update prices in Archtics. Zero integration required.

Phase 2 (3-6 months): API integration with Ticketmaster Host for automated price updates with human approval gate.

Phase 3 (6-12 months): Real-time feedback loop with live sell-through tracking, mid-season recalibration, and weekly pricing reviews.

**What This Role Delivers in 90 Days**

Days 1-30: Replace synthetic data with real Archtics/CRM feeds. Validate model accuracy on real transaction data.

Days 31-60: First live pricing recommendations for early-season home games. A/B test (control vs model-priced sections). Dashboard live for ticketing team.

Days 61-90: First measurable results. Revenue uplift report. STH health validation. Phase 2 roadmap presented.

---

Built with: Python, XGBoost, DuckDB, Streamlit, FastAPI, SHAP
Data: FBref, SeatGeek, StubHub, Vivid Seats, OddsPortal, MLS official, Ticketmaster (section template)
