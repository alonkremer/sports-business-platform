# SD FC Pricing Intelligence — User Guide
## How to Read the Dashboard and Draw Insights

---

### Getting Started

Open the dashboard at localhost:8501. The sidebar on the left has four views and a Strategic Mode selector. Start with the default: Balanced scenario, Revenue Optimization mode.


### View 1: Seat Map — "Where are the opportunities right now?"

This is your starting point for any game. Select a game from the dropdown. The stadium map lights up:

- Red sections = underpriced, raise price (secondary market is well above face)
- Light red/neutral sections = moderately underpriced (room to move)
- Gray sections = priced about right
- Blue sections = overpriced or low demand, consider lowering price

The Game Impact Score (1-10) at the top tells you how hot this game is overall.

EXAMPLE INSIGHT: You select "Sep 27 vs Club Tijuana (Baja Cup)". The map is mostly red and orange. Game Impact Score: 9.2. Lower bowl midfield (sections 111-115) shows face price $75, optimal $91, gap of $16 per seat. This is your highest-opportunity game — switch to Aggressive scenario using the radio buttons at the top. The map updates: now the optimal prices reflect maximum extraction. Field club sections show $180 face vs $245 secondary — even aggressive pricing at $220 keeps STH margins healthy.

ACTION: Flag this game for aggressive pricing. Present the three scenarios side by side to the VP of Ticketing.


### View 2: Pricing Workshop — "What happens if I change this price?"

Select a game and section. The slider lets you drag the price up or down. Four metrics update in real time:

- Projected Attendance for that section
- Section Revenue (price x expected seats sold)
- STH Resale Margin (does a season ticket holder profit if they resell?)
- Sell-through percentage

The demand curve chart shows your proposed price as a red dashed line against the revenue curve. The STH safe ceiling is a blue dashed line — go above it and you're hurting your season ticket holders.

EXAMPLE INSIGHT: You pick "Jun 21 vs Nashville" (a midweek game, cold market). Current face: $55 for upper bowl. You slide to $48. Projected sell-through jumps from 68% to 79%. Revenue actually increases because you're filling 385 more seats. The STH margin card stays green. This is a case where lowering the price makes more money — the Conservative scenario would have caught this automatically.

ACTION: For cold-market games, compare Conservative vs Balanced. Conservative often recommends a small price reduction that fills more seats and generates more total revenue.


### View 3: STH Value Dashboard — "Can I prove season tickets are worth it?"

This view is for your renewal campaign. It shows the average resale profit a season ticket holder can expect per game, broken down by section tier.

The bar chart shows margin by game — tall green bars mean high resale value (hot games), short or red bars mean the STH breaks even or loses (cold games). The overall message: across 17 games, your season ticket is worth more than you paid.

EXAMPLE INSIGHT: You filter to "lower_bowl_midfield". The dashboard shows: average season resale value $142 per seat above face across 17 games, average margin 11.2%. The blockquote at the bottom gives you copy-paste renewal language: "Your season tickets in the lower bowl consistently trade at 10-15% above face value on the secondary market."

But you notice 3 games (midweek, low-demand opponents) show negative margins. The Balanced pricing scenario already accounts for this — it prices those games slightly below face to protect STH value.

ACTION: Use the STH Value Statement in renewal emails. Filter by tier to customize messaging for each section group.


### View 4: Performance Report — "Is this actually working?"

This is your accountability view. It shows:

- Model accuracy (MAPE) — how close demand predictions were to reality
- Revenue uplift — how Balanced pricing compares to flat pricing
- Market health distribution — what percentage of section-games are hot vs cold
- 2025 retrospective — money left on table last season, broken down by game

EXAMPLE INSIGHT: The 2025 retrospective bar chart shows the Club Tijuana home match had the largest revenue opportunity at ~$38,000 for that single game. The LAFC rivalry match was second at ~$29,000. These are the games where aggressive pricing in 2026 has the strongest evidence. Meanwhile, the midweek Pachuca match shows near-zero opportunity — it was correctly priced or underdemanded. Conservative pricing is the right call for similar 2026 games.

ACTION: Use the retrospective to justify the pricing strategy to leadership. "$207K left on the table in Year 1" is the headline. The game-by-game breakdown shows exactly where it came from.


### Strategic Modes (Sidebar)

The dropdown at the bottom of the sidebar changes the guardrail behavior across all views:

- Revenue Optimization (default) — pure revenue maximization within STH guardrails
- Fan Acquisition — caps all increases at +10% above face, prioritizes filling seats for a growing fanbase
- Atmosphere (Sellout) — targets 90%+ sell-through on every game, prices down aggressively on cold games

EXAMPLE: Early in the season when you're still building the San Diego market, switch to Fan Acquisition. The seat map will show more blue (moderate) and less red (aggressive). As the brand matures and STH base grows, switch to Revenue Optimization.


### Quick Reference: Daily Workflow

1. Open the Seat Map. Check which upcoming games have red/orange sections.
2. For hot games: open the Pricing Workshop, experiment with Balanced and Aggressive prices, confirm STH margins stay healthy.
3. For cold games: check if Conservative recommends a price reduction that improves sell-through.
4. Before renewal season: pull up the STH Value Dashboard, export the resale margin data by tier.
5. Monthly: review the Performance Report to track model accuracy and cumulative revenue uplift.
