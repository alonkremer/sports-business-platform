# Feature Importance Report — SD FC Pricing Intelligence

**Generated:** 2026-03-27  
**Data:** `data/features/demand_features.parquet` (476 rows × 111 columns)  
**Target (Parts B/C):** `optimal_price_increase`  
**Target (Part A SHAP):** `target_demand_index` (XGBoost demand model)

---

## Part A — SHAP Feature Importances (Existing EMSR Pipeline)

### Pipeline Summary

The demand model (`src/pricing_model/demand_model.py`) is an XGBoost regressor trained on `target_demand_index` using 70 g-prefixed engineered features.

**MLflow runs found:** 5 (3 unique results)

| Run ID | CV MAPE | CV MAE | Folds | Train Rows | Features |
|--------|---------|--------|-------|------------|----------|
| `b2386700920d...` | 5.94% | 0.05316 | 3 | 238 | 70 |
| `91860a8d43dd...` | 9.44% | 0.07013 | 3 | 238 | 70 |
| `d9fec028f4c1...` | 9.45% | 0.07026 | 3 | 238 | 70 |

**Best run:** CV MAPE = **5.94%** (target: <15%) — model is well within spec.

### SHAP Values — Top 50 Features (demand index model, re-trained on full dataset)

*SHAP computed on 200 training samples. Feature group prefix legend: g1=schedule/calendar, g2=opponent quality, g3=match context, g4=secondary market, g5=sell-through velocity, g6=section/tier, g7=local market, g8=weather, g9=competing events, g10=promotions, g11=price benchmarks, g12=historical, g13=social/buzz, g14=cross-border.*

| Rank | Feature | Mean |SHAP| | % of Total | Feature Group |
|------|---------|------------|------------|---------------|
| 1 | `g3_match_significance` | 0.040888 | 56.85% | Match Context |
| 2 | `g2_away_xg` | 0.005835 | 8.11% | Opponent Quality |
| 3 | `g2_xg_diff` | 0.005484 | 7.63% | Opponent Quality |
| 4 | `g4_secondary_premium_pct` | 0.005375 | 7.47% | Secondary Market |
| 5 | `g2_home_xg` | 0.003308 | 4.60% | Opponent Quality |
| 6 | `g1_is_season_opener` | 0.002744 | 3.82% | Schedule/Calendar |
| 7 | `g2_opponent_strength` | 0.002484 | 3.45% | Opponent Quality |
| 8 | `g2_opponent_tier` | 0.002463 | 3.42% | Opponent Quality |
| 9 | `g1_season_progress` | 0.001168 | 1.62% | Schedule/Calendar |
| 10 | `g1_is_saturday` | 0.001011 | 1.41% | Schedule/Calendar |
| 11 | `g1_month` | 0.000620 | 0.86% | Schedule/Calendar |
| 12 | `g1_is_wednesday` | 0.000542 | 0.75% | Schedule/Calendar |
| 13 | `g1_day_of_week_num` | 0.000000 | 0.00% | Schedule/Calendar |
| 14 | `g1_is_baja_cup` | 0.000000 | 0.00% | Schedule/Calendar |
| 15 | `g1_is_decision_day` | 0.000000 | 0.00% | Schedule/Calendar |
| 16 | `g1_is_fifa_adjacent` | 0.000000 | 0.00% | Schedule/Calendar |
| 17 | `g2_home_win_prob` | 0.000000 | 0.00% | Opponent Quality |
| 18 | `g1_season` | 0.000000 | 0.00% | Schedule/Calendar |
| 19 | `g1_week_of_year` | 0.000000 | 0.00% | Schedule/Calendar |
| 20 | `g1_is_sunday` | 0.000000 | 0.00% | Schedule/Calendar |
| 21 | `g1_is_weekend` | 0.000000 | 0.00% | Schedule/Calendar |
| 22 | `g1_is_second_half` | 0.000000 | 0.00% | Schedule/Calendar |
| 23 | `g2_is_west_rival` | 0.000000 | 0.00% | Opponent Quality |
| 24 | `g2_is_underdog` | 0.000000 | 0.00% | Opponent Quality |
| 25 | `g2_is_rivalry` | 0.000000 | 0.00% | Opponent Quality |
| 26 | `g2_is_marquee` | 0.000000 | 0.00% | Opponent Quality |
| 27 | `g2_is_heavy_favorite` | 0.000000 | 0.00% | Opponent Quality |
| 28 | `g2_rivalry_tier1` | 0.000000 | 0.00% | Opponent Quality |
| 29 | `g3_is_cross_border` | 0.000000 | 0.00% | Match Context |
| 30 | `g2_marquee_tier1` | 0.000000 | 0.00% | Opponent Quality |
| 31 | `g3_star_player_on_opp` | 0.000000 | 0.00% | Match Context |
| 32 | `g4_home_win_prob` | 0.000000 | 0.00% | Secondary Market |
| 33 | `g4_is_healthy_market` | 0.000000 | 0.00% | Secondary Market |
| 34 | `g4_is_cold_market` | 0.000000 | 0.00% | Secondary Market |
| 35 | `g4_is_hot_market` | 0.000000 | 0.00% | Secondary Market |
| 36 | `g4_is_warm_market` | 0.000000 | 0.00% | Secondary Market |
| 37 | `g4_market_health_score` | 0.000000 | 0.00% | Secondary Market |
| 38 | `g4_n_transactions` | 0.000000 | 0.00% | Secondary Market |
| 39 | `g4_sold_price_avg` | 0.000000 | 0.00% | Secondary Market |
| 40 | `g4_sold_to_face_ratio` | 0.000000 | 0.00% | Secondary Market |
| 41 | `g5_is_soldout` | 0.000000 | 0.00% | Sell-Through Velocity |
| 42 | `g5_pct_remaining` | 0.000000 | 0.00% | Sell-Through Velocity |
| 43 | `g5_seats_remaining_t7` | 0.000000 | 0.00% | Sell-Through Velocity |
| 44 | `g5_sell_through_t30` | 0.000000 | 0.00% | Sell-Through Velocity |
| 45 | `g5_sell_through_t7` | 0.000000 | 0.00% | Sell-Through Velocity |
| 46 | `g5_velocity_acceleration` | 0.000000 | 0.00% | Sell-Through Velocity |
| 47 | `g5_velocity_t30` | 0.000000 | 0.00% | Sell-Through Velocity |
| 48 | `g5_velocity_t7` | 0.000000 | 0.00% | Sell-Through Velocity |
| 49 | `g6_capacity` | 0.000000 | 0.00% | Section/Tier |
| 50 | `g6_face_base_price` | 0.000000 | 0.00% | Section/Tier |

---

## Part B — Regression on `optimal_price_increase`

**Feature set:** 100 numeric columns (excludes target, identifiers, categoricals)  
**Rows:** 476 (after dropping NaN target)  
**Target stats:** mean=10.2686, std=20.1827, min=-47.3455, max=133.3091

### B.1 — Model Validation Metrics

#### Gradient Boosting Regressor (sklearn)

| Metric | Value |
|--------|-------|
| Train R² | 0.9999 |
| Train RMSE | 0.1922 |
| 5-Fold CV R² (mean) | 0.9869 |
| 5-Fold CV R² (std) | 0.0141 |
| 5-Fold CV RMSE (mean) | 2.1138 |
| 5-Fold CV RMSE (std) | 1.4865 |
| Fold 1 R² | 0.9856 |
| Fold 2 R² | 0.9600 |
| Fold 3 R² | 0.9954 |
| Fold 4 R² | 0.9982 |
| Fold 5 R² | 0.9956 |
| Fold 5 RMSE | 1.0251 |

> **Interpretation:** The model is fitted on synthetic data (price recommendations are deterministic from the pricing engine), so high R² is expected. CV R² measures out-of-fold generalization within the 476-row dataset.

### B.2 — Top 50 Features by Gradient Boosting Importance

*Trained on all 476 rows. Importance = mean impurity decrease across all trees.*

| Rank | Feature | Importance | % of Total | Feature Group |
|------|---------|------------|------------|---------------|
| 1 | `target_price_gap` | 0.369998 | 37.000% | Engineered/Other |
| 2 | `revenue_opp_per_seat` | 0.334534 | 33.453% | Engineered/Other |
| 3 | `sth_resale_margin` | 0.285546 | 28.555% | Engineered/Other |
| 4 | `secondary_premium_pct` | 0.002549 | 0.255% | Engineered/Other |
| 5 | `g4_secondary_premium_pct` | 0.001180 | 0.118% | Secondary Market |
| 6 | `g4_sold_to_face_ratio` | 0.001014 | 0.101% | Secondary Market |
| 7 | `sold_price_avg` | 0.000971 | 0.097% | Engineered/Other |
| 8 | `target_revenue_opp` | 0.000957 | 0.096% | Engineered/Other |
| 9 | `total_revenue_opportunity` | 0.000884 | 0.088% | Engineered/Other |
| 10 | `g4_sold_price_avg` | 0.000601 | 0.060% | Secondary Market |
| 11 | `face_price` | 0.000520 | 0.052% | Engineered/Other |
| 12 | `g4_n_transactions` | 0.000408 | 0.041% | Secondary Market |
| 13 | `g11_price_vs_mls_avg` | 0.000293 | 0.029% | Price Benchmarks |
| 14 | `g11_price_vs_padres` | 0.000165 | 0.016% | Price Benchmarks |
| 15 | `backlash_risk_score` | 0.000112 | 0.011% | Engineered/Other |
| 16 | `g6_face_base_price` | 0.000080 | 0.008% | Section/Tier |
| 17 | `g5_seats_remaining_t7` | 0.000043 | 0.004% | Sell-Through Velocity |
| 18 | `g8_temp_f` | 0.000031 | 0.003% | Weather |
| 19 | `g13_social_sentiment` | 0.000027 | 0.003% | Social/Buzz |
| 20 | `g4_home_win_prob` | 0.000011 | 0.001% | Secondary Market |
| 21 | `capacity` | 0.000011 | 0.001% | Engineered/Other |
| 22 | `g2_away_xg` | 0.000009 | 0.001% | Opponent Quality |
| 23 | `g2_xg_diff` | 0.000008 | 0.001% | Opponent Quality |
| 24 | `g1_week_of_year` | 0.000005 | 0.001% | Schedule/Calendar |
| 25 | `g1_season` | 0.000005 | 0.000% | Schedule/Calendar |
| 26 | `g2_is_heavy_favorite` | 0.000005 | 0.000% | Opponent Quality |
| 27 | `g1_season_progress` | 0.000004 | 0.000% | Schedule/Calendar |
| 28 | `g12_hist_sellthrough` | 0.000004 | 0.000% | Historical |
| 29 | `g10_promo_score` | 0.000003 | 0.000% | Promotions |
| 30 | `g2_home_win_prob` | 0.000003 | 0.000% | Opponent Quality |
| 31 | `g2_marquee_tier1` | 0.000003 | 0.000% | Opponent Quality |
| 32 | `g1_day_of_week_num` | 0.000003 | 0.000% | Schedule/Calendar |
| 33 | `g2_is_marquee` | 0.000002 | 0.000% | Opponent Quality |
| 34 | `g6_tier_ordinal` | 0.000002 | 0.000% | Section/Tier |
| 35 | `g3_match_significance` | 0.000002 | 0.000% | Match Context |
| 36 | `g2_rivalry_tier1` | 0.000001 | 0.000% | Opponent Quality |
| 37 | `g3_star_player_on_opp` | 0.000001 | 0.000% | Match Context |
| 38 | `g8_rain_prob` | 0.000001 | 0.000% | Weather |
| 39 | `g1_month` | 0.000001 | 0.000% | Schedule/Calendar |
| 40 | `g2_home_xg` | 0.000001 | 0.000% | Opponent Quality |
| 41 | `g12_hist_avg_att_same_opp` | 0.000001 | 0.000% | Historical |
| 42 | `g10_has_fireworks` | 0.000001 | 0.000% | Promotions |
| 43 | `g12_hist_demand_idx_opp` | 0.000001 | 0.000% | Historical |
| 44 | `g4_is_warm_market` | 0.000000 | 0.000% | Secondary Market |
| 45 | `g11_padres_equiv_price` | 0.000000 | 0.000% | Price Benchmarks |
| 46 | `g14_cross_border_index` | 0.000000 | 0.000% | Cross-Border |
| 47 | `g13_is_high_buzz` | 0.000000 | 0.000% | Social/Buzz |
| 48 | `target_log_sellthrough` | 0.000000 | 0.000% | Engineered/Other |
| 49 | `target_demand_index` | 0.000000 | 0.000% | Engineered/Other |
| 50 | `g10_has_giveaway` | 0.000000 | 0.000% | Promotions |

### B.3 — XGBoost Cross-Check (Top 20, for validation)

| Rank | Feature | XGB Importance |
|------|---------|----------------|
| 1 | `revenue_opp_per_seat` | 0.713795 |
| 2 | `sth_resale_margin` | 0.140562 |
| 3 | `target_price_gap` | 0.102662 |
| 4 | `target_revenue_opp` | 0.014030 |
| 5 | `total_revenue_opportunity` | 0.007270 |
| 6 | `secondary_premium_pct` | 0.005301 |
| 7 | `g4_sold_to_face_ratio` | 0.004517 |
| 8 | `sold_price_avg` | 0.002911 |
| 9 | `g4_secondary_premium_pct` | 0.001818 |
| 10 | `face_price` | 0.001735 |
| 11 | `g11_price_vs_mls_avg` | 0.001364 |
| 12 | `capacity` | 0.000630 |
| 13 | `g13_is_high_buzz` | 0.000498 |
| 14 | `g10_promo_score` | 0.000489 |
| 15 | `g4_sold_price_avg` | 0.000444 |
| 16 | `g8_temp_f` | 0.000325 |
| 17 | `backlash_risk_score` | 0.000256 |
| 18 | `g3_star_player_on_opp` | 0.000119 |
| 19 | `g6_face_base_price` | 0.000116 |
| 20 | `g4_n_transactions` | 0.000106 |

### B.4 — Top 50 Features by Ridge Regression Coefficient

*Features standardized (mean=0, std=1) before fitting Ridge(alpha=1.0). Coefficients are on the standardized scale — magnitude indicates sensitivity per 1 std dev change.*

| Rank | Feature | Coefficient | Abs Coef | Direction |
|------|---------|-------------|----------|-----------|
| 1 | `sth_resale_margin` | 8.606166 | 8.606166 | positive |
| 2 | `target_price_gap` | 8.606166 | 8.606166 | positive |
| 3 | `sold_price_avg` | 1.835842 | 1.835842 | positive |
| 4 | `g4_sold_price_avg` | 1.835842 | 1.835842 | positive |
| 5 | `revenue_opp_per_seat` | 1.222121 | 1.222121 | positive |
| 6 | `face_price` | -1.163569 | 1.163569 | negative |
| 7 | `g11_price_vs_mls_avg` | -1.163365 | 1.163365 | negative |
| 8 | `g11_padres_equiv_price` | -0.289598 | 0.289598 | negative |
| 9 | `g1_month` | -0.242675 | 0.242675 | negative |
| 10 | `g11_price_vs_padres` | -0.207143 | 0.207143 | negative |
| 11 | `g1_season_progress` | 0.164861 | 0.164861 | positive |
| 12 | `g6_face_base_price` | -0.162796 | 0.162796 | negative |
| 13 | `g1_week_of_year` | 0.154495 | 0.154495 | positive |
| 14 | `target_revenue_opp` | 0.154282 | 0.154282 | positive |
| 15 | `total_revenue_opportunity` | 0.154282 | 0.154282 | positive |
| 16 | `g4_secondary_premium_pct` | 0.147852 | 0.147852 | positive |
| 17 | `secondary_premium_pct` | 0.147852 | 0.147852 | positive |
| 18 | `g4_sold_to_face_ratio` | 0.147852 | 0.147852 | positive |
| 19 | `g1_is_season_opener` | -0.124641 | 0.124641 | negative |
| 20 | `g6_is_premium` | 0.122199 | 0.122199 | positive |
| 21 | `g6_tier_ordinal` | -0.113480 | 0.113480 | negative |
| 22 | `g12_face_price_lag` | -0.111939 | 0.111939 | negative |
| 23 | `g9_is_comic_con_weekend` | -0.110509 | 0.110509 | negative |
| 24 | `g2_rivalry_tier1` | 0.109061 | 0.109061 | positive |
| 25 | `target_demand_index` | -0.100841 | 0.100841 | negative |
| 26 | `g10_has_fireworks` | -0.083421 | 0.083421 | negative |
| 27 | `g2_opponent_strength` | -0.081873 | 0.081873 | negative |
| 28 | `g2_opponent_tier` | 0.081873 | 0.081873 | positive |
| 29 | `g1_season` | 0.081047 | 0.081047 | positive |
| 30 | `g12_hist_sellthrough` | 0.078016 | 0.078016 | positive |
| 31 | `g9_padres_conflict_prob` | 0.075769 | 0.075769 | positive |
| 32 | `backlash_risk_score` | 0.072484 | 0.072484 | positive |
| 33 | `g10_has_giveaway` | 0.061314 | 0.061314 | positive |
| 34 | `g3_star_player_on_opp` | -0.059794 | 0.059794 | negative |
| 35 | `g6_is_upper` | -0.058226 | 0.058226 | negative |
| 36 | `g6_is_midfield` | 0.050063 | 0.050063 | positive |
| 37 | `g3_match_significance` | -0.048090 | 0.048090 | negative |
| 38 | `g12_hist_demand_idx_opp` | 0.047469 | 0.047469 | positive |
| 39 | `g12_hist_avg_att_same_opp` | 0.047019 | 0.047019 | positive |
| 40 | `g8_temp_f` | -0.044817 | 0.044817 | negative |
| 41 | `g2_is_heavy_favorite` | 0.042002 | 0.042002 | positive |
| 42 | `g1_day_of_week_num` | -0.038867 | 0.038867 | negative |
| 43 | `g4_is_healthy_market` | 0.034583 | 0.034583 | positive |
| 44 | `g4_is_hot_market` | -0.033672 | 0.033672 | negative |
| 45 | `g1_is_decision_day` | 0.031524 | 0.031524 | positive |
| 46 | `g1_is_fifa_adjacent` | 0.030845 | 0.030845 | positive |
| 47 | `g1_is_sunday` | 0.024617 | 0.024617 | positive |
| 48 | `g6_is_lower_bowl` | -0.024131 | 0.024131 | negative |
| 49 | `g13_is_high_buzz` | -0.023281 | 0.023281 | negative |
| 50 | `g1_is_wednesday` | -0.023132 | 0.023132 | negative |

---

## Part C — Season Ticket Validation

**Method:** Group by `section`, average all numeric features (including `optimal_price_increase`). This mirrors the dashboard's season ticket aggregation mode.

### C.1 — Season-Level Price Recommendation by Tier

| Tier | Count | Mean Rec | Std Dev | Min Rec | Median Rec | Max Rec |
|------|-------|----------|---------|---------|------------|---------|
| field_club | 34 | 44.8262 | 41.6069 | -47.3455 | 49.2545 | 133.3091 |
| lower_bowl_corner | 136 | 7.3544 | 9.2613 | -17.0190 | 7.9318 | 33.9382 |
| lower_bowl_goal | 34 | 6.4002 | 8.9419 | -12.3003 | 5.5857 | 26.7382 |
| lower_bowl_midfield | 68 | 12.9946 | 14.8809 | -18.3771 | 12.8757 | 48.4655 |
| supporters_ga | 34 | -0.7231 | 1.5972 | -4.2151 | -0.7320 | 2.5564 |
| upper_bowl | 102 | 2.1314 | 4.4642 | -8.9917 | 2.3446 | 13.8600 |
| upper_concourse | 34 | -0.2297 | 2.1849 | -4.9644 | -0.0933 | 4.6327 |
| west_club | 34 | 31.6860 | 31.1752 | -23.9448 | 31.1135 | 109.4864 |

> **Note:** `optimal_price_increase` is expressed as a decimal multiplier (e.g., 0.10 = +10% increase recommended, 0.0 = no change, negative = reduce). Values represent the raw pricing engine output before any floor/ceiling rules applied in the dashboard optimizer.

### C.2 — Season-Level Recommendation by Section

*Table shows aggregated (averaged) recommendation per section when viewed in season ticket mode. Sorted by recommendation (highest first).*

| Section | Tier | Avg Rec (seasonal) | Row Count |
|---------|------|-------------------|-----------|
| FC_C124_C132 | field_club | 44.8262 | 34 |
| WC_C223_C231 | west_club | 31.6860 | 34 |
| LB_111_115 | lower_bowl_midfield | 13.5928 | 34 |
| LB_116_120 | lower_bowl_midfield | 12.3964 | 34 |
| LB_121_123 | lower_bowl_corner | 8.2550 | 34 |
| LB_133_135 | lower_bowl_corner | 7.5545 | 34 |
| LB_141 | lower_bowl_corner | 7.0333 | 34 |
| LB_101_105 | lower_bowl_corner | 6.5745 | 34 |
| LB_106_110 | lower_bowl_goal | 6.4002 | 34 |
| UB_208_212 | upper_bowl | 2.3753 | 34 |
| UB_202_207 | upper_bowl | 2.1348 | 34 |
| UB_235_238 | upper_bowl | 1.8841 | 34 |
| UC_323_334 | upper_concourse | -0.2297 | 34 |
| GA_136_140 | supporters_ga | -0.7231 | 34 |

### C.3 — Validation Assessment

| Check | Result | Status |
|-------|--------|--------|
| All section averages in valid range (-50% to +100%) | -0.7231 to 44.8262 | WARN |
| Section-to-section variation (std > 0.01) | 12.8566 | PASS |
| No null/missing recommendations | 0 nulls | PASS |
| Recommendations differ meaningfully across tiers | std=16.5571 | PASS |
| Averaging is sensible (mean across games ≈ season-level signal) | 10.2686 overall avg | PASS — linear pricing engine |

> **Conclusion:** Season ticket averaging is statistically sound for this pricing engine because `optimal_price_increase` is computed as a continuous regression output (not a discrete action), so the mean across games produces a meaningful season-level price adjustment signal per section.

---

*Report generated by `run_feature_analysis.py` — SD FC Pricing Intelligence Platform*