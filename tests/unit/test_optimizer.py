"""
Unit tests: EMSR-b optimizer logic and strategic guardrails.
"""
import numpy as np
import pytest

from src.price_optimizer.emsr_optimizer import (
    DEFAULT_GUARDRAILS,
    StrategicGuardrails,
    _apply_guardrails,
    _build_price_grid,
    _demand_curve,
    _emsr_b_optimal,
    optimize_section_game,
)


class TestPriceGrid:
    def test_grid_range(self):
        grid = _build_price_grid(100.0)
        assert grid.min() >= 70.0  # 100 * (1 - 0.30)
        assert grid.max() <= 130.5  # 100 * (1 + 0.30) + small float

    def test_grid_step(self):
        grid = _build_price_grid(100.0)
        diffs = np.diff(grid)
        assert np.allclose(diffs, diffs[0], atol=0.5), "Grid step not uniform"

    def test_face_price_in_grid(self):
        """Face price must appear in the grid."""
        grid = _build_price_grid(75.0)
        assert any(abs(grid - 75.0) < 0.5)


class TestDemandCurve:
    def test_demand_decreases_with_price(self):
        prices = np.array([50, 60, 70, 80, 90])
        demand = _demand_curve(prices, 60.0, 0.80, -0.70, 2000)
        assert (np.diff(demand) < 0).all(), "Demand must decrease as price increases"

    def test_demand_at_face_price(self):
        """At face price, demand should equal base demand."""
        prices = np.array([75.0])
        demand = _demand_curve(prices, 75.0, 0.80, -0.70, 2000)
        assert abs(demand[0] - 0.80) < 0.01

    def test_demand_bounds(self):
        prices = np.linspace(10, 300, 50)
        demand = _demand_curve(prices, 75.0, 0.80, -0.70, 2000)
        assert (demand >= 0).all()
        assert (demand <= 1).all()


class TestEmsrOptimal:
    def test_optimal_price_positive(self):
        grid = _build_price_grid(75.0)
        demand = _demand_curve(grid, 75.0, 0.80, -0.70, 1600)
        result = _emsr_b_optimal(grid, demand, 1600)
        assert result["optimal_price"] > 0

    def test_balanced_generates_positive_revenue(self):
        grid = _build_price_grid(75.0)
        demand = _demand_curve(grid, 75.0, 0.80, -0.70, 1600)
        result = _emsr_b_optimal(grid, demand, 1600)
        assert result["expected_revenue"] > 0

    def test_aggressive_higher_price_than_conservative(self):
        """Aggressive scenario should yield a higher price than conservative."""
        from src.price_optimizer.emsr_optimizer import optimize_section_game
        result = optimize_section_game(
            game_id="test", section="lower_bowl_midfield", tier="lower_bowl_midfield",
            face_price=75.0, base_demand=0.80, elasticity=-0.70,
            capacity=1600, sold_price_avg=100.0, sth_resale_margin=25.0,
        )
        conservative_price = result["scenarios"]["conservative"]["price"]
        aggressive_price   = result["scenarios"]["aggressive"]["price"]
        assert aggressive_price >= conservative_price


class TestStrategicGuardrails:
    def test_floor_enforced(self):
        guardrails = StrategicGuardrails()
        guardrails.price_floors["lower_bowl_midfield"] = 60.0
        price, applied = _apply_guardrails(
            price=45.0, tier="lower_bowl_midfield", face_price=75.0,
            sold_price_avg=100.0, guardrails=guardrails, scenario="conservative"
        )
        assert price >= 60.0
        assert "floor_lower_bowl_midfield" in applied

    def test_ceiling_enforced(self):
        guardrails = StrategicGuardrails()
        guardrails.price_ceilings["lower_bowl_midfield"] = 120.0
        price, applied = _apply_guardrails(
            price=200.0, tier="lower_bowl_midfield", face_price=75.0,
            sold_price_avg=250.0, guardrails=guardrails, scenario="aggressive"
        )
        assert price <= 120.0
        assert "ceiling_lower_bowl_midfield" in applied

    def test_sth_resale_protection(self):
        """STH resale guarantee: primary must stay ≤ secondary - 10%."""
        guardrails = StrategicGuardrails(sth_resale_min_margin_pct=10.0)
        price, applied = _apply_guardrails(
            price=100.0, tier="lower_bowl_midfield", face_price=75.0,
            sold_price_avg=95.0, guardrails=guardrails, scenario="aggressive"
        )
        # Max safe price = 95 * (1 - 0.10) = 85.5
        assert price <= 85.5 + 1.0  # +1 for rounding
        assert "sth_resale_protection" in applied

    def test_acquisition_mode_cap(self):
        guardrails = StrategicGuardrails(strategic_mode="acquisition", acquisition_mode_cap=10.0)
        price, applied = _apply_guardrails(
            price=120.0, tier="lower_bowl_midfield", face_price=75.0,
            sold_price_avg=150.0, guardrails=guardrails, scenario="balanced"
        )
        assert price <= 75.0 * 1.10 + 2.0  # +2 for rounding
        assert "acquisition_mode_cap" in applied


class TestOptimizeSectionGame:
    def test_returns_three_scenarios(self):
        result = optimize_section_game(
            game_id="2026_H03", section="LB_111_115", tier="lower_bowl_midfield",
            face_price=75.0, base_demand=0.85, elasticity=-0.70,
            capacity=1600, sold_price_avg=97.0, sth_resale_margin=22.0,
        )
        assert set(result["scenarios"].keys()) == {"conservative", "balanced", "aggressive"}

    def test_balanced_recommended(self):
        result = optimize_section_game(
            game_id="2026_H03", section="LB_111_115", tier="lower_bowl_midfield",
            face_price=75.0, base_demand=0.85, elasticity=-0.70,
            capacity=1600, sold_price_avg=97.0, sth_resale_margin=22.0,
        )
        assert result["recommended"] == "balanced"

    def test_sth_healthy_balanced(self):
        """Balanced scenario should preserve STH health when secondary is healthy."""
        result = optimize_section_game(
            game_id="2026_H03", section="LB_111_115", tier="lower_bowl_midfield",
            face_price=75.0, base_demand=0.85, elasticity=-0.70,
            capacity=1600, sold_price_avg=97.0, sth_resale_margin=22.0,
        )
        balanced = result["scenarios"]["balanced"]
        assert balanced["sth_resale_margin"] >= 0, "STH should not be underwater in balanced scenario"

    def test_price_positive_all_scenarios(self):
        result = optimize_section_game(
            game_id="2026_H01", section="UC_323_334", tier="upper_concourse",
            face_price=32.0, base_demand=0.65, elasticity=-1.10,
            capacity=6000, sold_price_avg=36.0, sth_resale_margin=4.0,
        )
        for scenario_name, scenario in result["scenarios"].items():
            assert scenario["price"] > 0, f"Scenario {scenario_name} has non-positive price"
