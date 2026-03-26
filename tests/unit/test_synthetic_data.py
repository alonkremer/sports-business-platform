"""
Unit tests: synthetic data generator calibration and data quality.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_generation.synthetic_sdfc import (
    SNAPDRAGON_CAPACITY,
    VERIFIED_2025,
    generate_games_2025,
    generate_inventory_snapshots,
    generate_secondary_market_consolidated,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def games_2025(rng):
    return generate_games_2025(rng)


class TestCalibration:
    """Verify synthetic data matches 2025 verified actuals."""

    def test_2025_game_count(self, games_2025):
        assert len(games_2025) == 17, f"Expected 17 home games, got {len(games_2025)}"

    def test_2025_avg_attendance_target(self, games_2025):
        """2025 average attendance must be 28,064 ±300."""
        avg = games_2025["attendance"].mean()
        assert abs(avg - 28_064) < 300, f"Avg attendance {avg:.0f} outside target ±300"

    def test_opener_attendance(self, games_2025):
        """Home opener (St. Louis City SC) must be 34,506."""
        opener = games_2025[games_2025["game_id"] == "2025_H01"]
        assert len(opener) == 1, "Opener game not found"
        att = opener.iloc[0]["attendance"]
        assert abs(att - 34_506) < 100, f"Opener attendance {att} should be 34,506"

    def test_min_attendance(self, games_2025):
        """Min attendance must be ~21,872 (Pachuca, midweek loss)."""
        min_att = games_2025["attendance"].min()
        assert abs(min_att - 21_872) < 200, f"Min attendance {min_att} should be ~21,872"

    def test_attendance_range_realistic(self, games_2025):
        """All attendances must be within [15K, 35K] for SD FC context."""
        assert games_2025["attendance"].min() >= 15_000
        assert games_2025["attendance"].max() <= SNAPDRAGON_CAPACITY + 100

    def test_demand_index_range(self, games_2025):
        """Demand index must be in [0, 1]."""
        assert games_2025["demand_index"].between(0, 1).all()

    def test_opponent_count(self, games_2025):
        """Must have 17 unique game IDs."""
        assert games_2025["game_id"].nunique() == 17

    def test_all_seasons_2025(self, games_2025):
        """All rows must be season 2025."""
        assert (games_2025["season"] == 2025).all()


class TestInventorySnapshots:
    """Test inventory snapshot generation."""

    def test_snapshot_days(self, games_2025, rng):
        inv = generate_inventory_snapshots(games_2025, rng)
        expected_days = {30, 14, 7, 3, 1}
        actual_days = set(inv["days_before_game"].unique())
        assert expected_days == actual_days, f"Expected {expected_days}, got {actual_days}"

    def test_sell_through_range(self, games_2025, rng):
        inv = generate_inventory_snapshots(games_2025, rng)
        assert inv["sell_through_pct"].between(0, 100).all()

    def test_seats_remaining_non_negative(self, games_2025, rng):
        inv = generate_inventory_snapshots(games_2025, rng)
        assert (inv["seats_remaining"] >= 0).all()

    def test_inventory_monotone(self, games_2025, rng):
        """Seats sold must be non-decreasing as game approaches (lower days_before)."""
        inv = generate_inventory_snapshots(games_2025, rng)
        for (gid, sec), grp in inv.groupby(["game_id", "section"]):
            grp_sorted = grp.sort_values("days_before_game", ascending=False)
            assert (grp_sorted["seats_sold"].diff().dropna() >= 0).all(), \
                f"Non-monotone inventory for {gid}/{sec}"


class TestSecondaryMarket:
    """Test secondary market data generation."""

    def test_secondary_premium_range(self, games_2025, rng):
        sec = generate_secondary_market_consolidated(games_2025, rng)
        assert sec["secondary_premium_pct"].between(-20, 120).all()

    def test_market_health_values(self, games_2025, rng):
        sec = generate_secondary_market_consolidated(games_2025, rng)
        valid_health = {"cold", "healthy", "warm", "hot"}
        assert set(sec["market_health"].unique()).issubset(valid_health)

    def test_baja_cup_premium(self, games_2025, rng):
        """Baja Cup game should have higher secondary premium than average."""
        sec = generate_secondary_market_consolidated(games_2025, rng)
        baja = sec[sec["opponent"] == "Club Tijuana"]["secondary_premium_pct"].mean()
        overall_avg = sec["secondary_premium_pct"].mean()
        assert baja > overall_avg, "Baja Cup should have above-average secondary premium"

    def test_sold_price_above_face(self, games_2025, rng):
        """For most games, secondary sold price should exceed face price."""
        sec = generate_secondary_market_consolidated(games_2025, rng)
        pct_above = (sec["sold_price_avg"] > sec["face_price"]).mean()
        assert pct_above > 0.60, f"Only {pct_above:.0%} of sold prices above face (expected >60%)"
