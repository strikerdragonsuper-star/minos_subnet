"""Comprehensive tests for utils.weight_tracking.ScoreTracker."""

import pytest

from utils.weight_tracking import (
    ScoreTracker,
    PARTICIPATION_WINDOW,
    WARMUP_WEIGHTS,
    SCORE_EPSILON,
    DEFAULT_BURN_RATE,
    DEFAULT_WINNER_WEIGHT,
    DEFAULT_DUST_TOP_N,
    DEFAULT_DUST_DECAY,
    CANONICAL_TIEBREAK_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eligible(tracker: ScoreTracker, hotkey: str, score: float = 0.80):
    """Shortcut: update + record enough rounds to make *hotkey* eligible.

    WARNING: calling this for multiple hotkeys on the same tracker causes
    decay and window-trim side-effects. Use _make_eligible_group instead.
    """
    for i in range(tracker.min_rounds):
        tracker.update(hotkey, score)
        tracker.record_round(f"elig-{hotkey}-{i}", [hotkey])


def _make_eligible_group(
    tracker: ScoreTracker,
    hotkey_scores: dict,
):
    """Make several miners eligible in the *same* rounds (avoids cross-decay).

    Args:
        hotkey_scores: {hotkey: raw_score, ...}
    """
    hotkeys = list(hotkey_scores.keys())
    for i in range(tracker.min_rounds):
        for hk, score in hotkey_scores.items():
            tracker.update(hk, score)
        tracker.record_round(f"elig-group-{i}", hotkeys)


def _make_active(tracker: ScoreTracker, hotkey: str, rounds: int = 1,
                 score: float = 0.80):
    """Record *rounds* rounds for *hotkey* (not necessarily eligible)."""
    for i in range(rounds):
        tracker.update(hotkey, score)
        tracker.record_round(f"active-{hotkey}-{i}", [hotkey])


def _seed_eligible_emas(tracker: ScoreTracker, hotkey_emas: dict):
    """Seed exact EMA values and eligibility for tolerance-boundary tests."""
    hotkeys = list(hotkey_emas.keys())
    for hk, ema in hotkey_emas.items():
        tracker.ema_scores[hk] = ema

    # Add participation history without changing the seeded EMA values.
    for i in range(tracker.min_rounds):
        tracker.round_history.append({
            "round_id": f"seed-{i}",
            "scored_hotkeys": list(hotkeys),
            "active_hotkeys": list(hotkeys),
        })
    tracker._recalculate_participation()


DEFAULT_REWARD_POLICY = {
    "burn_rate": DEFAULT_BURN_RATE,
    "winner_weight": DEFAULT_WINNER_WEIGHT,
    "dust_top_n": DEFAULT_DUST_TOP_N,
    "dust_decay": DEFAULT_DUST_DECAY,
}


# =========================================================================
# TestScoreTrackerInit
# =========================================================================

class TestScoreTrackerInit:
    """Verify constructor defaults and env-var overrides."""

    def test_default_values(self):
        tracker = ScoreTracker(alpha=0.1, decay_factor=0.95)
        assert tracker.alpha == 0.1
        assert tracker.decay_factor == 0.95
        assert tracker.min_rounds == 10  # module-level default

    def test_custom_values(self):
        tracker = ScoreTracker(alpha=0.25, min_rounds=5, decay_factor=0.90)
        assert tracker.alpha == 0.25
        assert tracker.min_rounds == 5
        assert tracker.decay_factor == 0.90

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("EMA_ALPHA", "0.42")
        monkeypatch.setenv("EMA_DECAY_FACTOR", "0.88")
        tracker = ScoreTracker(alpha=None, decay_factor=None)
        assert tracker.alpha == pytest.approx(0.42)
        assert tracker.decay_factor == pytest.approx(0.88)


# =========================================================================
# TestUpdate
# =========================================================================

class TestUpdate:
    """EMA update semantics."""

    def test_first_update_from_zero(self, score_tracker):
        """First update: EMA starts at 0, so new = alpha * score."""
        ema = score_tracker.update("hk_a", 1.0)
        assert ema == pytest.approx(0.1 * 1.0)

    def test_second_update(self, score_tracker):
        """Second update applies the full EMA formula."""
        score_tracker.update("hk_a", 1.0)
        ema2 = score_tracker.update("hk_a", 0.5)
        expected = (1 - 0.1) * 0.1 + 0.1 * 0.5  # 0.09 + 0.05 = 0.14
        assert ema2 == pytest.approx(expected)

    def test_converges_toward_repeated_score(self, score_tracker):
        """Repeated identical scores should make EMA converge to that value."""
        for _ in range(200):
            ema = score_tracker.update("hk_a", 0.75)
        assert ema == pytest.approx(0.75, abs=1e-4)

    def test_score_of_zero_decays_ema(self, score_tracker):
        """Feeding score=0 should shrink the EMA toward 0."""
        score_tracker.update("hk_a", 1.0)  # EMA = 0.1
        ema = score_tracker.update("hk_a", 0.0)
        assert ema == pytest.approx(0.9 * 0.1)
        assert ema < 0.1

    def test_last_raw_scores_set(self, score_tracker):
        """update() must record the raw score for reporting."""
        score_tracker.update("hk_a", 0.77)
        assert score_tracker.last_raw_scores["hk_a"] == pytest.approx(0.77)
        score_tracker.update("hk_a", 0.55)
        assert score_tracker.last_raw_scores["hk_a"] == pytest.approx(0.55)


# =========================================================================
# TestRecordRound
# =========================================================================

class TestRecordRound:
    """Round recording, decay, dedup, and window trimming."""

    def test_single_round_participation(self, score_tracker):
        score_tracker.update("hk_a", 0.5)
        score_tracker.record_round("r1", ["hk_a"])
        assert score_tracker.get_participation_count("hk_a") == 1

    def test_duplicate_round_is_idempotent(self, score_tracker):
        score_tracker.update("hk_a", 0.5)
        score_tracker.record_round("r1", ["hk_a"])
        score_tracker.record_round("r1", ["hk_a"])  # same id
        assert score_tracker.get_participation_count("hk_a") == 1
        assert len(score_tracker.round_history) == 1

    def test_decay_applied_to_absent_miners(self, score_tracker):
        score_tracker.update("hk_a", 1.0)  # EMA = 0.1
        score_tracker.update("hk_b", 1.0)  # EMA = 0.1
        # Only hk_a scored in this round → hk_b should decay
        score_tracker.record_round("r1", ["hk_a"])
        assert score_tracker.ema_scores["hk_b"] == pytest.approx(0.1 * 0.95)
        # hk_a should be unchanged
        assert score_tracker.ema_scores["hk_a"] == pytest.approx(0.1)

    def test_miner_removed_when_ema_below_threshold(self, score_tracker):
        """Miners whose EMA decays below 1e-6 get pruned."""
        score_tracker.ema_scores["hk_ghost"] = 1e-6  # just at boundary
        score_tracker.record_round("r1", [])  # hk_ghost absent
        assert "hk_ghost" not in score_tracker.ema_scores

    def test_window_trimming_at_21_rounds(self, score_tracker):
        """After 21 rounds, only the last 20 should remain."""
        for i in range(21):
            score_tracker.update("hk_a", 0.5)
            score_tracker.record_round(f"r{i}", ["hk_a"])
        assert len(score_tracker.round_history) == PARTICIPATION_WINDOW

    def test_participation_recalculated_after_trim(self, score_tracker):
        """Miner present only in the oldest round loses that count after trim."""
        # hk_b scored only in round 0
        score_tracker.update("hk_b", 0.5)
        score_tracker.update("hk_a", 0.5)
        score_tracker.record_round("r0", ["hk_a", "hk_b"])
        # Rounds 1..20 only have hk_a
        for i in range(1, 21):
            score_tracker.update("hk_a", 0.5)
            score_tracker.record_round(f"r{i}", ["hk_a"])
        # r0 has been trimmed; hk_b should have 0 participation
        assert score_tracker.get_participation_count("hk_b") == 0
        assert score_tracker.get_participation_count("hk_a") == PARTICIPATION_WINDOW

    def test_decay_factor_one_means_no_decay(self):
        tracker = ScoreTracker(alpha=0.1, min_rounds=10, decay_factor=1.0)
        tracker.update("hk_a", 1.0)  # EMA = 0.1
        tracker.record_round("r1", [])  # hk_a absent
        assert tracker.ema_scores["hk_a"] == pytest.approx(0.1)

    def test_empty_scored_hotkeys_decays_all(self, score_tracker):
        score_tracker.update("hk_a", 1.0)
        score_tracker.update("hk_b", 1.0)
        score_tracker.record_round("r1", [])
        assert score_tracker.ema_scores["hk_a"] == pytest.approx(0.1 * 0.95)
        assert score_tracker.ema_scores["hk_b"] == pytest.approx(0.1 * 0.95)


# =========================================================================
# TestEligibility
# =========================================================================

class TestEligibility:
    """Participation-based eligibility gating (min_rounds=10)."""

    def test_below_threshold(self, score_tracker):
        for i in range(9):
            score_tracker.update("hk_a", 0.5)
            score_tracker.record_round(f"r{i}", ["hk_a"])
        assert not score_tracker.is_eligible("hk_a")

    def test_at_threshold(self, score_tracker):
        for i in range(10):
            score_tracker.update("hk_a", 0.5)
            score_tracker.record_round(f"r{i}", ["hk_a"])
        assert score_tracker.is_eligible("hk_a")

    def test_above_threshold(self, score_tracker):
        for i in range(15):
            score_tracker.update("hk_a", 0.5)
            score_tracker.record_round(f"r{i}", ["hk_a"])
        assert score_tracker.is_eligible("hk_a")

    def test_unknown_hotkey(self, score_tracker):
        assert score_tracker.get_participation_count("hk_unknown") == 0
        assert not score_tracker.is_eligible("hk_unknown")


# =========================================================================
# TestWinnerHeavyPruningDust
# =========================================================================

class TestWinnerHeavyPruningDust:
    """Production normal mode: winner-heavy with pruning dust for top ranks."""

    def test_top_ten_distribution(self, score_tracker):
        scores = {
            f"hk_{i}": 1.0 - (i * 0.01)
            for i in range(1, 12)
        }
        _make_eligible_group(score_tracker, scores)

        hotkeys = list(scores.keys())
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            hotkeys, **DEFAULT_REWARD_POLICY
        )

        assert weights["hk_1"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert weights["hk_11"] == pytest.approx(0.0)

        dust_pool = 1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        dust_raw = [DEFAULT_DUST_DECAY ** i for i in range(DEFAULT_DUST_TOP_N - 1)]
        dust_total = sum(dust_raw)
        for rank in range(2, 11):
            expected = dust_pool * dust_raw[rank - 2] / dust_total
            assert weights[f"hk_{rank}"] == pytest.approx(expected)

        assert sum(weights.values()) == pytest.approx(1.0 - DEFAULT_BURN_RATE)

    def test_default_top_ten_dust_survives_u16_emit_rounding(self, score_tracker):
        scores = {
            f"hk_{i}": 1.0 - (i * 0.01)
            for i in range(1, 11)
        }
        _make_eligible_group(score_tracker, scores)

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            list(scores.keys()), **DEFAULT_REWARD_POLICY
        )
        burn_weight = 1.0 - sum(weights.values())
        max_weight = max(burn_weight, max(weights.values()))

        for rank in range(2, 11):
            encoded = round(weights[f"hk_{rank}"] / max_weight * 65535)
            assert encoded > 0

    def test_fewer_than_ten_eligible_renormalizes_dust(self, score_tracker):
        _make_eligible_group(
            score_tracker,
            {"hk_a": 0.90, "hk_b": 0.80, "hk_c": 0.70},
        )

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b", "hk_c"], **DEFAULT_REWARD_POLICY
        )

        dust_pool = 1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        dust_total = 1.0 + DEFAULT_DUST_DECAY
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert weights["hk_b"] == pytest.approx(dust_pool / dust_total)
        assert weights["hk_c"] == pytest.approx(dust_pool * DEFAULT_DUST_DECAY / dust_total)
        assert sum(weights.values()) == pytest.approx(1.0 - DEFAULT_BURN_RATE)

    def test_single_eligible_keeps_winner_weight_only(self, score_tracker):
        _make_eligible(score_tracker, "hk_a", score=0.90)

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a"], **DEFAULT_REWARD_POLICY
        )

        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert sum(weights.values()) == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_ineligible_high_ema_gets_zero(self, score_tracker):
        _make_eligible(score_tracker, "hk_a", score=0.80)
        _make_active(score_tracker, "hk_b", rounds=1, score=0.99)

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"], **DEFAULT_REWARD_POLICY
        )

        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert weights["hk_b"] == pytest.approx(0.0)

    def test_all_eligible_zero_ema_returns_all_zero(self, score_tracker):
        for i in range(score_tracker.min_rounds):
            score_tracker.record_round(f"r{i}", ["hk_a", "hk_b"])

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"], **DEFAULT_REWARD_POLICY
        )

        assert weights["hk_a"] == pytest.approx(0.0)
        assert weights["hk_b"] == pytest.approx(0.0)

    def test_warmup_scaled_to_non_burn_budget(self, score_tracker):
        _make_active(score_tracker, "hk_a", rounds=1, score=0.90)
        _make_active(score_tracker, "hk_b", rounds=1, score=0.70)

        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"], **DEFAULT_REWARD_POLICY
        )

        miner_budget = 1.0 - DEFAULT_BURN_RATE
        assert weights["hk_a"] == pytest.approx(miner_budget * 0.5 / 0.8)
        assert weights["hk_b"] == pytest.approx(miner_budget * 0.3 / 0.8)
        assert sum(weights.values()) == pytest.approx(miner_budget)


# =========================================================================
# TestCanonicalTiebreak
# =========================================================================

class TestCanonicalTiebreak:
    """Canonical-ranking tiebreak when local leaders are within tolerance.

    The canonical ranking comes from the platform's /scoring/canonical-ranking
    endpoint. The validator scans it for the first locally eligible candidate
    and uses that as a tiebreaker only inside the tolerance band.
    """

    def test_canonical_used_when_within_tolerance(self, score_tracker):
        # Local rank-1 = hk_a (0.700 EMA), rank-2 = hk_b (0.695 EMA).
        # Gap 0.5%, within 2% tolerance. Canonical says hk_b, so hk_b wins.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.695})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_b",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_b"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        dust_pool = 1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        assert weights["hk_a"] == pytest.approx(dust_pool)

    def test_canonical_ignored_when_outside_tolerance(self, score_tracker):
        # EMA gap is outside tolerance, so local rank 1 remains the winner.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.640})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_b",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        dust_pool = 1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        assert weights["hk_b"] == pytest.approx(dust_pool)

    def test_canonical_at_exact_tolerance_boundary(self, score_tracker):
        # Boundary is inclusive.
        gap = CANONICAL_TIEBREAK_TOLERANCE
        _seed_eligible_emas(
            score_tracker, {"hk_a": 0.70, "hk_b": 0.70 - gap}
        )
        actual_gap = (
            score_tracker.ema_scores["hk_a"] - score_tracker.ema_scores["hk_b"]
        )
        assert actual_gap == pytest.approx(CANONICAL_TIEBREAK_TOLERANCE)
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_b",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_b"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_just_outside_tolerance(self, score_tracker):
        # Just outside the tolerance band, local rank 1 remains the winner.
        gap = CANONICAL_TIEBREAK_TOLERANCE + 0.001
        _seed_eligible_emas(
            score_tracker, {"hk_a": 0.70, "hk_b": 0.70 - gap}
        )
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_b",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_lower_than_local_rank1_within_tolerance(self, score_tracker):
        _seed_eligible_emas(
            score_tracker, {"hk_a": 0.700, "hk_b": 0.715},
        )
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_a",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_not_in_eligible_set_falls_back(self, score_tracker):
        # A canonical hotkey outside local eligibility cannot be used.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.695})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_unknown_validator_in_some_other_subnet",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_ranking_skips_ineligible_rank1(self, score_tracker):
        # Scan past canonical entries that are not locally eligible.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.695})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_ranking=["hk_new_not_eligible", "hk_b", "hk_a"],
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_b"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_ranking_candidate_outside_tolerance_falls_back(self, score_tracker):
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.650})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_ranking=["hk_new_not_eligible", "hk_b"],
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_ranking_scans_past_out_of_band_candidate(self, score_tracker):
        _seed_eligible_emas(
            score_tracker,
            {"hk_a": 0.700, "hk_b": 0.690, "hk_c": 0.650},
        )
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b", "hk_c"],
            canonical_ranking=["hk_c", "hk_b", "hk_a"],
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_b"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_no_canonical_preserves_pure_function_baseline(self, score_tracker):
        # No canonical input preserves local EMA ranking.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.695})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top=None,
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_no_canonical_matches_default_call(self, score_tracker):
        # Explicit None and omitted canonical input should be equivalent.
        _seed_eligible_emas(
            score_tracker,
            {"hk_a": 0.70, "hk_b": 0.69, "hk_c": 0.68, "hk_d": 0.67},
        )
        with_none = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b", "hk_c", "hk_d"],
            canonical_top=None,
            **DEFAULT_REWARD_POLICY,
        )
        without_kwarg = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b", "hk_c", "hk_d"],
            **DEFAULT_REWARD_POLICY,
        )
        assert with_none == without_kwarg

    def test_canonical_matches_local_rank1_is_noop(self, score_tracker):
        _seed_eligible_emas(score_tracker, {"hk_a": 0.700, "hk_b": 0.695})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"],
            canonical_top="hk_a",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_dust_redistribution_after_canonical_override(self, score_tracker):
        # Three eligible miners (all within tolerance), canonical promotes
        # rank 2 to winner. The displaced local rank 1 keeps the top dust slot.
        _seed_eligible_emas(
            score_tracker,
            {"hk_a": 0.700, "hk_b": 0.695, "hk_c": 0.690},
        )
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b", "hk_c"],
            canonical_top="hk_b",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_b"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        dust_pool = 1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        dust_total = 1.0 + DEFAULT_DUST_DECAY
        assert weights["hk_a"] == pytest.approx(dust_pool / dust_total)
        assert weights["hk_c"] == pytest.approx(
            dust_pool * DEFAULT_DUST_DECAY / dust_total
        )

    def test_canonical_with_single_eligible_miner(self, score_tracker):
        # A single locally eligible miner remains the winner.
        _seed_eligible_emas(score_tracker, {"hk_a": 0.90})
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a"],
            canonical_top="hk_some_other",
            **DEFAULT_REWARD_POLICY,
        )
        assert weights["hk_a"] == pytest.approx(DEFAULT_WINNER_WEIGHT)

    def test_canonical_full_top10_dust_displaced_winner(self, score_tracker):
        # Canonical rank 2 wins; local rank 1 remains first dust recipient.
        emas = {f"hk_{i:02d}": 0.700 - i * 0.001 for i in range(1, 12)}
        _seed_eligible_emas(score_tracker, emas)
        miner_list = list(emas.keys())
        weights = score_tracker.get_winner_heavy_pruning_dust_weights(
            miner_list,
            canonical_top="hk_02",
            **DEFAULT_REWARD_POLICY,
        )
        non_zero = [hk for hk, w in weights.items() if w > 0]
        assert len(non_zero) == 10, f"expected 10 non-zero, got {len(non_zero)}"
        assert weights["hk_02"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert weights["hk_11"] == 0
        assert weights["hk_01"] > weights["hk_03"]


# =========================================================================
# TestRecoverFromPlatformState
# =========================================================================

class TestRecoverFromPlatformState:
    """State recovery from platform DB."""

    def test_empty_state(self, score_tracker):
        score_tracker.recover_from_platform_state([], [])
        assert score_tracker.ema_scores == {}
        assert score_tracker.round_history == []

    def test_recovery_restores_ema_and_participation(self, score_tracker):
        ema_entries = [
            {"miner_hotkey": "hk_a", "ema_score": 0.55, "participation_count": 5, "eligible": False},
            {"miner_hotkey": "hk_b", "ema_score": 0.30, "participation_count": 3, "eligible": False},
        ]
        round_history = [
            {"round_id": f"r{i}", "scored_hotkeys": ["hk_a"]}
            for i in range(5)
        ] + [
            {"round_id": f"r{i}", "scored_hotkeys": ["hk_a", "hk_b"]}
            for i in range(5, 8)
        ]
        score_tracker.recover_from_platform_state(ema_entries, round_history)
        assert score_tracker.ema_scores["hk_a"] == pytest.approx(0.55)
        assert score_tracker.ema_scores["hk_b"] == pytest.approx(0.30)
        assert score_tracker.get_participation_count("hk_a") == 8
        assert score_tracker.get_participation_count("hk_b") == 3

    def test_recovery_trims_to_window(self, score_tracker):
        """History exceeding PARTICIPATION_WINDOW gets trimmed on recovery."""
        ema_entries = [{"miner_hotkey": "hk_a", "ema_score": 0.5}]
        round_history = [
            {"round_id": f"r{i}", "scored_hotkeys": ["hk_a"]}
            for i in range(25)
        ]
        score_tracker.recover_from_platform_state(ema_entries, round_history)
        assert len(score_tracker.round_history) == PARTICIPATION_WINDOW
        # Only the last 20 rounds survive (r5..r24)
        assert score_tracker.round_history[0]["round_id"] == "r5"
        assert score_tracker.get_participation_count("hk_a") == PARTICIPATION_WINDOW


# =========================================================================
# TestRankingsAndHistory
# =========================================================================

class TestRankingsAndHistory:
    """get_rankings and build_weight_history."""

    def test_get_rankings(self, score_tracker):
        _make_eligible_group(score_tracker, {"hk_a": 0.90, "hk_b": 0.70})
        # hk_c is NOT eligible (only 1 round)
        _make_active(score_tracker, "hk_c", rounds=1, score=0.99)

        rankings = score_tracker.get_rankings(["hk_a", "hk_b", "hk_c"])
        assert rankings["hk_a"] == 1
        assert rankings["hk_b"] == 2
        assert rankings["hk_c"] is None  # ineligible

    def test_build_weight_history_structure(self, score_tracker_low_threshold):
        tracker = score_tracker_low_threshold
        _make_eligible_group(tracker, {"hk_a": 0.90, "hk_b": 0.70})

        weights = tracker.get_winner_heavy_pruning_dust_weights(
            ["hk_a", "hk_b"], **DEFAULT_REWARD_POLICY
        )
        entries = tracker.build_weight_history(
            round_id="r_final",
            validator_hotkey="val_hk",
            miner_hotkeys=["hk_a", "hk_b"],
            weights=weights,
        )

        assert len(entries) == 2
        keys = {"miner_hotkey", "raw_score", "ema_score", "rank",
                "weight", "eligible", "participation_count"}
        for entry in entries:
            assert set(entry.keys()) == keys

        # Winner entry
        winner = [e for e in entries if e["miner_hotkey"] == "hk_a"][0]
        assert winner["weight"] == pytest.approx(DEFAULT_WINNER_WEIGHT)
        assert winner["eligible"] is True
        assert winner["rank"] == 1

        loser = [e for e in entries if e["miner_hotkey"] == "hk_b"][0]
        assert loser["weight"] == pytest.approx(
            1.0 - DEFAULT_BURN_RATE - DEFAULT_WINNER_WEIGHT
        )
        assert loser["eligible"] is True
        assert loser["rank"] == 2
