"""Tests for field catalog and pre-match field validation.

These tests ensure that pre-match prediction models do not accidentally
use POST_MATCH_ONLY fields, which would cause data leakage.
"""

import pytest

from footbe_trader.football.field_catalog import (
    FIELD_CATALOG,
    FieldAvailability,
    FieldInfo,
    get_catalog_summary,
    get_endpoints,
    get_fields_by_availability,
    get_fields_by_endpoint,
    get_post_match_only_fields,
    get_pre_match_safe_fields,
    is_field_pre_match_safe,
    validate_pre_match_fields,
)


class TestFieldCatalog:
    """Tests for field catalog structure."""

    def test_catalog_not_empty(self):
        """Catalog should have fields."""
        assert len(FIELD_CATALOG) > 0

    def test_all_fields_have_required_attributes(self):
        """All fields must have required attributes."""
        for path, field in FIELD_CATALOG.items():
            assert isinstance(field, FieldInfo)
            assert field.path == path
            assert field.python_type in ("int", "str", "bool", "float", "datetime")
            assert isinstance(field.availability, FieldAvailability)
            assert field.endpoint != "", f"Field {path} missing endpoint"

    def test_catalog_has_all_availability_types(self):
        """Catalog should have fields of each availability type."""
        summary = get_catalog_summary()
        assert summary["pre_match_safe"] > 0, "No PRE_MATCH_SAFE fields"
        assert summary["pre_match_uncertain"] > 0, "No PRE_MATCH_UNCERTAIN fields"
        assert summary["post_match_only"] > 0, "No POST_MATCH_ONLY fields"

    def test_catalog_has_expected_endpoints(self):
        """Catalog should include expected endpoints."""
        endpoints = get_endpoints()
        expected = [
            "fixtures",
            "standings",
            "teams",
            "fixtures/statistics",
            "fixtures/lineups",
            "injuries",
            "odds",
        ]
        for ep in expected:
            assert ep in endpoints, f"Missing endpoint: {ep}"


class TestFieldAvailabilityLabels:
    """Tests for correct availability labeling."""

    def test_goals_are_post_match_only(self):
        """Goals must be POST_MATCH_ONLY to prevent data leakage."""
        goal_fields = [
            "goals.home",
            "goals.away",
            "score.halftime.home",
            "score.halftime.away",
            "score.fulltime.home",
            "score.fulltime.away",
        ]
        for path in goal_fields:
            assert path in FIELD_CATALOG, f"Field {path} not in catalog"
            assert FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY, \
                f"Field {path} must be POST_MATCH_ONLY"

    def test_winner_is_post_match_only(self):
        """Winner flags must be POST_MATCH_ONLY."""
        winner_fields = ["teams.home.winner", "teams.away.winner"]
        for path in winner_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY

    def test_match_statistics_are_post_match_only(self):
        """Match statistics must be POST_MATCH_ONLY."""
        stat_fields = [
            "statistics.Shots on Goal",
            "statistics.Ball Possession",
            "statistics.Corner Kicks",
            "statistics.expected_goals",
        ]
        for path in stat_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY

    def test_events_are_post_match_only(self):
        """Match events must be POST_MATCH_ONLY."""
        event_fields = [
            "events.type",
            "events.detail",
            "events.player.name",
        ]
        for path in event_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY

    def test_standings_are_pre_match_safe(self):
        """Standings should be PRE_MATCH_SAFE (known before match)."""
        standings_fields = [
            "standings.rank",
            "standings.points",
            "standings.all.played",
            "standings.all.win",
            "standings.form",
        ]
        for path in standings_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE

    def test_team_info_is_pre_match_safe(self):
        """Team info should be PRE_MATCH_SAFE."""
        team_fields = [
            "team.id",
            "team.name",
            "team.founded",
            "venue.capacity",
        ]
        for path in team_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE

    def test_fixture_scheduling_is_pre_match_safe(self):
        """Fixture scheduling info should be PRE_MATCH_SAFE."""
        fixture_fields = [
            "fixture.id",
            "fixture.date",
            "fixture.venue.name",
            "league.round",
            "teams.home.id",
            "teams.away.name",
        ]
        for path in fixture_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE

    def test_lineups_are_uncertain(self):
        """Lineups should be PRE_MATCH_UNCERTAIN (often missing/late)."""
        lineup_fields = [
            "lineups.formation",
            "lineups.startXI.player.id",
            "lineups.coach.name",
        ]
        for path in lineup_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_UNCERTAIN

    def test_injuries_are_uncertain(self):
        """Injuries should be PRE_MATCH_UNCERTAIN."""
        injury_fields = [
            "injuries.player.id",
            "injuries.player.type",
            "injuries.player.reason",
        ]
        for path in injury_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_UNCERTAIN

    def test_odds_are_pre_match_safe(self):
        """Odds should be PRE_MATCH_SAFE."""
        odds_fields = [
            "odds.bookmaker.name",
            "odds.bet.name",
            "odds.value.odd",
        ]
        for path in odds_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE

    def test_h2h_historical_is_pre_match_safe(self):
        """Head-to-head historical data should be PRE_MATCH_SAFE."""
        h2h_fields = [
            "h2h.fixture.id",
            "h2h.goals.home",
            "h2h.teams.home.name",
        ]
        for path in h2h_fields:
            assert path in FIELD_CATALOG
            assert FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE


class TestValidationFunctions:
    """Tests for field validation helper functions."""

    def test_is_field_pre_match_safe_true(self):
        """PRE_MATCH_SAFE fields should return True."""
        assert is_field_pre_match_safe("standings.points") is True
        assert is_field_pre_match_safe("teams.home.name") is True
        assert is_field_pre_match_safe("fixture.date") is True

    def test_is_field_pre_match_safe_false_for_post_match(self):
        """POST_MATCH_ONLY fields should return False."""
        assert is_field_pre_match_safe("goals.home") is False
        assert is_field_pre_match_safe("teams.home.winner") is False
        assert is_field_pre_match_safe("statistics.Shots on Goal") is False

    def test_is_field_pre_match_safe_false_for_uncertain(self):
        """PRE_MATCH_UNCERTAIN fields should return False (not strictly safe)."""
        assert is_field_pre_match_safe("lineups.formation") is False
        assert is_field_pre_match_safe("injuries.player.type") is False

    def test_is_field_pre_match_safe_false_for_unknown(self):
        """Unknown fields should return False (conservative)."""
        assert is_field_pre_match_safe("unknown.field.path") is False

    def test_validate_pre_match_fields_no_violations(self):
        """All safe fields should have no violations."""
        safe_fields = [
            "standings.points",
            "teams.home.name",
            "fixture.date",
            "league.round",
        ]
        violations = validate_pre_match_fields(safe_fields)
        assert violations == [], f"Unexpected violations: {violations}"

    def test_validate_pre_match_fields_with_violations(self):
        """POST_MATCH_ONLY fields should be violations."""
        fields = [
            "standings.points",  # safe
            "goals.home",  # violation
            "teams.home.winner",  # violation
        ]
        violations = validate_pre_match_fields(fields)
        assert len(violations) == 2
        assert any("goals.home" in v for v in violations)
        assert any("teams.home.winner" in v for v in violations)

    def test_validate_pre_match_fields_unknown_field(self):
        """Unknown fields should be violations."""
        fields = ["unknown.field"]
        violations = validate_pre_match_fields(fields)
        assert len(violations) == 1
        assert "unknown" in violations[0]

    def test_get_pre_match_safe_fields(self):
        """Should return all PRE_MATCH_SAFE fields."""
        safe_fields = get_pre_match_safe_fields()
        assert len(safe_fields) > 0
        for field in safe_fields:
            assert field.availability == FieldAvailability.PRE_MATCH_SAFE

    def test_get_post_match_only_fields(self):
        """Should return all POST_MATCH_ONLY fields."""
        post_fields = get_post_match_only_fields()
        assert len(post_fields) > 0
        for field in post_fields:
            assert field.availability == FieldAvailability.POST_MATCH_ONLY


class TestPreMatchModelFeatures:
    """Tests to ensure pre-match models don't use POST_MATCH_ONLY fields.

    This test class is critical for preventing data leakage.
    Add your feature definitions here and validate them.
    """

    # Define features that a pre-match model might use
    EXAMPLE_PRE_MATCH_FEATURES = [
        # Standings-based features
        "standings.rank",
        "standings.points",
        "standings.goalsDiff",
        "standings.form",
        "standings.all.win",
        "standings.all.draw",
        "standings.all.lose",
        "standings.home.win",
        "standings.away.win",
        # Team info
        "teams.home.id",
        "teams.home.name",
        "teams.away.id",
        "teams.away.name",
        # Fixture info
        "fixture.date",
        "fixture.venue.name",
        "league.round",
        # Historical H2H
        "h2h.goals.home",
        "h2h.goals.away",
    ]

    def test_example_features_are_pre_match_safe(self):
        """All example pre-match features must be safe to use."""
        violations = validate_pre_match_fields(self.EXAMPLE_PRE_MATCH_FEATURES)
        assert violations == [], \
            f"Pre-match features contain unsafe fields: {violations}"

    # List of known POST_MATCH_ONLY fields that must never appear in pre-match features
    FORBIDDEN_FEATURE_PATHS = [
        "goals.home",
        "goals.away",
        "teams.home.winner",
        "teams.away.winner",
        "score.halftime.home",
        "score.halftime.away",
        "score.fulltime.home",
        "score.fulltime.away",
        "statistics.Shots on Goal",
        "statistics.Ball Possession",
        "statistics.expected_goals",
        "events.type",
        "events.player.name",
    ]

    def test_forbidden_fields_are_post_match_only(self):
        """Verify forbidden fields are correctly labeled POST_MATCH_ONLY."""
        for path in self.FORBIDDEN_FEATURE_PATHS:
            assert path in FIELD_CATALOG, f"Forbidden field {path} not in catalog"
            assert FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY, \
                f"Forbidden field {path} should be POST_MATCH_ONLY"

    def test_no_forbidden_fields_in_example_features(self):
        """Example features must not include any forbidden fields."""
        for path in self.EXAMPLE_PRE_MATCH_FEATURES:
            assert path not in self.FORBIDDEN_FEATURE_PATHS, \
                f"Forbidden field {path} found in example features!"


class TestFieldCatalogIntegrity:
    """Tests for catalog data integrity."""

    def test_no_duplicate_paths(self):
        """Each path should be unique."""
        # Dict keys are inherently unique, but let's verify
        paths = list(FIELD_CATALOG.keys())
        assert len(paths) == len(set(paths))

    def test_paths_follow_naming_convention(self):
        """Paths should follow dot-notation convention."""
        for path in FIELD_CATALOG.keys():
            # Should not start or end with dot
            assert not path.startswith(".")
            assert not path.endswith(".")
            # Should not have consecutive dots
            assert ".." not in path
            # Each segment should have at least one character
            for segment in path.split("."):
                assert len(segment) > 0, f"Empty segment in path {path}"

    def test_all_fields_have_descriptions(self):
        """All fields should have descriptions."""
        for path, field in FIELD_CATALOG.items():
            assert field.description, f"Field {path} missing description"

    def test_all_fields_have_endpoints(self):
        """All fields should have an endpoint specified."""
        for path, field in FIELD_CATALOG.items():
            assert field.endpoint, f"Field {path} missing endpoint"


# =============================================================================
# FEATURE REGISTRY VALIDATION
# =============================================================================
# When you add a new feature engineering module, add a test here to validate
# that all its fields are pre-match safe.
# =============================================================================


class TestFeatureModulesUseSafeFields:
    """Validate that feature modules only use safe fields.

    Add tests here for each feature module you create.
    This is the critical test that prevents data leakage.
    """

    def test_placeholder_for_future_features(self):
        """Placeholder - replace with actual feature module tests."""
        # When you create a feature module like:
        # from footbe_trader.modeling.features import get_feature_fields
        #
        # You would test it like:
        # feature_fields = get_feature_fields()
        # violations = validate_pre_match_fields(feature_fields)
        # assert violations == [], f"Data leakage risk: {violations}"
        pass

    # Example of what a real test would look like:
    #
    # def test_standings_features_are_safe(self):
    #     """Standings feature module uses only safe fields."""
    #     from footbe_trader.modeling.features.standings import FIELDS_USED
    #     violations = validate_pre_match_fields(FIELDS_USED)
    #     assert violations == [], f"Standings features use unsafe fields: {violations}"
    #
    # def test_form_features_are_safe(self):
    #     """Form feature module uses only safe fields."""
    #     from footbe_trader.modeling.features.form import FIELDS_USED
    #     violations = validate_pre_match_fields(FIELDS_USED)
    #     assert violations == [], f"Form features use unsafe fields: {violations}"
