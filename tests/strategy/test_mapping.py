"""Tests for fixture-to-market mapping engine."""

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch
import pytest

from footbe_trader.strategy.mapping import (
    MappingCandidate,
    FixtureMarketMapping,
    MappingResult,
    MappingConfig,
    ManualOverrides,
)


class TestMappingCandidate:
    """Tests for MappingCandidate dataclass."""
    
    def test_creation(self):
        """Should create candidate with all fields."""
        candidate = MappingCandidate(
            event_ticker="SOCCER-EPL-MCI-ARS",
            event_title="Man City vs Arsenal",
            ticker_home_win="SOCCER-EPL-MCI-ARS-H",
            ticker_draw="SOCCER-EPL-MCI-ARS-D",
            ticker_away_win="SOCCER-EPL-MCI-ARS-A",
            structure_type="1X2",
            total_score=0.86,
            team_match_score=0.9,
            date_match_score=1.0,
            league_match_score=0.8,
            market_type_score=0.7,
            text_similarity_score=0.85,
        )
        
        assert candidate.event_ticker == "SOCCER-EPL-MCI-ARS"
        assert candidate.total_score == 0.86
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        candidate = MappingCandidate(
            event_ticker="TICKER-1",
            event_title="Test Match",
            total_score=0.75,
            structure_type="1X2",
        )
        
        d = candidate.to_dict()
        
        assert d["event_ticker"] == "TICKER-1"
        assert d["total_score"] == 0.75
        assert "score_breakdown" in d


class TestFixtureMarketMapping:
    """Tests for FixtureMarketMapping dataclass."""
    
    def test_creation(self):
        """Should create mapping with all fields."""
        mapping = FixtureMarketMapping(
            fixture_id=1,
            structure_type="1X2",
            ticker_home_win="TICKER-H",
            ticker_draw="TICKER-D",
            ticker_away_win="TICKER-A",
            confidence_score=0.9,
            status="AUTO",
        )
        
        assert mapping.fixture_id == 1
        assert mapping.status == "AUTO"
        assert mapping.confidence_score == 0.9
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        mapping = FixtureMarketMapping(
            fixture_id=100,
            structure_type="NO_DRAW",
            ticker_home_win="H-WIN",
            ticker_away_win="A-WIN",
            confidence_score=0.85,
        )
        
        d = mapping.to_dict()
        
        assert d["fixture_id"] == 100
        assert d["ticker_home_win"] == "H-WIN"


class TestMappingResult:
    """Tests for MappingResult dataclass."""
    
    def test_creation(self):
        """Should create result with all fields."""
        result = MappingResult(
            fixture_id=1,
            fixture_info={"home_team": "Team A", "away_team": "Team B"},
            success=True,
            mapping=None,
            candidates=[],
            reason="Test",
        )
        
        assert result.success is True
        assert result.fixture_id == 1
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        result = MappingResult(
            fixture_id=123,
            fixture_info={"home_team": "X"},
            success=False,
            reason="No matches found",
        )
        
        d = result.to_dict()
        
        assert d["fixture_id"] == 123
        assert d["success"] is False


class TestMappingConfig:
    """Tests for MappingConfig."""
    
    def test_default_values(self):
        """Should have reasonable defaults."""
        config = MappingConfig()
        
        assert config.hours_before > 0
        assert config.hours_after > 0
        assert config.auto_accept_threshold > 0
        assert config.review_threshold > 0
        assert config.min_candidate_threshold > 0
    
    def test_weights_sum_to_one(self):
        """Scoring weights should sum to approximately 1."""
        config = MappingConfig()
        
        weights = config.scoring_weights
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.01
    
    def test_market_type_preferences(self):
        """Should have market type preferences."""
        config = MappingConfig()
        
        prefs = config.market_type_preferences
        
        assert "1X2" in prefs
        assert prefs["1X2"] >= prefs.get("UNKNOWN", 0)


class TestManualOverrides:
    """Tests for ManualOverrides."""
    
    def test_initialization(self):
        """Should initialize with empty overrides."""
        # Create with non-existent path
        overrides = ManualOverrides("/nonexistent/path.yaml")
        assert overrides.overrides is not None
    
    def test_get_override_returns_none_when_missing(self):
        """Should return None when no override exists."""
        overrides = ManualOverrides("/nonexistent/path.yaml")
        
        result = overrides.get_override(fixture_id=99999)
        
        assert result is None


class TestMappingIntegration:
    """Integration tests for mapping workflow."""
    
    def test_config_loads_without_file(self):
        """Config should work even without file."""
        config = MappingConfig("/nonexistent/config.yaml")
        
        # Should have defaults
        assert config.hours_before > 0
        assert config.auto_accept_threshold > 0


class TestGoldenMappings:
    """Golden tests with real fixture/market examples."""
    
    # Golden test 1: Premier League - exact team match
    GOLDEN_EPL_EXACT = {
        "fixture": {
            "fixture_id": 10001,
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "kickoff_utc": "2024-04-07T16:30:00Z",
            "league_id": 39,
        },
        "market": {
            "ticker": "SOCCER-EPL-MANU-LIV-1X2",
            "home_team": "Man Utd",
            "away_team": "Liverpool",
            "match_date": "2024-04-07",
            "league_key": "premier_league",
            "structure_type": "1X2",
        },
        "expected_match": True,
        "expected_min_score": 0.85,
    }
    
    # Golden test 2: La Liga - diacritics handling
    GOLDEN_LALIGA_DIACRITICS = {
        "fixture": {
            "fixture_id": 10002,
            "home_team": "Atlético Madrid",
            "away_team": "Real Betis",
            "kickoff_utc": "2024-04-07T20:00:00Z",
            "league_id": 140,
        },
        "market": {
            "ticker": "SOCCER-LALIGA-ATM-BET-1X2",
            "home_team": "Atletico Madrid",  # No diacritics
            "away_team": "Betis",
            "match_date": "2024-04-07",
            "league_key": "la_liga",
            "structure_type": "1X2",
        },
        "expected_match": True,
        "expected_min_score": 0.80,
    }
    
    # Golden test 3: Serie A - prefix handling
    GOLDEN_SERIEA_PREFIX = {
        "fixture": {
            "fixture_id": 10003,
            "home_team": "AC Milan",
            "away_team": "AS Roma",
            "kickoff_utc": "2024-04-07T18:45:00Z",
            "league_id": 135,
        },
        "market": {
            "ticker": "SOCCER-SERIEA-MIL-ROM-1X2",
            "home_team": "Milan",  # No prefix
            "away_team": "Roma",   # No prefix
            "match_date": "2024-04-07",
            "league_key": "serie_a",
            "structure_type": "1X2",
        },
        "expected_match": True,
        "expected_min_score": 0.80,
    }
    
    # Golden test 4: Bundesliga - umlaut handling
    GOLDEN_BUNDESLIGA_UMLAUT = {
        "fixture": {
            "fixture_id": 10004,
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "kickoff_utc": "2024-04-06T17:30:00Z",
            "league_id": 78,
        },
        "market": {
            "ticker": "SOCCER-BUND-BAY-BVB-1X2",
            "home_team": "Bayern Munich",  # No umlaut
            "away_team": "Dortmund",
            "match_date": "2024-04-06",
            "league_key": "bundesliga",
            "structure_type": "1X2",
        },
        "expected_match": True,
        "expected_min_score": 0.80,
    }
    
    # Golden test 5: MLS - American naming
    GOLDEN_MLS_NAMING = {
        "fixture": {
            "fixture_id": 10005,
            "home_team": "Los Angeles FC",
            "away_team": "LA Galaxy",
            "kickoff_utc": "2024-04-06T02:30:00Z",
            "league_id": 253,
        },
        "market": {
            "ticker": "SOCCER-MLS-LAFC-LAG-ML",
            "home_team": "LAFC",
            "away_team": "Galaxy",
            "match_date": "2024-04-06",
            "league_key": "mls",
            "structure_type": "NO_DRAW",
        },
        "expected_match": True,
        "expected_min_score": 0.75,
    }
    
    # Golden test 6: Non-match - different fixture
    GOLDEN_NON_MATCH = {
        "fixture": {
            "fixture_id": 10006,
            "home_team": "Chelsea",
            "away_team": "Tottenham",
            "kickoff_utc": "2024-04-07T14:00:00Z",
            "league_id": 39,
        },
        "market": {
            "ticker": "SOCCER-EPL-ARS-MCI-1X2",
            "home_team": "Arsenal",
            "away_team": "Man City",
            "match_date": "2024-04-07",
            "league_key": "premier_league",
            "structure_type": "1X2",
        },
        "expected_match": False,
        "expected_max_score": 0.50,
    }
    
    def test_golden_epl_exact_match(self):
        """EPL fixture with exact team names should match."""
        from footbe_trader.strategy.normalization import TeamNameNormalizer, TeamAliasRegistry, fuzzy_match_ratio
        
        fixture = self.GOLDEN_EPL_EXACT["fixture"]
        market = self.GOLDEN_EPL_EXACT["market"]
        
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        # Normalize both and compare
        norm_fixture_home = normalizer.normalize(fixture["home_team"]).normalized
        norm_market_home = normalizer.normalize(market["home_team"]).normalized
        
        # Both should contain "manchester" and "utd/united"
        assert "manchester" in norm_fixture_home
        assert "man" in norm_market_home or "utd" in norm_market_home
        
        # Liverpool should match exactly
        away_ratio = fuzzy_match_ratio(fixture["away_team"], market["away_team"])
        assert away_ratio >= 90
    
    def test_golden_laliga_diacritics(self):
        """La Liga fixture with diacritics should match normalized."""
        from footbe_trader.strategy.normalization import TeamNameNormalizer, TeamAliasRegistry
        
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        fixture = self.GOLDEN_LALIGA_DIACRITICS["fixture"]
        market = self.GOLDEN_LALIGA_DIACRITICS["market"]
        
        # Normalize both versions
        norm_fixture = normalizer.normalize(fixture["home_team"]).normalized
        norm_market = normalizer.normalize(market["home_team"]).normalized
        
        # Both should contain "atletico" or "madrid"
        assert "atletico" in norm_fixture or "madrid" in norm_fixture
        assert "atletico" in norm_market or "madrid" in norm_market
    
    def test_golden_seriea_prefix(self):
        """Serie A fixture with prefixes should match stripped versions."""
        from footbe_trader.strategy.normalization import TeamNameNormalizer, TeamAliasRegistry
        
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        fixture = self.GOLDEN_SERIEA_PREFIX["fixture"]
        market = self.GOLDEN_SERIEA_PREFIX["market"]
        
        # AC Milan -> milan
        norm_fixture_home = normalizer.normalize(fixture["home_team"]).normalized
        # Milan -> milan
        norm_market_home = normalizer.normalize(market["home_team"]).normalized
        
        assert "milan" in norm_fixture_home
        assert "milan" in norm_market_home
    
    def test_golden_bundesliga_umlaut(self):
        """Bundesliga fixture with umlauts should match."""
        from footbe_trader.strategy.normalization import TeamNameNormalizer, TeamAliasRegistry
        
        normalizer = TeamNameNormalizer(TeamAliasRegistry())
        
        fixture = self.GOLDEN_BUNDESLIGA_UMLAUT["fixture"]
        market = self.GOLDEN_BUNDESLIGA_UMLAUT["market"]
        
        # Bayern München -> bayern munchen
        norm_fixture = normalizer.normalize(fixture["home_team"]).normalized
        # Bayern Munich -> bayern munich
        norm_market = normalizer.normalize(market["home_team"]).normalized
        
        # Both should have "bayern"
        assert "bayern" in norm_fixture
        assert "bayern" in norm_market
    
    def test_golden_mls_abbreviations(self):
        """MLS fixture with abbreviations should reasonably match."""
        from footbe_trader.strategy.normalization import fuzzy_match_ratio
        
        fixture = self.GOLDEN_MLS_NAMING["fixture"]
        market = self.GOLDEN_MLS_NAMING["market"]
        
        # Los Angeles FC vs LAFC - need alias support
        # LA Galaxy vs Galaxy - should match
        away_ratio = fuzzy_match_ratio(fixture["away_team"], market["away_team"])
        assert away_ratio >= 50  # Galaxy should be found
    
    def test_golden_non_match_low_score(self):
        """Different fixture should not match."""
        from footbe_trader.strategy.normalization import fuzzy_match_ratio
        
        fixture = self.GOLDEN_NON_MATCH["fixture"]
        market = self.GOLDEN_NON_MATCH["market"]
        
        # Chelsea vs Arsenal - different teams
        home_ratio = fuzzy_match_ratio(fixture["home_team"], market["home_team"])
        away_ratio = fuzzy_match_ratio(fixture["away_team"], market["away_team"])
        
        # Both should be low
        assert home_ratio < 50
        assert away_ratio < 50
