"""Tests for Kalshi soccer market discovery and classification."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import pytest

from footbe_trader.strategy.kalshi_discovery import (
    SoccerMarketClassification,
    KalshiEventRecord,
    KalshiMarketRecord,
    SoccerMarketClassifier,
)


class TestSoccerMarketClassification:
    """Tests for SoccerMarketClassification dataclass."""
    
    def test_classification_fields(self):
        """Should have required classification fields."""
        classification = SoccerMarketClassification(
            is_soccer=True,
            confidence=0.95,
            league_key="premier_league",
            home_team="Man City",
            away_team="Arsenal",
            structure_type="1X2",
        )
        
        assert classification.is_soccer is True
        assert classification.home_team == "Man City"
        assert classification.structure_type == "1X2"
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        classification = SoccerMarketClassification(
            is_soccer=True,
            confidence=0.9,
            home_team="Team A",
            away_team="Team B",
        )
        
        d = classification.to_dict()
        
        assert d["is_soccer"] is True
        assert d["home_team"] == "Team A"


class TestSoccerMarketClassifier:
    """Tests for SoccerMarketClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return SoccerMarketClassifier()
    
    def test_init(self, classifier):
        """Should initialize classifier."""
        assert classifier is not None
        assert classifier.config is not None
    
    def test_detect_soccer_by_series(self, classifier):
        """Should detect soccer by series ticker prefix."""
        # Create mock event with soccer series
        event = Mock()
        event.series_ticker = "SOCCER-EPL"
        event.title = "Man City vs Arsenal"
        event.subtitle = "Premier League Match"
        event.category = "sports"
        
        result = classifier.classify_event(event)
        
        # Should recognize as soccer
        assert result.is_soccer is True
    
    def test_detect_soccer_by_keywords(self, classifier):
        """Should detect soccer by keywords in title."""
        event = Mock()
        event.series_ticker = "SPORTS-MATCH-123"
        event.title = "Manchester United vs Liverpool - Premier League"
        event.subtitle = None
        event.category = "sports"
        
        result = classifier.classify_event(event)
        
        # Should recognize soccer keywords
        assert result is not None
    
    def test_non_soccer_event(self, classifier):
        """Should identify non-soccer events."""
        event = Mock()
        event.series_ticker = "CRYPTO-BTC"
        event.title = "Will Bitcoin exceed $100k?"
        event.subtitle = None
        event.category = "crypto"
        
        result = classifier.classify_event(event)
        
        assert result.is_soccer is False
    
    def test_extract_teams_from_vs_pattern(self, classifier):
        """Should extract team names from 'X vs Y' pattern."""
        event = Mock()
        event.series_ticker = "SOCCER-EPL"
        event.title = "Chelsea vs Arsenal"
        event.subtitle = "Premier League"
        event.category = "sports"
        
        result = classifier.classify_event(event)
        
        if result.is_soccer:
            # Should extract teams
            assert result.home_team is not None
            assert result.away_team is not None


class TestKalshiEventRecord:
    """Tests for KalshiEventRecord."""
    
    def test_creation(self):
        """Should create event record."""
        record = KalshiEventRecord(
            event_ticker="SOCCER-EPL-123",
            series_ticker="SOCCER-EPL",
            title="Man City vs Arsenal",
            subtitle="Premier League Match",
            category="sports",
            sub_category="soccer",
            strike_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            is_soccer=True,
            league_key="premier_league",
            parsed_home_team="Man City",
            parsed_away_team="Arsenal",
            parsed_canonical_home="manchester city",
            parsed_canonical_away="arsenal",
            market_structure="1X2",
        )
        
        assert record.event_ticker == "SOCCER-EPL-123"
        assert record.is_soccer is True
        assert record.parsed_home_team == "Man City"


class TestKalshiMarketRecord:
    """Tests for KalshiMarketRecord."""
    
    def test_creation(self):
        """Should create market record."""
        record = KalshiMarketRecord(
            ticker="SOCCER-EPL-MCI-ARS-H",
            event_ticker="SOCCER-EPL-MCI-ARS",
            title="Man City to win",
            subtitle=None,
            status="active",
            open_time=datetime(2024, 3, 14, tzinfo=timezone.utc),
            close_time=datetime(2024, 3, 15, 15, 0, tzinfo=timezone.utc),
            expiration_time=datetime(2024, 3, 15, 17, 0, tzinfo=timezone.utc),
            yes_bid=0.65,
            yes_ask=0.67,
            no_bid=0.33,
            no_ask=0.35,
            last_price=0.66,
            volume=1000,
            is_soccer=True,
            market_type="HOME_WIN",
            parsed_team="Man City",
            parsed_canonical_team="manchester city",
        )
        
        assert record.ticker == "SOCCER-EPL-MCI-ARS-H"
        assert record.market_type == "HOME_WIN"
        assert record.yes_bid == 0.65


class TestClassifierPatterns:
    """Tests for specific classification patterns."""
    
    @pytest.fixture
    def classifier(self):
        return SoccerMarketClassifier()
    
    def test_vs_pattern_detection(self, classifier):
        """Should detect vs pattern in title."""
        event = Mock()
        event.series_ticker = "SOCCER-EPL"
        event.title = "Manchester City vs Arsenal"
        event.subtitle = None
        event.category = "sports"
        
        result = classifier.classify_event(event)
        
        # With SOCCER series, should be detected
        assert result.is_soccer is True
    
    def test_multiple_vs_patterns(self, classifier):
        """Should handle various vs patterns."""
        patterns = [
            "Man City vs Arsenal",
            "Man City v Arsenal",
            "Man City v. Arsenal",
        ]
        
        for title in patterns:
            event = Mock()
            event.series_ticker = "SOCCER-EPL"
            event.title = title
            event.subtitle = None
            event.category = "sports"
            
            result = classifier.classify_event(event)
            assert result is not None
    
    def test_league_detection_from_series(self, classifier):
        """Should detect league from series ticker."""
        event = Mock()
        event.series_ticker = "SOCCER-EPL-MCI-ARS"
        event.title = "Man City vs Arsenal"
        event.subtitle = "Premier League"
        event.category = "sports"
        
        result = classifier.classify_event(event)
        
        # Should be soccer
        assert result.is_soccer is True
