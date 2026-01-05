"""Tests for league discovery and repository."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
import pytest

from footbe_trader.strategy.league_discovery import (
    LeagueInfo,
    LeagueDiscoveryResult,
    LeagueDiscovery,
    LeagueRepository,
)


class TestLeagueInfo:
    """Tests for LeagueInfo dataclass."""
    
    def test_creation(self):
        """Should create league info."""
        league = LeagueInfo(
            league_id=39,
            league_name="Premier League",
            country="England",
            type="League",
            logo_url="https://example.com/logo.png",
            seasons_available=[2022, 2023, 2024],
            league_key="premier_league",
        )
        
        assert league.league_id == 39
        assert league.league_name == "Premier League"
        assert 2024 in league.seasons_available
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        league = LeagueInfo(
            league_id=140,
            league_name="La Liga",
            country="Spain",
            type="League",
        )
        
        d = league.to_dict()
        
        assert d["league_id"] == 140
        assert d["league_name"] == "La Liga"
        assert d["country"] == "Spain"


class TestLeagueDiscoveryResult:
    """Tests for LeagueDiscoveryResult."""
    
    def test_creation(self):
        """Should create discovery result."""
        result = LeagueDiscoveryResult(
            total_found=10,
            new_leagues=5,
            updated_leagues=3,
            leagues=[],
            errors=["Error 1"],
        )
        
        assert result.total_found == 10
        assert result.new_leagues == 5
        assert len(result.errors) == 1


class TestLeagueKeyGeneration:
    """Tests for canonical league key generation."""
    
    def test_generate_key_from_name(self):
        """Should generate snake_case key from name."""
        # Test key generation via normalization module
        from footbe_trader.strategy.normalization import LeagueNameNormalizer, LeagueAliasRegistry
        
        normalizer = LeagueNameNormalizer(LeagueAliasRegistry())
        
        test_cases = [
            ("Premier League", "premier league"),
            ("La Liga", "la liga"),
            ("Serie A", "serie a"),
            ("Bundesliga", "bundesliga"),
            ("Ligue 1", "ligue 1"),
        ]
        
        for name, expected_base in test_cases:
            result = normalizer.normalize(name)
            # Should be lowercase
            assert result.normalized.islower()
            # Should contain expected base (spaces OK in normalized)
            assert expected_base in result.normalized or result.normalized.replace(" ", "_") == expected_base.replace(" ", "_")


class TestLeagueRepository:
    """Tests for LeagueRepository."""
    
    @pytest.fixture
    def repo_db(self):
        """Create database with leagues table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        
        conn.executescript("""
            CREATE TABLE leagues (
                id INTEGER PRIMARY KEY,
                league_id INTEGER UNIQUE,
                league_name TEXT,
                country TEXT,
                type TEXT,
                logo_url TEXT,
                seasons_available TEXT,
                league_key TEXT,
                is_active INTEGER DEFAULT 1,
                last_synced_at TEXT,
                raw_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
        """)
        
        return conn
    
    @pytest.fixture
    def repo(self, repo_db):
        """Create repository instance."""
        return LeagueRepository(repo_db)
    
    def test_upsert_league(self, repo, repo_db):
        """Should save league to database."""
        league = LeagueInfo(
            league_id=39,
            league_name="Premier League",
            country="England",
            type="League",
            logo_url="https://example.com/epl.png",
            seasons_available=[2022, 2023, 2024],
            league_key="premier_league",
        )
        
        repo.upsert_league(league)
        
        cursor = repo_db.execute(
            "SELECT * FROM leagues WHERE league_id = ?",
            (39,)
        )
        row = cursor.fetchone()
        
        assert row is not None
        assert row["league_name"] == "Premier League"
        assert row["country"] == "England"
    
    def test_get_league(self, repo, repo_db):
        """Should retrieve league by ID."""
        # Insert test data
        repo_db.execute("""
            INSERT INTO leagues 
            (league_id, league_name, country, type, league_key, seasons_available, 
             created_at, updated_at)
            VALUES (140, 'La Liga', 'Spain', 'League', 'la_liga', '[2023,2024]', 
                    '2024-01-01', '2024-01-01')
        """)
        repo_db.commit()
        
        league = repo.get_league(140)
        
        assert league is not None
        assert league.league_name == "La Liga"
        assert league.country == "Spain"
    
    def test_get_leagues_by_key(self, repo, repo_db):
        """Should retrieve leagues by canonical key."""
        # Insert test data
        repo_db.execute("""
            INSERT INTO leagues 
            (league_id, league_name, country, type, league_key, seasons_available,
             created_at, updated_at)
            VALUES (78, 'Bundesliga', 'Germany', 'League', 'bundesliga', '[2023,2024]', 
                    '2024-01-01', '2024-01-01')
        """)
        repo_db.commit()
        
        leagues = repo.get_leagues_by_key("bundesliga")
        
        assert len(leagues) == 1
        assert leagues[0].league_id == 78
    
    def test_get_all_leagues(self, repo, repo_db):
        """Should retrieve all leagues."""
        # Insert multiple leagues
        repo_db.execute("""
            INSERT INTO leagues 
            (league_id, league_name, country, type, league_key, seasons_available,
             created_at, updated_at)
            VALUES 
            (39, 'Premier League', 'England', 'League', 'premier_league', '[2024]', 
             '2024-01-01', '2024-01-01'),
            (140, 'La Liga', 'Spain', 'League', 'la_liga', '[2024]', 
             '2024-01-01', '2024-01-01'),
            (135, 'Serie A', 'Italy', 'League', 'serie_a', '[2024]', 
             '2024-01-01', '2024-01-01')
        """)
        repo_db.commit()
        
        leagues = repo.get_all_leagues()
        
        assert len(leagues) == 3
        league_ids = {l.league_id for l in leagues}
        assert 39 in league_ids
        assert 140 in league_ids
        assert 135 in league_ids
    
    def test_get_all_leagues_by_country(self, repo, repo_db):
        """Should filter leagues by country."""
        repo_db.execute("""
            INSERT INTO leagues 
            (league_id, league_name, country, type, league_key, seasons_available,
             created_at, updated_at)
            VALUES 
            (39, 'Premier League', 'England', 'League', 'premier_league', '[2024]', 
             '2024-01-01', '2024-01-01'),
            (40, 'Championship', 'England', 'League', 'championship', '[2024]', 
             '2024-01-01', '2024-01-01'),
            (140, 'La Liga', 'Spain', 'League', 'la_liga', '[2024]', 
             '2024-01-01', '2024-01-01')
        """)
        repo_db.commit()
        
        english_leagues = repo.get_all_leagues(country_filter="England")
        
        assert len(english_leagues) == 2
        for league in english_leagues:
            assert league.country == "England"
    
    def test_upsert_updates_existing(self, repo, repo_db):
        """Should update existing league on upsert."""
        # Insert initial data
        repo_db.execute("""
            INSERT INTO leagues 
            (league_id, league_name, country, type, league_key, seasons_available,
             created_at, updated_at)
            VALUES (39, 'EPL', 'England', 'League', 'epl', '[2023]', 
                    '2024-01-01', '2024-01-01')
        """)
        repo_db.commit()
        
        # Update with new data
        updated_league = LeagueInfo(
            league_id=39,
            league_name="Premier League",  # Updated name
            country="England",
            type="League",
            league_key="premier_league",  # Updated key
            seasons_available=[2023, 2024],  # Updated seasons
        )
        
        repo.upsert_league(updated_league)  # Should upsert
        
        league = repo.get_league(39)
        assert league.league_name == "Premier League"
        assert 2024 in league.seasons_available


class TestLeagueDiscovery:
    """Tests for LeagueDiscovery."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock API client."""
        client = Mock()
        return client
    
    def test_init(self, mock_client):
        """Should initialize with client."""
        discovery = LeagueDiscovery(mock_client)
        
        assert discovery.client == mock_client
    
    @pytest.mark.asyncio
    async def test_list_all_leagues_parses_response(self, mock_client):
        """Should parse API response correctly."""
        # Mock API response
        mock_client._request = AsyncMock(return_value={
            "response": [
                {
                    "league": {
                        "id": 39,
                        "name": "Premier League",
                        "type": "League",
                        "logo": "https://example.com/epl.png",
                    },
                    "country": {
                        "name": "England",
                    },
                    "seasons": [
                        {"year": 2023, "current": False},
                        {"year": 2024, "current": True},
                    ],
                },
                {
                    "league": {
                        "id": 140,
                        "name": "La Liga",
                        "type": "League",
                        "logo": "https://example.com/laliga.png",
                    },
                    "country": {
                        "name": "Spain",
                    },
                    "seasons": [
                        {"year": 2024, "current": True},
                    ],
                },
            ]
        })
        
        discovery = LeagueDiscovery(mock_client)
        leagues = await discovery.list_all_leagues()
        
        assert len(leagues) == 2
        
        epl = next(l for l in leagues if l.league_id == 39)
        assert epl.league_name == "Premier League"
        assert epl.country == "England"
        assert 2024 in epl.seasons_available


class TestLeagueSynchronization:
    """Tests for league sync workflow."""
    
    @pytest.fixture
    def sync_db(self):
        """Create database for sync tests."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        
        conn.executescript("""
            CREATE TABLE leagues (
                id INTEGER PRIMARY KEY,
                league_id INTEGER UNIQUE,
                league_name TEXT,
                country TEXT,
                type TEXT,
                logo_url TEXT,
                seasons_available TEXT,
                league_key TEXT,
                is_active INTEGER DEFAULT 1,
                last_synced_at TEXT,
                raw_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
        """)
        
        return conn
    
    def test_sync_inserts_new_leagues(self, sync_db):
        """Sync should insert new leagues."""
        repo = LeagueRepository(sync_db)
        
        leagues = [
            LeagueInfo(39, "Premier League", "England", "League", None, [2024], "premier_league"),
            LeagueInfo(140, "La Liga", "Spain", "League", None, [2024], "la_liga"),
        ]
        
        for league in leagues:
            repo.upsert_league(league)
        
        all_leagues = repo.get_all_leagues()
        assert len(all_leagues) == 2
    
    def test_sync_updates_existing_leagues(self, sync_db):
        """Sync should update existing leagues."""
        repo = LeagueRepository(sync_db)
        
        # Initial sync
        repo.upsert_league(LeagueInfo(39, "EPL", "England", "League", None, [2023], "epl"))
        
        # Second sync with updated info
        repo.upsert_league(LeagueInfo(39, "Premier League", "England", "League", None, [2023, 2024], "premier_league"))
        
        league = repo.get_league(39)
        assert league.league_name == "Premier League"
        assert 2024 in league.seasons_available
