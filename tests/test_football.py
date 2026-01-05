"""Tests for Football API client and ingestion."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from footbe_trader.football.client import FootballApiClient, FootballApiError
from footbe_trader.football.ingestion import (
    EPL_LEAGUE_ID,
    EPLIngestion,
    IngestionResult,
    generate_integrity_report,
)
from footbe_trader.football.interfaces import (
    FixtureData,
    FixtureStatus,
    LeagueData,
    SeasonData,
    StandingData,
    TeamData,
)
from footbe_trader.storage.database import Database


class TestFixtureStatus:
    """Tests for FixtureStatus enum."""

    def test_from_short_finished(self):
        """Test finished status codes."""
        assert FixtureStatus.from_short("FT") == FixtureStatus.FULL_TIME
        assert FixtureStatus.from_short("AET") == FixtureStatus.AFTER_EXTRA_TIME
        assert FixtureStatus.from_short("PEN") == FixtureStatus.AFTER_PENALTIES

    def test_from_short_live(self):
        """Test live status codes."""
        assert FixtureStatus.from_short("1H") == FixtureStatus.FIRST_HALF
        assert FixtureStatus.from_short("HT") == FixtureStatus.HALFTIME
        assert FixtureStatus.from_short("2H") == FixtureStatus.SECOND_HALF

    def test_from_short_scheduled(self):
        """Test scheduled status codes."""
        assert FixtureStatus.from_short("TBD") == FixtureStatus.TIME_TO_BE_DEFINED
        assert FixtureStatus.from_short("NS") == FixtureStatus.NOT_STARTED

    def test_from_short_unknown(self):
        """Test unknown status code returns UNKNOWN."""
        assert FixtureStatus.from_short("INVALID") == FixtureStatus.UNKNOWN

    def test_is_finished(self):
        """Test is_finished property."""
        assert FixtureStatus.FULL_TIME.is_finished is True
        assert FixtureStatus.AFTER_EXTRA_TIME.is_finished is True
        assert FixtureStatus.AFTER_PENALTIES.is_finished is True
        assert FixtureStatus.NOT_STARTED.is_finished is False
        assert FixtureStatus.FIRST_HALF.is_finished is False

    def test_is_live(self):
        """Test is_live property."""
        assert FixtureStatus.FIRST_HALF.is_live is True
        assert FixtureStatus.HALFTIME.is_live is True
        assert FixtureStatus.SECOND_HALF.is_live is True
        assert FixtureStatus.FULL_TIME.is_live is False
        assert FixtureStatus.NOT_STARTED.is_live is False

    def test_is_scheduled(self):
        """Test is_scheduled property."""
        assert FixtureStatus.NOT_STARTED.is_scheduled is True
        assert FixtureStatus.TIME_TO_BE_DEFINED.is_scheduled is True
        assert FixtureStatus.FULL_TIME.is_scheduled is False
        assert FixtureStatus.FIRST_HALF.is_scheduled is False


class TestFixtureData:
    """Tests for FixtureData parsing."""

    def test_create_fixture_data(self):
        """Test creating FixtureData from API response."""
        fixture = FixtureData(
            fixture_id=123456,
            kickoff_utc=datetime(2024, 1, 15, 15, 0, 0),
            venue="Anfield",
            status=FixtureStatus.FULL_TIME,
            league_id=39,
            season=2023,
            round="Regular Season - 20",
            home_team_id=40,
            home_team_name="Liverpool",
            away_team_id=33,
            away_team_name="Manchester United",
            home_goals=4,
            away_goals=0,
        )

        assert fixture.fixture_id == 123456
        assert fixture.home_team_name == "Liverpool"
        assert fixture.away_team_name == "Manchester United"
        assert fixture.home_goals == 4
        assert fixture.away_goals == 0
        assert fixture.status.is_finished is True

    def test_fixture_data_no_score(self):
        """Test FixtureData for upcoming match without score."""
        fixture = FixtureData(
            fixture_id=789012,
            kickoff_utc=datetime(2024, 5, 15, 15, 0, 0),
            venue="Emirates Stadium",
            status=FixtureStatus.NOT_STARTED,
            league_id=39,
            season=2023,
            round="Regular Season - 38",
            home_team_id=42,
            home_team_name="Arsenal",
            away_team_id=47,
            away_team_name="Tottenham",
            home_goals=None,
            away_goals=None,
        )

        assert fixture.home_goals is None
        assert fixture.away_goals is None
        assert fixture.status.is_scheduled is True


class TestTeamData:
    """Tests for TeamData."""

    def test_create_team_data(self):
        """Test creating TeamData."""
        team = TeamData(
            team_id=40,
            name="Liverpool",
            code="LIV",
            logo_url="https://media.api-sports.io/football/teams/40.png",
            country="England",
            founded=1892,
        )

        assert team.team_id == 40
        assert team.name == "Liverpool"
        assert team.code == "LIV"
        assert team.founded == 1892


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_empty_result(self):
        """Test empty ingestion result."""
        result = IngestionResult()
        assert result.fixtures_count == 0
        assert result.teams_count == 0
        assert result.standings_count == 0
        assert result.errors == []

    def test_result_with_data(self):
        """Test ingestion result with data via properties."""
        result = IngestionResult()
        result.fixtures_count = 380
        result.teams_count = 20
        result.standings_count = 20
        result.add_error("Some warning")
        assert result.fixtures_count == 380
        assert result.teams_count == 20
        assert len(result.errors) == 1


class TestFootballApiClient:
    """Tests for FootballApiClient with mocked responses."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.api_key = "test_api_key"
        config.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        config.rate_limit_per_minute = 100
        return config

    @pytest.fixture
    def sample_league_response(self):
        """Sample league API response."""
        return {
            "get": "leagues",
            "parameters": {"id": "39"},
            "errors": [],
            "results": 1,
            "response": [
                {
                    "league": {
                        "id": 39,
                        "name": "Premier League",
                        "type": "League",
                        "logo": "https://media.api-sports.io/football/leagues/39.png",
                    },
                    "country": {
                        "name": "England",
                        "code": "GB",
                        "flag": "https://media.api-sports.io/flags/gb.svg",
                    },
                    "seasons": [
                        {"year": 2023, "start": "2023-08-11", "end": "2024-05-19", "current": True},
                        {"year": 2022, "start": "2022-08-05", "end": "2023-05-28", "current": False},
                    ],
                }
            ],
        }

    @pytest.fixture
    def sample_fixtures_response(self):
        """Sample fixtures API response."""
        return {
            "get": "fixtures",
            "parameters": {"league": "39", "season": "2023"},
            "errors": [],
            "results": 2,
            "response": [
                {
                    "fixture": {
                        "id": 1035037,
                        "referee": "Michael Oliver, England",
                        "timezone": "UTC",
                        "date": "2023-08-11T19:00:00+00:00",
                        "timestamp": 1691780400,
                        "venue": {
                            "id": 10503,
                            "name": "Vitality Stadium",
                            "city": "Bournemouth",
                        },
                        "status": {"long": "Match Finished", "short": "FT", "elapsed": 90},
                    },
                    "league": {
                        "id": 39,
                        "name": "Premier League",
                        "country": "England",
                        "logo": "https://media.api-sports.io/football/leagues/39.png",
                        "flag": "https://media.api-sports.io/flags/gb.svg",
                        "season": 2023,
                        "round": "Regular Season - 1",
                    },
                    "teams": {
                        "home": {
                            "id": 35,
                            "name": "Bournemouth",
                            "logo": "https://media.api-sports.io/football/teams/35.png",
                            "winner": False,
                        },
                        "away": {
                            "id": 48,
                            "name": "West Ham",
                            "logo": "https://media.api-sports.io/football/teams/48.png",
                            "winner": True,
                        },
                    },
                    "goals": {"home": 1, "away": 1},
                    "score": {
                        "halftime": {"home": 0, "away": 1},
                        "fulltime": {"home": 1, "away": 1},
                    },
                },
                {
                    "fixture": {
                        "id": 1035038,
                        "referee": "Robert Jones",
                        "timezone": "UTC",
                        "date": "2023-08-12T11:30:00+00:00",
                        "timestamp": 1691839800,
                        "venue": {"id": 494, "name": "Emirates Stadium", "city": "London"},
                        "status": {"long": "Match Finished", "short": "FT", "elapsed": 90},
                    },
                    "league": {
                        "id": 39,
                        "name": "Premier League",
                        "country": "England",
                        "logo": "https://media.api-sports.io/football/leagues/39.png",
                        "flag": "https://media.api-sports.io/flags/gb.svg",
                        "season": 2023,
                        "round": "Regular Season - 1",
                    },
                    "teams": {
                        "home": {
                            "id": 42,
                            "name": "Arsenal",
                            "logo": "https://media.api-sports.io/football/teams/42.png",
                            "winner": True,
                        },
                        "away": {
                            "id": 65,
                            "name": "Nottingham Forest",
                            "logo": "https://media.api-sports.io/football/teams/65.png",
                            "winner": False,
                        },
                    },
                    "goals": {"home": 2, "away": 1},
                    "score": {
                        "halftime": {"home": 1, "away": 0},
                        "fulltime": {"home": 2, "away": 1},
                    },
                },
            ],
        }

    @pytest.fixture
    def sample_teams_response(self):
        """Sample teams API response."""
        return {
            "get": "teams",
            "parameters": {"league": "39", "season": "2023"},
            "errors": [],
            "results": 2,
            "response": [
                {
                    "team": {
                        "id": 33,
                        "name": "Manchester United",
                        "code": "MUN",
                        "country": "England",
                        "founded": 1878,
                        "logo": "https://media.api-sports.io/football/teams/33.png",
                    },
                    "venue": {"id": 556, "name": "Old Trafford", "city": "Manchester"},
                },
                {
                    "team": {
                        "id": 40,
                        "name": "Liverpool",
                        "code": "LIV",
                        "country": "England",
                        "founded": 1892,
                        "logo": "https://media.api-sports.io/football/teams/40.png",
                    },
                    "venue": {"id": 550, "name": "Anfield", "city": "Liverpool"},
                },
            ],
        }

    def test_parse_league_response(self, sample_league_response):
        """Test parsing league API response."""
        response_data = sample_league_response["response"][0]

        league = LeagueData(
            league_id=response_data["league"]["id"],
            name=response_data["league"]["name"],
            logo_url=response_data["league"]["logo"],
            country=response_data["country"]["name"],
        )

        assert league.league_id == 39
        assert league.name == "Premier League"
        assert league.country == "England"

    def test_parse_seasons_response(self, sample_league_response):
        """Test parsing seasons from league response."""
        response_data = sample_league_response["response"][0]
        seasons = []

        for s in response_data["seasons"]:
            season = SeasonData(
                year=s["year"],
                current=s["current"],
            )
            seasons.append(season)

        assert len(seasons) == 2
        assert seasons[0].year == 2023
        assert seasons[0].current is True
        assert seasons[1].year == 2022
        assert seasons[1].current is False

    def test_parse_fixtures_response(self, sample_fixtures_response):
        """Test parsing fixtures from API response."""
        fixtures = []

        for item in sample_fixtures_response["response"]:
            fixture_data = item["fixture"]
            league_data = item["league"]
            teams_data = item["teams"]
            goals_data = item["goals"]

            fixture = FixtureData(
                fixture_id=fixture_data["id"],
                kickoff_utc=datetime.fromisoformat(fixture_data["date"].replace("+00:00", "")),
                venue=fixture_data["venue"]["name"],
                status=FixtureStatus.from_short(fixture_data["status"]["short"]),
                league_id=league_data["id"],
                season=league_data["season"],
                round=league_data["round"],
                home_team_id=teams_data["home"]["id"],
                home_team_name=teams_data["home"]["name"],
                away_team_id=teams_data["away"]["id"],
                away_team_name=teams_data["away"]["name"],
                home_goals=goals_data.get("home"),
                away_goals=goals_data.get("away"),
            )
            fixtures.append(fixture)

        assert len(fixtures) == 2
        assert fixtures[0].fixture_id == 1035037
        assert fixtures[0].home_team_name == "Bournemouth"
        assert fixtures[0].away_team_name == "West Ham"
        assert fixtures[0].home_goals == 1
        assert fixtures[0].away_goals == 1

        assert fixtures[1].fixture_id == 1035038
        assert fixtures[1].home_team_name == "Arsenal"
        assert fixtures[1].home_goals == 2

    def test_parse_teams_response(self, sample_teams_response):
        """Test parsing teams from API response."""
        teams = []

        for item in sample_teams_response["response"]:
            team_data = item["team"]
            team = TeamData(
                team_id=team_data["id"],
                name=team_data["name"],
                code=team_data.get("code", ""),
                logo_url=team_data.get("logo", ""),
                country=team_data.get("country", ""),
                founded=team_data.get("founded"),
            )
            teams.append(team)

        assert len(teams) == 2
        assert teams[0].team_id == 33
        assert teams[0].name == "Manchester United"
        assert teams[0].code == "MUN"
        assert teams[1].team_id == 40
        assert teams[1].name == "Liverpool"


class TestDatabaseFixtureOperations:
    """Tests for database operations with fixtures."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a test database."""
        db = Database(tmp_path / "test.db")
        db.connect()
        db.migrate()
        yield db
        db.close()

    def test_upsert_team(self, db):
        """Test upserting a team."""
        from footbe_trader.storage.models import Team

        team = Team(
            team_id=40,
            name="Liverpool",
            code="LIV",
            logo_url="https://example.com/liv.png",
            country="England",
            founded=1892,
        )

        db.upsert_team(team)
        result = db.get_team(40)

        assert result is not None
        assert result.name == "Liverpool"
        assert result.code == "LIV"

    def test_upsert_team_update(self, db):
        """Test upserting updates existing team."""
        from footbe_trader.storage.models import Team

        team1 = Team(team_id=40, name="Liverpool FC")
        db.upsert_team(team1)

        team2 = Team(team_id=40, name="Liverpool Football Club")
        db.upsert_team(team2)

        result = db.get_team(40)
        assert result.name == "Liverpool Football Club"

    def test_upsert_fixture(self, db):
        """Test upserting a fixture."""
        from footbe_trader.storage.models import FixtureV2, Team

        # First add teams
        db.upsert_team(Team(team_id=40, name="Liverpool"))
        db.upsert_team(Team(team_id=33, name="Manchester United"))

        fixture = FixtureV2(
            fixture_id=123456,
            kickoff_utc=datetime(2024, 1, 15, 15, 0, 0),
            venue="Anfield",
            status="FT",
            league_id=39,
            season=2023,
            round="Regular Season - 20",
            home_team_id=40,
            away_team_id=33,
            home_goals=4,
            away_goals=0,
        )

        db.upsert_fixture(fixture)
        result = db.get_fixture_by_id(123456)

        assert result is not None
        assert result.home_goals == 4
        assert result.away_goals == 0
        assert result.status == "FT"

    def test_upsert_fixture_update_score(self, db):
        """Test upserting updates fixture score."""
        from footbe_trader.storage.models import FixtureV2, Team

        db.upsert_team(Team(team_id=40, name="Liverpool"))
        db.upsert_team(Team(team_id=33, name="Manchester United"))

        # First insert without score
        fixture1 = FixtureV2(
            fixture_id=123456,
            kickoff_utc=datetime(2024, 1, 15, 15, 0, 0),
            status="NS",
            league_id=39,
            season=2023,
            round="Regular Season - 20",
            home_team_id=40,
            away_team_id=33,
        )
        db.upsert_fixture(fixture1)

        # Update with score
        fixture2 = FixtureV2(
            fixture_id=123456,
            kickoff_utc=datetime(2024, 1, 15, 15, 0, 0),
            status="FT",
            league_id=39,
            season=2023,
            round="Regular Season - 20",
            home_team_id=40,
            away_team_id=33,
            home_goals=3,
            away_goals=1,
        )
        db.upsert_fixture(fixture2)

        result = db.get_fixture_by_id(123456)
        assert result.status == "FT"
        assert result.home_goals == 3
        assert result.away_goals == 1

    def test_get_fixtures_by_season(self, db):
        """Test getting fixtures by season."""
        from footbe_trader.storage.models import FixtureV2, Team

        db.upsert_team(Team(team_id=1, name="Team A"))
        db.upsert_team(Team(team_id=2, name="Team B"))

        for i in range(5):
            fixture = FixtureV2(
                fixture_id=1000 + i,
                kickoff_utc=datetime(2024, 1, 10 + i, 15, 0, 0),
                status="FT",
                league_id=39,
                season=2023,
                round=f"Round {i+1}",
                home_team_id=1,
                away_team_id=2,
            )
            db.upsert_fixture(fixture)

        fixtures = db.get_fixtures_by_season(2023)
        assert len(fixtures) == 5

    def test_get_fixtures_count_by_season(self, db):
        """Test getting fixture counts per season."""
        from footbe_trader.storage.models import FixtureV2, Team

        db.upsert_team(Team(team_id=1, name="Team A"))
        db.upsert_team(Team(team_id=2, name="Team B"))

        # Add fixtures for two seasons
        for i in range(3):
            db.upsert_fixture(
                FixtureV2(
                    fixture_id=1000 + i,
                    kickoff_utc=datetime(2023, 1, 10 + i, 15, 0, 0),
                    status="FT",
                    league_id=39,
                    season=2022,
                    round=f"Round {i+1}",
                    home_team_id=1,
                    away_team_id=2,
                )
            )

        for i in range(5):
            db.upsert_fixture(
                FixtureV2(
                    fixture_id=2000 + i,
                    kickoff_utc=datetime(2024, 1, 10 + i, 15, 0, 0),
                    status="FT",
                    league_id=39,
                    season=2023,
                    round=f"Round {i+1}",
                    home_team_id=1,
                    away_team_id=2,
                )
            )

        counts = db.get_fixtures_count_by_season()
        assert counts[2022] == 3
        assert counts[2023] == 5

    def test_ingestion_log(self, db):
        """Test ingestion logging."""
        db.log_ingestion("fixtures", 2023, 380)

        log = db.get_ingestion_log("fixtures", 2023)
        assert log is not None
        assert log.record_count == 380

    def test_get_ingested_seasons(self, db):
        """Test getting list of ingested seasons."""
        db.log_ingestion("fixtures", 2022, 380)
        db.log_ingestion("fixtures", 2023, 380)
        db.log_ingestion("fixtures", 2024, 200)

        seasons = db.get_ingested_seasons("fixtures")
        assert 2022 in seasons
        assert 2023 in seasons
        assert 2024 in seasons


class TestIntegrityReport:
    """Tests for integrity report generation."""

    @pytest.fixture
    def db_with_data(self, tmp_path):
        """Create a test database with sample data."""
        from footbe_trader.storage.models import FixtureV2, Team

        db = Database(tmp_path / "test.db")
        db.connect()
        db.migrate()

        # Add teams
        for i in range(4):
            db.upsert_team(Team(team_id=i + 1, name=f"Team {i + 1}"))

        # Add fixtures for 2023 (some with scores, some without)
        for i in range(10):
            db.upsert_fixture(
                FixtureV2(
                    fixture_id=1000 + i,
                    kickoff_utc=datetime(2023, 8, 10 + i, 15, 0, 0),
                    status="FT",
                    league_id=39,
                    season=2023,
                    round=f"Round {i+1}",
                    home_team_id=(i % 4) + 1,
                    away_team_id=((i + 1) % 4) + 1,
                    home_goals=2 if i % 2 == 0 else None,  # Half have scores
                    away_goals=1 if i % 2 == 0 else None,
                )
            )

        yield db
        db.close()

    def test_generate_integrity_report(self, db_with_data):
        """Test generating integrity report."""
        report = generate_integrity_report(db_with_data)

        assert "generated_at" in report
        assert report["total_fixtures"] == 10
        assert report["total_teams"] == 4
        assert 2023 in report["fixtures_per_season"]
        assert report["fixtures_per_season"][2023] == 10
