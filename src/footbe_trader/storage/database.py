"""Database connection and access layer."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.storage.models import (
    BacktestEquity,
    BacktestTrade,
    FixtureV2,
    HistoricalSnapshot,
    IngestionLog,
    OrderbookSnapshot,
    Run,
    Snapshot,
    SnapshotSession,
    StandingSnapshot,
    StrategyBacktest,
    Team,
)
from footbe_trader.storage.schema import MIGRATIONS, SCHEMA_VERSION

logger = get_logger(__name__)


class Database:
    """SQLite database connection and operations."""

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open database connection and ensure schema is applied."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")

        logger.info("database_connected", path=str(self.db_path))

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("database_closed", path=str(self.db_path))

    @property
    def connection(self) -> sqlite3.Connection:
        """Get active connection, raising if not connected."""
        if self._connection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection

    def migrate(self) -> None:
        """Apply database migrations."""
        conn = self.connection
        cursor = conn.cursor()

        # Get current schema version
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            current_version = 0
        else:
            cursor.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            current_version = row[0] if row[0] is not None else 0

        # Apply pending migrations
        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                logger.info("applying_migration", version=version)
                cursor.executescript(MIGRATIONS[version])
                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,),
                )
                conn.commit()
                logger.info("migration_applied", version=version)

        logger.info(
            "migrations_complete",
            from_version=current_version,
            to_version=SCHEMA_VERSION,
        )

    def get_schema_version(self) -> int:
        """Get current schema version."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            return 0
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row[0] is not None else 0

    # --- Run operations ---

    def create_run(self, run: Run) -> int:
        """Create a new run record.

        Args:
            run: Run data to insert.

        Returns:
            ID of created run.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO runs (run_type, status, config_hash, started_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run.run_type,
                run.status,
                run.config_hash,
                run.started_at.isoformat(),
                json.dumps(run.metadata),
            ),
        )
        self.connection.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        logger.info("run_created", run_id=run_id, run_type=run.run_type)
        return run_id

    def complete_run(
        self, run_id: int, status: str = "completed", error_message: str | None = None
    ) -> None:
        """Mark a run as complete.

        Args:
            run_id: ID of run to complete.
            status: Final status (completed, failed).
            error_message: Error message if failed.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE runs
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ?
            """,
            (status, utc_now().isoformat(), error_message, run_id),
        )
        self.connection.commit()
        logger.info("run_completed", run_id=run_id, status=status)

    def get_run(self, run_id: int) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID.

        Returns:
            Run object or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        """Convert database row to Run object."""
        return Run(
            id=row["id"],
            run_type=row["run_type"],
            status=row["status"],
            config_hash=row["config_hash"] or "",
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # --- Snapshot operations ---

    def create_snapshot(self, snapshot: Snapshot) -> int:
        """Create a new snapshot record.

        Args:
            snapshot: Snapshot data to insert.

        Returns:
            ID of created snapshot.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO snapshots (run_id, snapshot_type, data, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                snapshot.run_id,
                snapshot.snapshot_type,
                json.dumps(snapshot.data),
                snapshot.created_at.isoformat(),
            ),
        )
        self.connection.commit()
        snapshot_id = cursor.lastrowid
        assert snapshot_id is not None
        logger.info(
            "snapshot_created",
            snapshot_id=snapshot_id,
            snapshot_type=snapshot.snapshot_type,
        )
        return snapshot_id

    def get_snapshot(self, snapshot_id: int) -> Snapshot | None:
        """Get a snapshot by ID.

        Args:
            snapshot_id: Snapshot ID.

        Returns:
            Snapshot object or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_snapshots_for_run(self, run_id: int) -> list[Snapshot]:
        """Get all snapshots for a run.

        Args:
            run_id: Run ID.

        Returns:
            List of snapshots.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM snapshots WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        )
        return [self._row_to_snapshot(row) for row in cursor.fetchall()]

    def _row_to_snapshot(self, row: sqlite3.Row) -> Snapshot:
        """Convert database row to Snapshot object."""
        return Snapshot(
            id=row["id"],
            run_id=row["run_id"],
            snapshot_type=row["snapshot_type"],
            data=json.loads(row["data"]) if row["data"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Orderbook Snapshot operations ---

    def create_orderbook_snapshot(self, snapshot: OrderbookSnapshot) -> int:
        """Create a new orderbook snapshot record.

        Args:
            snapshot: OrderbookSnapshot data to insert.

        Returns:
            ID of created snapshot.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO orderbook_snapshots 
            (timestamp, ticker, best_bid, best_ask, mid, spread, 
             bid_volume, ask_volume, volume, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.timestamp.isoformat(),
                snapshot.ticker,
                snapshot.best_bid,
                snapshot.best_ask,
                snapshot.mid,
                snapshot.spread,
                snapshot.bid_volume,
                snapshot.ask_volume,
                snapshot.volume,
                json.dumps(snapshot.raw_json),
                snapshot.created_at.isoformat(),
            ),
        )
        self.connection.commit()
        snapshot_id = cursor.lastrowid
        assert snapshot_id is not None
        logger.info(
            "orderbook_snapshot_created",
            snapshot_id=snapshot_id,
            ticker=snapshot.ticker,
        )
        return snapshot_id

    def get_orderbook_snapshots(
        self,
        ticker: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[OrderbookSnapshot]:
        """Get orderbook snapshots, optionally filtered.

        Args:
            ticker: Filter by ticker.
            since: Filter to snapshots after this time.
            limit: Maximum number of snapshots to return.

        Returns:
            List of orderbook snapshots.
        """
        cursor = self.connection.cursor()
        query = "SELECT * FROM orderbook_snapshots WHERE 1=1"
        params: list[Any] = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [self._row_to_orderbook_snapshot(row) for row in cursor.fetchall()]

    def _row_to_orderbook_snapshot(self, row: sqlite3.Row) -> OrderbookSnapshot:
        """Convert database row to OrderbookSnapshot object."""
        return OrderbookSnapshot(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            ticker=row["ticker"],
            best_bid=row["best_bid"],
            best_ask=row["best_ask"],
            mid=row["mid"],
            spread=row["spread"],
            bid_volume=row["bid_volume"],
            ask_volume=row["ask_volume"],
            volume=row["volume"],
            raw_json=json.loads(row["raw_json"]) if row["raw_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Team operations ---

    def upsert_team(self, team: Team) -> int:
        """Insert or update a team record.

        Args:
            team: Team data to upsert.

        Returns:
            ID of the team record.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO teams 
            (team_id, name, code, country, logo_url, founded, 
             venue_name, venue_capacity, raw_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                name = excluded.name,
                code = excluded.code,
                country = excluded.country,
                logo_url = excluded.logo_url,
                founded = excluded.founded,
                venue_name = excluded.venue_name,
                venue_capacity = excluded.venue_capacity,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
            """,
            (
                team.team_id,
                team.name,
                team.code,
                team.country,
                team.logo_url,
                team.founded,
                team.venue_name,
                team.venue_capacity,
                json.dumps(team.raw_json),
                utc_now().isoformat(),
            ),
        )
        self.connection.commit()
        # Get the id (either inserted or existing)
        cursor.execute("SELECT id FROM teams WHERE team_id = ?", (team.team_id,))
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_team(self, team_id: int) -> Team | None:
        """Get a team by API-Football team ID.

        Args:
            team_id: API-Football team ID.

        Returns:
            Team object or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_team(row)

    def get_all_teams(self) -> list[Team]:
        """Get all teams.

        Returns:
            List of all teams.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM teams ORDER BY name")
        return [self._row_to_team(row) for row in cursor.fetchall()]

    def _row_to_team(self, row: sqlite3.Row) -> Team:
        """Convert database row to Team object."""
        return Team(
            id=row["id"],
            team_id=row["team_id"],
            name=row["name"],
            code=row["code"] or "",
            country=row["country"] or "",
            logo_url=row["logo_url"] or "",
            founded=row["founded"],
            venue_name=row["venue_name"] or "",
            venue_capacity=row["venue_capacity"],
            raw_json=json.loads(row["raw_json"]) if row["raw_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # --- Fixture V2 operations ---

    def upsert_fixture(self, fixture: FixtureV2) -> int:
        """Insert or update a fixture record.

        Args:
            fixture: Fixture data to upsert.

        Returns:
            ID of the fixture record.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO fixtures_v2 
            (fixture_id, league_id, season, round, home_team_id, away_team_id,
             kickoff_utc, status, home_goals, away_goals, venue, referee,
             raw_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                league_id = excluded.league_id,
                season = excluded.season,
                round = excluded.round,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                kickoff_utc = excluded.kickoff_utc,
                status = excluded.status,
                home_goals = excluded.home_goals,
                away_goals = excluded.away_goals,
                venue = excluded.venue,
                referee = excluded.referee,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
            """,
            (
                fixture.fixture_id,
                fixture.league_id,
                fixture.season,
                fixture.round,
                fixture.home_team_id,
                fixture.away_team_id,
                fixture.kickoff_utc.isoformat() if fixture.kickoff_utc else None,
                fixture.status,
                fixture.home_goals,
                fixture.away_goals,
                fixture.venue,
                fixture.referee,
                json.dumps(fixture.raw_json),
                utc_now().isoformat(),
            ),
        )
        self.connection.commit()
        # Get the id (either inserted or existing)
        cursor.execute("SELECT id FROM fixtures_v2 WHERE fixture_id = ?", (fixture.fixture_id,))
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_fixture_by_id(self, fixture_id: int) -> FixtureV2 | None:
        """Get a fixture by API-Football fixture ID.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            Fixture object or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM fixtures_v2 WHERE fixture_id = ?", (fixture_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_fixture_v2(row)

    def get_fixtures_by_season(
        self,
        season: int,
        status: str | None = None,
    ) -> list[FixtureV2]:
        """Get fixtures for a season.

        Args:
            season: Season year.
            status: Optional status filter.

        Returns:
            List of fixtures.
        """
        cursor = self.connection.cursor()
        query = "SELECT * FROM fixtures_v2 WHERE season = ?"
        params: list[Any] = [season]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY kickoff_utc"
        cursor.execute(query, params)
        return [self._row_to_fixture_v2(row) for row in cursor.fetchall()]

    def get_fixtures_count_by_season(self) -> dict[int, int]:
        """Get count of fixtures per season.

        Returns:
            Dictionary mapping season to fixture count.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT season, COUNT(*) as count FROM fixtures_v2 GROUP BY season ORDER BY season"
        )
        return {row["season"]: row["count"] for row in cursor.fetchall()}

    def get_missing_scores_by_season(self) -> dict[int, tuple[int, int]]:
        """Get count of fixtures with missing scores per season.

        Returns:
            Dictionary mapping season to (missing, total) counts.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT 
                season,
                SUM(CASE WHEN status = 'FT' AND (home_goals IS NULL OR away_goals IS NULL) THEN 1 ELSE 0 END) as missing,
                SUM(CASE WHEN status = 'FT' THEN 1 ELSE 0 END) as total
            FROM fixtures_v2 
            GROUP BY season 
            ORDER BY season
            """
        )
        return {row["season"]: (row["missing"], row["total"]) for row in cursor.fetchall()}

    def get_teams_by_season(self, season: int) -> list[int]:
        """Get unique team IDs for a season.

        Args:
            season: Season year.

        Returns:
            List of team IDs.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT DISTINCT team_id FROM (
                SELECT home_team_id as team_id FROM fixtures_v2 WHERE season = ?
                UNION
                SELECT away_team_id as team_id FROM fixtures_v2 WHERE season = ?
            )
            ORDER BY team_id
            """,
            (season, season),
        )
        return [row[0] for row in cursor.fetchall()]

    def _row_to_fixture_v2(self, row: sqlite3.Row) -> FixtureV2:
        """Convert database row to FixtureV2 object."""
        return FixtureV2(
            id=row["id"],
            fixture_id=row["fixture_id"],
            league_id=row["league_id"],
            season=row["season"],
            round=row["round"] or "",
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_utc=(
                datetime.fromisoformat(row["kickoff_utc"])
                if row["kickoff_utc"]
                else None
            ),
            status=row["status"],
            home_goals=row["home_goals"],
            away_goals=row["away_goals"],
            venue=row["venue"] or "",
            referee=row["referee"] or "",
            raw_json=json.loads(row["raw_json"]) if row["raw_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # --- Standing Snapshot operations ---

    def upsert_standing_snapshot(self, standing: StandingSnapshot) -> int:
        """Insert or update a standing snapshot.

        Args:
            standing: Standing snapshot data.

        Returns:
            ID of the standing snapshot.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO standings_snapshots 
            (league_id, season, snapshot_date, team_id, rank, points, played,
             wins, draws, losses, goals_for, goals_against, goal_difference,
             form, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, snapshot_date, team_id) DO UPDATE SET
                rank = excluded.rank,
                points = excluded.points,
                played = excluded.played,
                wins = excluded.wins,
                draws = excluded.draws,
                losses = excluded.losses,
                goals_for = excluded.goals_for,
                goals_against = excluded.goals_against,
                goal_difference = excluded.goal_difference,
                form = excluded.form,
                raw_json = excluded.raw_json
            """,
            (
                standing.league_id,
                standing.season,
                standing.snapshot_date,
                standing.team_id,
                standing.rank,
                standing.points,
                standing.played,
                standing.wins,
                standing.draws,
                standing.losses,
                standing.goals_for,
                standing.goals_against,
                standing.goal_difference,
                standing.form,
                json.dumps(standing.raw_json),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    # --- Ingestion Log operations ---

    def log_ingestion(
        self,
        data_type: str,
        season: int,
        record_count: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> int:
        """Log an ingestion run.

        Args:
            data_type: Type of data ingested ('fixtures', 'teams', 'standings').
            season: Season year.
            record_count: Number of records processed.
            status: Ingestion status.
            error_message: Error message if failed.

        Returns:
            ID of the log entry.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO ingestion_log 
            (data_type, season, record_count, status, error_message)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(data_type, season) DO UPDATE SET
                ingested_at = datetime('now'),
                record_count = excluded.record_count,
                status = excluded.status,
                error_message = excluded.error_message
            """,
            (data_type, season, record_count, status, error_message),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def get_ingestion_log(self, data_type: str, season: int) -> IngestionLog | None:
        """Get ingestion log for a data type and season.

        Args:
            data_type: Type of data.
            season: Season year.

        Returns:
            IngestionLog or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM ingestion_log WHERE data_type = ? AND season = ?",
            (data_type, season),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return IngestionLog(
            id=row["id"],
            data_type=row["data_type"],
            season=row["season"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            record_count=row["record_count"],
            status=row["status"],
            error_message=row["error_message"],
        )

    def get_ingested_seasons(self, data_type: str) -> list[int]:
        """Get list of seasons that have been ingested.

        Args:
            data_type: Type of data.

        Returns:
            List of season years.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT season FROM ingestion_log WHERE data_type = ? AND status = 'success' ORDER BY season",
            (data_type,),
        )
        return [row["season"] for row in cursor.fetchall()]

    # --- Historical Snapshot operations ---

    def create_snapshot_session(self, session: "SnapshotSession") -> int:
        """Create a new snapshot collection session.

        Args:
            session: Session data to insert.

        Returns:
            ID of created session.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO snapshot_sessions 
            (session_id, started_at, status, interval_minutes, 
             fixtures_tracked, snapshots_collected, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.started_at.isoformat(),
                session.status,
                session.interval_minutes,
                session.fixtures_tracked,
                session.snapshots_collected,
                json.dumps(session.config_json),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def update_snapshot_session(self, session: "SnapshotSession") -> None:
        """Update an existing snapshot session.

        Args:
            session: Session data to update.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE snapshot_sessions SET
                ended_at = ?,
                status = ?,
                fixtures_tracked = ?,
                snapshots_collected = ?,
                error_message = ?
            WHERE session_id = ?
            """,
            (
                session.ended_at.isoformat() if session.ended_at else None,
                session.status,
                session.fixtures_tracked,
                session.snapshots_collected,
                session.error_message,
                session.session_id,
            ),
        )
        self.connection.commit()

    def create_historical_snapshot(self, snapshot: "HistoricalSnapshot") -> int:
        """Create a historical snapshot record.

        Args:
            snapshot: Snapshot data to insert.

        Returns:
            ID of created snapshot.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO historical_snapshots 
            (session_id, fixture_id, ticker, outcome, timestamp,
             best_bid, best_ask, mid, spread, bid_volume, ask_volume,
             yes_price, no_price, volume_24h, open_interest,
             model_prob, model_version, raw_orderbook_json, raw_market_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.session_id,
                snapshot.fixture_id,
                snapshot.ticker,
                snapshot.outcome,
                snapshot.timestamp.isoformat(),
                snapshot.best_bid,
                snapshot.best_ask,
                snapshot.mid,
                snapshot.spread,
                snapshot.bid_volume,
                snapshot.ask_volume,
                snapshot.yes_price,
                snapshot.no_price,
                snapshot.volume_24h,
                snapshot.open_interest,
                snapshot.model_prob,
                snapshot.model_version,
                json.dumps(snapshot.raw_orderbook_json),
                json.dumps(snapshot.raw_market_json),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def get_historical_snapshots(
        self,
        fixture_id: int | None = None,
        session_id: str | None = None,
        ticker: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 10000,
    ) -> list["HistoricalSnapshot"]:
        """Get historical snapshots with optional filters.

        Args:
            fixture_id: Filter by fixture ID.
            session_id: Filter by session ID.
            ticker: Filter by ticker.
            since: Filter to snapshots after this time.
            until: Filter to snapshots before this time.
            limit: Maximum number of snapshots.

        Returns:
            List of historical snapshots.
        """
        cursor = self.connection.cursor()
        query = "SELECT * FROM historical_snapshots WHERE 1=1"
        params: list[Any] = []

        if fixture_id is not None:
            query += " AND fixture_id = ?"
            params.append(fixture_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [self._row_to_historical_snapshot(row) for row in cursor.fetchall()]

    def _row_to_historical_snapshot(self, row: sqlite3.Row) -> "HistoricalSnapshot":
        """Convert database row to HistoricalSnapshot."""
        return HistoricalSnapshot(
            id=row["id"],
            session_id=row["session_id"],
            fixture_id=row["fixture_id"],
            ticker=row["ticker"],
            outcome=row["outcome"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            best_bid=row["best_bid"],
            best_ask=row["best_ask"],
            mid=row["mid"],
            spread=row["spread"],
            bid_volume=row["bid_volume"],
            ask_volume=row["ask_volume"],
            yes_price=row["yes_price"],
            no_price=row["no_price"],
            volume_24h=row["volume_24h"],
            open_interest=row["open_interest"],
            model_prob=row["model_prob"],
            model_version=row["model_version"],
            raw_orderbook_json=json.loads(row["raw_orderbook_json"]) if row["raw_orderbook_json"] else {},
            raw_market_json=json.loads(row["raw_market_json"]) if row["raw_market_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Strategy Backtest operations ---

    def create_strategy_backtest(self, backtest: "StrategyBacktest") -> int:
        """Create a strategy backtest record.

        Args:
            backtest: Backtest data to insert.

        Returns:
            ID of created backtest.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO strategy_backtests 
            (backtest_id, started_at, status, strategy_config_hash, strategy_config_json,
             snapshot_start, snapshot_end, fixtures_included, initial_bankroll)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                backtest.backtest_id,
                backtest.started_at.isoformat(),
                backtest.status,
                backtest.strategy_config_hash,
                json.dumps(backtest.strategy_config_json),
                backtest.snapshot_start.isoformat() if backtest.snapshot_start else None,
                backtest.snapshot_end.isoformat() if backtest.snapshot_end else None,
                json.dumps(backtest.fixtures_included),
                backtest.initial_bankroll,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def update_strategy_backtest(self, backtest: "StrategyBacktest") -> None:
        """Update a strategy backtest record.

        Args:
            backtest: Backtest data to update.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE strategy_backtests SET
                completed_at = ?,
                status = ?,
                snapshot_start = ?,
                snapshot_end = ?,
                fixtures_included = ?,
                final_bankroll = ?,
                total_return = ?,
                max_drawdown = ?,
                sharpe_ratio = ?,
                total_trades = ?,
                winning_trades = ?,
                losing_trades = ?,
                avg_hold_time_minutes = ?,
                per_outcome_stats_json = ?,
                per_fixture_stats_json = ?,
                edge_calibration_json = ?,
                results_json = ?,
                error_message = ?
            WHERE backtest_id = ?
            """,
            (
                backtest.completed_at.isoformat() if backtest.completed_at else None,
                backtest.status,
                backtest.snapshot_start.isoformat() if backtest.snapshot_start else None,
                backtest.snapshot_end.isoformat() if backtest.snapshot_end else None,
                json.dumps(backtest.fixtures_included),
                backtest.final_bankroll,
                backtest.total_return,
                backtest.max_drawdown,
                backtest.sharpe_ratio,
                backtest.total_trades,
                backtest.winning_trades,
                backtest.losing_trades,
                backtest.avg_hold_time_minutes,
                json.dumps(backtest.per_outcome_stats_json),
                json.dumps(backtest.per_fixture_stats_json),
                json.dumps(backtest.edge_calibration_json),
                json.dumps(backtest.results_json),
                backtest.error_message,
                backtest.backtest_id,
            ),
        )
        self.connection.commit()

    def create_backtest_trade(self, trade: "BacktestTrade") -> int:
        """Create a backtest trade record.

        Args:
            trade: Trade data to insert.

        Returns:
            ID of created trade.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO backtest_trades 
            (backtest_id, trade_id, fixture_id, ticker, outcome,
             entry_timestamp, entry_price, entry_quantity, entry_edge,
             entry_model_prob, entry_reason, exit_timestamp, exit_price,
             exit_quantity, exit_reason, realized_pnl, return_pct,
             hold_time_minutes, mtm_history_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.backtest_id,
                trade.trade_id,
                trade.fixture_id,
                trade.ticker,
                trade.outcome,
                trade.entry_timestamp.isoformat(),
                trade.entry_price,
                trade.entry_quantity,
                trade.entry_edge,
                trade.entry_model_prob,
                trade.entry_reason,
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                trade.exit_price,
                trade.exit_quantity,
                trade.exit_reason,
                trade.realized_pnl,
                trade.return_pct,
                trade.hold_time_minutes,
                json.dumps(trade.mtm_history_json),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def create_backtest_equity(self, equity: "BacktestEquity") -> int:
        """Create a backtest equity point.

        Args:
            equity: Equity data to insert.

        Returns:
            ID of created record.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO backtest_equity 
            (backtest_id, timestamp, bankroll, total_exposure, position_count,
             realized_pnl, unrealized_pnl, total_pnl, drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                equity.backtest_id,
                equity.timestamp.isoformat(),
                equity.bankroll,
                equity.total_exposure,
                equity.position_count,
                equity.realized_pnl,
                equity.unrealized_pnl,
                equity.total_pnl,
                equity.drawdown,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid or 0

    def get_backtest_trades(self, backtest_id: str) -> list["BacktestTrade"]:
        """Get all trades for a backtest.

        Args:
            backtest_id: Backtest ID.

        Returns:
            List of trades.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM backtest_trades WHERE backtest_id = ? ORDER BY entry_timestamp",
            (backtest_id,),
        )
        return [self._row_to_backtest_trade(row) for row in cursor.fetchall()]

    def _row_to_backtest_trade(self, row: sqlite3.Row) -> "BacktestTrade":
        """Convert database row to BacktestTrade."""
        return BacktestTrade(
            id=row["id"],
            backtest_id=row["backtest_id"],
            trade_id=row["trade_id"],
            fixture_id=row["fixture_id"],
            ticker=row["ticker"],
            outcome=row["outcome"],
            entry_timestamp=datetime.fromisoformat(row["entry_timestamp"]),
            entry_price=row["entry_price"],
            entry_quantity=row["entry_quantity"],
            entry_edge=row["entry_edge"],
            entry_model_prob=row["entry_model_prob"],
            entry_reason=row["entry_reason"],
            exit_timestamp=datetime.fromisoformat(row["exit_timestamp"]) if row["exit_timestamp"] else None,
            exit_price=row["exit_price"],
            exit_quantity=row["exit_quantity"],
            exit_reason=row["exit_reason"],
            realized_pnl=row["realized_pnl"],
            return_pct=row["return_pct"],
            hold_time_minutes=row["hold_time_minutes"],
            mtm_history_json=json.loads(row["mtm_history_json"]) if row["mtm_history_json"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_backtest_equity(self, backtest_id: str) -> list["BacktestEquity"]:
        """Get equity curve for a backtest.

        Args:
            backtest_id: Backtest ID.

        Returns:
            List of equity points.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM backtest_equity WHERE backtest_id = ? ORDER BY timestamp",
            (backtest_id,),
        )
        return [self._row_to_backtest_equity(row) for row in cursor.fetchall()]

    def _row_to_backtest_equity(self, row: sqlite3.Row) -> "BacktestEquity":
        """Convert database row to BacktestEquity."""
        return BacktestEquity(
            id=row["id"],
            backtest_id=row["backtest_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bankroll=row["bankroll"],
            total_exposure=row["total_exposure"],
            position_count=row["position_count"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            total_pnl=row["total_pnl"],
            drawdown=row["drawdown"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Generic operations ---

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute raw SQL.

        Args:
            sql: SQL statement.
            params: Query parameters.

        Returns:
            Cursor with results.
        """
        return self.connection.execute(sql, params)

    def commit(self) -> None:
        """Commit current transaction."""
        self.connection.commit()

    def __enter__(self) -> "Database":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
