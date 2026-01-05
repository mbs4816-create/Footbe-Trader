"""Tests for database migrations and operations."""

from pathlib import Path

from footbe_trader.common.time_utils import utc_now
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import Run, Snapshot
from footbe_trader.storage.schema import SCHEMA_VERSION


class TestDatabaseMigrations:
    """Test database migrations."""

    def test_migrate_creates_tables(self, temp_dir: Path):
        """Test that migrations create all required tables."""
        db_path = temp_dir / "test_migrate.db"
        db = Database(db_path)
        db.connect()
        db.migrate()

        # Check all tables exist
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "schema_version",
            "runs",
            "fixtures",
            "markets",
            "snapshots",
            "predictions",
            "orders",
            "fills",
            "positions",
            "pnl_marks",
        }
        assert expected_tables.issubset(tables)

        db.close()

    def test_schema_version_tracked(self, temp_dir: Path):
        """Test that schema version is properly tracked."""
        db_path = temp_dir / "test_version.db"
        db = Database(db_path)
        db.connect()

        # Before migration
        assert db.get_schema_version() == 0

        # After migration
        db.migrate()
        assert db.get_schema_version() == SCHEMA_VERSION

        db.close()

    def test_migrate_is_idempotent(self, temp_dir: Path):
        """Test that running migrations multiple times is safe."""
        db_path = temp_dir / "test_idempotent.db"
        db = Database(db_path)
        db.connect()

        # Run migrations twice
        db.migrate()
        version_after_first = db.get_schema_version()

        db.migrate()
        version_after_second = db.get_schema_version()

        assert version_after_first == version_after_second == SCHEMA_VERSION

        db.close()

    def test_indexes_created(self, temp_dir: Path):
        """Test that indexes are created."""
        db_path = temp_dir / "test_indexes.db"
        db = Database(db_path)
        db.connect()
        db.migrate()

        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        # Check key indexes exist
        assert "idx_runs_started_at" in indexes
        assert "idx_fixtures_external_id" in indexes
        assert "idx_markets_external_id" in indexes
        assert "idx_snapshots_run_id" in indexes

        db.close()


class TestDatabaseOperations:
    """Test database CRUD operations."""

    def test_create_run(self, db: Database):
        """Test creating a run record."""
        run = Run(
            run_type="heartbeat",
            status="running",
            config_hash="abc123",
            started_at=utc_now(),
            metadata={"test": True},
        )

        run_id = db.create_run(run)

        assert run_id is not None
        assert run_id > 0

    def test_get_run(self, db: Database):
        """Test retrieving a run record."""
        run = Run(
            run_type="trading",
            status="running",
            config_hash="xyz789",
            started_at=utc_now(),
            metadata={"mode": "live"},
        )
        run_id = db.create_run(run)

        retrieved = db.get_run(run_id)

        assert retrieved is not None
        assert retrieved.id == run_id
        assert retrieved.run_type == "trading"
        assert retrieved.status == "running"
        assert retrieved.config_hash == "xyz789"

    def test_complete_run(self, db: Database):
        """Test completing a run record."""
        run = Run(run_type="heartbeat", status="running")
        run_id = db.create_run(run)

        db.complete_run(run_id, status="completed")

        retrieved = db.get_run(run_id)
        assert retrieved is not None
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None

    def test_complete_run_with_error(self, db: Database):
        """Test completing a run with error."""
        run = Run(run_type="heartbeat", status="running")
        run_id = db.create_run(run)

        db.complete_run(run_id, status="failed", error_message="Connection timeout")

        retrieved = db.get_run(run_id)
        assert retrieved is not None
        assert retrieved.status == "failed"
        assert retrieved.error_message == "Connection timeout"

    def test_create_snapshot(self, db: Database):
        """Test creating a snapshot record."""
        run = Run(run_type="heartbeat", status="running")
        run_id = db.create_run(run)

        snapshot = Snapshot(
            run_id=run_id,
            snapshot_type="fixtures",
            data={"count": 10, "fixtures": []},
            created_at=utc_now(),
        )

        snapshot_id = db.create_snapshot(snapshot)

        assert snapshot_id is not None
        assert snapshot_id > 0

    def test_get_snapshot(self, db: Database):
        """Test retrieving a snapshot record."""
        run = Run(run_type="heartbeat", status="running")
        run_id = db.create_run(run)

        snapshot = Snapshot(
            run_id=run_id,
            snapshot_type="markets",
            data={"count": 5},
        )
        snapshot_id = db.create_snapshot(snapshot)

        retrieved = db.get_snapshot(snapshot_id)

        assert retrieved is not None
        assert retrieved.id == snapshot_id
        assert retrieved.snapshot_type == "markets"
        assert retrieved.data == {"count": 5}

    def test_get_snapshots_for_run(self, db: Database):
        """Test retrieving all snapshots for a run."""
        run = Run(run_type="heartbeat", status="running")
        run_id = db.create_run(run)

        # Create multiple snapshots
        for snapshot_type in ["fixtures", "markets", "prices"]:
            snapshot = Snapshot(
                run_id=run_id,
                snapshot_type=snapshot_type,
                data={"type": snapshot_type},
            )
            db.create_snapshot(snapshot)

        snapshots = db.get_snapshots_for_run(run_id)

        assert len(snapshots) == 3
        snapshot_types = {s.snapshot_type for s in snapshots}
        assert snapshot_types == {"fixtures", "markets", "prices"}

    def test_context_manager(self, temp_dir: Path):
        """Test database context manager."""
        db_path = temp_dir / "test_context.db"

        with Database(db_path) as db:
            db.migrate()
            run = Run(run_type="test", status="running")
            run_id = db.create_run(run)
            assert run_id > 0

        # Connection should be closed
        assert db._connection is None
