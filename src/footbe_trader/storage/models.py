"""Data models for storage layer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from footbe_trader.common.time_utils import utc_now


@dataclass
class Fixture:
    """Football fixture/match."""

    id: int | None = None
    external_id: str = ""  # API-Football fixture ID
    league: str = "EPL"
    season: str = ""
    home_team: str = ""
    away_team: str = ""
    kickoff_time: datetime | None = None
    status: str = "scheduled"  # scheduled, live, finished, postponed
    home_score: int | None = None
    away_score: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class Market:
    """Trading market on Kalshi."""

    id: int | None = None
    external_id: str = ""  # Kalshi market ID
    fixture_id: int | None = None
    market_type: str = ""  # e.g., "match_winner", "total_goals"
    title: str = ""
    status: str = "open"  # open, closed, settled
    close_time: datetime | None = None
    settlement_value: float | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class Snapshot:
    """Point-in-time snapshot of market/system state."""

    id: int | None = None
    run_id: int | None = None
    snapshot_type: str = ""  # "fixtures", "markets", "prices"
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Prediction:
    """Model prediction for a market."""

    id: int | None = None
    run_id: int | None = None
    fixture_id: int | None = None
    market_id: int | None = None
    model_name: str = ""
    model_version: str = ""
    prediction_type: str = ""  # e.g., "home_win_prob"
    value: float = 0.0
    confidence: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Order:
    """Trading order."""

    id: int | None = None
    external_id: str = ""  # Kalshi order ID
    run_id: int | None = None
    market_id: int | None = None
    side: str = ""  # "buy", "sell"
    order_type: str = "limit"  # "limit", "market"
    price: float = 0.0
    quantity: int = 0
    filled_quantity: int = 0
    status: str = "pending"  # pending, open, filled, cancelled, rejected
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class Fill:
    """Order fill/execution."""

    id: int | None = None
    external_id: str = ""  # Kalshi fill ID
    order_id: int | None = None
    price: float = 0.0
    quantity: int = 0
    fee: float = 0.0
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Position:
    """Current position in a market."""

    id: int | None = None
    market_id: int | None = None
    quantity: int = 0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class PnlMark:
    """P&L mark at a point in time."""

    id: int | None = None
    run_id: int | None = None
    position_id: int | None = None
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Run:
    """Agent run/heartbeat record."""

    id: int | None = None
    run_type: str = "heartbeat"  # heartbeat, trading, backtest
    status: str = "running"  # running, completed, failed
    config_hash: str = ""
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot for market data analysis."""

    id: int | None = None
    timestamp: datetime = field(default_factory=utc_now)
    ticker: str = ""
    best_bid: float | None = None
    best_ask: float | None = None
    mid: float | None = None
    spread: float | None = None
    bid_volume: int | None = None
    ask_volume: int | None = None
    volume: int | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Team:
    """Football team record."""

    id: int | None = None
    team_id: int = 0  # API-Football team ID
    name: str = ""
    code: str = ""
    country: str = ""
    logo_url: str = ""
    founded: int | None = None
    venue_name: str = ""
    venue_capacity: int | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class FixtureV2:
    """Enhanced football fixture/match record."""

    id: int | None = None
    fixture_id: int = 0  # API-Football fixture ID
    league_id: int = 39  # 39 = EPL
    season: int = 0
    round: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    kickoff_utc: datetime | None = None
    status: str = "NS"  # API-Football status code
    home_goals: int | None = None
    away_goals: int | None = None
    venue: str = ""
    referee: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class StandingSnapshot:
    """League standings snapshot at a point in time."""

    id: int | None = None
    league_id: int = 39
    season: int = 0
    snapshot_date: str = ""  # YYYY-MM-DD
    team_id: int = 0
    rank: int = 0
    points: int = 0
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0
    form: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class IngestionLog:
    """Log of data ingestion runs."""

    id: int | None = None
    data_type: str = ""  # 'fixtures', 'teams', 'standings'
    season: int = 0
    ingested_at: datetime = field(default_factory=utc_now)
    record_count: int = 0
    status: str = "success"  # 'success', 'failed', 'partial'
    error_message: str | None = None


@dataclass
class SnapshotSession:
    """Collection session for historical snapshots."""

    id: int | None = None
    session_id: str = ""
    started_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None
    status: str = "running"  # 'running', 'completed', 'failed'
    interval_minutes: int = 5
    fixtures_tracked: int = 0
    snapshots_collected: int = 0
    config_json: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class HistoricalSnapshot:
    """Historical orderbook snapshot with fixture linkage."""

    id: int | None = None
    session_id: str | None = None
    fixture_id: int = 0
    ticker: str = ""
    outcome: str = ""  # 'home_win', 'draw', 'away_win'
    timestamp: datetime = field(default_factory=utc_now)

    # Orderbook data
    best_bid: float | None = None
    best_ask: float | None = None
    mid: float | None = None
    spread: float | None = None
    bid_volume: int | None = None
    ask_volume: int | None = None

    # Market data
    yes_price: float | None = None
    no_price: float | None = None
    volume_24h: int | None = None
    open_interest: int | None = None

    # Model predictions
    model_prob: float | None = None
    model_version: str | None = None

    # Raw data
    raw_orderbook_json: dict[str, Any] = field(default_factory=dict)
    raw_market_json: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)


@dataclass
class StrategyBacktest:
    """Strategy backtest run record."""

    id: int | None = None
    backtest_id: str = ""
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = "running"  # 'running', 'completed', 'failed'

    # Configuration
    strategy_config_hash: str = ""
    strategy_config_json: dict[str, Any] = field(default_factory=dict)
    snapshot_start: datetime | None = None
    snapshot_end: datetime | None = None
    fixtures_included: list[int] = field(default_factory=list)

    # Summary metrics
    initial_bankroll: float = 10000.0
    final_bankroll: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_hold_time_minutes: float | None = None

    # Breakdown (JSON)
    per_outcome_stats_json: dict[str, Any] = field(default_factory=dict)
    per_fixture_stats_json: dict[str, Any] = field(default_factory=dict)
    edge_calibration_json: dict[str, Any] = field(default_factory=dict)
    results_json: dict[str, Any] = field(default_factory=dict)

    error_message: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class BacktestTrade:
    """Individual trade within a backtest."""

    id: int | None = None
    backtest_id: str = ""
    trade_id: str = ""
    fixture_id: int = 0
    ticker: str = ""
    outcome: str = ""

    # Entry
    entry_timestamp: datetime = field(default_factory=utc_now)
    entry_price: float = 0.0
    entry_quantity: int = 0
    entry_edge: float | None = None
    entry_model_prob: float | None = None
    entry_reason: str | None = None

    # Exit
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_quantity: int | None = None
    exit_reason: str | None = None

    # P&L
    realized_pnl: float = 0.0
    return_pct: float | None = None
    hold_time_minutes: float | None = None

    # MTM history
    mtm_history_json: list[dict[str, Any]] = field(default_factory=list)

    created_at: datetime = field(default_factory=utc_now)


@dataclass
class BacktestEquity:
    """Point-in-time portfolio state during backtest."""

    id: int | None = None
    backtest_id: str = ""
    timestamp: datetime = field(default_factory=utc_now)
    bankroll: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    drawdown: float = 0.0

    created_at: datetime = field(default_factory=utc_now)


# ============================================================================
# NBA Models
# ============================================================================


@dataclass
class NBATeamRecord:
    """NBA team database record."""

    id: int | None = None
    team_id: int = 0  # API-NBA team ID
    name: str = ""
    nickname: str = ""
    code: str = ""  # 3-letter code like "LAL", "BOS"
    city: str = ""
    conference: str = ""  # "East" or "West"
    division: str = ""
    logo_url: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class NBAGameRecord:
    """NBA game database record."""

    id: int | None = None
    game_id: int = 0  # API-NBA game ID
    season: int = 0
    league: str = "standard"
    stage: int | None = None
    date_utc: datetime = field(default_factory=utc_now)
    timestamp: int = 0  # Unix timestamp
    status: int = 1  # NBAGameStatus enum value
    home_team_id: int = 0
    away_team_id: int = 0
    home_score: int | None = None
    away_score: int | None = None
    arena: str = ""
    city: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class NBAGameMarketMap:
    """Mapping between NBA game and Kalshi markets."""

    id: int | None = None
    game_id: int = 0  # API-NBA game ID
    mapping_version: int = 1
    
    # 2-way market structure (no draw in basketball)
    ticker_home_win: str | None = None
    ticker_away_win: str | None = None
    
    event_ticker: str | None = None  # Kalshi event ticker
    confidence_score: float = 0.0
    confidence_components: dict[str, Any] = field(default_factory=dict)
    status: str = "AUTO"  # 'AUTO', 'MANUAL_OVERRIDE', 'REJECTED'
    metadata_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
