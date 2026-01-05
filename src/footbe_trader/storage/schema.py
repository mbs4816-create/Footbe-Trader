"""SQLite database schema definitions."""

SCHEMA_VERSION = 7

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Agent runs/heartbeats
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL DEFAULT 'heartbeat',
    status TEXT NOT NULL DEFAULT 'running',
    config_hash TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    error_message TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

-- Football fixtures
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE NOT NULL,
    league TEXT NOT NULL DEFAULT 'EPL',
    season TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_time TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled',
    home_score INTEGER,
    away_score INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fixtures_external_id ON fixtures(external_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(kickoff_time);
CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status);

-- Trading markets
CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE NOT NULL,
    fixture_id INTEGER REFERENCES fixtures(id),
    market_type TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    close_time TEXT,
    settlement_value REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_markets_external_id ON markets(external_id);
CREATE INDEX IF NOT EXISTS idx_markets_fixture_id ON markets(fixture_id);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);

-- Point-in-time snapshots
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    snapshot_type TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON snapshots(snapshot_type);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    fixture_id INTEGER REFERENCES fixtures(id),
    market_id INTEGER REFERENCES markets(id),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    value REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    features TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_market_id ON predictions(market_id);

-- Trading orders
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT,
    run_id INTEGER REFERENCES runs(id),
    market_id INTEGER REFERENCES markets(id),
    side TEXT NOT NULL,
    order_type TEXT NOT NULL DEFAULT 'limit',
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_run_id ON orders(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_market_id ON orders(market_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- Order fills/executions
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT,
    order_id INTEGER REFERENCES orders(id),
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    fee REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);

-- Current positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id INTEGER UNIQUE REFERENCES markets(id),
    quantity INTEGER NOT NULL DEFAULT 0,
    average_price REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_market_id ON positions(market_id);

-- P&L marks
CREATE TABLE IF NOT EXISTS pnl_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    position_id INTEGER REFERENCES positions(id),
    mark_price REAL NOT NULL,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pnl_marks_run_id ON pnl_marks(run_id);
CREATE INDEX IF NOT EXISTS idx_pnl_marks_position_id ON pnl_marks(position_id);
"""

# Migration 2: Add orderbook_snapshots table
MIGRATION_2_ORDERBOOK_SNAPSHOTS = """
-- Orderbook snapshots for market data
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    ticker TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    mid REAL,
    spread REAL,
    bid_volume INTEGER,
    ask_volume INTEGER,
    volume INTEGER,
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_ticker ON orderbook_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_timestamp ON orderbook_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_ticker_timestamp ON orderbook_snapshots(ticker, timestamp);
"""

# Migration 3: Enhanced fixtures table with team IDs and new teams table
MIGRATION_3_FIXTURES_TEAMS = """
-- Teams table for storing team information
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER UNIQUE NOT NULL,  -- API-Football team ID
    name TEXT NOT NULL,
    code TEXT,
    country TEXT,
    logo_url TEXT,
    founded INTEGER,
    venue_name TEXT,
    venue_capacity INTEGER,
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_teams_team_id ON teams(team_id);
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);

-- Enhanced fixtures table (new table to replace old one)
CREATE TABLE IF NOT EXISTS fixtures_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER UNIQUE NOT NULL,  -- API-Football fixture ID
    league_id INTEGER NOT NULL DEFAULT 39,  -- 39 = EPL
    season INTEGER NOT NULL,
    round TEXT,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    kickoff_utc TEXT,
    status TEXT NOT NULL DEFAULT 'NS',
    home_goals INTEGER,
    away_goals INTEGER,
    venue TEXT,
    referee TEXT,
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_fixtures_v2_fixture_id ON fixtures_v2(fixture_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_season ON fixtures_v2(season);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_kickoff ON fixtures_v2(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_status ON fixtures_v2(status);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_season_round ON fixtures_v2(season, round);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_home_team ON fixtures_v2(home_team_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_v2_away_team ON fixtures_v2(away_team_id);

-- Standings snapshots for historical tracking
CREATE TABLE IF NOT EXISTS standings_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL DEFAULT 39,
    season INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    points INTEGER NOT NULL,
    played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    goal_difference INTEGER NOT NULL,
    form TEXT,
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season, snapshot_date, team_id)
);

CREATE INDEX IF NOT EXISTS idx_standings_season ON standings_snapshots(season);
CREATE INDEX IF NOT EXISTS idx_standings_date ON standings_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_standings_team ON standings_snapshots(team_id);

-- Ingestion log for tracking what has been ingested
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL,  -- 'fixtures', 'teams', 'standings'
    season INTEGER NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    record_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'success',
    error_message TEXT,
    UNIQUE(data_type, season)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_type_season ON ingestion_log(data_type, season);
"""

# Migration 4: Universal market mapping tables
MIGRATION_4_MARKET_MAPPING = """
-- Leagues table for all API-Football leagues
CREATE TABLE IF NOT EXISTS leagues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER UNIQUE NOT NULL,  -- API-Football league ID
    league_name TEXT NOT NULL,
    country TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'League',  -- League or Cup
    logo_url TEXT,
    seasons_available TEXT DEFAULT '[]',  -- JSON array of seasons
    league_key TEXT,  -- Canonical league identifier for cross-platform mapping
    is_active INTEGER NOT NULL DEFAULT 1,
    last_synced_at TEXT,
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_leagues_league_id ON leagues(league_id);
CREATE INDEX IF NOT EXISTS idx_leagues_country ON leagues(country);
CREATE INDEX IF NOT EXISTS idx_leagues_league_key ON leagues(league_key);
CREATE INDEX IF NOT EXISTS idx_leagues_is_active ON leagues(is_active);

-- Add canonical_name to teams table (safe add - will fail silently if exists)
-- Note: SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we handle in code

-- Kalshi events table
CREATE TABLE IF NOT EXISTS kalshi_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_ticker TEXT UNIQUE NOT NULL,
    series_ticker TEXT,
    title TEXT NOT NULL,
    subtitle TEXT,
    category TEXT,
    sub_category TEXT,
    strike_date TEXT,
    is_soccer INTEGER NOT NULL DEFAULT 0,
    league_key TEXT,  -- Canonical league identifier if detected
    parsed_home_team TEXT,  -- Parsed team name from title
    parsed_away_team TEXT,
    parsed_canonical_home TEXT,  -- Canonical team name
    parsed_canonical_away TEXT,
    market_structure TEXT,  -- '1X2', 'MONEYLINE', 'BINARY', etc.
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_kalshi_events_event_ticker ON kalshi_events(event_ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_events_series_ticker ON kalshi_events(series_ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_events_is_soccer ON kalshi_events(is_soccer);
CREATE INDEX IF NOT EXISTS idx_kalshi_events_league_key ON kalshi_events(league_key);
CREATE INDEX IF NOT EXISTS idx_kalshi_events_strike_date ON kalshi_events(strike_date);

-- Kalshi markets table
CREATE TABLE IF NOT EXISTS kalshi_markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    event_ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    subtitle TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    open_time TEXT,
    close_time TEXT,
    expiration_time TEXT,
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    last_price REAL,
    volume INTEGER DEFAULT 0,
    is_soccer INTEGER NOT NULL DEFAULT 0,
    market_type TEXT,  -- 'HOME_WIN', 'AWAY_WIN', 'DRAW', 'MONEYLINE_YES', etc.
    parsed_team TEXT,  -- Which team this market references
    parsed_canonical_team TEXT,
    raw_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (event_ticker) REFERENCES kalshi_events(event_ticker)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_markets_ticker ON kalshi_markets(ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_event_ticker ON kalshi_markets(event_ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_status ON kalshi_markets(status);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_is_soccer ON kalshi_markets(is_soccer);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_close_time ON kalshi_markets(close_time);

-- Fixture to market mapping
CREATE TABLE IF NOT EXISTS fixture_market_map (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL,  -- API-Football fixture ID
    mapping_version INTEGER NOT NULL DEFAULT 1,
    structure_type TEXT NOT NULL,  -- '1X2', 'HOME_WIN_BINARY', 'AWAY_WIN_BINARY', 'NO_DRAW', etc.
    ticker_home_win TEXT,
    ticker_draw TEXT,
    ticker_away_win TEXT,
    ticker_home_win_yes TEXT,
    ticker_home_win_no TEXT,
    ticker_away_win_yes TEXT,
    ticker_away_win_no TEXT,
    event_ticker TEXT,  -- Kalshi event ticker
    confidence_score REAL NOT NULL DEFAULT 0.0,
    confidence_components TEXT DEFAULT '{}',  -- JSON with score breakdown
    status TEXT NOT NULL DEFAULT 'AUTO',  -- 'AUTO', 'MANUAL_OVERRIDE', 'REJECTED'
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(fixture_id, mapping_version)
);

CREATE INDEX IF NOT EXISTS idx_fixture_market_map_fixture_id ON fixture_market_map(fixture_id);
CREATE INDEX IF NOT EXISTS idx_fixture_market_map_status ON fixture_market_map(status);
CREATE INDEX IF NOT EXISTS idx_fixture_market_map_confidence ON fixture_market_map(confidence_score);
CREATE INDEX IF NOT EXISTS idx_fixture_market_map_event_ticker ON fixture_market_map(event_ticker);

-- Mapping reviews for low confidence matches
CREATE TABLE IF NOT EXISTS mapping_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL,
    fixture_info TEXT NOT NULL,  -- JSON: league, teams, kickoff
    candidate_count INTEGER NOT NULL DEFAULT 0,
    top_candidates TEXT DEFAULT '[]',  -- JSON array of candidate info
    review_status TEXT NOT NULL DEFAULT 'PENDING',  -- 'PENDING', 'RESOLVED', 'SKIPPED'
    resolution TEXT,  -- 'CONFIRMED', 'REJECTED', 'MANUAL_MAP'
    resolved_mapping_id INTEGER REFERENCES fixture_market_map(id),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mapping_reviews_fixture_id ON mapping_reviews(fixture_id);
CREATE INDEX IF NOT EXISTS idx_mapping_reviews_status ON mapping_reviews(review_status);
"""

# Migration 5: Paper trading and agent run tables
MIGRATION_5_PAPER_TRADING = """
-- Agent runs table (tracks each trading loop execution)
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL DEFAULT 'paper',  -- 'paper', 'live', 'backtest'
    status TEXT NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'failed'
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    config_hash TEXT NOT NULL,
    config_json TEXT DEFAULT '{}',
    
    -- Run statistics
    fixtures_evaluated INTEGER DEFAULT 0,
    markets_evaluated INTEGER DEFAULT 0,
    decisions_made INTEGER DEFAULT 0,
    orders_placed INTEGER DEFAULT 0,
    orders_filled INTEGER DEFAULT 0,
    orders_rejected INTEGER DEFAULT 0,
    
    -- P&L summary
    total_realized_pnl REAL DEFAULT 0.0,
    total_unrealized_pnl REAL DEFAULT 0.0,
    total_exposure REAL DEFAULT 0.0,
    position_count INTEGER DEFAULT 0,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_run_type ON agent_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at ON agent_runs(started_at);

-- Decision records table (captures every trading decision)
CREATE TABLE IF NOT EXISTS decision_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL UNIQUE,
    run_id INTEGER REFERENCES agent_runs(id),
    fixture_id INTEGER,
    market_ticker TEXT NOT NULL,
    outcome TEXT NOT NULL,  -- 'home_win', 'draw', 'away_win'
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    
    -- Inputs (stored as JSON)
    market_snapshot_json TEXT,  -- MarketSnapshot as JSON
    model_prediction_json TEXT,  -- ModelPrediction as JSON
    current_position_json TEXT,  -- PositionState as JSON
    
    -- Calculations
    edge_calculation_json TEXT,  -- EdgeCalculation as JSON
    kelly_sizing_json TEXT,  -- KellySizing as JSON
    
    -- Decision
    action TEXT NOT NULL,  -- 'buy', 'sell', 'hold', 'exit', 'skip'
    exit_reason TEXT,  -- 'take_profit', 'stop_loss', 'edge_flip', 'market_close'
    order_params_json TEXT,  -- OrderParams as JSON
    
    -- Rationale
    rationale TEXT,
    filters_passed_json TEXT,  -- Dict of filter results
    rejection_reason TEXT,
    
    -- Execution
    order_placed INTEGER DEFAULT 0,  -- Boolean
    order_id TEXT,
    execution_error TEXT,
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decision_records_run_id ON decision_records(run_id);
CREATE INDEX IF NOT EXISTS idx_decision_records_fixture_id ON decision_records(fixture_id);
CREATE INDEX IF NOT EXISTS idx_decision_records_ticker ON decision_records(market_ticker);
CREATE INDEX IF NOT EXISTS idx_decision_records_action ON decision_records(action);
CREATE INDEX IF NOT EXISTS idx_decision_records_timestamp ON decision_records(timestamp);

-- Paper positions table (tracks simulated positions)
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    outcome TEXT,  -- 'home_win', 'draw', 'away_win'
    fixture_id INTEGER,
    quantity INTEGER NOT NULL DEFAULT 0,
    average_entry_price REAL NOT NULL DEFAULT 0.0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    mark_price REAL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker)
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_ticker ON paper_positions(ticker);
CREATE INDEX IF NOT EXISTS idx_paper_positions_fixture_id ON paper_positions(fixture_id);

-- Paper orders table (tracks simulated orders)
CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    run_id INTEGER REFERENCES agent_runs(id),
    decision_id TEXT REFERENCES decision_records(decision_id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'yes', 'no'
    action TEXT NOT NULL,  -- 'buy', 'sell'
    order_type TEXT NOT NULL DEFAULT 'limit',
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'filled', 'partial', 'rejected'
    fill_price REAL DEFAULT 0.0,
    fees REAL DEFAULT 0.0,
    rejection_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    filled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_orders_run_id ON paper_orders(run_id);
CREATE INDEX IF NOT EXISTS idx_paper_orders_ticker ON paper_orders(ticker);
CREATE INDEX IF NOT EXISTS idx_paper_orders_status ON paper_orders(status);

-- Paper fills table (tracks simulated fills)
CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fill_id TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL REFERENCES paper_orders(order_id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    fees REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_paper_fills_order_id ON paper_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_paper_fills_ticker ON paper_fills(ticker);

-- PnL snapshots table (point-in-time P&L records)
CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL UNIQUE,
    run_id INTEGER REFERENCES agent_runs(id),
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    total_realized_pnl REAL NOT NULL DEFAULT 0.0,
    total_unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    total_exposure REAL NOT NULL DEFAULT 0.0,
    position_count INTEGER NOT NULL DEFAULT 0,
    bankroll REAL NOT NULL DEFAULT 0.0,
    positions_json TEXT DEFAULT '[]'  -- Array of position snapshots
);

CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_run_id ON pnl_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_timestamp ON pnl_snapshots(timestamp);
"""

# Migration 6: Historical snapshots for strategy backtesting
MIGRATION_6_HISTORICAL_SNAPSHOTS = """
-- Historical snapshots collection sessions
CREATE TABLE IF NOT EXISTS snapshot_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'failed'
    interval_minutes INTEGER NOT NULL DEFAULT 5,
    fixtures_tracked INTEGER DEFAULT 0,
    snapshots_collected INTEGER DEFAULT 0,
    config_json TEXT DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_snapshot_sessions_status ON snapshot_sessions(status);
CREATE INDEX IF NOT EXISTS idx_snapshot_sessions_started ON snapshot_sessions(started_at);

-- Historical orderbook snapshots with fixture linkage
CREATE TABLE IF NOT EXISTS historical_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES snapshot_sessions(session_id),
    fixture_id INTEGER NOT NULL,  -- API-Football fixture ID
    ticker TEXT NOT NULL,
    outcome TEXT NOT NULL,  -- 'home_win', 'draw', 'away_win'
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    
    -- Orderbook data
    best_bid REAL,
    best_ask REAL,
    mid REAL,
    spread REAL,
    bid_volume INTEGER,
    ask_volume INTEGER,
    
    -- Market data
    yes_price REAL,
    no_price REAL,
    volume_24h INTEGER,
    open_interest INTEGER,
    
    -- Model predictions at snapshot time (if available)
    model_prob REAL,
    model_version TEXT,
    
    -- Full raw data
    raw_orderbook_json TEXT DEFAULT '{}',
    raw_market_json TEXT DEFAULT '{}',
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_historical_snapshots_session ON historical_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_fixture ON historical_snapshots(fixture_id);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_ticker ON historical_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_timestamp ON historical_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_fixture_timestamp ON historical_snapshots(fixture_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_session_fixture ON historical_snapshots(session_id, fixture_id);

-- Strategy backtest runs
CREATE TABLE IF NOT EXISTS strategy_backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',  -- 'running', 'completed', 'failed'
    
    -- Configuration
    strategy_config_hash TEXT NOT NULL,
    strategy_config_json TEXT DEFAULT '{}',
    snapshot_start TEXT,  -- Start of snapshot range
    snapshot_end TEXT,    -- End of snapshot range
    fixtures_included TEXT DEFAULT '[]',  -- JSON array of fixture IDs
    
    -- Summary metrics
    initial_bankroll REAL NOT NULL DEFAULT 10000.0,
    final_bankroll REAL,
    total_return REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    
    -- Trade statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    avg_hold_time_minutes REAL,
    
    -- Per-outcome breakdown (JSON)
    per_outcome_stats_json TEXT DEFAULT '{}',
    per_fixture_stats_json TEXT DEFAULT '{}',
    
    -- Edge calibration
    edge_calibration_json TEXT DEFAULT '{}',
    
    -- Full results
    results_json TEXT DEFAULT '{}',
    error_message TEXT,
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_strategy_backtests_status ON strategy_backtests(status);
CREATE INDEX IF NOT EXISTS idx_strategy_backtests_started ON strategy_backtests(started_at);
CREATE INDEX IF NOT EXISTS idx_strategy_backtests_config ON strategy_backtests(strategy_config_hash);

-- Backtest trades (individual trades within a backtest)
CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT NOT NULL REFERENCES strategy_backtests(backtest_id),
    trade_id TEXT NOT NULL,
    fixture_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    outcome TEXT NOT NULL,
    
    -- Entry
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_quantity INTEGER NOT NULL,
    entry_edge REAL,
    entry_model_prob REAL,
    entry_reason TEXT,
    
    -- Exit
    exit_timestamp TEXT,
    exit_price REAL,
    exit_quantity INTEGER,
    exit_reason TEXT,  -- 'take_profit', 'stop_loss', 'edge_flip', 'market_close', 'settlement'
    
    -- P&L
    realized_pnl REAL DEFAULT 0.0,
    return_pct REAL,
    hold_time_minutes REAL,
    
    -- Mark-to-market history (JSON array of {timestamp, mid, unrealized_pnl})
    mtm_history_json TEXT DEFAULT '[]',
    
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(backtest_id, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_fixture ON backtest_trades(fixture_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_outcome ON backtest_trades(outcome);

-- Backtest equity curve (point-in-time portfolio state)
CREATE TABLE IF NOT EXISTS backtest_equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT NOT NULL REFERENCES strategy_backtests(backtest_id),
    timestamp TEXT NOT NULL,
    bankroll REAL NOT NULL,
    total_exposure REAL NOT NULL DEFAULT 0.0,
    position_count INTEGER NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    drawdown REAL DEFAULT 0.0,
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_backtest_equity_backtest ON backtest_equity(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_equity_timestamp ON backtest_equity(timestamp);
"""

# Migration 7: Reporting subsystem - extended decision capture
MIGRATION_7_REPORTING = """
-- Extend decision_records with additional fields for comprehensive reporting
ALTER TABLE decision_records ADD COLUMN league_key TEXT;
ALTER TABLE decision_records ADD COLUMN structure_type TEXT DEFAULT 'match_result';  -- match_result, btts, etc.
ALTER TABLE decision_records ADD COLUMN hours_to_kickoff REAL;
ALTER TABLE decision_records ADD COLUMN time_category TEXT;  -- very_early, early, standard, optimal, late

-- Pacing and drawdown state at decision time
ALTER TABLE decision_records ADD COLUMN pace_state TEXT;  -- ahead_of_pace, on_pace, behind_pace
ALTER TABLE decision_records ADD COLUMN drawdown REAL DEFAULT 0.0;
ALTER TABLE decision_records ADD COLUMN drawdown_band TEXT;  -- none, light, moderate, severe
ALTER TABLE decision_records ADD COLUMN gross_exposure REAL DEFAULT 0.0;
ALTER TABLE decision_records ADD COLUMN equity REAL DEFAULT 0.0;

-- Throttle and sizing adjustments
ALTER TABLE decision_records ADD COLUMN throttle_multiplier REAL DEFAULT 1.0;
ALTER TABLE decision_records ADD COLUMN pacing_edge_mult REAL DEFAULT 1.0;
ALTER TABLE decision_records ADD COLUMN pacing_size_mult REAL DEFAULT 1.0;
ALTER TABLE decision_records ADD COLUMN time_edge_mult REAL DEFAULT 1.0;
ALTER TABLE decision_records ADD COLUMN time_size_mult REAL DEFAULT 1.0;

-- Size tracking
ALTER TABLE decision_records ADD COLUMN size_intended INTEGER;  -- Kelly-optimal size before throttle
ALTER TABLE decision_records ADD COLUMN size_capped INTEGER;  -- After exposure cap
ALTER TABLE decision_records ADD COLUMN size_executed INTEGER;  -- Actually executed (0 if skipped)

-- Full policy decision for audit trail
ALTER TABLE decision_records ADD COLUMN policy_decision_json TEXT;

-- Edge bucketing for evaluation
ALTER TABLE decision_records ADD COLUMN edge_bucket TEXT;  -- <5%, 5-10%, 10-15%, 15%+

-- Index for common reporting queries
CREATE INDEX IF NOT EXISTS idx_decision_records_league ON decision_records(league_key);
CREATE INDEX IF NOT EXISTS idx_decision_records_pace_state ON decision_records(pace_state);
CREATE INDEX IF NOT EXISTS idx_decision_records_edge_bucket ON decision_records(edge_bucket);
CREATE INDEX IF NOT EXISTS idx_decision_records_created ON decision_records(created_at);
"""

MIGRATIONS: dict[int, str] = {
    1: SCHEMA_SQL,
    2: MIGRATION_2_ORDERBOOK_SNAPSHOTS,
    3: MIGRATION_3_FIXTURES_TEAMS,
    4: MIGRATION_4_MARKET_MAPPING,
    5: MIGRATION_5_PAPER_TRADING,
    6: MIGRATION_6_HISTORICAL_SNAPSHOTS,
    7: MIGRATION_7_REPORTING,
}
