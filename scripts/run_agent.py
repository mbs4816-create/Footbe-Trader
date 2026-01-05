#!/usr/bin/env python3
"""Trading Agent CLI.

Run the trading agent in paper or live mode to evaluate fixtures,
generate trading signals, and execute trades.

Usage:
    python scripts/run_agent.py --mode paper --interval 30
    python scripts/run_agent.py --mode paper --once
    python scripts/run_agent.py --mode live --interval 60 --dry-run
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderbookLevel, OrderData
from footbe_trader.modeling.interfaces import PredictionResult
from footbe_trader.storage.database import Database
from footbe_trader.strategy.decision_record import (
    AgentRunSummary,
    DecisionRecord,
    ModelPrediction,
)
from footbe_trader.strategy.mapping import (
    FixtureMarketMapping,
    FixtureMarketMapper,
    MappingConfig,
)
from footbe_trader.strategy.paper_trading import (
    PaperFill,
    PaperOrder,
    PaperPosition,
    PaperTradingSimulator,
    PnlSnapshot,
)
from footbe_trader.strategy.trading_strategy import (
    EdgeStrategy,
    FixtureContext,
    OutcomeContext,
    StrategyConfig,
    create_agent_run_summary,
)

logger = get_logger(__name__)

# Global shutdown flag
_shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info("shutdown_requested", signal=signum)
    _shutdown_requested = True


class TradingAgent:
    """Trading agent that runs the strategy loop."""

    def __init__(
        self,
        db: Database,
        kalshi_client: KalshiClient | None,
        strategy: EdgeStrategy,
        simulator: PaperTradingSimulator,
        mode: str = "paper",
        dry_run: bool = False,
    ):
        """Initialize trading agent.

        Args:
            db: Database connection.
            kalshi_client: Kalshi API client (None for testing).
            strategy: Trading strategy.
            simulator: Paper trading simulator.
            mode: "paper" or "live".
            dry_run: If True, don't execute trades even in live mode.
        """
        self.db = db
        self.kalshi_client = kalshi_client
        self.strategy = strategy
        self.simulator = simulator
        self.mode = mode
        self.dry_run = dry_run
        self._current_run: AgentRunSummary | None = None

    async def run_once(self) -> AgentRunSummary:
        """Run a single iteration of the trading loop.

        Returns:
            Run summary with statistics.
        """
        # Create run record
        run_summary = create_agent_run_summary(
            run_type=self.mode,
            config=self.strategy.config,
        )

        # Store run in database
        run_id = self._create_agent_run(run_summary)
        run_summary.run_id = run_id
        self.strategy.set_run_id(run_id)
        self.simulator.set_run_id(run_id)
        self._current_run = run_summary

        logger.info(
            "agent_run_started",
            run_id=run_id,
            mode=self.mode,
            dry_run=self.dry_run,
        )

        try:
            # Step 1: Get mapped fixtures
            fixtures = await self._get_mapped_fixtures()
            run_summary.fixtures_evaluated = len(fixtures)

            logger.info("fixtures_loaded", count=len(fixtures))

            # Step 2: Process each fixture
            for fixture_ctx in fixtures:
                if _shutdown_requested:
                    break

                await self._process_fixture(fixture_ctx, run_summary)

            # Step 3: Check exit conditions for open positions
            await self._check_exits(run_summary)

            # Step 4: Take P&L snapshot
            pnl_snapshot = self.simulator.take_pnl_snapshot()
            self._store_pnl_snapshot(pnl_snapshot)

            # Step 5: Update summary
            self.simulator.update_run_summary(run_summary)
            run_summary.status = "completed"
            run_summary.completed_at = utc_now()

            # Decision records are stored inline during _process_fixture
            # to satisfy foreign key constraints with paper_orders

            # Update run in database
            self._complete_agent_run(run_summary)

            logger.info(
                "agent_run_completed",
                run_id=run_id,
                fixtures=run_summary.fixtures_evaluated,
                markets=run_summary.markets_evaluated,
                decisions=run_summary.decisions_made,
                orders_placed=run_summary.orders_placed,
                orders_filled=run_summary.orders_filled,
                realized_pnl=run_summary.total_realized_pnl,
                unrealized_pnl=run_summary.total_unrealized_pnl,
            )

        except Exception as e:
            run_summary.status = "failed"
            run_summary.error_message = str(e)
            run_summary.error_count += 1
            run_summary.completed_at = utc_now()
            self._complete_agent_run(run_summary)
            logger.error(
                "agent_run_failed",
                run_id=run_id,
                error=str(e),
                exc_info=True,
            )
            raise

        return run_summary

    def _get_market_data_from_db(
        self, ticker: str
    ) -> tuple[MarketData | None, OrderbookData | None]:
        """Fetch market data from database for paper trading.
        
        Args:
            ticker: Market ticker.
            
        Returns:
            Tuple of (MarketData, OrderbookData) or (None, None).
        """
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT 
                ticker, event_ticker, title, subtitle, status,
                close_time, yes_bid, yes_ask, no_bid, no_ask, 
                last_price, volume
            FROM kalshi_markets
            WHERE ticker = ?
            """,
            (ticker,),
        )
        row = cursor.fetchone()
        
        if not row:
            return None, None
        
        # Parse close_time string to datetime
        close_time = None
        if row["close_time"]:
            try:
                close_time_str = row["close_time"]
                if close_time_str.endswith("Z"):
                    close_time_str = close_time_str[:-1] + "+00:00"
                close_time = datetime.fromisoformat(close_time_str)
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=UTC)
            except ValueError:
                pass
        
        market_data = MarketData(
            ticker=row["ticker"],
            event_ticker=row["event_ticker"],
            title=row["title"],
            subtitle=row["subtitle"] or "",
            status=row["status"],
            close_time=close_time,
            yes_bid=row["yes_bid"] or 0,
            yes_ask=row["yes_ask"] or 0,
            volume=row["volume"] or 0,
            open_interest=0,
        )
        
        # Create synthetic orderbook from bid/ask using OrderbookLevel
        yes_bid = row["yes_bid"] or 0
        yes_ask = row["yes_ask"] or 0
        no_bid = row["no_bid"] or 0
        no_ask = row["no_ask"] or 0
        
        orderbook = OrderbookData(
            ticker=row["ticker"],
            yes_bids=[OrderbookLevel(price=yes_bid, quantity=1000)] if yes_bid > 0 else [],
            yes_asks=[OrderbookLevel(price=yes_ask, quantity=1000)] if yes_ask > 0 else [],
            no_bids=[OrderbookLevel(price=no_bid, quantity=1000)] if no_bid > 0 else [],
            no_asks=[OrderbookLevel(price=no_ask, quantity=1000)] if no_ask > 0 else [],
        )
        
        return market_data, orderbook

    async def run_loop(self, interval_minutes: int) -> None:
        """Run the trading loop continuously.

        Args:
            interval_minutes: Minutes between iterations.
        """
        logger.info(
            "agent_loop_started",
            mode=self.mode,
            interval_minutes=interval_minutes,
        )

        while not _shutdown_requested:
            try:
                summary = await self.run_once()

                # Log summary
                print(f"\n{'=' * 60}")
                print(f"Run {summary.run_id} completed at {summary.completed_at}")
                print(f"  Fixtures evaluated: {summary.fixtures_evaluated}")
                print(f"  Markets evaluated: {summary.markets_evaluated}")
                print(f"  Decisions made: {summary.decisions_made}")
                print(f"  Orders placed: {summary.orders_placed}")
                print(f"  Orders filled: {summary.orders_filled}")
                print(f"  Realized P&L: ${summary.total_realized_pnl:.2f}")
                print(f"  Unrealized P&L: ${summary.total_unrealized_pnl:.2f}")
                print(f"  Open positions: {summary.position_count}")
                print(f"  Exposure: ${summary.total_exposure:.2f}")
                print(f"{'=' * 60}\n")

            except Exception as e:
                logger.error("run_loop_error", error=str(e), exc_info=True)
                # Continue loop despite errors

            # Sleep until next iteration
            if not _shutdown_requested:
                logger.info("sleeping", minutes=interval_minutes)
                await asyncio.sleep(interval_minutes * 60)

        logger.info("agent_loop_stopped")

    async def _get_mapped_fixtures(self) -> list[FixtureContext]:
        """Get fixtures with valid market mappings."""
        fixtures: list[FixtureContext] = []

        # Query fixtures with mappings from database
        cursor = self.db.connection.cursor()

        # Get fixtures in the next 7 days with active mappings
        now = utc_now()
        cutoff = now + timedelta(days=7)

        cursor.execute(
            """
            SELECT
                f.fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.kickoff_utc,
                f.league_id,
                m.mapping_version,
                m.structure_type,
                m.ticker_home_win,
                m.ticker_draw,
                m.ticker_away_win,
                m.event_ticker,
                m.confidence_score,
                th.name as home_team_name,
                ta.name as away_team_name
            FROM fixtures_v2 f
            JOIN fixture_market_map m ON f.fixture_id = m.fixture_id
            LEFT JOIN teams th ON f.home_team_id = th.team_id
            LEFT JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.kickoff_utc >= ?
            AND f.kickoff_utc <= ?
            AND m.status = 'AUTO'
            AND m.confidence_score >= 0.70
            ORDER BY f.kickoff_utc ASC
            """,
            (now.isoformat(), cutoff.isoformat()),
        )

        for row in cursor.fetchall():
            mapping = FixtureMarketMapping(
                fixture_id=row["fixture_id"],
                mapping_version=row["mapping_version"],
                structure_type=row["structure_type"],
                ticker_home_win=row["ticker_home_win"],
                ticker_draw=row["ticker_draw"],
                ticker_away_win=row["ticker_away_win"],
                event_ticker=row["event_ticker"],
                confidence_score=row["confidence_score"],
            )

            kickoff = (
                datetime.fromisoformat(row["kickoff_utc"])
                if row["kickoff_utc"]
                else now
            )

            fixture_ctx = FixtureContext(
                fixture_id=row["fixture_id"],
                home_team=row["home_team_name"] or str(row["home_team_id"]),
                away_team=row["away_team_name"] or str(row["away_team_id"]),
                kickoff_time=kickoff,
                league=str(row["league_id"]),
                mapping=mapping,
                current_exposure=self.simulator.get_fixture_exposure(row["fixture_id"]),
            )
            fixtures.append(fixture_ctx)

        return fixtures

    async def _process_fixture(
        self,
        fixture: FixtureContext,
        summary: AgentRunSummary,
    ) -> None:
        """Process a single fixture."""
        logger.info(
            "processing_fixture",
            fixture_id=fixture.fixture_id,
            match=f"{fixture.home_team} vs {fixture.away_team}",
        )

        # Build outcome contexts
        outcomes = await self._build_outcome_contexts(fixture)
        summary.markets_evaluated += len(outcomes)

        if not outcomes:
            logger.debug("no_valid_outcomes", fixture_id=fixture.fixture_id)
            return

        # Get model predictions (placeholder - would integrate with actual model)
        model_prediction = await self._get_model_prediction(fixture)

        # Evaluate fixture with strategy
        decisions = self.strategy.evaluate_fixture(
            fixture=fixture,
            outcomes=outcomes,
            model_prediction=model_prediction,
            global_exposure=self.simulator.total_exposure,
            bankroll=self.simulator.bankroll,
        )

        summary.decisions_made += len(decisions)

        # Execute decisions
        for decision in decisions:
            # Store decision record first (before order, due to FK constraint)
            self._store_decision_record(decision)
            
            if decision.order_params is not None:
                if not self.dry_run and self.kalshi_client is not None:
                    # Live trading: place real order via Kalshi API
                    order_data = await self._execute_live_order(decision, summary)
                    if order_data:
                        decision.order_placed = True
                        decision.order_id = order_data.order_id
                        self._update_decision_with_order(decision)
                        
                        # Also update simulator for position tracking
                        orderbook = None
                        for outcome in outcomes:
                            if outcome.ticker == decision.market_ticker:
                                orderbook = outcome.orderbook
                                break
                        self.simulator.execute_decision(decision, orderbook)
                else:
                    # Paper trading: simulate order locally
                    orderbook = None
                    for outcome in outcomes:
                        if outcome.ticker == decision.market_ticker:
                            orderbook = outcome.orderbook
                            break

                    order = self.simulator.execute_decision(decision, orderbook)

                    if order:
                        decision.order_placed = True
                        decision.order_id = order.order_id
                        
                        # Update the decision record with order info
                        self._update_decision_with_order(decision)

                        # Store order in database
                        self._store_paper_order(order)

                        # Store fills
                        for fill in self.simulator.fills:
                            if fill.order_id == order.order_id:
                                self._store_paper_fill(fill)
        
        # Clear decision records since we stored them inline
        self.strategy.clear_decision_records()

    async def _build_outcome_contexts(
        self,
        fixture: FixtureContext,
    ) -> list[OutcomeContext]:
        """Build outcome contexts for a fixture."""
        outcomes: list[OutcomeContext] = []
        mapping = fixture.mapping

        # Define outcome -> ticker mapping
        outcome_tickers = [
            ("home_win", mapping.ticker_home_win),
            ("draw", mapping.ticker_draw),
            ("away_win", mapping.ticker_away_win),
        ]

        for outcome_type, ticker in outcome_tickers:
            if not ticker:
                continue

            market_data: MarketData | None = None
            orderbook: OrderbookData | None = None

            if self.kalshi_client:
                try:
                    market_data = await self.kalshi_client.get_market(ticker)
                    orderbook = await self.kalshi_client.get_orderbook(ticker)
                except Exception as e:
                    logger.warning(
                        "failed_to_fetch_market",
                        ticker=ticker,
                        error=str(e),
                    )
                    continue
            else:
                # Paper trading without live client - fetch from database
                market_data, orderbook = self._get_market_data_from_db(ticker)

            # Get existing position
            position = self.simulator.get_position(ticker)

            outcome_ctx = OutcomeContext(
                outcome=outcome_type,
                ticker=ticker,
                market_data=market_data,
                orderbook=orderbook,
                current_position=position.quantity if position else 0,
                average_entry_price=position.average_entry_price if position else 0.0,
            )
            outcomes.append(outcome_ctx)

        return outcomes

    async def _get_model_prediction(
        self,
        fixture: FixtureContext,
    ) -> ModelPrediction:
        """Get model prediction for a fixture.

        This is a placeholder that would integrate with the actual prediction
        models from the modeling module.
        """
        # TODO: Integrate with actual prediction models
        # For now, use simple home advantage priors
        # Home win: 45%, Draw: 25%, Away win: 30%
        return ModelPrediction(
            model_name="home_advantage_prior",
            model_version="1.0.0",
            prob_home_win=0.45,
            prob_draw=0.25,
            prob_away_win=0.30,
            confidence=0.7,
        )

    async def _check_exits(self, summary: AgentRunSummary) -> None:
        """Check exit conditions for open positions."""
        open_positions = self.simulator.open_positions

        for ticker, position in open_positions.items():
            if not position.is_open:
                continue

            # Get current market data
            market_data: MarketData | None = None
            orderbook: OrderbookData | None = None

            if self.kalshi_client:
                try:
                    market_data = await self.kalshi_client.get_market(ticker)
                    orderbook = await self.kalshi_client.get_orderbook(ticker)
                except Exception as e:
                    logger.warning(
                        "failed_to_fetch_market_for_exit",
                        ticker=ticker,
                        error=str(e),
                    )
                    continue

            # Mark position to market
            if orderbook and orderbook.mid_price:
                self.simulator.mark_to_market(ticker, orderbook.mid_price)

            # Get fixture context
            fixture_id = position.fixture_id or 0
            fixture_ctx = await self._get_fixture_by_id(fixture_id)
            if not fixture_ctx:
                continue

            # Build outcome context
            outcome_ctx = OutcomeContext(
                outcome=position.outcome,
                ticker=ticker,
                market_data=market_data,
                orderbook=orderbook,
                current_position=position.quantity,
                average_entry_price=position.average_entry_price,
            )

            # Get updated model prediction
            model_prediction = await self._get_model_prediction(fixture_ctx)

            # Evaluate exit
            decision = self.strategy.evaluate_exit(
                outcome=outcome_ctx,
                model_prediction=model_prediction,
                entry_price=position.average_entry_price,
                fixture_id=fixture_id,
            )

            summary.decisions_made += 1

            # Store decision record first (before order, due to FK constraint)
            self._store_decision_record(decision)

            # Execute exit if needed
            if decision.order_params is not None:
                if not self.dry_run and self.kalshi_client is not None:
                    # Live trading: place real order via Kalshi API
                    order_data = await self._execute_live_order(decision, summary)
                    if order_data:
                        decision.order_placed = True
                        decision.order_id = order_data.order_id
                        self._update_decision_with_order(decision)
                        self.simulator.execute_decision(decision, orderbook)
                else:
                    # Paper trading
                    order = self.simulator.execute_decision(decision, orderbook)

                    if order:
                        decision.order_placed = True
                        decision.order_id = order.order_id
                        self._update_decision_with_order(decision)
                        self._store_paper_order(order)

    async def _get_fixture_by_id(self, fixture_id: int) -> FixtureContext | None:
        """Get fixture context by ID."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT
                f.fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.kickoff_utc,
                f.league_id,
                m.mapping_version,
                m.structure_type,
                m.ticker_home_win,
                m.ticker_draw,
                m.ticker_away_win,
                m.event_ticker,
                m.confidence_score,
                th.name as home_team_name,
                ta.name as away_team_name
            FROM fixtures_v2 f
            JOIN fixture_market_map m ON f.fixture_id = m.fixture_id
            LEFT JOIN teams th ON f.home_team_id = th.team_id
            LEFT JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.fixture_id = ?
            LIMIT 1
            """,
            (fixture_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        mapping = FixtureMarketMapping(
            fixture_id=row["fixture_id"],
            mapping_version=row["mapping_version"],
            structure_type=row["structure_type"],
            ticker_home_win=row["ticker_home_win"],
            ticker_draw=row["ticker_draw"],
            ticker_away_win=row["ticker_away_win"],
            event_ticker=row["event_ticker"],
            confidence_score=row["confidence_score"],
        )

        kickoff = (
            datetime.fromisoformat(row["kickoff_utc"])
            if row["kickoff_utc"]
            else utc_now()
        )

        return FixtureContext(
            fixture_id=row["fixture_id"],
            home_team=row["home_team_name"] or str(row["home_team_id"]),
            away_team=row["away_team_name"] or str(row["away_team_id"]),
            kickoff_time=kickoff,
            league=str(row["league_id"]),
            mapping=mapping,
        )

    # --- Database operations ---

    def _create_agent_run(self, summary: AgentRunSummary) -> int:
        """Create agent run record in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO agent_runs (
                run_type, status, started_at, config_hash, config_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary.run_type,
                summary.status,
                summary.started_at.isoformat(),
                summary.config_hash,
                json.dumps(summary.config_summary),
            ),
        )
        self.db.connection.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def _complete_agent_run(self, summary: AgentRunSummary) -> None:
        """Update completed agent run in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            UPDATE agent_runs SET
                status = ?,
                completed_at = ?,
                fixtures_evaluated = ?,
                markets_evaluated = ?,
                decisions_made = ?,
                orders_placed = ?,
                orders_filled = ?,
                orders_rejected = ?,
                total_realized_pnl = ?,
                total_unrealized_pnl = ?,
                total_exposure = ?,
                position_count = ?,
                error_count = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                summary.status,
                summary.completed_at.isoformat() if summary.completed_at else None,
                summary.fixtures_evaluated,
                summary.markets_evaluated,
                summary.decisions_made,
                summary.orders_placed,
                summary.orders_filled,
                summary.orders_rejected,
                summary.total_realized_pnl,
                summary.total_unrealized_pnl,
                summary.total_exposure,
                summary.position_count,
                summary.error_count,
                summary.error_message,
                summary.run_id,
            ),
        )
        self.db.connection.commit()

    def _store_decision_record(self, decision: DecisionRecord) -> None:
        """Store decision record in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome, timestamp,
                market_snapshot_json, model_prediction_json, current_position_json,
                edge_calculation_json, kelly_sizing_json,
                action, exit_reason, order_params_json,
                rationale, filters_passed_json, rejection_reason,
                order_placed, order_id, execution_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.run_id,
                decision.fixture_id,
                decision.market_ticker,
                decision.outcome,
                decision.timestamp.isoformat(),
                json.dumps(decision.market_snapshot.to_dict())
                if decision.market_snapshot
                else None,
                json.dumps(decision.model_prediction.to_dict())
                if decision.model_prediction
                else None,
                json.dumps(decision.current_position.to_dict())
                if decision.current_position
                else None,
                json.dumps(decision.edge_calculation.to_dict())
                if decision.edge_calculation
                else None,
                json.dumps(decision.kelly_sizing.to_dict())
                if decision.kelly_sizing
                else None,
                decision.action.value,
                decision.exit_reason.value if decision.exit_reason else None,
                json.dumps(decision.order_params.to_dict())
                if decision.order_params
                else None,
                decision.rationale,
                json.dumps(decision.filters_passed),
                decision.rejection_reason,
                1 if decision.order_placed else 0,
                decision.order_id,
                decision.execution_error,
            ),
        )
        self.db.connection.commit()

    def _update_decision_with_order(self, decision: DecisionRecord) -> None:
        """Update decision record with order info after execution."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            UPDATE decision_records 
            SET order_placed = ?, order_id = ?
            WHERE decision_id = ?
            """,
            (
                1 if decision.order_placed else 0,
                decision.order_id,
                decision.decision_id,
            ),
        )
        self.db.connection.commit()

    def _store_paper_order(self, order: PaperOrder) -> None:
        """Store paper order in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO paper_orders (
                order_id, run_id, decision_id, ticker, side, action,
                order_type, price, quantity, filled_quantity, status,
                fill_price, fees, rejection_reason, created_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.run_id,
                order.decision_id,
                order.ticker,
                order.side,
                order.action,
                order.order_type,
                order.price,
                order.quantity,
                order.filled_quantity,
                order.status,
                order.fill_price,
                order.fees,
                order.rejection_reason,
                order.created_at.isoformat(),
                order.filled_at.isoformat() if order.filled_at else None,
            ),
        )
        self.db.connection.commit()

    def _store_paper_fill(self, fill: PaperFill) -> None:
        """Store paper fill in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO paper_fills (
                fill_id, order_id, ticker, side, action, price, quantity, fees, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.ticker,
                fill.side,
                fill.action,
                fill.price,
                fill.quantity,
                fill.fees,
                fill.timestamp.isoformat(),
            ),
        )
        self.db.connection.commit()

    def _store_pnl_snapshot(self, snapshot: PnlSnapshot) -> None:
        """Store P&L snapshot in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO pnl_snapshots (
                snapshot_id, run_id, timestamp,
                total_realized_pnl, total_unrealized_pnl, total_pnl,
                total_exposure, position_count, bankroll, positions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.run_id,
                snapshot.timestamp.isoformat(),
                snapshot.total_realized_pnl,
                snapshot.total_unrealized_pnl,
                snapshot.total_pnl,
                snapshot.total_exposure,
                snapshot.position_count,
                snapshot.bankroll,
                json.dumps(snapshot.positions_snapshot),
            ),
        )
        self.db.connection.commit()

    async def _execute_live_order(
        self,
        decision: DecisionRecord,
        run_summary: AgentRunSummary,
    ) -> OrderData | None:
        """Execute a live order via Kalshi API.
        
        Args:
            decision: Decision record with order_params.
            run_summary: Current run summary for tracking.
            
        Returns:
            OrderData if order was placed, None otherwise.
        """
        if decision.order_params is None:
            return None
            
        if self.kalshi_client is None:
            logger.error("kalshi_client_missing", msg="Cannot place live order without Kalshi client")
            return None

        order_params = decision.order_params
        
        try:
            logger.info(
                "placing_live_order",
                ticker=order_params.ticker,
                side=order_params.side,
                action=order_params.action,
                price=order_params.price,
                quantity=order_params.quantity,
            )
            
            order_data = await self.kalshi_client.place_limit_order(
                ticker=order_params.ticker,
                side=order_params.side,
                action=order_params.action,
                price=order_params.price,
                quantity=order_params.quantity,
                client_order_id=decision.decision_id,
            )
            
            logger.info(
                "live_order_placed",
                order_id=order_data.order_id,
                ticker=order_params.ticker,
                status=order_data.status,
                filled_quantity=order_data.filled_quantity,
            )
            
            run_summary.orders_placed += 1
            if order_data.filled_quantity > 0:
                run_summary.orders_filled += 1
            
            # Store in live_orders table
            self._store_live_order(order_data, decision)
            
            return order_data
            
        except Exception as e:
            logger.error(
                "live_order_failed",
                ticker=order_params.ticker,
                error=str(e),
            )
            run_summary.error_count += 1
            return None

    def _store_live_order(self, order_data: OrderData, decision: DecisionRecord) -> None:
        """Store live order in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO live_orders (
                order_id, decision_id, ticker, side, action,
                order_type, price, quantity, filled_quantity, status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_data.order_id,
                decision.decision_id,
                order_data.ticker,
                order_data.side,
                order_data.action,
                order_data.order_type,
                order_data.price,
                order_data.quantity,
                order_data.filled_quantity,
                order_data.status,
                order_data.created_time.isoformat() if order_data.created_time else utc_now().isoformat(),
            ),
        )
        self.db.connection.commit()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Footbe trading agent"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="paper",
        help="Trading mode: paper (simulated) or live",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minutes between trading loop iterations",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once instead of continuous loop",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades (even in live mode)",
    )

    parser.add_argument(
        "--strategy-config",
        type=str,
        default="configs/strategy_config.yaml",
        help="Path to strategy configuration file",
    )

    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Override initial bankroll",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for paper trading (for reproducibility)",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configurations
    config = load_config(args.config)
    strategy_config = StrategyConfig.from_yaml(args.strategy_config)

    # Override bankroll if specified
    if args.bankroll is not None:
        strategy_config.initial_bankroll = args.bankroll

    # Initialize database
    db = Database(config.database.path)
    db.connect()
    db.migrate()

    # Initialize Kalshi client (if credentials available)
    kalshi_client: KalshiClient | None = None
    if config.kalshi.api_key_id and config.kalshi.private_key_path:
        kalshi_client = KalshiClient(config.kalshi)

    # Initialize strategy and simulator
    strategy = EdgeStrategy(strategy_config)
    simulator = PaperTradingSimulator(
        config=strategy_config,
        initial_bankroll=strategy_config.initial_bankroll,
    )

    if args.seed is not None:
        simulator.set_seed(args.seed)

    # Create trading agent
    agent = TradingAgent(
        db=db,
        kalshi_client=kalshi_client,
        strategy=strategy,
        simulator=simulator,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    try:
        if kalshi_client:
            async with kalshi_client:
                if args.once:
                    await agent.run_once()
                else:
                    await agent.run_loop(args.interval)
        else:
            logger.warning("running_without_kalshi_client")
            if args.once:
                await agent.run_once()
            else:
                await agent.run_loop(args.interval)

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
