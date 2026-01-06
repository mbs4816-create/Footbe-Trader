"""Google Sheets integration for report export.

Provides functionality to export trading reports to Google Sheets,
including runs, decisions, P&L snapshots, and summary statistics.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from footbe_trader.common.logging import get_logger
from footbe_trader.reporting.queries import (
    DayStats,
    DecisionSummary,
    EdgeBucketStats,
    PnLSnapshot,
    RunSummary,
    WeekStats,
)

logger = get_logger(__name__)

# OAuth scopes required for Google Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

# Token cache location
TOKEN_PATH = Path.home() / ".footbe_trader" / "google_token.json"


@dataclass
class SheetConfig:
    """Configuration for Google Sheets export."""
    
    client_id: str
    client_secret: str
    spreadsheet_url: str
    
    @classmethod
    def from_env(cls) -> "SheetConfig":
        """Load configuration from environment variables."""
        client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        spreadsheet_url = os.getenv("GOOGLE_SHEETS_URL", "")
        
        if not client_id or not client_secret:
            raise ValueError(
                "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in environment"
            )
        if not spreadsheet_url:
            raise ValueError("GOOGLE_SHEETS_URL must be set in environment")
        
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            spreadsheet_url=spreadsheet_url,
        )
    
    @property
    def spreadsheet_id(self) -> str:
        """Extract spreadsheet ID from URL."""
        # URL format: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/...
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", self.spreadsheet_url)
        if not match:
            raise ValueError(f"Could not extract spreadsheet ID from URL: {self.spreadsheet_url}")
        return match.group(1)


class GoogleSheetsClient:
    """Client for exporting reports to Google Sheets."""
    
    def __init__(self, config: SheetConfig | None = None):
        """Initialize the Google Sheets client.
        
        Args:
            config: Sheet configuration. If None, loads from environment.
        """
        self.config = config or SheetConfig.from_env()
        self._client: gspread.Client | None = None
        self._spreadsheet: gspread.Spreadsheet | None = None
    
    def _get_credentials(self) -> Credentials:
        """Get or refresh OAuth credentials.
        
        Returns:
            Valid Google credentials.
        """
        creds = None
        
        # Try to load existing token
        if TOKEN_PATH.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
            except Exception as e:
                logger.warning("failed_to_load_token", error=str(e))
        
        # If no valid credentials, run OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
            else:
                # Create OAuth flow with client credentials (Web application type)
                client_config = {
                    "web": {
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost:8080", "http://127.0.0.1:8080"],
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                creds = flow.run_local_server(port=8080)
            
            # Save credentials for next time
            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())
            logger.info("google_credentials_saved", path=str(TOKEN_PATH))
        
        return creds
    
    def connect(self) -> gspread.Spreadsheet:
        """Connect to the Google Spreadsheet.
        
        Returns:
            The connected Spreadsheet object.
        """
        if self._spreadsheet is not None:
            return self._spreadsheet
        
        creds = self._get_credentials()
        self._client = gspread.authorize(creds)
        self._spreadsheet = self._client.open_by_key(self.config.spreadsheet_id)
        
        logger.info(
            "connected_to_spreadsheet",
            title=self._spreadsheet.title,
            spreadsheet_id=self.config.spreadsheet_id,
        )
        
        return self._spreadsheet
    
    def _get_or_create_worksheet(
        self,
        name: str,
        rows: int = 1000,
        cols: int = 26,
    ) -> gspread.Worksheet:
        """Get or create a worksheet by name.
        
        Args:
            name: Worksheet name.
            rows: Number of rows if creating new.
            cols: Number of columns if creating new.
            
        Returns:
            The worksheet.
        """
        spreadsheet = self.connect()
        
        try:
            worksheet = spreadsheet.worksheet(name)
            logger.debug("worksheet_found", name=name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=name, rows=rows, cols=cols)
            logger.info("worksheet_created", name=name)
        
        return worksheet
    
    def export_runs(self, runs: list[RunSummary], sheet_name: str = "Runs") -> int:
        """Export run summaries to a worksheet.
        
        Args:
            runs: List of run summaries to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not runs:
            logger.info("no_runs_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Run ID", "Type", "Status", "Started At", "Completed At",
            "Fixtures", "Markets", "Decisions", "Orders Placed", "Orders Filled",
            "Realized P&L", "Unrealized P&L", "Total P&L", "Exposure", "Positions",
            "Errors",
        ]
        
        # Data rows
        rows = [headers]
        for run in runs:
            rows.append([
                run.run_id,
                run.run_type,
                run.status,
                run.started_at.isoformat() if run.started_at else "",
                run.completed_at.isoformat() if run.completed_at else "",
                run.fixtures_evaluated,
                run.markets_evaluated,
                run.decisions_made,
                run.orders_placed,
                run.orders_filled,
                round(run.total_realized_pnl, 2),
                round(run.total_unrealized_pnl, 2),
                round(run.total_pnl, 2),
                round(run.total_exposure, 2),
                run.position_count,
                run.error_count,
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:P1", {
            "backgroundColor": {"red": 0.2, "green": 0.4, "blue": 0.8},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_runs", count=len(runs), sheet=sheet_name)
        return len(runs)
    
    def export_decisions(
        self,
        decisions: list[DecisionSummary],
        sheet_name: str = "Decisions",
    ) -> int:
        """Export decision records to a worksheet.
        
        Args:
            decisions: List of decisions to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not decisions:
            logger.info("no_decisions_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Timestamp", "Run ID", "Market", "Outcome", "Action",
            "Edge", "Edge Bucket", "Size Intended", "Size Executed",
            "Order Placed", "Rejection Reason",
            "Bid", "Ask", "Model Prob",
            "League", "Hours to Kickoff",
            "Pace State", "Drawdown", "Throttle",
        ]
        
        # Data rows
        rows = [headers]
        for d in decisions:
            rows.append([
                d.timestamp.isoformat() if d.timestamp else "",
                d.run_id,
                d.market_ticker,
                d.outcome,
                d.action,
                round(d.edge * 100, 2) if d.edge else "",
                d.edge_bucket or "",
                d.size_intended or "",
                d.size_executed or "",
                "Yes" if d.order_placed else "No",
                d.rejection_reason or "",
                round(d.best_bid, 2) if d.best_bid else "",
                round(d.best_ask, 2) if d.best_ask else "",
                round(d.model_prob * 100, 2) if d.model_prob else "",
                d.league_key or "",
                round(d.hours_to_kickoff, 1) if d.hours_to_kickoff else "",
                d.pace_state or "",
                round(d.drawdown * 100, 2) if d.drawdown else "",
                round(d.throttle_multiplier, 2) if d.throttle_multiplier else "",
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:S1", {
            "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.4},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_decisions", count=len(decisions), sheet=sheet_name)
        return len(decisions)
    
    def export_pnl_snapshots(
        self,
        snapshots: list[PnLSnapshot],
        sheet_name: str = "P&L Snapshots",
    ) -> int:
        """Export P&L snapshots to a worksheet.
        
        Args:
            snapshots: List of P&L snapshots to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not snapshots:
            logger.info("no_snapshots_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Timestamp", "Run ID", "Realized P&L", "Unrealized P&L",
            "Total P&L", "Exposure", "Positions", "Bankroll",
        ]
        
        # Data rows
        rows = [headers]
        for s in snapshots:
            rows.append([
                s.timestamp.isoformat() if s.timestamp else "",
                s.run_id,
                round(s.total_realized_pnl, 2),
                round(s.total_unrealized_pnl, 2),
                round(s.total_pnl, 2),
                round(s.total_exposure, 2),
                s.position_count,
                round(s.bankroll, 2),
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:H1", {
            "backgroundColor": {"red": 0.6, "green": 0.4, "blue": 0.2},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_pnl_snapshots", count=len(snapshots), sheet=sheet_name)
        return len(snapshots)
    
    def export_daily_stats(
        self,
        stats: list[DayStats],
        sheet_name: str = "Daily Stats",
    ) -> int:
        """Export daily statistics to a worksheet.
        
        Args:
            stats: List of daily stats to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not stats:
            logger.info("no_daily_stats_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Date", "Runs", "Decisions", "Trades", "Skips",
            "Total P&L", "Ending Equity", "Max Drawdown %",
        ]
        
        # Data rows
        rows = [headers]
        for s in stats:
            rows.append([
                s.date,
                s.run_count,
                s.decisions_count,
                s.trades_count,
                s.skips_count,
                round(s.total_pnl, 2),
                round(s.ending_equity, 2),
                round(s.max_drawdown * 100, 2),
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:H1", {
            "backgroundColor": {"red": 0.5, "green": 0.3, "blue": 0.6},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_daily_stats", count=len(stats), sheet=sheet_name)
        return len(stats)
    
    def export_weekly_stats(
        self,
        stats: list[WeekStats],
        sheet_name: str = "Weekly Stats",
    ) -> int:
        """Export weekly statistics to a worksheet.
        
        Args:
            stats: List of weekly stats to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not stats:
            logger.info("no_weekly_stats_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Week Start", "Week End", "Runs", "Decisions", "Trades", "Skips",
            "Total P&L", "Return %", "Target %", "Pace Status",
            "Ending Equity", "Max Drawdown %",
        ]
        
        # Data rows
        rows = [headers]
        for s in stats:
            rows.append([
                s.week_start,
                s.week_end,
                s.run_count,
                s.decisions_count,
                s.trades_count,
                s.skips_count,
                round(s.total_pnl, 2),
                round(s.return_pct * 100, 2),
                round(s.target_return * 100, 2),
                s.pace_status,
                round(s.ending_equity, 2),
                round(s.max_drawdown * 100, 2),
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:L1", {
            "backgroundColor": {"red": 0.3, "green": 0.5, "blue": 0.7},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_weekly_stats", count=len(stats), sheet=sheet_name)
        return len(stats)
    
    def export_edge_buckets(
        self,
        buckets: list[EdgeBucketStats],
        sheet_name: str = "Edge Buckets",
    ) -> int:
        """Export edge bucket statistics to a worksheet.
        
        Args:
            buckets: List of edge bucket stats to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not buckets:
            logger.info("no_edge_buckets_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Bucket", "Total Count", "Trades", "Skips",
            "Total P&L", "Avg Edge %", "Win Rate %",
        ]
        
        # Data rows
        rows = [headers]
        for b in buckets:
            rows.append([
                b.bucket,
                b.count,
                b.trades_count,
                b.skips_count,
                round(b.total_pnl, 2),
                round(b.avg_edge * 100, 2) if b.avg_edge else "",
                round(b.win_rate * 100, 2) if b.win_rate else "",
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:G1", {
            "backgroundColor": {"red": 0.7, "green": 0.5, "blue": 0.3},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_edge_buckets", count=len(buckets), sheet=sheet_name)
        return len(buckets)
    
    def export_positions(
        self,
        positions: list[dict[str, Any]],
        sheet_name: str = "Positions",
    ) -> int:
        """Export current positions to a worksheet.
        
        Args:
            positions: List of position dicts to export.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not positions:
            logger.info("no_positions_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers
        headers = [
            "Ticker", "Quantity", "Avg Entry Price", "Mark Price",
            "Unrealized P&L", "Realized P&L", "Updated At",
        ]
        
        # Data rows
        rows = [headers]
        for p in positions:
            rows.append([
                p.get("ticker", ""),
                p.get("quantity", 0),
                round(p.get("average_entry_price", 0), 2),
                round(p.get("mark_price", 0), 2),
                round(p.get("unrealized_pnl", 0), 2),
                round(p.get("realized_pnl", 0), 2),
                p.get("updated_at", ""),
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:G1", {
            "backgroundColor": {"red": 0.4, "green": 0.6, "blue": 0.5},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_positions", count=len(positions), sheet=sheet_name)
        return len(positions)
    
    def export_summary(
        self,
        summary: dict[str, Any],
        sheet_name: str = "Summary",
    ) -> None:
        """Export a summary overview to a worksheet.
        
        Args:
            summary: Summary data dict.
            sheet_name: Name of the worksheet.
        """
        worksheet = self._get_or_create_worksheet(sheet_name, rows=50, cols=5)
        
        # Create summary rows
        rows = [
            ["Footbe Trader Report Summary"],
            ["Generated At", datetime.now().isoformat()],
            [""],
            ["=== Performance ==="],
            ["Total Realized P&L", f"${summary.get('realized_pnl', 0):.2f}"],
            ["Total Unrealized P&L", f"${summary.get('unrealized_pnl', 0):.2f}"],
            ["Total P&L", f"${summary.get('total_pnl', 0):.2f}"],
            [""],
            ["=== Activity ==="],
            ["Total Runs", summary.get("total_runs", 0)],
            ["Total Decisions", summary.get("total_decisions", 0)],
            ["Total Trades", summary.get("total_trades", 0)],
            ["Total Skips", summary.get("total_skips", 0)],
            [""],
            ["=== Positions ==="],
            ["Open Positions", summary.get("position_count", 0)],
            ["Total Exposure", f"${summary.get('exposure', 0):.2f}"],
            [""],
            ["=== Risk ==="],
            ["Max Drawdown", f"{summary.get('max_drawdown', 0) * 100:.2f}%"],
            ["Current Drawdown", f"{summary.get('current_drawdown', 0) * 100:.2f}%"],
        ]
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format title
        worksheet.format("A1", {
            "textFormat": {"bold": True, "fontSize": 14},
        })
        
        # Format section headers
        for i, row in enumerate(rows, start=1):
            if row and isinstance(row[0], str) and row[0].startswith("==="):
                worksheet.format(f"A{i}", {
                    "textFormat": {"bold": True},
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                })
        
        logger.info("exported_summary", sheet=sheet_name)
    
    def export_live_orders(
        self,
        orders: list[dict[str, Any]],
        sheet_name: str = "Live Orders",
    ) -> int:
        """Export live orders to a worksheet with plain English game names.
        
        Args:
            orders: List of live order dicts to export (should include 'game_name' key).
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not orders:
            logger.info("no_live_orders_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name)
        
        # Headers - added Game Name column
        headers = [
            "ID", "Game", "Bet On", "Sport", "Order ID", "Ticker",
            "Price", "Quantity", "Filled", "Status", "Created At",
        ]
        
        # Data rows
        rows = [headers]
        for o in orders:
            rows.append([
                o.get("id", ""),
                o.get("game_name", ""),  # Plain English name
                o.get("bet_outcome", ""),  # What we're betting on
                o.get("sport", ""),  # Sport category
                o.get("order_id", "")[:8] + "..." if o.get("order_id") else "",  # Shortened
                o.get("ticker", ""),
                f"${o.get('price', 0):.2f}",
                o.get("quantity", 0),
                o.get("filled_quantity", 0),
                o.get("status", ""),
                o.get("created_at", "")[:19] if o.get("created_at") else "",  # Trim timezone
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:K1", {
            "backgroundColor": {"red": 0.1, "green": 0.5, "blue": 0.8},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_live_orders", count=len(orders), sheet=sheet_name)
        return len(orders)
    
    def export_sport_summary(
        self,
        summary_data: list[dict[str, Any]],
        sheet_name: str = "Sport Summary",
    ) -> int:
        """Export summary by sport to a worksheet.
        
        Args:
            summary_data: List of sport summary dicts.
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        if not summary_data:
            logger.info("no_sport_summary_to_export")
            return 0
        
        worksheet = self._get_or_create_worksheet(sheet_name, rows=20, cols=10)
        
        # Headers
        headers = [
            "Sport", "Total Orders", "Total Contracts", "Filled Orders",
            "Pending Orders", "Total Cost ($)", "Avg Price ($)",
        ]
        
        # Data rows
        rows = [headers]
        for s in summary_data:
            rows.append([
                s.get("sport", ""),
                s.get("total_orders", 0),
                s.get("total_contracts", 0),
                s.get("filled_orders", 0),
                s.get("pending_orders", 0),
                f"${s.get('total_cost', 0):.2f}",
                f"${s.get('avg_price', 0):.2f}",
            ])
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format header row
        worksheet.format("A1:G1", {
            "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.3},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        logger.info("exported_sport_summary", count=len(summary_data), sheet=sheet_name)
        return len(summary_data)
    
    def export_narrative(
        self,
        narrative: str,
        sheet_name: str = "Trading Narrative",
    ) -> None:
        """Export a narrative summary to a worksheet.
        
        Args:
            narrative: The narrative text to export.
            sheet_name: Name of the worksheet.
        """
        worksheet = self._get_or_create_worksheet(sheet_name, rows=100, cols=5)
        
        # Split narrative into rows
        lines = narrative.split("\n")
        rows = [[line] for line in lines]
        
        # Clear and update
        worksheet.clear()
        worksheet.update(rows, "A1")
        
        # Format title row
        worksheet.format("A1", {
            "textFormat": {"bold": True, "fontSize": 14},
        })
        
        logger.info("exported_narrative", sheet=sheet_name)
    
    def export_paper_experiments(
        self,
        experiments: list[dict[str, Any]],
        sheet_name: str = "Paper Experiments",
    ) -> int:
        """Export paper trading experiments with explanations to a worksheet.
        
        This tab explains what the system is testing in paper mode, why it's
        doing it, and what the results are. Paper trading is used to test
        new strategies before risking real money.
        
        Args:
            experiments: List of paper experiment dicts with:
                - experiment_name: Name of the experiment
                - hypothesis: What we're testing
                - strategy_changes: Changes being tested vs live
                - start_date: When experiment started
                - end_date: When experiment ended (or "Ongoing")
                - runs: Number of runs executed
                - trades: Number of trades made
                - pnl: Simulated P&L
                - conclusion: What we learned
                - status: "testing", "completed", "promoted", "rejected"
            sheet_name: Name of the worksheet.
            
        Returns:
            Number of rows written.
        """
        worksheet = self._get_or_create_worksheet(sheet_name, rows=100, cols=15)
        
        # Introduction section
        intro_rows = [
            ["📊 Paper Trading Experiments"],
            [""],
            ["Purpose: Paper trading runs simulated trades in the background to test"],
            ["new strategies before deploying them to live trading with real money."],
            [""],
            ["This tab shows what experiments are running, what they're testing,"],
            ["and whether the results suggest promoting the strategy to live."],
            [""],
            ["Paper trading does NOT send Telegram notifications (live only)."],
            [""],
        ]
        
        # Headers for experiments table
        headers = [
            "Experiment", "Status", "Hypothesis", "Strategy Changes",
            "Start Date", "End Date", "Runs", "Trades", "Simulated P&L",
            "Win Rate %", "Conclusion",
        ]
        
        # Data rows
        data_rows = [headers]
        for exp in experiments:
            data_rows.append([
                exp.get("experiment_name", ""),
                exp.get("status", ""),
                exp.get("hypothesis", ""),
                exp.get("strategy_changes", ""),
                exp.get("start_date", ""),
                exp.get("end_date", ""),
                exp.get("runs", 0),
                exp.get("trades", 0),
                f"${exp.get('pnl', 0):.2f}",
                f"{exp.get('win_rate', 0) * 100:.1f}%" if exp.get("win_rate") else "",
                exp.get("conclusion", ""),
            ])
        
        # Combine intro and data
        all_rows = intro_rows + data_rows
        
        # Clear and update
        worksheet.clear()
        worksheet.update(all_rows, "A1")
        
        # Format title
        worksheet.format("A1", {
            "textFormat": {"bold": True, "fontSize": 16},
        })
        
        # Format header row (after intro section)
        header_row = len(intro_rows) + 1
        worksheet.format(f"A{header_row}:K{header_row}", {
            "backgroundColor": {"red": 0.6, "green": 0.2, "blue": 0.8},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        
        # Color-code status column
        for i, exp in enumerate(experiments):
            row_num = len(intro_rows) + 2 + i
            status = exp.get("status", "")
            if status == "promoted":
                color = {"red": 0.2, "green": 0.8, "blue": 0.2}  # Green
            elif status == "rejected":
                color = {"red": 0.8, "green": 0.2, "blue": 0.2}  # Red
            elif status == "testing":
                color = {"red": 0.9, "green": 0.7, "blue": 0.2}  # Yellow/orange
            else:
                color = {"red": 0.8, "green": 0.8, "blue": 0.8}  # Gray
            worksheet.format(f"B{row_num}", {"backgroundColor": color})
        
        logger.info("exported_paper_experiments", count=len(experiments), sheet=sheet_name)
        return len(experiments)


def export_all_reports(
    connection,
    config: SheetConfig | None = None,
    live_only: bool = False,
) -> dict[str, int]:
    """Export all report data to Google Sheets.
    
    Args:
        connection: SQLite database connection.
        config: Optional sheet configuration.
        live_only: If True, only export live trading data (not paper).
        
    Returns:
        Dict with counts of exported items per sheet.
    """
    from footbe_trader.reporting.queries import ReportingQueries
    
    client = GoogleSheetsClient(config)
    queries = ReportingQueries(connection)
    cursor = connection.cursor()
    
    results = {}
    
    if live_only:
        # Export only live trading data
        
        # Get live orders with game names from kalshi_markets
        cursor.execute("""
            SELECT 
                lo.*,
                km.title as game_title,
                CASE 
                    WHEN lo.ticker LIKE '%NBA%' THEN 'NBA'
                    WHEN lo.ticker LIKE '%EPL%' THEN 'EPL'
                    WHEN lo.ticker LIKE '%LALIGA%' THEN 'La Liga'
                    WHEN lo.ticker LIKE '%BUNDESLIGA%' THEN 'Bundesliga'
                    WHEN lo.ticker LIKE '%SERIEA%' THEN 'Serie A'
                    ELSE 'Other'
                END as sport
            FROM live_orders lo
            LEFT JOIN kalshi_markets km ON lo.ticker = km.ticker
            ORDER BY lo.created_at DESC
        """)
        
        live_orders = []
        for row in cursor.fetchall():
            order = dict(row)
            # Parse the bet outcome from ticker (e.g., -BOU means Bournemouth, -TIE means Tie)
            ticker = order.get("ticker", "")
            bet_outcome = ticker.split("-")[-1] if "-" in ticker else order.get("side", "")
            
            # Map common abbreviations to readable names
            outcome_map = {
                "TIE": "Draw",
                "BOU": "Bournemouth",
                "MCI": "Man City",
                "BRE": "Brentford",
                "EVE": "Everton",
                "NEW": "Newcastle",
                "MUN": "Man United",
                "ARS": "Arsenal",
                "LFC": "Liverpool",
                "TOT": "Tottenham",
                "CHE": "Chelsea",
                "GSW": "Golden State",
                "LAC": "LA Clippers",
                "LAL": "LA Lakers",
                "BOS": "Boston",
                "PHI": "Philadelphia",
                "OKC": "OKC Thunder",
                "TOR": "Toronto",
                "DET": "Detroit",
                "NYK": "New York",
                "HOU": "Houston",
                "POR": "Portland",
            }
            readable_outcome = outcome_map.get(bet_outcome, bet_outcome)
            
            # Extract game name from title or construct from ticker
            game_name = order.get("game_title", "")
            if game_name:
                game_name = game_name.replace(" Winner?", "")
            else:
                # Fallback: parse from ticker
                game_name = ticker
            
            order["game_name"] = game_name
            order["bet_outcome"] = readable_outcome
            order["sport"] = order.get("sport", "Unknown")
            live_orders.append(order)
        
        results["Live Orders"] = client.export_live_orders(live_orders)
        
        # Sport Summary
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN ticker LIKE '%NBA%' THEN 'NBA'
                    WHEN ticker LIKE '%EPL%' THEN 'EPL'
                    WHEN ticker LIKE '%LALIGA%' THEN 'La Liga'
                    WHEN ticker LIKE '%BUNDESLIGA%' THEN 'Bundesliga'
                    WHEN ticker LIKE '%SERIEA%' THEN 'Serie A'
                    ELSE 'Other'
                END as sport,
                COUNT(*) as total_orders,
                SUM(quantity) as total_contracts,
                SUM(CASE WHEN status = 'executed' OR status = 'filled' THEN 1 ELSE 0 END) as filled_orders,
                SUM(CASE WHEN status = 'resting' OR status = 'pending' THEN 1 ELSE 0 END) as pending_orders,
                SUM(price * quantity) as total_cost,
                AVG(price) as avg_price
            FROM live_orders
            GROUP BY sport
            ORDER BY total_orders DESC
        """)
        
        sport_summary = [dict(row) for row in cursor.fetchall()]
        results["Sport Summary"] = client.export_sport_summary(sport_summary)
        
        # Get decisions that resulted in live orders
        cursor.execute("""
            SELECT dr.* FROM decision_records dr
            INNER JOIN live_orders lo ON dr.decision_id = lo.decision_id
            ORDER BY dr.timestamp DESC
        """)
        live_decisions = []
        for row in cursor.fetchall():
            live_decisions.append(queries._row_to_decision_summary(row))
        results["Decisions"] = client.export_decisions(live_decisions, sheet_name="Live Decisions")
        
        # Build live summary
        cursor.execute("SELECT COUNT(*) FROM live_orders")
        total_orders = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM live_orders WHERE status = 'executed' OR status = 'filled'")
        filled_orders = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM live_orders WHERE status = 'resting' OR status = 'pending'")
        pending_orders = cursor.fetchone()[0]
        cursor.execute("SELECT SUM(price * quantity) FROM live_orders")
        total_exposure = cursor.fetchone()[0] or 0
        cursor.execute("SELECT SUM(price * filled_quantity) FROM live_orders WHERE status = 'executed' OR status = 'filled'")
        total_cost = cursor.fetchone()[0] or 0
        
        summary = {
            "realized_pnl": 0,  # Would need fills data to calculate
            "unrealized_pnl": 0,
            "total_pnl": 0,
            "total_runs": 0,
            "total_decisions": len(live_decisions),
            "total_trades": filled_orders,
            "total_skips": 0,
            "position_count": pending_orders,
            "exposure": total_exposure,
            "max_drawdown": 0,
            "current_drawdown": 0,
            "total_orders": total_orders,
            "pending_orders": pending_orders,
            "filled_orders": filled_orders,
        }
        client.export_summary(summary, sheet_name="Live Summary")
        
    else:
        # Export paper trading data (original behavior)
        
        # Get all runs
        runs = queries.get_all_runs()
        results["Runs"] = client.export_runs(runs)
        
        # Get all decisions
        decisions = queries.get_all_decisions()
        results["Decisions"] = client.export_decisions(decisions)
        
        # Get P&L snapshots
        snapshots = queries.get_all_pnl_snapshots()
        results["P&L Snapshots"] = client.export_pnl_snapshots(snapshots)
        
        # Get daily stats
        daily_stats = queries.get_all_daily_stats()
        results["Daily Stats"] = client.export_daily_stats(daily_stats)
        
        # Get weekly stats
        weekly_stats = queries.get_all_weekly_stats()
        results["Weekly Stats"] = client.export_weekly_stats(weekly_stats)
        
        # Get positions
        positions = queries.get_all_positions()
        results["Positions"] = client.export_positions(positions)
        
        # Build summary
        summary = {
            "realized_pnl": sum(r.total_realized_pnl for r in runs),
            "unrealized_pnl": sum(r.total_unrealized_pnl for r in runs[-1:]),
            "total_pnl": sum(r.total_pnl for r in runs),
            "total_runs": len(runs),
            "total_decisions": len(decisions),
            "total_trades": sum(1 for d in decisions if d.action != "skip"),
            "total_skips": sum(1 for d in decisions if d.action == "skip"),
            "position_count": len(positions),
            "exposure": runs[-1].total_exposure if runs else 0,
            "max_drawdown": max((d.max_drawdown for d in daily_stats), default=0),
            "current_drawdown": daily_stats[-1].max_drawdown if daily_stats else 0,
        }
        client.export_summary(summary)
    
    logger.info("export_complete", results=results)
    return results


def generate_trading_narrative(connection) -> str:
    """Generate a narrative summary of actual trading decisions and performance.
    
    Args:
        connection: SQLite database connection.
        
    Returns:
        A narrative string summarizing trading activity.
    """
    cursor = connection.cursor()
    now = datetime.now()
    
    lines = [
        f"# Trading Narrative - {now.strftime('%Y-%m-%d %H:%M')} UTC",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Overall stats
    cursor.execute("SELECT COUNT(*) FROM live_orders")
    total_orders = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM live_orders WHERE status = 'executed' OR status = 'filled'")
    filled_orders = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM live_orders WHERE status = 'resting'")
    resting_orders = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(price * quantity) FROM live_orders")
    total_exposure = cursor.fetchone()[0] or 0
    cursor.execute("SELECT SUM(price * filled_quantity) FROM live_orders WHERE status = 'executed' OR status = 'filled'")
    filled_cost = cursor.fetchone()[0] or 0
    
    lines.append(f"**Total Orders Placed:** {total_orders}")
    lines.append(f"**Orders Filled:** {filled_orders}")
    lines.append(f"**Orders Pending:** {resting_orders}")
    lines.append(f"**Total Exposure:** ${total_exposure:.2f}")
    lines.append(f"**Capital Deployed:** ${filled_cost:.2f}")
    lines.append("")
    
    # Sport breakdown
    lines.append("## Breakdown by Sport")
    lines.append("")
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN ticker LIKE '%NBA%' THEN 'NBA Basketball'
                WHEN ticker LIKE '%EPL%' THEN 'English Premier League'
                WHEN ticker LIKE '%LALIGA%' THEN 'La Liga (Spain)'
                WHEN ticker LIKE '%BUNDESLIGA%' THEN 'Bundesliga (Germany)'
                WHEN ticker LIKE '%SERIEA%' THEN 'Serie A (Italy)'
                ELSE 'Other'
            END as sport,
            COUNT(*) as orders,
            SUM(quantity) as contracts,
            SUM(CASE WHEN status = 'executed' OR status = 'filled' THEN 1 ELSE 0 END) as filled,
            SUM(price * quantity) as exposure,
            AVG(price) as avg_price
        FROM live_orders
        GROUP BY sport
        ORDER BY orders DESC
    """)
    
    for row in cursor.fetchall():
        sport, orders, contracts, filled, exposure, avg_price = row
        lines.append(f"### {sport}")
        lines.append(f"- Orders: {orders} ({filled} filled)")
        lines.append(f"- Total Contracts: {contracts}")
        lines.append(f"- Exposure: ${exposure:.2f}")
        lines.append(f"- Average Price: ${avg_price:.2f}")
        lines.append("")
    
    # Recent activity (last 4 hours)
    lines.append("## Recent Activity (Last 4 Hours)")
    lines.append("")
    
    cursor.execute("""
        SELECT 
            lo.ticker,
            km.title,
            lo.side,
            lo.price,
            lo.quantity,
            lo.status,
            lo.created_at
        FROM live_orders lo
        LEFT JOIN kalshi_markets km ON lo.ticker = km.ticker
        WHERE datetime(lo.created_at) > datetime('now', '-4 hours')
        ORDER BY lo.created_at DESC
        LIMIT 20
    """)
    
    recent_orders = cursor.fetchall()
    if recent_orders:
        for row in recent_orders:
            ticker, title, side, price, qty, status, created_at = row
            game_name = title.replace(" Winner?", "") if title else ticker
            bet_on = ticker.split("-")[-1] if "-" in ticker else side
            lines.append(f"- **{game_name}**: Bet {qty} contracts on {bet_on} @ ${price:.2f} - Status: {status}")
    else:
        lines.append("No new orders in the last 4 hours.")
    lines.append("")
    
    # Decisions made
    lines.append("## Key Decisions Made")
    lines.append("")
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN dr.action = 'buy' THEN 1 ELSE 0 END) as buys,
            SUM(CASE WHEN dr.action = 'skip' THEN 1 ELSE 0 END) as skips
        FROM decision_records dr
        INNER JOIN live_orders lo ON dr.decision_id = lo.decision_id
    """)
    row = cursor.fetchone()
    if row:
        total, buys, skips = row
        lines.append(f"- Total decisions leading to orders: {total}")
        lines.append(f"- Buy decisions: {buys}")
        lines.append("")
    
    # Changes implemented
    lines.append("## Changes Implemented")
    lines.append("")
    lines.append("Based on current market conditions and model outputs:")
    lines.append("")
    
    # Check for any patterns in the decisions
    cursor.execute("""
        SELECT 
            CASE 
                WHEN ticker LIKE '%TIE%' THEN 'Draw bets'
                WHEN ticker LIKE '%-' || substr(ticker, -3) THEN 'Win bets'
                ELSE 'Other'
            END as bet_type,
            COUNT(*) as cnt
        FROM live_orders
        GROUP BY bet_type
    """)
    
    for row in cursor.fetchall():
        bet_type, count = row
        lines.append(f"- Placed {count} {bet_type}")
    
    lines.append("")
    lines.append("## Performance Notes")
    lines.append("")
    
    # Filled orders performance
    if filled_orders > 0:
        lines.append(f"- {filled_orders} orders have been filled out of {total_orders} placed ({filled_orders/total_orders*100:.1f}% fill rate)")
        lines.append(f"- Average order size: {total_exposure/total_orders:.0f} contracts")
    else:
        lines.append("- No orders have been filled yet. All orders are currently resting in the order book.")
    
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC*")
    
    return "\n".join(lines)


def export_with_narrative(
    connection,
    config: SheetConfig | None = None,
) -> dict[str, Any]:
    """Export live trading data with narrative to Google Sheets.
    
    Args:
        connection: SQLite database connection.
        config: Optional sheet configuration.
        
    Returns:
        Dict with export results.
    """
    client = GoogleSheetsClient(config)
    
    # Export regular reports
    results = export_all_reports(connection, config, live_only=True)
    
    # Generate and export narrative
    narrative = generate_trading_narrative(connection)
    client.export_narrative(narrative)
    
    logger.info("export_with_narrative_complete", results=results)
    return {"exports": results, "narrative_generated": True}
