#!/usr/bin/env python3
"""Sync portfolio data from Kalshi and export to Google Sheets.

This script fetches actual positions, fills, and P&L from Kalshi
and exports them to Google Sheets with accurate performance data.

Usage:
    python scripts/sync_portfolio_to_sheets.py
"""

import asyncio
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from footbe_trader.common.config import load_config
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.reporting.sheets import GoogleSheetsClient


# Team code mappings for readable names
TEAM_CODES = {
    # NBA
    "GSW": "Golden State Warriors", "LAC": "LA Clippers", "LAL": "LA Lakers",
    "DEN": "Denver Nuggets", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "HOU": "Houston Rockets", "MIA": "Miami Heat", "BOS": "Boston Celtics",
    "NYK": "New York Knicks", "CHI": "Chicago Bulls", "DAL": "Dallas Mavericks",
    "MIL": "Milwaukee Bucks", "ATL": "Atlanta Hawks", "CLE": "Cleveland Cavaliers",
    # EPL
    "BUR": "Burnley", "MUN": "Manchester United", "LIV": "Liverpool", "MCI": "Man City",
    "ARS": "Arsenal", "CHE": "Chelsea", "TOT": "Tottenham", "NEW": "Newcastle",
    # La Liga
    "SEV": "Sevilla", "RCC": "Real Madrid", "BAR": "Barcelona", "LEV": "Levante",
    "ESP": "Espanyol",
    # Serie A  
    "GEN": "Genoa", "CAG": "Cagliari", "VER": "Verona", "LAZ": "Lazio",
    "FIO": "Fiorentina", "ACM": "AC Milan", "LEC": "Lecce", "PAR": "Parma",
}


def parse_ticker_to_game_name(ticker: str) -> str:
    """Convert ticker like KXNBAGAME-26JAN05GSWLAC-GSW to 'Warriors vs Clippers: Warriors Win'."""
    try:
        # Extract parts: KXNBAGAME-26JAN05GSWLAC-GSW
        parts = ticker.split("-")
        if len(parts) < 3:
            return ticker
        
        prefix = parts[0]  # KXNBAGAME
        match_info = parts[1]  # 26JAN05GSWLAC
        outcome = parts[2]  # GSW or TIE
        
        # Extract teams from match_info (last 6 chars are usually team codes)
        teams_str = match_info[-6:]  # GSWLAC
        team1_code = teams_str[:3]  # GSW
        team2_code = teams_str[3:]  # LAC
        
        team1 = TEAM_CODES.get(team1_code, team1_code)
        team2 = TEAM_CODES.get(team2_code, team2_code)
        
        if outcome == "TIE":
            outcome_str = "Draw"
        else:
            outcome_str = f"{TEAM_CODES.get(outcome, outcome)} Win"
        
        # Determine sport
        if "NBA" in prefix:
            sport = "🏀"
        elif "EPL" in prefix:
            sport = "⚽ EPL"
        elif "LALIGA" in prefix:
            sport = "⚽ La Liga"
        elif "SERIEA" in prefix:
            sport = "⚽ Serie A"
        elif "BUNDESLIGA" in prefix:
            sport = "⚽ Bundesliga"
        else:
            sport = "🎯"
        
        return f"{sport}: {team1} vs {team2} - {outcome_str}"
    except:
        return ticker


async def fetch_kalshi_data(config):
    """Fetch all portfolio data from Kalshi."""
    result = {
        "balance": None,
        "positions": [],
        "settled_positions": [],
        "fills": [],
        "total_realized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
    }
    
    async with KalshiClient(config.kalshi) as kalshi:
        # Get account balance
        print("Fetching account balance...")
        balance = await kalshi.get_balance()
        result["balance"] = balance
        print(f"  Balance: ${balance.balance:.2f}")
        print(f"  Portfolio Value: ${balance.portfolio_value:.2f}")
        print(f"  Payout: ${balance.payout:.2f}")
        
        # Get all positions
        print("\nFetching positions...")
        all_positions = []
        try:
            positions, cursor = await kalshi.get_positions()
            all_positions.extend(positions)
            print(f"  Found {len(positions)} positions")
            
            while cursor:
                positions, cursor = await kalshi.get_positions(cursor=cursor)
                all_positions.extend(positions)
        except Exception as e:
            print(f"  Warning: Could not fetch positions: {e}")
        
        # Separate into active and settled based on position count
        # Active: position != 0 (still holding contracts)
        # Settled: position == 0 but has realized P&L or was traded
        for pos in all_positions:
            if pos.position != 0:
                result["positions"].append(pos)
            elif pos.realized_pnl != 0 or pos.total_cost > 0:
                result["settled_positions"].append(pos)
        
        print(f"  Active positions: {len(result['positions'])}")
        print(f"  Settled positions: {len(result['settled_positions'])}")
        
        # Get recent fills
        print("\nFetching recent fills...")
        try:
            fills, cursor = await kalshi.get_fills()
            result["fills"].extend(fills)
            print(f"  Found {len(fills)} fills")
            
            while cursor:
                fills, cursor = await kalshi.get_fills(cursor=cursor)
                result["fills"].extend(fills)
        except Exception as e:
            print(f"  Warning: Could not fetch fills: {e}")
        
        # Calculate P&L
        result["total_realized_pnl"] = sum(p.realized_pnl for p in result["positions"]) + \
                                        sum(p.realized_pnl for p in result["settled_positions"])
        result["total_unrealized_pnl"] = balance.portfolio_value - balance.balance
        
        print(f"\n--- P&L Summary ---")
        print(f"  Realized P&L: ${result['total_realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${result['total_unrealized_pnl']:.2f}")
        print(f"  Total P&L: ${result['total_realized_pnl'] + result['total_unrealized_pnl']:.2f}")
    
    return result


def export_to_sheets(data):
    """Export portfolio data to Google Sheets."""
    print("\nExporting to Google Sheets...")
    client = GoogleSheetsClient()
    
    balance = data["balance"]
    all_positions = data["positions"]
    settled_positions = data["settled_positions"]
    all_fills = data["fills"]
    total_realized_pnl = data["total_realized_pnl"]
    total_unrealized = data["total_unrealized_pnl"]
    
    # Export Portfolio Summary FIRST
    worksheet = client._get_or_create_worksheet("Portfolio Summary", rows=35, cols=5)
    summary_rows = [
        ["🏆 Kalshi Portfolio Summary"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
        [""],
        ["=== Account Balance ==="],
        ["Cash Balance", f"${balance.balance:.2f}"],
        ["Portfolio Value", f"${balance.portfolio_value:.2f}"],
        ["Pending Payout", f"${balance.payout:.2f}"],
        [""],
        ["=== Performance ==="],
        ["Realized P&L", f"${total_realized_pnl:.2f}"],
        ["Unrealized P&L", f"${total_unrealized:.2f}"],
        ["Total P&L", f"${total_realized_pnl + total_unrealized:.2f}"],
        [""],
        ["=== Activity ==="],
        ["Active Positions", len([p for p in all_positions if p.position != 0])],
        ["Settled Positions", len(settled_positions)],
        ["Total Fills", len(all_fills)],
        ["Resting Orders", sum(p.resting_orders_count for p in all_positions)],
        [""],
        ["=== Breakdown by Sport ==="],
    ]
    
    # Calculate P&L by sport
    sport_pnl = {"NBA": 0.0, "EPL": 0.0, "La Liga": 0.0, "Bundesliga": 0.0, "Serie A": 0.0, "Other": 0.0}
    all_pos = all_positions + settled_positions
    for p in all_pos:
        ticker = p.ticker.upper()
        if "NBA" in ticker:
            sport_pnl["NBA"] += p.realized_pnl
        elif "EPL" in ticker:
            sport_pnl["EPL"] += p.realized_pnl
        elif "LALIGA" in ticker:
            sport_pnl["La Liga"] += p.realized_pnl
        elif "BUNDESLIGA" in ticker:
            sport_pnl["Bundesliga"] += p.realized_pnl
        elif "SERIEA" in ticker:
            sport_pnl["Serie A"] += p.realized_pnl
        else:
            sport_pnl["Other"] += p.realized_pnl
    
    for sport, pnl in sport_pnl.items():
        summary_rows.append([sport, f"${pnl:.2f}"])
    
    worksheet.clear()
    worksheet.update(summary_rows, "A1")
    worksheet.format("A1", {"textFormat": {"bold": True, "fontSize": 14}})
    print("  Exported Portfolio Summary")
    
    # Export Active Positions
    active_positions = [p for p in all_positions if p.position != 0]
    if active_positions:
        worksheet = client._get_or_create_worksheet("Active Positions")
        headers = ["Game", "Ticker", "Qty", "Avg Price", "Exposure", "Resting Orders"]
        rows = [headers]
        for p in active_positions:
            rows.append([
                parse_ticker_to_game_name(p.ticker),
                p.ticker,
                p.position,
                f"${p.total_cost / abs(p.position) if p.position else 0:.2f}",
                f"${p.market_exposure:.2f}",
                p.resting_orders_count,
            ])
        worksheet.clear()
        worksheet.update(rows, "A1")
        worksheet.format("A1:F1", {
            "backgroundColor": {"red": 0.1, "green": 0.6, "blue": 0.4},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        print(f"  Exported {len(active_positions)} active positions")
    
    # Export Settled Positions (with outcomes)
    if settled_positions:
        worksheet = client._get_or_create_worksheet("Settled Positions")
        headers = ["Game", "Ticker", "Outcome", "Position", "Realized P&L", "Total Cost"]
        rows = [headers]
        for p in settled_positions:
            # Determine outcome based on realized P&L
            if p.realized_pnl > 0:
                outcome = "WON ✓"
            elif p.realized_pnl < 0:
                outcome = "LOST ✗"
            else:
                outcome = "PUSH"
            
            rows.append([
                parse_ticker_to_game_name(p.ticker),
                p.ticker,
                outcome,
                p.position,
                f"${p.realized_pnl:.2f}",
                f"${p.total_cost:.2f}",
            ])
        worksheet.clear()
        worksheet.update(rows, "A1")
        worksheet.format("A1:F1", {
            "backgroundColor": {"red": 0.5, "green": 0.5, "blue": 0.2},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        print(f"  Exported {len(settled_positions)} settled positions")
    
    # Export Fills with game names
    if all_fills:
        worksheet = client._get_or_create_worksheet("Kalshi Fills")
        headers = ["Time", "Game", "Side", "Action", "Price", "Contracts", "Cost/Payout"]
        rows = [headers]
        for f in all_fills[:200]:  # Limit to 200 most recent
            # Calculate cost (buy) or potential payout (sell)
            cost = f.price * f.count
            rows.append([
                f.created_time.strftime("%Y-%m-%d %H:%M") if f.created_time else "",
                parse_ticker_to_game_name(f.ticker),
                f.side.upper(),
                f.action.upper(),
                f"${f.price:.2f}",
                f.count,
                f"${cost:.2f}",
            ])
        worksheet.clear()
        worksheet.update(rows, "A1")
        worksheet.format("A1:G1", {
            "backgroundColor": {"red": 0.6, "green": 0.3, "blue": 0.5},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        print(f"  Exported {min(len(all_fills), 200)} fills")
    
    # Export Trade Summary (aggregated by game)
    if all_fills:
        worksheet = client._get_or_create_worksheet("Trade Summary")
        
        # Aggregate fills by ticker
        trade_summary = defaultdict(lambda: {"buys": 0, "sells": 0, "buy_cost": 0.0, "sell_cost": 0.0, "contracts": 0})
        for f in all_fills:
            ticker = f.ticker
            if f.action == "buy":
                trade_summary[ticker]["buys"] += f.count
                trade_summary[ticker]["buy_cost"] += f.price * f.count
            else:
                trade_summary[ticker]["sells"] += f.count
                trade_summary[ticker]["sell_cost"] += f.price * f.count
            trade_summary[ticker]["contracts"] += f.count
        
        headers = ["Game", "Total Contracts", "Buys", "Buy Cost", "Sells", "Sell Credit"]
        rows = [headers]
        for ticker, summary in sorted(trade_summary.items(), key=lambda x: x[1]["contracts"], reverse=True):
            rows.append([
                parse_ticker_to_game_name(ticker),
                summary["contracts"],
                summary["buys"],
                f"${summary['buy_cost']:.2f}",
                summary["sells"],
                f"${summary['sell_cost']:.2f}",
            ])
        
        worksheet.clear()
        worksheet.update(rows, "A1")
        worksheet.format("A1:F1", {
            "backgroundColor": {"red": 0.3, "green": 0.5, "blue": 0.7},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })
        print(f"  Exported trade summary for {len(trade_summary)} games")
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60 + "\n")


async def main():
    """Main entry point."""
    print(f"\n{'='*60}")
    print(f"Syncing Portfolio from Kalshi - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "dev.yaml"
    try:
        config = load_config(config_path)
        print(f"Config loaded: Kalshi {'DEMO' if config.kalshi.use_demo else 'PRODUCTION'}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    # Fetch data from Kalshi
    data = await fetch_kalshi_data(config)
    
    # Export to Google Sheets
    export_to_sheets(data)


if __name__ == "__main__":
    asyncio.run(main())
