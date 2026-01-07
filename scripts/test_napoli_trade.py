#!/usr/bin/env python3
"""Test why Napoli trade isn't being placed."""

import sys
from pathlib import Path
from datetime import datetime, UTC

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.storage.database import Database
from footbe_trader.strategy.trading_strategy import EdgeStrategy, StrategyConfig

# Load config
config = StrategyConfig.from_yaml("configs/strategy_config_aggressive.yaml")

print(f"Strategy Config:")
print(f"  min_edge_to_enter: {config.min_edge_to_enter} ({config.min_edge_to_enter * 100:.1f}%)")
print(f"  min_model_confidence: {config.min_model_confidence}")
print(f"  require_pre_game: {config.require_pre_game}")
print(f"  min_minutes_before_kickoff: {config.min_minutes_before_kickoff}")
print(f"  max_spread: {config.max_spread}")
print(f"  min_ask_volume: {config.min_ask_volume}")
print()

# Napoli vs Verona data
model_prob = 0.792  # 79.2%
market_ask = 0.73  # 73¢
edge = model_prob - market_ask

print(f"Napoli vs Verona:")
print(f"  Model probability: {model_prob:.1%}")
print(f"  Market ask: ${market_ask:.2f}")
print(f"  Edge: {edge:.1%}")
print()

# Check filters
print("Filter checks:")
print(f"  Edge >= threshold: {edge} >= {config.min_edge_to_enter} = {edge >= config.min_edge_to_enter}")
print(f"  Model confidence: 0.72 >= {config.min_model_confidence} = {0.72 >= config.min_model_confidence}")

# Check kickoff time
kickoff = datetime(2026, 1, 7, 17, 30, 0).replace(tzinfo=UTC)
now = datetime.now(UTC)
minutes_to_kickoff = (kickoff - now).total_seconds() / 60
print(f"  Minutes to kickoff: {minutes_to_kickoff:.0f}")
print(f"  Pre-game check: {minutes_to_kickoff} >= {config.min_minutes_before_kickoff} = {minutes_to_kickoff >= config.min_minutes_before_kickoff}")
print()

if edge >= config.min_edge_to_enter and 0.72 >= config.min_model_confidence and minutes_to_kickoff >= config.min_minutes_before_kickoff:
    print("✅ ALL FILTERS PASS - Trade SHOULD be placed!")
else:
    print("❌ FILTERS FAIL - Trade rejected")
