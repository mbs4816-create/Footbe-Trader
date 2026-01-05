# Universal Market Mapping System

This module provides a universal system for mapping API-Football fixtures to Kalshi market tickers across any soccer league.

## Overview

The mapping system consists of several components:

1. **League Discovery** - Discovers and syncs leagues from API-Football
2. **Kalshi Market Discovery** - Identifies and classifies Kalshi soccer markets  
3. **Name Normalization** - Normalizes team and league names for cross-platform matching
4. **Fixture-Market Mapping** - Links fixtures to markets using candidate scoring

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  API-Football   │     │     Kalshi       │
│   Fixtures      │     │    Markets       │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────────┐
│ League         │      │ Soccer Market      │
│ Discovery      │      │ Classifier         │
└────────┬───────┘      └────────┬───────────┘
         │                       │
         ▼                       ▼
┌──────────────────────────────────────────┐
│        Name Normalization                │
│   • Team aliases (300+ entries)          │
│   • Diacritic removal                    │
│   • Token stripping (FC, AC, etc.)       │
│   • Fuzzy matching                       │
└────────────────────────┬─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────┐
│       Fixture-Market Mapper              │
│   • Candidate generation                 │
│   • Weighted scoring                     │
│   • Confidence thresholds                │
│   • Manual override support              │
└────────────────────────┬─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────┐
│         fixture_market_map               │
│   (Confirmed mappings)                   │
└──────────────────────────────────────────┘
```

## Key Features

### League-Agnostic Design

The mapper works for **any** soccer league without code changes. To add support for a new league:

1. Add league to `configs/league_aliases.yaml`
2. Add team aliases to `configs/team_aliases.yaml`
3. Run `sync_leagues.py` and `sync_kalshi_soccer_markets.py`

### Configurable Scoring

All scoring parameters are in `configs/mapping_config.yaml`:

```yaml
scoring_weights:
  team_match: 0.40      # How well do team names match?
  date_match: 0.20      # Does kickoff date match market close?
  league_match: 0.15    # Are both in same league?
  market_type: 0.10     # Preference for 1X2 vs moneyline
  text_similarity: 0.15 # Fuzzy title comparison

confidence:
  auto_accept: 0.85     # Auto-accept if score >= 85%
  review: 0.70          # Send to review queue if 70-85%
  min_candidate: 0.30   # Discard candidates below 30%
```

### Manual Override Support

For edge cases, add overrides to `configs/manual_market_overrides.yaml`:

```yaml
overrides:
  # By fixture ID
  fixture_12345:
    event_ticker: "SOCCER-EPL-MCI-ARS"
    ticker_home_win: "SOCCER-EPL-MCI-ARS-H-YES"
    ticker_away_win: "SOCCER-EPL-MCI-ARS-A-YES"
    structure_type: "1X2"
    
  # By team + date
  match_mancity_vs_arsenal_20240315:
    home_team: "manchester city"
    away_team: "arsenal"
    date: "2024-03-15"
    event_ticker: "SOCCER-EPL-MCI-ARS"
```

## CLI Tools

### Sync Leagues

```bash
python scripts/sync_leagues.py --country England
python scripts/sync_leagues.py --country "United States"
python scripts/sync_leagues.py --all
```

### Ingest Fixtures

```bash
# Ingest EPL 2024 season
python scripts/ingest_fixtures.py --league-id 39 --season 2024

# Ingest MLS current season
python scripts/ingest_fixtures.py --league-id 253 --season 2024
```

### Sync Kalshi Markets

```bash
python scripts/sync_kalshi_soccer_markets.py
```

### Run Mapping

```bash
# Map all pending fixtures
python scripts/map_fixtures_to_markets.py --league-id 39

# Show review queue
python scripts/map_fixtures_to_markets.py --show-reviews

# Export mappings
python scripts/map_fixtures_to_markets.py --export mappings.json
```

## Database Schema

### leagues

| Column | Type | Description |
|--------|------|-------------|
| league_id | INT | API-Football league ID |
| league_name | TEXT | Full league name |
| country | TEXT | Country name |
| league_key | TEXT | Canonical key for matching |
| seasons_available | JSON | Available seasons |

### kalshi_events

| Column | Type | Description |
|--------|------|-------------|
| event_ticker | TEXT | Kalshi event ticker |
| series_ticker | TEXT | Kalshi series ticker |
| is_soccer | BOOL | Classified as soccer? |
| league_key | TEXT | Detected league key |
| parsed_home_team | TEXT | Extracted home team |
| parsed_away_team | TEXT | Extracted away team |

### kalshi_markets

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Market ticker |
| event_ticker | TEXT | Parent event |
| is_soccer | BOOL | Classified as soccer? |
| market_type | TEXT | HOME_WIN, AWAY_WIN, DRAW |

### fixture_market_map

| Column | Type | Description |
|--------|------|-------------|
| fixture_id | INT | API-Football fixture ID |
| event_ticker | TEXT | Kalshi event ticker |
| ticker_home_win | TEXT | Home win market |
| ticker_draw | TEXT | Draw market |
| ticker_away_win | TEXT | Away win market |
| confidence_score | REAL | Mapping confidence |
| status | TEXT | AUTO, MANUAL_OVERRIDE |

### mapping_reviews

| Column | Type | Description |
|--------|------|-------------|
| fixture_id | INT | Fixture needing review |
| event_ticker | TEXT | Best candidate event |
| score | REAL | Candidate score |
| reason | TEXT | Why review needed |

## Name Normalization

### Team Name Processing

1. **Lowercase**: "Manchester United" → "manchester united"
2. **Remove diacritics**: "Atlético Madrid" → "atletico madrid"
3. **Strip tokens**: "Chelsea FC" → "chelsea", "AC Milan" → "milan"
4. **Alias lookup**: "Man Utd" → "manchester united"

### Token Stripping

These suffixes/prefixes are removed:
- FC, CF, SC, AFC, AC, AS, SS
- VfB, VfL, SV, FSV, RB, BSC
- Numbers: 04, 05, 1899, etc.

### Alias Configuration

Team aliases are league-specific:

```yaml
epl:
  "man utd": "manchester united"
  "man united": "manchester united"
  "mufc": "manchester united"

la_liga:
  "atletico": "atletico madrid"
  "atleti": "atletico madrid"
  "real": "real madrid"

global:
  "barca": "barcelona"
```

## Market Structure Types

| Type | Description | Outcomes |
|------|-------------|----------|
| 1X2 | Traditional European | Home, Draw, Away |
| NO_DRAW | Moneyline/MLS style | Home, Away |
| HOME_WIN_BINARY | Binary market | Home Yes/No |
| AWAY_WIN_BINARY | Binary market | Away Yes/No |

## Scoring Details

### Team Match Score (40%)

```python
# Normalize both team names
norm_fixture_home = normalizer.normalize(fixture_home)
norm_market_home = normalizer.normalize(market_home)

# Compare with fuzzy matching
home_score = fuzzy_match_ratio(norm_fixture_home, norm_market_home) / 100

# Check if teams are swapped
if home_matches_away and away_matches_home:
    score *= (1 - swap_penalty)  # Usually 0.85
```

### Date Match Score (20%)

```python
# Perfect match = 1.0
if fixture_date == market_close_date:
    score = 1.0
# Same day = 0.9
elif abs(fixture_date - market_close_date) < timedelta(hours=12):
    score = 0.9
# Within window
elif within_time_window:
    score = 0.5
else:
    score = 0.0
```

### League Match Score (15%)

```python
if fixture_league_key == market_league_key:
    score = 1.0
elif fixture_league_key and market_league_key is None:
    score = 0.5  # Market league unknown
else:
    score = 0.0
```

### Market Type Score (10%)

Uses preference weights from config:

```python
preferences = {
    "1X2": 1.0,
    "NO_DRAW": 0.9,
    "MONEYLINE": 0.8,
    "BINARY": 0.6,
    "UNKNOWN": 0.3,
}
score = preferences.get(market_structure, 0.3)
```

### Text Similarity Score (15%)

```python
# Compare full event titles
title_ratio = fuzz.ratio(fixture_title, event_title) / 100
```

## Usage Example

```python
from footbe_trader.strategy.mapping import (
    FixtureMarketMapper,
    MappingConfig,
    ManualOverrides,
)
from footbe_trader.storage import Database

# Initialize
db = Database("footbe.db")
config = MappingConfig()
overrides = ManualOverrides()

mapper = FixtureMarketMapper(db.conn, config, overrides)

# Map a fixture
result = await mapper.map_fixture(fixture_id=12345)

if result.success:
    mapping = result.mapping
    print(f"Home win ticker: {mapping.ticker_home_win}")
    print(f"Away win ticker: {mapping.ticker_away_win}")
    print(f"Confidence: {mapping.confidence_score:.2%}")
else:
    print(f"Mapping failed: {result.reason}")
    print(f"Candidates found: {len(result.candidates)}")
```

## Testing

```bash
# Run mapping tests
pytest tests/strategy/test_normalization.py -v
pytest tests/strategy/test_mapping.py -v
pytest tests/strategy/test_kalshi_discovery.py -v
pytest tests/strategy/test_league_discovery.py -v

# Run all tests
pytest -v
```

## Troubleshooting

### Low Confidence Scores

1. Check team aliases - is the variation missing?
2. Verify league is synced properly
3. Check time window configuration

### Missing Markets

1. Run `sync_kalshi_soccer_markets.py` to refresh
2. Check if Kalshi has markets for that league
3. Verify market classification is detecting soccer

### Wrong Matches

1. Add manual override in `manual_market_overrides.yaml`
2. Review scoring weights in `mapping_config.yaml`
3. Add team aliases for problematic names
