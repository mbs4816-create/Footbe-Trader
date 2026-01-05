# API-Football Field Catalog

This document catalogs all fields from API-Football endpoints used for EPL match prediction modeling.

## Field Availability Labels

| Label | Description |
|-------|-------------|
| 🟢 `PRE_MATCH_SAFE` | Available before kickoff reliably. Safe for pre-match predictions. |
| 🟡 `PRE_MATCH_UNCERTAIN` | May be available before kickoff but often missing (e.g., lineups ~1hr before). |
| 🔴 `POST_MATCH_ONLY` | Only available after match completion. **DO NOT use for pre-match predictions.** |

## Summary

- **Total Fields Cataloged:** 152
- **PRE_MATCH_SAFE:** 82 fields
- **PRE_MATCH_UNCERTAIN:** 27 fields
- **POST_MATCH_ONLY:** 43 fields

*Generated: 2026-01-05 15:35:13 UTC*

## Endpoints

- [fixtures](#fixtures)
- [fixtures/events](#fixturesevents)
- [fixtures/headtohead](#fixturesheadtohead)
- [fixtures/lineups](#fixtureslineups)
- [fixtures/statistics](#fixturesstatistics)
- [injuries](#injuries)
- [odds](#odds)
- [standings](#standings)
- [teams](#teams)

---

## fixtures

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `fixture.date` | `datetime` | 🟢 `PRE_MATCH_SAFE` | Match date/time ISO format | `"2023-08-11T19:00:00+00:00"` |
| `fixture.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Unique fixture identifier | `1035037` |
| `fixture.status.long` | `str` | 🟢 `PRE_MATCH_SAFE` | Full status description | `"Not Started"` |
| `fixture.status.short` | `str` | 🟢 `PRE_MATCH_SAFE` | Short status code (NS, FT, etc.) | `"NS"` |
| `fixture.timestamp` | `int` | 🟢 `PRE_MATCH_SAFE` | Unix timestamp of kickoff | `1691780400` |
| `fixture.timezone` | `str` | 🟢 `PRE_MATCH_SAFE` | Timezone of date field | `"UTC"` |
| `fixture.venue.city` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue city | `"Bournemouth"` |
| `fixture.venue.id` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue unique ID | `10503` |
| `fixture.venue.name` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue name | `"Vitality Stadium"` |
| `league.country` | `str` | 🟢 `PRE_MATCH_SAFE` | League country | `"England"` |
| `league.flag` | `str` | 🟢 `PRE_MATCH_SAFE` | Country flag URL | `"https://media.api-sports.io..."` |
| `league.id` | `int` | 🟢 `PRE_MATCH_SAFE` | League unique ID (39 for EPL) | `39` |
| `league.logo` | `str` | 🟢 `PRE_MATCH_SAFE` | League logo URL | `"https://media.api-sports.io..."` |
| `league.name` | `str` | 🟢 `PRE_MATCH_SAFE` | League name | `"Premier League"` |
| `league.round` | `str` | 🟢 `PRE_MATCH_SAFE` | Match round/gameweek | `"Regular Season - 1"` |
| `league.season` | `int` | 🟢 `PRE_MATCH_SAFE` | Season year | `2023` |
| `teams.away.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Away team unique ID | `48` |
| `teams.away.logo` | `str` | 🟢 `PRE_MATCH_SAFE` | Away team logo URL | `"https://media.api-sports.io..."` |
| `teams.away.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Away team name | `"West Ham"` |
| `teams.home.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Home team unique ID | `35` |
| `teams.home.logo` | `str` | 🟢 `PRE_MATCH_SAFE` | Home team logo URL | `"https://media.api-sports.io..."` |
| `teams.home.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Home team name | `"Bournemouth"` |
| `fixture.referee` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Match referee name | `"Michael Oliver, England"` |
| `fixture.status.elapsed` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Minutes elapsed in match | `90` |
| `goals.away` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Away team final goals | `1` |
| `goals.home` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Home team final goals | `1` |
| `score.extratime.away` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Away goals in extra time | `null` |
| `score.extratime.home` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Home goals in extra time | `null` |
| `score.fulltime.away` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Away goals at fulltime | `1` |
| `score.fulltime.home` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Home goals at fulltime | `1` |
| `score.halftime.away` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Away goals at halftime | `1` |
| `score.halftime.home` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Home goals at halftime | `0` |
| `score.penalty.away` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Away penalty shootout goals | `null` |
| `score.penalty.home` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Home penalty shootout goals | `null` |
| `teams.away.winner` | `bool` (nullable) | 🔴 `POST_MATCH_ONLY` | Whether away team won | `false` |
| `teams.home.winner` | `bool` (nullable) | 🔴 `POST_MATCH_ONLY` | Whether home team won | `true` |

## fixtures/events

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `events.assist.id` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Assist player ID | `747` |
| `events.assist.name` | `str` (nullable) | 🔴 `POST_MATCH_ONLY` | Assist player name | `"Marcus Rashford"` |
| `events.comments` | `str` (nullable) | 🔴 `POST_MATCH_ONLY` | Event comments | `null` |
| `events.detail` | `str` | 🔴 `POST_MATCH_ONLY` | Event detail (Normal Goal, Penalty) | `"Normal Goal"` |
| `events.player.id` | `int` | 🔴 `POST_MATCH_ONLY` | Player ID of event | `882` |
| `events.player.name` | `str` | 🔴 `POST_MATCH_ONLY` | Player name of event | `"Bruno Fernandes"` |
| `events.team.id` | `int` | 🔴 `POST_MATCH_ONLY` | Team ID of event | `33` |
| `events.team.name` | `str` | 🔴 `POST_MATCH_ONLY` | Team name of event | `"Manchester United"` |
| `events.time.elapsed` | `int` | 🔴 `POST_MATCH_ONLY` | Minute of event | `45` |
| `events.time.extra` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Extra time minutes | `2` |
| `events.type` | `str` | 🔴 `POST_MATCH_ONLY` | Event type (Goal, Card, Subst) | `"Goal"` |

## fixtures/headtohead

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `h2h.fixture.date` | `datetime` | 🟢 `PRE_MATCH_SAFE` | Historical fixture date | `"2023-03-12T15:00:00"` |
| `h2h.fixture.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Historical fixture ID | `867946` |
| `h2h.goals.away` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | H2H away goals (historical) | `1` |
| `h2h.goals.home` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | H2H home goals (historical) | `2` |
| `h2h.teams.away.id` | `int` | 🟢 `PRE_MATCH_SAFE` | H2H away team ID | `40` |
| `h2h.teams.away.name` | `str` | 🟢 `PRE_MATCH_SAFE` | H2H away team name | `"Liverpool"` |
| `h2h.teams.home.id` | `int` | 🟢 `PRE_MATCH_SAFE` | H2H home team ID | `33` |
| `h2h.teams.home.name` | `str` | 🟢 `PRE_MATCH_SAFE` | H2H home team name | `"Manchester United"` |

## fixtures/lineups

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `lineups.coach.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Coach ID | `19` |
| `lineups.coach.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Coach name | `"Erik ten Hag"` |
| `lineups.formation` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Team formation (4-3-3) | `"4-3-3"` |
| `lineups.startXI.player.grid` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Player grid position | `"3:2"` |
| `lineups.startXI.player.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Starting player ID | `882` |
| `lineups.startXI.player.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Starting player name | `"Bruno Fernandes"` |
| `lineups.startXI.player.number` | `int` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Player shirt number | `8` |
| `lineups.startXI.player.pos` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Player position (G/D/M/F) | `"M"` |
| `lineups.substitutes.player.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Substitute player ID | `747` |
| `lineups.substitutes.player.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Substitute player name | `"Harry Maguire"` |
| `lineups.substitutes.player.number` | `int` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Substitute shirt number | `5` |
| `lineups.substitutes.player.pos` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Substitute position | `"D"` |
| `lineups.team.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Team ID | `33` |
| `lineups.team.logo` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Team logo URL | `"https://..."` |
| `lineups.team.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Team name | `"Manchester United"` |

## fixtures/statistics

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `statistics.Ball Possession` | `str` (nullable) | 🔴 `POST_MATCH_ONLY` | Ball possession percentage | `"54%"` |
| `statistics.Blocked Shots` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Shots blocked | `4` |
| `statistics.Corner Kicks` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Corners taken | `6` |
| `statistics.Fouls` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Fouls committed | `10` |
| `statistics.Goalkeeper Saves` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Goalkeeper saves | `4` |
| `statistics.Offsides` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Offsides | `2` |
| `statistics.Passes %` | `str` (nullable) | 🔴 `POST_MATCH_ONLY` | Pass accuracy percentage | `"85%"` |
| `statistics.Passes accurate` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Accurate passes | `387` |
| `statistics.Red Cards` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Red cards | `0` |
| `statistics.Shots insidebox` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Shots inside box | `8` |
| `statistics.Shots off Goal` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Shots off target | `3` |
| `statistics.Shots on Goal` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Shots on target | `5` |
| `statistics.Shots outsidebox` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Shots outside box | `4` |
| `statistics.Total Shots` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Total shots | `12` |
| `statistics.Total passes` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Total passes | `456` |
| `statistics.Yellow Cards` | `int` (nullable) | 🔴 `POST_MATCH_ONLY` | Yellow cards | `2` |
| `statistics.expected_goals` | `float` (nullable) | 🔴 `POST_MATCH_ONLY` | Expected goals (xG) | `1.45` |
| `statistics.team.id` | `int` | 🔴 `POST_MATCH_ONLY` | Team ID for these stats | `33` |
| `statistics.team.name` | `str` | 🔴 `POST_MATCH_ONLY` | Team name for these stats | `"Manchester United"` |

## injuries

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `injuries.fixture.date` | `datetime` | 🟡 `PRE_MATCH_UNCERTAIN` | Fixture date for injury report | `"2024-01-15T15:00:00"` |
| `injuries.fixture.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Fixture ID for injury report | `1035037` |
| `injuries.league.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | League ID | `39` |
| `injuries.league.season` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Season year | `2023` |
| `injuries.player.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Injured player ID | `882` |
| `injuries.player.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Injured player name | `"Bruno Fernandes"` |
| `injuries.player.photo` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Player photo URL | `"https://..."` |
| `injuries.player.reason` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Injury reason/description | `"Muscle Injury"` |
| `injuries.player.type` | `str` (nullable) | 🟡 `PRE_MATCH_UNCERTAIN` | Injury type | `"Hamstring"` |
| `injuries.team.id` | `int` | 🟡 `PRE_MATCH_UNCERTAIN` | Injured player team ID | `33` |
| `injuries.team.name` | `str` | 🟡 `PRE_MATCH_UNCERTAIN` | Injured player team name | `"Manchester United"` |

## odds

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `odds.bet.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Bet type ID | `1` |
| `odds.bet.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Bet type name (Match Winner) | `"Match Winner"` |
| `odds.bookmaker.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Bookmaker ID | `8` |
| `odds.bookmaker.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Bookmaker name | `"Bet365"` |
| `odds.fixture.date` | `datetime` | 🟢 `PRE_MATCH_SAFE` | Fixture date | `"2024-01-15T15:00:00"` |
| `odds.fixture.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Fixture ID for odds | `1035037` |
| `odds.league.id` | `int` | 🟢 `PRE_MATCH_SAFE` | League ID | `39` |
| `odds.value.odd` | `str` | 🟢 `PRE_MATCH_SAFE` | Decimal odds | `"1.85"` |
| `odds.value.value` | `str` | 🟢 `PRE_MATCH_SAFE` | Bet selection (Home/Draw/Away) | `"Home"` |

## standings

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `standings.all.draw` | `int` | 🟢 `PRE_MATCH_SAFE` | Total draws | `6` |
| `standings.all.goals.against` | `int` | 🟢 `PRE_MATCH_SAFE` | Goals conceded | `43` |
| `standings.all.goals.for` | `int` | 🟢 `PRE_MATCH_SAFE` | Goals scored | `88` |
| `standings.all.lose` | `int` | 🟢 `PRE_MATCH_SAFE` | Total losses | `6` |
| `standings.all.played` | `int` | 🟢 `PRE_MATCH_SAFE` | Total matches played | `38` |
| `standings.all.win` | `int` | 🟢 `PRE_MATCH_SAFE` | Total wins | `26` |
| `standings.away.draw` | `int` | 🟢 `PRE_MATCH_SAFE` | Away draws | `3` |
| `standings.away.goals.against` | `int` | 🟢 `PRE_MATCH_SAFE` | Away goals conceded | `25` |
| `standings.away.goals.for` | `int` | 🟢 `PRE_MATCH_SAFE` | Away goals scored | `43` |
| `standings.away.lose` | `int` | 🟢 `PRE_MATCH_SAFE` | Away losses | `4` |
| `standings.away.played` | `int` | 🟢 `PRE_MATCH_SAFE` | Away matches played | `19` |
| `standings.away.win` | `int` | 🟢 `PRE_MATCH_SAFE` | Away wins | `12` |
| `standings.description` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Position description | `"Champions League"` |
| `standings.form` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Last 5 match results (WWDLW) | `"WDWWW"` |
| `standings.goalsDiff` | `int` | 🟢 `PRE_MATCH_SAFE` | Goal difference | `45` |
| `standings.group` | `str` | 🟢 `PRE_MATCH_SAFE` | League group (for cup formats) | `"Premier League"` |
| `standings.home.draw` | `int` | 🟢 `PRE_MATCH_SAFE` | Home draws | `3` |
| `standings.home.goals.against` | `int` | 🟢 `PRE_MATCH_SAFE` | Home goals conceded | `18` |
| `standings.home.goals.for` | `int` | 🟢 `PRE_MATCH_SAFE` | Home goals scored | `45` |
| `standings.home.lose` | `int` | 🟢 `PRE_MATCH_SAFE` | Home losses | `2` |
| `standings.home.played` | `int` | 🟢 `PRE_MATCH_SAFE` | Home matches played | `19` |
| `standings.home.win` | `int` | 🟢 `PRE_MATCH_SAFE` | Home wins | `14` |
| `standings.points` | `int` | 🟢 `PRE_MATCH_SAFE` | Total points | `84` |
| `standings.rank` | `int` | 🟢 `PRE_MATCH_SAFE` | Team league position | `1` |
| `standings.status` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Promotion/relegation status | `"same"` |
| `standings.team.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Team unique ID | `42` |
| `standings.team.logo` | `str` | 🟢 `PRE_MATCH_SAFE` | Team logo URL | `"https://media.api-sports.io..."` |
| `standings.team.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Team name | `"Arsenal"` |
| `standings.update` | `datetime` | 🟢 `PRE_MATCH_SAFE` | Last update timestamp | `"2024-01-15T00:00:00+00:00"` |

## teams

| Field Path | Type | Availability | Description | Example |
|------------|------|--------------|-------------|---------|
| `team.code` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Team short code | `"MUN"` |
| `team.country` | `str` | 🟢 `PRE_MATCH_SAFE` | Team country | `"England"` |
| `team.founded` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | Year team founded | `1878` |
| `team.id` | `int` | 🟢 `PRE_MATCH_SAFE` | Team unique ID | `33` |
| `team.logo` | `str` | 🟢 `PRE_MATCH_SAFE` | Team logo URL | `"https://media.api-sports.io..."` |
| `team.name` | `str` | 🟢 `PRE_MATCH_SAFE` | Team name | `"Manchester United"` |
| `team.national` | `bool` | 🟢 `PRE_MATCH_SAFE` | Is national team | `false` |
| `venue.address` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue address | `"Sir Matt Busby Way"` |
| `venue.capacity` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue capacity | `76212` |
| `venue.city` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue city | `"Manchester"` |
| `venue.id` | `int` (nullable) | 🟢 `PRE_MATCH_SAFE` | Team venue ID | `556` |
| `venue.image` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Venue image URL | `"https://media.api-sports.io..."` |
| `venue.name` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Team venue name | `"Old Trafford"` |
| `venue.surface` | `str` (nullable) | 🟢 `PRE_MATCH_SAFE` | Pitch surface type | `"grass"` |

---

## Usage Guidelines

### For Pre-Match Prediction Models

Only use fields marked with 🟢 `PRE_MATCH_SAFE` for features in pre-match prediction models.

Fields marked 🟡 `PRE_MATCH_UNCERTAIN` (like lineups) can be used with proper handling:
- Check if data is available before using
- Have fallback logic when data is missing
- Document the uncertainty in your model

### Validation

The `field_catalog.py` module provides validation functions:

```python
from footbe_trader.football.field_catalog import (
    is_field_pre_match_safe,
    validate_pre_match_fields,
)

# Check single field
assert is_field_pre_match_safe('standings.points')  # True
assert not is_field_pre_match_safe('goals.home')     # False - post match only

# Validate multiple fields
violations = validate_pre_match_fields(['standings.points', 'goals.home'])
if violations:
    raise ValueError(f'Post-match fields in pre-match model: {violations}')
```

### Adding New Fields

When adding new fields to the catalog:

1. Add to `src/footbe_trader/football/field_catalog.py`
2. Run `python scripts/generate_field_catalog.py` to regenerate this doc
3. Run tests to ensure no pre-match models use POST_MATCH_ONLY fields
