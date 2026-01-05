# Footbe Trader

Automated EPL prediction-market trading agent.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
# Run heartbeat (dev mode)
python scripts/run_heartbeat.py --config configs/dev.yaml

# Or using the installed command
footbe-heartbeat --config configs/dev.yaml
```

## Development

```bash
# Run tests
make test

# Lint and format
make lint
make format

# Run all checks
make check
```

## Project Structure

```
src/footbe_trader/
├── common/       # Config, logging, time utils
├── storage/      # SQLite schema + access layer
├── football/     # API-Football client stub
├── kalshi/       # Kalshi client stub
├── modeling/     # Probability model interface
├── strategy/     # Signal generation
├── execution/    # Order placement
└── agent/        # Orchestrator loop, policy, pacing
```

## Trading Policy

The agent uses a goal-driven policy system to manage risk and aggressiveness:

- **Target**: 10% weekly equity growth
- **Drawdown Throttle**: Reduces sizing at 5%, 10%, 15% drawdown thresholds
- **Pacing**: Adjusts aggressiveness based on progress toward target
- **Time-to-Kickoff**: Different edge requirements for early vs optimal vs late markets

See [docs/AGENT_POLICY.md](docs/AGENT_POLICY.md) for full documentation.

### Run Modes

| Mode | Target | Max DD | Use Case |
|------|--------|--------|----------|
| `paper_conservative` | 3% | 10% | Model validation |
| `paper_aggressive` | 10% | 15% | Stress testing |
| `live_small` | 5% | 12% | Small stake validation |
| `live_scaled` | 8% | 15% | Production trading |

Configuration files: `config/run_modes/*.yaml`
