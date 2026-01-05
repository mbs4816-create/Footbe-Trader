#!/usr/bin/env python3
"""Run heartbeat agent.

Usage:
    python scripts/run_heartbeat.py --config configs/dev.yaml
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from footbe_trader.agent.heartbeat import main

if __name__ == "__main__":
    main()
