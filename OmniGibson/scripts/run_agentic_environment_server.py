#!/usr/bin/env python3
"""Preload the conda websockets package, then launch the agentic control service."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    omnigibson_root = repo_root / "OmniGibson"
    if str(omnigibson_root) not in sys.path:
        sys.path.insert(0, str(omnigibson_root))
    websockets = importlib.import_module("websockets")
    importlib.import_module("websockets.asyncio.server")
    importlib.import_module("websockets.sync.client")
    print(
        f"BEHAVIOR_AGENTIC_WEBSOCKETS {getattr(websockets, '__version__', 'unknown')} "
        f"{getattr(websockets, '__file__', 'unknown')}",
        flush=True,
    )
    from omnigibson.learning.agentic.runtime_server import main as runtime_main

    runtime_main()


if __name__ == "__main__":
    main()
