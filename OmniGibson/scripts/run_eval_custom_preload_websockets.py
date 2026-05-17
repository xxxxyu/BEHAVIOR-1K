#!/usr/bin/env python3
"""Run eval_custom.py after pinning the conda websockets package.

Some Isaac Sim installs ship a prebundled websockets 12 package that can shadow
the BEHAVIOR conda environment after Kit starts. The websocket eval client uses
websockets.asyncio, which is provided by the conda package used in our behavior
environment. Preloading it here keeps the eval startup path aligned without
changing the policy or networking implementation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preload_websockets() -> None:
    websockets = importlib.import_module("websockets")
    importlib.import_module("websockets.asyncio.server")
    importlib.import_module("websockets.sync.client")
    version = getattr(websockets, "__version__", "unknown")
    location = getattr(websockets, "__file__", "unknown")
    print(f"BEHAVIOR_CLIENT_WEBSOCKETS {version} {location}", flush=True)


def main() -> None:
    repo_root = _repo_root()
    omnigibson_root = repo_root / "OmniGibson"
    if str(omnigibson_root) not in sys.path:
        sys.path.insert(0, str(omnigibson_root))

    _preload_websockets()

    eval_script = omnigibson_root / "omnigibson" / "learning" / "eval_custom.py"
    sys.argv = [str(eval_script), *sys.argv[1:]]
    runpy.run_path(str(eval_script), run_name="__main__")


if __name__ == "__main__":
    main()
