#!/usr/bin/env python3
"""
Thin shim to expose the legacy top-level build_data.py as a package entrypoint.
Keeps a single source of truth while providing `python -m sft.build_data` and `sft-build-data`.
"""
from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    # Add repo root (where the original build_data.py lives) to sys.path
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../SFT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import build_data as _legacy
    return _legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())

