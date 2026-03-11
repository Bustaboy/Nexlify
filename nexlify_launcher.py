#!/usr/bin/env python3
"""Nexlify root launcher.

Provides the documented `python nexlify_launcher.py` entrypoint with
lightweight preflight checks before handing off to the main GUI module.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    PROJECT_ROOT / "nexlify" / "gui" / "cyber_gui.py",
    PROJECT_ROOT / "config",
]

REQUIRED_MODULES = ["PyQt5", "qasync", "aiohttp", "pyotp"]


def missing_modules(modules: Iterable[str] = REQUIRED_MODULES) -> List[str]:
    """Return a list of required modules that are not importable."""
    return [module for module in modules if importlib.util.find_spec(module) is None]


def missing_files(files: Iterable[Path] = REQUIRED_FILES) -> List[str]:
    """Return a list of required file paths that do not exist."""
    missing: List[str] = []
    for path in files:
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    return missing


def run_preflight() -> int:
    """Run preflight checks and return process exit code semantics."""
    missing_mods = missing_modules()
    missing_paths = missing_files()

    if not missing_mods and not missing_paths:
        print("✅ Nexlify preflight checks passed")
        return 0

    if missing_paths:
        print("❌ Missing required files/paths:")
        for path in missing_paths:
            print(f"   - {path}")

    if missing_mods:
        print("❌ Missing required Python modules:")
        for module in missing_mods:
            print(f"   - {module}")
        print("   Install dependencies with: pip install -r requirements.txt")

    return 1


def main() -> None:
    """CLI entry point for the root launcher."""
    parser = argparse.ArgumentParser(description="Launch the Nexlify desktop GUI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run launcher preflight checks and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks without opening the GUI",
    )
    args = parser.parse_args()

    exit_code = run_preflight()
    if args.check or args.dry_run or exit_code != 0:
        raise SystemExit(exit_code)

    from nexlify.gui.cyber_gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()

