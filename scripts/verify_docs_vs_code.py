#!/usr/bin/env python3
"""Verify core documentation claims against repository code layout.

This script is intentionally lightweight and uses only the Python standard library
so it can run early in CI as a fast drift guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CYBER_GUI = ROOT / "nexlify" / "gui" / "cyber_gui.py"
GUI_INTEGRATION = ROOT / "nexlify" / "gui" / "nexlify_gui_integration.py"

REQUIRED_PATHS = [
    ROOT / "nexlify_launcher.py",
    ROOT / "nexlify" / "gui" / "cyber_gui.py",
    ROOT / "nexlify" / "core" / "arasaka_neural_net.py",
    ROOT / "nexlify" / "financial" / "nexlify_tax_reporter.py",
]

REQUIRED_MODULE_SPECS = [
    "nexlify.gui.cyber_gui",
    "nexlify.gui.nexlify_gui_integration",
    "nexlify.financial.nexlify_tax_reporter",
]

REQUIRED_README_SNIPPETS = [
    "python nexlify_launcher.py",
    "Dashboard",
    "Trading",
    "Portfolio",
    "Strategies",
    "Settings",
    "Logs",
    "nexlify/gui/cyber_gui.py",
    "nexlify/financial/nexlify_tax_reporter.py",
]

FORBIDDEN_README_SNIPPETS = [
    "nexlify_tax_reporting.py",
]

REQUIRED_CYBER_GUI_SNIPPETS = [
    'self.tab_widget.addTab(dashboard, "Dashboard")',
    'self.tab_widget.addTab(trading, "Trading")',
    'self.tab_widget.addTab(portfolio, "Portfolio")',
    'self.tab_widget.addTab(strategies, "Strategies")',
    'self.tab_widget.addTab(settings, "Settings")',
    'self.tab_widget.addTab(logs, "Logs")',
    'Exchange API Configuration (Encrypted Storage)',
]

REQUIRED_INTEGRATION_SNIPPETS = [
    'tabs.addTab(kill_switch_tab, "🚨 Emergency")',
    'tabs.addTab(tax_tab, "💰 Tax Reports")',
    'tabs.addTab(defi_tab, "🌊 DeFi")',
    'tabs.addTab(profit_tab, "💸 Withdrawals")',
]


def missing_paths(paths: Iterable[Path] = REQUIRED_PATHS) -> List[str]:
    """Return required repository paths that are missing."""
    missing: List[str] = []
    for path in paths:
        if not path.exists():
            try:
                display_path = str(path.relative_to(ROOT))
            except ValueError:
                display_path = str(path)
            missing.append(display_path)
    return missing


def missing_module_specs(modules: Iterable[str] = REQUIRED_MODULE_SPECS) -> List[str]:
    """Return module names that cannot be resolved to source paths/specs.

    For local repo modules, this avoids importing package `__init__` files (which may
    pull optional runtime deps) by resolving expected source paths directly.
    """
    missing: List[str] = []

    for module in modules:
        if module.startswith("nexlify."):
            module_rel = Path(*module.split("."))
            module_file = ROOT / f"{module_rel}.py"
            module_pkg_init = ROOT / module_rel / "__init__.py"
            if not module_file.exists() and not module_pkg_init.exists():
                missing.append(module)
            continue

        if importlib.util.find_spec(module) is None:
            missing.append(module)

    return missing


def main() -> int:
    errors: list[str] = []

    for path in missing_paths():
        errors.append(f"Missing required path: {path}")

    for module in missing_module_specs():
        errors.append(f"Missing module import spec: {module}")

    if not README.exists():
        errors.append("Missing README.md")
    else:
        content = README.read_text(encoding="utf-8")

        for snippet in REQUIRED_README_SNIPPETS:
            if snippet not in content:
                errors.append(f"README missing required text: {snippet}")

        for snippet in FORBIDDEN_README_SNIPPETS:
            if snippet in content:
                errors.append(f"README contains stale text: {snippet}")

    if CYBER_GUI.exists():
        cyber_gui_content = CYBER_GUI.read_text(encoding="utf-8")
        for snippet in REQUIRED_CYBER_GUI_SNIPPETS:
            if snippet not in cyber_gui_content:
                errors.append(f"cyber_gui missing required UI snippet: {snippet}")
    else:
        errors.append("Missing nexlify/gui/cyber_gui.py")

    if GUI_INTEGRATION.exists():
        integration_content = GUI_INTEGRATION.read_text(encoding="utf-8")
        for snippet in REQUIRED_INTEGRATION_SNIPPETS:
            if snippet not in integration_content:
                errors.append(f"nexlify_gui_integration missing tab snippet: {snippet}")
    else:
        errors.append("Missing nexlify/gui/nexlify_gui_integration.py")

    if errors:
        print("❌ docs-vs-code verification failed")
        for error in errors:
            print(f" - {error}")
        return 1

    print("✅ docs-vs-code verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
