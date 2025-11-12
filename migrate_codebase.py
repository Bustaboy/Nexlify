#!/usr/bin/env python3
"""
Automated migration script for Nexlify reorganization.
Safely moves files and updates imports to new package structure.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

# File mapping: source -> destination (in dependency order)
FILE_MAPPING = {
    # Phase 1: Utils (move first - everything depends on these)
    'error_handler.py': 'nexlify/utils/error_handler.py',
    'utils_module.py': 'nexlify/utils/utils_module.py',

    # Phase 2: Risk (Level 0 - no internal dependencies except error_handler)
    'nexlify_risk_manager.py': 'nexlify/risk/nexlify_risk_manager.py',
    'nexlify_circuit_breaker.py': 'nexlify/risk/nexlify_circuit_breaker.py',
    'nexlify_flash_crash_protection.py': 'nexlify/risk/nexlify_flash_crash_protection.py',
    'nexlify_emergency_kill_switch.py': 'nexlify/risk/nexlify_emergency_kill_switch.py',

    # Phase 2: Analytics (Level 0)
    'nexlify_performance_tracker.py': 'nexlify/analytics/nexlify_performance_tracker.py',
    'nexlify_advanced_analytics.py': 'nexlify/analytics/nexlify_advanced_analytics.py',
    'nexlify_ai_companion.py': 'nexlify/analytics/nexlify_ai_companion.py',

    # Phase 2: Strategies (Level 0)
    'nexlify_multi_strategy.py': 'nexlify/strategies/nexlify_multi_strategy.py',
    'nexlify_multi_timeframe.py': 'nexlify/strategies/nexlify_multi_timeframe.py',
    'nexlify_predictive_features.py': 'nexlify/strategies/nexlify_predictive_features.py',
    'nexlify_rl_agent.py': 'nexlify/strategies/nexlify_rl_agent.py',

    # Phase 2: Security (Level 0 - except security_suite)
    'nexlify_advanced_security.py': 'nexlify/security/nexlify_advanced_security.py',
    'nexlify_pin_manager.py': 'nexlify/security/nexlify_pin_manager.py',
    'nexlify_integrity_monitor.py': 'nexlify/security/nexlify_integrity_monitor.py',
    'nexlify_audit_trail.py': 'nexlify/security/nexlify_audit_trail.py',

    # Phase 2: Financial (Level 0)
    'nexlify_profit_manager.py': 'nexlify/financial/nexlify_profit_manager.py',
    'nexlify_tax_reporter.py': 'nexlify/financial/nexlify_tax_reporter.py',
    'nexlify_portfolio_rebalancer.py': 'nexlify/financial/nexlify_portfolio_rebalancer.py',
    'nexlify_defi_integration.py': 'nexlify/financial/nexlify_defi_integration.py',

    # Phase 2: Integrations (Level 0)
    'nexlify_websocket_feeds.py': 'nexlify/integrations/nexlify_websocket_feeds.py',
    'nexlify_telegram_bot.py': 'nexlify/integrations/nexlify_telegram_bot.py',

    # Phase 2: Backtesting (Level 0)
    'nexlify_backtester.py': 'nexlify/backtesting/nexlify_backtester.py',
    'nexlify_paper_trading.py': 'nexlify/backtesting/nexlify_paper_trading.py',
    'backtest_phase1_phase2_integration.py': 'nexlify/backtesting/backtest_phase1_phase2_integration.py',

    # Phase 2: GUI (Level 0)
    'nexlify_cyberpunk_effects.py': 'nexlify/gui/nexlify_cyberpunk_effects.py',
    'nexlify_hardware_detection.py': 'nexlify/gui/nexlify_hardware_detection.py',

    # Phase 2: Core (Level 0 - auto_trader)
    'nexlify_auto_trader.py': 'nexlify/core/nexlify_auto_trader.py',

    # Phase 3: Level 1 files (depend on Level 0)
    'arasaka_neural_net.py': 'nexlify/core/arasaka_neural_net.py',
    'nexlify_security_suite.py': 'nexlify/security/nexlify_security_suite.py',

    # Phase 4: Level 2 files
    'nexlify_neural_net.py': 'nexlify/core/nexlify_neural_net.py',
    'nexlify_gui_integration.py': 'nexlify/gui/nexlify_gui_integration.py',
    'nexlify_trading_integration.py': 'nexlify/core/nexlify_trading_integration.py',

    # Phase 5: Level 3 files (depend on all previous)
    'cyber_gui.py': 'nexlify/gui/cyber_gui.py',

    # Phase 6: Scripts (last - may import any module)
    'nexlify_launcher.py': 'scripts/nexlify_launcher.py',
    'setup_nexlify.py': 'scripts/setup_nexlify.py',
    'train_rl_agent.py': 'scripts/train_rl_agent.py',
    'example_integration.py': 'scripts/example_integration.py',
}

# Import replacement patterns
IMPORT_REPLACEMENTS = {
    'from error_handler import': 'from nexlify.utils.error_handler import',
    'from utils_module import': 'from nexlify.utils.utils_module import',
    'import error_handler': 'import nexlify.utils.error_handler',
    'import utils_module': 'import nexlify.utils.utils_module',

    # Risk
    'from nexlify_risk_manager import': 'from nexlify.risk.nexlify_risk_manager import',
    'from nexlify_circuit_breaker import': 'from nexlify.risk.nexlify_circuit_breaker import',
    'from nexlify_flash_crash_protection import': 'from nexlify.risk.nexlify_flash_crash_protection import',
    'from nexlify_emergency_kill_switch import': 'from nexlify.risk.nexlify_emergency_kill_switch import',

    # Analytics
    'from nexlify_performance_tracker import': 'from nexlify.analytics.nexlify_performance_tracker import',
    'from nexlify_advanced_analytics import': 'from nexlify.analytics.nexlify_advanced_analytics import',
    'from nexlify_ai_companion import': 'from nexlify.analytics.nexlify_ai_companion import',

    # Strategies
    'from nexlify_multi_strategy import': 'from nexlify.strategies.nexlify_multi_strategy import',
    'from nexlify_multi_timeframe import': 'from nexlify.strategies.nexlify_multi_timeframe import',
    'from nexlify_predictive_features import': 'from nexlify.strategies.nexlify_predictive_features import',
    'from nexlify_rl_agent import': 'from nexlify.strategies.nexlify_rl_agent import',

    # Security
    'from nexlify_advanced_security import': 'from nexlify.security.nexlify_advanced_security import',
    'from nexlify_pin_manager import': 'from nexlify.security.nexlify_pin_manager import',
    'from nexlify_integrity_monitor import': 'from nexlify.security.nexlify_integrity_monitor import',
    'from nexlify_audit_trail import': 'from nexlify.security.nexlify_audit_trail import',
    'from nexlify_security_suite import': 'from nexlify.security.nexlify_security_suite import',

    # Financial
    'from nexlify_profit_manager import': 'from nexlify.financial.nexlify_profit_manager import',
    'from nexlify_tax_reporter import': 'from nexlify.financial.nexlify_tax_reporter import',
    'from nexlify_portfolio_rebalancer import': 'from nexlify.financial.nexlify_portfolio_rebalancer import',
    'from nexlify_defi_integration import': 'from nexlify.financial.nexlify_defi_integration import',

    # Integrations
    'from nexlify_websocket_feeds import': 'from nexlify.integrations.nexlify_websocket_feeds import',
    'from nexlify_telegram_bot import': 'from nexlify.integrations.nexlify_telegram_bot import',

    # Backtesting
    'from nexlify_backtester import': 'from nexlify.backtesting.nexlify_backtester import',
    'from nexlify_paper_trading import': 'from nexlify.backtesting.nexlify_paper_trading import',
    'from backtest_phase1_phase2_integration import': 'from nexlify.backtesting.backtest_phase1_phase2_integration import',

    # GUI
    'from nexlify_cyberpunk_effects import': 'from nexlify.gui.nexlify_cyberpunk_effects import',
    'from nexlify_hardware_detection import': 'from nexlify.gui.nexlify_hardware_detection import',
    'from nexlify_gui_integration import': 'from nexlify.gui.nexlify_gui_integration import',
    'from cyber_gui import': 'from nexlify.gui.cyber_gui import',

    # Core
    'from nexlify_auto_trader import': 'from nexlify.core.nexlify_auto_trader import',
    'from arasaka_neural_net import': 'from nexlify.core.arasaka_neural_net import',
    'from nexlify_neural_net import': 'from nexlify.core.nexlify_neural_net import',
    'from nexlify_trading_integration import': 'from nexlify.core.nexlify_trading_integration import',
}

# __init__.py content for each package
INIT_FILES = {
    'nexlify/__init__.py': '''"""
Nexlify - AI-Powered Cryptocurrency Trading Platform
"""
__version__ = "2.0.7.7"

# Re-export main components for easy access
from nexlify.core import (
    ArasakaNeuralNet,
    NexlifyNeuralNet,
    AutoTrader,
    TradingIntegrationManager
)

from nexlify.risk import (
    RiskManager,
    CircuitBreaker,
    FlashCrashProtection,
    EmergencyKillSwitch
)

from nexlify.security import SecuritySuite

__all__ = [
    'ArasakaNeuralNet',
    'NexlifyNeuralNet',
    'AutoTrader',
    'TradingIntegrationManager',
    'RiskManager',
    'CircuitBreaker',
    'FlashCrashProtection',
    'EmergencyKillSwitch',
    'SecuritySuite',
]
''',

    'nexlify/core/__init__.py': '''"""Core trading and neural network components."""

from nexlify.core.arasaka_neural_net import ArasakaNeuralNet
from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.core.nexlify_auto_trader import AutoTrader
from nexlify.core.nexlify_trading_integration import TradingIntegrationManager

__all__ = [
    'ArasakaNeuralNet',
    'NexlifyNeuralNet',
    'AutoTrader',
    'TradingIntegrationManager',
]
''',

    'nexlify/strategies/__init__.py': '''"""Trading strategies and ML models."""

from nexlify.strategies.nexlify_multi_strategy import MultiStrategy
from nexlify.strategies.nexlify_multi_timeframe import MultiTimeframe
from nexlify.strategies.nexlify_predictive_features import PredictiveFeatures
from nexlify.strategies.nexlify_rl_agent import RLAgent

__all__ = [
    'MultiStrategy',
    'MultiTimeframe',
    'PredictiveFeatures',
    'RLAgent',
]
''',

    'nexlify/risk/__init__.py': '''"""Risk management and protection systems."""

from nexlify.risk.nexlify_risk_manager import RiskManager
from nexlify.risk.nexlify_circuit_breaker import CircuitBreaker
from nexlify.risk.nexlify_flash_crash_protection import (
    FlashCrashProtection,
    CrashSeverity
)
from nexlify.risk.nexlify_emergency_kill_switch import (
    EmergencyKillSwitch,
    KillSwitchTrigger
)

__all__ = [
    'RiskManager',
    'CircuitBreaker',
    'FlashCrashProtection',
    'CrashSeverity',
    'EmergencyKillSwitch',
    'KillSwitchTrigger',
]
''',

    'nexlify/security/__init__.py': '''"""Security, authentication, and audit systems."""

from nexlify.security.nexlify_advanced_security import AdvancedSecurity
from nexlify.security.nexlify_pin_manager import PINManager
from nexlify.security.nexlify_integrity_monitor import IntegrityMonitor
from nexlify.security.nexlify_audit_trail import AuditTrail
from nexlify.security.nexlify_security_suite import SecuritySuite

__all__ = [
    'AdvancedSecurity',
    'PINManager',
    'IntegrityMonitor',
    'AuditTrail',
    'SecuritySuite',
]
''',

    'nexlify/financial/__init__.py': '''"""Financial management, tax reporting, and DeFi integration."""

from nexlify.financial.nexlify_profit_manager import (
    ProfitManager,
    WithdrawalDestination
)
from nexlify.financial.nexlify_tax_reporter import TaxReporter
from nexlify.financial.nexlify_portfolio_rebalancer import PortfolioRebalancer
from nexlify.financial.nexlify_defi_integration import DeFiIntegration

__all__ = [
    'ProfitManager',
    'WithdrawalDestination',
    'TaxReporter',
    'PortfolioRebalancer',
    'DeFiIntegration',
]
''',

    'nexlify/analytics/__init__.py': '''"""Performance tracking and analytics."""

from nexlify.analytics.nexlify_performance_tracker import PerformanceTracker
from nexlify.analytics.nexlify_advanced_analytics import AdvancedAnalytics
from nexlify.analytics.nexlify_ai_companion import AICompanion

__all__ = [
    'PerformanceTracker',
    'AdvancedAnalytics',
    'AICompanion',
]
''',

    'nexlify/backtesting/__init__.py': '''"""Backtesting and paper trading systems."""

from nexlify.backtesting.nexlify_backtester import Backtester
from nexlify.backtesting.nexlify_paper_trading import PaperTrading
from nexlify.backtesting.backtest_phase1_phase2_integration import (
    BacktestIntegration
)

__all__ = [
    'Backtester',
    'PaperTrading',
    'BacktestIntegration',
]
''',

    'nexlify/integrations/__init__.py': '''"""External service integrations."""

from nexlify.integrations.nexlify_websocket_feeds import WebSocketFeeds
from nexlify.integrations.nexlify_telegram_bot import TelegramBot

__all__ = [
    'WebSocketFeeds',
    'TelegramBot',
]
''',

    'nexlify/gui/__init__.py': '''"""GUI components and cyberpunk effects."""

from nexlify.gui.cyber_gui import CyberGUI
from nexlify.gui.nexlify_gui_integration import GUIIntegration
from nexlify.gui.nexlify_cyberpunk_effects import CyberpunkEffects
from nexlify.gui.nexlify_hardware_detection import HardwareDetection

__all__ = [
    'CyberGUI',
    'GUIIntegration',
    'CyberpunkEffects',
    'HardwareDetection',
]
''',

    'nexlify/utils/__init__.py': '''"""Utility modules and error handling."""

from nexlify.utils.error_handler import get_error_handler, handle_errors

__all__ = [
    'get_error_handler',
    'handle_errors',
]
''',
}


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def create_directories():
    """Create all required directories."""
    directories = [
        'nexlify/core',
        'nexlify/strategies',
        'nexlify/risk',
        'nexlify/security',
        'nexlify/financial',
        'nexlify/analytics',
        'nexlify/backtesting',
        'nexlify/integrations',
        'nexlify/gui',
        'nexlify/utils',
        'scripts',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")


def update_imports(content: str) -> str:
    """Update all import statements in the content."""
    for old_import, new_import in IMPORT_REPLACEMENTS.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
    return content


def move_file(source: str, destination: str, dry_run: bool = False) -> bool:
    """Move a file and update its imports."""
    if not os.path.exists(source):
        print_warning(f"Source file not found: {source}")
        return False

    try:
        # Read the file content
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update imports
        updated_content = update_imports(content)

        if dry_run:
            print(f"  [DRY RUN] Would move: {source} -> {destination}")
            if updated_content != content:
                print(f"  [DRY RUN] Would update imports in: {destination}")
            return True

        # Ensure destination directory exists
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        # Write to new location
        with open(destination, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print_success(f"Moved: {source} -> {destination}")

        if updated_content != content:
            print(f"  → Updated imports")

        return True

    except Exception as e:
        print_error(f"Failed to move {source}: {str(e)}")
        return False


def create_init_files():
    """Create __init__.py files for all packages."""
    for init_file, content in INIT_FILES.items():
        try:
            # Ensure directory exists
            Path(init_file).parent.mkdir(parents=True, exist_ok=True)

            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print_success(f"Created: {init_file}")
        except Exception as e:
            print_error(f"Failed to create {init_file}: {str(e)}")


def verify_migration():
    """Verify that all expected files exist in new locations."""
    print_header("VERIFICATION")

    missing_files = []
    for destination in FILE_MAPPING.values():
        if not os.path.exists(destination):
            missing_files.append(destination)
            print_error(f"Missing: {destination}")

    if not missing_files:
        print_success("All files migrated successfully!")
        return True
    else:
        print_warning(f"{len(missing_files)} files missing")
        return False


def cleanup_old_files(dry_run: bool = False):
    """Remove old files from root directory."""
    print_header("CLEANUP")

    for source in FILE_MAPPING.keys():
        if os.path.exists(source):
            if dry_run:
                print(f"  [DRY RUN] Would remove: {source}")
            else:
                try:
                    os.remove(source)
                    print_success(f"Removed: {source}")
                except Exception as e:
                    print_error(f"Failed to remove {source}: {str(e)}")


def main():
    """Run the migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Migrate Nexlify codebase to new package structure'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not remove old files after migration'
    )

    args = parser.parse_args()

    print_header("NEXLIFY CODEBASE REORGANIZATION")

    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be modified")

    # Step 1: Create directory structure
    print_header("STEP 1: Creating directory structure")
    if not args.dry_run:
        create_directories()
    else:
        print("[DRY RUN] Would create directory structure")

    # Step 2: Move and update files
    print_header("STEP 2: Moving and updating files")
    success_count = 0
    fail_count = 0

    for source, destination in FILE_MAPPING.items():
        if move_file(source, destination, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n  Total: {success_count} successful, {fail_count} failed")

    # Step 3: Create __init__.py files
    print_header("STEP 3: Creating __init__.py files")
    if not args.dry_run:
        create_init_files()
    else:
        print("[DRY RUN] Would create __init__.py files")

    # Step 4: Verify migration
    if not args.dry_run:
        verification_ok = verify_migration()
    else:
        verification_ok = True
        print("[DRY RUN] Skipping verification")

    # Step 5: Cleanup (optional)
    if not args.no_cleanup and verification_ok:
        cleanup_old_files(dry_run=args.dry_run)
    elif args.no_cleanup:
        print_header("CLEANUP SKIPPED")
        print_warning("Old files remain in root directory (use --no-cleanup=False to remove)")

    # Final summary
    print_header("MIGRATION COMPLETE!" if not args.dry_run else "DRY RUN COMPLETE!")

    if not args.dry_run:
        print("Next steps:")
        print("1. Run tests to verify everything works: pytest tests/ -v")
        print("2. Check imports: python3 -c 'from nexlify.core import AutoTrader'")
        print("3. Update external documentation and scripts")
        print("4. Commit changes: git add . && git commit -m 'Reorganize codebase'")
    else:
        print("Run without --dry-run to perform actual migration")


if __name__ == "__main__":
    main()
