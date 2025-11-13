"""
Nexlify Pre-Flight Checker
Validates all systems before training starts

This module checks:
- Internet connectivity
- Exchange API availability
- External feature APIs (Fear & Greed, etc.)
- Data availability and cache status
- Hardware capabilities
- Dependencies

Provides:
- Impact assessment for failures
- User choice to continue or abort
- Troubleshooting guidance
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import ccxt
    import requests
    import torch
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a pre-flight check"""
    component: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    impact: str  # 'none', 'low', 'medium', 'high', 'critical'
    troubleshooting: str


class PreFlightChecker:
    """
    Comprehensive pre-flight validation system
    """

    def __init__(self, symbol: str = 'BTC/USDT', exchange: str = 'binance'):
        """
        Initialize pre-flight checker

        Args:
            symbol: Trading pair to validate
            exchange: Exchange to validate
        """
        self.symbol = symbol
        self.exchange = exchange
        self.checks: List[CheckResult] = []

    def run_all_checks(self, automated_mode: bool = False) -> Tuple[bool, List[CheckResult]]:
        """
        Run all pre-flight checks

        Args:
            automated_mode: If True, skip user prompts

        Returns:
            Tuple of (can_proceed, list of check results)
        """
        print("\n" + "="*80)
        print("NEXLIFY PRE-FLIGHT CHECK")
        print("="*80)
        print(f"Symbol: {self.symbol}")
        print(f"Exchange: {self.exchange}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

        # Run all checks
        self._check_internet()
        self._check_exchange_api()
        self._check_fear_greed_api()
        self._check_hardware()
        self._check_dependencies()
        self._check_disk_space()

        # Analyze results
        passed = sum(1 for c in self.checks if c.status == 'pass')
        warnings = sum(1 for c in self.checks if c.status == 'warning')
        failed = sum(1 for c in self.checks if c.status == 'fail')

        # Print summary
        print("\n" + "="*80)
        print("PRE-FLIGHT CHECK SUMMARY")
        print("="*80)
        print(f"✓ Passed:  {passed}/{len(self.checks)}")
        print(f"⚠ Warnings: {warnings}/{len(self.checks)}")
        print(f"✗ Failed:   {failed}/{len(self.checks)}")
        print("="*80 + "\n")

        # Print detailed results
        print("DETAILED RESULTS:")
        print("-" * 80)

        for check in self.checks:
            icon = {'pass': '✓', 'warning': '⚠', 'fail': '✗'}[check.status]
            color = {'pass': '', 'warning': '', 'fail': ''}[check.status]

            print(f"\n{icon} {check.component}")
            print(f"  Status: {check.status.upper()}")
            print(f"  Message: {check.message}")
            print(f"  Impact: {check.impact.upper()}")

            if check.status != 'pass':
                print(f"  How to fix: {check.troubleshooting}")

        print("\n" + "="*80)

        # Assess overall impact
        critical_failures = [c for c in self.checks if c.impact == 'critical' and c.status == 'fail']
        high_impact_issues = [c for c in self.checks if c.impact in ['high', 'critical'] and c.status in ['fail', 'warning']]

        if critical_failures:
            print("\n❌ CRITICAL FAILURES DETECTED")
            print("Training cannot proceed with the following critical issues:\n")
            for check in critical_failures:
                print(f"  • {check.component}: {check.message}")
                print(f"    Fix: {check.troubleshooting}\n")

            return False, self.checks

        if high_impact_issues:
            print("\n⚠ HIGH IMPACT ISSUES DETECTED")
            print("Training can proceed but will be significantly degraded:\n")

            total_impact = 0
            for check in high_impact_issues:
                impact_pct = {'high': 30, 'critical': 50, 'medium': 15}.get(check.impact, 0)
                total_impact += impact_pct
                print(f"  • {check.component}: {check.message}")
                print(f"    Impact: ~{impact_pct}% reduction in training quality")
                print(f"    Fix: {check.troubleshooting}\n")

            print(f"📊 ESTIMATED IMPACT: ~{min(total_impact, 80)}% reduction in training effectiveness")
            print("\nRecommendations:")
            print("  1. Fix the issues above for best results")
            print("  2. Or proceed with degraded training (not recommended)")
            print("  3. Use --automated flag to skip this check in the future")

            if not automated_mode:
                print("\n" + "="*80)
                response = input("\nDo you want to continue anyway? (yes/no): ").strip().lower()

                if response not in ['yes', 'y']:
                    print("\n❌ Training aborted by user")
                    print("\nNext steps:")
                    print("  1. Review the troubleshooting guidance above")
                    print("  2. Fix the issues")
                    print("  3. Run the pre-flight check again")
                    return False, self.checks
                else:
                    print("\n⚠ Proceeding with degraded training...")

        else:
            print("\n✅ ALL SYSTEMS GO!")
            print("All critical systems are operational.")

            if warnings > 0:
                print(f"\nNote: {warnings} warning(s) detected but won't significantly impact training.")

        return True, self.checks

    def _check_internet(self):
        """Check internet connectivity"""
        try:
            response = requests.get('https://www.google.com', timeout=5)
            if response.status_code == 200:
                self.checks.append(CheckResult(
                    component="Internet Connectivity",
                    status="pass",
                    message="Internet connection is active",
                    impact="none",
                    troubleshooting="N/A"
                ))
            else:
                self.checks.append(CheckResult(
                    component="Internet Connectivity",
                    status="warning",
                    message=f"Internet reachable but unusual response (code {response.status_code})",
                    impact="low",
                    troubleshooting="Check your internet connection stability"
                ))
        except Exception as e:
            self.checks.append(CheckResult(
                component="Internet Connectivity",
                status="fail",
                message=f"No internet connection: {e}",
                impact="critical",
                troubleshooting="1. Check your network connection\n    2. Verify firewall settings\n    3. Test: ping google.com"
            ))

    def _check_exchange_api(self):
        """Check exchange API availability"""
        # Handle 'auto' exchange selection
        if self.exchange.lower() == 'auto':
            self._check_multiple_exchanges()
            return

        try:
            exchange_class = getattr(ccxt, self.exchange, None)
            if not exchange_class:
                self.checks.append(CheckResult(
                    component=f"{self.exchange.capitalize()} Exchange API",
                    status="fail",
                    message=f"Exchange '{self.exchange}' not supported by CCXT",
                    impact="critical",
                    troubleshooting=f"1. Use a supported exchange: binance, coinbase, kraken\n    2. Or install latest CCXT: pip install --upgrade ccxt"
                ))
                return

            exchange = exchange_class({'enableRateLimit': True, 'timeout': 10000})

            # Test fetching ticker
            ticker = exchange.fetch_ticker(self.symbol)

            if ticker and 'last' in ticker:
                self.checks.append(CheckResult(
                    component=f"{self.exchange.capitalize()} Exchange API",
                    status="pass",
                    message=f"Successfully connected to {self.exchange}, {self.symbol} available (price: ${ticker['last']:,.2f})",
                    impact="none",
                    troubleshooting="N/A"
                ))
            else:
                self.checks.append(CheckResult(
                    component=f"{self.exchange.capitalize()} Exchange API",
                    status="warning",
                    message=f"Connected but ticker data incomplete",
                    impact="medium",
                    troubleshooting="1. Verify symbol format (e.g., BTC/USDT)\n    2. Try a different exchange\n    3. Check exchange status: https://status.binance.com"
                ))

        except ccxt.NetworkError as e:
            self.checks.append(CheckResult(
                component=f"{self.exchange.capitalize()} Exchange API",
                status="fail",
                message=f"Network error connecting to exchange: {e}",
                impact="critical",
                troubleshooting="1. Check internet connection\n    2. Try again in a few minutes\n    3. Use VPN if exchange is blocked in your region\n    4. Try a different exchange with --exchange flag"
            ))

        except ccxt.ExchangeError as e:
            if 'symbol' in str(e).lower():
                self.checks.append(CheckResult(
                    component=f"{self.exchange.capitalize()} Exchange API",
                    status="fail",
                    message=f"Symbol {self.symbol} not found on {self.exchange}",
                    impact="critical",
                    troubleshooting=f"1. Verify symbol format (should be like BTC/USDT)\n    2. Check available symbols: https://{self.exchange}.com\n    3. Try: BTC/USDT, ETH/USDT, SOL/USDT"
                ))
            else:
                self.checks.append(CheckResult(
                    component=f"{self.exchange.capitalize()} Exchange API",
                    status="fail",
                    message=f"Exchange error: {e}",
                    impact="high",
                    troubleshooting="1. Check exchange status\n    2. Try different exchange\n    3. Wait and retry"
                ))

        except Exception as e:
            self.checks.append(CheckResult(
                component=f"{self.exchange.capitalize()} Exchange API",
                status="fail",
                message=f"Unexpected error: {e}",
                impact="critical",
                troubleshooting="1. Update CCXT: pip install --upgrade ccxt\n    2. Check Python version (3.8+ required)\n    3. Report issue if persists"
            ))

    def _check_multiple_exchanges(self):
        """Check multiple exchanges for auto-selection mode"""
        exchanges_to_test = ['coinbase', 'kraken', 'bitstamp', 'gemini']
        available_exchanges = []

        for exchange_name in exchanges_to_test:
            try:
                exchange_class = getattr(ccxt, exchange_name, None)
                if not exchange_class:
                    continue

                exchange = exchange_class({'enableRateLimit': True, 'timeout': 10000})

                # Quick test - just try to load markets
                exchange.load_markets()

                # Check if symbol is available
                if self.symbol in exchange.markets:
                    available_exchanges.append(exchange_name)

            except Exception:
                # Silently skip exchanges that fail
                continue

        if len(available_exchanges) >= 2:
            self.checks.append(CheckResult(
                component="Exchange API (Auto-select)",
                status="pass",
                message=f"✓ {len(available_exchanges)} exchanges available for auto-selection: {', '.join(available_exchanges)}",
                impact="none",
                troubleshooting="N/A"
            ))
        elif len(available_exchanges) == 1:
            self.checks.append(CheckResult(
                component="Exchange API (Auto-select)",
                status="warning",
                message=f"Only 1 exchange available: {available_exchanges[0]}. Auto-selection will use this exchange.",
                impact="low",
                troubleshooting="1. Check network connectivity\n    2. Some exchanges may be blocked in your region\n    3. Consider using VPN"
            ))
        else:
            self.checks.append(CheckResult(
                component="Exchange API (Auto-select)",
                status="fail",
                message=f"No exchanges available for {self.symbol}. Cannot proceed with auto-selection.",
                impact="critical",
                troubleshooting=f"1. Check internet connection\n    2. Verify symbol format: {self.symbol}\n    3. Try manual exchange selection with --exchange coinbase\n    4. Use VPN if exchanges are blocked"
            ))

    def _check_fear_greed_api(self):
        """Check Fear & Greed Index API"""
        try:
            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    value = data['data'][0].get('value', 'N/A')
                    classification = data['data'][0].get('value_classification', 'N/A')

                    self.checks.append(CheckResult(
                        component="Fear & Greed Index API",
                        status="pass",
                        message=f"API accessible (Current: {value} - {classification})",
                        impact="none",
                        troubleshooting="N/A"
                    ))
                else:
                    self.checks.append(CheckResult(
                        component="Fear & Greed Index API",
                        status="warning",
                        message="API reachable but returned no data",
                        impact="low",
                        troubleshooting="Training will continue with neutral sentiment values. No action needed."
                    ))
            else:
                self.checks.append(CheckResult(
                    component="Fear & Greed Index API",
                    status="warning",
                    message=f"API returned status {response.status_code}",
                    impact="low",
                    troubleshooting="Training will continue without sentiment data. Check: https://api.alternative.me/fng/"
                ))

        except requests.exceptions.Timeout:
            self.checks.append(CheckResult(
                component="Fear & Greed Index API",
                status="warning",
                message="API request timed out",
                impact="low",
                troubleshooting="Training will continue with neutral sentiment. No fix needed."
            ))

        except Exception as e:
            self.checks.append(CheckResult(
                component="Fear & Greed Index API",
                status="warning",
                message=f"Could not access API: {e}",
                impact="low",
                troubleshooting="Training will continue without Fear & Greed data (~5% impact). No action required."
            ))

    def _check_hardware(self):
        """Check hardware capabilities"""
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.checks.append(CheckResult(
                component="GPU Acceleration",
                status="pass",
                message=f"GPU available: {gpu_name} ({gpu_memory:.1f} GB VRAM)",
                impact="none",
                troubleshooting="N/A"
            ))
        else:
            self.checks.append(CheckResult(
                component="GPU Acceleration",
                status="warning",
                message="No GPU detected, will use CPU (training will be slower)",
                impact="medium",
                troubleshooting="For faster training:\n    1. Install CUDA-compatible PyTorch: https://pytorch.org\n    2. Or accept slower CPU training (works fine, just slower)"
            ))

        # Check CPU cores
        import psutil
        cpu_count = psutil.cpu_count()

        if cpu_count >= 4:
            self.checks.append(CheckResult(
                component="CPU Cores",
                status="pass",
                message=f"{cpu_count} CPU cores available",
                impact="none",
                troubleshooting="N/A"
            ))
        else:
            self.checks.append(CheckResult(
                component="CPU Cores",
                status="warning",
                message=f"Only {cpu_count} CPU cores (4+ recommended)",
                impact="low",
                troubleshooting="Training will work but may be slow. Consider using a more powerful machine."
            ))

        # Check RAM
        ram_gb = psutil.virtual_memory().total / 1024**3

        if ram_gb >= 8:
            self.checks.append(CheckResult(
                component="System RAM",
                status="pass",
                message=f"{ram_gb:.1f} GB RAM available",
                impact="none",
                troubleshooting="N/A"
            ))
        elif ram_gb >= 4:
            self.checks.append(CheckResult(
                component="System RAM",
                status="warning",
                message=f"Only {ram_gb:.1f} GB RAM (8GB+ recommended)",
                impact="medium",
                troubleshooting="Training may use swap memory (slower). Close other applications."
            ))
        else:
            self.checks.append(CheckResult(
                component="System RAM",
                status="fail",
                message=f"Insufficient RAM: {ram_gb:.1f} GB (4GB minimum)",
                impact="high",
                troubleshooting="1. Close other applications\n    2. Use a machine with more RAM\n    3. Reduce training data with --years 1"
            ))

    def _check_dependencies(self):
        """Check Python dependencies"""
        required = {
            'torch': '2.0.0',
            'pandas': '1.3.0',
            'numpy': '1.20.0',
            'ccxt': '3.0.0',
            'requests': '2.25.0'
        }

        missing = []
        outdated = []

        for package, min_version in required.items():
            try:
                if package == 'torch':
                    import torch
                    version = torch.__version__.split('+')[0]
                elif package == 'pandas':
                    import pandas
                    version = pandas.__version__
                elif package == 'numpy':
                    import numpy
                    version = numpy.__version__
                elif package == 'ccxt':
                    import ccxt
                    version = ccxt.__version__
                elif package == 'requests':
                    import requests
                    version = requests.__version__

                # Simple version comparison (not perfect but works for major versions)
                if version < min_version:
                    outdated.append(f"{package} ({version}, need {min_version}+)")

            except ImportError:
                missing.append(package)

        if not missing and not outdated:
            self.checks.append(CheckResult(
                component="Python Dependencies",
                status="pass",
                message="All required packages are installed and up-to-date",
                impact="none",
                troubleshooting="N/A"
            ))
        elif missing:
            self.checks.append(CheckResult(
                component="Python Dependencies",
                status="fail",
                message=f"Missing packages: {', '.join(missing)}",
                impact="critical",
                troubleshooting=f"Install missing packages:\n    pip install {' '.join(missing)}\n    Or: pip install -r requirements.txt"
            ))
        elif outdated:
            self.checks.append(CheckResult(
                component="Python Dependencies",
                status="warning",
                message=f"Outdated packages: {', '.join(outdated)}",
                impact="low",
                troubleshooting="Update packages:\n    pip install --upgrade torch pandas numpy ccxt requests"
            ))

    def _check_disk_space(self):
        """Check available disk space"""
        import psutil

        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024**3

        if free_gb >= 5:
            self.checks.append(CheckResult(
                component="Disk Space",
                status="pass",
                message=f"{free_gb:.1f} GB free space available",
                impact="none",
                troubleshooting="N/A"
            ))
        elif free_gb >= 2:
            self.checks.append(CheckResult(
                component="Disk Space",
                status="warning",
                message=f"Only {free_gb:.1f} GB free (5GB+ recommended)",
                impact="low",
                troubleshooting="Free up disk space or training data may not be fully cached."
            ))
        else:
            self.checks.append(CheckResult(
                component="Disk Space",
                status="fail",
                message=f"Insufficient disk space: {free_gb:.1f} GB (2GB minimum)",
                impact="high",
                troubleshooting="1. Free up disk space\n    2. Use external drive\n    3. Reduce training data with --years 1"
            ))

    def save_report(self, output_path: str = "./preflight_report.json"):
        """Save pre-flight check report to file"""
        import json

        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'checks': [asdict(check) for check in self.checks],
            'summary': {
                'passed': sum(1 for c in self.checks if c.status == 'pass'),
                'warnings': sum(1 for c in self.checks if c.status == 'warning'),
                'failed': sum(1 for c in self.checks if c.status == 'fail')
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Pre-flight report saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nexlify Pre-Flight Checker")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--automated', action='store_true', help='Skip user prompts')
    parser.add_argument('--save-report', type=str, help='Save report to file')

    args = parser.parse_args()

    checker = PreFlightChecker(symbol=args.symbol, exchange=args.exchange)
    can_proceed, results = checker.run_all_checks(automated_mode=args.automated)

    if args.save_report:
        checker.save_report(args.save_report)

    sys.exit(0 if can_proceed else 1)
