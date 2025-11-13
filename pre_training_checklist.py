#!/usr/bin/env python3
"""
Nexlify Pre-Training Checklist
===============================
Comprehensive validation script to run before training or launching the app.
Checks all critical components and provides actionable recommendations.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_check(name, status, message=""):
    """Print a check result"""
    if status == "PASS":
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        color = Colors.GREEN
    elif status == "WARN":
        symbol = f"{Colors.YELLOW}⚠{Colors.END}"
        color = Colors.YELLOW
    else:  # FAIL
        symbol = f"{Colors.RED}✗{Colors.END}"
        color = Colors.RED

    print(f"{symbol} {Colors.BOLD}{name}{Colors.END}: {color}{status}{Colors.END}")
    if message:
        print(f"  → {message}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_check("Python Version", "PASS", f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_check("Python Version", "FAIL", f"Python {version.major}.{version.minor}.{version.micro} (3.9+ required)")
        return False

def check_dependencies():
    """Check if critical dependencies are installed"""
    critical_packages = [
        'ccxt', 'pandas', 'numpy', 'tensorflow', 'torch',
        'scikit-learn', 'xgboost', 'sqlalchemy', 'aiohttp'
    ]

    missing = []
    for package in critical_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if not missing:
        print_check("Python Dependencies", "PASS", "All critical packages installed")
        return True
    else:
        print_check("Python Dependencies", "FAIL", f"Missing: {', '.join(missing)}")
        print(f"  → Run: pip install -r requirements.txt")
        return False

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'config', 'data', 'models', 'logs', 'nexlify',
        'data/historical_cache', 'data/sample_datasets'
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)

    if not missing:
        print_check("Directory Structure", "PASS", "All required directories exist")
        return True
    else:
        print_check("Directory Structure", "WARN", f"Missing: {', '.join(missing)}")
        print(f"  → Will be auto-created on first run")
        return None  # Warning, not critical

def check_sample_data():
    """Check if sample training data exists"""
    sample_files = [
        'data/sample_datasets/btc_usdt_raw.csv',
        'data/sample_datasets/btc_usdt_quick_test.csv'
    ]

    existing = [f for f in sample_files if Path(f).exists()]

    if existing:
        print_check("Sample Training Data", "PASS", f"{len(existing)} sample dataset(s) found")
        return True
    else:
        print_check("Sample Training Data", "WARN", "No sample data found")
        print(f"  → Training will fetch data from exchanges (requires internet)")
        return None  # Warning, not critical

def check_config_files():
    """Check if configuration files exist"""
    if Path('config/neural_config.json').exists():
        print_check("Configuration Files", "PASS", "neural_config.json found")
        return True
    elif Path('config/neural_config.example.json').exists():
        print_check("Configuration Files", "WARN", "Using example config")
        print(f"  → Copy config/neural_config.example.json to config/neural_config.json")
        return None  # Warning, not critical
    else:
        print_check("Configuration Files", "FAIL", "No config files found")
        print(f"  → Run scripts/setup_nexlify.py first")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import psutil
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)

        if free_gb >= 5:
            print_check("Disk Space", "PASS", f"{free_gb:.1f} GB available")
            return True
        elif free_gb >= 2:
            print_check("Disk Space", "WARN", f"{free_gb:.1f} GB available (5GB+ recommended)")
            return None
        else:
            print_check("Disk Space", "FAIL", f"{free_gb:.1f} GB available (2GB minimum)")
            return False
    except ImportError:
        print_check("Disk Space", "WARN", "Could not check (psutil not installed)")
        return None

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_check("GPU Availability", "PASS", f"GPU found: {gpu_name}")
            return True
        else:
            print_check("GPU Availability", "WARN", "No GPU found (will use CPU)")
            print(f"  → Training will be slower on CPU")
            return None
    except ImportError:
        print_check("GPU Availability", "WARN", "Could not check (torch not installed)")
        return None

def check_internet_connectivity():
    """Check internet connectivity"""
    try:
        import requests
        response = requests.get('https://google.com', timeout=5)
        if response.status_code == 200:
            print_check("Internet Connectivity", "PASS", "Connected")
            return True
        else:
            print_check("Internet Connectivity", "WARN", "Limited connectivity")
            return None
    except:
        print_check("Internet Connectivity", "WARN", "No internet connection")
        print(f"  → Can still train with cached/sample data")
        return None

def check_exchange_api():
    """Check if exchange API is accessible"""
    try:
        import ccxt
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        print_check("Exchange API", "PASS", f"Binance accessible (BTC: ${ticker['last']:.2f})")
        return True
    except Exception as e:
        print_check("Exchange API", "WARN", "Cannot access exchange")
        print(f"  → {str(e)[:80]}")
        print(f"  → Can still train with cached/sample data")
        return None

def check_database():
    """Check database connectivity"""
    try:
        from sqlalchemy import create_engine
        from pathlib import Path

        db_path = Path('data/trading.db')
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(f'sqlite:///{db_path}')
        with engine.connect() as conn:
            pass

        print_check("Database", "PASS", "SQLite database accessible")
        return True
    except Exception as e:
        print_check("Database", "FAIL", f"Database error: {str(e)[:80]}")
        return False

def main():
    """Run all pre-training checks"""
    print_header("NEXLIFY PRE-TRAINING CHECKLIST")

    results = {
        'Python Version': check_python_version(),
        'Python Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Sample Data': check_sample_data(),
        'Config Files': check_config_files(),
        'Disk Space': check_disk_space(),
        'GPU': check_gpu_availability(),
        'Internet': check_internet_connectivity(),
        'Exchange API': check_exchange_api(),
        'Database': check_database(),
    }

    # Summary
    print_header("SUMMARY")

    critical_pass = all(v is not False for k, v in results.items() if k in ['Python Version', 'Python Dependencies', 'Database'])
    warnings = sum(1 for v in results.values() if v is None)
    failures = sum(1 for v in results.values() if v is False)

    if critical_pass and failures == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.END}")
        print(f"\n{Colors.GREEN}You're ready to start training or run the app.{Colors.END}")
        if warnings > 0:
            print(f"{Colors.YELLOW}Note: {warnings} warning(s) - review above for details{Colors.END}")
        return 0
    elif critical_pass:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Some checks failed (non-critical){Colors.END}")
        print(f"\n{Colors.YELLOW}You can proceed, but review the issues above.{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Critical checks failed{Colors.END}")
        print(f"\n{Colors.RED}Please fix the critical issues before proceeding.{Colors.END}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
