"""
Nexlify Setup Script - Complete System Installation and Configuration
Handles database initialization, dependency installation, and system validation
"""

import os
import sys
import json
import shutil
import platform
import subprocess
import sqlite3
import socket
import time
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import psutil
import pkg_resources

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NexlifySetup:
    """Enhanced setup script with comprehensive system validation and initialization"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.config = {}
        self.errors = []
        self.warnings = []
        
        # System requirements
        self.requirements = {
            'python_version': '3.11',
            'min_ram_gb': 8,
            'recommended_ram_gb': 16,
            'min_disk_gb': 20,
            'recommended_disk_gb': 50,
            'gpu_models': [
                'GTX 1060', 'GTX 1070', 'GTX 1080',
                'RTX 2060', 'RTX 2070', 'RTX 2080',
                'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090',
                'RTX 4060', 'RTX 4070', 'RTX 4080', 'RTX 4090'
            ]
        }
        
        # Required ports
        self.required_ports = {
            8000: 'Nexlify API Server',
            8001: 'Mobile API Server',
            5432: 'PostgreSQL Database',
            6379: 'Redis Cache',
            8888: 'Jupyter Notebook (Optional)'
        }
        
        # Color codes for terminal output
        self.colors = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'RESET': '\033[0m',
            'BOLD': '\033[1m'
        }
    
    def print_banner(self):
        """Display cyberpunk-style setup banner"""
        banner = f"""
{self.colors['CYAN']}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  {self.colors['BOLD']}███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗{self.colors['CYAN']}      ║
║  {self.colors['BOLD']}████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝{self.colors['CYAN']}      ║
║  {self.colors['BOLD']}██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝{self.colors['CYAN']}       ║
║  {self.colors['BOLD']}██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝{self.colors['CYAN']}        ║
║  {self.colors['BOLD']}██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║{self.colors['CYAN']}         ║
║  {self.colors['BOLD']}╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝{self.colors['CYAN']}         ║
║                                                               ║
║          {self.colors['GREEN']}Neural Trading System Setup v2.0.8{self.colors['CYAN']}                  ║
║          {self.colors['BLUE']}Cyberpunk Trading Engine Installation{self.colors['CYAN']}              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{self.colors['RESET']}
        """
        print(banner)
    
    def run(self):
        """Main setup orchestration"""
        try:
            self.print_banner()
            
            # Phase 1: System Validation
            print(f"\n{self.colors['CYAN']}[Phase 1/6] System Validation{self.colors['RESET']}")
            if not self.check_system_requirements():
                return False
            
            # Phase 2: Dependency Check
            print(f"\n{self.colors['CYAN']}[Phase 2/6] Dependency Installation{self.colors['RESET']}")
            if not self.install_dependencies():
                return False
            
            # Phase 3: Directory Structure
            print(f"\n{self.colors['CYAN']}[Phase 3/6] Creating Directory Structure{self.colors['RESET']}")
            self.create_directory_structure()
            
            # Phase 4: Database Initialization
            print(f"\n{self.colors['CYAN']}[Phase 4/6] Database Initialization{self.colors['RESET']}")
            self.initialize_databases()
            
            # Phase 5: Configuration
            print(f"\n{self.colors['CYAN']}[Phase 5/6] Configuration Setup{self.colors['RESET']}")
            self.create_configurations()
            
            # Phase 6: Final Setup
            print(f"\n{self.colors['CYAN']}[Phase 6/6] Finalizing Installation{self.colors['RESET']}")
            self.finalize_setup()
            
            # Display summary
            self.display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            print(f"\n{self.colors['RED']}✗ Setup failed: {str(e)}{self.colors['RESET']}")
            return False
    
    def check_system_requirements(self) -> bool:
        """Comprehensive system requirement validation"""
        print("\nChecking system requirements...")
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if float(python_version) < float(self.requirements['python_version']):
            self.errors.append(
                f"Python {self.requirements['python_version']}+ required, found {python_version}"
            )
            print(f"  {self.colors['RED']}✗ Python version{self.colors['RESET']}")
        else:
            print(f"  {self.colors['GREEN']}✓ Python {python_version}{self.colors['RESET']}")
        
        # Operating System
        os_name = platform.system()
        if os_name == 'Windows':
            # Check for Visual C++ runtime
            try:
                import ctypes
                ctypes.cdll.msvcrt
                print(f"  {self.colors['GREEN']}✓ Windows with C++ runtime{self.colors['RESET']}")
            except:
                self.warnings.append("Visual C++ runtime may be required for some packages")
                print(f"  {self.colors['YELLOW']}! Windows (C++ runtime recommended){self.colors['RESET']}")
        else:
            print(f"  {self.colors['GREEN']}✓ {os_name} OS{self.colors['RESET']}")
        
        # RAM Check
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < self.requirements['min_ram_gb']:
            self.errors.append(
                f"Minimum {self.requirements['min_ram_gb']}GB RAM required, found {ram_gb:.1f}GB"
            )
            print(f"  {self.colors['RED']}✗ RAM: {ram_gb:.1f}GB{self.colors['RESET']}")
        elif ram_gb < self.requirements['recommended_ram_gb']:
            self.warnings.append(
                f"Recommended {self.requirements['recommended_ram_gb']}GB RAM, found {ram_gb:.1f}GB"
            )
            print(f"  {self.colors['YELLOW']}! RAM: {ram_gb:.1f}GB (16GB recommended){self.colors['RESET']}")
        else:
            print(f"  {self.colors['GREEN']}✓ RAM: {ram_gb:.1f}GB{self.colors['RESET']}")
        
        # Disk Space
        disk_usage = psutil.disk_usage(str(self.base_path))
        free_gb = disk_usage.free / (1024**3)
        if free_gb < self.requirements['min_disk_gb']:
            self.errors.append(
                f"Minimum {self.requirements['min_disk_gb']}GB disk space required, found {free_gb:.1f}GB"
            )
            print(f"  {self.colors['RED']}✗ Disk space: {free_gb:.1f}GB{self.colors['RESET']}")
        elif free_gb < self.requirements['recommended_disk_gb']:
            self.warnings.append(
                f"Recommended {self.requirements['recommended_disk_gb']}GB disk space, found {free_gb:.1f}GB"
            )
            print(f"  {self.colors['YELLOW']}! Disk space: {free_gb:.1f}GB (50GB recommended){self.colors['RESET']}")
        else:
            print(f"  {self.colors['GREEN']}✓ Disk space: {free_gb:.1f}GB{self.colors['RESET']}")
        
        # CPU Cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            self.warnings.append(f"4+ CPU cores recommended, found {cpu_count}")
            print(f"  {self.colors['YELLOW']}! CPU cores: {cpu_count} (4+ recommended){self.colors['RESET']}")
        else:
            print(f"  {self.colors['GREEN']}✓ CPU cores: {cpu_count}{self.colors['RESET']}")
        
        # GPU Check
        gpu_info = self._check_gpu()
        if gpu_info:
            print(f"  {self.colors['GREEN']}✓ GPU: {gpu_info}{self.colors['RESET']}")
        else:
            self.warnings.append(
                "No compatible GPU detected. ML features will run on CPU (slower)"
            )
            print(f"  {self.colors['YELLOW']}! No compatible GPU detected{self.colors['RESET']}")
        
        # Port availability
        print("\nChecking port availability...")
        for port, service in self.required_ports.items():
            if self._is_port_available(port):
                print(f"  {self.colors['GREEN']}✓ Port {port} ({service}){self.colors['RESET']}")
            else:
                self.errors.append(f"Port {port} ({service}) is already in use")
                print(f"  {self.colors['RED']}✗ Port {port} ({service}) - IN USE{self.colors['RESET']}")
        
        # Display system compatibility
        if os_name == 'Linux':
            # Check for GUI dependencies
            if not self._check_gui_dependencies():
                self.warnings.append(
                    "GUI dependencies missing. Install: sudo apt-get install python3-pyqt5"
                )
        
        # Docker check (optional)
        docker_available = self._check_docker()
        if docker_available:
            print(f"  {self.colors['GREEN']}✓ Docker available{self.colors['RESET']}")
        else:
            self.warnings.append("Docker not found (optional for containerized deployment)")
            print(f"  {self.colors['YELLOW']}! Docker not found (optional){self.colors['RESET']}")
        
        # Internet connectivity
        if self._check_internet():
            print(f"  {self.colors['GREEN']}✓ Internet connection{self.colors['RESET']}")
        else:
            self.errors.append("No internet connection detected")
            print(f"  {self.colors['RED']}✗ No internet connection{self.colors['RESET']}")
        
        # Check if errors exist
        if self.errors:
            print(f"\n{self.colors['RED']}System requirements not met:{self.colors['RESET']}")
            for error in self.errors:
                print(f"  • {error}")
            return False
        
        return True
    
    def _check_gpu(self) -> Optional[str]:
        """Check for compatible GPU"""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                # Check if it's a supported model
                for model in self.requirements['gpu_models']:
                    if model in gpu_name:
                        return gpu_name
                return f"{gpu_name} (compatibility unknown)"
        except:
            pass
        
        # Try alternative methods
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].name
        except:
            pass
        
        return None
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
    
    def _check_gui_dependencies(self) -> bool:
        """Check GUI dependencies on Linux"""
        if platform.system() != 'Linux':
            return True
        
        try:
            # Check for X11 display
            if 'DISPLAY' not in os.environ:
                return False
            
            # Check for PyQt5
            subprocess.run(['python3', '-c', 'import PyQt5'], check=True)
            return True
        except:
            return False
    
    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_internet(self) -> bool:
        """Check internet connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies with profile selection"""
        print("\nSelect installation profile:")
        print("  1. Standard - Core trading features (recommended)")
        print("  2. Full - All features including ML/GPU support")
        
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == '2':
            profile = 'full'
            requirements_file = 'requirements_full.txt'
        else:
            profile = 'standard'
            requirements_file = 'requirements_standard.txt'
        
        print(f"\nInstalling {profile} profile dependencies...")
        
        # Create requirements files
        self._create_requirements_files()
        
        # Upgrade pip first
        print("  Upgrading pip...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            print(f"  {self.colors['GREEN']}✓ Pip upgraded{self.colors['RESET']}")
        except:
            self.warnings.append("Failed to upgrade pip")
        
        # Install requirements
        print(f"  Installing from {requirements_file}...")
        try:
            # Use --no-deps first to avoid conflicts
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
                check=True
            )
            print(f"  {self.colors['GREEN']}✓ Dependencies installed{self.colors['RESET']}")
            
            # Verify critical packages
            critical_packages = ['ccxt', 'PyQt5', 'pandas', 'numpy', 'asyncio', 'aiohttp']
            missing = []
            
            for package in critical_packages:
                try:
                    pkg_resources.get_distribution(package)
                except:
                    missing.append(package)
            
            if missing:
                print(f"  {self.colors['YELLOW']}! Missing packages: {', '.join(missing)}{self.colors['RESET']}")
                print("  Attempting individual installation...")
                for package in missing:
                    try:
                        subprocess.run(
                            [sys.executable, '-m', 'pip', 'install', package],
                            check=True
                        )
                    except:
                        self.errors.append(f"Failed to install {package}")
            
            return len(self.errors) == 0
            
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Dependency installation failed: {e}")
            print(f"  {self.colors['RED']}✗ Installation failed{self.colors['RESET']}")
            return False
    
    def _create_requirements_files(self):
        """Create requirements files for different profiles"""
        # Standard requirements (no ML)
        standard_reqs = """# Nexlify Standard Requirements
ccxt>=4.1.22
pandas>=2.0.3
numpy>=1.24.3
asyncio>=3.4.3
aiohttp>=3.8.5
PyQt5>=5.15.9
qasync>=0.23.0
python-dotenv>=1.0.0
requests>=2.31.0
websocket-client>=1.6.1
pycryptodome>=3.18.0
argon2-cffi>=21.3.0
pyotp>=2.8.0
qrcode>=7.4.2
Pillow>=10.0.0
psutil>=5.9.5
colorama>=0.4.6
pyyaml>=6.0.1
jsonschema>=4.19.1
marshmallow>=3.20.1
pydantic>=2.4.2
sqlalchemy>=2.0.21
alembic>=1.12.0
redis>=5.0.0
celery>=5.3.1
pytest>=7.4.2
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
black>=23.9.1
flake8>=6.1.0
mypy>=1.5.1
sphinx>=7.2.6
sphinx-rtd-theme>=1.3.0
loguru>=0.7.2
python-json-logger>=2.0.7
sentry-sdk>=1.32.0
prometheus-client>=0.17.1
py-cpuinfo>=9.0.0
psycopg2-binary>=2.9.7
cryptography>=41.0.4
web3>=6.9.0
eth-account>=0.9.0
"""
        
        # Full requirements (with ML)
        full_reqs = standard_reqs + """
# ML and Advanced Features
tensorflow>=2.13.0
keras>=2.13.0
scikit-learn>=1.3.0
xgboost>=1.7.6
lightgbm>=4.1.0
torch>=2.0.1
pandas-ta>=0.3.14b0
statsmodels>=0.14.0
scipy>=1.11.3
matplotlib>=3.7.2
seaborn>=0.12.2
plotly>=5.17.0
bokeh>=3.2.2
jupyter>=1.0.0
ipykernel>=6.25.2
nbformat>=5.9.2
"""
        
        # Write files
        with open('requirements_standard.txt', 'w') as f:
            f.write(standard_reqs)
        
        with open('requirements_full.txt', 'w') as f:
            f.write(full_reqs)
    
    def create_directory_structure(self):
        """Create complete directory structure with proper permissions"""
        print("\nCreating directory structure...")
        
        directories = {
            'config': 0o700,  # Sensitive configs
            'data': 0o755,
            'data/market': 0o755,
            'data/models': 0o755,
            'logs': 0o755,
            'logs/trading': 0o755,
            'logs/errors': 0o755,
            'logs/audit': 0o700,  # Audit logs need protection
            'logs/performance': 0o755,
            'logs/crash_reports': 0o755,
            'backups': 0o700,  # Backups are sensitive
            'backups/config': 0o700,
            'backups/database': 0o700,
            'backups/logs': 0o700,
            'scripts': 0o755,
            'assets': 0o755,
            'assets/sounds': 0o755,
            'assets/images': 0o755,
            'temp': 0o755,
            'reports': 0o755,
            'keys': 0o700  # Encryption keys
        }
        
        for dir_path, permissions in directories.items():
            full_path = self.base_path / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                
                # Set permissions on Unix-like systems
                if platform.system() != 'Windows':
                    os.chmod(full_path, permissions)
                
                print(f"  {self.colors['GREEN']}✓ Created {dir_path}{self.colors['RESET']}")
                
            except Exception as e:
                self.errors.append(f"Failed to create {dir_path}: {e}")
                print(f"  {self.colors['RED']}✗ Failed to create {dir_path}{self.colors['RESET']}")
        
        # Create .gitignore
        self._create_gitignore()
    
    def _create_gitignore(self):
        """Create comprehensive .gitignore file"""
        gitignore_content = """# Nexlify .gitignore

# Sensitive files
config/enhanced_config.json
config/neural_config.json
.env
*.key
*.pem
keys/

# Database
*.db
*.sqlite
*.sqlite3
data/trading.db

# Logs
logs/
*.log

# Backups
backups/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
temp/
*.tmp
*.temp

# Model files
*.h5
*.pkl
*.joblib
models/

# Reports
reports/*.pdf
reports/*.csv

# Emergency stop file
EMERGENCY_STOP_ACTIVE
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
    
    def initialize_databases(self):
        """Initialize all databases with full schema"""
        print("\nInitializing databases...")
        
        # SQLite main database
        self._init_trading_db()
        
        # Audit database
        self._init_audit_db()
        
        # Create backup schedule
        self._setup_backup_schedule()
    
    def _init_trading_db(self):
        """Initialize main trading database with full schema"""
        db_path = self.base_path / 'data' / 'trading.db'
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_admin BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                two_factor_secret TEXT,
                api_key_hash TEXT
            )
            """)
            
            # Trading tables
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                pair TEXT NOT NULL,
                exchange TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                type TEXT NOT NULL CHECK(type IN ('market', 'limit', 'stop-limit')),
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fee REAL DEFAULT 0,
                fee_currency TEXT,
                profit_usdt REAL,
                profit_percentage REAL,
                strategy TEXT,
                order_id TEXT,
                status TEXT DEFAULT 'completed',
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_exchange ON trades(exchange)")
            
            # Positions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                pair TEXT NOT NULL,
                exchange TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                pnl REAL,
                pnl_percentage REAL,
                strategy TEXT,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            
            # Withdrawals table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS withdrawals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                amount_usdt REAL NOT NULL,
                amount_btc REAL NOT NULL,
                btc_price REAL NOT NULL,
                btc_address TEXT NOT NULL,
                tx_hash TEXT,
                exchange TEXT,
                status TEXT DEFAULT 'pending',
                confirmed_at TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            
            # Strategies table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                config TEXT,
                performance_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Strategy performance
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trades_count INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                sharpe_ratio REAL,
                max_drawdown REAL,
                metadata TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            )
            """)
            
            # Exchange connections
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchange_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                exchange TEXT NOT NULL,
                api_key_encrypted TEXT NOT NULL,
                secret_encrypted TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                testnet BOOLEAN DEFAULT 0,
                last_connected TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            
            # System configuration
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                type TEXT DEFAULT 'string',
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Performance metrics
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                component TEXT,
                metadata TEXT
            )
            """)
            
            # Create triggers for updated_at
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_strategies_timestamp 
            AFTER UPDATE ON strategies
            BEGIN
                UPDATE strategies SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """)
            
            # Insert default data
            cursor.execute("""
            INSERT OR IGNORE INTO users (username, password_hash, is_admin) 
            VALUES ('admin', '', 1)
            """)
            
            # Insert default strategies
            default_strategies = [
                ('arbitrage', 'arbitrage'),
                ('momentum', 'trend_following'),
                ('market_making', 'liquidity_provision'),
                ('dex_integration', 'defi'),
                ('ai_predictions', 'machine_learning')
            ]
            
            for name, strategy_type in default_strategies:
                cursor.execute("""
                INSERT OR IGNORE INTO strategies (name, type) 
                VALUES (?, ?)
                """, (name, strategy_type))
            
            conn.commit()
            conn.close()
            
            print(f"  {self.colors['GREEN']}✓ Trading database initialized{self.colors['RESET']}")
            
        except Exception as e:
            self.errors.append(f"Failed to initialize trading database: {e}")
            print(f"  {self.colors['RED']}✗ Trading database initialization failed{self.colors['RESET']}")
    
    def _init_audit_db(self):
        """Initialize audit trail database"""
        db_path = self.base_path / 'logs' / 'audit' / 'audit_trail.db'
        
        try:
            # Ensure directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Audit entries table (from nexlify_audit_trail.py)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT,
                ip_address TEXT,
                component TEXT,
                action TEXT,
                details TEXT,
                severity TEXT,
                hash TEXT NOT NULL,
                previous_hash TEXT,
                signature TEXT
            )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_entries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_entries(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_entries(user_id)")
            
            conn.commit()
            conn.close()
            
            print(f"  {self.colors['GREEN']}✓ Audit database initialized{self.colors['RESET']}")
            
        except Exception as e:
            self.errors.append(f"Failed to initialize audit database: {e}")
            print(f"  {self.colors['RED']}✗ Audit database initialization failed{self.colors['RESET']}")
    
    def _setup_backup_schedule(self):
        """Setup automated backup configuration"""
        backup_config = {
            'enabled': True,
            'schedule': {
                'databases': {
                    'frequency': 'daily',
                    'time': '03:00',
                    'retention_days': 30
                },
                'configs': {
                    'frequency': 'on_change',
                    'retention_count': 10
                },
                'logs': {
                    'frequency': 'weekly',
                    'day': 'sunday',
                    'time': '02:00',
                    'retention_days': 90
                }
            },
            'compression': 'zip',
            'encryption': True
        }
        
        # Save backup configuration
        backup_config_path = self.base_path / 'config' / 'backup_config.json'
        backup_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_config_path, 'w') as f:
            json.dump(backup_config, f, indent=4)
        
        # Create backup script
        self._create_backup_script()
    
    def _create_backup_script(self):
        """Create automated backup script"""
        script_content = '''#!/usr/bin/env python3
"""
Nexlify Automated Backup Script
Handles database, configuration, and log backups
"""

import os
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

def backup_databases():
    """Backup all databases"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups/database") / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup trading database
    src_db = Path("data/trading.db")
    if src_db.exists():
        dst_db = backup_dir / "trading.db"
        
        # Use SQLite backup API for consistency
        src_conn = sqlite3.connect(str(src_db))
        dst_conn = sqlite3.connect(str(dst_db))
        
        with dst_conn:
            src_conn.backup(dst_conn)
        
        src_conn.close()
        dst_conn.close()
        
        print(f"✓ Backed up trading database")
    
    # Backup audit database
    src_audit = Path("logs/audit/audit_trail.db")
    if src_audit.exists():
        dst_audit = backup_dir / "audit_trail.db"
        shutil.copy2(src_audit, dst_audit)
        print(f"✓ Backed up audit database")
    
    # Create compressed archive
    archive_path = backup_dir.parent / f"db_backup_{timestamp}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in backup_dir.rglob('*'):
            zf.write(file, file.relative_to(backup_dir))
    
    # Remove uncompressed files
    shutil.rmtree(backup_dir)
    
    return archive_path

def backup_configs():
    """Backup configuration files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups/config") / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Config files to backup
    config_files = [
        "config/enhanced_config.json",
        "config/neural_config.json",
        "config/backup_config.json",
        ".env"
    ]
    
    for config_file in config_files:
        src = Path(config_file)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            print(f"✓ Backed up {config_file}")
    
    # Create archive
    archive_path = backup_dir.parent / f"config_backup_{timestamp}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in backup_dir.rglob('*'):
            zf.write(file, file.relative_to(backup_dir))
    
    shutil.rmtree(backup_dir)
    
    return archive_path

def cleanup_old_backups(directory: Path, retention_days: int):
    """Remove backups older than retention period"""
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    for backup_file in directory.glob("*.zip"):
        if backup_file.stat().st_mtime < cutoff_date.timestamp():
            backup_file.unlink()
            print(f"✓ Removed old backup: {backup_file.name}")

def main():
    """Main backup routine"""
    print(f"Starting backup at {datetime.now()}")
    
    # Load backup configuration
    with open("config/backup_config.json", 'r') as f:
        config = json.load(f)
    
    if not config.get('enabled', True):
        print("Backups are disabled")
        return
    
    # Perform backups
    try:
        # Database backup
        db_backup = backup_databases()
        print(f"Database backup created: {db_backup}")
        
        # Configuration backup
        config_backup = backup_configs()
        print(f"Configuration backup created: {config_backup}")
        
        # Cleanup old backups
        cleanup_old_backups(
            Path("backups/database"),
            config['schedule']['databases']['retention_days']
        )
        
        print("Backup completed successfully")
        
    except Exception as e:
        print(f"Backup failed: {e}")
        # Log to error file
        with open("logs/errors/backup_errors.log", 'a') as f:
            f.write(f"{datetime.now()}: Backup failed - {str(e)}\\n")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.base_path / 'scripts' / 'backup.py'
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix
        if platform.system() != 'Windows':
            os.chmod(script_path, 0o755)
    
    def create_configurations(self):
        """Create all configuration files"""
        print("\nCreating configuration files...")
        
        # Enhanced configuration
        self._create_enhanced_config()
        
        # Environment file
        self._create_env_file()
        
        # Docker configuration
        self._create_docker_config()
        
        # Systemd service (Linux)
        if platform.system() == 'Linux':
            self._create_systemd_service()
    
    def _create_enhanced_config(self):
        """Create enhanced configuration file"""
        config = {
            "version": "2.0.8",
            "environment": "development",
            "debug": False,
            
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 300
            },
            
            "mobile_api": {
                "enabled": True,
                "port": 8001,
                "max_connections": 100
            },
            
            "database": {
                "main_db": "sqlite:///data/trading.db",
                "audit_db": "sqlite:///logs/audit/audit_trail.db",
                "pool_size": 10,
                "pool_timeout": 30
            },
            
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": "",
                "decode_responses": True
            },
            
            "exchanges": {},
            
            "trading": {
                "initial_capital": 10000,
                "risk_level": "medium",
                "min_profit_threshold": 0.5,
                "max_spread_percentage": 2.0,
                "min_volume_usdt": 10000,
                "max_concurrent_trades": 10,
                "scan_interval_seconds": 300,
                "auto_trade": False,
                "testnet": True
            },
            
            "withdrawal": {
                "btc_address": "",
                "min_withdrawal_usdt": 100,
                "withdrawal_percentage": 50,
                "auto_withdraw": False
            },
            
            "security": {
                "master_password_enabled": False,
                "master_password": "",
                "2fa_enabled": False,
                "ip_whitelist_enabled": False,
                "ip_whitelist": ["127.0.0.1"],
                "session_timeout_minutes": 30,
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 15
            },
            
            "notifications": {
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": ""
                },
                "email": {
                    "enabled": False,
                    "smtp_host": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "",
                    "emergency_contact": ""
                }
            },
            
            "features": {
                "enable_predictive": True,
                "enable_multi_strategy": True,
                "enable_dex_integration": False,
                "enable_ai_companion": True,
                "enable_audit": True,
                "enable_mobile_api": True,
                "enable_gpu": False
            },
            
            "performance": {
                "use_cython": False,
                "cache_size_mb": 512,
                "log_rotation_mb": 100,
                "log_retention_days": 30,
                "parallel_strategies": True,
                "batch_size": 1000
            },
            
            "ui": {
                "theme": "cyberpunk",
                "sound_effects": True,
                "animations": True,
                "auto_refresh_seconds": 5
            },
            
            "backup": {
                "auto_backup": True,
                "backup_interval_hours": 24,
                "retention_days": 30
            },
            
            "pin": "2077"
        }
        
        config_path = self.base_path / 'config' / 'enhanced_config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"  {self.colors['GREEN']}✓ Enhanced configuration created{self.colors['RESET']}")
    
    def _create_env_file(self):
        """Create environment file template"""
        env_content = """# Nexlify Environment Configuration

# Database
DATABASE_URL=sqlite:///data/trading.db
AUDIT_DB_URL=sqlite:///logs/audit/audit_trail.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
MASTER_PASSWORD=
SECRET_KEY=

# Exchange API Keys (Encrypted)
BINANCE_API_KEY=
BINANCE_SECRET=
BYBIT_API_KEY=
BYBIT_SECRET=
OKX_API_KEY=
OKX_SECRET=

# Blockchain RPC
ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
BSC_RPC_URL=https://bsc-dataseed.binance.org/

# Mobile API
MOBILE_API_SECRET=your-secret-key-here

# Telegram Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
EMERGENCY_CONTACT=

# OpenAI (for AI Companion)
OPENAI_API_KEY=

# Sentry Error Tracking (Optional)
SENTRY_DSN=

# Development
DEBUG=False
LOG_LEVEL=INFO
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        # Also create .env.example
        with open('.env.example', 'w') as f:
            f.write(env_content)
        
        print(f"  {self.colors['GREEN']}✓ Environment files created{self.colors['RESET']}")
    
    def _create_docker_config(self):
        """Create Docker configuration files"""
        # Dockerfile
        dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    libssl-dev \\
    libffi-dev \\
    python3-dev \\
    gcc \\
    g++ \\
    make \\
    git \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_standard.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_standard.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data logs backups config keys

# Set permissions
RUN chmod -R 755 /app

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "smart_launcher.py"]
"""
        
        # Docker Compose
        docker_compose_content = """version: '3.8'

services:
  nexlify:
    build: .
    container_name: nexlify_trading
    restart: unless-stopped
    ports:
      - "8000:8000"  # API
      - "8001:8001"  # Mobile API
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
      - ./backups:/app/backups
    environment:
      - DATABASE_URL=postgresql://nexlify:nexlify@postgres:5432/nexlify
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - nexlify_network

  postgres:
    image: postgres:15-alpine
    container_name: nexlify_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: nexlify
      POSTGRES_USER: nexlify
      POSTGRES_PASSWORD: nexlify_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexlify"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nexlify_network

  redis:
    image: redis:7-alpine
    container_name: nexlify_redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nexlify_network

volumes:
  postgres_data:
  redis_data:

networks:
  nexlify_network:
    driver: bridge
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        with open('docker-compose.yml', 'w') as f:
            f.write(docker_compose_content)
        
        print(f"  {self.colors['GREEN']}✓ Docker configuration created{self.colors['RESET']}")
    
    def _create_systemd_service(self):
        """Create systemd service file for Linux"""
        service_content = f"""[Unit]
Description=Nexlify Neural Trading System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'nexlify')}
WorkingDirectory={self.base_path}
Environment="PATH={sys.prefix}/bin"
ExecStart={sys.executable} {self.base_path}/smart_launcher.py
Restart=on-failure
RestartSec=10

# Security
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={self.base_path}/data {self.base_path}/logs {self.base_path}/backups

[Install]
WantedBy=multi-user.target
"""
        
        service_path = self.base_path / 'nexlify.service'
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        print(f"  {self.colors['GREEN']}✓ Systemd service file created{self.colors['RESET']}")
        print(f"    To install: sudo cp nexlify.service /etc/systemd/system/")
        print(f"    To enable: sudo systemctl enable nexlify")
        print(f"    To start: sudo systemctl start nexlify")
    
    def finalize_setup(self):
        """Finalize installation"""
        print("\nFinalizing installation...")
        
        # Create launch scripts
        self._create_launch_scripts()
        
        # Create README
        self._create_readme()
        
        # Set up cron jobs
        if platform.system() != 'Windows':
            self._setup_cron_jobs()
        
        # Initialize encryption keys
        self._init_encryption_keys()
    
    def _create_launch_scripts(self):
        """Create platform-specific launch scripts"""
        # Windows batch script
        if platform.system() == 'Windows':
            bat_content = f"""@echo off
echo Starting Nexlify Neural Trading System...
"{sys.executable}" smart_launcher.py
pause
"""
            with open('start_nexlify.bat', 'w') as f:
                f.write(bat_content)
        
        # Unix shell script
        else:
            sh_content = f"""#!/bin/bash
echo "Starting Nexlify Neural Trading System..."
{sys.executable} smart_launcher.py
"""
            with open('start_nexlify.sh', 'w') as f:
                f.write(sh_content)
            os.chmod('start_nexlify.sh', 0o755)
        
        print(f"  {self.colors['GREEN']}✓ Launch scripts created{self.colors['RESET']}")
    
    def _create_readme(self):
        """Create comprehensive README"""
        readme_content = """# Nexlify Neural Trading System v2.0.8

## 🚀 Quick Start

1. **Start the system:**
   - Windows: Double-click `start_nexlify.bat`
   - Linux/Mac: Run `./start_nexlify.sh`

2. **Access the GUI:**
   - The trading interface will open automatically
   - Default PIN: 2077 (change this immediately!)

3. **Configure exchanges:**
   - Go to Settings tab
   - Add your exchange API keys
   - Test connections before trading

## 📋 System Requirements

- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 20GB disk space minimum
- Internet connection
- GPU (optional): NVIDIA GTX 1060+ for ML features

## 🔧 Configuration

- Main config: `config/enhanced_config.json`
- Environment: `.env` file
- Logs: `logs/` directory
- Backups: `backups/` directory

## 🔒 Security

1. Change default PIN immediately
2. Enable 2FA in settings (recommended)
3. Configure IP whitelist for production
4. Keep your API keys secure

## 📱 Mobile App

- Mobile API runs on port 8001
- Use QR code in settings to pair devices
- Supports iOS and Android apps

## 🆘 Troubleshooting

- Check logs in `logs/errors/`
- Emergency stop: Create `EMERGENCY_STOP_ACTIVE` file
- Database issues: Run backup script first

## 📞 Support

- Documentation: https://docs.nexlify.com
- Issues: https://github.com/nexlify/support
- Email: support@nexlify.com

## ⚠️ Disclaimer

Trading cryptocurrencies involves substantial risk. Only trade with funds you can afford to lose.
"""
        
        with open('README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"  {self.colors['GREEN']}✓ README created{self.colors['RESET']}")
    
    def _setup_cron_jobs(self):
        """Setup cron jobs for automated tasks"""
        cron_content = f"""# Nexlify Automated Tasks

# Daily backup at 3 AM
0 3 * * * cd {self.base_path} && {sys.executable} scripts/backup.py >> logs/backup.log 2>&1

# Clean old logs weekly
0 2 * * 0 find {self.base_path}/logs -name "*.log" -mtime +30 -delete

# Monitor disk space
0 * * * * df -h {self.base_path} | tail -1 | awk '{{if($(NF-1) > 90) print "Disk usage critical: " $(NF-1)}}' >> logs/system.log
"""
        
        cron_file = self.base_path / 'nexlify.cron'
        with open(cron_file, 'w') as f:
            f.write(cron_content)
        
        print(f"  {self.colors['GREEN']}✓ Cron jobs configured{self.colors['RESET']}")
        print(f"    To install: crontab nexlify.cron")
    
    def _init_encryption_keys(self):
        """Initialize encryption keys directory"""
        keys_dir = self.base_path / 'keys'
        keys_dir.mkdir(exist_ok=True)
        
        # Create key generation script
        keygen_content = '''#!/usr/bin/env python3
"""Generate encryption keys for Nexlify"""

import os
import secrets
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

# Generate master key
master_key = secrets.token_bytes(32)
with open('keys/master.key', 'wb') as f:
    f.write(master_key)

# Generate RSA key pair for audit signatures
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# Save private key
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)
with open('keys/audit_key.pem', 'wb') as f:
    f.write(pem_private)

# Save public key
public_key = private_key.public_key()
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
with open('keys/audit_key_public.pem', 'wb') as f:
    f.write(pem_public)

print("✓ Encryption keys generated")
'''
        
        keygen_path = self.base_path / 'scripts' / 'generate_keys.py'
        with open(keygen_path, 'w') as f:
            f.write(keygen_content)
        
        if platform.system() != 'Windows':
            os.chmod(keygen_path, 0o755)
    
    def display_summary(self):
        """Display installation summary"""
        print(f"\n{self.colors['CYAN']}{'='*60}{self.colors['RESET']}")
        print(f"{self.colors['GREEN']}✓ NEXLIFY INSTALLATION COMPLETE!{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}{'='*60}{self.colors['RESET']}")
        
        # System info
        print(f"\n{self.colors['BOLD']}System Information:{self.colors['RESET']}")
        print(f"  • Python: {sys.version.split()[0]}")
        print(f"  • Platform: {platform.system()} {platform.release()}")
        print(f"  • Installation path: {self.base_path}")
        
        # Warnings
        if self.warnings:
            print(f"\n{self.colors['YELLOW']}⚠️  Warnings:{self.colors['RESET']}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Next steps
        print(f"\n{self.colors['BOLD']}Next Steps:{self.colors['RESET']}")
        print(f"  1. Configure your exchange API keys in Settings")
        print(f"  2. Change the default PIN (2077) immediately")
        print(f"  3. Enable 2FA for enhanced security (optional)")
        print(f"  4. Run backups regularly")
        
        # Launch instructions
        print(f"\n{self.colors['BOLD']}To start Nexlify:{self.colors['RESET']}")
        if platform.system() == 'Windows':
            print(f"  • Double-click: start_nexlify.bat")
            print(f"  • Or run: python smart_launcher.py")
        else:
            print(f"  • Run: ./start_nexlify.sh")
            print(f"  • Or: python3 smart_launcher.py")
        
        print(f"\n{self.colors['GREEN']}Happy Trading! 🚀{self.colors['RESET']}\n")


def main():
    """Main entry point"""
    setup = NexlifySetup()
    
    try:
        success = setup.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{setup.colors['RED']}Fatal error: {e}{setup.colors['RESET']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
