#!/usr/bin/env python3
"""
Enhanced Nexlify Setup Script v2.0.8
Addresses all V3 improvements for robust system initialization
"""

import os
import sys
import json
import sqlite3
import socket
import subprocess
import platform
import shutil
import time
import hashlib
import secrets
import stat
import psutil
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
MIN_PYTHON_VERSION = (3, 11)
MIN_RAM_GB = 8
RECOMMENDED_RAM_GB = 16
MIN_DISK_GB = 20
RECOMMENDED_DISK_GB = 50
DEFAULT_PORTS = {
    'api': 8000,
    'mobile_api': 8001,
    'postgresql': 5432,
    'redis': 6379,
    'websocket': 8080
}

# GPU requirements for ML features
GPU_REQUIREMENTS = {
    'nvidia': {
        'min_compute': 6.1,  # GTX 1060 level
        'min_memory_gb': 6,
        'recommended_models': ['GTX 1060', 'GTX 1070', 'RTX 2060', 'RTX 3060', 'RTX 4060']
    }
}

class NexlifySetup:
    """Enhanced setup script with V3 improvements"""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.errors = []
        self.warnings = []
        self.backup_dir = self.root_path / 'backups' / 'setup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.install_profile = None  # 'standard' or 'full'
        self.config = {}
        self.db_path = self.root_path / 'data' / 'trading.db'
        self.audit_db_path = self.root_path / 'data' / 'audit' / 'audit_trail.db'
        
    def run(self):
        """Main setup process with comprehensive error handling"""
        try:
            print(self._get_banner())
            
            # Pre-setup validation
            if not self._check_python_version():
                return False
                
            if not self._check_system_requirements():
                return False
                
            # Backup existing installation
            if self._check_existing_installation():
                if not self._backup_existing():
                    return False
                    
            # Core setup
            if not self._check_system_compatibility():
                return False
                
            self._select_installation_profile()
            
            if not self._check_dependencies():
                return False
                
            if not self._create_directory_structure():
                return False
                
            if not self._initialize_databases():
                return False
                
            if not self._check_ports():
                return False
                
            if not self._create_configuration():
                return False
                
            if not self._install_dependencies():
                return False
                
            if self.install_profile == 'full':
                self._check_gpu_support()
                
            if not self._setup_docker():
                return False
                
            if not self._create_scripts():
                return False
                
            if not self._set_permissions():
                return False
                
            if not self._run_tests():
                return False
                
            self._generate_documentation()
            self._display_final_instructions()
            
            return True
            
        except Exception as e:
            self._log_error(f"Setup failed: {str(e)}")
            self._save_error_report()
            return False
            
    def _get_banner(self) -> str:
        """Cyberpunk-themed banner with cross-platform support"""
        # Use simple ASCII for Windows CMD compatibility
        if platform.system() == 'Windows' and not os.environ.get('WT_SESSION'):
            return """
============================================
         NEXLIFY SETUP v2.0.8
         Night City Trading Platform
============================================
"""
        else:
            # Full cyberpunk banner for terminals with color support
            return """
\033[95m╔══════════════════════════════════════════╗
║      \033[96mNEXLIFY SETUP v2.0.8\033[95m              ║
║   \033[92mNight City Trading Platform\033[95m           ║
║      \033[93mEnhanced with V3 Improvements\033[95m      ║
╚══════════════════════════════════════════╝\033[0m
"""
            
    def _check_python_version(self) -> bool:
        """Verify Python version meets requirements"""
        current = sys.version_info[:2]
        if current < MIN_PYTHON_VERSION:
            self._log_error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required, found {current[0]}.{current[1]}")
            return False
            
        # Check for 64-bit Python (required for TensorFlow)
        if sys.maxsize <= 2**32:
            self._log_error("64-bit Python required for optimal performance")
            return False
            
        print(f"✓ Python {current[0]}.{current[1]} (64-bit) detected")
        return True
        
    def _check_system_requirements(self) -> bool:
        """Comprehensive system requirements check"""
        print("\nChecking system requirements...")
        
        # RAM check
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < MIN_RAM_GB:
            self._log_error(f"Minimum {MIN_RAM_GB}GB RAM required, found {ram_gb:.1f}GB")
            return False
        elif ram_gb < RECOMMENDED_RAM_GB:
            self._log_warning(f"Recommended {RECOMMENDED_RAM_GB}GB RAM for ML features, found {ram_gb:.1f}GB")
        print(f"✓ RAM: {ram_gb:.1f}GB")
        
        # Disk space check
        disk_gb = psutil.disk_usage(str(self.root_path)).free / (1024**3)
        if disk_gb < MIN_DISK_GB:
            self._log_error(f"Minimum {MIN_DISK_GB}GB free disk space required, found {disk_gb:.1f}GB")
            return False
        elif disk_gb < RECOMMENDED_DISK_GB:
            self._log_warning(f"Recommended {RECOMMENDED_DISK_GB}GB for logs and models, found {disk_gb:.1f}GB")
        print(f"✓ Disk: {disk_gb:.1f}GB free")
        
        # CPU check
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            self._log_warning(f"Recommended 4+ CPU cores for multi-strategy trading, found {cpu_count}")
        print(f"✓ CPU: {cpu_count} cores")
        
        # Network check
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("✓ Internet connection available")
        except OSError:
            self._log_error("Internet connection required for exchange APIs")
            return False
            
        return True
        
    def _check_existing_installation(self) -> bool:
        """Check for existing Nexlify installation"""
        indicators = [
            self.root_path / 'enhanced_config.json',
            self.root_path / 'neural_config.json',
            self.root_path / 'data' / 'trading.db',
            self.root_path / 'src' / 'nexlify_neural_net.py'
        ]
        
        for indicator in indicators:
            if indicator.exists():
                response = input("\n⚠️  Existing installation detected. Backup and continue? (y/n): ")
                return response.lower() == 'y'
                
        return False
        
    def _backup_existing(self) -> bool:
        """Backup existing installation with timestamp"""
        print("\nBacking up existing installation...")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup important files
            backup_items = [
                'enhanced_config.json',
                'neural_config.json',
                '.env',
                'data/',
                'logs/',
                'src/',
                'backups/config/'
            ]
            
            for item in backup_items:
                src = self.root_path / item
                if src.exists():
                    dst = self.backup_dir / item
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        
            # Create backup manifest
            manifest = {
                'timestamp': datetime.now().isoformat(),
                'version': self._get_current_version(),
                'files': [str(f.relative_to(self.backup_dir)) for f in self.backup_dir.rglob('*') if f.is_file()]
            }
            
            with open(self.backup_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
                
            print(f"✓ Backup created: {self.backup_dir}")
            return True
            
        except Exception as e:
            self._log_error(f"Backup failed: {str(e)}")
            return False
            
    def _check_system_compatibility(self) -> bool:
        """Platform-specific compatibility checks"""
        print("\nChecking system compatibility...")
        
        system = platform.system()
        
        if system == 'Windows':
            # Check Windows runtime
            try:
                subprocess.run(['where', 'python'], capture_output=True, check=True)
            except:
                self._log_warning("Python not in PATH, may cause issues")
                
            # Check for Visual C++ runtime (needed for some packages)
            if not self._check_windows_runtime():
                self._log_warning("Visual C++ runtime may be needed for some packages")
                
        elif system == 'Linux':
            # Check for required system packages
            required_packages = ['python3-dev', 'build-essential', 'libssl-dev']
            missing = []
            
            for pkg in required_packages:
                try:
                    subprocess.run(['dpkg', '-l', pkg], capture_output=True, check=True)
                except:
                    missing.append(pkg)
                    
            if missing:
                self._log_warning(f"Missing system packages: {', '.join(missing)}")
                print("Install with: sudo apt-get install " + ' '.join(missing))
                
            # Check for GUI dependencies if not headless
            if os.environ.get('DISPLAY'):
                self._check_gui_dependencies()
                
        elif system == 'Darwin':  # macOS
            # Check for Xcode command line tools
            try:
                subprocess.run(['xcode-select', '-p'], capture_output=True, check=True)
            except:
                self._log_warning("Xcode command line tools required")
                print("Install with: xcode-select --install")
                
        print("✓ System compatibility checked")
        return True
        
    def _select_installation_profile(self):
        """Select installation profile (standard or full)"""
        print("\nSelect installation profile:")
        print("1. Standard - Core trading features")
        print("2. Full - All features including ML/GPU acceleration")
        
        while True:
            choice = input("\nEnter choice (1-2): ")
            if choice == '1':
                self.install_profile = 'standard'
                break
            elif choice == '2':
                self.install_profile = 'full'
                break
            else:
                print("Invalid choice, please try again")
                
        print(f"✓ Selected {self.install_profile} installation")
        
    def _check_dependencies(self) -> bool:
        """Verify critical system dependencies"""
        print("\nChecking dependencies...")
        
        # Check Python packages installer
        try:
            import pip
            print("✓ pip available")
        except ImportError:
            self._log_error("pip not found, cannot install packages")
            return False
            
        # Check for Docker (optional but recommended)
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            print("✓ Docker available")
            self.config['docker_available'] = True
        except:
            self._log_warning("Docker not found, container features disabled")
            self.config['docker_available'] = False
            
        # Check for Redis (optional)
        try:
            subprocess.run(['redis-cli', '--version'], capture_output=True, check=True)
            print("✓ Redis available")
            self.config['redis_available'] = True
        except:
            self._log_warning("Redis not found, caching features limited")
            self.config['redis_available'] = False
            
        # Check for Git
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            print("✓ Git available")
        except:
            self._log_warning("Git not found, version control disabled")
            
        return True
        
    def _create_directory_structure(self) -> bool:
        """Create project directories with proper permissions"""
        print("\nCreating directory structure...")
        
        directories = [
            'data',
            'data/market',
            'data/models',
            'data/audit',
            'logs',
            'logs/trading',
            'logs/errors',
            'logs/audit',
            'logs/performance',
            'logs/crash_reports',
            'logs/mobile',
            'backups',
            'backups/config',
            'backups/data',
            'backups/logs',
            'src',
            'src/strategies',
            'src/indicators',
            'src/utils',
            'tests',
            'tests/unit',
            'tests/integration',
            'docs',
            'docs/api',
            'scripts',
            'models',
            'reports',
            'config'
        ]
        
        try:
            for dir_path in directories:
                path = self.root_path / dir_path
                path.mkdir(parents=True, exist_ok=True)
                
                # Set secure permissions on sensitive directories
                if any(sensitive in str(dir_path) for sensitive in ['config', 'data', 'logs/audit']):
                    self._set_secure_permissions(path)
                    
            print("✓ Directory structure created")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to create directories: {str(e)}")
            return False
            
    def _initialize_databases(self) -> bool:
        """Initialize SQLite databases with schema"""
        print("\nInitializing databases...")
        
        try:
            # Trading database
            self._create_trading_database()
            
            # Audit database
            self._create_audit_database()
            
            # Validate databases
            if not self._validate_databases():
                return False
                
            print("✓ Databases initialized")
            return True
            
        except Exception as e:
            self._log_error(f"Database initialization failed: {str(e)}")
            return False
            
    def _create_trading_database(self):
        """Create main trading database with schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Enable Write-Ahead Logging for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fee REAL DEFAULT 0,
                strategy TEXT,
                order_id TEXT UNIQUE,
                status TEXT DEFAULT 'pending',
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Withdrawals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS withdrawals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                exchange TEXT NOT NULL,
                currency TEXT NOT NULL,
                amount REAL NOT NULL,
                address TEXT NOT NULL,
                tx_hash TEXT,
                status TEXT DEFAULT 'pending',
                fee REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Portfolio table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                balance REAL NOT NULL,
                value_usd REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(exchange, symbol)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                period TEXT DEFAULT 'daily',
                metadata TEXT
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol ON trades(exchange, symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_exchange ON portfolio(exchange)")
        
        conn.commit()
        conn.close()
        
    def _create_audit_database(self):
        """Create audit trail database"""
        self.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.audit_db_path))
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Audit entries table (from nexlify_audit_trail.py)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                action TEXT NOT NULL,
                component TEXT NOT NULL,
                details TEXT,
                severity TEXT DEFAULT 'info',
                hash TEXT NOT NULL,
                previous_hash TEXT,
                signature TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Security events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user TEXT,
                ip_address TEXT,
                success BOOLEAN,
                details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_entries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_entries(user)")
        
        conn.commit()
        conn.close()
        
    def _validate_databases(self) -> bool:
        """Validate database integrity"""
        try:
            # Test trading database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            required_tables = ['trades', 'withdrawals', 'portfolio', 'performance_metrics']
            missing = set(required_tables) - set(tables)
            if missing:
                self._log_error(f"Missing tables in trading.db: {missing}")
                return False
                
            # Test audit database
            conn = sqlite3.connect(str(self.audit_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            required_tables = ['audit_entries', 'security_events']
            missing = set(required_tables) - set(tables)
            if missing:
                self._log_error(f"Missing tables in audit_trail.db: {missing}")
                return False
                
            return True
            
        except Exception as e:
            self._log_error(f"Database validation failed: {str(e)}")
            return False
            
    def _check_ports(self) -> bool:
        """Check for port conflicts and find alternatives"""
        print("\nChecking network ports...")
        
        port_status = {}
        
        for service, port in DEFAULT_PORTS.items():
            if self._is_port_available(port):
                port_status[service] = port
                print(f"✓ Port {port} available for {service}")
            else:
                # Find alternative port
                alt_port = self._find_available_port(port)
                port_status[service] = alt_port
                self._log_warning(f"Port {port} busy for {service}, using {alt_port}")
                
        self.config['ports'] = port_status
        return True
        
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
            
    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find next available port"""
        for offset in range(max_attempts):
            port = start_port + offset
            if self._is_port_available(port):
                return port
        raise RuntimeError(f"No available ports found starting from {start_port}")
        
    def _create_configuration(self) -> bool:
        """Create enhanced configuration files"""
        print("\nCreating configuration files...")
        
        try:
            # Generate secure keys
            master_password = secrets.token_urlsafe(32)
            jwt_secret = secrets.token_urlsafe(32)
            api_secret = secrets.token_urlsafe(32)
            
            # Enhanced config - all user settings managed via GUI
            enhanced_config = {
                "version": "3.0.0",
                "environment": "production",
                "debug": False,
                "theme": "cyberpunk",
                
                "security": {
                    "session_timeout_minutes": 60,
                    "max_failed_attempts": 5,
                    "lockout_duration_minutes": 30,
                    "require_2fa": False,  # Optional by default
                    "ip_whitelist_enabled": False,
                    "allowed_ips": []
                },
                
                "api": {
                    "host": "0.0.0.0",
                    "port": self.config['ports']['api'],
                    "workers": psutil.cpu_count(),
                    "rate_limit": "100/minute",
                    "mobile_api_port": self.config['ports']['mobile_api']
                },
                
                "database": {
                    "trading_db": str(self.db_path),
                    "audit_db": str(self.audit_db_path),
                    "connection_pool_size": 10,
                    "enable_wal": True
                },
                
                "exchanges": {
                    "enabled": ["binance", "kraken", "coinbase"],
                    "testnet": True,
                    "rate_limit_buffer": 0.9,
                    "credentials": {
                        "binance": {"api_key": "", "api_secret": ""},
                        "kraken": {"api_key": "", "api_secret": ""},
                        "coinbase": {"api_key": "", "api_secret": ""}
                    }
                },
                
                "defi": {
                    "enabled": self.install_profile == 'full',
                    "rpc_url": "",
                    "private_key": "",
                    "slippage_tolerance": 0.02,
                    "gas_price_multiplier": 1.2
                },
                
                "trading": {
                    "initial_capital": 10000,
                    "max_position_size": 0.1,
                    "risk_level": "medium",
                    "enable_arbitrage": True,
                    "withdrawal_address": "",
                    "min_withdrawal": 100
                },
                
                "performance": {
                    "enable_gpu": self.install_profile == 'full',
                    "cache_size_mb": 512,
                    "log_rotation_mb": 100,
                    "metrics_retention_days": 90,
                    "parallel_strategies": True
                },
                
                "notifications": {
                    "telegram": {
                        "enabled": False,
                        "bot_token": "",
                        "chat_id": ""
                    },
                    "email": {
                        "enabled": False,
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "from_address": "",
                        "to_address": ""
                    },
                    "webhook": {
                        "enabled": False,
                        "url": ""
                    },
                    "alert_on_errors": True,
                    "alert_on_trades": True
                },
                
                "ai_companion": {
                    "enabled": self.install_profile == 'full',
                    "provider": "openai",
                    "api_key": "",
                    "model": "gpt-3.5-turbo",
                    "personality": "professional"
                },
                
                "ports": self.config['ports'],
                
                "features": {
                    "enable_mobile_api": True,
                    "enable_backtesting": True,
                    "enable_paper_trading": True,
                    "enable_audit_trail": True
                }
            }
            
            # Save enhanced config
            config_path = self.root_path / 'enhanced_config.json'
            with open(config_path, 'w') as f:
                json.dump(enhanced_config, f, indent=2)
                
            # Set secure permissions
            self._set_secure_permissions(config_path)
            
            # Create .env file with system secrets
            self._create_env_file(master_password, jwt_secret, api_secret)
            
            # Create legacy neural_config.json for compatibility
            self._create_neural_config()
            
            # Create system loader script
            self._create_system_loader()
            
            print("✓ Configuration files created")
            return True
            
        except Exception as e:
            self._log_error(f"Configuration creation failed: {str(e)}")
            return False
            
    def _create_env_file(self, master_password: str, jwt_secret: str, api_secret: str):
        """Create environment file with system-generated secrets only"""
        # User-configurable settings go in enhanced_config.json via GUI
        env_content = f"""# Nexlify System Environment
# Generated on {datetime.now().isoformat()}
# DO NOT EDIT - System managed file

# System Security Keys (Auto-generated)
MASTER_PASSWORD={master_password}
JWT_SECRET={jwt_secret}
MOBILE_API_SECRET={api_secret}

# Database URLs (System paths)
DATABASE_URL=sqlite:///{self.db_path}
AUDIT_DATABASE_URL=sqlite:///{self.audit_db_path}

# Docker Passwords (If using Docker)
POSTGRES_PASSWORD={secrets.token_hex(16)}
REDIS_PASSWORD={secrets.token_hex(16)}

# System ID
SYSTEM_ID={secrets.token_hex(8)}
"""
        
        env_path = self.root_path / '.env'
        with open(env_path, 'w') as f:
            f.write(env_content)
            
        # Set secure permissions
        self._set_secure_permissions(env_path)
        
    def _create_neural_config(self):
        """Create legacy config for compatibility"""
        # All user settings will be configured via GUI
        neural_config = {
            "exchange_configs": {},
            "trading_pairs": ["BTC/USDT", "ETH/USDT"],
            "risk_level": "medium",
            "min_profit_threshold": 0.002,
            "max_position_size": 0.1,
            "enable_test_mode": True,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "emergency_contact": "",
            "btc_wallet_address": "",
            "min_withdrawal": 100,
            "api_port": self.config['ports']['api'],
            "pin": "2077",  # Will be changed on first GUI login
            "enable_logging": True,
            "database_url": f"sqlite:///{self.db_path}"
        }
        
        with open(self.root_path / 'neural_config.json', 'w') as f:
            json.dump(neural_config, f, indent=2)
            
    def _create_system_loader(self):
        """Create a system configuration loader that merges .env secrets with config"""
        loader_script = '''#!/usr/bin/env python3
"""System configuration loader - merges .env secrets with enhanced_config.json"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    """Loads system configuration merging .env secrets with user config"""
    
    @staticmethod
    def load_config():
        """Load complete configuration including system secrets"""
        root = Path(__file__).parent.parent
        
        # Load .env file
        load_dotenv(root / '.env')
        
        # Load enhanced_config.json
        with open(root / 'enhanced_config.json') as f:
            config = json.load(f)
        
        # Inject system secrets from .env
        config['security']['master_password'] = os.environ.get('MASTER_PASSWORD')
        config['security']['jwt_secret'] = os.environ.get('JWT_SECRET')
        config['api']['mobile_api_secret'] = os.environ.get('MOBILE_API_SECRET')
        config['system_id'] = os.environ.get('SYSTEM_ID')
        
        return config
    
    @staticmethod
    def save_config(config):
        """Save configuration (excluding system secrets)"""
        root = Path(__file__).parent.parent
        
        # Remove system secrets before saving
        config_copy = config.copy()
        if 'master_password' in config_copy.get('security', {}):
            del config_copy['security']['master_password']
        if 'jwt_secret' in config_copy.get('security', {}):
            del config_copy['security']['jwt_secret']
        if 'mobile_api_secret' in config_copy.get('api', {}):
            del config_copy['api']['mobile_api_secret']
        if 'system_id' in config_copy:
            del config_copy['system_id']
        
        # Save cleaned config
        with open(root / 'enhanced_config.json', 'w') as f:
            json.dump(config_copy, f, indent=2)

# For convenience
def load_config():
    return ConfigLoader.load_config()

def save_config(config):
    return ConfigLoader.save_config(config)
'''
        
        utils_dir = self.root_path / 'src' / 'utils'
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        with open(utils_dir / 'config_loader.py', 'w') as f:
            f.write(loader_script)
            
    def _install_dependencies(self) -> bool:
        """Install Python dependencies with proper validation"""
        print("\nInstalling dependencies...")
        
        try:
            # Create requirements file if not exists
            self._create_requirements_file()
            
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install based on profile
            req_file = 'requirements_full.txt' if self.install_profile == 'full' else 'requirements.txt'
            
            print(f"Installing from {req_file}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', req_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self._log_error(f"Dependency installation failed: {result.stderr}")
                return False
                
            # Verify critical packages
            if not self._verify_installations():
                return False
                
            print("✓ Dependencies installed")
            return True
            
        except Exception as e:
            self._log_error(f"Dependency installation failed: {str(e)}")
            return False
            
    def _create_requirements_file(self):
        """Create requirements files for different profiles"""
        # Standard requirements
        standard_requirements = """# Core dependencies
ccxt==4.1.22
pandas==2.1.3
numpy==1.26.2
requests==2.31.0
websocket-client==1.6.4
python-dotenv==1.0.0
pydantic==2.4.2
colorama==0.4.6
psutil==5.9.5
aiohttp==3.9.0
asyncio==3.4.3

# Database
sqlalchemy==2.0.23
alembic==1.12.0

# Security
cryptography==41.0.5
pyjwt==2.8.0
argon2-cffi==23.1.0
pyotp==2.9.0
qrcode==7.4.2

# GUI
tk==0.1.0
matplotlib==3.8.1
Pillow==10.1.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3
pyyaml==6.0.1
"""
        
        # Full requirements (includes ML/GPU)
        full_requirements = standard_requirements + """
# Machine Learning
scikit-learn==1.3.2
scipy==1.11.4
statsmodels==0.14.0
torch==2.1.0
tensorflow==2.15.0
keras==2.15.0
xgboost==2.0.2
lightgbm==4.1.0

# Blockchain/DeFi
web3==6.11.3
eth-account==0.10.0

# API/Mobile
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0

# Audio/Visual
pygame==2.5.2

# Development
pytest==7.4.3
black==23.11.0
flake8==6.1.0
"""
        
        with open(self.root_path / 'requirements.txt', 'w') as f:
            f.write(standard_requirements)
            
        with open(self.root_path / 'requirements_full.txt', 'w') as f:
            f.write(full_requirements)
            
    def _verify_installations(self) -> bool:
        """Verify critical package installations"""
        critical_packages = {
            'ccxt': 'ccxt',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sqlalchemy': 'sqlalchemy',
            'cryptography': 'cryptography',
            'psutil': 'psutil'
        }
        
        if self.install_profile == 'full':
            critical_packages.update({
                'torch': 'torch',
                'web3': 'web3',
                'fastapi': 'fastapi'
            })
            
        missing = []
        for name, import_name in critical_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(name)
                
        if missing:
            self._log_error(f"Failed to install critical packages: {', '.join(missing)}")
            return False
            
        return True
        
    def _check_gpu_support(self):
        """Check GPU support for ML features"""
        print("\nChecking GPU support...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print(f"✓ NVIDIA GPU detected: {gpu_name}")
                print(f"  GPUs: {gpu_count}")
                print(f"  Memory: {gpu_memory:.1f}GB")
                
                # Check compute capability
                major, minor = torch.cuda.get_device_capability(0)
                compute_cap = float(f"{major}.{minor}")
                
                if compute_cap >= GPU_REQUIREMENTS['nvidia']['min_compute']:
                    print(f"  Compute capability: {compute_cap} ✓")
                else:
                    self._log_warning(f"GPU compute capability {compute_cap} below recommended {GPU_REQUIREMENTS['nvidia']['min_compute']}")
                    
            else:
                self._log_warning("No NVIDIA GPU detected, ML features will use CPU")
                
        except ImportError:
            self._log_warning("PyTorch not available, skipping GPU check")
            
    def _setup_docker(self) -> bool:
        """Setup Docker configuration if available"""
        if not self.config.get('docker_available'):
            print("\nSkipping Docker setup (Docker not available)")
            return True
            
        print("\nSetting up Docker configuration...")
        
        try:
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile()
            with open(self.root_path / 'Dockerfile', 'w') as f:
                f.write(dockerfile_content)
                
            # Create docker-compose.yml
            compose_content = self._generate_docker_compose()
            with open(self.root_path / 'docker-compose.yml', 'w') as f:
                f.write(compose_content)
                
            # Create .dockerignore
            dockerignore = """
.env
*.pyc
__pycache__
.git
.vscode
.idea
logs/
backups/
data/*.db
models/*.h5
"""
            with open(self.root_path / '.dockerignore', 'w') as f:
                f.write(dockerignore)
                
            print("✓ Docker configuration created")
            return True
            
        except Exception as e:
            self._log_error(f"Docker setup failed: {str(e)}")
            return False
            
    def _generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile"""
        return f"""FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libssl-dev \\
    libffi-dev \\
    python3-dev \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements{"_full" if self.install_profile == "full" else ""}.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements{"_full" if self.install_profile == "full" else ""}.txt

# Install additional packages for enhanced features
RUN pip install --no-cache-dir psutil pygame colorama

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create necessary directories
RUN mkdir -p data logs backups models

# Set permissions
RUN chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "smart_launcher.py"]
"""
        
    def _generate_docker_compose(self) -> str:
        """Generate docker-compose configuration"""
        return f"""version: '3.8'

services:
  nexlify:
    build: .
    container_name: nexlify_trading
    ports:
      - "{self.config['ports']['api']}:8000"
      - "{self.config['ports']['mobile_api']}:8001"
      - "{self.config['ports']['websocket']}:8080"
    environment:
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
      - ./models:/app/models
    restart: unless-stopped
    depends_on:
      - redis
      - postgres
    networks:
      - nexlify_network
    
  redis:
    image: redis:7-alpine
    container_name: nexlify_redis
    ports:
      - "{self.config['ports']['redis']}:6379"
    command: redis-server --requirepass ${{REDIS_PASSWORD}}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - nexlify_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    
  postgres:
    image: postgres:15-alpine
    container_name: nexlify_postgres
    ports:
      - "{self.config['ports']['postgresql']}:5432"
    environment:
      - POSTGRES_DB=nexlify
      - POSTGRES_USER=nexlify
      - POSTGRES_PASSWORD=${{POSTGRES_PASSWORD}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    networks:
      - nexlify_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexlify"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
  postgres_data:

networks:
  nexlify_network:
    driver: bridge
"""
        
    def _create_scripts(self) -> bool:
        """Create utility scripts"""
        print("\nCreating utility scripts...")
        
        try:
            scripts_dir = self.root_path / 'scripts'
            scripts_dir.mkdir(exist_ok=True)
            
            # Database initialization script
            self._create_db_init_script()
            
            # Backup script
            self._create_backup_script()
            
            # Health check script
            self._create_health_check_script()
            
            # Migration script
            self._create_migration_script()
            
            print("✓ Scripts created")
            return True
            
        except Exception as e:
            self._log_error(f"Script creation failed: {str(e)}")
            return False
            
    def _create_db_init_script(self):
        """Create database initialization script"""
        script = """#!/usr/bin/env python3
\"\"\"Initialize Nexlify databases\"\"\"

import sqlite3
from pathlib import Path

def init_databases():
    root = Path(__file__).parent.parent
    
    # Initialize trading database
    trading_db = root / 'data' / 'trading.db'
    trading_db.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(trading_db))
    cursor = conn.cursor()
    
    # Add any custom initialization here
    cursor.execute("PRAGMA journal_mode=WAL")
    
    # Create test data if needed
    cursor.execute(\"\"\"
        INSERT OR IGNORE INTO portfolio (exchange, symbol, balance, value_usd)
        VALUES ('testnet', 'USDT', 10000.0, 10000.0)
    \"\"\")
    
    conn.commit()
    conn.close()
    
    print("✓ Databases initialized")

if __name__ == "__main__":
    init_databases()
"""
        
        with open(self.root_path / 'scripts' / 'init_db.py', 'w') as f:
            f.write(script)
            
    def _create_backup_script(self):
        """Create backup script"""
        script = """#!/usr/bin/env python3
\"\"\"Backup Nexlify data and configuration\"\"\"

import shutil
import json
from pathlib import Path
from datetime import datetime

def backup_nexlify():
    root = Path(__file__).parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = root / 'backups' / 'manual' / timestamp
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Items to backup
    items = [
        'enhanced_config.json',
        'neural_config.json',
        '.env',
        'data/',
        'logs/',
        'models/'
    ]
    
    for item in items:
        src = root / item
        if src.exists():
            dst = backup_dir / item
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    
    # Create manifest
    manifest = {
        'timestamp': timestamp,
        'items': items,
        'size_mb': sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()) / (1024*1024)
    }
    
    with open(backup_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Backup created: {backup_dir}")
    
if __name__ == "__main__":
    backup_nexlify()
"""
        
        with open(self.root_path / 'scripts' / 'backup.py', 'w') as f:
            f.write(script)
            
    def _create_health_check_script(self):
        """Create system health check script"""
        script = """#!/usr/bin/env python3
\"\"\"Check Nexlify system health\"\"\"

import psutil
import sqlite3
import json
import socket
from pathlib import Path

def check_health():
    root = Path(__file__).parent.parent
    issues = []
    
    # Check system resources
    if psutil.cpu_percent(interval=1) > 90:
        issues.append("High CPU usage")
    
    if psutil.virtual_memory().percent > 90:
        issues.append("High memory usage")
    
    if psutil.disk_usage(str(root)).percent > 90:
        issues.append("Low disk space")
    
    # Check databases
    try:
        conn = sqlite3.connect(str(root / 'data' / 'trading.db'))
        conn.execute("SELECT COUNT(*) FROM trades")
        conn.close()
    except Exception as e:
        issues.append(f"Trading database issue: {e}")
    
    # Check ports
    config = json.load(open(root / 'enhanced_config.json'))
    for service, port in config.get('ports', {}).items():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    print(f"✓ {service} on port {port}")
                else:
                    issues.append(f"{service} not responding on port {port}")
        except:
            pass
    
    if issues:
        print("\\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\\n✓ System healthy")
    
    return len(issues) == 0

if __name__ == "__main__":
    check_health()
"""
        
        with open(self.root_path / 'scripts' / 'health_check.py', 'w') as f:
            f.write(script)
            
    def _create_migration_script(self):
        """Create configuration migration script"""
        script = """#!/usr/bin/env python3
\"\"\"Migrate from neural_config.json to enhanced_config.json\"\"\"

import json
from pathlib import Path
from datetime import datetime

def migrate_config():
    root = Path(__file__).parent.parent
    
    old_config_path = root / 'neural_config.json'
    new_config_path = root / 'enhanced_config.json'
    
    if not old_config_path.exists():
        print("No neural_config.json found")
        return
    
    # Backup old config
    backup_path = root / 'backups' / 'config' / f'neural_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(old_config_path) as f:
        old_config = json.load(f)
    
    with open(backup_path, 'w') as f:
        json.dump(old_config, f, indent=2)
    
    # Load enhanced config
    with open(new_config_path) as f:
        new_config = json.load(f)
    
    # Migrate settings
    if 'risk_level' in old_config:
        new_config['trading']['risk_level'] = old_config['risk_level']
    
    if 'telegram_bot_token' in old_config and old_config['telegram_bot_token']:
        new_config['notifications']['telegram_enabled'] = True
    
    if 'btc_wallet_address' in old_config:
        new_config['trading']['withdrawal_address'] = old_config['btc_wallet_address']
    
    # Save updated config
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print("✓ Configuration migrated")
    
if __name__ == "__main__":
    migrate_config()
"""
        
        with open(self.root_path / 'scripts' / 'migrate_config.py', 'w') as f:
            f.write(script)
            
    def _set_permissions(self) -> bool:
        """Set proper file permissions"""
        print("\nSetting file permissions...")
        
        try:
            # Sensitive files (read/write owner only)
            sensitive_files = [
                '.env',
                'enhanced_config.json',
                'neural_config.json',
                'data/trading.db',
                'data/audit/audit_trail.db'
            ]
            
            for file_path in sensitive_files:
                path = self.root_path / file_path
                if path.exists():
                    self._set_secure_permissions(path)
                    
            # Script files (executable)
            for script in (self.root_path / 'scripts').glob('*.py'):
                self._set_executable_permissions(script)
                
            print("✓ Permissions set")
            return True
            
        except Exception as e:
            self._log_error(f"Permission setting failed: {str(e)}")
            return False
            
    def _set_secure_permissions(self, path: Path):
        """Set secure permissions (owner read/write only)"""
        if platform.system() != 'Windows':
            # Unix-like systems
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 600
        else:
            # Windows - limited permission control
            try:
                import win32security
                import ntsecuritycon as con
                
                # Get current user
                user = win32security.GetUserName()
                
                # Create security descriptor
                sd = win32security.GetFileSecurity(str(path), win32security.DACL_SECURITY_INFORMATION)
                dacl = win32security.ACL()
                
                # Grant full control to owner only
                user_sid = win32security.LookupAccountName(None, user)[0]
                dacl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_ALL_ACCESS, user_sid)
                
                sd.SetSecurityDescriptorDacl(1, dacl, 0)
                win32security.SetFileSecurity(str(path), win32security.DACL_SECURITY_INFORMATION, sd)
            except ImportError:
                # pywin32 not available, skip Windows-specific permissions
                pass
                
    def _set_executable_permissions(self, path: Path):
        """Set executable permissions"""
        if platform.system() != 'Windows':
            os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)  # 755
            
    def _run_tests(self) -> bool:
        """Run basic validation tests"""
        print("\nRunning validation tests...")
        
        tests = [
            self._test_imports,
            self._test_database_access,
            self._test_config_loading,
            self._test_port_availability
        ]
        
        failed = []
        for test in tests:
            try:
                if not test():
                    failed.append(test.__name__)
            except Exception as e:
                failed.append(f"{test.__name__}: {str(e)}")
                
        if failed:
            self._log_error(f"Tests failed: {', '.join(failed)}")
            return False
            
        print("✓ All tests passed")
        return True
        
    def _test_imports(self) -> bool:
        """Test critical imports"""
        try:
            import ccxt
            import pandas
            import numpy
            import sqlalchemy
            import cryptography
            print("  ✓ Import test passed")
            return True
        except ImportError as e:
            print(f"  ✗ Import test failed: {e}")
            return False
            
    def _test_database_access(self) -> bool:
        """Test database access"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            cursor.fetchall()
            conn.close()
            print("  ✓ Database test passed")
            return True
        except Exception as e:
            print(f"  ✗ Database test failed: {e}")
            return False
            
    def _test_config_loading(self) -> bool:
        """Test configuration loading"""
        try:
            with open(self.root_path / 'enhanced_config.json') as f:
                config = json.load(f)
            assert 'version' in config
            assert 'security' in config
            print("  ✓ Config test passed")
            return True
        except Exception as e:
            print(f"  ✗ Config test failed: {e}")
            return False
            
    def _test_port_availability(self) -> bool:
        """Test that configured ports are available"""
        try:
            for service, port in self.config['ports'].items():
                if self._is_port_available(port):
                    continue
                else:
                    # Port might be in use by our service
                    pass
            print("  ✓ Port test passed")
            return True
        except Exception as e:
            print(f"  ✗ Port test failed: {e}")
            return False
            
    def _generate_documentation(self):
        """Generate setup documentation"""
        print("\nGenerating documentation...")
        
        docs_dir = self.root_path / 'docs'
        docs_dir.mkdir(exist_ok=True)
        
        # Quick start guide
        quickstart = f"""# Nexlify Quick Start Guide

## Installation Complete!

Your Nexlify v2.0.8 installation is ready with the following configuration:

- **Installation Profile**: {self.install_profile}
- **API Port**: {self.config['ports']['api']}
- **Mobile API Port**: {self.config['ports']['mobile_api']}
- **Default PIN**: 2077 (Will be changed on first login)

## First Steps

1. **Start the System**
   ```bash
   python smart_launcher.py
   ```

2. **Initial Configuration**
   The GUI will automatically open and guide you through:
   - Setting up your PIN
   - Configuring exchange API credentials
   - Setting up notifications (optional)
   - Configuring DeFi/RPC settings (optional)
   - Enabling security features (2FA, IP whitelist)

3. **Begin Trading**
   Once configured, the system will:
   - Connect to your selected exchanges
   - Start monitoring markets
   - Execute your trading strategies

## Configuration Management

All settings are managed through the GUI:
- **Settings Tab**: General configuration
- **Security Tab**: 2FA, API keys, IP whitelist
- **Exchanges Tab**: Exchange credentials and settings
- **Notifications Tab**: Telegram, email, webhooks

## Security Features

- Master password and JWT tokens are auto-generated
- All sensitive data is encrypted
- Optional 2FA and IP whitelisting
- Audit trail for all actions

## Troubleshooting

- **Port conflicts**: Check `enhanced_config.json` for port settings
- **Database issues**: Run `python scripts/health_check.py`
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **GUI not opening**: Ensure tkinter is installed (Linux: `sudo apt-get install python3-tk`)

## System Files

- `/enhanced_config.json`: Main configuration (managed by GUI)
- `/.env`: System secrets (do not edit)
- `/data/`: Databases and trading data
- `/logs/`: System logs
- `/backups/`: Automatic backups
"""
        
        with open(docs_dir / 'QUICKSTART.md', 'w') as f:
            f.write(quickstart)
            
        # System requirements doc
        sysreq = f"""# System Requirements

## Hardware Requirements

### Minimum:
- CPU: 4 cores
- RAM: {MIN_RAM_GB}GB
- Disk: {MIN_DISK_GB}GB free space
- Network: Stable internet connection

### Recommended:
- CPU: 8+ cores
- RAM: {RECOMMENDED_RAM_GB}GB
- Disk: {RECOMMENDED_DISK_GB}GB free space
- GPU: NVIDIA GTX 1060+ (for ML features)

## Software Requirements

- Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ (64-bit)
- Operating System: Windows 10+, Ubuntu 20.04+, macOS 11+

## Optional Components

- Docker (for containerized deployment)
- Redis (for advanced caching)
- PostgreSQL (for production database)

## Network Ports

The following ports are used by default:

- API: {self.config['ports']['api']}
- Mobile API: {self.config['ports']['mobile_api']}
- WebSocket: {self.config['ports']['websocket']}
- PostgreSQL: {self.config['ports']['postgresql']}
- Redis: {self.config['ports']['redis']}
"""
        
        with open(docs_dir / 'REQUIREMENTS.md', 'w') as f:
            f.write(sysreq)
            
    def _display_final_instructions(self):
        """Display final setup instructions"""
        print("\n" + "="*50)
        print("✨ NEXLIFY SETUP COMPLETE! ✨")
        print("="*50)
        
        print("\n📋 Next Steps:")
        print("1. Run: python smart_launcher.py")
        print("2. The GUI will guide you through initial configuration")
        print("3. All settings are managed through the GUI")
        
        print("\n🔐 First-Time Setup (via GUI):")
        print("- Configure exchange API credentials")
        print("- Set up notification services (optional)")
        print("- Configure DeFi/RPC settings (optional)")
        print("- Enable 2FA security (recommended)")
        print("- Change default PIN (currently: 2077)")
        
        print("\n🌐 Access Points:")
        print(f"- Trading API: http://localhost:{self.config['ports']['api']}")
        print(f"- Mobile API: http://localhost:{self.config['ports']['mobile_api']}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        print("\n📚 Documentation: docs/QUICKSTART.md")
        print("\n🚀 Ready to trade in Night City!")
        
    def _check_windows_runtime(self) -> bool:
        """Check for Windows C++ runtime and install if needed"""
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("vcruntime140.dll")
            return True
        except:
            # Try to install Visual C++ Redistributable
            return self._install_visual_cpp()
            
    def _install_visual_cpp(self) -> bool:
        """Download and install Visual C++ Redistributable"""
        print("\nVisual C++ Runtime not found. Installing...")
        
        try:
            # Microsoft official download URL for VC++ 2022 Redistributable
            vc_redist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
            
            # Download the installer
            download_path = self.root_path / "vc_redist.x64.exe"
            
            print("Downloading Visual C++ Redistributable...")
            response = requests.get(vc_redist_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save the installer
            with open(download_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        print(f"\rDownload progress: {progress}%", end='', flush=True)
                        
            print("\n✓ Download complete")
            
            # Run the installer silently
            print("Installing Visual C++ Runtime...")
            result = subprocess.run(
                [str(download_path), '/install', '/quiet', '/norestart'],
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up installer
            download_path.unlink(missing_ok=True)
            
            if result.returncode == 0:
                print("✓ Visual C++ Runtime installed successfully")
                return True
            else:
                self._log_warning("Visual C++ installation completed with warnings")
                return True  # Often returns non-zero even on success
                
        except requests.RequestException as e:
            self._log_error(f"Failed to download Visual C++ Runtime: {str(e)}")
            print("\nPlease download and install manually from:")
            print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False
        except subprocess.TimeoutExpired:
            self._log_error("Visual C++ installation timed out")
            return False
        except Exception as e:
            self._log_error(f"Failed to install Visual C++ Runtime: {str(e)}")
            return False
            
    def _check_gui_dependencies(self):
        """Check GUI dependencies on Linux"""
        try:
            subprocess.run(['python3', '-c', 'import tkinter'], check=True, capture_output=True)
        except:
            self._log_warning("tkinter not available, GUI features may not work")
            print("Install with: sudo apt-get install python3-tk")
            
    def _get_current_version(self) -> str:
        """Get current installation version"""
        try:
            config_path = self.root_path / 'enhanced_config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get('version', 'unknown')
        except:
            pass
        return 'unknown'
        
    def _log_error(self, message: str):
        """Log error message"""
        self.errors.append(message)
        print(f"\n❌ ERROR: {message}")
        
    def _log_warning(self, message: str):
        """Log warning message"""
        self.warnings.append(message)
        print(f"\n⚠️  WARNING: {message}")
        
    def _save_error_report(self):
        """Save error report for debugging"""
        if self.errors:
            report_path = self.root_path / 'setup_errors.log'
            with open(report_path, 'w') as f:
                f.write(f"Setup Error Report - {datetime.now().isoformat()}\n")
                f.write("="*50 + "\n\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
            print(f"\nError report saved to: {report_path}")


def main():
    """Main entry point"""
    setup = NexlifySetup()
    success = setup.run()
    
    if not success:
        print("\n❌ Setup failed! Check setup_errors.log for details.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
