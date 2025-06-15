#!/usr/bin/env python3
"""
Nexlify Smart Launcher v2.0.8
Enhanced launcher with comprehensive checks, health monitoring, and failover support
"""

import sys
import os
import subprocess
import json
import time
import psutil
import asyncio
import signal
import socket
import logging
import shutil
import platform
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import hashlib
import fcntl

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback color definitions
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = ''
    class Style:
        BRIGHT = RESET_ALL = ''

@dataclass
class ComponentInfo:
    """Information about a system component"""
    name: str
    module: str
    dependencies: List[str]
    process: Optional[subprocess.Popen] = None
    port: Optional[int] = None
    health_check: Optional[callable] = None
    critical: bool = True
    startup_timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3

class HealthMonitor:
    """Monitors component health"""
    
    def __init__(self, launcher):
        self.launcher = launcher
        self.running = False
        self.health_thread = None
        self.check_interval = 30  # seconds
        
    def start(self):
        """Start health monitoring"""
        self.running = True
        self.health_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.health_thread.start()
        
    def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.health_thread:
            self.health_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check all components
                for component in self.launcher.components.values():
                    if component.process and component.health_check:
                        if not component.health_check():
                            self.launcher.logger.warning(
                                f"Health check failed for {component.name}"
                            )
                            # Attempt recovery
                            if component.critical and component.retry_count < component.max_retries:
                                self.launcher._restart_component(component)
                                
                # Check system resources
                self._check_system_resources()
                
            except Exception as e:
                self.launcher.logger.error(f"Health monitor error: {e}")
                
            time.sleep(self.check_interval)
            
    def _check_system_resources(self):
        """Check system resource usage"""
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.launcher.logger.warning(f"High memory usage: {memory.percent}%")
            
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.launcher.logger.warning(f"High CPU usage: {cpu_percent}%")
            
        # Disk check
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                if usage.percent > 95:
                    self.launcher.logger.warning(
                        f"Low disk space on {partition.mountpoint}: {usage.percent}%"
                    )
            except:
                pass

class DynamicResourceAllocator:
    """Dynamically allocates resources based on system capabilities"""
    
    @staticmethod
    def get_optimal_settings() -> Dict[str, Any]:
        """Calculate optimal settings based on system resources"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base settings
        settings = {
            'num_workers': min(cpu_count - 1, 8),  # Leave one core for system
            'cache_size_mb': min(int(memory_gb * 100), 2000),  # ~10% of RAM for cache
            'enable_gpu': False,
            'parallel_strategies': True,
            'max_concurrent_trades': 10,
            'api_rate_limit': 60,
        }
        
        # Adjust based on system specs
        if cpu_count >= 8 and memory_gb >= 16:
            # High-end system
            settings['num_workers'] = min(cpu_count - 2, 16)
            settings['cache_size_mb'] = min(int(memory_gb * 200), 4000)
            settings['max_concurrent_trades'] = 20
            settings['api_rate_limit'] = 120
            
        elif cpu_count <= 2 or memory_gb <= 4:
            # Low-end system
            settings['num_workers'] = 1
            settings['cache_size_mb'] = 200
            settings['max_concurrent_trades'] = 5
            settings['parallel_strategies'] = False
            settings['api_rate_limit'] = 30
            
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                settings['enable_gpu'] = True
        except:
            pass
            
        return settings

class NexlifyLauncher:
    def __init__(self):
        # Determine project root more reliably
        self.project_root = self._find_project_root()
        os.chdir(self.project_root)
        
        # Setup logging first
        self._setup_logging()
        
        # Configuration
        self.config_path = Path("config/enhanced_config.json")
        self.old_config_path = Path("config/neural_config.json")
        self.config = None
        
        # Component registry with dependency graph
        self.components = self._build_component_registry()
        
        # Processes
        self.processes = {}
        
        # Health monitor
        self.health_monitor = HealthMonitor(self)
        
        # Resource allocator
        self.resource_allocator = DynamicResourceAllocator()
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _find_project_root(self) -> Path:
        """Find project root directory reliably"""
        # Start from script location
        current = Path(__file__).resolve().parent
        
        # Look for key project indicators
        indicators = ['config', 'src', 'requirements.txt', '.git']
        
        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent
            
        # Fallback to script parent directory
        return Path(__file__).resolve().parent
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure launcher log
        self.logger = logging.getLogger('NexlifyLauncher')
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_dir / 'launcher.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows-specific
        if sys.platform == 'win32':
            signal.signal(signal.SIGBREAK, signal_handler)
            
    def _build_component_registry(self) -> Dict[str, ComponentInfo]:
        """Build component dependency graph"""
        return {
            'config_migration': ComponentInfo(
                name='Config Migration',
                module='scripts.migrate_config',
                dependencies=[],
                critical=True,
                startup_timeout=10
            ),
            'error_handler': ComponentInfo(
                name='Error Handler',
                module='src.core.error_handler',
                dependencies=['config_migration'],
                critical=True,
                startup_timeout=5
            ),
            'database': ComponentInfo(
                name='Database',
                module='scripts.setup_database',
                dependencies=['config_migration'],
                critical=True,
                startup_timeout=10
            ),
            'security': ComponentInfo(
                name='Security Manager',
                module='src.security.nexlify_advanced_security',
                dependencies=['database', 'error_handler'],
                critical=True,
                startup_timeout=10
            ),
            'audit': ComponentInfo(
                name='Audit Trail',
                module='src.security.nexlify_audit_trail',
                dependencies=['database', 'security'],
                critical=True,
                startup_timeout=10
            ),
            'neural_net': ComponentInfo(
                name='Neural Network API',
                module='src.core.arasaka_neural_net',
                dependencies=['security', 'audit', 'error_handler'],
                port=8000,
                critical=True,
                startup_timeout=30,
                health_check=lambda: self._check_api_health()
            ),
            'multi_strategy': ComponentInfo(
                name='Multi-Strategy Optimizer',
                module='src.trading.nexlify_multi_strategy',
                dependencies=['neural_net'],
                critical=False,
                startup_timeout=20
            ),
            'predictive': ComponentInfo(
                name='Predictive Engine',
                module='src.predictive.nexlify_predictive_features',
                dependencies=['neural_net'],
                critical=False,
                startup_timeout=20
            ),
            'dex': ComponentInfo(
                name='DEX Integration',
                module='src.trading.nexlify_dex_integration',
                dependencies=['neural_net'],
                critical=False,
                startup_timeout=15
            ),
            'mobile_api': ComponentInfo(
                name='Mobile API',
                module='src.api.nexlify_mobile_api',
                dependencies=['neural_net', 'security'],
                port=8001,
                critical=False,
                startup_timeout=15,
                health_check=lambda: self._check_port(8001)
            ),
            'gui': ComponentInfo(
                name='GUI',
                module='src.gui.nexlify_enhanced_gui',
                dependencies=['neural_net', 'security', 'mobile_api'],
                critical=False,
                startup_timeout=20
            )
        }
        
    def print_banner(self):
        """Print startup banner"""
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     ███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗  ║
║     ████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝  ║
║     ██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝   ║
║     ██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝    ║
║     ██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║     ║
║     ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝     ║
║                                                                ║
║            NEURAL TRADING SYSTEM v2.0.8                        ║
║            Enhanced Security & Performance                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
        """
        print(banner)
        
    def check_requirements(self) -> bool:
        """Check all system requirements"""
        print(f"\n{Fore.YELLOW}🔍 Checking system requirements...{Style.RESET_ALL}")
        
        # Python version check
        python_version = sys.version_info
        required_version = (3, 9)  # Minimum 3.9
        
        # Check for Docker environment
        if os.path.exists('/.dockerenv'):
            # In Docker, we might need 3.11
            if 'python:3.11' in os.environ.get('DOCKER_IMAGE', ''):
                required_version = (3, 11)
                
        if python_version < required_version:
            print(f"{Fore.RED}❌ Python {required_version[0]}.{required_version[1]}+ required, found {python_version.major}.{python_version.minor}{Style.RESET_ALL}")
            return False
            
        print(f"{Fore.GREEN}✅ Python {python_version.major}.{python_version.minor} OK{Style.RESET_ALL}")
        
        # Check system resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        if memory.total < 2 * 1024**3:  # 2GB minimum
            print(f"{Fore.RED}❌ Insufficient RAM: {memory.total / 1024**3:.1f}GB (2GB minimum){Style.RESET_ALL}")
            return False
            
        if disk.free < 5 * 1024**3:  # 5GB free space
            print(f"{Fore.RED}❌ Insufficient disk space: {disk.free / 1024**3:.1f}GB free (5GB minimum){Style.RESET_ALL}")
            return False
            
        print(f"{Fore.GREEN}✅ System resources OK - RAM: {memory.total / 1024**3:.1f}GB, Disk: {disk.free / 1024**3:.1f}GB free{Style.RESET_ALL}")
        
        # Check required packages with versions
        required_packages = {
            'ccxt': '4.1.22',
            'pandas': '2.1.3',
            'numpy': '1.26.2',
            'aiohttp': '3.9.1',
            'psutil': '5.9.6',
            'cryptography': '41.0.7',
            'sqlalchemy': '2.0.23',
            'pydantic': '2.5.2',
            'sklearn': '0.0',  # scikit-learn
            'torch': None,  # Optional, version varies
        }
        
        missing = []
        version_mismatches = []
        
        for package, required_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                
                # Check version if specified
                if required_version and hasattr(module, '__version__'):
                    installed_version = module.__version__
                    if not self._check_version_compatibility(installed_version, required_version):
                        version_mismatches.append(f"{package} (need {required_version}, have {installed_version})")
                        
            except ImportError:
                if package == 'torch':  # Optional
                    print(f"{Fore.YELLOW}⚠️  PyTorch not installed (optional for GPU support){Style.RESET_ALL}")
                else:
                    missing.append(package)
                    
        # Check for ccxt exchange support
        try:
            import ccxt
            exchanges = ['binance', 'kraken', 'coinbase', 'kucoin']
            unsupported = []
            
            for exchange in exchanges:
                if not hasattr(ccxt, exchange):
                    unsupported.append(exchange)
                    
            if unsupported:
                print(f"{Fore.YELLOW}⚠️  Unsupported exchanges: {', '.join(unsupported)}{Style.RESET_ALL}")
                
        except:
            pass
            
        # Check numpy BLAS/LAPACK
        try:
            import numpy as np
            config = np.show_config(mode='dicts')
            if 'blas' not in str(config).lower():
                print(f"{Fore.YELLOW}⚠️  NumPy compiled without BLAS optimization{Style.RESET_ALL}")
        except:
            pass
            
        if missing:
            print(f"{Fore.RED}❌ Missing packages: {', '.join(missing)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Run: pip install -r requirements.txt{Style.RESET_ALL}")
            return False
            
        if version_mismatches:
            print(f"{Fore.YELLOW}⚠️  Version mismatches: {', '.join(version_mismatches)}{Style.RESET_ALL}")
            
        print(f"{Fore.GREEN}✅ All required packages installed{Style.RESET_ALL}")
        
        # Platform-specific checks
        if platform.system() == 'Windows':
            # Check for Visual C++ runtime
            if not self._check_windows_runtime():
                print(f"{Fore.YELLOW}⚠️  Visual C++ runtime may be missing{Style.RESET_ALL}")
                
        # Check GUI dependencies
        if not self._check_tkinter():
            print(f"{Fore.YELLOW}⚠️  Tkinter not available - GUI will not work{Style.RESET_ALL}")
            print(f"On Linux: sudo apt-get install python3-tk")
            
        return True
        
    def _check_version_compatibility(self, installed: str, required: str) -> bool:
        """Check if installed version meets requirements"""
        def parse_version(v):
            return tuple(map(int, v.split('.')[:3]))
            
        try:
            installed_tuple = parse_version(installed)
            required_tuple = parse_version(required)
            return installed_tuple >= required_tuple
        except:
            return True  # Assume OK if can't parse
            
    def _check_windows_runtime(self) -> bool:
        """Check for Windows Visual C++ runtime"""
        try:
            import ctypes
            # Try to load a common MSVC runtime DLL
            ctypes.cdll.LoadLibrary("msvcp140.dll")
            return True
        except:
            return False
            
    def _check_tkinter(self) -> bool:
        """Check if tkinter is available"""
        try:
            import tkinter
            return True
        except ImportError:
            return False
            
    def check_emergency_stop(self) -> bool:
        """Check for emergency stop with integrity validation"""
        stop_file = Path("EMERGENCY_STOP_ACTIVE")
        
        if stop_file.exists():
            try:
                # Validate file integrity
                content = stop_file.read_text()
                
                # Check if content is valid (contains timestamp)
                if "triggered at" in content:
                    # Parse timestamp
                    parts = content.split("triggered at ")
                    if len(parts) > 1:
                        timestamp_str = parts[1].split(" due to")[0]
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.strip())
                            age = datetime.now() - timestamp
                            
                            print(f"\n{Fore.RED}🛑 EMERGENCY STOP ACTIVE{Style.RESET_ALL}")
                            print(f"Triggered: {timestamp_str}")
                            print(f"Age: {age}")
                            
                            # Auto-clear if older than 24 hours
                            if age > timedelta(hours=24):
                                print(f"{Fore.YELLOW}Emergency stop is older than 24 hours. Auto-clearing...{Style.RESET_ALL}")
                                stop_file.unlink()
                                return False
                                
                        except:
                            pass
                            
                    return True
                else:
                    # Invalid content, might be corrupted
                    print(f"{Fore.YELLOW}⚠️  Emergency stop file appears corrupted. Removing...{Style.RESET_ALL}")
                    stop_file.unlink()
                    
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  Error reading emergency stop file: {e}{Style.RESET_ALL}")
                
        return False
        
    def check_directories(self) -> bool:
        """Create required directories with proper permissions"""
        print(f"\n{Fore.YELLOW}📁 Checking directories...{Style.RESET_ALL}")
        
        directories = [
            "config",
            "data",
            "data/market",
            "data/models", 
            "logs",
            "logs/crash_reports",
            "logs/errors",
            "logs/audit",
            "logs/performance",
            "logs/mobile",
            "models",
            "backups",
            "backups/config",
            "backups/database",
            "backups/models",
            "reports",
            "reports/compliance",
            "reports/performance",
            "reports/tax",
            "assets",
            "assets/sounds",
            "assets/images"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Set appropriate permissions
                if platform.system() != 'Windows':
                    # Unix-like systems
                    if 'config' in str(dir_path) or 'backups' in str(dir_path):
                        # Restrictive permissions for sensitive directories
                        os.chmod(dir_path, 0o700)
                    else:
                        # Standard permissions
                        os.chmod(dir_path, 0o755)
                        
                # Check if writable
                test_file = dir_path / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except:
                    print(f"{Fore.RED}❌ Directory not writable: {directory}{Style.RESET_ALL}")
                    return False
                    
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to create directory {directory}: {e}{Style.RESET_ALL}")
                return False
                
        print(f"{Fore.GREEN}✅ All directories ready{Style.RESET_ALL}")
        return True
        
    def check_config(self) -> bool:
        """Check and load configuration"""
        print(f"\n{Fore.YELLOW}⚙️  Checking configuration...{Style.RESET_ALL}")
        
        # First check if we need to migrate
        if not self.config_path.exists() and self.old_config_path.exists():
            print(f"{Fore.YELLOW}📝 Old configuration found. Running migration...{Style.RESET_ALL}")
            
            # Run migration
            try:
                from scripts.migrate_config import ConfigMigrator
                migrator = ConfigMigrator()
                if not migrator.migrate():
                    print(f"{Fore.RED}❌ Configuration migration failed{Style.RESET_ALL}")
                    return False
            except Exception as e:
                print(f"{Fore.RED}❌ Migration error: {e}{Style.RESET_ALL}")
                return False
                
        # Check for enhanced config
        if not self.config_path.exists():
            print(f"{Fore.RED}❌ Configuration file not found: {self.config_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Creating default configuration...{Style.RESET_ALL}")
            
            # Create default config
            try:
                from scripts.migrate_config import ConfigMigrator
                migrator = ConfigMigrator()
                migrator._create_default_config()
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to create default config: {e}{Style.RESET_ALL}")
                return False
                
        # Load and validate config
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Validate structure
            required_sections = [
                'version', 'environment', 'security', 'trading',
                'exchanges', 'notifications', 'api', 'database'
            ]
            
            missing = [s for s in required_sections if s not in self.config]
            if missing:
                print(f"{Fore.RED}❌ Missing config sections: {', '.join(missing)}{Style.RESET_ALL}")
                return False
                
            # Check version
            if self.config.get('version') != '2.0.8':
                print(f"{Fore.YELLOW}⚠️  Config version mismatch: {self.config.get('version')} != 2.0.8{Style.RESET_ALL}")
                
            # Validate database URL
            db_url = self.config.get('database', {}).get('url', '')
            if not db_url:
                print(f"{Fore.RED}❌ Database URL not configured{Style.RESET_ALL}")
                return False
                
            # Check file permissions
            if platform.system() != 'Windows':
                stat_info = os.stat(self.config_path)
                if stat_info.st_mode & 0o077:
                    print(f"{Fore.YELLOW}⚠️  Config file has loose permissions. Fixing...{Style.RESET_ALL}")
                    os.chmod(self.config_path, 0o600)
                    
            print(f"{Fore.GREEN}✅ Configuration loaded (v{self.config.get('version')}){Style.RESET_ALL}")
            
            # Apply dynamic resource allocation
            optimal_settings = self.resource_allocator.get_optimal_settings()
            print(f"{Fore.CYAN}📊 System resources: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🔧 Optimal settings: {optimal_settings['num_workers']} workers, {optimal_settings['cache_size_mb']}MB cache{Style.RESET_ALL}")
            
            # Update config with optimal settings
            if 'performance' not in self.config:
                self.config['performance'] = {}
            self.config['performance'].update(optimal_settings)
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}❌ Invalid JSON in config file: {e}{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading config: {e}{Style.RESET_ALL}")
            return False
            
    def check_ports(self) -> bool:
        """Check if required ports are available"""
        print(f"\n{Fore.YELLOW}🔌 Checking ports...{Style.RESET_ALL}")
        
        required_ports = {
            self.config.get('api', {}).get('port', 8000): 'Neural Net API',
            8001: 'Mobile API',
            5432: 'PostgreSQL (if used)',
            6379: 'Redis (if used)'
        }
        
        blocked_ports = []
        
        for port, service in required_ports.items():
            if self._is_port_in_use(port):
                blocked_ports.append(f"{port} ({service})")
                
        if blocked_ports:
            print(f"{Fore.RED}❌ Ports in use: {', '.join(blocked_ports)}{Style.RESET_ALL}")
            
            # Try alternative ports for API
            if self._is_port_in_use(8000):
                alt_port = self._find_free_port(8000, 8100)
                if alt_port:
                    print(f"{Fore.YELLOW}📡 Using alternative API port: {alt_port}{Style.RESET_ALL}")
                    self.config['api']['port'] = alt_port
                    # Update component
                    self.components['neural_net'].port = alt_port
                else:
                    return False
                    
        print(f"{Fore.GREEN}✅ Required ports available{Style.RESET_ALL}")
        return True
        
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return False
            except:
                return True
                
    def _find_free_port(self, start: int, end: int) -> Optional[int]:
        """Find a free port in range"""
        for port in range(start, end):
            if not self._is_port_in_use(port):
                return port
        return None
        
    def check_database(self) -> bool:
        """Initialize and check database"""
        print(f"\n{Fore.YELLOW}🗄️  Checking database...{Style.RESET_ALL}")
        
        db_url = self.config.get('database', {}).get('url', '')
        
        if db_url.startswith('sqlite:///'):
            # SQLite database
            db_path = Path(db_url.replace('sqlite:///', ''))
            
            if not db_path.exists():
                print(f"{Fore.YELLOW}Creating database: {db_path}{Style.RESET_ALL}")
                
                # Run database setup
                try:
                    # Import and run setup
                    setup_script = Path("scripts/setup_database.py")
                    if setup_script.exists():
                        subprocess.run([sys.executable, str(setup_script)], check=True)
                    else:
                        # Create basic database
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        db_path.touch()
                        
                    print(f"{Fore.GREEN}✅ Database initialized{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ Database setup failed: {e}{Style.RESET_ALL}")
                    return False
                    
            else:
                # Check if writable
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    conn.execute("SELECT 1")
                    conn.close()
                    print(f"{Fore.GREEN}✅ Database accessible{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ Database error: {e}{Style.RESET_ALL}")
                    return False
                    
        elif db_url.startswith('postgresql://'):
            # PostgreSQL
            print(f"{Fore.YELLOW}⚠️  PostgreSQL database configured. Ensure it's running.{Style.RESET_ALL}")
            
        return True
        
    def check_external_services(self) -> bool:
        """Check external service connectivity"""
        print(f"\n{Fore.YELLOW}🌐 Checking external services...{Style.RESET_ALL}")
        
        # Check Redis if configured
        if self.config.get('performance', {}).get('cache_enabled'):
            if self._is_port_in_use(6379):
                print(f"{Fore.GREEN}✅ Redis appears to be running{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  Redis not detected. Caching will use memory.{Style.RESET_ALL}")
                
        # Check Docker if using containers
        if Path('/.dockerenv').exists():
            print(f"{Fore.GREEN}✅ Running in Docker container{Style.RESET_ALL}")
            
        return True
        
    def start_component(self, component_name: str) -> bool:
        """Start a single component with dependency resolution"""
        component = self.components.get(component_name)
        if not component:
            self.logger.error(f"Unknown component: {component_name}")
            return False
            
        # Check dependencies first
        for dep in component.dependencies:
            if dep not in self.processes or not self._is_component_running(dep):
                self.logger.info(f"Starting dependency: {dep}")
                if not self.start_component(dep):
                    return False
                    
        # Component-specific startup
        if component_name == 'config_migration':
            # Already handled in check_config
            self.processes[component_name] = True
            return True
            
        elif component_name == 'error_handler':
            # Initialize error handler
            try:
                from src.core.error_handler import get_error_handler
                handler = get_error_handler()
                self.processes[component_name] = handler
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize error handler: {e}")
                return False
                
        elif component_name == 'database':
            # Already handled in check_database
            self.processes[component_name] = True
            return True
            
        elif component_name == 'security':
            # Initialize security manager
            try:
                # Security manager will be initialized by other components
                self.processes[component_name] = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize security: {e}")
                return False
                
        elif component_name == 'audit':
            # Initialize audit trail
            try:
                # Audit trail will be initialized by other components
                self.processes[component_name] = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize audit: {e}")
                return False
                
        elif component_name == 'neural_net':
            return self.start_neural_net()
            
        elif component_name == 'gui':
            return self.start_gui()
            
        elif component_name == 'mobile_api':
            return self.start_mobile_api()
            
        else:
            # Generic module startup
            self.logger.info(f"Initializing {component.name}...")
            self.processes[component_name] = True
            return True
            
    def start_neural_net(self) -> bool:
        """Start the neural network API with enhanced checks"""
        print(f"\n{Fore.YELLOW}🧠 Starting Neural Network API...{Style.RESET_ALL}")
        
        # Check if already running
        port = self.components['neural_net'].port
        if self._is_port_in_use(port):
            print(f"{Fore.YELLOW}⚠️  API port {port} already in use{Style.RESET_ALL}")
            
            # Check if it's our API
            if self._check_api_health():
                print(f"{Fore.GREEN}✅ API already running{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ Port {port} blocked by another service{Style.RESET_ALL}")
                return False
                
        try:
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Add config path
            env['NEXLIFY_CONFIG'] = str(self.config_path)
            
            # Start process with proper output handling
            process = subprocess.Popen(
                [sys.executable, "-m", "src.core.arasaka_neural_net"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            self.components['neural_net'].process = process
            self.processes['neural_net'] = process
            
            # Monitor startup
            start_time = time.time()
            timeout = self.components['neural_net'].startup_timeout
            
            # Start output monitoring threads
            self._start_output_monitor(process, "Neural Net")
            
            while time.time() - start_time < timeout:
                # Check if process is still running
                if process.poll() is not None:
                    # Process terminated
                    print(f"{Fore.RED}❌ Neural Net API crashed during startup{Style.RESET_ALL}")
                    return False
                    
                # Check if API is responding
                if self._check_api_health():
                    print(f"{Fore.GREEN}✅ Neural Net API started successfully{Style.RESET_ALL}")
                    return True
                    
                time.sleep(0.5)
                
            print(f"{Fore.RED}❌ Neural Net API startup timeout{Style.RESET_ALL}")
            return False
            
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start Neural Net API: {e}{Style.RESET_ALL}")
            self.logger.error(f"Neural Net startup error: {e}", exc_info=True)
            return False
            
    def start_gui(self) -> bool:
        """Start the GUI with display checks"""
        print(f"\n{Fore.YELLOW}🖥️  Starting GUI...{Style.RESET_ALL}")
        
        # Check display availability
        if platform.system() != 'Windows':
            if not os.environ.get('DISPLAY'):
                print(f"{Fore.RED}❌ No display detected (DISPLAY not set){Style.RESET_ALL}")
                print(f"{Fore.YELLOW}GUI cannot run in headless environment{Style.RESET_ALL}")
                return False
                
        try:
            # Test tkinter
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            root.destroy()
        except Exception as e:
            print(f"{Fore.RED}❌ GUI initialization failed: {e}{Style.RESET_ALL}")
            return False
            
        try:
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['NEXLIFY_CONFIG'] = str(self.config_path)
            
            # Start GUI process
            process = subprocess.Popen(
                [sys.executable, "-m", "src.gui.nexlify_enhanced_gui"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.components['gui'].process = process
            self.processes['gui'] = process
            
            # Monitor startup
            start_time = time.time()
            timeout = self.components['gui'].startup_timeout
            
            # Start output monitoring
            self._start_output_monitor(process, "GUI")
            
            # Give GUI time to initialize
            time.sleep(2)
            
            # Check if still running
            if process.poll() is None:
                print(f"{Fore.GREEN}✅ GUI started successfully{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ GUI crashed during startup{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start GUI: {e}{Style.RESET_ALL}")
            self.logger.error(f"GUI startup error: {e}", exc_info=True)
            return False
            
    def start_mobile_api(self) -> bool:
        """Start mobile API service"""
        print(f"\n{Fore.YELLOW}📱 Starting Mobile API...{Style.RESET_ALL}")
        
        # Check port
        port = self.components['mobile_api'].port
        if self._is_port_in_use(port):
            print(f"{Fore.YELLOW}⚠️  Mobile API port {port} already in use{Style.RESET_ALL}")
            return False
            
        try:
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['NEXLIFY_CONFIG'] = str(self.config_path)
            
            # Start process
            process = subprocess.Popen(
                [sys.executable, "-m", "src.api.nexlify_mobile_api"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.components['mobile_api'].process = process
            self.processes['mobile_api'] = process
            
            # Monitor startup
            self._start_output_monitor(process, "Mobile API")
            
            # Wait for startup
            time.sleep(3)
            
            if process.poll() is None and self._check_port(port):
                print(f"{Fore.GREEN}✅ Mobile API started on port {port}{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ Mobile API failed to start{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start Mobile API: {e}{Style.RESET_ALL}")
            return False
            
    def _start_output_monitor(self, process: subprocess.Popen, name: str):
        """Start threads to monitor process output"""
        def monitor_output(pipe, prefix):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        self.logger.info(f"[{prefix}] {line.strip()}")
            except:
                pass
                
        # Start monitoring threads
        threading.Thread(
            target=monitor_output,
            args=(process.stdout, f"{name}-OUT"),
            daemon=True
        ).start()
        
        threading.Thread(
            target=monitor_output,
            args=(process.stderr, f"{name}-ERR"),
            daemon=True
        ).start()
        
    def _check_api_health(self) -> bool:
        """Check if API is healthy"""
        port = self.components['neural_net'].port
        try:
            import urllib.request
            response = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return response.status == 200
        except:
            return False
            
    def _check_port(self, port: int) -> bool:
        """Check if service is listening on port"""
        return self._is_port_in_use(port)
        
    def _is_component_running(self, component_name: str) -> bool:
        """Check if component is running"""
        if component_name not in self.processes:
            return False
            
        process = self.processes.get(component_name)
        
        if isinstance(process, subprocess.Popen):
            return process.poll() is None
        else:
            return bool(process)
            
    def _restart_component(self, component: ComponentInfo):
        """Restart a failed component"""
        self.logger.info(f"Attempting to restart {component.name}")
        
        # Stop if running
        if component.process and component.process.poll() is None:
            component.process.terminate()
            component.process.wait(timeout=5)
            
        # Increment retry count
        component.retry_count += 1
        
        # Restart
        self.start_component(component.name)
        
    def cleanup(self):
        """Cleanup and shutdown all components"""
        print(f"\n{Fore.YELLOW}🛑 Shutting down...{Style.RESET_ALL}")
        
        # Stop health monitor
        self.health_monitor.stop()
        
        # Notify components of shutdown
        for name, component in reversed(list(self.components.items())):
            if component.process and isinstance(component.process, subprocess.Popen):
                if component.process.poll() is None:
                    print(f"Stopping {component.name}...")
                    
                    # Try graceful shutdown first
                    component.process.terminate()
                    
                    try:
                        component.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if needed
                        component.process.kill()
                        component.process.wait()
                        
                    # Log exit code
                    exit_code = component.process.returncode
                    self.logger.info(f"{component.name} exited with code {exit_code}")
                    
        print(f"{Fore.GREEN}✅ Shutdown complete{Style.RESET_ALL}")
        
    def run(self):
        """Main launcher sequence"""
        try:
            # Print banner
            self.print_banner()
            
            # System checks
            if not self.check_requirements():
                return 1
                
            if self.check_emergency_stop():
                print(f"\n{Fore.YELLOW}To clear emergency stop, delete EMERGENCY_STOP_ACTIVE file{Style.RESET_ALL}")
                return 1
                
            if not self.check_directories():
                return 1
                
            if not self.check_config():
                return 1
                
            if not self.check_ports():
                return 1
                
            if not self.check_database():
                return 1
                
            if not self.check_external_services():
                return 1
                
            # Pre-launch health check
            print(f"\n{Fore.YELLOW}🔍 Running pre-launch health check...{Style.RESET_ALL}")
            
            # Test exchange connectivity
            if self.config.get('exchanges', {}).get('testnet'):
                print(f"{Fore.CYAN}📡 Using TESTNET mode{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  Using MAINNET mode - real funds at risk!{Style.RESET_ALL}")
                
            # Start components in dependency order
            print(f"\n{Fore.YELLOW}🚀 Starting Nexlify components...{Style.RESET_ALL}")
            
            # Critical components
            critical_components = ['error_handler', 'database', 'security', 'audit', 'neural_net']
            
            for component_name in critical_components:
                if not self.start_component(component_name):
                    print(f"{Fore.RED}❌ Failed to start critical component: {component_name}{Style.RESET_ALL}")
                    self.cleanup()
                    return 1
                    
            # Optional components
            optional_components = ['multi_strategy', 'predictive', 'dex', 'mobile_api', 'gui']
            
            for component_name in optional_components:
                component = self.components[component_name]
                
                # Check if enabled in config
                if component_name == 'dex' and not self.config.get('dex_integration', {}).get('enabled'):
                    continue
                if component_name == 'mobile_api' and not self.config.get('mobile', {}).get('enabled'):
                    continue
                    
                if not self.start_component(component_name):
                    print(f"{Fore.YELLOW}⚠️  Failed to start optional component: {component_name}{Style.RESET_ALL}")
                    
            # Start health monitoring
            self.health_monitor.start()
            
            # Final instructions
            print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ NEXLIFY STARTED SUCCESSFULLY!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            
            api_port = self.config.get('api', {}).get('port', 8000)
            print(f"\n{Fore.CYAN}📡 API: http://localhost:{api_port}{Style.RESET_ALL}")
            
            if 'mobile_api' in self.processes:
                print(f"{Fore.CYAN}📱 Mobile API: http://localhost:8001{Style.RESET_ALL}")
                
            if 'gui' not in self.processes:
                print(f"\n{Fore.YELLOW}No GUI running. Use the API or start GUI manually.{Style.RESET_ALL}")
                
            # Security reminder
            if not self.config.get('security', {}).get('master_password_enabled'):
                print(f"\n{Fore.YELLOW}⚠️  Master password not enabled. Configure in GUI settings.{Style.RESET_ALL}")
                
            if not self.config.get('security', {}).get('2fa_enabled'):
                print(f"{Fore.YELLOW}⚠️  2FA not enabled. Configure in GUI settings for better security.{Style.RESET_ALL}")
                
            print(f"\n{Fore.CYAN}Press Ctrl+C to shutdown{Style.RESET_ALL}")
            
            # Keep running
            while True:
                time.sleep(1)
                
                # Check critical components
                for component_name in critical_components:
                    if not self._is_component_running(component_name):
                        self.logger.error(f"Critical component {component_name} has stopped!")
                        self.cleanup()
                        return 1
                        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Shutdown requested...{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Fatal error: {e}{Style.RESET_ALL}")
            self.logger.error(f"Fatal launcher error: {e}", exc_info=True)
            return 1
            
        finally:
            self.cleanup()
            
        return 0

def main():
    """Main entry point"""
    launcher = NexlifyLauncher()
    sys.exit(launcher.run())

if __name__ == "__main__":
    main()
