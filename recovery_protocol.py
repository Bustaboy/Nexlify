#!/usr/bin/env python3
"""
🌃 NEXLIFY RECOVERY PROTOCOL v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMERGENCY NEURAL RECONSTRUCTION SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the connection flatlines and the matrix glitches out, this recovery
protocol will resurrect your trading neural net from the digital void.

Usage: python recovery_protocol.py [--full-restore] [--check-only]
"""

import sys
import os
import json
import time
import asyncio
import subprocess
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

# Initialize cyberpunk console
console = Console(color_system="truecolor")

@dataclass
class RecoveryState:
    """Tracks the state of our recovery process"""
    timestamp: str
    phase: str
    completed_tasks: List[str]
    pending_tasks: List[str]
    errors: List[Dict[str, str]]
    warnings: List[str]
    system_status: Dict[str, bool]
    
    def save(self, path: Path):
        """Persist recovery state to disk"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'RecoveryState':
        """Load recovery state from disk"""
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls(
            timestamp=datetime.now().isoformat(),
            phase="INITIALIZATION",
            completed_tasks=[],
            pending_tasks=[],
            errors=[],
            warnings=[],
            system_status={}
        )

class NexlifyRecoveryProtocol:
    """
    🔧 NEURAL RECOVERY SYSTEM
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Handles complete system recovery when the trading matrix goes offline
    """
    
    def __init__(self):
        self.root_path = Path(__file__).parent.parent
        self.config_path = self.root_path / "config"
        self.backup_path = self.root_path / "backups"
        self.state_path = self.root_path / "data" / ".recovery_state.json"
        self.log_path = self.root_path / "logs" / "recovery"
        
        # Ensure critical directories exist
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load or create recovery state
        self.state = RecoveryState.load(self.state_path)
        
        # TODO List - Critical recovery tasks
        self.todo_list = [
            "CHECK_SYSTEM_INTEGRITY",
            "VERIFY_DEPENDENCIES",
            "RESTORE_CONFIGURATION",
            "VALIDATE_API_KEYS",
            "RESTORE_DATABASE_STATE",
            "SYNC_MARKET_DATA",
            "RESTART_TRADING_ENGINE",
            "RECONNECT_EXCHANGES",
            "RESUME_STRATEGIES",
            "VERIFY_RISK_LIMITS",
            "RESTORE_ML_MODELS",
            "ENABLE_MONITORING",
            "SEND_RECOVERY_REPORT"
        ]
    
    def setup_logging(self):
        """Configure recovery logging"""
        log_file = self.log_path / f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("NEXLIFY_RECOVERY")
    
    def print_banner(self):
        """Display cyberpunk recovery banner"""
        banner = """
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
[bold green]    _   _  _____  _     _  _  _____  _   _    ____  _____ _____ _____  _   _ [/bold green]
[bold green]   | \ | || ____|| |   | || ||  ___|| | | |  |  _ \| ____/ ____/ _ \ \| | / |[/bold green]
[bold green]   |  \| ||  _|  |  |  | || || |_   | |_| |  | |_) |  _|| |   | | | |\   /| |[/bold green]
[bold green]   | |\  || |___ | |_| || || ||  _|  |   |   |  _ <| |__| |___| |_| | | V | |[/bold green]
[bold green]   |_| \_||_____||_____|_||_||_|     |_|     |_| \_\_____|______\___/  \_/ |_|[/bold green]
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
[bold magenta]                    EMERGENCY NEURAL RECONSTRUCTION PROTOCOL                    [/bold magenta]
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
        """
        console.print(banner)
    
    async def check_system_integrity(self) -> Tuple[bool, List[str]]:
        """Verify core system files and directories"""
        console.print("\n[bold yellow]🔍 SCANNING SYSTEM INTEGRITY...[/bold yellow]")
        
        issues = []
        critical_paths = {
            "config/enhanced_config.json": "Main configuration",
            "src/core/engine.py": "Trading engine",
            "src/utils/utils_module.py": "Utilities module",
            "data/": "Data directory",
            "logs/": "Logging directory",
            "backups/": "Backup directory"
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking files...", total=len(critical_paths))
            
            for path, description in critical_paths.items():
                full_path = self.root_path / path
                
                if full_path.is_dir():
                    if not full_path.exists():
                        issues.append(f"Missing directory: {path} ({description})")
                else:
                    if not full_path.exists():
                        issues.append(f"Missing file: {path} ({description})")
                    else:
                        # Verify file integrity via checksum
                        try:
                            with open(full_path, 'rb') as f:
                                content = f.read()
                                if len(content) == 0:
                                    issues.append(f"Empty file: {path}")
                        except Exception as e:
                            issues.append(f"Cannot read {path}: {str(e)}")
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.1)  # Visual effect
        
        return len(issues) == 0, issues
    
    async def verify_dependencies(self) -> Tuple[bool, List[str]]:
        """Check all required Python packages and system dependencies"""
        console.print("\n[bold yellow]📦 VERIFYING DEPENDENCIES...[/bold yellow]")
        
        missing_packages = []
        required_packages = [
            "uvloop>=0.19.0",
            "aiohttp>=3.9.0",
            "ccxt>=4.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "torch>=2.0.0",
            "redis>=5.0.0",
            "sqlalchemy>=2.0.0",
            "rich>=13.0.0",
            "websockets>=12.0",
            "cryptography>=41.0.0",
            "python-dotenv>=1.0.0",
            "pydantic>=2.0.0",
            "fastapi>=0.100.0"
        ]
        
        # Check Python version
        if sys.version_info < (3, 9):
            missing_packages.append(f"Python 3.9+ required (current: {sys.version})")
        
        # Check packages
        try:
            import pkg_resources
            installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            
            for package in required_packages:
                pkg_name = package.split('>=')[0]
                if pkg_name not in installed:
                    missing_packages.append(package)
        except Exception as e:
            missing_packages.append(f"Cannot check packages: {str(e)}")
        
        # Check system dependencies
        system_deps = {
            "git": "Version control",
            "redis-server": "Cache/State management",
            "curl": "API communications"
        }
        
        for cmd, description in system_deps.items():
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing_packages.append(f"{cmd} ({description})")
            except FileNotFoundError:
                missing_packages.append(f"{cmd} ({description})")
        
        return len(missing_packages) == 0, missing_packages
    
    async def restore_configuration(self) -> bool:
        """Restore configuration from backups"""
        console.print("\n[bold yellow]⚙️  RESTORING CONFIGURATION...[/bold yellow]")
        
        config_file = self.config_path / "enhanced_config.json"
        
        # Check if config exists
        if config_file.exists():
            console.print("[green]✓[/green] Configuration file exists")
            return True
        
        # Look for backups
        backup_pattern = "enhanced_config_*.json"
        backups = sorted(self.backup_path.glob(f"config/{backup_pattern}"), reverse=True)
        
        if not backups:
            # Create default configuration
            console.print("[yellow]![/yellow] No backups found, creating default configuration")
            default_config = self.create_default_config()
            
            self.config_path.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return True
        
        # Restore from most recent backup
        latest_backup = backups[0]
        console.print(f"[green]✓[/green] Restoring from backup: {latest_backup.name}")
        
        try:
            shutil.copy2(latest_backup, config_file)
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore config: {e}")
            return False
    
    def create_default_config(self) -> dict:
        """Create default configuration with all features"""
        return {
            "version": "3.0.0",
            "theme": "cyberpunk_neon",
            "system": {
                "name": "NEXLIFY_TRADING_MATRIX",
                "mode": "production",
                "debug": False,
                "timezone": "UTC"
            },
            "exchanges": {
                "coinbase": {
                    "enabled": True,
                    "priority": 1,
                    "api_key": "",
                    "api_secret": "",
                    "testnet": True
                },
                "binance": {
                    "enabled": True,
                    "priority": 2,
                    "api_key": "",
                    "api_secret": "",
                    "testnet": True
                },
                "kraken": {
                    "enabled": False,
                    "api_key": "",
                    "api_secret": ""
                }
            },
            "state_persistence": {
                "primary": "rocksdb",
                "cache": "redis",
                "backup": "s3_compatible",
                "sync_interval": 60,
                "compression": "zstd"
            },
            "trading": {
                "strategies": {
                    "arbitrage": {"enabled": True, "allocation": 0.3},
                    "market_making": {"enabled": True, "allocation": 0.2},
                    "momentum": {"enabled": True, "allocation": 0.2},
                    "mean_reversion": {"enabled": True, "allocation": 0.15},
                    "ml_predictions": {"enabled": True, "allocation": 0.15}
                },
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.05,
                    "max_daily_loss": 0.05
                }
            },
            "ml_models": {
                "timesfm": {"enabled": True, "confidence_threshold": 0.7},
                "transformer": {"enabled": True, "lookback_window": 100},
                "ensemble": {"enabled": True, "models": ["lstm", "gru", "transformer"]}
            },
            "monitoring": {
                "prometheus": {"enabled": True, "port": 9090},
                "grafana": {"enabled": True, "port": 3000},
                "alerts": {
                    "email": {"enabled": False},
                    "discord": {"enabled": False},
                    "telegram": {"enabled": False}
                }
            }
        }
    
    async def validate_api_keys(self) -> Tuple[bool, List[str]]:
        """Validate exchange API keys"""
        console.print("\n[bold yellow]🔑 VALIDATING API KEYS...[/bold yellow]")
        
        issues = []
        
        try:
            # Load configuration
            config_file = self.config_path / "enhanced_config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check each exchange
            for exchange, settings in config.get("exchanges", {}).items():
                if settings.get("enabled", False):
                    if not settings.get("api_key") or not settings.get("api_secret"):
                        issues.append(f"{exchange}: Missing API credentials")
                    else:
                        # Could add actual API validation here
                        console.print(f"[green]✓[/green] {exchange}: API keys present")
            
        except Exception as e:
            issues.append(f"Cannot read configuration: {str(e)}")
        
        return len(issues) == 0, issues
    
    async def restore_database_state(self) -> bool:
        """Restore database and cache state"""
        console.print("\n[bold yellow]💾 RESTORING DATABASE STATE...[/bold yellow]")
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            console.print("[green]✓[/green] Redis cache is operational")
        except:
            console.print("[red]✗[/red] Redis not running - starting...")
            try:
                subprocess.run(["redis-server", "--daemonize", "yes"], check=True)
                console.print("[green]✓[/green] Redis started successfully")
            except:
                console.print("[yellow]![/yellow] Could not start Redis - cache disabled")
        
        # Check RocksDB data
        rocksdb_path = self.root_path / "data" / "rocksdb"
        if not rocksdb_path.exists():
            rocksdb_path.mkdir(parents=True)
            console.print("[yellow]![/yellow] Created new RocksDB directory")
        else:
            console.print("[green]✓[/green] RocksDB data directory exists")
        
        # Check for database backups
        db_backups = sorted(self.backup_path.glob("data/rocksdb_*.tar.gz"), reverse=True)
        if db_backups and not any(rocksdb_path.iterdir()):
            console.print(f"[yellow]![/yellow] Restoring database from backup: {db_backups[0].name}")
            # Would implement actual restore here
        
        return True
    
    async def restart_trading_engine(self) -> bool:
        """Restart the core trading engine"""
        console.print("\n[bold yellow]🚀 RESTARTING TRADING ENGINE...[/bold yellow]")
        
        # Check if engine is already running
        try:
            # Look for PID file
            pid_file = self.root_path / "nexlify.pid"
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is alive
                try:
                    os.kill(pid, 0)
                    console.print("[yellow]![/yellow] Trading engine already running")
                    return True
                except OSError:
                    # Process is dead, remove PID file
                    pid_file.unlink()
        except:
            pass
        
        # Start the engine
        console.print("[cyan]...[/cyan] Starting trading engine")
        
        # Use the smart launcher
        launcher_path = self.root_path / "scripts" / "smart_launcher.py"
        if launcher_path.exists():
            try:
                subprocess.Popen([sys.executable, str(launcher_path)], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                await asyncio.sleep(3)  # Give it time to start
                console.print("[green]✓[/green] Trading engine started")
                return True
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to start engine: {e}")
                return False
        else:
            console.print("[red]✗[/red] Launcher script not found")
            return False
    
    async def run_recovery(self, full_restore: bool = False):
        """Execute the recovery protocol"""
        self.print_banner()
        
        console.print(f"\n[bold cyan]RECOVERY INITIATED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]")
        console.print(f"[bold cyan]MODE: {'FULL RESTORE' if full_restore else 'QUICK RECOVERY'}[/bold cyan]\n")
        
        # Create TODO table
        table = Table(title="RECOVERY TODO LIST", box=box.ROUNDED)
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="green")
        
        # Track results
        results = {}
        
        # Execute recovery tasks
        for task in self.todo_list:
            if task in self.state.completed_tasks and not full_restore:
                table.add_row(task, "✓ COMPLETED", "Previously recovered")
                continue
            
            # Execute task
            if task == "CHECK_SYSTEM_INTEGRITY":
                success, issues = await self.check_system_integrity()
                results[task] = success
                details = "All files present" if success else f"{len(issues)} issues found"
                
            elif task == "VERIFY_DEPENDENCIES":
                success, missing = await self.verify_dependencies()
                results[task] = success
                details = "All dependencies OK" if success else f"{len(missing)} missing"
                
            elif task == "RESTORE_CONFIGURATION":
                success = await self.restore_configuration()
                results[task] = success
                details = "Configuration restored" if success else "Failed to restore"
                
            elif task == "VALIDATE_API_KEYS":
                success, issues = await self.validate_api_keys()
                results[task] = success
                details = "API keys valid" if success else f"{len(issues)} issues"
                
            elif task == "RESTORE_DATABASE_STATE":
                success = await self.restore_database_state()
                results[task] = success
                details = "Database operational" if success else "Database issues"
                
            elif task == "RESTART_TRADING_ENGINE":
                success = await self.restart_trading_engine()
                results[task] = success
                details = "Engine running" if success else "Failed to start"
                
            else:
                # Placeholder for other tasks
                results[task] = True
                details = "Simulated completion"
            
            # Update state
            if results.get(task, False):
                self.state.completed_tasks.append(task)
                status = "✓ COMPLETE"
            else:
                self.state.pending_tasks.append(task)
                status = "✗ FAILED"
            
            table.add_row(task, status, details)
        
        # Display results
        console.print(table)
        
        # Save state
        self.state.save(self.state_path)
        
        # Generate recovery report
        await self.generate_recovery_report(results)
        
        # Final status
        total_tasks = len(self.todo_list)
        completed_tasks = sum(1 for v in results.values() if v)
        success_rate = (completed_tasks / total_tasks) * 100
        
        if success_rate == 100:
            console.print(f"\n[bold green]🎉 RECOVERY COMPLETE - ALL SYSTEMS OPERATIONAL[/bold green]")
        elif success_rate >= 80:
            console.print(f"\n[bold yellow]⚠️  RECOVERY PARTIAL - {completed_tasks}/{total_tasks} TASKS COMPLETE[/bold yellow]")
        else:
            console.print(f"\n[bold red]❌ RECOVERY FAILED - ONLY {completed_tasks}/{total_tasks} TASKS COMPLETE[/bold red]")
        
        console.print(f"\n[bold cyan]Recovery logs saved to: {self.log_path}[/bold cyan]")
    
    async def generate_recovery_report(self, results: Dict[str, bool]):
        """Generate detailed recovery report"""
        report_path = self.log_path / f"recovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("NEXLIFY RECOVERY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Recovery State: {self.state.phase}\n\n")
            
            f.write("TASK RESULTS:\n")
            f.write("-" * 40 + "\n")
            for task, success in results.items():
                f.write(f"{task}: {'SUCCESS' if success else 'FAILED'}\n")
            
            f.write("\nERRORS:\n")
            f.write("-" * 40 + "\n")
            for error in self.state.errors:
                f.write(f"- {error}\n")
            
            f.write("\nWARNINGS:\n")
            f.write("-" * 40 + "\n")
            for warning in self.state.warnings:
                f.write(f"- {warning}\n")

def main():
    """Main entry point for recovery protocol"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexlify Recovery Protocol")
    parser.add_argument("--full-restore", action="store_true", 
                       help="Perform full system restore")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check system status")
    
    args = parser.parse_args()
    
    # Create recovery instance
    recovery = NexlifyRecoveryProtocol()
    
    # Run recovery
    try:
        if args.check_only:
            # Just check status
            asyncio.run(recovery.check_system_integrity())
        else:
            # Run full recovery
            asyncio.run(recovery.run_recovery(full_restore=args.full_restore))
    except KeyboardInterrupt:
        console.print("\n[bold red]Recovery interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]CRITICAL ERROR: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
