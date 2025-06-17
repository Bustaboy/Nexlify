# nexlify_installer.py
"""
Nexlify Installer - Smooth as a Braindance, Powerful as Cyberware
Deploys the entire trading platform with style
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import psutil
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import print as rprint

# Initialize rich console for that cyberpunk aesthetic
console = Console()

# ASCII art logo - because style matters
NEXLIFY_LOGO = """
[cyan]
███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗
████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝
██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝ 
██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝  
██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║   
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝   
[/cyan]
[magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/magenta]
[yellow]     Cyberpunk Trading Platform v2.0 - Night City Edition[/yellow]
[magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/magenta]
"""

class NexlifyInstaller:
    """
    The master installer - deploys Nexlify like a pro
    Handles everything from system checks to neural net initialization
    """
    
    def __init__(self):
        self.install_dir = Path.home() / "nexlify"
        self.venv_dir = self.install_dir / "venv"
        self.config_dir = self.install_dir / "config"
        self.data_dir = self.install_dir / "data"
        self.logs_dir = self.install_dir / "logs"
        self.models_dir = self.install_dir / "models"
        
        self.min_requirements = {
            'python': (3, 11),
            'ram_gb': 8,
            'disk_gb': 20,
            'cpu_cores': 4
        }
        
        self.dependencies = {
            'system': ['postgresql', 'redis', 'nodejs', 'npm'],
            'python': [
                'fastapi==0.104.1',
                'uvicorn[standard]==0.24.0',
                'sqlalchemy==2.0.23',
                'alembic==1.12.1',
                'psycopg2-binary==2.9.9',
                'redis==5.0.1',
                'celery==5.3.4',
                'torch==2.1.0',
                'transformers==4.35.2',
                'ray[default]==2.8.0',
                'pandas==2.1.3',
                'numpy==1.24.3',
                'ta==0.10.2',
                'scikit-learn==1.3.2',
                'prometheus-client==0.19.0',
                'opentelemetry-api==1.21.0',
                'structlog==23.2.0',
                'pydantic==2.5.0',
                'python-jose[cryptography]==3.3.0',
                'passlib[argon2]==1.7.4',
                'python-multipart==0.0.6',
                'pyotp==2.9.0',
                'qrcode==7.4.2',
                'aiohttp==3.9.0',
                'websockets==12.0',
                'click==8.1.7',
                'rich==13.7.0',
                'pytest==7.4.3',
                'pytest-asyncio==0.21.1',
                'pytest-cov==4.1.0',
                'faker==20.0.3',
                'factory-boy==3.3.0'
            ]
        }
    
    def show_welcome(self):
        """Display welcome screen - first impressions matter"""
        console.clear()
        rprint(NEXLIFY_LOGO)
        
        rprint("\n[bold cyan]Welcome to the Nexlify Installation Matrix[/bold cyan]")
        rprint("[yellow]Preparing to jack you into the most advanced trading platform in Night City...[/yellow]\n")
        
        # Show system info
        info_table = Table(title="[bold magenta]System Information[/bold magenta]", show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Operating System", f"{platform.system()} {platform.release()}")
        info_table.add_row("Architecture", platform.machine())
        info_table.add_row("Python Version", platform.python_version())
        info_table.add_row("CPU Cores", str(psutil.cpu_count()))
        info_table.add_row("Total RAM", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
        info_table.add_row("Available Disk", f"{shutil.disk_usage('/').free / (1024**3):.1f} GB")
        
        console.print(info_table)
        console.print()
    
    def check_system_requirements(self) -> Tuple[bool, List[str]]:
        """Check if system meets requirements - no weak chrome allowed"""
        issues = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Scanning system capabilities...", total=None)
            
            # Check Python version
            py_version = sys.version_info[:2]
            if py_version < self.min_requirements['python']:
                issues.append(f"Python {self.min_requirements['python'][0]}.{self.min_requirements['python'][1]}+ required (found {py_version[0]}.{py_version[1]})")
            
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb < self.min_requirements['ram_gb']:
                issues.append(f"Minimum {self.min_requirements['ram_gb']}GB RAM required (found {ram_gb:.1f}GB)")
            
            # Check disk space
            disk_gb = shutil.disk_usage('/').free / (1024**3)
            if disk_gb < self.min_requirements['disk_gb']:
                issues.append(f"Minimum {self.min_requirements['disk_gb']}GB free disk space required (found {disk_gb:.1f}GB)")
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            if cpu_cores < self.min_requirements['cpu_cores']:
                issues.append(f"Minimum {self.min_requirements['cpu_cores']} CPU cores required (found {cpu_cores})")
            
            # Check for CUDA (optional but recommended)
            try:
                import torch
                if torch.cuda.is_available():
                    console.print("[green]✓[/green] CUDA detected - neural acceleration available")
                else:
                    console.print("[yellow]![/yellow] No CUDA detected - will use CPU for ML (slower)")
            except:
                pass
        
        return len(issues) == 0, issues
    
    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies - the supporting chrome"""
        deps_status = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Checking system dependencies...", total=len(self.dependencies['system']))
            
            for dep in self.dependencies['system']:
                progress.update(task, advance=1, description=f"[cyan]Checking {dep}...")
                
                if dep == 'postgresql':
                    deps_status[dep] = self._check_command(['psql', '--version'])
                elif dep == 'redis':
                    deps_status[dep] = self._check_command(['redis-cli', '--version'])
                elif dep == 'nodejs':
                    deps_status[dep] = self._check_command(['node', '--version'])
                elif dep == 'npm':
                    deps_status[dep] = self._check_command(['npm', '--version'])
        
        return deps_status
    
    def _check_command(self, cmd: List[str]) -> bool:
        """Check if a command exists - simple and clean"""
        try:
            subprocess.run(cmd, capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False
    
    def create_directory_structure(self):
        """Create directory structure - organize the digital space"""
        directories = [
            self.install_dir,
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.install_dir / "src",
            self.install_dir / "frontend",
            self.install_dir / "scripts",
            self.data_dir / "backups",
            self.models_dir / "checkpoints"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Creating directory structure...", total=len(directories))
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                progress.update(task, advance=1)
        
        console.print("[green]✓[/green] Directory structure created")
    
    def setup_virtual_environment(self):
        """Setup Python virtual environment - isolated neural space"""
        console.print("\n[bold cyan]Setting up virtual environment...[/bold cyan]")
        
        try:
            # Create venv
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            
            # Get pip path
            pip_path = self.venv_dir / "bin" / "pip" if platform.system() != "Windows" else self.venv_dir / "Scripts" / "pip.exe"
            
            # Upgrade pip
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            console.print("[green]✓[/green] Virtual environment created")
            return pip_path
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Failed to create virtual environment: {e}")
            return None
    
    def install_python_dependencies(self, pip_path: Path):
        """Install Python dependencies - load the neural augmentations"""
        console.print("\n[bold cyan]Installing Python dependencies...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Installing packages...", total=len(self.dependencies['python']))
            
            for package in self.dependencies['python']:
                progress.update(task, advance=1, description=f"[cyan]Installing {package.split('==')[0]}...")
                
                try:
                    subprocess.run(
                        [str(pip_path), "install", package],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    console.print(f"[yellow]![/yellow] Failed to install {package}: {e.stderr.decode()}")
    
    def setup_database(self):
        """Setup PostgreSQL database - the data vault"""
        console.print("\n[bold cyan]Setting up PostgreSQL database...[/bold cyan]")
        
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'nexlify_trading',
            'user': 'nexlify_user',
            'password': self._generate_secure_password()
        }
        
        # Save database config
        config_path = self.config_dir / "database.json"
        with open(config_path, 'w') as f:
            json.dump(db_config, f, indent=2)
        
        console.print("[green]✓[/green] Database configuration saved")
        console.print(f"[yellow]![/yellow] Please create PostgreSQL database and user:")
        
        sql_commands = f"""
-- Run these commands as PostgreSQL superuser:
CREATE USER {db_config['user']} WITH PASSWORD '{db_config['password']}';
CREATE DATABASE {db_config['database']} OWNER {db_config['user']};
GRANT ALL PRIVILEGES ON DATABASE {db_config['database']} TO {db_config['user']};
"""
        
        syntax = Syntax(sql_commands, "sql", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[bold yellow]PostgreSQL Setup Commands[/bold yellow]"))
    
    def setup_frontend(self):
        """Setup frontend - the chrome UI"""
        console.print("\n[bold cyan]Setting up frontend...[/bold cyan]")
        
        # Create package.json
        package_json = {
            "name": "nexlify-frontend",
            "version": "2.0.0",
            "description": "Nexlify Trading Platform - Cyberpunk UI",
            "main": "electron/main.js",
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "electron": "electron .",
                "dev": "concurrently \"npm start\" \"wait-on http://localhost:3000 && electron .\"",
                "dist": "npm run build && electron-builder"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-scripts": "5.0.1",
                "recharts": "^2.9.0",
                "lucide-react": "^0.290.0",
                "axios": "^1.6.0",
                "tailwindcss": "^3.3.5",
                "electron": "^27.0.0"
            },
            "devDependencies": {
                "concurrently": "^8.2.2",
                "wait-on": "^7.2.0",
                "electron-builder": "^24.6.4"
            }
        }
        
        package_path = self.install_dir / "frontend" / "package.json"
        with open(package_path, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        console.print("[green]✓[/green] Frontend configuration created")
        console.print("[yellow]![/yellow] Run 'npm install' in the frontend directory to install dependencies")
    
    def create_launcher_scripts(self):
        """Create launcher scripts - one-click startup"""
        console.print("\n[bold cyan]Creating launcher scripts...[/bold cyan]")
        
        # Create main launcher script
        if platform.system() == "Windows":
            launcher_path = self.install_dir / "launch_nexlify.bat"
            launcher_content = f"""@echo off
echo Starting Nexlify Trading Platform...
cd /d "{self.install_dir}"

REM Start Redis
start "Redis" redis-server

REM Start PostgreSQL (if not running as service)
REM start "PostgreSQL" pg_ctl start -D "C:\\Program Files\\PostgreSQL\\data"

REM Start Backend API
start "Nexlify API" cmd /k "cd src && ..\\venv\\Scripts\\python -m uvicorn api.main:app --reload"

REM Start Frontend
start "Nexlify Frontend" cmd /k "cd frontend && npm run electron"

echo Nexlify is starting up...
echo Check the opened windows for status.
pause
"""
        else:  # Unix-like systems
            launcher_path = self.install_dir / "launch_nexlify.sh"
            launcher_content = f"""#!/bin/bash
echo "Starting Nexlify Trading Platform..."
cd "{self.install_dir}"

# Start Redis
redis-server &

# Start Backend API
source venv/bin/activate
cd src && python -m uvicorn api.main:app --reload &
cd ..

# Start Frontend
cd frontend && npm run electron &

echo "Nexlify is starting up..."
echo "Press Ctrl+C to stop all services"
wait
"""
            
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        if platform.system() != "Windows":
            os.chmod(launcher_path, 0o755)
        
        console.print(f"[green]✓[/green] Launcher script created: {launcher_path}")
    
    def _generate_secure_password(self) -> str:
        """Generate secure password - strong as cyberware encryption"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(32))
    
    def create_default_config(self):
        """Create default configuration - the neural presets"""
        config = {
            "app_name": "Nexlify Trading Platform",
            "version": "2.0.0",
            "environment": "production",
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "security": {
                "enable_2fa": True,
                "session_timeout": 3600,
                "rate_limit": 100
            },
            "ml": {
                "default_model": "CyberTransformer-v1",
                "confidence_threshold": 0.7,
                "use_gpu": True
            },
            "monitoring": {
                "enable_prometheus": True,
                "metrics_port": 9090,
                "alert_webhook": ""
            }
        }
        
        config_path = self.config_dir / "nexlify.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print("[green]✓[/green] Default configuration created")
    
    async def run_installation(self):
        """Main installation flow - the full neural upgrade"""
        self.show_welcome()
        
        # Check requirements
        console.print("\n[bold cyan]Checking system requirements...[/bold cyan]")
        meets_requirements, issues = self.check_system_requirements()
        
        if not meets_requirements:
            console.print("\n[bold red]System does not meet minimum requirements:[/bold red]")
            for issue in issues:
                console.print(f"  [red]✗[/red] {issue}")
            
            if not click.confirm("\nContinue anyway?", default=False):
                console.print("\n[yellow]Installation cancelled.[/yellow]")
                return
        else:
            console.print("[green]✓[/green] System meets all requirements")
        
        # Check dependencies
        console.print("\n[bold cyan]Checking system dependencies...[/bold cyan]")
        deps_status = self.check_system_dependencies()
        
        missing_deps = [dep for dep, installed in deps_status.items() if not installed]
        if missing_deps:
            console.print("\n[bold yellow]Missing system dependencies:[/bold yellow]")
            for dep in missing_deps:
                console.print(f"  [yellow]![/yellow] {dep} not found")
            
            console.print("\n[yellow]Please install missing dependencies before continuing.[/yellow]")
            if not click.confirm("\nContinue anyway?", default=False):
                console.print("\n[yellow]Installation cancelled.[/yellow]")
                return
        
        # Create directories
        self.create_directory_structure()
        
        # Setup virtual environment
        pip_path = self.setup_virtual_environment()
        if not pip_path:
            console.print("\n[red]Failed to setup virtual environment. Aborting.[/red]")
            return
        
        # Install Python dependencies
        self.install_python_dependencies(pip_path)
        
        # Setup database
        self.setup_database()
        
        # Setup frontend
        self.setup_frontend()
        
        # Create config files
        self.create_default_config()
        
        # Create launcher scripts
        self.create_launcher_scripts()
        
        # Final summary
        console.print("\n" + "="*60)
        console.print(Panel(
            "[bold green]Installation Complete![/bold green]\n\n"
            f"Nexlify has been installed to: [cyan]{self.install_dir}[/cyan]\n\n"
            "[bold yellow]Next Steps:[/bold yellow]\n"
            "1. Set up PostgreSQL database using the provided commands\n"
            "2. Configure Redis server\n"
            "3. Install frontend dependencies: cd frontend && npm install\n"
            "4. Launch Nexlify using the launcher script\n\n"
            "[bold magenta]Welcome to Night City's finest trading platform![/bold magenta]",
            title="[bold cyan]Installation Successful[/bold cyan]",
            border_style="green"
        ))

@click.command()
@click.option('--install-dir', default=None, help='Custom installation directory')
@click.option('--skip-deps', is_flag=True, help='Skip dependency installation')
@click.option('--dev-mode', is_flag=True, help='Install in development mode')
def main(install_dir, skip_deps, dev_mode):
    """
    Nexlify Installer - Deploy the future of trading
    """
    installer = NexlifyInstaller()
    
    if install_dir:
        installer.install_dir = Path(install_dir)
    
    if dev_mode:
        console.print("[yellow]Installing in development mode...[/yellow]")
    
    try:
        asyncio.run(installer.run_installation())
    except KeyboardInterrupt:
        console.print("\n[red]Installation interrupted by user.[/red]")
    except Exception as e:
        console.print(f"\n[red]Installation failed: {e}[/red]")
        logging.exception("Installation error")

if __name__ == "__main__":
    main()
