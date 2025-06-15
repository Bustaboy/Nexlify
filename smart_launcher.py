#!/usr/bin/env python3
"""
Nexlify - Smart Launcher
Handles proper startup sequence and dependency checking
"""

import os
import sys
import subprocess
import time
import json
import threading
from pathlib import Path

# Add project root to path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

from error_handler import get_error_handler, handle_errors

# Initialize error handler
error_handler = get_error_handler()

class NightCityLauncher:
    def __init__(self):
        self.project_root = project_root
        os.chdir(self.project_root)
        
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'cyan': '\033[96m',
            'reset': '\033[0m'
        }
        
        self.processes = []
    
    def print_banner(self):
        """Print cyberpunk startup banner"""
        print(self.colors['cyan'] + """
        ███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗
        ████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝
        ██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝ 
        ██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝  
        ██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║   
        ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝   
                    ARASAKA NEURAL-NET LAUNCHER v2.0.7.7
        """ + self.colors['reset'])
        print(self.colors['yellow'] + "="*75 + self.colors['reset'])
    
    @handle_errors("Checking Requirements", reraise=False)
    def check_requirements(self) -> bool:
        """Check if all requirements are met"""
        print(f"{self.colors['yellow']}🔍 Checking system requirements...{self.colors['reset']}")
        
        # Check Python version
        if sys.version_info < (3, 9):
            print(f"{self.colors['red']}❌ Python 3.9+ required! You have {sys.version}{self.colors['reset']}")
            return False
        print(f"{self.colors['green']}✅ Python {sys.version.split()[0]} detected{self.colors['reset']}")
        
        # Check required files
        required_files = [
            'arasaka_neural_net.py',
            'cyber_gui.py',
            'error_handler.py',
            'requirements.txt'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"{self.colors['red']}❌ Missing required file: {file}{self.colors['reset']}")
                return False
        print(f"{self.colors['green']}✅ All required files present{self.colors['reset']}")
        
        # Check/create directories
        required_dirs = ['config', 'logs', 'data', 'models', 'backups', 'reports']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                print(f"{self.colors['yellow']}📁 Created directory: {dir_name}{self.colors['reset']}")
        
        # Check Python packages
        print(f"{self.colors['yellow']}📦 Checking Python packages...{self.colors['reset']}")
        missing_packages = self.check_packages()
        
        if missing_packages:
            print(f"{self.colors['yellow']}⚠️  Missing packages: {', '.join(missing_packages)}{self.colors['reset']}")
            print(f"{self.colors['cyan']}Installing missing packages...{self.colors['reset']}")
            
            if self.install_packages(missing_packages):
                print(f"{self.colors['green']}✅ All packages installed{self.colors['reset']}")
            else:
                print(f"{self.colors['red']}❌ Failed to install some packages{self.colors['reset']}")
                return False
        else:
            print(f"{self.colors['green']}✅ All required packages installed{self.colors['reset']}")
        
        return True
    
    def check_packages(self) -> list:
        """Check which required packages are missing"""
        required = [
            'ccxt', 'pandas', 'numpy', 'matplotlib', 'aiohttp',
            'tkinter', 'asyncio', 'websockets'
        ]
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        return missing
    
    def install_packages(self, packages: list) -> bool:
        """Install missing packages"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + packages
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            error_handler.log_error(e, "Package installation failed")
            return False
    
    @handle_errors("Checking Configuration", reraise=False)
    def check_config(self) -> bool:
        """Check if configuration exists"""
        config_path = self.project_root / 'config' / 'neural_config.json'
        
        if not config_path.exists():
            print(f"{self.colors['yellow']}⚠️  No configuration found - creating default...{self.colors['reset']}")
            
            default_config = {
                "exchanges": {},
                "trading": {
                    "min_profit_percent": 0.5,
                    "max_position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.05
                },
                "neural_net": {
                    "confidence_threshold": 0.7,
                    "retrain_interval_hours": 168,
                    "max_concurrent_trades": 5
                },
                "security": {
                    "pin": "2077",
                    "encryption_enabled": True,
                    "2fa_enabled": False
                },
                "btc_wallet_address": "",
                "auto_withdraw": {
                    "enabled": False,
                    "min_amount_usd": 100,
                    "schedule": "monthly"
                },
                "risk_level": "low",
                "auto_trade": True,
                "environment": {
                    "debug": False,
                    "log_level": "INFO",
                    "api_port": 8000,
                    "database_url": "sqlite:///data/trading.db",
                    "emergency_contact": "",
                    "telegram_bot_token": "",
                    "telegram_chat_id": ""
                }
            }
            
            # Create config directory if it doesn't exist
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"{self.colors['green']}✅ Default configuration created{self.colors['reset']}")
            print(f"{self.colors['cyan']}📝 You'll be guided through setup on first launch{self.colors['reset']}")
            return False
        
        # Check if API keys are configured
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            has_exchanges = False
            if 'exchanges' in config:
                for exchange, settings in config['exchanges'].items():
                    if settings.get('api_key') and settings['api_key'] != 'YOUR_API_KEY_HERE':
                        has_exchanges = True
                        break
            
            return has_exchanges
            
        except Exception as e:
            error_handler.log_error(e, "Failed to read configuration")
            return False
    
    def check_emergency_stop(self) -> bool:
        """Check for emergency stop file"""
        if os.path.exists('EMERGENCY_STOP_ACTIVE'):
            print(f"\n{self.colors['red']}⚠️  EMERGENCY STOP DETECTED!{self.colors['reset']}")
            print("The kill switch was activated in a previous session.")
            
            # Show when it was activated
            try:
                with open('EMERGENCY_STOP_ACTIVE', 'r') as f:
                    stop_time = f.read().strip()
                print(f"Activated at: {stop_time}")
            except:
                pass
            
            response = input("\nRemove emergency stop and continue? (y/n): ")
            if response.lower() == 'y':
                os.remove('EMERGENCY_STOP_ACTIVE')
                print(f"{self.colors['green']}✅ Emergency stop cleared{self.colors['reset']}")
                return True
            else:
                print("Exiting...")
                return False
        return True
    
    @handle_errors("Starting Neural Net", severity="critical", reraise=False)
    def start_neural_net(self):
        """Start the neural net trading engine"""
        print(f"\n{self.colors['cyan']}🧠 Starting Arasaka Neural-Net...{self.colors['reset']}")
        
        # Start neural net in subprocess
        neural_process = subprocess.Popen(
            [sys.executable, 'arasaka_neural_net.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output in thread
        def monitor_neural_net():
            for line in neural_process.stdout:
                if line.strip():
                    print(f"  [NEURAL-NET] {line.strip()}")
        
        monitor_thread = threading.Thread(target=monitor_neural_net, daemon=True)
        monitor_thread.start()
        
        # Give it time to initialize
        time.sleep(3)
        
        if neural_process.poll() is not None:
            stderr = neural_process.stderr.read()
            error_handler.log_error(
                Exception(f"Neural-Net failed to start: {stderr}"),
                "Neural-Net startup failure",
                severity="critical"
            )
            print(f"{self.colors['red']}❌ Neural-Net failed to start!{self.colors['reset']}")
            return None
        
        print(f"{self.colors['green']}✅ Neural-Net online and scanning markets{self.colors['reset']}")
        self.processes.append(neural_process)
        return neural_process
    
    @handle_errors("Starting GUI", severity="critical", reraise=False)
    def start_gui(self):
        """Start the GUI interface"""
        print(f"\n{self.colors['cyan']}🖥️  Starting Cyberpunk GUI...{self.colors['reset']}")
        
        # Start GUI
        gui_process = subprocess.Popen(
            [sys.executable, 'cyber_gui.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        time.sleep(2)
        
        if gui_process.poll() is not None:
            stderr = gui_process.stderr.read()
            error_handler.log_error(
                Exception(f"GUI failed to start: {stderr}"),
                "GUI startup failure",
                severity="critical"
            )
            print(f"{self.colors['red']}❌ GUI failed to start!{self.colors['reset']}")
            if stderr:
                print(f"Error: {stderr}")
            return None
        
        print(f"{self.colors['green']}✅ GUI interface ready{self.colors['reset']}")
        self.processes.append(gui_process)
        return gui_process
    
    def cleanup(self):
        """Clean up processes on exit"""
        print(f"\n{self.colors['yellow']}Shutting down...{self.colors['reset']}")
        
        for process in self.processes:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print(f"{self.colors['green']}✅ All processes stopped{self.colors['reset']}")
    
    @handle_errors("Main Launcher Loop", severity="fatal")
    def run(self):
        """Main launcher sequence"""
        self.print_banner()
        
        # Check requirements
        if not self.check_requirements():
            print(f"\n{self.colors['red']}Please install missing requirements first!{self.colors['reset']}")
            input("\nPress Enter to exit...")
            return
        
        # Check emergency stop
        if not self.check_emergency_stop():
            return
        
        # Check config
        has_config = self.check_config()
        
        print(f"\n{self.colors['yellow']}🚀 Initializing Nexlify Trading Matrix...{self.colors['reset']}")
        
        # Start neural net
        neural_process = self.start_neural_net()
        if not neural_process:
            input("\nPress Enter to exit...")
            return
        
        # Start GUI
        gui_process = self.start_gui()
        if not gui_process:
            self.cleanup()
            input("\nPress Enter to exit...")
            return
        
        # Print instructions
        print("\n" + "="*75)
        print(f"{self.colors['green']}✅ NEXLIFY IS RUNNING!{self.colors['reset']}")
        print("\n📋 Instructions:")
        
        if not has_config:
            print("1. The GUI will show the onboarding screen")
            print("2. Enter your exchange API credentials")
            print("3. Set your BTC wallet address")
            print("4. Choose your risk level")
            print("5. Click 'JACK INTO THE MATRIX' to start trading")
        else:
            print("1. Enter PIN: 2077 (default)")
            print("2. Monitor active trading pairs")
            print("3. Check profit charts")
            print("4. Adjust settings as needed")
        
        print("\n⚠️  To stop:")
        print("- Use the KILL SWITCH in the GUI")
        print("- Or press Ctrl+C in this window")
        print("- Or close all windows")
        print("\n📊 Error Monitoring:")
        print("- Check the ERROR REPORT tab in GUI")
        print("- Logs are in the logs/ directory")
        print("\n" + "="*75)
        
        # Wait for processes
        try:
            print(f"\n{self.colors['cyan']}System running... Press Ctrl+C to stop{self.colors['reset']}")
            gui_process.wait()
        except KeyboardInterrupt:
            print(f"\n{self.colors['yellow']}Received interrupt signal...{self.colors['reset']}")
        except Exception as e:
            error_handler.log_error(e, "Launcher error", severity="error")
        finally:
            self.cleanup()
            print(f"\n{self.colors['cyan']}Thanks for trading with Nexlify! 👋{self.colors['reset']}")

if __name__ == "__main__":
    launcher = NightCityLauncher()
    launcher.run()