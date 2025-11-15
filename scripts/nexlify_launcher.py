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
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(project_root))

from nexlify.utils.error_handler import get_error_handler, handle_errors

# Initialize error handler
error_handler = get_error_handler()

class NexlifyLauncher:
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
        if sys.version_info < (3, 11):
            print(f"{self.colors['red']}❌ Python 3.11+ required! You have {sys.version_info.major}.{sys.version_info.minor}{self.colors['reset']}")
            return False

        print(f"{self.colors['green']}✅ Python {sys.version_info.major}.{sys.version_info.minor} detected{self.colors['reset']}")
        
        # Check critical modules
        required_modules = ['ccxt', 'pandas', 'numpy', 'aiohttp', 'colorama']
        missing = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            print(f"{self.colors['red']}❌ Missing modules: {', '.join(missing)}{self.colors['reset']}")
            print(f"{self.colors['yellow']}Run: pip install -r requirements.txt{self.colors['reset']}")
            return False
        
        print(f"{self.colors['green']}✅ All required modules installed{self.colors['reset']}")
        return True
    
    def check_emergency_stop(self) -> bool:
        """Check if emergency stop is active"""
        stop_file = self.project_root / "EMERGENCY_STOP_ACTIVE"
        if stop_file.exists():
            print(f"{self.colors['red']}⛔ EMERGENCY STOP is active!{self.colors['reset']}")
            print(f"{self.colors['yellow']}To resume trading, delete: {stop_file}{self.colors['reset']}")
            return False
        return True
    
    def check_directories(self):
        """Ensure required directories exist"""
        print(f"{self.colors['yellow']}📁 Checking directories...{self.colors['reset']}")
        
        dirs = ['config', 'logs', 'data', 'models', 'backups', 'reports', 'logs/crash_reports']
        for dir_name in dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"{self.colors['green']}✅ All directories ready{self.colors['reset']}")
    
    def check_config(self) -> bool:
        """Check if configuration exists"""
        config_file = self.project_root / "config" / "neural_config.json"
        if not config_file.exists():
            print(f"{self.colors['yellow']}📝 No configuration found - GUI will show onboarding{self.colors['reset']}")
            return False
        return True
    
    @handle_errors("Starting Neural Net", reraise=False)
    def start_neural_net(self):
        """Start the neural net API server"""
        print(f"\n{self.colors['cyan']}🧠 Starting Neural-Net API...{self.colors['reset']}")

        api_script = self.project_root / "nexlify" / "core" / "arasaka_neural_net.py"
        if not api_script.exists():
            print(f"{self.colors['red']}❌ Missing arasaka_neural_net.py!{self.colors['reset']}")
            return None
        
        process = subprocess.Popen(
            [sys.executable, str(api_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        
        # Wait for API to start
        print("Waiting for API to initialize...")
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"{self.colors['red']}❌ API failed to start!{self.colors['reset']}")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Error: {stderr}")
            return None
        
        print(f"{self.colors['green']}✅ Neural-Net API running on http://127.0.0.1:8000{self.colors['reset']}")
        return process
    
    @handle_errors("Starting GUI", reraise=False)
    def start_gui(self):
        """Start the cyberpunk GUI"""
        print(f"\n{self.colors['cyan']}🎮 Starting Cyberpunk GUI...{self.colors['reset']}")

        gui_script = self.project_root / "nexlify" / "gui" / "cyber_gui.py"
        if not gui_script.exists():
            print(f"{self.colors['red']}❌ Missing cyber_gui.py!{self.colors['reset']}")
            return None
        
        process = subprocess.Popen(
            [sys.executable, str(gui_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        
        # Give GUI time to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"{self.colors['red']}❌ GUI failed to start!{self.colors['reset']}")
            stdout, stderr = process.communicate()
            if stderr:
                print(f"Error: {stderr}")
            return None
        
        print(f"{self.colors['green']}✅ Cyberpunk GUI is running{self.colors['reset']}")
        return process
    
    def cleanup(self):
        """Clean up processes on exit"""
        print(f"\n{self.colors['yellow']}🧹 Shutting down processes...{self.colors['reset']}")
        
        for process in self.processes:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print(f"{self.colors['green']}✅ All processes stopped{self.colors['reset']}")
    
    def run(self):
        """Run the launcher"""
        self.print_banner()
        
        # Check requirements
        if not self.check_requirements():
            print(f"\n{self.colors['red']}❌ Requirements not met! Run setup_nexlify.py first{self.colors['reset']}")
            input("\nPress Enter to exit...")
            return
        
        # Check emergency stop
        if not self.check_emergency_stop():
            return
        
        # Check directories
        self.check_directories()
        
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
    launcher = NexlifyLauncher()
    launcher.run()
