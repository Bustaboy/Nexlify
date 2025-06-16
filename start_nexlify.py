#!/usr/bin/env python3
"""
Enhanced Nexlify Startup Script (Cross-Platform)
Addresses all V3 improvements with graceful shutdown and process management
"""

import os
import sys
import subprocess
import signal
import time
import json
import socket
import atexit
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import platform
import psutil

# Try to import colorama for cross-platform colored output
try:
    from colorama import init, Fore, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback color definitions
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = ""

class NexlifyStarter:
    """Enhanced startup manager with process tracking and graceful shutdown"""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_dir = self.root_path / 'logs' / 'startup'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.shutdown_requested = False
        self.config = self._load_config()
        self.api_port = self.config.get('ports', {}).get('api', 8000)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Platform-specific signal handling
        if platform.system() != 'Windows':
            signal.signal(signal.SIGHUP, self._signal_handler)
            
    def _load_config(self) -> dict:
        """Load configuration to get ports and settings"""
        config_path = self.root_path / 'enhanced_config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except:
                pass
        return {}
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n{Fore.YELLOW}Received shutdown signal...{Style.RESET_ALL}")
        self.shutdown_requested = True
        self.shutdown()
        
    def run(self):
        """Main startup sequence"""
        self._print_banner()
        
        # Check Python version
        if not self._check_python_version():
            return False
            
        # Check system resources
        if not self._check_system_resources():
            print(f"{Fore.YELLOW}Warning: System resources may be limited{Style.RESET_ALL}")
            
        # Check if using smart_launcher.py
        if (self.root_path / 'smart_launcher.py').exists():
            print(f"{Fore.CYAN}Using smart_launcher.py for integrated startup...{Style.RESET_ALL}")
            success = self._start_smart_launcher()
        else:
            # Direct component startup
            print(f"{Fore.CYAN}Starting components directly...{Style.RESET_ALL}")
            success = self._start_components_directly()
            
        if success:
            print(f"\n{Fore.GREEN}✓ Nexlify is running!{Style.RESET_ALL}")
            self._print_access_info()
            self._monitor_processes()
        else:
            print(f"\n{Fore.RED}✗ Startup failed. Check logs in {self.log_dir}{Style.RESET_ALL}")
            return False
            
        return True
        
    def _print_banner(self):
        """Display startup banner"""
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════╗
║       NEXLIFY TRADING PLATFORM           ║
║         Night City Trader                ║
║            v2.0.8                        ║
╚══════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
        
    def _check_python_version(self) -> bool:
        """Verify Python version meets requirements"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 11):
            print(f"{Fore.RED}ERROR: Python 3.11+ required, found {version.major}.{version.minor}{Style.RESET_ALL}")
            return False
            
        print(f"{Fore.GREEN}✓ Python {version.major}.{version.minor} detected{Style.RESET_ALL}")
        return True
        
    def _check_system_resources(self) -> bool:
        """Check available system resources"""
        try:
            # Check RAM
            ram_gb = psutil.virtual_memory().available / (1024**3)
            if ram_gb < 2:
                print(f"{Fore.YELLOW}⚠ Low RAM: {ram_gb:.1f}GB available{Style.RESET_ALL}")
                return False
                
            # Check disk space
            disk_gb = psutil.disk_usage(str(self.root_path)).free / (1024**3)
            if disk_gb < 1:
                print(f"{Fore.YELLOW}⚠ Low disk space: {disk_gb:.1f}GB free{Style.RESET_ALL}")
                return False
                
            return True
        except:
            # If psutil not available, continue anyway
            return True
            
    def _start_smart_launcher(self) -> bool:
        """Start using smart_launcher.py"""
        log_file = self.log_dir / f'launcher_{self.timestamp}.log'
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [sys.executable, 'smart_launcher.py'],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
                
            self.processes['launcher'] = process
            
            # Monitor launcher output
            return self._wait_for_startup(log_file, "launcher")
            
        except Exception as e:
            print(f"{Fore.RED}Failed to start launcher: {e}{Style.RESET_ALL}")
            return False
            
    def _start_components_directly(self) -> bool:
        """Start neural net and GUI directly"""
        # Start neural net
        if not self._start_neural_net():
            return False
            
        # Wait for API to be ready
        if not self._wait_for_api():
            return False
            
        # Start GUI
        if not self._start_gui():
            return False
            
        return True
        
    def _start_neural_net(self) -> bool:
        """Start the neural net engine"""
        print(f"\n{Fore.CYAN}Starting Neural Net Engine...{Style.RESET_ALL}")
        
        neural_script = self.root_path / 'src' / 'nexlify_neural_net.py'
        if not neural_script.exists():
            # Try alternative location
            neural_script = self.root_path / 'nexlify_neural_net.py'
            
        if not neural_script.exists():
            print(f"{Fore.RED}Neural net script not found{Style.RESET_ALL}")
            return False
            
        log_file = self.log_dir / f'neural_net_{self.timestamp}.log'
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [sys.executable, '-u', str(neural_script)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
            self.processes['neural_net'] = process
            print(f"{Fore.GREEN}✓ Neural Net started (PID: {process.pid}){Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Failed to start neural net: {e}{Style.RESET_ALL}")
            return False
            
    def _wait_for_api(self, timeout: int = 30) -> bool:
        """Wait for API to be ready with dynamic timeout"""
        print(f"\n{Fore.CYAN}Waiting for API on port {self.api_port}...{Style.RESET_ALL}")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._is_port_listening(self.api_port):
                print(f"{Fore.GREEN}✓ API is ready{Style.RESET_ALL}")
                return True
                
            # Check if process died
            if 'neural_net' in self.processes:
                if self.processes['neural_net'].poll() is not None:
                    print(f"{Fore.RED}Neural net process died{Style.RESET_ALL}")
                    return False
                    
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            print(f"  Waiting... {elapsed}s", end='\r')
            
        print(f"\n{Fore.YELLOW}⚠ API took too long to start, continuing anyway{Style.RESET_ALL}")
        return True
        
    def _is_port_listening(self, port: int) -> bool:
        """Check if a port is listening"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except:
            return False
            
    def _start_gui(self) -> bool:
        """Start the GUI"""
        print(f"\n{Fore.CYAN}Starting GUI...{Style.RESET_ALL}")
        
        gui_script = self.root_path / 'cyber_gui.py'
        if not gui_script.exists():
            print(f"{Fore.RED}GUI script not found{Style.RESET_ALL}")
            return False
            
        log_file = self.log_dir / f'gui_{self.timestamp}.log'
        
        try:
            # Platform-specific GUI startup
            env = os.environ.copy()
            
            if platform.system() == 'Linux' and not env.get('DISPLAY'):
                print(f"{Fore.YELLOW}Warning: No DISPLAY set, GUI may not work{Style.RESET_ALL}")
                
            with open(log_file, 'w') as log:
                if platform.system() == 'Windows':
                    # Windows: Use pythonw to avoid console window
                    executable = sys.executable.replace('python.exe', 'pythonw.exe')
                    if not Path(executable).exists():
                        executable = sys.executable
                else:
                    executable = sys.executable
                    
                process = subprocess.Popen(
                    [executable, str(gui_script)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env=env
                )
                
            self.processes['gui'] = process
            print(f"{Fore.GREEN}✓ GUI started (PID: {process.pid}){Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Failed to start GUI: {e}{Style.RESET_ALL}")
            return False
            
    def _wait_for_startup(self, log_file: Path, component: str, timeout: int = 30) -> bool:
        """Wait for component to start successfully"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if component in self.processes:
                process = self.processes[component]
                if process.poll() is not None:
                    print(f"{Fore.RED}{component} process exited with code {process.returncode}{Style.RESET_ALL}")
                    return False
                    
            # Check log file for success indicators
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if 'error' in content.lower() and 'critical' in content.lower():
                            print(f"{Fore.RED}Critical error in {component}{Style.RESET_ALL}")
                            return False
                        if 'ready' in content.lower() or 'started' in content.lower():
                            return True
                except:
                    pass
                    
            time.sleep(1)
            
        return True
        
    def _print_access_info(self):
        """Display access information"""
        info = f"""
{Fore.GREEN}Access Points:{Style.RESET_ALL}
  • Trading API: http://localhost:{self.api_port}
  • GUI: Running in separate window
  
{Fore.YELLOW}Commands:{Style.RESET_ALL}
  • Press Ctrl+C for graceful shutdown
  • Type 'stop' and press Enter to stop
  • Type 'status' to check component status
  • Type 'logs' to show log locations
"""
        print(info)
        
    def _monitor_processes(self):
        """Monitor processes and handle user commands"""
        print(f"\n{Fore.CYAN}Monitoring processes...{Style.RESET_ALL}\n")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._process_monitor_thread, daemon=True)
        monitor_thread.start()
        
        # Handle user input
        try:
            while not self.shutdown_requested:
                try:
                    user_input = input().strip().lower()
                    
                    if user_input in ['stop', 'exit', 'quit']:
                        self.shutdown()
                        break
                    elif user_input == 'status':
                        self._show_status()
                    elif user_input == 'logs':
                        self._show_logs()
                    elif user_input == 'help':
                        self._show_help()
                        
                except KeyboardInterrupt:
                    self.shutdown()
                    break
                except EOFError:
                    # Handle when running without terminal
                    time.sleep(1)
                    
        except Exception as e:
            print(f"{Fore.RED}Monitor error: {e}{Style.RESET_ALL}")
            
    def _process_monitor_thread(self):
        """Monitor process health in background"""
        while not self.shutdown_requested:
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    print(f"\n{Fore.YELLOW}⚠ {name} stopped unexpectedly (exit code: {process.returncode}){Style.RESET_ALL}")
                    del self.processes[name]
                    
                    # Try to restart critical components
                    if name == 'neural_net' and not self.shutdown_requested:
                        print(f"{Fore.CYAN}Attempting to restart {name}...{Style.RESET_ALL}")
                        self._start_neural_net()
                        
            time.sleep(5)
            
    def _show_status(self):
        """Show status of all components"""
        print(f"\n{Fore.CYAN}Component Status:{Style.RESET_ALL}")
        for name, process in self.processes.items():
            if process.poll() is None:
                # Get process info
                try:
                    p = psutil.Process(process.pid)
                    cpu = p.cpu_percent(interval=0.1)
                    mem = p.memory_info().rss / 1024 / 1024  # MB
                    print(f"  • {name}: {Fore.GREEN}Running{Style.RESET_ALL} (PID: {process.pid}, CPU: {cpu:.1f}%, RAM: {mem:.1f}MB)")
                except:
                    print(f"  • {name}: {Fore.GREEN}Running{Style.RESET_ALL} (PID: {process.pid})")
            else:
                print(f"  • {name}: {Fore.RED}Stopped{Style.RESET_ALL}")
                
        # Check API status
        if self._is_port_listening(self.api_port):
            print(f"  • API Port {self.api_port}: {Fore.GREEN}Listening{Style.RESET_ALL}")
        else:
            print(f"  • API Port {self.api_port}: {Fore.RED}Not listening{Style.RESET_ALL}")
            
    def _show_logs(self):
        """Show log file locations"""
        print(f"\n{Fore.CYAN}Log Files:{Style.RESET_ALL}")
        for log_file in sorted(self.log_dir.glob(f'*_{self.timestamp}.log')):
            size = log_file.stat().st_size / 1024  # KB
            print(f"  • {log_file.name} ({size:.1f}KB)")
        print(f"\nLog directory: {self.log_dir}")
        
    def _show_help(self):
        """Show available commands"""
        help_text = f"""
{Fore.CYAN}Available Commands:{Style.RESET_ALL}
  • stop/exit/quit - Graceful shutdown
  • status - Show component status
  • logs - Show log file locations
  • help - Show this help message
"""
        print(help_text)
        
    def shutdown(self):
        """Graceful shutdown of all components"""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        print(f"\n{Fore.YELLOW}Initiating graceful shutdown...{Style.RESET_ALL}")
        
        # Create emergency stop file
        stop_file = self.root_path / 'EMERGENCY_STOP_ACTIVE'
        stop_file.write_text('STOP')
        
        # Shutdown in reverse order
        shutdown_order = ['gui', 'neural_net', 'launcher']
        
        for component in shutdown_order:
            if component in self.processes:
                process = self.processes[component]
                if process.poll() is None:
                    print(f"Stopping {component}...")
                    
                    # Try graceful termination first
                    try:
                        if platform.system() == 'Windows':
                            process.terminate()
                        else:
                            process.send_signal(signal.SIGTERM)
                            
                        # Wait for graceful shutdown
                        try:
                            process.wait(timeout=5)
                            print(f"  ✓ {component} stopped gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if needed
                            print(f"  ⚠ Force stopping {component}")
                            process.kill()
                            process.wait()
                            
                    except Exception as e:
                        print(f"  ✗ Error stopping {component}: {e}")
                        
        # Clean up emergency stop file
        if stop_file.exists():
            stop_file.unlink()
            
        print(f"\n{Fore.GREEN}✓ Shutdown complete{Style.RESET_ALL}")
        print(f"Logs saved to: {self.log_dir}")
        
    def cleanup(self):
        """Cleanup on exit"""
        if not self.shutdown_requested:
            self.shutdown()


def main():
    """Main entry point"""
    starter = NexlifyStarter()
    
    try:
        success = starter.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"{Fore.RED}Startup error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
