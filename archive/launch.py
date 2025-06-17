#!/usr/bin/env python3
"""
Nexlify - Quick Launcher
Simple launcher for the Arasaka Neural-Net Trading Matrix
"""

import subprocess
import time
import os
import sys

def main():
    """Quick launch Nexlify trading system"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🌃 Starting Nexlify Trading Matrix...")
    print("=" * 50)
    
    # Start Neural-Net API
    print("Starting Neural-Net API...")
    api_process = subprocess.Popen([sys.executable, "arasaka_neural_net.py"])
    time.sleep(3)
    
    # Start Cyberpunk GUI
    print("Starting Cyberpunk GUI...")
    gui_process = subprocess.Popen([sys.executable, "cyber_gui.py"])
    
    print("\n✅ Nexlify is running!")
    print("\nPress Ctrl+C to stop all processes")
    
    try:
        gui_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()
        gui_process.terminate()

if __name__ == "__main__":
    main()
