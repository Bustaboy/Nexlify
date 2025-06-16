# scripts/install_requirements.py
"""
Nexlify Requirements Installation Helper
Helps users choose and install the appropriate requirements
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class RequirementsInstaller:
    """Interactive requirements installer for Nexlify"""
    
    def __init__(self):
        self.root_path = Path(__file__).parent.parent
        self.python_cmd = sys.executable
        
    def run(self):
        """Main installation process"""
        print("\n" + "="*60)
        print("🌃 NEXLIFY REQUIREMENTS INSTALLER v2.0.8")
        print("="*60 + "\n")
        
        # Check Python version
        if not self._check_python_version():
            return
        
        # Choose installation type
        install_type = self._choose_install_type()
        
        # Install requirements
        if install_type:
            self._install_requirements(install_type)
        
        print("\n✅ Installation complete!")
        print("📋 Next step: Run 'python setup_nexlify.py' to complete setup")
        
    def _check_python_version(self):
        """Verify Python version"""
        version = sys.version_info
        if version < (3, 11):
            print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}")
            print("📥 Download from: https://python.org")
            return False
        
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True
        
    def _choose_install_type(self):
        """Let user choose installation type"""
        print("\n📦 Choose installation type:")
        print("\n1. STANDARD - Core features only (recommended)")
        print("   - Exchange trading")
        print("   - GUI interface")
        print("   - Basic strategies")
        print("   - ~500MB download")
        
        print("\n2. FULL - All features including ML/AI")
        print("   - Everything in Standard")
        print("   - Machine Learning models")
        print("   - AI trading companion")
        print("   - DeFi/DEX integration")
        print("   - Advanced analytics")
        print("   - ~3GB download")
        
        print("\n3. DEVELOPMENT - For contributors")
        print("   - Everything in Full")
        print("   - Testing frameworks")
        print("   - Code quality tools")
        print("   - Documentation builders")
        print("   - ~4GB download")
        
        print("\n4. CUSTOM - Manual selection")
        print("   - Choose your own requirements file")
        
        print("\n0. EXIT - Cancel installation")
        
        while True:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("Installation cancelled.")
                return None
            elif choice == '1':
                return 'standard'
            elif choice == '2':
                return 'full'
            elif choice == '3':
                return 'development'
            elif choice == '4':
                return self._choose_custom_file()
            else:
                print("Invalid choice. Please enter 0-4.")
                
    def _choose_custom_file(self):
        """Let user choose a custom requirements file"""
        print("\n📄 Available requirements files:")
        
        req_files = list(self.root_path.glob("requirements*.txt"))
        
        for i, file in enumerate(req_files, 1):
            print(f"{i}. {file.name}")
            
        while True:
            choice = input(f"\nSelect file (1-{len(req_files)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(req_files):
                    return req_files[idx].name
            except ValueError:
                pass
            print("Invalid selection.")
            
    def _install_requirements(self, install_type):
        """Install the selected requirements"""
        # Map install types to files
        file_map = {
            'standard': 'requirements.txt',
            'full': 'requirements-full.txt',
            'development': 'requirements-full.txt'  # Start with full
        }
        
        req_file = file_map.get(install_type, install_type)
        req_path = self.root_path / req_file
        
        if not req_path.exists():
            print(f"❌ Requirements file not found: {req_file}")
            return
            
        print(f"\n📥 Installing from: {req_file}")
        print("⏳ This may take several minutes...\n")
        
        # Upgrade pip first
        print("📦 Upgrading pip...")
        subprocess.run([self.python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print(f"\n📦 Installing {install_type} requirements...")
        result = subprocess.run(
            [self.python_cmd, "-m", "pip", "install", "-r", str(req_path)],
            capture_output=False  # Show output to user
        )
        
        if result.returncode != 0:
            print("\n⚠️ Some packages failed to install")
            print("💡 Try running: pip install -r {} --no-deps".format(req_file))
        else:
            print(f"\n✅ {install_type.title()} requirements installed successfully!")
            
        # Install dev requirements if development mode
        if install_type == 'development':
            dev_path = self.root_path / 'requirements-dev.txt'
            if dev_path.exists():
                print("\n📦 Installing development tools...")
                subprocess.run([self.python_cmd, "-m", "pip", "install", "-r", str(dev_path)])
                
        # Platform-specific instructions
        self._show_platform_notes()
        
    def _show_platform_notes(self):
        """Show platform-specific notes"""
        system = platform.system()
        
        print("\n📌 Platform-specific notes:")
        
        if system == "Windows":
            print("- If TA-Lib fails, download wheel from:")
            print("  https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("- Visual C++ Runtime may be required for some packages")
            
        elif system == "Darwin":  # macOS
            print("- If issues with PyQt5, try: brew install pyqt5")
            print("- For TA-Lib: brew install ta-lib")
            
        elif system == "Linux":
            print("- If PyQt5 fails: sudo apt-get install python3-pyqt5")
            print("- For TA-Lib: sudo apt-get install libta-lib-dev")
            print("- May need: sudo apt-get install python3-dev")

if __name__ == "__main__":
    installer = RequirementsInstaller()
    installer.run()
