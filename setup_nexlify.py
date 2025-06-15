#!/usr/bin/env python3
"""
Nexlify - Automated Setup Script
Installs dependencies and configures the trading matrix
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

class NexlifySetup:
    def __init__(self):
        self.project_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self.python_version = sys.version_info
        
        # Required packages
        self.requirements = [
            'ccxt>=4.0.0',
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'matplotlib>=3.7.0',
            'aiohttp>=3.8.0',
            'python-dotenv>=1.0.0',
            'colorama>=0.4.6',
            'asyncio>=3.4.3',
            'websockets>=11.0',
            'scikit-learn>=1.3.0',
            'xgboost>=2.0.0',
            'tensorflow>=2.13.0',
            'ta>=0.10.2',
            'schedule>=1.2.0'
        ]
        
    def print_banner(self):
        """Print cyberpunk banner"""
        print("\033[92m" + """
        ███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗
        ████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝
        ██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝ 
        ██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝  
        ██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║   
        ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝   
                          SETUP WIZARD v2.0.7.7
        """ + "\033[0m")
        print("\033[93m" + "="*70 + "\033[0m")
    
    def check_python(self):
        """Check Python version"""
        print("🔍 Checking Python version...")
        if self.python_version < (3, 9):
            print(f"❌ Python {self.python_version.major}.{self.python_version.minor} detected")
            print("❌ Python 3.9+ required!")
            return False
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")
        return True
    
    def create_directories(self):
        """Create project directory structure"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            self.project_root / "config",
            self.project_root / "logs",
            self.project_root / "logs" / "crash_reports",
            self.project_root / "data",
            self.project_root / "models",
            self.project_root / "backups",
            self.project_root / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def install_requirements(self):
        """Install Python packages"""
        print("\n📦 Installing dependencies...")
        
        # Create requirements.txt
        req_file = self.project_root / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write('\n'.join(self.requirements))
        
        # Install packages
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
            ])
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install some packages")
            return False
    
    def create_config_files(self):
        """Create configuration files"""
        print("\n⚙️ Creating configuration files...")
        
        # Main config - No API keys, those are entered through GUI
        config = {
            "exchanges": {},  # Will be populated through GUI
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
            "btc_wallet_address": "",  # Set through GUI
            "auto_withdraw": {
                "enabled": False,
                "min_amount_usd": 100,
                "schedule": "monthly"
            },
            "risk_level": "low",  # Default to low risk
            "auto_trade": True,
            "environment": {  # Environment settings managed by GUI
                "debug": False,
                "log_level": "INFO",
                "api_port": 8000,
                "database_url": "sqlite:///data/trading.db",
                "emergency_contact": "",
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            }
        }
        
        config_file = self.project_root / "config" / "neural_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created: {config_file}")
        print("📝 Note: All settings will be configured through the GUI")
        
        # Create minimal .env file (for compatibility, but managed by GUI)
        env_content = """# Nexlify Environment Configuration
# ⚠️ DO NOT EDIT THIS FILE MANUALLY!
# All settings are managed through the GUI
# Go to Environment tab to change these values

# This file is auto-generated and will be overwritten
# Use the GUI's Environment tab for all configuration
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created: {env_file} (managed by GUI)")
    
    def create_github_files(self):
        """Create GitHub-specific files"""
        print("\n📝 Creating GitHub files...")
        
        # README.md
        readme_content = """# 🌃 Nexlify - Arasaka Neural-Net Trading Matrix

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> "In Night City, you're either the hunter or the prey. This Neural-Net ensures you're always the hunter."

## 🚀 Features

- **100% Autonomous Trading**: AI-driven pair selection and execution
- **Multi-Exchange Arbitrage**: Automatic opportunity detection across exchanges
- **Cyberpunk GUI**: Real-time visualization with full configuration management
- **Smart Fee Calculation**: All trades account for gas and exchange fees
- **Custom BTC Withdrawals**: Set your wallet, automate your profits
- **Neural Confidence Scoring**: AI ranks pairs by profit potential
- **Emergency Kill Switch**: Instant shutdown for safety
- **All Settings in GUI**: No config files to edit - everything through the interface!

## 🛠️ Installation

### Quick Start (Windows)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nexlify.git
cd nexlify
```

2. Run the setup script:
```bash
python setup_nexlify.py
```

3. Launch the trader:
```bash
python nexlify_launcher.py
```

4. **First Launch**:
   - GUI shows onboarding screen
   - Enter your exchange API keys
   - Set your BTC wallet address
   - Configure environment settings
   - Click "JACK INTO THE MATRIX"!

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir config logs data models backups reports

# Run the smart launcher
python nexlify_launcher.py
```

## 📋 Configuration

**Everything is configured through the GUI!**

- **API Keys**: Enter on first launch or in "🔐 API CONFIG" tab
- **BTC Wallet**: Set in control panel
- **Environment**: Configure in "🌐 ENVIRONMENT" tab
- **Risk Settings**: Adjust in "⚙️ NEURAL CONFIG" tab

No manual file editing required!

## 🎮 Usage

1. **Start the System**: Run `nexlify_launcher.py`
2. **Enter PIN**: Default is `2077`
3. **Monitor**: Watch the AI work in real-time
4. **Adjust**: All settings accessible through GUI tabs

## 🛡️ Security

- PIN-protected access (change default!)
- Encrypted API keys storage
- Emergency kill switch
- Automatic position limits
- Testnet mode for safe testing

## 📊 GUI Features

- **Active Pairs Tab**: Real-time view of all trading pairs
- **Profit Matrix**: Visual profit tracking
- **Neural Config**: Risk and trading settings
- **Environment**: Debug, logging, and notifications
- **API Config**: Manage exchange credentials
- **Neural Logs**: Monitor all system activities

## ⚠️ Disclaimer

Trading cryptocurrencies involves significant risk. This bot is for educational purposes. Always test thoroughly on testnet before using real funds. The developers are not responsible for any financial losses.

## 🤝 Contributing

Pull requests welcome! Please test all changes on testnet first.

## 📜 License

MIT License - see LICENSE file

---

*Built with 💚 in Night City*
"""
        
        readme_file = self.project_root / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print(f"✅ Created: {readme_file}")
    
    def final_instructions(self):
        """Print final setup instructions"""
        print("\n" + "="*70)
        print("\033[92m✅ SETUP COMPLETE!\033[0m")
        print("\n📋 Next Steps:")
        print("1. Run: python nexlify_launcher.py (or start_nexlify.bat)")
        print("2. The GUI will guide you through initial setup")
        print("3. Enter your exchange API keys in the onboarding screen")
        print("4. Set your BTC wallet address")
        print("5. Choose your risk level")
        print("6. Click 'JACK INTO THE MATRIX' to start trading")
        print("\n⚠️  IMPORTANT:")
        print("- Default PIN: 2077 (change this in settings!)")
        print("- Start with TESTNET mode enabled!")
        print("- Test thoroughly before using real funds")
        print("- Monitor logs/neural_net.log for issues")
        print("\n🚀 To upload to GitHub:")
        print("1. Create new repo on GitHub called 'nexlify'")
        print("2. Run: git init")
        print("3. Run: git add .")
        print("4. Run: git commit -m 'Initial commit - Nexlify v2.0.7.7'")
        print("5. Run: git remote add origin YOUR_GITHUB_URL")
        print("6. Run: git push -u origin main")
        print("\n" + "="*70)
    
    def run(self):
        """Run the complete setup process"""
        self.print_banner()
        
        if not self.check_python():
            return False
        
        self.create_directories()
        
        if not self.install_requirements():
            print("\n⚠️  Some packages failed to install")
            print("Try: pip install -r requirements.txt manually")
        
        self.create_config_files()
        self.create_github_files()
        
        print("\n📂 Don't forget to copy:")
        print("✅ arasaka_neural_net.py (main trading engine)")
        print("✅ cyber_gui.py (GUI interface)")
        print("✅ error_handler.py (error management)")
        print("✅ utils.py (utility functions)")
        
        self.final_instructions()
        return True

if __name__ == "__main__":
    setup = NexlifySetup()
    setup.run()
    input("\nPress Enter to exit...")
