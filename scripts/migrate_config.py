"""
Configuration Migration Script for Nexlify v2.0.8
Migrates from neural_config.json to enhanced_config.json
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ConfigMigrator:
    def __init__(self):
        self.old_config_path = Path("config/neural_config.json")
        self.new_config_path = Path("config/enhanced_config.json")
        self.backup_dir = Path("backups/config")
        
    def migrate(self) -> bool:
        """Perform configuration migration"""
        print("🔄 Starting configuration migration to v2.0.8...")
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if migration is needed
        if not self.old_config_path.exists() and self.new_config_path.exists():
            print("✅ Already using enhanced_config.json")
            return True
            
        if not self.old_config_path.exists():
            print("⚠️  No existing configuration found. Creating default enhanced_config.json")
            self._create_default_config()
            return True
        
        # Backup old config
        backup_path = self._backup_old_config()
        print(f"📁 Backed up old config to: {backup_path}")
        
        # Load old config
        try:
            with open(self.old_config_path, 'r') as f:
                old_config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading old config: {e}")
            return False
        
        # Migrate to new structure
        new_config = self._migrate_config_structure(old_config)
        
        # Save new config
        try:
            with open(self.new_config_path, 'w') as f:
                json.dump(new_config, f, indent=2)
            print(f"✅ Created enhanced_config.json")
        except Exception as e:
            print(f"❌ Error saving new config: {e}")
            return False
        
        # Keep old config for reference but rename it
        try:
            old_backup = self.old_config_path.with_suffix('.json.old')
            shutil.move(str(self.old_config_path), str(old_backup))
            print(f"📝 Renamed old config to: {old_backup}")
        except Exception as e:
            print(f"⚠️  Could not rename old config: {e}")
        
        print("✅ Migration completed successfully!")
        return True
    
    def _backup_old_config(self) -> Path:
        """Create backup of old configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"neural_config_{timestamp}.json"
        shutil.copy2(self.old_config_path, backup_path)
        return backup_path
    
    def _migrate_config_structure(self, old: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old config structure to new enhanced structure"""
        return {
            "version": "2.0.8",
            "environment": {
                "mode": old.get("mode", "testnet"),
                "debug": old.get("debug", False),
                "log_level": old.get("log_level", "INFO")
            },
            "security": {
                "master_password_enabled": False,  # Optional by default
                "master_password": "",  # User must set via UI
                "2fa_enabled": False,  # Optional by default
                "2fa_secret": "",  # Generated when enabled
                "session_timeout_minutes": old.get("session_timeout", 60),
                "ip_whitelist_enabled": False,
                "ip_whitelist": [],
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30,
                "api_key_rotation_days": 30
            },
            "trading": {
                "mode": old.get("trading_mode", "hybrid"),
                "risk_level": old.get("risk_level", "balanced"),
                "max_position_size": old.get("max_position_size", 1000),
                "stop_loss_percentage": old.get("stop_loss", 2.0),
                "take_profit_percentage": old.get("take_profit", 5.0),
                "trailing_stop_enabled": old.get("trailing_stop", False),
                "daily_loss_limit": old.get("max_daily_loss", 1000),
                "withdraw_profits": old.get("auto_withdraw", True),
                "min_withdrawal": old.get("min_withdrawal", 100),
                "btc_wallet": old.get("btc_wallet", ""),
                "compound_profits": old.get("compound_profits", True)
            },
            "exchanges": {
                "enabled": old.get("exchanges", ["binance", "kraken", "coinbase"]),
                "testnet": old.get("use_testnet", True),
                "credentials": self._migrate_exchange_credentials(old)
            },
            "multi_strategy": {
                "enabled": old.get("multi_strategy_enabled", True),
                "initial_capital": old.get("initial_capital", 10000),
                "rebalance_interval_hours": old.get("rebalance_interval", 24),
                "strategies": {
                    "arbitrage": {
                        "enabled": old.get("arbitrage_enabled", True),
                        "allocation_percentage": old.get("arbitrage_allocation", 30)
                    },
                    "grid_trading": {
                        "enabled": old.get("grid_enabled", True),
                        "allocation_percentage": old.get("grid_allocation", 20)
                    },
                    "momentum": {
                        "enabled": old.get("momentum_enabled", True),
                        "allocation_percentage": old.get("momentum_allocation", 20)
                    },
                    "mean_reversion": {
                        "enabled": old.get("mean_reversion_enabled", True),
                        "allocation_percentage": old.get("mean_reversion_allocation", 20)
                    },
                    "sentiment": {
                        "enabled": old.get("sentiment_enabled", True),
                        "allocation_percentage": old.get("sentiment_allocation", 10)
                    }
                }
            },
            "dex_integration": {
                "enabled": old.get("dex_enabled", False),
                "networks": ["ethereum", "arbitrum", "optimism"],
                "slippage_tolerance": old.get("dex_slippage", 0.5),
                "gas_price_multiplier": 1.2,
                "min_liquidity_usd": 50000
            },
            "predictive_features": {
                "enabled": old.get("ml_enabled", True),
                "models": {
                    "volatility": True,
                    "liquidity": True,
                    "fee_prediction": True,
                    "anomaly_detection": True
                },
                "update_interval_minutes": 15,
                "confidence_threshold": 0.75
            },
            "notifications": {
                "telegram": {
                    "enabled": old.get("telegram_notifications", False),
                    "bot_token": old.get("telegram_bot_token", ""),
                    "chat_id": old.get("telegram_chat_id", ""),
                    "trade_alerts": True,
                    "error_alerts": True,
                    "profit_alerts": True
                },
                "email": {
                    "enabled": old.get("email_notifications", False),
                    "smtp_server": old.get("smtp_server", ""),
                    "smtp_port": old.get("smtp_port", 587),
                    "smtp_user": old.get("smtp_user", ""),
                    "smtp_password": "",  # Must be set via UI
                    "recipient": old.get("emergency_contact", "")
                },
                "mobile_push": {
                    "enabled": False,
                    "min_profit_alert": 100
                }
            },
            "api": {
                "enabled": True,
                "host": old.get("api_host", "0.0.0.0"),
                "port": old.get("api_port", 8000),
                "cors_enabled": True,
                "rate_limit_per_minute": 60
            },
            "mobile": {
                "enabled": True,
                "max_devices": 5,
                "session_duration_days": 30,
                "biometric_auth": True
            },
            "gui": {
                "theme": "cyberpunk_neon",
                "animations_enabled": True,
                "sound_enabled": True,
                "sound_volume": 0.7,
                "refresh_interval_seconds": 5,
                "chart_history_days": 30
            },
            "audit": {
                "enabled": True,
                "retention_days": old.get("audit_retention_days", 2555),  # 7 years
                "blockchain_verification": True,
                "export_format": "json",
                "compliance_mode": "auto"  # auto, mifid2, fatf, custom
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl_seconds": 300,
                "max_cache_size_mb": 500,
                "parallel_processing": True,
                "num_workers": 4,
                "gpu_enabled": False
            },
            "error_handling": {
                "telegram_notifications": old.get("telegram_notifications", False),
                "telegram_bot_token": old.get("telegram_bot_token", ""),
                "telegram_chat_id": old.get("telegram_chat_id", ""),
                "email_notifications": old.get("email_notifications", False),
                "emergency_contact": old.get("emergency_contact", ""),
                "emergency_stop_threshold": 100,
                "error_deduplication_window": 300,
                "suppress_non_critical": True
            },
            "logging": {
                "level": old.get("log_level", "INFO"),
                "error_log": "logs/errors.log",
                "crash_reports": "logs/crash_reports",
                "max_log_size": 100,
                "log_rotation_count": 10
            },
            "database": {
                "url": old.get("database_url", "sqlite:///data/trading.db"),
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30
            },
            "paths": {
                "data": "data",
                "logs": "logs",
                "models": "models",
                "backups": "backups",
                "reports": "reports"
            }
        }
    
    def _migrate_exchange_credentials(self, old: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate exchange credentials structure"""
        creds = {}
        
        # Map old structure to new
        exchange_mappings = {
            "binance": ["binance_api_key", "binance_api_secret"],
            "kraken": ["kraken_api_key", "kraken_api_secret"],
            "coinbase": ["coinbase_api_key", "coinbase_api_secret"],
            "kucoin": ["kucoin_api_key", "kucoin_api_secret"],
            "okx": ["okx_api_key", "okx_api_secret"]
        }
        
        for exchange, [key_field, secret_field] in exchange_mappings.items():
            if old.get(key_field):
                creds[exchange] = {
                    "api_key": old.get(key_field, ""),
                    "api_secret": old.get(secret_field, ""),
                    "testnet": old.get("use_testnet", True)
                }
        
        return creds
    
    def _create_default_config(self):
        """Create default enhanced configuration"""
        default_config = {
            "version": "2.0.8",
            "environment": {
                "mode": "testnet",
                "debug": False,
                "log_level": "INFO"
            },
            "security": {
                "master_password_enabled": False,
                "master_password": "",
                "2fa_enabled": False,
                "2fa_secret": "",
                "session_timeout_minutes": 60,
                "ip_whitelist_enabled": False,
                "ip_whitelist": [],
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30,
                "api_key_rotation_days": 30
            },
            "trading": {
                "mode": "hybrid",
                "risk_level": "balanced",
                "max_position_size": 1000,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 5.0,
                "trailing_stop_enabled": False,
                "daily_loss_limit": 1000,
                "withdraw_profits": True,
                "min_withdrawal": 100,
                "btc_wallet": "",
                "compound_profits": True
            },
            "exchanges": {
                "enabled": ["binance", "kraken", "coinbase"],
                "testnet": True,
                "credentials": {}
            },
            "multi_strategy": {
                "enabled": True,
                "initial_capital": 10000,
                "rebalance_interval_hours": 24,
                "strategies": {
                    "arbitrage": {"enabled": True, "allocation_percentage": 30},
                    "grid_trading": {"enabled": True, "allocation_percentage": 20},
                    "momentum": {"enabled": True, "allocation_percentage": 20},
                    "mean_reversion": {"enabled": True, "allocation_percentage": 20},
                    "sentiment": {"enabled": True, "allocation_percentage": 10}
                }
            },
            "dex_integration": {
                "enabled": False,
                "networks": ["ethereum", "arbitrum", "optimism"],
                "slippage_tolerance": 0.5,
                "gas_price_multiplier": 1.2,
                "min_liquidity_usd": 50000
            },
            "predictive_features": {
                "enabled": True,
                "models": {
                    "volatility": True,
                    "liquidity": True,
                    "fee_prediction": True,
                    "anomaly_detection": True
                },
                "update_interval_minutes": 15,
                "confidence_threshold": 0.75
            },
            "notifications": {
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": "",
                    "trade_alerts": True,
                    "error_alerts": True,
                    "profit_alerts": True
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "smtp_user": "",
                    "smtp_password": "",
                    "recipient": ""
                },
                "mobile_push": {
                    "enabled": False,
                    "min_profit_alert": 100
                }
            },
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8000,
                "cors_enabled": True,
                "rate_limit_per_minute": 60
            },
            "mobile": {
                "enabled": True,
                "max_devices": 5,
                "session_duration_days": 30,
                "biometric_auth": True
            },
            "gui": {
                "theme": "cyberpunk_neon",
                "animations_enabled": True,
                "sound_enabled": True,
                "sound_volume": 0.7,
                "refresh_interval_seconds": 5,
                "chart_history_days": 30
            },
            "audit": {
                "enabled": True,
                "retention_days": 2555,
                "blockchain_verification": True,
                "export_format": "json",
                "compliance_mode": "auto"
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl_seconds": 300,
                "max_cache_size_mb": 500,
                "parallel_processing": True,
                "num_workers": 4,
                "gpu_enabled": False
            },
            "error_handling": {
                "telegram_notifications": False,
                "telegram_bot_token": "",
                "telegram_chat_id": "",
                "email_notifications": False,
                "emergency_contact": "",
                "emergency_stop_threshold": 100,
                "error_deduplication_window": 300,
                "suppress_non_critical": True
            },
            "logging": {
                "level": "INFO",
                "error_log": "logs/errors.log",
                "crash_reports": "logs/crash_reports",
                "max_log_size": 100,
                "log_rotation_count": 10
            },
            "database": {
                "url": "sqlite:///data/trading.db",
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30
            },
            "paths": {
                "data": "data",
                "logs": "logs",
                "models": "models",
                "backups": "backups",
                "reports": "reports"
            }
        }
        
        # Create config directory if needed
        self.new_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default config
        with open(self.new_config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

if __name__ == "__main__":
    migrator = ConfigMigrator()
    success = migrator.migrate()
    
    if success:
        print("\n✨ Configuration migration completed!")
        print("📝 Please review config/enhanced_config.json")
        print("🔐 Security settings can be configured via the GUI")
    else:
        print("\n❌ Migration failed. Please check the errors above.")
