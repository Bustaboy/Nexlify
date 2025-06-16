# nexlify/config/config_manager.py
"""
Nexlify Configuration Manager - The Neural Core
Handles all config ops like a fixer handles Night City deals
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet
from pydantic import BaseSettings, Field, validator
import secrets
import hashlib
from datetime import datetime, timezone

class SecurityConfig(BaseSettings):
    """Security protocols tighter than Arasaka HQ"""
    
    # Dynamic PIN generation - no more hardcoded "2077" nonsense
    pin_length: int = Field(default=6, ge=4, le=12)
    pin_salt: str = Field(default_factory=lambda: secrets.token_hex(32))
    
    # Encryption keys - generated fresh, stored encrypted
    master_key: Optional[str] = Field(default=None)
    api_key_encryption: bool = Field(default=True)
    
    # 2FA Config
    enable_2fa: bool = Field(default=True)
    totp_issuer: str = Field(default="Nexlify_TradingOS")
    backup_codes_count: int = Field(default=10)
    
    # Rate limiting - stop the script kiddies
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)  # seconds
    
    # IP Security
    ip_whitelist_enabled: bool = Field(default=False)
    allowed_ips: list[str] = Field(default_factory=lambda: ["127.0.0.1"])
    
    # Session management
    session_timeout: int = Field(default=3600)  # 1 hour
    refresh_token_expiry: int = Field(default=604800)  # 7 days
    
    @validator('master_key', pre=True, always=True)
    def generate_master_key(cls, v):
        if not v:
            return Fernet.generate_key().decode()
        return v
    
    class Config:
        env_prefix = "NEXLIFY_SECURITY_"

class DatabaseConfig(BaseSettings):
    """Database config - PostgreSQL ready for the big leagues"""
    
    db_type: str = Field(default="postgresql")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    username: str = Field(default="nexlify_user")
    password: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    database: str = Field(default="nexlify_trading")
    
    # Connection pooling for performance
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=40)
    pool_timeout: int = Field(default=30)
    
    # Backup config
    auto_backup: bool = Field(default=True)
    backup_interval: int = Field(default=86400)  # 24 hours
    backup_retention_days: int = Field(default=30)
    
    @property
    def connection_string(self) -> str:
        """Generate secure connection string"""
        return f"{self.db_type}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "NEXLIFY_DB_"

class PerformanceConfig(BaseSettings):
    """Performance tuning - make it run like a street samurai"""
    
    # GUI Performance
    enable_gpu_acceleration: bool = Field(default=True)
    animation_fps: int = Field(default=60)
    reduce_effects_on_low_spec: bool = Field(default=True)
    
    # Queue Management
    max_queue_size: int = Field(default=10000)
    queue_overflow_strategy: str = Field(default="circular_buffer")
    worker_threads: int = Field(default=os.cpu_count() or 4)
    
    # Caching
    enable_redis_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=300)  # 5 minutes
    
    # Backtesting optimization
    backtest_chunk_size: int = Field(default=1000)
    parallel_backtests: int = Field(default=4)
    use_gpu_backtesting: bool = Field(default=True)
    
    class Config:
        env_prefix = "NEXLIFY_PERF_"

class MLConfig(BaseSettings):
    """Machine Learning config - NetRunner grade AI"""
    
    # Model settings
    model_architecture: str = Field(default="transformer")
    model_checkpoint_dir: Path = Field(default=Path("./models/checkpoints"))
    
    # Training params
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    epochs: int = Field(default=100)
    early_stopping_patience: int = Field(default=10)
    
    # Real-time inference
    inference_device: str = Field(default="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu")
    model_quantization: bool = Field(default=True)
    
    # Feature engineering
    feature_window: int = Field(default=100)  # candles
    technical_indicators: list[str] = Field(default_factory=lambda: [
        "RSI", "MACD", "BB", "EMA", "SMA", "ATR", "OBV", "VWAP"
    ])
    
    # Model validation
    validate_predictions: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.7)
    
    class Config:
        env_prefix = "NEXLIFY_ML_"

class MonitoringConfig(BaseSettings):
    """Real-time monitoring - know everything, miss nothing"""
    
    # Metrics collection
    enable_prometheus: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_rotation: str = Field(default="1 day")
    log_retention: str = Field(default="30 days")
    
    # Alerting
    enable_alerts: bool = Field(default=True)
    alert_channels: list[str] = Field(default_factory=lambda: ["email", "webhook"])
    alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0,
        "latency_p99": 1000.0  # ms
    })
    
    # Tracing
    enable_jaeger: bool = Field(default=True)
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces")
    
    class Config:
        env_prefix = "NEXLIFY_MONITOR_"

class NexlifyConfig:
    """Main configuration brain - coordinates all subsystems"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("./config/nexlify.yaml")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.performance = PerformanceConfig()
        self.ml = MLConfig()
        self.monitoring = MonitoringConfig()
        
        # Encryption for sensitive data
        self._cipher = Fernet(self.security.master_key.encode())
        
        # Load existing config or create new
        self.load_or_create_config()
    
    def load_or_create_config(self):
        """Load existing config or create fresh one"""
        if self.config_path.exists():
            self.load_config()
        else:
            self.save_config()
            print(f"[{datetime.now(timezone.utc).isoformat()}] Fresh config initialized at {self.config_path}")
    
    def load_config(self):
        """Load and decrypt configuration"""
        try:
            with open(self.config_path, 'r') as f:
                encrypted_data = yaml.safe_load(f)
            
            # Decrypt sensitive fields
            if encrypted_data.get('encrypted_fields'):
                for field_path in encrypted_data['encrypted_fields']:
                    self._decrypt_field(encrypted_data, field_path)
            
            # Update configurations
            self._update_from_dict(encrypted_data)
            
        except Exception as e:
            print(f"[ERROR] Config load failed: {e}")
            raise
    
    def save_config(self):
        """Save configuration with encryption"""
        config_dict = self.to_dict()
        
        # Encrypt sensitive fields
        sensitive_fields = [
            'security.pin_salt',
            'security.master_key',
            'database.password',
            'monitoring.alert_channels'
        ]
        
        encrypted_fields = []
        for field_path in sensitive_fields:
            if self._encrypt_field(config_dict, field_path):
                encrypted_fields.append(field_path)
        
        config_dict['encrypted_fields'] = encrypted_fields
        config_dict['config_version'] = "2.0.0"
        config_dict['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Atomic write
        temp_path = self.config_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        temp_path.replace(self.config_path)
    
    def _encrypt_field(self, data: dict, field_path: str) -> bool:
        """Encrypt a specific field in nested dict"""
        parts = field_path.split('.')
        current = data
        
        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]
        
        if parts[-1] in current:
            value = str(current[parts[-1]])
            current[parts[-1]] = self._cipher.encrypt(value.encode()).decode()
            return True
        return False
    
    def _decrypt_field(self, data: dict, field_path: str) -> bool:
        """Decrypt a specific field in nested dict"""
        parts = field_path.split('.')
        current = data
        
        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]
        
        if parts[-1] in current:
            try:
                encrypted_value = current[parts[-1]].encode()
                current[parts[-1]] = self._cipher.decrypt(encrypted_value).decode()
                return True
            except:
                return False
        return False
    
    def to_dict(self) -> dict:
        """Convert all configs to dictionary"""
        return {
            'security': self.security.dict(),
            'database': self.database.dict(),
            'performance': self.performance.dict(),
            'ml': self.ml.dict(),
            'monitoring': self.monitoring.dict()
        }
    
    def _update_from_dict(self, data: dict):
        """Update configurations from dictionary"""
        if 'security' in data:
            self.security = SecurityConfig(**data['security'])
        if 'database' in data:
            self.database = DatabaseConfig(**data['database'])
        if 'performance' in data:
            self.performance = PerformanceConfig(**data['performance'])
        if 'ml' in data:
            self.ml = MLConfig(**data['ml'])
        if 'monitoring' in data:
            self.monitoring = MonitoringConfig(**data['monitoring'])
    
    def generate_pin(self) -> tuple[str, str]:
        """Generate secure PIN with hash"""
        pin = ''.join(secrets.choice('0123456789') for _ in range(self.security.pin_length))
        pin_hash = hashlib.pbkdf2_hmac(
            'sha256',
            pin.encode(),
            self.security.pin_salt.encode(),
            100000
        )
        return pin, pin_hash.hex()
    
    def verify_pin(self, pin: str, pin_hash: str) -> bool:
        """Verify PIN against hash"""
        test_hash = hashlib.pbkdf2_hmac(
            'sha256',
            pin.encode(),
            self.security.pin_salt.encode(),
            100000
        )
        return test_hash.hex() == pin_hash
    
    def get_system_status(self) -> dict:
        """Get current system configuration status"""
        return {
            'config_version': '2.0.0',
            'security_enabled': self.security.enable_2fa,
            'database_type': self.database.db_type,
            'performance_mode': 'GPU' if self.performance.enable_gpu_acceleration else 'CPU',
            'ml_device': self.ml.inference_device,
            'monitoring_active': self.monitoring.enable_prometheus
        }

# Singleton instance
_config_instance: Optional[NexlifyConfig] = None

def get_config() -> NexlifyConfig:
    """Get or create config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = NexlifyConfig()
    return _config_instance
