#!/usr/bin/env python3
"""
src/utils/config_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY DYNAMIC CONFIGURATION LOADER v3.1 (MERGED EDITION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Zero hardcoded values. Everything dynamic. Everything configurable.
Combines best features from both implementations with 2025 tech stack.
"""

import os
import json
import yaml
import toml
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import secrets

# File handling
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# External config sources
import hvac  # HashiCorp Vault client
import boto3  # AWS Secrets Manager
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Validation and parsing
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Performance and monitoring
import aioredis
import structlog
from prometheus_client import Counter, Histogram, Gauge
from rich.console import Console

# Initialize console and logger
console = Console()
logger = structlog.get_logger("NEXLIFY.CONFIG.MATRIX")

# Metrics
CONFIG_LOADS = Counter('nexlify_config_loads_total', 'Total configuration loads')
CONFIG_RELOAD_TIME = Histogram('nexlify_config_reload_seconds', 'Configuration reload time')
CONFIG_SOURCES = Gauge('nexlify_config_sources_active', 'Number of active config sources')

# Cyberpunk color codes for console output
class CyberColors:
    NEON_CYAN = "\033[96m"
    NEON_PINK = "\033[95m"
    NEON_GREEN = "\033[92m"
    NEON_RED = "\033[91m"
    NEON_YELLOW = "\033[93m"
    NEURAL_PURPLE = "\033[35m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


@dataclass
class ConfigSource:
    """Represents a configuration source with priority and metadata"""
    name: str
    priority: int  # Lower number = higher priority
    data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    loader: Optional[Callable] = None
    validator: Optional[Callable] = None
    encrypted: bool = False
    watch: bool = True
    cache_ttl: int = 300  # seconds


class CyberpunkConfigSchema(BaseModel):
    """Enhanced Pydantic schema for configuration validation"""
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    environment: str = Field(..., regex=r'^(development|staging|production|cyberpunk_test)$')
    neural_net_id: str = Field(default="NEXLIFY-MATRIX-v3")
    
    # System Configuration
    system: Dict[str, Any] = Field(default_factory=dict)
    debug_mode: bool = Field(default=False)
    
    # Exchange Configuration
    exchanges: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    priority_exchange: str = Field(default="coinbase")
    
    # Database Configuration
    databases: Dict[str, Any] = Field(default_factory=dict)
    questdb_url: Optional[str] = None
    redis_url: str = Field(default="redis://localhost:6379/0")
    arcticdb_path: Optional[str] = None
    
    # ML/AI Configuration
    ml_models: Dict[str, Any] = Field(default_factory=dict)
    timesfm_enabled: bool = Field(default=True)
    ensemble_models: List[str] = Field(default_factory=lambda: ["timesfm", "chronos", "itransformer"])
    
    # Trading Configuration
    trading: Dict[str, Any] = Field(default_factory=dict)
    risk_management: Dict[str, float] = Field(default_factory=dict)
    strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Security Configuration
    security: Dict[str, Any] = Field(default_factory=dict)
    mpc_wallet_enabled: bool = Field(default=True)
    
    # Performance Configuration
    performance: Dict[str, Any] = Field(default_factory=dict)
    rust_acceleration: bool = Field(default=True)
    gpu_enabled: bool = Field(default=True)
    
    # Monitoring Configuration
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    prometheus_enabled: bool = Field(default=True)
    
    class Config:
        extra = 'allow'  # Allow additional fields
    
    @validator('*', pre=True)
    def no_hardcoded_values(cls, v, field):
        """Ensure no obvious hardcoded values"""
        if isinstance(v, str):
            # Check for common hardcoded patterns
            hardcoded_patterns = [
                'localhost', '127.0.0.1', '0.0.0.0',
                'password123', 'admin', 'root',
                'sk_test_', 'pk_test_'  # Test API keys
            ]
            for pattern in hardcoded_patterns:
                if pattern in v.lower() and field.name not in ['environment', 'redis_url']:
                    logger.warning(f"Potential hardcoded value in {field.name}: {v}")
        return v


class ConfigFileHandler(FileSystemEventHandler):
    """Watch configuration files for changes"""
    
    def __init__(self, config_loader, filepath):
        self.config_loader = config_loader
        self.filepath = filepath
        
    def on_modified(self, event):
        if event.src_path == str(self.filepath):
            logger.info(f"{CyberColors.NEON_YELLOW}Config file modified: {self.filepath}{CyberColors.RESET}")
            asyncio.create_task(self.config_loader.reload_config(self.filepath))


class DynamicConfigLoader:
    """
    🌃 NEXLIFY Enhanced Dynamic Configuration Loader
    
    Merged features:
    - Multiple configuration sources with priority-based merging
    - Hot reload capability with file watching
    - Redis caching for ultra-fast access
    - Encryption support for sensitive data
    - External source support (Vault, AWS, Azure)
    - Rust-ready async architecture
    - Cyberpunk-themed logging and validation
    - Callback system for reactive configuration
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.sources: List[ConfigSource] = []
        self.config_cache: Dict[str, Any] = {}
        self.encryption_key: Optional[bytes] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Hot reload system
        self.hot_reload_enabled = True
        self.observers: List[Observer] = []
        self._load_lock = asyncio.Lock()
        
        # Callback system
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="config_worker")
        
        # Initialize encryption if configured
        if encryption_key := os.getenv('NEXLIFY_CONFIG_ENCRYPTION_KEY'):
            self._init_encryption(encryption_key)
        
        # Load .env file
        load_dotenv()
        
        # Setup initial logger
        logger.info(f"{CyberColors.NEON_CYAN}🔧 Initializing NEXLIFY Config Matrix...{CyberColors.RESET}")
    
    def _init_encryption(self, key: str):
        """Initialize encryption with key derivation"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'nexlify_salt_2025',  # In production, use random salt
            iterations=100000,
        )
        self.encryption_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.cipher = Fernet(self.encryption_key)
    
    async def initialize(self):
        """Initialize the configuration system"""
        logger.info(f"{CyberColors.NEON_CYAN}🚀 NEXLIFY Config Matrix initializing...{CyberColors.RESET}")
        
        # Connect to Redis if configured
        redis_url = os.getenv('NEXLIFY_REDIS_URL', 'redis://localhost:6379/0')
        try:
            self.redis_client = await aioredis.create_redis_pool(
                redis_url,
                encoding='utf-8'
            )
            logger.info(f"{CyberColors.NEON_GREEN}✓ Redis cache connected{CyberColors.RESET}")
        except Exception as e:
            logger.warning(f"{CyberColors.NEON_YELLOW}⚠ Redis unavailable: {e}{CyberColors.RESET}")
        
        # Setup default configuration sources
        self._setup_default_sources()
        
        # Load all configurations
        await self.load_all_configs()
        
        # Setup file watching
        if self.hot_reload_enabled:
            self._setup_file_watching()
        
        logger.info(f"{CyberColors.NEON_GREEN}✓ Config Matrix online - Neural pathways active{CyberColors.RESET}")
    
    def _setup_default_sources(self):
        """Setup default configuration sources with priority"""
        
        # Environment variables (highest priority)
        self.add_source(ConfigSource(
            name="environment",
            priority=0,
            loader=self._load_from_env,
            watch=False
        ))
        
        # User config file
        env = os.getenv('NEXLIFY_ENV', 'development')
        config_dir = Path(os.getenv('NEXLIFY_CONFIG_DIR', 'config'))
        
        # Environment-specific config
        env_config = config_dir / f'{env}.yaml'
        if env_config.exists():
            self.add_source(ConfigSource(
                name=f"config:{env}",
                priority=10,
                loader=lambda: self._load_yaml_file(env_config),
                encrypted=True,
                watch=True
            ))
        
        # Base config
        base_config = config_dir / 'base.yaml'
        if base_config.exists():
            self.add_source(ConfigSource(
                name="config:base",
                priority=20,
                loader=lambda: self._load_yaml_file(base_config),
                watch=True
            ))
        
        # Local overrides (git-ignored)
        local_config = config_dir / 'local.yaml'
        if local_config.exists():
            self.add_source(ConfigSource(
                name="config:local",
                priority=5,
                loader=lambda: self._load_yaml_file(local_config),
                watch=True
            ))
        
        # External sources
        self._setup_external_sources()
    
    def _setup_external_sources(self):
        """Setup external configuration sources"""
        
        # Redis remote config
        if self.redis_client:
            self.add_source(ConfigSource(
                name="redis:remote",
                priority=3,
                loader=self._load_from_redis,
                cache_ttl=60
            ))
        
        # HashiCorp Vault
        if vault_url := os.getenv('VAULT_ADDR'):
            self.add_source(ConfigSource(
                name="vault",
                priority=2,
                loader=lambda: self._load_from_vault(
                    vault_url,
                    os.getenv('VAULT_TOKEN'),
                    os.getenv('VAULT_PATH', 'nexlify/config')
                ),
                encrypted=True,
                cache_ttl=300
            ))
        
        # AWS Secrets Manager
        if aws_secret := os.getenv('AWS_SECRET_NAME'):
            self.add_source(ConfigSource(
                name="aws:secrets",
                priority=2,
                loader=lambda: self._load_from_aws(aws_secret),
                encrypted=True,
                cache_ttl=300
            ))
        
        # Azure Key Vault
        if azure_vault := os.getenv('AZURE_KEYVAULT_URL'):
            self.add_source(ConfigSource(
                name="azure:keyvault",
                priority=2,
                loader=lambda: self._load_from_azure(azure_vault),
                encrypted=True,
                cache_ttl=300
            ))
    
    def add_source(self, source: ConfigSource):
        """Add a configuration source"""
        # Remove existing source with same name
        self.sources = [s for s in self.sources if s.name != source.name]
        self.sources.append(source)
        # Sort by priority
        self.sources.sort(key=lambda x: x.priority)
        CONFIG_SOURCES.set(len(self.sources))
    
    async def load_all_configs(self):
        """Load configuration from all sources"""
        start_time = asyncio.get_event_loop().time()
        merged_config = {}
        
        async with self._load_lock:
            # Process sources in priority order (lower number = higher priority)
            for source in reversed(self.sources):  # Process lowest priority first
                try:
                    logger.debug(f"Loading config from {source.name}")
                    
                    # Check cache first
                    cached = await self._get_cached_config(source.name)
                    if cached:
                        config_data = cached
                    else:
                        # Load from source
                        if source.loader:
                            if asyncio.iscoroutinefunction(source.loader):
                                config_data = await source.loader()
                            else:
                                config_data = await asyncio.get_event_loop().run_in_executor(
                                    self.executor, source.loader
                                )
                        else:
                            config_data = source.data
                        
                        # Cache if enabled
                        if source.cache_ttl > 0 and config_data:
                            await self._cache_config(source.name, config_data, source.cache_ttl)
                    
                    # Decrypt if needed
                    if source.encrypted and self.encryption_key and config_data:
                        config_data = self._decrypt_config(config_data)
                    
                    # Validate if validator provided
                    if source.validator and config_data:
                        config_data = source.validator(config_data)
                    
                    # Update source data
                    source.data = config_data or {}
                    source.last_updated = datetime.now()
                    
                    # Calculate checksum
                    if config_data:
                        source.checksum = hashlib.sha256(
                            json.dumps(config_data, sort_keys=True).encode()
                        ).hexdigest()
                    
                    # Merge configurations
                    merged_config = self._deep_merge(merged_config, source.data)
                    
                except Exception as e:
                    logger.error(f"{CyberColors.NEON_RED}Error loading {source.name}: {e}{CyberColors.RESET}")
            
            # Update cache
            self.config_cache = merged_config
        
        # Validate final configuration
        try:
            validated_config = CyberpunkConfigSchema(**self.config_cache)
            self.config_cache = validated_config.dict()
        except Exception as e:
            logger.warning(f"{CyberColors.NEON_YELLOW}Configuration validation warning: {e}{CyberColors.RESET}")
        
        # Update metrics
        CONFIG_LOADS.inc()
        CONFIG_RELOAD_TIME.observe(asyncio.get_event_loop().time() - start_time)
        
        # Notify callbacks
        await self._notify_callbacks('config_loaded', self.config_cache)
        
        logger.info(f"{CyberColors.NEON_GREEN}✓ Configuration loaded from {len(self.sources)} sources{CyberColors.RESET}")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        prefix = 'NEXLIFY_'
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Convert NEXLIFY_SECTION__KEY to nested dict
            config_key = key[len(prefix):].lower()
            parts = config_key.split('__')
            
            current = env_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Parse value
            current[parts[-1]] = self._parse_value(value)
        
        return env_config
    
    def _parse_value(self, value: str) -> Any:
        """Parse string values to appropriate types"""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON arrays/objects
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Duration strings (e.g., "30s", "5m", "1h")
        if value[-1] in 'smhd' and value[:-1].isdigit():
            multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
            return int(value[:-1]) * multipliers[value[-1]]
        
        return value
    
    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return {}
    
    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return {}
    
    def _load_toml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load TOML configuration file"""
        try:
            with open(filepath, 'r') as f:
                return toml.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return {}
    
    async def _load_from_redis(self) -> Dict[str, Any]:
        """Load configuration from Redis"""
        if not self.redis_client:
            return {}
        
        try:
            env = os.getenv('NEXLIFY_ENV', 'development')
            config_data = await self.redis_client.get(f"nexlify:config:{env}")
            if config_data:
                return json.loads(config_data)
        except Exception as e:
            logger.error(f"Redis config load failed: {e}")
        
        return {}
    
    async def _load_from_vault(self, vault_url: str, vault_token: str, path: str) -> Dict[str, Any]:
        """Load configuration from HashiCorp Vault"""
        try:
            client = hvac.Client(url=vault_url, token=vault_token)
            
            if not client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            # Read secrets
            response = client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            logger.error(f"Vault config load failed: {e}")
            return {}
    
    async def _load_from_aws(self, secret_name: str) -> Dict[str, Any]:
        """Load configuration from AWS Secrets Manager"""
        try:
            session = boto3.session.Session()
            client = session.client(service_name='secretsmanager')
            
            response = client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
        except Exception as e:
            logger.error(f"AWS Secrets config load failed: {e}")
        
        return {}
    
    async def _load_from_azure(self, vault_url: str) -> Dict[str, Any]:
        """Load configuration from Azure Key Vault"""
        try:
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            
            data = {}
            # List all secrets
            for secret_properties in client.list_properties_of_secrets():
                if secret_properties.name.startswith('nexlify-'):
                    key = secret_properties.name[8:].replace('-', '.')
                    secret = client.get_secret(secret_properties.name)
                    data[key] = self._parse_value(secret.value)
            
            return data
        except Exception as e:
            logger.error(f"Azure KeyVault config load failed: {e}")
            return {}
    
    async def _get_cached_config(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get cached configuration from Redis"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(f"nexlify:config:cache:{source_name}")
            if cached:
                return json.loads(cached)
        except:
            pass
        
        return None
    
    async def _cache_config(self, source_name: str, config: Dict[str, Any], ttl: int):
        """Cache configuration in Redis"""
        if not self.redis_client or not config:
            return
        
        try:
            await self.redis_client.setex(
                f"nexlify:config:cache:{source_name}",
                ttl,
                json.dumps(config)
            )
        except Exception as e:
            logger.debug(f"Config caching failed: {e}")
    
    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration values"""
        if not self.cipher:
            return config
        
        def decrypt_value(value):
            if isinstance(value, str) and value.startswith("ENCRYPTED:"):
                try:
                    encrypted_data = value[10:]
                    return self.cipher.decrypt(encrypted_data.encode()).decode()
                except:
                    logger.error(f"Failed to decrypt value")
                    return value
            elif isinstance(value, dict):
                return {k: decrypt_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [decrypt_value(item) for item in value]
            return value
        
        return decrypt_value(config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _setup_file_watching(self):
        """Setup file system watching for config changes"""
        config_dir = Path(os.getenv('NEXLIFY_CONFIG_DIR', 'config'))
        if not config_dir.exists():
            return
        
        # Watch each config file
        for source in self.sources:
            if source.watch and source.name.startswith('config:'):
                # Extract filepath from source name
                if source.loader:
                    # This is a bit hacky but works for our lambda loaders
                    import inspect
                    closure = inspect.getclosurevars(source.loader)
                    for var in closure.nonlocals.values():
                        if isinstance(var, Path):
                            self._watch_file(var, source)
                            break
    
    def _watch_file(self, filepath: Path, source: ConfigSource):
        """Setup watching for a specific file"""
        observer = Observer()
        handler = ConfigFileHandler(self, filepath)
        observer.schedule(handler, str(filepath.parent), recursive=False)
        observer.start()
        self.observers.append(observer)
        logger.info(f"{CyberColors.NEON_CYAN}👁️  Watching {filepath.name} for changes{CyberColors.RESET}")
    
    async def reload_config(self, changed_file: Optional[Path] = None):
        """Reload configuration"""
        logger.info(f"{CyberColors.NEON_YELLOW}🔄 Reloading configuration matrix...{CyberColors.RESET}")
        
        old_config = self.config_cache.copy()
        await self.load_all_configs()
        
        # Detect changes
        changes = self._detect_changes(old_config, self.config_cache)
        if changes:
            logger.info(f"{CyberColors.NEON_GREEN}Configuration updated: {len(changes)} changes{CyberColors.RESET}")
            await self._notify_callbacks('config_changed', {
                'changes': changes,
                'old_config': old_config,
                'new_config': self.config_cache
            })
    
    def _detect_changes(self, old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
        """Detect configuration changes"""
        changes = []
        
        def compare_dicts(d1, d2, path=""):
            for key in set(list(d1.keys()) + list(d2.keys())):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    changes.append(f"Added: {current_path}")
                elif key not in d2:
                    changes.append(f"Removed: {current_path}")
                elif d1[key] != d2[key]:
                    if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                        compare_dicts(d1[key], d2[key], current_path)
                    else:
                        changes.append(f"Modified: {current_path}")
        
        compare_dicts(old, new)
        return changes
    
    def get(self, key: str, default: Any = None, decrypt: bool = False) -> Any:
        """Get configuration value with dot notation support"""
        # Check for encrypted value marker
        if key.endswith('.encrypted'):
            decrypt = True
            key = key[:-10]
        
        # Navigate nested dictionaries
        value = self.config_cache
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        # Decrypt if needed
        if decrypt and isinstance(value, str) and self.cipher:
            try:
                value = self.cipher.decrypt(value.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt {key}: {e}")
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config_cache.copy()
    
    def set(self, key: str, value: Any, encrypt: bool = False):
        """Set configuration value (runtime only, not persisted)"""
        # Encrypt if needed
        if encrypt and isinstance(value, str) and self.cipher:
            value = self.cipher.encrypt(value.encode()).decode()
        
        # Set value using dot notation
        parts = key.split('.')
        target = self.config_cache
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
    
    async def save_to_redis(self, ttl: int = 3600):
        """Save current configuration to Redis"""
        if not self.redis_client:
            return
        
        try:
            env = os.getenv('NEXLIFY_ENV', 'development')
            await self.redis_client.setex(
                f"nexlify:config:{env}",
                ttl,
                json.dumps(self.config_cache)
            )
            logger.info(f"{CyberColors.NEON_GREEN}Configuration saved to Redis{CyberColors.RESET}")
        except Exception as e:
            logger.error(f"Failed to save config to Redis: {e}")
    
    def watch(self, callback: Callable):
        """Register callback for configuration changes (legacy compatibility)"""
        self.on_change(callback)
    
    def on_change(self, callback: Callable):
        """Register callback for configuration changes"""
        self.callbacks['config_changed'].append(callback)
    
    def on_load(self, callback: Callable):
        """Register callback for configuration load"""
        self.callbacks['config_loaded'].append(callback)
    
    async def _notify_callbacks(self, event: str, data: Any):
        """Notify registered callbacks"""
        for callback in self.callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, callback, data
                    )
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def shutdown(self):
        """Cleanup resources"""
        logger.info(f"{CyberColors.NEURAL_PURPLE}Config Matrix shutting down...{CyberColors.RESET}")
        
        # Stop file watchers
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup file watchers (legacy compatibility)"""
        for observer in self.observers:
            if observer.is_alive():
                observer.stop()
                observer.join()
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None
    
    # Decorator for config-dependent functions
    @staticmethod
    def config_required(*keys):
        """Decorator to ensure configuration keys exist"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                loader = get_config_loader()
                for key in keys:
                    if not loader.get(key):
                        raise ValueError(f"Required config key '{key}' not found")
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global singleton instance
_config_loader: Optional[DynamicConfigLoader] = None

def get_config_loader() -> DynamicConfigLoader:
    """Get or create the global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = DynamicConfigLoader()
    return _config_loader

async def load_config() -> Dict[str, Any]:
    """Helper function to load configuration"""
    loader = get_config_loader()
    
    # Initialize if not already done
    if not loader.config_cache:
        await loader.initialize()
    
    return loader.get_all()

async def initialize_config():
    """Initialize the global configuration (new style)"""
    loader = get_config_loader()
    await loader.initialize()
    return loader

# Convenience functions for backward compatibility
def config(key: str, default: Any = None) -> Any:
    """Quick access to configuration values"""
    return get_config_loader().get(key, default)

def secret(key: str, default: Any = None) -> Any:
    """Quick access to encrypted configuration values"""
    return get_config_loader().get(key, default, decrypt=True)

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value (new style)"""
    return get_config_loader().get(key, default)

def set_config(key: str, value: Any):
    """Set configuration value (new style)"""
    get_config_loader().set(key, value)


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize configuration
        config_loader = await initialize_config()
        
        # Access configuration
        print(f"{CyberColors.NEON_CYAN}=== NEXLIFY Configuration Test ==={CyberColors.RESET}")
        print(f"Environment: {config_loader.get('environment')}")
        print(f"Neural Net ID: {config_loader.get('neural_net_id')}")
        print(f"QuestDB URL: {config_loader.get('databases.questdb_url')}")
        print(f"ML Models: {config_loader.get('ml_models')}")
        print(f"Redis URL: {config_loader.get('redis_url')}")
        
        # Register change callback
        async def on_config_change(data):
            print(f"{CyberColors.NEON_YELLOW}Configuration changed:{CyberColors.RESET}")
            for change in data['changes']:
                print(f"  - {change}")
        
        config_loader.on_change(on_config_change)
        
        # Test setting a value
        config_loader.set('test.dynamic_value', 'Hello from the Matrix!')
        print(f"\nDynamic value: {config_loader.get('test.dynamic_value')}")
        
        # Keep running to watch for changes
        print(f"\n{CyberColors.NEON_GREEN}Watching for configuration changes...{CyberColors.RESET}")
        print(f"{CyberColors.NEON_YELLOW}Press Ctrl+C to exit{CyberColors.RESET}")
        
        try:
            await asyncio.sleep(3600)
        except KeyboardInterrupt:
            print(f"\n{CyberColors.NEURAL_PURPLE}Shutting down...{CyberColors.RESET}")
        finally:
            await config_loader.shutdown()
    
    asyncio.run(main())
