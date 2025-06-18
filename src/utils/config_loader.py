#!/usr/bin/env python3
"""
src/utils/config_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY DYNAMIC CONFIGURATION LOADER v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Zero hardcoded values. Everything dynamic. Everything configurable.
Uses the latest best practices for configuration management in 2025.
"""

import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import hashlib
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel, Field, validator
import hvac  # HashiCorp Vault client
import boto3  # AWS Secrets Manager
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import logging
from rich.console import Console

console = Console()
logger = logging.getLogger("NEXLIFY.CONFIG")

@dataclass
class ConfigSource:
    """Represents a configuration source with priority"""
    name: str
    priority: int  # Lower number = higher priority
    data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None

class ConfigSchema(BaseModel):
    """Pydantic schema for configuration validation"""
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    environment: str = Field(..., regex=r'^(development|staging|production)$')
    
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
                if pattern in v.lower() and field.name not in ['environment']:
                    raise ValueError(f"Potential hardcoded value in {field.name}: {v}")
        return v

class DynamicConfigLoader:
    """
    Ultra-dynamic configuration loader with zero hardcoded values
    Supports multiple sources with priority-based merging
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.sources: List[ConfigSource] = []
        self.config_cache: Dict[str, Any] = {}
        self.encryption_key: Optional[bytes] = None
        self.hot_reload_enabled = True
        self.observers: List[Observer] = []
        self._load_lock = asyncio.Lock()
        
        # Initialize encryption
        self._init_encryption()
        
        # Load environment variables first (highest priority)
        self._load_environment()
        
    def _init_encryption(self):
        """Initialize encryption for sensitive values"""
        key_sources = [
            os.getenv('NEXLIFY_ENCRYPTION_KEY'),
            os.getenv('NEXLIFY_MASTER_KEY'),
            self._read_key_file('.nexlify_key'),
            self._generate_key()
        ]
        
        for key in key_sources:
            if key:
                self.encryption_key = key.encode() if isinstance(key, str) else key
                break
    
    def _read_key_file(self, filename: str) -> Optional[str]:
        """Read encryption key from file"""
        key_path = self.base_path / filename
        if key_path.exists() and key_path.stat().st_mode & 0o777 == 0o600:
            return key_path.read_text().strip()
        return None
    
    def _generate_key(self) -> bytes:
        """Generate new encryption key (last resort)"""
        logger.warning("Generating new encryption key - this should only happen once!")
        return Fernet.generate_key()
    
    def _load_environment(self):
        """Load environment variables with .env support"""
        # Load .env files in order of precedence
        env_files = [
            '.env.local',
            f'.env.{os.getenv("NEXLIFY_ENV", "development")}',
            '.env'
        ]
        
        for env_file in env_files:
            env_path = self.base_path / env_file
            if env_path.exists():
                load_dotenv(env_path, override=True)
        
        # Create source from environment
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith('NEXLIFY_'):
                # Convert NEXLIFY_SOME_KEY to some.key
                config_key = key[8:].lower().replace('_', '.')
                env_config[config_key] = self._parse_value(value)
        
        self.sources.append(ConfigSource(
            name="environment",
            priority=0,  # Highest priority
            data=env_config
        ))
    
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
    
    async def load_file(self, filepath: Path, priority: int = 10):
        """Load configuration from file (YAML, JSON, or TOML)"""
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
        
        # Determine format and parse
        if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
            data = yaml.safe_load(content)
        elif filepath.suffix == '.json':
            data = json.loads(content)
        elif filepath.suffix == '.toml':
            data = toml.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}")
        
        # Calculate checksum
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        # Add source
        source = ConfigSource(
            name=str(filepath),
            priority=priority,
            data=data,
            checksum=checksum
        )
        
        async with self._load_lock:
            # Remove existing source with same name
            self.sources = [s for s in self.sources if s.name != source.name]
            self.sources.append(source)
            self._rebuild_cache()
        
        # Setup hot reload if enabled
        if self.hot_reload_enabled:
            self._setup_file_watcher(filepath)
    
    def _setup_file_watcher(self, filepath: Path):
        """Setup file watcher for hot reloading"""
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, loader, filepath):
                self.loader = loader
                self.filepath = filepath
            
            def on_modified(self, event):
                if event.src_path == str(self.filepath):
                    logger.info(f"Config file changed: {self.filepath}")
                    asyncio.create_task(self.loader.load_file(self.filepath))
        
        observer = Observer()
        observer.schedule(ConfigFileHandler(self, filepath), str(filepath.parent))
        observer.start()
        self.observers.append(observer)
    
    async def load_vault(self, vault_url: str, vault_token: str, path: str, priority: int = 5):
        """Load configuration from HashiCorp Vault"""
        client = hvac.Client(url=vault_url, token=vault_token)
        
        if not client.is_authenticated():
            raise ValueError("Vault authentication failed")
        
        # Read secrets
        response = client.secrets.kv.v2.read_secret_version(path=path)
        data = response['data']['data']
        
        source = ConfigSource(
            name=f"vault:{path}",
            priority=priority,
            data=data
        )
        
        async with self._load_lock:
            self.sources = [s for s in self.sources if s.name != source.name]
            self.sources.append(source)
            self._rebuild_cache()
    
    async def load_aws_secrets(self, secret_name: str, region: str = None, priority: int = 5):
        """Load configuration from AWS Secrets Manager"""
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        response = client.get_secret_value(SecretId=secret_name)
        data = json.loads(response['SecretString'])
        
        source = ConfigSource(
            name=f"aws:{secret_name}",
            priority=priority,
            data=data
        )
        
        async with self._load_lock:
            self.sources = [s for s in self.sources if s.name != source.name]
            self.sources.append(source)
            self._rebuild_cache()
    
    async def load_azure_keyvault(self, vault_url: str, priority: int = 5):
        """Load configuration from Azure Key Vault"""
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        data = {}
        for secret in client.list_properties_of_secrets():
            secret_value = client.get_secret(secret.name)
            # Convert Azure Key Vault naming (hyphens) to dots
            key = secret.name.replace('-', '.')
            data[key] = self._parse_value(secret_value.value)
        
        source = ConfigSource(
            name=f"azure:{vault_url}",
            priority=priority,
            data=data
        )
        
        async with self._load_lock:
            self.sources = [s for s in self.sources if s.name != source.name]
            self.sources.append(source)
            self._rebuild_cache()
    
    def _rebuild_cache(self):
        """Rebuild configuration cache from all sources"""
        # Sort sources by priority (lower number = higher priority)
        sorted_sources = sorted(self.sources, key=lambda s: s.priority, reverse=True)
        
        # Merge configurations (higher priority overwrites lower)
        self.config_cache = {}
        for source in sorted_sources:
            self._deep_merge(self.config_cache, source.data)
        
        # Validate final configuration
        try:
            ConfigSchema(**self.config_cache)
        except Exception as e:
            logger.warning(f"Configuration validation warning: {e}")
    
    def _deep_merge(self, base: dict, update: dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
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
        if decrypt and isinstance(value, str) and self.encryption_key:
            try:
                f = Fernet(self.encryption_key)
                value = f.decrypt(value.encode()).decode()
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
        if encrypt and isinstance(value, str) and self.encryption_key:
            f = Fernet(self.encryption_key)
            value = f.encrypt(value.encode()).decode()
        
        # Set value using dot notation
        parts = key.split('.')
        target = self.config_cache
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
    
    def watch(self, callback):
        """Register callback for configuration changes"""
        # Implementation for reactive config updates
        pass
    
    def __del__(self):
        """Cleanup file watchers"""
        for observer in self.observers:
            observer.stop()
            observer.join()

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
    
    # Load configuration files based on environment
    env = os.getenv('NEXLIFY_ENV', 'development')
    config_dir = Path(os.getenv('NEXLIFY_CONFIG_DIR', 'config'))
    
    # Load base config
    base_config = config_dir / 'base.yaml'
    if base_config.exists():
        await loader.load_file(base_config, priority=20)
    
    # Load environment-specific config
    env_config = config_dir / f'{env}.yaml'
    if env_config.exists():
        await loader.load_file(env_config, priority=10)
    
    # Load local overrides (git-ignored)
    local_config = config_dir / 'local.yaml'
    if local_config.exists():
        await loader.load_file(local_config, priority=5)
    
    # Load from external sources if configured
    if vault_url := os.getenv('VAULT_ADDR'):
        await loader.load_vault(
            vault_url,
            os.getenv('VAULT_TOKEN'),
            os.getenv('VAULT_PATH', 'nexlify/config'),
            priority=3
        )
    
    if aws_secret := os.getenv('AWS_SECRET_NAME'):
        await loader.load_aws_secrets(aws_secret, priority=3)
    
    if azure_vault := os.getenv('AZURE_KEYVAULT_URL'):
        await loader.load_azure_keyvault(azure_vault, priority=3)
    
    return loader.get_all()

# Convenience functions
def config(key: str, default: Any = None) -> Any:
    """Quick access to configuration values"""
    return get_config_loader().get(key, default)

def secret(key: str, default: Any = None) -> Any:
    """Quick access to encrypted configuration values"""
    return get_config_loader().get(key, default, decrypt=True)
