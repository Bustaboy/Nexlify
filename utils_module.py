"""
Nexlify Utilities Module - Comprehensive utilities for trading operations
Enhanced with validation, security, and performance optimizations
"""

import os
import sys
import json
import time
import hashlib
import hmac
import re
import asyncio
import aiohttp
import socket
import shutil
import threading
import functools
import logging
import tempfile
import zipfile
import platform
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Tuple
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict, deque
from contextlib import contextmanager
import concurrent.futures
from enum import Enum

import numpy as np
import pandas as pd
import pytz
from dateutil import parser as date_parser

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import py7zr
    HAS_7ZIP = True
except ImportError:
    HAS_7ZIP = False


T = TypeVar('T')


class TimeFrame(Enum):
    """Standard timeframes for trading"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class FileUtils:
    """Enhanced file operation utilities"""
    
    @staticmethod
    def safe_json_save(data: Dict[str, Any], filepath: Union[str, Path], 
                      create_backup: bool = True, check_space: bool = True,
                      min_space_mb: float = 10) -> bool:
        """
        Safely save JSON with atomic write and disk space check
        
        Args:
            data: Data to save
            filepath: Target file path
            create_backup: Create backup before overwriting
            check_space: Check disk space before writing
            min_space_mb: Minimum required disk space in MB
        """
        filepath = Path(filepath)
        
        try:
            # Check disk space
            if check_space and HAS_PSUTIL:
                disk_usage = psutil.disk_usage(str(filepath.parent))
                free_mb = disk_usage.free / (1024 * 1024)
                if free_mb < min_space_mb:
                    logging.error(f"Insufficient disk space: {free_mb:.1f}MB < {min_space_mb}MB")
                    return False
            
            # Create backup if file exists
            if create_backup and filepath.exists():
                FileUtils.create_backup(filepath)
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Atomic rename
            temp_file.replace(filepath)
            
            # Set appropriate permissions on Unix
            if platform.system() != "Windows":
                os.chmod(filepath, 0o600)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save JSON to {filepath}: {e}")
            # Clean up temp file if exists
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
            return False
    
    @staticmethod
    def safe_json_load(filepath: Union[str, Path], default: Any = None,
                      validate_schema: Optional[Dict] = None) -> Any:
        """
        Safely load JSON with validation
        
        Args:
            filepath: File to load
            default: Default value if load fails
            validate_schema: Optional schema to validate against
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logging.warning(f"File not found: {filepath}")
            return default if default is not None else {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # TODO: Add schema validation if provided
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {filepath}: {e}")
            return default if default is not None else {}
        except Exception as e:
            logging.error(f"Failed to load JSON from {filepath}: {e}")
            return default if default is not None else {}
    
    @staticmethod
    def create_backup(filepath: Union[str, Path], max_backups: int = 5,
                     compress: bool = False) -> Optional[Path]:
        """
        Create timestamped backup with rotation
        
        Args:
            filepath: File to backup
            max_backups: Maximum number of backups to keep
            compress: Compress backup using 7zip if available
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        try:
            # Create backup directory
            backup_dir = filepath.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copy file
            shutil.copy2(filepath, backup_path)
            
            # Compress if requested
            if compress and HAS_7ZIP:
                archive_path = backup_path.with_suffix('.7z')
                with py7zr.SevenZipFile(archive_path, 'w') as archive:
                    archive.write(backup_path, backup_path.name)
                backup_path.unlink()  # Remove uncompressed
                backup_path = archive_path
            
            # Rotate old backups
            FileUtils._rotate_backups(backup_dir, filepath.stem, max_backups)
            
            logging.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return None
    
    @staticmethod
    def _rotate_backups(backup_dir: Path, base_name: str, max_backups: int):
        """Rotate old backups, keeping only the newest"""
        pattern = f"{base_name}_*"
        backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logging.info(f"Removed old backup: {oldest}")
    
    @staticmethod
    def cleanup_old_files(directory: Union[str, Path], days: int = 7,
                         pattern: str = "*", check_in_use: bool = True) -> int:
        """
        Clean up old files with safety checks
        
        Args:
            directory: Directory to clean
            days: Files older than this are deleted
            pattern: File pattern to match
            check_in_use: Check if files are in use (Windows)
        """
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            
            try:
                # Check file age
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime >= cutoff_time:
                    continue
                
                # Check if file is in use (Windows)
                if check_in_use and platform.system() == "Windows":
                    if FileUtils._is_file_in_use(file_path):
                        logging.warning(f"File in use, skipping: {file_path}")
                        continue
                
                file_path.unlink()
                deleted_count += 1
                logging.debug(f"Deleted old file: {file_path}")
                
            except Exception as e:
                logging.error(f"Failed to delete {file_path}: {e}")
        
        return deleted_count
    
    @staticmethod
    def _is_file_in_use(filepath: Path) -> bool:
        """Check if file is in use (Windows)"""
        if platform.system() != "Windows":
            return False
        
        try:
            # Try to open file exclusively
            with open(filepath, 'rb+'):
                return False
        except (IOError, OSError):
            return True
    
    @staticmethod
    def encrypt_file(filepath: Union[str, Path], encryption_manager=None) -> bool:
        """
        Encrypt file using EncryptionManager from nexlify_advanced_security
        
        Args:
            filepath: File to encrypt
            encryption_manager: EncryptionManager instance (will create if None)
        """
        filepath = Path(filepath)
        
        try:
            # Get or create EncryptionManager
            if encryption_manager is None:
                try:
                    from nexlify_advanced_security import EncryptionManager
                    # Load config to get master password
                    config = FileUtils.safe_json_load("config/enhanced_config.json", {})
                    master_password = config.get("security", {}).get("master_password", "")
                    
                    if not master_password:
                        logging.warning("No master password set, using default encryption")
                        master_password = "nexlify_default_2077"  # Fallback for non-sensitive files
                    
                    encryption_manager = EncryptionManager(master_password)
                except ImportError:
                    logging.error("nexlify_advanced_security not available, using basic encryption")
                    # Fallback to basic Fernet encryption
                    if HAS_CRYPTO:
                        key = Fernet.generate_key()
                        fernet = Fernet(key)
                        
                        with open(filepath, 'rb') as f:
                            data = f.read()
                        
                        encrypted_data = fernet.encrypt(data)
                        enc_path = filepath.with_suffix(filepath.suffix + '.enc')
                        
                        with open(enc_path, 'wb') as f:
                            f.write(encrypted_data)
                        
                        # Save key separately (insecure but better than nothing)
                        key_path = enc_path.with_suffix('.key')
                        with open(key_path, 'wb') as f:
                            f.write(key)
                        
                        filepath.unlink()
                        return True
                    return False
            
            # Use EncryptionManager to encrypt
            return encryption_manager.encrypt_file(str(filepath))
            
        except Exception as e:
            logging.error(f"Failed to encrypt file: {e}")
            return False
    
    @staticmethod
    def decrypt_file(filepath: Union[str, Path], encryption_manager=None) -> bool:
        """
        Decrypt file using EncryptionManager from nexlify_advanced_security
        
        Args:
            filepath: File to decrypt
            encryption_manager: EncryptionManager instance (will create if None)
        """
        filepath = Path(filepath)
        
        try:
            # Get or create EncryptionManager
            if encryption_manager is None:
                try:
                    from nexlify_advanced_security import EncryptionManager
                    # Load config to get master password
                    config = FileUtils.safe_json_load("config/enhanced_config.json", {})
                    master_password = config.get("security", {}).get("master_password", "")
                    
                    if not master_password:
                        logging.warning("No master password set, using default decryption")
                        master_password = "nexlify_default_2077"  # Fallback for non-sensitive files
                    
                    encryption_manager = EncryptionManager(master_password)
                except ImportError:
                    logging.error("nexlify_advanced_security not available, using basic decryption")
                    # Fallback to basic Fernet decryption
                    if HAS_CRYPTO:
                        key_path = filepath.with_suffix('.key')
                        if not key_path.exists():
                            logging.error("Encryption key not found")
                            return False
                        
                        with open(key_path, 'rb') as f:
                            key = f.read()
                        
                        fernet = Fernet(key)
                        
                        with open(filepath, 'rb') as f:
                            encrypted_data = f.read()
                        
                        data = fernet.decrypt(encrypted_data)
                        dec_path = filepath.with_suffix('')  # Remove .enc
                        
                        with open(dec_path, 'wb') as f:
                            f.write(data)
                        
                        filepath.unlink()
                        key_path.unlink()
                        return True
                    return False
            
            # Use EncryptionManager to decrypt
            return encryption_manager.decrypt_file(str(filepath))
            
        except Exception as e:
            logging.error(f"Failed to decrypt file: {e}")
            return False
    
    @staticmethod
    def save_encrypted_json(data: Dict[str, Any], filepath: Union[str, Path],
                           encryption_manager=None) -> bool:
        """
        Save JSON data encrypted
        
        Args:
            data: Data to save
            filepath: Target file path
            encryption_manager: EncryptionManager instance
        """
        filepath = Path(filepath)
        
        try:
            # First save as regular JSON
            if not FileUtils.safe_json_save(data, filepath):
                return False
            
            # Then encrypt the file
            if not FileUtils.encrypt_file(filepath, encryption_manager):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save encrypted JSON: {e}")
            return False
    
    @staticmethod
    def load_encrypted_json(filepath: Union[str, Path], default: Any = None,
                           encryption_manager=None) -> Any:
        """
        Load encrypted JSON data
        
        Args:
            filepath: File to load
            default: Default value if load fails
            encryption_manager: EncryptionManager instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return default if default is not None else {}
        
        try:
            # Create temp copy
            temp_path = filepath.with_suffix('.tmp')
            shutil.copy2(filepath, temp_path)
            
            # Decrypt temp file
            if FileUtils.decrypt_file(temp_path, encryption_manager):
                decrypted_path = temp_path.with_suffix('')
                data = FileUtils.safe_json_load(decrypted_path, default)
                
                # Clean up
                if decrypted_path.exists():
                    decrypted_path.unlink()
                
                return data
            else:
                if temp_path.exists():
                    temp_path.unlink()
                return default if default is not None else {}
                
        except Exception as e:
            logging.error(f"Failed to load encrypted JSON: {e}")
            return default if default is not None else {}


class NetworkUtils:
    """Enhanced network operation utilities"""
    
    # Exchange-specific rate limits (requests per minute)
    EXCHANGE_RATE_LIMITS = {
        'binance': 1200,
        'kraken': 60,
        'coinbase': 600,
        'huobi': 100,
        'okx': 300,
        'default': 60
    }
    
    # Rate limiter storage
    _rate_limiters = defaultdict(lambda: {'last_request': 0, 'request_count': 0})
    _rate_limit_lock = threading.Lock()
    _detected_rate_limits = {}  # Cache for detected rate limits
    
    @staticmethod
    def detect_exchange_rate_limit(exchange: str, force_refresh: bool = False) -> int:
        """
        Detect and cache exchange rate limit from ccxt
        
        Args:
            exchange: Exchange name
            force_refresh: Force re-detection even if cached
        
        Returns:
            Rate limit in requests per minute
        """
        # Check cache first
        if not force_refresh and exchange in NetworkUtils._detected_rate_limits:
            return NetworkUtils._detected_rate_limits[exchange]
        
        rate_limit = NetworkUtils.EXCHANGE_RATE_LIMITS.get(exchange, 60)
        
        if HAS_CCXT and hasattr(ccxt, exchange):
            try:
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange)
                ex = exchange_class({'enableRateLimit': True})
                
                # Try multiple methods to get rate limit
                detected_limit = None
                
                # Method 1: Direct rateLimit property
                if hasattr(ex, 'rateLimit') and ex.rateLimit:
                    # Convert milliseconds to requests per minute
                    detected_limit = int(60000 / ex.rateLimit)
                
                # Method 2: Check describe() method
                elif hasattr(ex, 'describe'):
                    info = ex.describe()
                    
                    # Look for rate limit in various places
                    if 'rateLimit' in info:
                        if isinstance(info['rateLimit'], (int, float)):
                            detected_limit = int(60000 / info['rateLimit'])
                        elif isinstance(info['rateLimit'], dict):
                            # Some exchanges provide detailed limits
                            for key in ['api', 'public', 'private', 'get']:
                                if key in info['rateLimit']:
                                    val = info['rateLimit'][key]
                                    if isinstance(val, (int, float)):
                                        detected_limit = int(60000 / val)
                                        break
                    
                    # Method 3: Check API section
                    elif 'api' in info and isinstance(info['api'], dict):
                        if 'rateLimit' in info['api']:
                            detected_limit = int(60000 / info['api']['rateLimit'])
                
                # Method 4: Load markets and check
                if detected_limit is None and hasattr(ex, 'load_markets'):
                    try:
                        ex.load_markets()
                        if hasattr(ex, 'rateLimit') and ex.rateLimit:
                            detected_limit = int(60000 / ex.rateLimit)
                    except:
                        pass
                
                if detected_limit and detected_limit > 0:
                    rate_limit = detected_limit
                    logging.info(f"Detected rate limit for {exchange}: {rate_limit} requests/minute")
                
            except Exception as e:
                logging.debug(f"Could not detect rate limit for {exchange}: {e}")
        
        # Cache the result
        NetworkUtils._detected_rate_limits[exchange] = rate_limit
        return rate_limit
    
    @staticmethod
    def rate_limited_request(func: Callable, *args, exchange: str = 'default',
                           custom_limit: Optional[int] = None, **kwargs) -> Any:
        """
        Execute function with exchange-specific rate limiting
        
        Args:
            func: Function to execute
            exchange: Exchange name for rate limit lookup
            custom_limit: Override rate limit (requests per minute)
        """
        # Get rate limit
        if custom_limit:
            rate_limit = custom_limit
        else:
            # Use automatic detection
            rate_limit = NetworkUtils.detect_exchange_rate_limit(exchange)
        
        # Ensure rate limit is reasonable
        rate_limit = max(1, min(rate_limit, 10000))  # Between 1 and 10000 requests per minute
        
        # Calculate minimum time between requests
        min_interval = 60.0 / rate_limit
        
        with NetworkUtils._rate_limit_lock:
            limiter = NetworkUtils._rate_limiters[exchange]
            now = time.time()
            
            # Reset counter if minute has passed
            if now - limiter['last_request'] > 60:
                limiter['request_count'] = 0
            
            # Check if we need to wait
            time_since_last = now - limiter['last_request']
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
                now = time.time()
            
            # Update limiter
            limiter['last_request'] = now
            limiter['request_count'] += 1
        
        # Execute function
        return func(*args, **kwargs)
    
    @staticmethod
    async def async_rate_limited_request(func: Callable, *args, exchange: str = 'default',
                                       custom_limit: Optional[int] = None, **kwargs) -> Any:
        """Async version of rate limited request with automatic exchange detection"""
        # Get rate limit
        if custom_limit:
            rate_limit = custom_limit
        else:
            # Use automatic detection
            rate_limit = NetworkUtils.detect_exchange_rate_limit(exchange)
        
        rate_limit = max(1, min(rate_limit, 10000))
        min_interval = 60.0 / rate_limit
        
        with NetworkUtils._rate_limit_lock:
            limiter = NetworkUtils._rate_limiters[exchange]
            now = time.time()
            
            if now - limiter['last_request'] > 60:
                limiter['request_count'] = 0
            
            time_since_last = now - limiter['last_request']
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
                now = time.time()
            
            limiter['last_request'] = now
            limiter['request_count'] += 1
        
        return await func(*args, **kwargs)
    
    @staticmethod
    def generate_signature(params: Dict[str, Any], secret: str, 
                          exchange: str = 'binance', algorithm: str = None) -> str:
        """
        Generate exchange-specific API signature
        
        Args:
            params: Request parameters
            secret: API secret
            exchange: Exchange name
            algorithm: Override algorithm (sha256, sha512, etc.)
        """
        # Exchange-specific algorithms
        exchange_algorithms = {
            'binance': 'sha256',
            'kraken': 'sha512',
            'coinbase': 'sha256',
            'okx': 'sha256',
            'huobi': 'sha256'
        }
        
        algo = algorithm or exchange_algorithms.get(exchange, 'sha256')
        
        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generate signature
        if algo == 'sha256':
            signature = hmac.new(
                secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        elif algo == 'sha512':
            signature = hmac.new(
                secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        
        return signature
    
    @staticmethod
    def check_internet(timeout: float = 5.0) -> bool:
        """Check internet connectivity"""
        test_hosts = [
            ("8.8.8.8", 53),  # Google DNS
            ("1.1.1.1", 53),  # Cloudflare DNS
            ("api.binance.com", 443),  # Binance API
        ]
        
        for host, port in test_hosts:
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                sock.close()
                return True
            except:
                continue
        
        return False
    
    @staticmethod
    async def check_exchange_status(exchange: str) -> Dict[str, Any]:
        """Check exchange API status"""
        status = {
            'online': False,
            'latency': None,
            'message': None
        }
        
        if not HAS_CCXT:
            status['message'] = "CCXT not available"
            return status
        
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange, None)
            if not exchange_class:
                status['message'] = f"Unknown exchange: {exchange}"
                return status
            
            ex = exchange_class()
            
            # Measure latency
            start_time = time.time()
            await ex.load_markets()
            latency = (time.time() - start_time) * 1000  # ms
            
            status['online'] = True
            status['latency'] = round(latency, 2)
            status['message'] = "Online"
            
        except Exception as e:
            status['message'] = str(e)
        
        return status
    
    @staticmethod
    def parse_websocket_url(exchange: str, stream_type: str = 'ticker') -> str:
        """Get websocket URL for exchange"""
        ws_urls = {
            'binance': {
                'ticker': 'wss://stream.binance.com:9443/ws',
                'depth': 'wss://stream.binance.com:9443/ws',
                'trade': 'wss://stream.binance.com:9443/ws'
            },
            'kraken': {
                'ticker': 'wss://ws.kraken.com',
                'depth': 'wss://ws.kraken.com',
                'trade': 'wss://ws.kraken.com'
            },
            'coinbase': {
                'ticker': 'wss://ws-feed.exchange.coinbase.com',
                'depth': 'wss://ws-feed.exchange.coinbase.com',
                'trade': 'wss://ws-feed.exchange.coinbase.com'
            }
        }
        
        return ws_urls.get(exchange, {}).get(stream_type, '')


class ValidationUtils:
    """Enhanced validation utilities"""
    
    @staticmethod
    def validate_email(email: str, allow_empty: bool = False) -> bool:
        """Validate email address"""
        if not email:
            return allow_empty
        
        # RFC 5322 compliant regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_api_credentials(api_key: str, api_secret: str, 
                               exchange: str = None) -> Dict[str, Any]:
        """
        Validate API credentials format
        
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        result = {'valid': True, 'errors': []}
        
        # Check if empty
        if not api_key or not api_secret:
            result['valid'] = False
            result['errors'].append("API key and secret cannot be empty")
            return result
        
        # Exchange-specific validation
        if exchange:
            if exchange == 'binance':
                if len(api_key) != 64:
                    result['valid'] = False
                    result['errors'].append("Binance API key should be 64 characters")
                if len(api_secret) != 64:
                    result['valid'] = False
                    result['errors'].append("Binance API secret should be 64 characters")
            
            elif exchange == 'kraken':
                # Kraken uses different format with hyphens
                if not re.match(r'^[A-Za-z0-9+/=-]+
    
    @staticmethod
    def validate_port(port: Any, check_available: bool = False) -> bool:
        """Validate port number and optionally check if available"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                return False
            
            if check_available:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port_num))
                        return True
                    except OSError:
                        return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ip_address(ip: str, allow_ranges: bool = True) -> bool:
        """Validate IP address or CIDR range"""
        if not ip:
            return False
        
        # Check single IP
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, ip):
            # Validate octets
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # Check CIDR range
        if allow_ranges and '/' in ip:
            try:
                addr, mask = ip.split('/')
                mask_int = int(mask)
                if not 0 <= mask_int <= 32:
                    return False
                return ValidationUtils.validate_ip_address(addr, allow_ranges=False)
            except:
                return False
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = None) -> bool:
        """Validate trading symbol format"""
        if not symbol:
            return False
        
        # General format: BASE/QUOTE
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")
, api_key):
                    result['valid'] = False
                    result['errors'].append("Invalid Kraken API key format")
            
            elif exchange == 'coinbase':
                # Coinbase uses UUID format
                uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}
    
    @staticmethod
    def validate_port(port: Any, check_available: bool = False) -> bool:
        """Validate port number and optionally check if available"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                return False
            
            if check_available:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port_num))
                        return True
                    except OSError:
                        return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ip_address(ip: str, allow_ranges: bool = True) -> bool:
        """Validate IP address or CIDR range"""
        if not ip:
            return False
        
        # Check single IP
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, ip):
            # Validate octets
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # Check CIDR range
        if allow_ranges and '/' in ip:
            try:
                addr, mask = ip.split('/')
                mask_int = int(mask)
                if not 0 <= mask_int <= 32:
                    return False
                return ValidationUtils.validate_ip_address(addr, allow_ranges=False)
            except:
                return False
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = None) -> bool:
        """Validate trading symbol format"""
        if not symbol:
            return False
        
        # General format: BASE/QUOTE
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
        if not re.match(pattern, symbol.upper()):
            return False
        
        # Exchange-specific validation
        if exchange == 'binance':
            # Binance uses no separator
            base, quote = symbol.upper().split('/')
            valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
            return quote in valid_quotes
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate trading timeframe"""
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe in valid_timeframes


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")

                if not re.match(uuid_pattern, api_key, re.IGNORECASE):
                    result['valid'] = False
                    result['errors'].append("Invalid Coinbase API key format")
        
        # Additional validation with APIKeyRotation if available
        try:
            from nexlify_advanced_security import APIKeyRotation
            
            # Create temporary rotation manager to validate
            rotation_manager = APIKeyRotation()
            
            # Validate using the security module
            is_valid = rotation_manager.validate_api_key(exchange, api_key, api_secret)
            if not is_valid:
                result['valid'] = False
                result['errors'].append(f"API credentials failed {exchange} validation")
                
        except ImportError:
            # Security module not available, use basic validation only
            pass
        except Exception as e:
            logging.warning(f"Could not validate with APIKeyRotation: {e}")
        
        return result
    
    @staticmethod
    def validate_port(port: Any, check_available: bool = False) -> bool:
        """Validate port number and optionally check if available"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                return False
            
            if check_available:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port_num))
                        return True
                    except OSError:
                        return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ip_address(ip: str, allow_ranges: bool = True) -> bool:
        """Validate IP address or CIDR range"""
        if not ip:
            return False
        
        # Check single IP
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, ip):
            # Validate octets
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # Check CIDR range
        if allow_ranges and '/' in ip:
            try:
                addr, mask = ip.split('/')
                mask_int = int(mask)
                if not 0 <= mask_int <= 32:
                    return False
                return ValidationUtils.validate_ip_address(addr, allow_ranges=False)
            except:
                return False
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = None) -> bool:
        """Validate trading symbol format"""
        if not symbol:
            return False
        
        # General format: BASE/QUOTE
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
        if not re.match(pattern, symbol.upper()):
            return False
        
        # Exchange-specific validation
        if exchange == 'binance':
            # Binance uses no separator
            base, quote = symbol.upper().split('/')
            valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
            return quote in valid_quotes
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate trading timeframe"""
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe in valid_timeframes


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")

        if not re.match(pattern, symbol.upper()):
            return False
        
        # Exchange-specific validation
        if exchange == 'binance':
            # Binance uses no separator
            base, quote = symbol.upper().split('/')
            valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
            return quote in valid_quotes
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate trading timeframe"""
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe in valid_timeframes
    
    @staticmethod
    def validate_exchange_config(exchange: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete exchange configuration
        
        Args:
            exchange: Exchange name
            config: Configuration dict with api_key, api_secret, etc.
        
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        result = {'valid': True, 'errors': []}
        
        # Validate API credentials
        api_key = config.get('api_key', '')
        api_secret = config.get('api_secret', '')
        
        cred_result = ValidationUtils.validate_api_credentials(api_key, api_secret, exchange)
        if not cred_result['valid']:
            result['valid'] = False
            result['errors'].extend(cred_result['errors'])
        
        # Test connection if credentials are valid
        if result['valid'] and HAS_CCXT:
            try:
                exchange_class = getattr(ccxt, exchange, None)
                if exchange_class:
                    # Create test instance
                    test_exchange = exchange_class({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True
                    })
                    
                    # Test API key permissions
                    try:
                        # This will fail if API keys are invalid
                        test_exchange.fetch_balance()
                    except ccxt.AuthenticationError:
                        result['valid'] = False
                        result['errors'].append("Invalid API credentials - authentication failed")
                    except ccxt.InsufficientFunds:
                        # This is actually good - means API works but no funds
                        pass
                    except Exception as e:
                        # Network or other errors
                        result['errors'].append(f"Could not verify credentials: {str(e)}")
                else:
                    result['errors'].append(f"Unknown exchange: {exchange}")
            except Exception as e:
                result['errors'].append(f"Exchange validation error: {str(e)}")
        
        return result


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")
, api_key):
                    result['valid'] = False
                    result['errors'].append("Invalid Kraken API key format")
            
            elif exchange == 'coinbase':
                # Coinbase uses UUID format
                uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}
    
    @staticmethod
    def validate_port(port: Any, check_available: bool = False) -> bool:
        """Validate port number and optionally check if available"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                return False
            
            if check_available:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port_num))
                        return True
                    except OSError:
                        return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ip_address(ip: str, allow_ranges: bool = True) -> bool:
        """Validate IP address or CIDR range"""
        if not ip:
            return False
        
        # Check single IP
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, ip):
            # Validate octets
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # Check CIDR range
        if allow_ranges and '/' in ip:
            try:
                addr, mask = ip.split('/')
                mask_int = int(mask)
                if not 0 <= mask_int <= 32:
                    return False
                return ValidationUtils.validate_ip_address(addr, allow_ranges=False)
            except:
                return False
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = None) -> bool:
        """Validate trading symbol format"""
        if not symbol:
            return False
        
        # General format: BASE/QUOTE
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
        if not re.match(pattern, symbol.upper()):
            return False
        
        # Exchange-specific validation
        if exchange == 'binance':
            # Binance uses no separator
            base, quote = symbol.upper().split('/')
            valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
            return quote in valid_quotes
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate trading timeframe"""
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe in valid_timeframes


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")

                if not re.match(uuid_pattern, api_key, re.IGNORECASE):
                    result['valid'] = False
                    result['errors'].append("Invalid Coinbase API key format")
        
        # Additional validation with APIKeyRotation if available
        try:
            from nexlify_advanced_security import APIKeyRotation
            
            # Create temporary rotation manager to validate
            rotation_manager = APIKeyRotation()
            
            # Validate using the security module
            is_valid = rotation_manager.validate_api_key(exchange, api_key, api_secret)
            if not is_valid:
                result['valid'] = False
                result['errors'].append(f"API credentials failed {exchange} validation")
                
        except ImportError:
            # Security module not available, use basic validation only
            pass
        except Exception as e:
            logging.warning(f"Could not validate with APIKeyRotation: {e}")
        
        return result
    
    @staticmethod
    def validate_port(port: Any, check_available: bool = False) -> bool:
        """Validate port number and optionally check if available"""
        try:
            port_num = int(port)
            if not 1 <= port_num <= 65535:
                return False
            
            if check_available:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port_num))
                        return True
                    except OSError:
                        return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ip_address(ip: str, allow_ranges: bool = True) -> bool:
        """Validate IP address or CIDR range"""
        if not ip:
            return False
        
        # Check single IP
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, ip):
            # Validate octets
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # Check CIDR range
        if allow_ranges and '/' in ip:
            try:
                addr, mask = ip.split('/')
                mask_int = int(mask)
                if not 0 <= mask_int <= 32:
                    return False
                return ValidationUtils.validate_ip_address(addr, allow_ranges=False)
            except:
                return False
        
        return False
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = None) -> bool:
        """Validate trading symbol format"""
        if not symbol:
            return False
        
        # General format: BASE/QUOTE
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
        if not re.match(pattern, symbol.upper()):
            return False
        
        # Exchange-specific validation
        if exchange == 'binance':
            # Binance uses no separator
            base, quote = symbol.upper().split('/')
            valid_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
            return quote in valid_quotes
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate trading timeframe"""
        valid_timeframes = [tf.value for tf in TimeFrame]
        return timeframe in valid_timeframes


class CryptoUtils:
    """Enhanced cryptocurrency utilities"""
    
    # Supported cryptocurrencies
    SUPPORTED_CHAINS = {
        'BTC': {
            'name': 'Bitcoin',
            'decimals': 8,
            'address_pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$'
        },
        'ETH': {
            'name': 'Ethereum',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'BNB': {
            'name': 'Binance Smart Chain',
            'decimals': 18,
            'address_pattern': r'^0x[a-fA-F0-9]{40}$'
        },
        'SOL': {
            'name': 'Solana',
            'decimals': 9,
            'address_pattern': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
        }
    }
    
    @staticmethod
    def validate_address(address: str, chain: str = 'BTC') -> bool:
        """
        Validate cryptocurrency address with checksum verification
        
        Args:
            address: Crypto address
            chain: Blockchain (BTC, ETH, BNB, SOL)
        """
        if not address or chain not in CryptoUtils.SUPPORTED_CHAINS:
            return False
        
        chain_info = CryptoUtils.SUPPORTED_CHAINS[chain]
        pattern = chain_info['address_pattern']
        
        # Basic pattern check
        if not re.match(pattern, address):
            return False
        
        # Additional validation for specific chains
        if chain == 'BTC':
            return CryptoUtils._validate_btc_address(address)
        elif chain in ['ETH', 'BNB']:
            return CryptoUtils._validate_eth_address(address)
        elif chain == 'SOL':
            return CryptoUtils._validate_sol_address(address)
        
        return True
    
    @staticmethod
    def _validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address with checksum"""
        # This is a simplified check - full validation requires base58 decoding
        # For production, use a library like bitcoinlib
        
        # Check length and characters
        if address.startswith('bc1'):
            # Bech32 address
            return len(address) in range(42, 63)
        else:
            # Legacy or SegWit
            return len(address) in range(26, 36)
    
    @staticmethod
    def _validate_eth_address(address: str) -> bool:
        """Validate Ethereum address with checksum"""
        if not address.startswith('0x'):
            return False
        
        # Remove 0x prefix
        address = address[2:]
        
        # Check if all lowercase or all uppercase (non-checksummed)
        if address == address.lower() or address == address.upper():
            return True
        
        # Validate EIP-55 checksum
        # For production, use web3.py for proper validation
        try:
            address_hash = hashlib.sha3_256(address.lower().encode()).hexdigest()
            for i in range(len(address)):
                char = address[i]
                if char in '0123456789':
                    continue
                hash_char = int(address_hash[i], 16)
                if (hash_char >= 8 and char.upper() != char) or \
                   (hash_char < 8 and char.lower() != char):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def _validate_sol_address(address: str) -> bool:
        """Validate Solana address"""
        # Basic check - Solana uses base58
        try:
            # Check valid base58 characters
            base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return all(c in base58_chars for c in address) and len(address) in range(32, 45)
        except:
            return False
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize trading pair symbol across exchanges
        
        Handles:
        - Case normalization
        - Separator differences (BTC-USDT vs BTC/USDT)
        - Futures/perpetual contracts
        - Stablecoin variants
        """
        if not symbol:
            return ""
        
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Replace common separators with /
        symbol = symbol.replace('-', '/')
        symbol = symbol.replace('_', '/')
        
        # Handle perpetual/futures markers
        futures_markers = ['-PERP', '_PERP', '.PERP', '-SWAP', '_SWAP']
        for marker in futures_markers:
            if marker in symbol:
                symbol = symbol.replace(marker, '')
                symbol += '-PERP'
        
        # Normalize stablecoin variants
        stablecoin_map = {
            'USDT': ['USDT', 'TETHER'],
            'USDC': ['USDC', 'USDCOIN'],
            'BUSD': ['BUSD', 'BINANCEUSD'],
            'DAI': ['DAI', 'MAKERDAI']
        }
        
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            
            # Check quote currency
            for standard, variants in stablecoin_map.items():
                if quote in variants:
                    quote = standard
                    break
            
            symbol = f"{base}/{quote}"
        
        return symbol
    
    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float,
                              stop_loss_percent: float, price: float,
                              min_size: Optional[float] = None,
                              max_size: Optional[float] = None,
                              exchange_minimums: Optional[Dict] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Account balance
            risk_percent: Risk per trade (e.g., 1.0 for 1%)
            stop_loss_percent: Stop loss percentage
            price: Entry price
            min_size: Minimum position size
            max_size: Maximum position size
            exchange_minimums: Exchange-specific minimums
        """
        if balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_amount = balance * (risk_percent / 100)
        position_value = risk_amount / (stop_loss_percent / 100)
        position_size = position_value / price
        
        # Apply exchange minimums
        if exchange_minimums:
            # Common exchange minimums
            default_minimums = {
                'binance': 0.001,  # 0.001 BTC
                'kraken': 0.002,
                'coinbase': 0.001,
                'default': 0.001
            }
            
            for exchange, minimum in exchange_minimums.items():
                if minimum > position_size:
                    position_size = max(position_size, minimum)
        
        # Apply bounds
        if min_size:
            position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
        
        # Round to 8 decimal places
        return round(position_size, 8)
    
    @staticmethod
    def convert_to_base_unit(amount: float, decimals: int) -> int:
        """Convert amount to base unit (e.g., ETH to wei)"""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def convert_from_base_unit(amount: int, decimals: int) -> float:
        """Convert from base unit to decimal (e.g., wei to ETH)"""
        return amount / (10 ** decimals)


class TimeUtils:
    """Enhanced time and scheduling utilities"""
    
    @staticmethod
    def get_next_schedule_time(schedule: str, timezone_str: str = 'UTC',
                             reference_time: Optional[datetime] = None) -> datetime:
        """
        Get next scheduled time with timezone support
        
        Args:
            schedule: Schedule string (e.g., "daily@09:00", "hourly", "*/5m")
            timezone_str: Timezone string (e.g., "US/Eastern")
            reference_time: Reference time (default: now)
        """
        tz = pytz.timezone(timezone_str)
        now = reference_time or datetime.now(tz)
        
        # Ensure reference time is timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        
        # Parse schedule
        if schedule == "hourly":
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif schedule.startswith("daily@"):
            # Daily at specific time
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_time <= now:
                next_time += timedelta(days=1)
        
        elif schedule.startswith("*/"):
            # Interval (e.g., */5m, */30m, */2h)
            interval_str = schedule[2:]
            
            if interval_str.endswith('m'):
                minutes = int(interval_str[:-1])
                # Round to next interval
                current_minutes = now.minute
                next_minutes = ((current_minutes // minutes) + 1) * minutes
                
                if next_minutes >= 60:
                    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            elif interval_str.endswith('h'):
                hours = int(interval_str[:-1])
                current_hour = now.hour
                next_hour = ((current_hour // hours) + 1) * hours
                
                if next_hour >= 24:
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        else:
            # Default to next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return next_time
    
    @staticmethod
    def format_duration(seconds: float, short: bool = False) -> str:
        """Format duration in human-readable format"""
        if seconds < 0:
            return "Invalid duration"
        
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if short:
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        else:
            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            return ", ".join(parts)
    
    @staticmethod
    def parse_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def get_market_sessions() -> Dict[str, Dict[str, Any]]:
        """Get current market sessions status"""
        now_utc = datetime.now(pytz.UTC)
        
        sessions = {
            'asia': {
                'name': 'Asia/Tokyo',
                'open': 0,  # 00:00 UTC
                'close': 9,  # 09:00 UTC
                'active': False
            },
            'europe': {
                'name': 'Europe/London',
                'open': 8,  # 08:00 UTC
                'close': 17,  # 17:00 UTC
                'active': False
            },
            'america': {
                'name': 'America/New_York',
                'open': 13,  # 13:00 UTC
                'close': 22,  # 22:00 UTC
                'active': False
            }
        }
        
        current_hour = now_utc.hour
        
        for session_name, session in sessions.items():
            if session['open'] <= current_hour < session['close']:
                session['active'] = True
        
        return sessions
    
    @staticmethod
    def sleep_until(target_time: datetime):
        """Sleep until target time"""
        now = datetime.now(target_time.tzinfo or pytz.UTC)
        delta = (target_time - now).total_seconds()
        
        if delta > 0:
            time.sleep(delta)


class MathUtils:
    """Enhanced mathematical utilities for trading"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float,
                   default: Optional[float] = None) -> Optional[float]:
        """
        Safe division with customizable default
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero (None or 0.0)
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float,
                                  precision: int = 2) -> float:
        """Calculate percentage change with precision"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        change = ((new_value - old_value) / old_value) * 100
        return round(change, precision)
    
    @staticmethod
    def moving_average(values: List[float], window: int,
                      ma_type: str = 'sma') -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of values
            window: Window size
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        if len(values) < window:
            return []
        
        if ma_type == 'sma':
            # Simple Moving Average
            ma = []
            for i in range(window - 1, len(values)):
                avg = sum(values[i - window + 1:i + 1]) / window
                ma.append(avg)
            return ma
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            multiplier = 2 / (window + 1)
            ema = [sum(values[:window]) / window]  # First EMA is SMA
            
            for i in range(window, len(values)):
                ema_val = (values[i] - ema[-1]) * multiplier + ema[-1]
                ema.append(ema_val)
            
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            wma = []
            
            for i in range(window - 1, len(values)):
                weighted_sum = sum(
                    values[i - j] * weights[window - 1 - j]
                    for j in range(window)
                )
                wma.append(weighted_sum / weight_sum)
            
            return wma
        
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percent, peak_index, trough_index)
        """
        if not values:
            return 0.0, 0, 0
        
        peak = values[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd * 100, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def kelly_criterion(win_probability: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
        
        Returns:
            Fraction of capital to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        loss_probability = 1 - win_probability
        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Cap at 25% for safety
        return max(0, min(kelly, 0.25))


class AsyncUtils:
    """Enhanced asynchronous utilities"""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync context"""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Create a new thread to run the coroutine
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            # No running loop, create new one
            return asyncio.run(coro)
    
    @staticmethod
    async def gather_with_limit(coros: List, limit: int = 10,
                               return_exceptions: bool = True):
        """Run multiple coroutines with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros],
            return_exceptions=return_exceptions
        )
    
    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: tuple = (Exception,),
                   logger: Optional[logging.Logger] = None):
        """
        Async retry decorator with exponential backoff and logging
        
        Args:
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            exceptions: Exceptions to catch
            logger: Logger for retry attempts
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                            )
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                if logger:
                    logger.error(
                        f"All retries failed for {func.__name__}: {last_exception}"
                    )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    async def timeout_with_fallback(coro, timeout: float,
                                   fallback: Any = None):
        """Execute coroutine with timeout and fallback value"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return fallback


class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, source_tf: str,
                      target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            source_tf: Source timeframe (e.g., '1m')
            target_tf: Target timeframe (e.g., '5m')
        """
        if source_tf == target_tf:
            return data.copy()
        
        # Convert timeframes to pandas freq
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        target_freq = freq_map.get(target_tf, '5T')
        
        # Set timestamp as index
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled.dropna(inplace=True)
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr',
                       threshold: float = 1.5) -> List[int]:
        """
        Detect outliers in data
        
        Args:
            data: List of values
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for detection
        
        Returns:
            List of outlier indices
        """
        if not data or len(data) < 3:
            return []
        
        arr = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(arr):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std > 0:
                for i, value in enumerate(arr):
                    z_score = abs((value - mean) / std)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """
        Normalize data
        
        Args:
            data: List of values
            method: Normalization method ('minmax', 'zscore')
        """
        if not data:
            return []
        
        arr = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            if max_val == min_val:
                return [0.5] * len(data)
            
            normalized = (arr - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                return [0.0] * len(data)
            
            normalized = (arr - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()


# Singleton error handler instance getter
_error_handler_instance = None

def get_error_handler():
    """Get or create error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        from .error_handler import NexlifyErrorHandler
        _error_handler_instance = NexlifyErrorHandler()
    return _error_handler_instance


# Example usage and testing
if __name__ == "__main__":
    # Test utilities
    print("Testing Nexlify Utilities...")
    
    # Test file utils
    test_data = {"test": "data", "timestamp": datetime.now()}
    FileUtils.safe_json_save(test_data, "test.json")
    loaded = FileUtils.safe_json_load("test.json")
    print(f"File utils: {loaded}")
    
    # Test crypto utils
    btc_address = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    print(f"BTC address valid: {CryptoUtils.validate_address(btc_address, 'BTC')}")
    
    # Test validation
    email = "trader@nexlify.com"
    print(f"Email valid: {ValidationUtils.validate_email(email)}")
    
    # Test math utils
    prices = [100, 102, 98, 105, 103, 99, 101]
    ma = MathUtils.moving_average(prices, 3)
    print(f"Moving average: {ma}")
    
    print("\nAll tests completed!")
