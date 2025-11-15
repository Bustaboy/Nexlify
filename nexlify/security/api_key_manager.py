"""
Encrypted API Key Manager

Provides secure storage and retrieval of exchange API keys using
Fernet symmetric encryption. Keys are stored in an encrypted file
and can be accessed by both the training UI and main application.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from base64 import urlsafe_b64encode
from hashlib import sha256

logger = logging.getLogger(__name__)

# Try to import cryptography, fall back to warning if not available
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not installed - API keys will be stored unencrypted!")
    ENCRYPTION_AVAILABLE = False


class APIKeyManager:
    """
    Manages encrypted storage of exchange API keys

    Features:
    - Fernet symmetric encryption
    - Master password-based key derivation
    - Support for multiple exchanges
    - Test connection functionality
    - Automatic file creation

    Example:
        >>> manager = APIKeyManager(password="my_secure_password")
        >>> manager.add_api_key("binance", "api_key_here", "secret_here")
        >>> keys = manager.get_api_key("binance")
        >>> print(keys['api_key'])
    """

    def __init__(
        self,
        password: str,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize API key manager

        Args:
            password: Master password for encryption
            storage_path: Path to encrypted storage file
        """
        self.storage_path = storage_path or Path('config/api_keys.encrypted')
        self.password = password

        # Derive encryption key from password
        if ENCRYPTION_AVAILABLE:
            key = self._derive_key(password)
            self.cipher = Fernet(key)
        else:
            self.cipher = None
            logger.warning("Encryption not available - using plaintext storage")

        # Load existing keys
        self.api_keys: Dict[str, Dict[str, str]] = {}
        self._load_keys()

    def _derive_key(self, password: str) -> bytes:
        """
        Derive encryption key from password using SHA-256

        Args:
            password: Master password

        Returns:
            32-byte Fernet key
        """
        # Hash password to create 32-byte key
        key_material = sha256(password.encode()).digest()
        # Encode for Fernet (needs base64)
        return urlsafe_b64encode(key_material)

    def _load_keys(self) -> None:
        """Load API keys from encrypted storage file"""
        if not self.storage_path.exists():
            logger.info(f"No existing API keys file at {self.storage_path}")
            return

        try:
            with open(self.storage_path, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt if encryption is available
            if self.cipher and ENCRYPTION_AVAILABLE:
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self.api_keys = json.loads(decrypted_data.decode())
            else:
                # Fallback to plaintext
                self.api_keys = json.loads(encrypted_data.decode())

            logger.info(f"Loaded API keys for {len(self.api_keys)} exchanges")

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            logger.warning("Starting with empty API keys")
            self.api_keys = {}

    def _save_keys(self) -> None:
        """Save API keys to encrypted storage file"""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to JSON
            json_data = json.dumps(self.api_keys, indent=2)

            # Encrypt if available
            if self.cipher and ENCRYPTION_AVAILABLE:
                encrypted_data = self.cipher.encrypt(json_data.encode())
            else:
                encrypted_data = json_data.encode()

            # Write to file
            with open(self.storage_path, 'wb') as f:
                f.write(encrypted_data)

            logger.info(f"Saved API keys for {len(self.api_keys)} exchanges")

        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            raise

    def add_api_key(
        self,
        exchange: str,
        api_key: str,
        secret: str,
        testnet: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add or update API key for an exchange

        Args:
            exchange: Exchange name (e.g., 'binance', 'kraken')
            api_key: API key
            secret: API secret
            testnet: Whether this is a testnet API key
            additional_params: Additional exchange-specific parameters
        """
        self.api_keys[exchange] = {
            'api_key': api_key,
            'secret': secret,
            'testnet': testnet
        }

        if additional_params:
            self.api_keys[exchange].update(additional_params)

        self._save_keys()
        logger.info(f"Added API key for {exchange} (testnet={testnet})")

    def get_api_key(self, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Get API key for an exchange

        Args:
            exchange: Exchange name

        Returns:
            Dict with api_key, secret, and other parameters, or None if not found
        """
        return self.api_keys.get(exchange)

    def remove_api_key(self, exchange: str) -> bool:
        """
        Remove API key for an exchange

        Args:
            exchange: Exchange name

        Returns:
            True if removed, False if not found
        """
        if exchange in self.api_keys:
            del self.api_keys[exchange]
            self._save_keys()
            logger.info(f"Removed API key for {exchange}")
            return True
        return False

    def list_exchanges(self) -> list:
        """
        Get list of exchanges with stored API keys

        Returns:
            List of exchange names
        """
        return list(self.api_keys.keys())

    def has_api_key(self, exchange: str) -> bool:
        """
        Check if API key exists for exchange

        Args:
            exchange: Exchange name

        Returns:
            True if API key exists
        """
        return exchange in self.api_keys

    def get_ccxt_config(self, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Get API key in CCXT-compatible format

        Args:
            exchange: Exchange name

        Returns:
            Dict with 'apiKey' and 'secret' for CCXT, or None if not found
        """
        keys = self.get_api_key(exchange)
        if not keys:
            return None

        config = {
            'apiKey': keys['api_key'],
            'secret': keys['secret'],
        }

        # Add testnet flag if set
        if keys.get('testnet'):
            config['options'] = {'defaultType': 'future'}

        return config

    async def test_connection(self, exchange: str) -> tuple[bool, str]:
        """
        Test API connection to exchange

        Args:
            exchange: Exchange name

        Returns:
            (success, message) tuple
        """
        try:
            import ccxt.async_support as ccxt

            # Get API config
            config = self.get_ccxt_config(exchange)
            if not config:
                return False, f"No API key found for {exchange}"

            # Create exchange instance
            exchange_class = getattr(ccxt, exchange.lower(), None)
            if not exchange_class:
                return False, f"Exchange {exchange} not supported by CCXT"

            exchange_instance = exchange_class(config)

            try:
                # Test by fetching balance
                await exchange_instance.load_markets()
                balance = await exchange_instance.fetch_balance()

                await exchange_instance.close()

                # Check if we got valid response
                if balance and 'total' in balance:
                    return True, f"Successfully connected to {exchange}"
                else:
                    return False, f"Connected but received invalid response from {exchange}"

            except ccxt.AuthenticationError as e:
                await exchange_instance.close()
                return False, f"Authentication failed: {str(e)}"

            except ccxt.NetworkError as e:
                await exchange_instance.close()
                return False, f"Network error: {str(e)}"

            except Exception as e:
                await exchange_instance.close()
                return False, f"Connection test failed: {str(e)}"

        except ImportError:
            return False, "CCXT not installed"

        except Exception as e:
            return False, f"Error testing connection: {str(e)}"

    def export_to_neural_config(self, config_path: Path) -> None:
        """
        Export API keys to neural_config.json format

        Args:
            config_path: Path to neural_config.json
        """
        try:
            # Load existing config
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            # Ensure exchanges section exists
            if 'exchanges' not in config:
                config['exchanges'] = {}

            # Add all exchanges
            for exchange, keys in self.api_keys.items():
                config['exchanges'][exchange] = {
                    'api_key': keys['api_key'],
                    'secret': keys['secret'],
                    'testnet': keys.get('testnet', False)
                }

            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Exported {len(self.api_keys)} API keys to {config_path}")

        except Exception as e:
            logger.error(f"Failed to export to neural_config: {e}")
            raise

    def import_from_neural_config(self, config_path: Path) -> int:
        """
        Import API keys from neural_config.json

        Args:
            config_path: Path to neural_config.json

        Returns:
            Number of API keys imported
        """
        try:
            with open(config_path) as f:
                config = json.load(f)

            exchanges = config.get('exchanges', {})
            count = 0

            for exchange, exchange_config in exchanges.items():
                api_key = exchange_config.get('api_key')
                secret = exchange_config.get('secret')

                if api_key and secret:
                    self.add_api_key(
                        exchange=exchange,
                        api_key=api_key,
                        secret=secret,
                        testnet=exchange_config.get('testnet', False)
                    )
                    count += 1

            logger.info(f"Imported {count} API keys from {config_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to import from neural_config: {e}")
            raise


# Singleton instance management
_api_key_manager_instance: Optional[APIKeyManager] = None


def get_api_key_manager(password: Optional[str] = None) -> Optional[APIKeyManager]:
    """
    Get singleton API key manager instance

    Args:
        password: Master password (required on first call)

    Returns:
        APIKeyManager instance or None if password not provided
    """
    global _api_key_manager_instance

    if _api_key_manager_instance is None:
        if password is None:
            logger.error("Password required to initialize API key manager")
            return None

        _api_key_manager_instance = APIKeyManager(password)

    return _api_key_manager_instance


def reset_api_key_manager() -> None:
    """Reset singleton instance (useful for testing)"""
    global _api_key_manager_instance
    _api_key_manager_instance = None
