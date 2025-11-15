# API Key Management Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-15

## Overview

The Nexlify API Key Management system provides secure, encrypted storage for exchange API keys with centralized access from both the training UI and main application. Keys are stored in an encrypted file and persist across sessions, eliminating the need to re-enter credentials multiple times.

## Key Features

### ✅ Encrypted Storage
- API keys stored using Fernet symmetric encryption
- Master password-based key derivation using SHA-256
- Keys protected at rest

### ✅ Centralized Management
- Single storage location for all exchange credentials
- Access from both training UI and main application
- Automatic synchronization between interfaces

### ✅ Connection Testing
- Built-in test functionality for each exchange
- Real-time validation of API credentials
- Visual status indicators

### ✅ Multi-Exchange Support
- Binance, Kraken, Coinbase, Bitfinex, Bitstamp
- Gemini, Poloniex, KuCoin, Huobi, OKX
- Testnet/Mainnet toggle for each exchange

## Architecture

### Storage Location

```
config/
└── api_keys.encrypted    # Encrypted API key storage
```

### Encryption Method

- **Algorithm:** Fernet (symmetric encryption)
- **Key Derivation:** SHA-256 hash of master password
- **Key Encoding:** URL-safe base64
- **Storage Format:** Encrypted JSON

### Security Considerations

⚠️ **Important Security Notes:**

1. **Master Password:** The default password is `"nexlify_default_password_change_me"`. You should change this in production to use the user's PIN or a secure password.

2. **File Permissions:** Ensure `config/api_keys.encrypted` has restricted permissions (e.g., `chmod 600`).

3. **Never Commit:** Add `config/api_keys.encrypted` to `.gitignore`.

4. **Backup:** Back up encrypted keys securely before system changes.

## Usage

### From Training UI

#### 1. Launch Training UI

```bash
python launch_training_ui.py
```

#### 2. Configure API Keys

In the left panel, scroll to the **"Exchange API Settings"** section:

1. **Select Exchange:** Choose from dropdown (Binance, Kraken, etc.)
2. **Enter API Key:** Type or paste your API key
3. **Enter Secret:** Type or paste your secret key
4. **Testnet:** Check if using testnet credentials
5. **Test Connection:** Click to verify credentials work
6. **Save API Keys:** Click to save to encrypted storage

#### 3. Status Indicators

- ✓ Green: API keys loaded/saved successfully
- ✗ Red: Connection failed or error
- Gray: No API keys loaded

### From Main Application

#### 1. Launch Main Application

```bash
python scripts/nexlify_launcher.py
```

#### 2. Navigate to Settings

Click the **"Settings"** tab in the main interface.

#### 3. Configure API Keys

In the **"Exchange API Configuration (Encrypted Storage)"** section:

For each exchange:
1. **Key:** Enter API key
2. **Secret:** Enter secret key
3. **Testnet:** Check if using testnet
4. **Save:** Click to save credentials
5. **Test:** Click to test connection

#### 4. View Status

Status label shows:
- "✓ API keys loaded for {exchange}" - Keys found in storage
- "✓ API keys saved for {exchange}" - Keys successfully saved
- "✓ Connection successful to {exchange}" - Test passed
- "✗ Connection failed" - Test failed
- "No API keys stored" - No keys found

## API Reference

### APIKeyManager Class

```python
from nexlify.security.api_key_manager import APIKeyManager

# Initialize
manager = APIKeyManager(password="your_secure_password")

# Add API key
manager.add_api_key(
    exchange="binance",
    api_key="your_api_key",
    secret="your_secret",
    testnet=False
)

# Get API key
keys = manager.get_api_key("binance")
if keys:
    print(keys['api_key'])
    print(keys['secret'])
    print(keys['testnet'])

# Remove API key
manager.remove_api_key("binance")

# List all exchanges with keys
exchanges = manager.list_exchanges()

# Test connection
success, message = await manager.test_connection("binance")
```

### Integration with CCXT

```python
# Get CCXT-compatible config
ccxt_config = manager.get_ccxt_config("binance")

import ccxt
exchange = ccxt.binance(ccxt_config)
balance = await exchange.fetch_balance()
```

### Singleton Access

```python
from nexlify.security.api_key_manager import get_api_key_manager

# Get singleton instance (initializes on first call)
manager = get_api_key_manager(password="your_password")

# Subsequent calls return same instance
manager2 = get_api_key_manager()  # Same instance
```

## Configuration

### Default Password

In both `training_ui.py` and `cyber_gui.py`:

```python
# CHANGE THIS IN PRODUCTION!
password = "nexlify_default_password_change_me"
self.api_key_manager = APIKeyManager(password)
```

**Recommended:** Use user's PIN or prompt for password:

```python
from nexlify.security.nexlify_pin_manager import PINManager

pin_manager = PINManager()
password = pin_manager.get_pin_hash()  # Use hashed PIN as password
self.api_key_manager = APIKeyManager(password)
```

### Storage Path

Default: `config/api_keys.encrypted`

To change:

```python
from pathlib import Path

manager = APIKeyManager(
    password="your_password",
    storage_path=Path("custom/path/api_keys.encrypted")
)
```

## Import/Export

### Export to neural_config.json

```python
manager.export_to_neural_config(Path("config/neural_config.json"))
```

This adds all API keys to the `exchanges` section:

```json
{
  "exchanges": {
    "binance": {
      "api_key": "...",
      "secret": "...",
      "testnet": false
    },
    "kraken": {
      "api_key": "...",
      "secret": "...",
      "testnet": false
    }
  }
}
```

### Import from neural_config.json

```python
count = manager.import_from_neural_config(Path("config/neural_config.json"))
print(f"Imported {count} API keys")
```

## Troubleshooting

### "cryptography not installed"

**Error:** `cryptography not installed - API keys will be stored unencrypted!`

**Solution:** Install cryptography:

```bash
pip install cryptography==41.0.4
```

### "Failed to decrypt"

**Error:** `Failed to load API keys: Failed to decrypt`

**Cause:** Wrong password or corrupted file

**Solution:**
1. Check password matches what was used to encrypt
2. Delete `config/api_keys.encrypted` and re-enter keys
3. Restore from backup if available

### "API key manager not initialized"

**Error:** `API key manager not initialized`

**Cause:** Manager failed to initialize

**Solution:**
1. Check logs for initialization error
2. Verify cryptography is installed
3. Check file permissions on config directory

### Connection Test Fails

**Error:** Test connection shows "✗ Connection failed"

**Possible Causes:**
1. **Invalid credentials:** Double-check API key and secret
2. **IP whitelist:** Check exchange IP whitelist settings
3. **Testnet mismatch:** Testnet keys won't work on mainnet
4. **Network issues:** Check internet connection
5. **Rate limiting:** Wait a moment and try again
6. **Exchange API down:** Check exchange status

## Best Practices

### 1. Secure Password

❌ **Don't:**
```python
password = "password123"  # Too weak
password = "nexlify_default_password_change_me"  # Default, change this!
```

✅ **Do:**
```python
password = user_pin_hash  # From PIN manager
password = hashlib.sha256(user_input.encode()).hexdigest()  # User-provided password
```

### 2. File Permissions

```bash
# Restrict access to encrypted file
chmod 600 config/api_keys.encrypted

# Ensure only user can read/write
ls -l config/api_keys.encrypted
# -rw------- 1 user user 1234 Nov 15 10:30 api_keys.encrypted
```

### 3. Backup

```bash
# Backup before changes
cp config/api_keys.encrypted config/api_keys.encrypted.backup

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp config/api_keys.encrypted backups/api_keys_$DATE.encrypted
```

### 4. Git Ignore

Ensure `.gitignore` includes:

```gitignore
# API Keys
config/api_keys.encrypted
config/api_keys.encrypted.backup
backups/api_keys_*.encrypted
```

### 5. Test Before Live Trading

1. ✅ Use testnet credentials first
2. ✅ Test connection for each exchange
3. ✅ Verify with small test trade
4. ✅ Monitor logs for errors
5. ✅ Only then use mainnet credentials

## Example Workflows

### Setup New Exchange

1. Go to exchange website (e.g., Binance)
2. Generate API key with appropriate permissions:
   - ✅ Read
   - ✅ Trade
   - ❌ Withdraw (unless needed)
3. Whitelist your IP address
4. Open Nexlify Training UI or Main App
5. Navigate to API settings
6. Enter API key and secret
7. Check "Testnet" if applicable
8. Click "Test Connection"
9. If successful, click "Save API Keys"
10. Keys are now saved and available in both UIs

### Migrate from neural_config.json

If you have API keys in `config/neural_config.json`:

```python
from nexlify.security.api_key_manager import APIKeyManager
from pathlib import Path

# Initialize manager
manager = APIKeyManager("your_secure_password")

# Import from neural_config.json
count = manager.import_from_neural_config(
    Path("config/neural_config.json")
)

print(f"Migrated {count} API keys to encrypted storage")

# Now keys are in encrypted format
# You can remove them from neural_config.json
```

### Update Existing Keys

1. Open either Training UI or Main Application
2. Navigate to API settings
3. Select exchange from dropdown
4. Existing keys will auto-load (masked)
5. Update API key or secret as needed
6. Click "Save API Keys"
7. Click "Test Connection" to verify
8. Changes immediately available in both UIs

## Security Checklist

Before deploying to production:

- [ ] Change default password from `nexlify_default_password_change_me`
- [ ] Set file permissions to 600 on `api_keys.encrypted`
- [ ] Add `api_keys.encrypted` to `.gitignore`
- [ ] Use testnet for initial testing
- [ ] Whitelist IP addresses on exchanges
- [ ] Limit API key permissions (no withdraw unless needed)
- [ ] Enable 2FA on exchange accounts
- [ ] Create backup of encrypted keys
- [ ] Document password recovery procedure
- [ ] Test connection before live trading

## Migration from Old System

If upgrading from a previous version:

### Step 1: Export Old Keys

```bash
# Backup old config
cp config/neural_config.json config/neural_config.backup.json
```

### Step 2: Use Import Feature

```python
from nexlify.security.api_key_manager import APIKeyManager
from pathlib import Path

manager = APIKeyManager("your_new_secure_password")
manager.import_from_neural_config(Path("config/neural_config.json"))
```

### Step 3: Verify

```bash
# Launch UI and check all exchanges load correctly
python launch_training_ui.py

# Test connections for each exchange
# Should see "✓ API keys loaded"
```

### Step 4: Clean Up (Optional)

```bash
# Remove API keys from neural_config.json
# Keep other settings
# Edit manually or use script
```

## Frequently Asked Questions

### Q: Are my API keys safe?

**A:** Yes, if you:
1. Use a strong master password
2. Protect file permissions (chmod 600)
3. Never commit to git
4. Keep backups secure

### Q: Can I use different passwords for each exchange?

**A:** No, currently all keys use the same master password. You could create separate `APIKeyManager` instances with different passwords and storage paths.

### Q: What happens if I forget the password?

**A:** You'll need to re-enter all API keys. There's no password recovery - the encryption is designed to be unbreakable without the password.

### Q: Can I share keys between multiple users?

**A:** Not recommended. Each user should have their own API keys and encrypted storage.

### Q: Is the encryption strong enough?

**A:** Yes, Fernet uses AES-128 in CBC mode with HMAC authentication. It's secure for protecting API keys at rest.

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages in UI
3. Consult CLAUDE.md for architecture details
4. File issue on GitHub with logs

---

**Version:** 1.0.0
**Last Updated:** 2025-11-15
**Maintainer:** Nexlify Development Team
