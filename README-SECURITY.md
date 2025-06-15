# Nexlify Security Module v2.0.8

## Overview

The Nexlify security module provides comprehensive protection for your trading system with **optional** security features. All security features are disabled by default and can be enabled based on your needs.

## Key Features

### 1. **Master Password (Optional)**
- Encrypts sensitive configuration data
- Uses Argon2 hashing (industry standard)
- Configurable via GUI or CLI

### 2. **Two-Factor Authentication (Optional)**
- TOTP-based (works with Google Authenticator, Authy, etc.)
- Backup codes for recovery
- Rate limiting to prevent brute force

### 3. **Session Management**
- JWT-based sessions with configurable timeout
- IP validation (optional)
- Automatic session cleanup

### 4. **API Key Rotation**
- Automated rotation with overlap period
- Exchange-specific validation
- Encrypted storage

### 5. **Access Control**
- IP whitelisting with CIDR support
- Automatic lockout after failed attempts
- Rate limiting per IP

### 6. **Encryption**
- AES-256 encryption via Fernet
- Secure key derivation with PBKDF2
- Automatic key rotation support

## Quick Start

### 1. Configure Security Settings

Run the security configuration helper:
```bash
python scripts/configure_security.py
```

Or configure via the GUI after starting Nexlify.

### 2. Security Recommendations

For maximum security, we recommend:
- Enable master password
- Enable 2FA
- Configure IP whitelist for your IPs
- Set API key rotation to 30 days
- Use session timeout of 60 minutes

For convenience with reasonable security:
- Skip master password
- Skip 2FA
- Disable IP whitelist
- Set API key rotation to 90 days
- Use session timeout of 240 minutes

### 3. First Login

If you enabled master password or 2FA:
1. Start Nexlify normally
2. Open the GUI
3. On first login:
   - Set your master password (if enabled)
   - Scan QR code for 2FA (if enabled)
   - Save backup codes securely

## Configuration Options

All security settings are in `config/enhanced_config.json`:

```json
{
  "security": {
    "master_password_enabled": false,      // Enable master password
    "master_password": "",                 // Set via GUI only
    "2fa_enabled": false,                  // Enable 2FA
    "2fa_secret": "",                      // Generated automatically
    "session_timeout_minutes": 60,         // Session duration
    "ip_whitelist_enabled": false,         // Enable IP restrictions
    "ip_whitelist": [],                    // Allowed IPs/CIDRs
    "max_failed_attempts": 5,              // Before lockout
    "lockout_duration_minutes": 30,        // Lockout time
    "api_key_rotation_days": 30           // 0 to disable
  }
}
```

## Security Best Practices

### 1. **Password Security**
- Use a strong, unique master password
- Never share your password
- Change it regularly

### 2. **2FA Security**
- Keep backup codes offline
- Don't screenshot QR codes
- Use a secure authenticator app

### 3. **API Key Security**
- Use read-only keys when possible
- Rotate keys regularly
- Revoke unused keys

### 4. **Network Security**
- Use IP whitelisting on public servers
- Always use HTTPS
- Monitor failed login attempts

### 5. **Operational Security**
- Review security logs regularly
- Update Nexlify regularly
- Backup encrypted configs

## Troubleshooting

### Locked Out?

1. **IP Blocked**: Wait for lockout duration or delete `EMERGENCY_STOP_ACTIVE`
2. **Lost 2FA**: Use backup codes or disable via config file
3. **Forgot Password**: 
   - Without 2FA: Delete `config/.users.enc` and `config/.keys/`
   - With 2FA: Requires config file recovery

### Common Issues

1. **"Invalid credentials"**
   - Check caps lock
   - Ensure master password is enabled
   - Verify IP is whitelisted

2. **"2FA token invalid"**
   - Check device time sync
   - Try next/previous code
   - Use backup code

3. **"Session expired"**
   - Normal behavior after timeout
   - Login again
   - Increase timeout if needed

## Emergency Access

If completely locked out:

1. Stop Nexlify
2. Edit `config/enhanced_config.json`:
   ```json
   {
     "security": {
       "master_password_enabled": false,
       "2fa_enabled": false,
       "ip_whitelist_enabled": false
     }
   }
   ```
3. Delete security files:
   ```bash
   rm -rf config/.keys config/.users.enc config/.2fa_secrets.enc
   ```
4. Restart Nexlify

⚠️ **Warning**: This will reset all security settings!

## Security Logs

Security events are logged to:
- `logs/errors.log` - Critical security events
- `logs/audit/` - Audit trail (if enabled)
- GUI Security tab - Real-time events

## API Integration

For developers integrating with Nexlify:

```python
from src.security.nexlify_advanced_security import get_security_manager

security = get_security_manager()

# Authenticate
token = security.authenticate_user(
    username="admin",
    password="your_password",
    ip_address="127.0.0.1",
    totp_token="123456"  # If 2FA enabled
)

# Validate session
session = security.validate_session(token, ip_address)
if session:
    # Access granted
    pass
```

## Support

For security issues:
- Check logs in `logs/errors.log`
- Review this documentation
- Contact support with security logs

Remember: Security is optional but recommended for production use!
