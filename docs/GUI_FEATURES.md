# Nexlify GUI Features - Phase 1 & 2

## Overview

This document describes the new GUI features added in Phase 1 (Security & Safety) and Phase 2 (Financial Management).

## New GUI Tabs

### 🚨 Emergency Tab

**Emergency Kill Switch Control Panel**

The Emergency tab provides instant access to emergency shutdown functionality:

- **Big Red Button**: Instantly stops all trading when clicked
  - Closes all open positions
  - Cancels all pending orders
  - Creates system backup
  - Locks the system

- **Status Display**: Shows current system state
  - ✅ System Operational (green) - Normal trading
  - ⚠️ KILL SWITCH ACTIVE (red) - Trading stopped

- **Reset Button**: Allows resetting the kill switch after review
  - Only enabled when system is not locked
  - Requires confirmation

- **Event History**: Shows recent emergency events
  - Timestamps
  - Trigger reasons
  - Actions taken

**When to Use:**
- Flash crash detected
- Suspicious activity
- System anomaly
- Need to stop trading immediately

---

### 💰 Tax Reports Tab

**Tax Reporting and Compliance**

Comprehensive tax reporting tools for cryptocurrency trading:

- **Tax Summary**: Real-time overview
  - Total trades for current year
  - Short-term vs long-term gains
  - Total gain/loss calculation
  - Estimated tax liability

- **Report Generation**:
  - **Form 8949**: IRS-compliant CSV export
  - **TurboTax Export**: Direct import format
  - Year selection (2020-present)

- **Cost Basis Methods**:
  - FIFO (First In, First Out) - Default
  - LIFO (Last In, First Out)
  - HIFO (Highest In, First Out)

**How to Use:**
1. View current year summary automatically
2. Select tax year from dropdown
3. Click "Generate Form 8949" for IRS reporting
4. Click "Export TurboTax" for tax software import
5. Files saved to `reports/tax/` directory

**Note:** Always consult with a tax professional. These reports are for reference only.

---

### 🌊 DeFi Tab

**DeFi Integration and Yield Farming**

Manage decentralized finance positions and passive income:

- **Portfolio Summary**: Top-level metrics
  - Total portfolio value (USD)
  - Total rewards earned
  - Number of active positions

- **Active Positions Table**: All liquidity positions
  - Protocol (Uniswap, PancakeSwap, etc.)
  - Token pair (e.g., ETH/USDC)
  - Position value in USD
  - Rewards earned
  - Impermanent loss percentage
  - Withdraw button per position

- **Actions**:
  - **🌾 Harvest All Rewards**: Collect rewards from all positions
  - **🔄 Refresh**: Update position data
  - **Withdraw**: Remove liquidity from specific position

**Supported Protocols:**
- Uniswap V3 (Ethereum)
- PancakeSwap (BSC)
- Aave (Lending)
- Compound (Lending)
- Curve Finance (Stablecoin pools)

**How to Use:**
1. View all active positions in the table
2. Click "Harvest All Rewards" to collect earnings
3. Monitor impermanent loss percentage
4. Click "Withdraw" to exit a position
5. Confirm withdrawal in popup dialog

**Risk Information:**
- Impermanent loss occurs when token prices diverge
- Higher APY often means higher risk
- Always monitor IL% regularly

---

### 💸 Withdrawals Tab

**Profit Management and Automated Withdrawals**

Manage trading profits and automate withdrawals:

- **Summary Bar**: Real-time profit tracking
  - Total profit (realized + unrealized)
  - Total withdrawn to date
  - Available for withdrawal

- **Manual Withdrawal**:
  - Amount input (USD)
  - Destination selection:
    - Cold Wallet (secure storage)
    - Bank Account (fiat)
    - Reinvest (back to trading)
  - Execute button

- **Recent Withdrawals Table**:
  - Date and time
  - Amount withdrawn
  - Destination
  - Status (pending/completed/failed)

**Safety Features:**
- Minimum operating balance protection ($1,000 default)
- Cannot withdraw more than available profit
- Confirmation dialogs for all withdrawals
- Audit trail of all transactions

**How to Use:**
1. Check "Available" amount in summary
2. Enter withdrawal amount
3. Select destination
4. Click "Execute Withdrawal"
5. Confirm in dialog
6. View status in Recent Withdrawals table

**Automated Withdrawals** (configured in Settings):
- Percentage-based: Withdraw X% of profits periodically
- Threshold-based: Withdraw when profit exceeds threshold
- Time-based: Fixed amount on schedule
- Hybrid: Combination of strategies

---

## Security Features (Backend)

### 🔐 PIN Authentication

- Secure login with Argon2 password hashing
- Account lockout after 3 failed attempts
- 15-minute lockout duration
- IP address logging for security audits

### 📊 Flash Crash Protection

Automatic monitoring of all trading pairs:
- **Minor Alert**: -5% drop
- **Major Alert**: -10% drop
- **Critical Alert**: -15% drop (triggers emergency response)

Multi-timeframe analysis (1m, 5m, 15m) prevents false alarms.

### 🛡️ System Integrity Monitoring

Continuous monitoring of critical files:
- SHA-256 checksums
- File size verification
- Modification time tracking
- Tamper detection alerts

**Monitored Files:**
- nexlify_neural_net.py
- nexlify_emergency_kill_switch.py
- nexlify_flash_crash_protection.py
- nexlify_pin_manager.py
- nexlify_integrity_monitor.py
- config/neural_config.json
- And more...

---

## Configuration

All features can be configured in `config/neural_config.json`:

```json
{
  "pin_authentication": {
    "enabled": true,
    "min_length": 4,
    "max_failed_attempts": 3
  },
  "emergency_kill_switch": {
    "enabled": true,
    "auto_backup": true
  },
  "tax_reporting": {
    "jurisdiction": "us",
    "cost_basis_method": "fifo"
  },
  "defi_integration": {
    "idle_threshold": 1000,
    "min_apy": 5.0
  },
  "profit_management": {
    "min_operating_balance": 1000
  }
}
```

---

## Keyboard Shortcuts

- **Emergency Stop**: No shortcut (deliberate - must click button)
- **Refresh Data**: F5 (all tabs)
- **Switch Tabs**: Ctrl+Tab / Ctrl+Shift+Tab

---

## Troubleshooting

### Emergency Kill Switch Won't Reset
- Check if system is locked (requires manual unlock)
- Verify no critical issues exist
- Review event history for details

### Tax Reports Show Wrong Data
- Click "Refresh Summary"
- Verify all trades are recorded
- Check cost basis method in config
- Ensure correct tax year selected

### DeFi Positions Not Showing
- Click "Refresh" button
- Check network connectivity
- Verify protocol is enabled in config
- Check logs tab for errors

### Withdrawals Failing
- Verify sufficient available profit
- Check destination configuration
- Ensure above minimum operating balance
- Review recent withdrawals for conflicts

---

## Support

For issues or questions:
1. Check the Logs tab for error messages
2. Review configuration file
3. Consult backtest results: `backtest_phase1_phase2_integration.py`
4. Report issues at: https://github.com/Bustaboy/Nexlify/issues

---

## Version History

- **v2.0.9** (2025-11-12): Phase 1 & 2 GUI Integration
  - Added Emergency Kill Switch tab
  - Added Tax Reports tab
  - Added DeFi Integration tab
  - Added Profit Management tab
  - Comprehensive backtest suite

- **v2.0.8**: Previous version (before Phase 1 & 2)

---

## Credits

Phase 1 & 2 implementation inspired by Night-City-Trader's security and financial features, adapted for Nexlify's cryptocurrency arbitrage focus.
