# Nexlify Project Structure v2.0.8

```
nexlify/
│
├── config/                          # Configuration files
│   ├── enhanced_config.json        # Main configuration (v2.0.8)
│   ├── neural_config.json.old      # Old config (after migration)
│   └── .env.example               # Environment variables template
│
├── src/                           # Source code
│   ├── core/                      # Core components
│   │   ├── __init__.py
│   │   ├── arasaka_neural_net.py # Main trading engine
│   │   ├── error_handler.py      # Enhanced error handling
│   │   └── utils_module.py       # Utility functions
│   │
│   ├── security/                  # Security components
│   │   ├── __init__.py
│   │   ├── nexlify_advanced_security.py
│   │   └── nexlify_audit_trail.py
│   │
│   ├── trading/                   # Trading strategies
│   │   ├── __init__.py
│   │   ├── nexlify_multi_strategy.py
│   │   ├── nexlify_dex_integration.py
│   │   └── strategies/
│   │       ├── arbitrage.py
│   │       ├── grid_trading.py
│   │       ├── momentum.py
│   │       ├── mean_reversion.py
│   │       └── sentiment.py
│   │
│   ├── predictive/               # ML and predictive features
│   │   ├── __init__.py
│   │   ├── nexlify_predictive_features.py
│   │   └── models/
│   │       ├── volatility.py
│   │       ├── liquidity.py
│   │       └── transformer.py
│   │
│   ├── gui/                      # GUI components
│   │   ├── __init__.py
│   │   ├── cyber_gui.py         # Legacy GUI (for compatibility)
│   │   ├── nexlify_enhanced_gui.py
│   │   └── nexlify_cyberpunk_effects.py
│   │
│   ├── api/                      # API components
│   │   ├── __init__.py
│   │   ├── nexlify_mobile_api.py
│   │   └── nexlify_ai_companion.py
│   │
│   └── backtesting/             # Backtesting components
│       ├── __init__.py
│       └── nexlify_advanced_backtesting.py
│
├── scripts/                      # Utility scripts
│   ├── migrate_config.py        # Config migration script
│   ├── smart_launcher.py        # Enhanced launcher
│   ├── nexlify_implementation_script.py
│   ├── setup_nexlify.py
│   ├── setup_database.py
│   └── compile_cython.py
│
├── data/                        # Data storage
│   ├── trading.db              # Main database
│   ├── market/                 # Market data cache
│   └── models/                 # Trained ML models
│
├── logs/                        # Log files
│   ├── errors.log              # Error log
│   ├── trading.log             # Trading log
│   ├── audit/                  # Audit logs
│   │   └── audit_trail.db
│   ├── crash_reports/          # Crash reports
│   ├── performance/            # Performance logs
│   └── mobile/                 # Mobile API logs
│
├── backups/                     # Backup files
│   ├── config/                 # Config backups
│   ├── database/               # Database backups
│   └── models/                 # Model backups
│
├── reports/                     # Generated reports
│   ├── compliance/             # Compliance reports
│   ├── performance/            # Performance reports
│   └── tax/                    # Tax reports
│
├── assets/                      # Static assets
│   ├── sounds/                 # Sound effects
│   │   ├── neural_boot.wav
│   │   ├── trade_execute.wav
│   │   ├── profit.wav
│   │   ├── loss.wav
│   │   ├── alert_low.wav
│   │   ├── alert_medium.wav
│   │   ├── alert_high.wav
│   │   └── click.wav
│   └── images/                 # GUI images
│       └── nexlify_logo.png
│
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── tests/                       # Test files
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── QUICK_REFERENCE.md
│   ├── setup_guide.md
│   ├── api_reference.md
│   └── troubleshooting.md
│
├── requirements.txt             # Python dependencies
├── requirements_enhanced.txt    # Additional v2.0.8 dependencies
├── .gitignore
├── LICENSE
├── README.md
├── start_nexlify.bat           # Windows launcher
├── start_nexlify.sh            # Linux/Mac launcher
└── VERSION                     # Version file (2.0.8)
```

## Key Changes in v2.0.8

### New Files
- `config/enhanced_config.json` - New unified configuration
- `scripts/migrate_config.py` - Configuration migration tool
- `src/core/error_handler.py` - Enhanced error handling with deduplication
- `requirements_enhanced.txt` - Additional dependencies for v2.0.8

### Modified Structure
- Organized source code into logical modules (core, security, trading, etc.)
- Separated strategies into individual files
- Added dedicated directories for different log types
- Enhanced backup structure

### Migration Path
1. Run `python scripts/migrate_config.py` to migrate configuration
2. Old config is backed up to `backups/config/` and renamed to `.old`
3. All components now use `enhanced_config.json`
4. Settings can be adjusted via the enhanced GUI

### Security Enhancements
- Master password and 2FA are now optional (configured via GUI)
- IP whitelisting available but disabled by default
- API key rotation configurable
- Session management with configurable timeouts

### Performance Improvements
- Error deduplication to reduce log spam
- Configurable error suppression for non-critical components
- Better resource management
- Parallel processing support
