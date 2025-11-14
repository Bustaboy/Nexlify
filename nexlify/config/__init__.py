"""
Nexlify Configuration Module
=============================

Centralized configuration management for all Nexlify components.
"""

from nexlify.config.crypto_trading_config import (
    CryptoTradingConfig,
    CRYPTO_24_7_CONFIG,
    DEFAULT_CONFIG,
    FEATURE_PERIODS,
    PERIODS_PER_YEAR,
    validate_config,
    ConservativeCryptoConfig,
    AggressiveCryptoConfig,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
)

from nexlify.config.fee_providers import (
    FeeEstimate,
    FeeProvider,
    BinanceFeeProvider,
    CoinbaseFeeProvider,
    EthereumFeeProvider,
    PolygonFeeProvider,
    BSCFeeProvider,
    StaticFeeProvider,
    get_fee_provider,
    compare_fee_providers,
)

__all__ = [
    # Config classes
    "CryptoTradingConfig",
    "CRYPTO_24_7_CONFIG",
    "DEFAULT_CONFIG",
    "FEATURE_PERIODS",
    "PERIODS_PER_YEAR",
    "validate_config",
    "ConservativeCryptoConfig",
    "AggressiveCryptoConfig",
    "CONSERVATIVE_CONFIG",
    "AGGRESSIVE_CONFIG",
    # Fee providers
    "FeeEstimate",
    "FeeProvider",
    "BinanceFeeProvider",
    "CoinbaseFeeProvider",
    "EthereumFeeProvider",
    "PolygonFeeProvider",
    "BSCFeeProvider",
    "StaticFeeProvider",
    "get_fee_provider",
    "compare_fee_providers",
]
