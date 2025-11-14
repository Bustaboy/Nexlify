#!/usr/bin/env python3
"""
Dynamic Fee Provider System
============================

Provides real-time fee calculation for different networks and exchanges.

CRITICAL: Fees are incurred TWICE in round-trip trades (USD→ETH→USD)
- Entry fee: when buying crypto with USD
- Exit fee: when selling crypto back to USD

For ETH/L1 chains: gas fees can vary 10-100x based on network congestion
For CEX: trading fees are typically fixed percentage

Author: Nexlify Team
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FEE DATA STRUCTURES
# ============================================================================

@dataclass
class FeeEstimate:
    """
    Fee estimate for a trade

    Attributes:
        entry_fee_rate: Fee rate when entering position (buy)
        exit_fee_rate: Fee rate when exiting position (sell)
        entry_fixed_cost: Fixed cost for entry (e.g., gas in USD)
        exit_fixed_cost: Fixed cost for exit (e.g., gas in USD)
        total_round_trip_rate: Combined fee rate for round trip
        network: Network/exchange name
        fee_type: Type of fee ('percentage', 'gas', 'hybrid')
    """
    entry_fee_rate: float  # Percentage (0.001 = 0.1%)
    exit_fee_rate: float   # Percentage (0.001 = 0.1%)
    entry_fixed_cost: float = 0.0  # USD
    exit_fixed_cost: float = 0.0   # USD
    network: str = "unknown"
    fee_type: str = "percentage"

    @property
    def total_round_trip_rate(self) -> float:
        """Calculate total percentage cost for round trip (buy + sell)"""
        return self.entry_fee_rate + self.exit_fee_rate

    def calculate_entry_cost(self, trade_value_usd: float) -> Tuple[float, float]:
        """
        Calculate entry costs (buying crypto)

        Args:
            trade_value_usd: USD amount being invested

        Returns:
            (percentage_fee, fixed_fee) in USD
        """
        percentage_fee = trade_value_usd * self.entry_fee_rate
        return percentage_fee, self.entry_fixed_cost

    def calculate_exit_cost(self, trade_value_usd: float) -> Tuple[float, float]:
        """
        Calculate exit costs (selling crypto)

        Args:
            trade_value_usd: USD value of crypto being sold

        Returns:
            (percentage_fee, fixed_fee) in USD
        """
        percentage_fee = trade_value_usd * self.exit_fee_rate
        return percentage_fee, self.exit_fixed_cost

    def calculate_round_trip_cost(self, trade_value_usd: float) -> float:
        """
        Calculate total cost for USD→Crypto→USD round trip

        Args:
            trade_value_usd: USD amount of trade

        Returns:
            Total fees in USD
        """
        entry_pct, entry_fixed = self.calculate_entry_cost(trade_value_usd)

        # After entry, we have less USD due to fees
        remaining_value = trade_value_usd - entry_pct - entry_fixed

        # Exit fees calculated on remaining value
        exit_pct, exit_fixed = self.calculate_exit_cost(remaining_value)

        total = entry_pct + entry_fixed + exit_pct + exit_fixed
        return total


# ============================================================================
# FEE PROVIDER BASE CLASS
# ============================================================================

class FeeProvider(ABC):
    """
    Abstract base class for fee providers

    Fee providers query real-time fee data from networks/exchanges
    and return FeeEstimate objects.
    """

    @abstractmethod
    def get_fee_estimate(self,
                        asset: str = "ETH",
                        trade_size_usd: float = 1000.0) -> FeeEstimate:
        """
        Get current fee estimate for trading asset

        Args:
            asset: Asset symbol (e.g., 'ETH', 'BTC')
            trade_size_usd: Size of trade in USD (affects gas cost calculation)

        Returns:
            FeeEstimate with current fee rates
        """
        pass

    @abstractmethod
    def get_network_name(self) -> str:
        """Get network/exchange name"""
        pass


# ============================================================================
# CENTRALIZED EXCHANGE FEE PROVIDERS
# ============================================================================

class BinanceFeeProvider(FeeProvider):
    """
    Binance CEX fee provider

    Default: 0.1% maker/taker (can be reduced with BNB or volume)
    """

    def __init__(self,
                 maker_fee: float = 0.001,  # 0.1%
                 taker_fee: float = 0.001):  # 0.1%
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """Get Binance fee estimate (typically uses taker fees for market orders)"""
        return FeeEstimate(
            entry_fee_rate=self.taker_fee,
            exit_fee_rate=self.taker_fee,
            entry_fixed_cost=0.0,
            exit_fixed_cost=0.0,
            network="Binance",
            fee_type="percentage"
        )

    def get_network_name(self) -> str:
        return "Binance"


class CoinbaseFeeProvider(FeeProvider):
    """
    Coinbase Pro fee provider

    Tiered fees based on 30-day volume
    Default: 0.5% (low volume)
    """

    def __init__(self, fee_rate: float = 0.005):  # 0.5% default
        self.fee_rate = fee_rate

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """Get Coinbase fee estimate"""
        return FeeEstimate(
            entry_fee_rate=self.fee_rate,
            exit_fee_rate=self.fee_rate,
            entry_fixed_cost=0.0,
            exit_fixed_cost=0.0,
            network="Coinbase",
            fee_type="percentage"
        )

    def get_network_name(self) -> str:
        return "Coinbase"


# ============================================================================
# BLOCKCHAIN NETWORK FEE PROVIDERS
# ============================================================================

class EthereumFeeProvider(FeeProvider):
    """
    Ethereum L1 fee provider

    CRITICAL: ETH gas fees are HIGHLY variable
    - Low congestion: $2-5 per swap
    - Medium: $10-30 per swap
    - High: $50-200+ per swap

    For RL training, this is CRITICAL - a $100 trade with $10 gas fees
    means 10% fee, not the 0.1% CEX rate!

    TODO: Integrate with Web3.py to query real-time gas prices
    """

    def __init__(self,
                 gas_price_gwei: Optional[float] = None,
                 swap_gas_limit: int = 150000):
        """
        Args:
            gas_price_gwei: Gas price in Gwei (None = use estimate)
            swap_gas_limit: Gas limit for DEX swap (Uniswap ~150k)
        """
        self.gas_price_gwei = gas_price_gwei
        self.swap_gas_limit = swap_gas_limit

        # DEX trading fee (e.g., Uniswap 0.3%)
        self.dex_fee_rate = 0.003

    def estimate_gas_price_gwei(self) -> float:
        """
        Estimate current gas price

        TODO: Replace with Web3.py real-time query:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(RPC_URL))
            gas_price = w3.eth.gas_price

        Returns:
            Estimated gas price in Gwei
        """
        if self.gas_price_gwei is not None:
            return self.gas_price_gwei

        # Conservative estimate: 30 Gwei (medium congestion)
        # PRODUCTION: MUST query real-time from network
        return 30.0

    def calculate_gas_cost_usd(self, eth_price_usd: float = 2000.0) -> float:
        """
        Calculate gas cost in USD for one swap

        Args:
            eth_price_usd: Current ETH price

        Returns:
            Gas cost in USD
        """
        gas_price_gwei = self.estimate_gas_price_gwei()
        gas_price_eth = gas_price_gwei / 1e9  # Gwei to ETH
        gas_cost_eth = gas_price_eth * self.swap_gas_limit
        gas_cost_usd = gas_cost_eth * eth_price_usd

        return gas_cost_usd

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """
        Get Ethereum fee estimate

        Includes both:
        - DEX trading fee (0.3% for Uniswap)
        - Gas cost (variable, can be $5-$200)
        """
        # Get current ETH price (in production, query from oracle)
        # For now, use conservative estimate
        eth_price_usd = 2000.0

        gas_cost_usd = self.calculate_gas_cost_usd(eth_price_usd)

        logger.info(f"ETH gas estimate: {self.estimate_gas_price_gwei():.1f} Gwei = ${gas_cost_usd:.2f} per swap")

        return FeeEstimate(
            entry_fee_rate=self.dex_fee_rate,  # 0.3% DEX fee
            exit_fee_rate=self.dex_fee_rate,   # 0.3% DEX fee
            entry_fixed_cost=gas_cost_usd,     # Gas for buy swap
            exit_fixed_cost=gas_cost_usd,      # Gas for sell swap
            network="Ethereum",
            fee_type="hybrid"  # Both percentage and fixed
        )

    def get_network_name(self) -> str:
        return "Ethereum"


class PolygonFeeProvider(FeeProvider):
    """
    Polygon (Matic) L2 fee provider

    Much cheaper than Ethereum L1:
    - Gas costs: $0.01 - $0.50 per swap
    - DEX fees: 0.3% (same as Ethereum)
    """

    def __init__(self, dex_fee_rate: float = 0.003):
        self.dex_fee_rate = dex_fee_rate
        # Polygon gas is cheap enough to approximate as fixed cost
        self.avg_gas_cost_usd = 0.05

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """Get Polygon fee estimate"""
        return FeeEstimate(
            entry_fee_rate=self.dex_fee_rate,
            exit_fee_rate=self.dex_fee_rate,
            entry_fixed_cost=self.avg_gas_cost_usd,
            exit_fixed_cost=self.avg_gas_cost_usd,
            network="Polygon",
            fee_type="hybrid"
        )

    def get_network_name(self) -> str:
        return "Polygon"


class BSCFeeProvider(FeeProvider):
    """
    Binance Smart Chain fee provider

    Cheaper than Ethereum but more expensive than Polygon:
    - Gas costs: $0.10 - $2.00 per swap
    - DEX fees: 0.25% (PancakeSwap)
    """

    def __init__(self, dex_fee_rate: float = 0.0025):
        self.dex_fee_rate = dex_fee_rate
        self.avg_gas_cost_usd = 0.30

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """Get BSC fee estimate"""
        return FeeEstimate(
            entry_fee_rate=self.dex_fee_rate,
            exit_fee_rate=self.dex_fee_rate,
            entry_fixed_cost=self.avg_gas_cost_usd,
            exit_fixed_cost=self.avg_gas_cost_usd,
            network="BSC",
            fee_type="hybrid"
        )

    def get_network_name(self) -> str:
        return "BSC"


# ============================================================================
# DEFAULT/STATIC FEE PROVIDER
# ============================================================================

class StaticFeeProvider(FeeProvider):
    """
    Static fee provider for testing/backward compatibility

    Uses fixed fee rates (not recommended for production)
    """

    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate

    def get_fee_estimate(self, asset: str = "ETH", trade_size_usd: float = 1000.0) -> FeeEstimate:
        """Get static fee estimate"""
        return FeeEstimate(
            entry_fee_rate=self.fee_rate,
            exit_fee_rate=self.fee_rate,
            entry_fixed_cost=0.0,
            exit_fixed_cost=0.0,
            network="Static",
            fee_type="percentage"
        )

    def get_network_name(self) -> str:
        return "Static"


# ============================================================================
# FEE PROVIDER FACTORY
# ============================================================================

def get_fee_provider(network: str = "binance", **kwargs) -> FeeProvider:
    """
    Factory function to create fee providers

    Args:
        network: Network/exchange name
        **kwargs: Provider-specific arguments

    Returns:
        FeeProvider instance

    Examples:
        >>> provider = get_fee_provider("binance")
        >>> provider = get_fee_provider("ethereum", gas_price_gwei=50)
        >>> provider = get_fee_provider("polygon")
    """
    network_lower = network.lower()

    providers = {
        "binance": BinanceFeeProvider,
        "coinbase": CoinbaseFeeProvider,
        "ethereum": EthereumFeeProvider,
        "eth": EthereumFeeProvider,
        "polygon": PolygonFeeProvider,
        "matic": PolygonFeeProvider,
        "bsc": BSCFeeProvider,
        "static": StaticFeeProvider,
    }

    provider_class = providers.get(network_lower)

    if provider_class is None:
        logger.warning(f"Unknown network '{network}', using StaticFeeProvider")
        return StaticFeeProvider(**kwargs)

    return provider_class(**kwargs)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compare_fee_providers(trade_size_usd: float = 1000.0) -> Dict[str, FeeEstimate]:
    """
    Compare fees across different networks/exchanges

    Args:
        trade_size_usd: Trade size for comparison

    Returns:
        Dict mapping network name to FeeEstimate
    """
    networks = ["binance", "coinbase", "ethereum", "polygon", "bsc"]
    results = {}

    print(f"\nFee Comparison for ${trade_size_usd:,.2f} Round-Trip Trade (Buy + Sell)")
    print("=" * 80)
    print(f"{'Network':<15} {'Entry Fee':<15} {'Exit Fee':<15} {'Round Trip':<15} {'% Cost'}")
    print("-" * 80)

    for network in networks:
        provider = get_fee_provider(network)
        estimate = provider.get_fee_estimate(trade_size_usd=trade_size_usd)

        # Calculate costs
        entry_pct, entry_fixed = estimate.calculate_entry_cost(trade_size_usd)
        entry_total = entry_pct + entry_fixed

        remaining = trade_size_usd - entry_total
        exit_pct, exit_fixed = estimate.calculate_exit_cost(remaining)
        exit_total = exit_pct + exit_fixed

        round_trip = estimate.calculate_round_trip_cost(trade_size_usd)
        pct_cost = (round_trip / trade_size_usd) * 100

        print(f"{network:<15} ${entry_total:<14.2f} ${exit_total:<14.2f} ${round_trip:<14.2f} {pct_cost:.3f}%")

        results[network] = estimate

    print("=" * 80)
    print("\nKEY INSIGHTS:")
    print("- CEX fees are consistent across trade sizes (percentage-based)")
    print("- ETH L1 gas fees dominate for small trades (can be 5-10% of $100 trade!)")
    print("- L2 solutions (Polygon, BSC) offer good compromise")
    print("- For RL training: fee structure dramatically affects optimal trade size\n")

    return results


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
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


if __name__ == "__main__":
    # Demo fee comparison
    print("\nDemonstrating fee impact on different trade sizes:")
    print("\n" + "="*80)

    for trade_size in [100, 500, 1000, 5000]:
        compare_fee_providers(trade_size)
