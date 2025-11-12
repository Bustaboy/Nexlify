"""
Enhanced Multi-Mode RL Trading Agent

Comprehensive RL environment supporting:
- Spot trading (buy/sell with partial position sizing)
- Futures trading (long/short with leverage)
- Margin trading
- DeFi operations (liquidity pools, staking, yield farming, DEX swaps)
- Risk management (liquidation prevention, position reduction)
- Comprehensive fee tracking (trading fees, gas fees, funding fees)
- Liquidity modeling (slippage calculation, liquidity depth checks)

Actions: 30 (vs 3 in basic agent)
State Features: 31 (vs 8 in basic agent)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode types"""
    SPOT = "spot"
    FUTURES_LONG = "futures_long"
    FUTURES_SHORT = "futures_short"
    MARGIN = "margin"
    DEX_SWAP = "dex_swap"
    LIQUIDITY_POOL = "liquidity_pool"
    YIELD_FARM = "yield_farm"
    STAKING = "staking"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class PositionSize(Enum):
    """Partial position sizing"""
    QUARTER = 0.25
    HALF = 0.50
    THREE_QUARTER = 0.75
    FULL = 1.00


@dataclass
class Position:
    """Trading position"""
    mode: TradingMode
    size: float
    entry_price: float
    leverage: float = 1.0
    liquidation_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: int = 0

    # DeFi-specific fields
    apy: float = 0.0
    impermanent_loss: float = 0.0
    pool_share: float = 0.0


class EnhancedTradingEnvironment:
    """
    Enhanced trading environment with multi-mode support

    Action Space (30 actions):
    - 0: Hold
    - 1-4: Buy Spot (25%, 50%, 75%, 100%)
    - 5-8: Sell Spot (25%, 50%, 75%, 100%)
    - 9-12: Long Futures (25%, 50%, 75%, 100%)
    - 13: Close All Long Futures
    - 14-17: Short Futures (25%, 50%, 75%, 100%)
    - 18: Close All Short Futures
    - 19-22: Add Liquidity (25%, 50%, 75%, 100%)
    - 23: Remove All Liquidity
    - 24-26: Stake (25%, 50%, 100%)
    - 27: Unstake All
    - 28: Close All Positions (emergency exit)
    - 29: Reduce All Positions by 50%

    State Space (31 features):
    - Account status (5): balance, margin, total_value, equity, margin_ratio
    - Market data (8): price, price_change, rsi, macd, volume, volatility, trend, momentum
    - Spot positions (3): position, entry_ratio, pnl_ratio
    - Futures long (4): position, entry_ratio, pnl_ratio, liquidation_distance
    - Futures short (4): position, entry_ratio, pnl_ratio, liquidation_distance
    - Total exposure (1): total_exposure_ratio
    - DeFi status (5): lp_value, lp_apy, lp_impermanent_loss, staked_amount, staking_apy
    - Fee tracking (1): cumulative_fees_ratio
    """

    def __init__(
        self,
        price_data: np.ndarray,
        initial_balance: float = 10000,
        max_leverage: float = 10.0,
        trading_fee: float = 0.001,  # 0.1%
        funding_rate: float = 0.0001,  # 0.01% per 8h
        margin_interest: float = 0.0002,  # 0.02% per day
        gas_fee_usd: float = 5.0,  # $5 per DeFi operation
        dex_swap_fee: float = 0.003,  # 0.3% for DEX swaps
        market_liquidity_ratio: float = 100.0,  # Market liquidity as multiple of balance
        max_slippage_tolerance: float = 0.05,  # 5% max slippage
        lp_base_apy: float = 0.15,  # 15% APY for liquidity pools
        staking_base_apy: float = 0.08,  # 8% APY for staking
    ):
        """
        Initialize enhanced trading environment

        Args:
            price_data: Historical price data
            initial_balance: Starting capital in USD
            max_leverage: Maximum leverage allowed
            trading_fee: Trading fee as decimal (0.001 = 0.1%)
            funding_rate: Funding rate for futures (per 8 hours)
            margin_interest: Interest on leveraged positions (per day)
            gas_fee_usd: Gas fee for DeFi operations in USD
            dex_swap_fee: DEX swap fee as decimal
            market_liquidity_ratio: Market liquidity depth (multiple of initial balance)
            max_slippage_tolerance: Maximum acceptable slippage
            lp_base_apy: Base APY for liquidity pools
            staking_base_apy: Base APY for staking
        """
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.trading_fee = trading_fee
        self.funding_rate = funding_rate
        self.margin_interest = margin_interest
        self.gas_fee_usd = gas_fee_usd
        self.dex_swap_fee = dex_swap_fee
        self.market_liquidity_ratio = market_liquidity_ratio
        self.max_slippage_tolerance = max_slippage_tolerance
        self.lp_base_apy = lp_base_apy
        self.staking_base_apy = staking_base_apy

        # Liquidity modeling
        self.liquidity_depth = initial_balance * market_liquidity_ratio

        # Environment state
        self.current_step = 0
        self.max_steps = len(price_data) - 1

        # Account state
        self.balance = initial_balance
        self.equity = initial_balance

        # Positions
        self.spot_positions: List[Position] = []
        self.futures_long_positions: List[Position] = []
        self.futures_short_positions: List[Position] = []
        self.lp_positions: List[Position] = []
        self.staked_positions: List[Position] = []

        # Tracking
        self.total_fees_paid = 0.0
        self.total_trades = 0
        self.liquidation_count = 0
        self.trade_history = []

        # Action/state space
        self.action_space_n = 30
        self.state_space_n = 31

        # Price indicators cache
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.price_history = deque(maxlen=100)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        # Clear positions
        self.spot_positions = []
        self.futures_long_positions = []
        self.futures_short_positions = []
        self.lp_positions = []
        self.staked_positions = []

        # Reset tracking
        self.total_fees_paid = 0.0
        self.total_trades = 0
        self.liquidation_count = 0
        self.trade_history = []
        self.price_history.clear()

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state (31 features)

        Returns:
            State vector of shape (31,)
        """
        current_price = self.price_data[self.current_step]
        self.price_history.append(current_price)

        # Get portfolio value
        portfolio_value = self.get_portfolio_value()

        # Calculate positions
        spot_value = sum(p.size * current_price for p in self.spot_positions)
        long_value = sum(p.size * p.leverage for p in self.futures_long_positions)
        short_value = sum(p.size * p.leverage for p in self.futures_short_positions)
        lp_value = sum(p.size for p in self.lp_positions)
        staked_value = sum(p.size for p in self.staked_positions)

        # Calculate P&L
        spot_pnl = sum((current_price - p.entry_price) * p.size for p in self.spot_positions)
        long_pnl = sum((current_price - p.entry_price) * p.size * p.leverage for p in self.futures_long_positions)
        short_pnl = sum((p.entry_price - current_price) * p.size * p.leverage for p in self.futures_short_positions)

        # Account status (5 features)
        balance_ratio = self.balance / self.initial_balance
        margin_available_ratio = max(0, self.balance) / self.initial_balance
        total_value_ratio = portfolio_value / self.initial_balance
        equity_ratio = self.equity / self.initial_balance
        margin_ratio = (long_value + short_value) / max(self.equity, 1) if self.equity > 0 else 0

        # Market data (8 features)
        price_normalized = current_price / self.initial_balance
        price_change = self._calculate_price_change()
        rsi = self._calculate_rsi()
        macd = self._calculate_macd()
        volume = self._calculate_volume()
        volatility = self._calculate_volatility()
        trend = self._calculate_trend()
        momentum = self._calculate_momentum()

        # Spot positions (3 features)
        spot_position_ratio = spot_value / self.initial_balance
        spot_entry_ratio = (
            np.mean([p.entry_price for p in self.spot_positions]) / current_price
            if self.spot_positions else 1.0
        )
        spot_pnl_ratio = spot_pnl / self.initial_balance

        # Futures long (4 features)
        long_position_ratio = long_value / self.initial_balance
        long_entry_ratio = (
            np.mean([p.entry_price for p in self.futures_long_positions]) / current_price
            if self.futures_long_positions else 1.0
        )
        long_pnl_ratio = long_pnl / self.initial_balance
        long_liquidation_distance = self._calculate_liquidation_distance(self.futures_long_positions, current_price)

        # Futures short (4 features)
        short_position_ratio = short_value / self.initial_balance
        short_entry_ratio = (
            np.mean([p.entry_price for p in self.futures_short_positions]) / current_price
            if self.futures_short_positions else 1.0
        )
        short_pnl_ratio = short_pnl / self.initial_balance
        short_liquidation_distance = self._calculate_liquidation_distance(self.futures_short_positions, current_price, is_short=True)

        # Total exposure (1 feature)
        total_exposure_ratio = (spot_value + long_value + short_value + lp_value + staked_value) / self.initial_balance

        # DeFi status (5 features)
        lp_value_ratio = lp_value / self.initial_balance
        lp_apy = np.mean([p.apy for p in self.lp_positions]) if self.lp_positions else 0.0
        lp_impermanent_loss_ratio = sum(p.impermanent_loss for p in self.lp_positions) / self.initial_balance
        staked_amount_ratio = staked_value / self.initial_balance
        staking_apy = self.staking_base_apy if self.staked_positions else 0.0

        # Fee tracking (1 feature)
        cumulative_fees_ratio = self.total_fees_paid / self.initial_balance

        # Construct state vector (31 features)
        state = np.array([
            # Account (5)
            balance_ratio,
            margin_available_ratio,
            total_value_ratio,
            equity_ratio,
            margin_ratio,
            # Market (8)
            price_normalized,
            price_change,
            rsi,
            macd,
            volume,
            volatility,
            trend,
            momentum,
            # Spot (3)
            spot_position_ratio,
            spot_entry_ratio,
            spot_pnl_ratio,
            # Long (4)
            long_position_ratio,
            long_entry_ratio,
            long_pnl_ratio,
            long_liquidation_distance,
            # Short (4)
            short_position_ratio,
            short_entry_ratio,
            short_pnl_ratio,
            short_liquidation_distance,
            # Total (1)
            total_exposure_ratio,
            # DeFi (5)
            lp_value_ratio,
            lp_apy,
            lp_impermanent_loss_ratio,
            staked_amount_ratio,
            staking_apy,
            # Fees (1)
            cumulative_fees_ratio,
        ], dtype=np.float32)

        # Replace NaN/inf with 0
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state

        Args:
            action: Action index (0-29)

        Returns:
            (next_state, reward, done, info)
        """
        # Execute action
        reward, info = self._execute_action(action)

        # Apply fees (funding, margin interest)
        self._apply_periodic_fees()

        # Check for liquidations
        liquidated = self._check_liquidations()
        if liquidated:
            reward -= 1.0  # Heavy penalty for liquidation
            info['liquidation'] = True
            self.liquidation_count += 1

        # Update equity
        self.equity = self.get_portfolio_value()

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.equity <= 0

        # Get next state
        next_state = self._get_state()

        # Add unrealized P&L to reward (small weight)
        unrealized_pnl = self.equity - self.initial_balance
        reward += unrealized_pnl * 0.01

        return next_state, reward, done, info

    def _execute_action(self, action: int) -> Tuple[float, Dict]:
        """
        Execute trading action

        Args:
            action: Action index (0-29)

        Returns:
            (reward, info)
        """
        current_price = self.price_data[self.current_step]
        reward = 0.0
        info = {'action': self._get_action_name(action), 'trade': False}

        # Action 0: Hold
        if action == 0:
            reward -= 0.0001  # Small penalty to encourage action

        # Actions 1-4: Buy Spot (25%, 50%, 75%, 100%)
        elif 1 <= action <= 4:
            size_pct = [0.25, 0.50, 0.75, 1.00][action - 1]
            reward_delta = self._open_spot_position(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Actions 5-8: Sell Spot (25%, 50%, 75%, 100%)
        elif 5 <= action <= 8:
            size_pct = [0.25, 0.50, 0.75, 1.00][action - 5]
            reward_delta = self._close_spot_position(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Actions 9-12: Long Futures (25%, 50%, 75%, 100%)
        elif 9 <= action <= 12:
            size_pct = [0.25, 0.50, 0.75, 1.00][action - 9]
            reward_delta = self._open_futures_long(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Action 13: Close All Long Futures
        elif action == 13:
            reward_delta = self._close_futures_long(current_price, 1.0)
            reward += reward_delta
            info['trade'] = True

        # Actions 14-17: Short Futures (25%, 50%, 75%, 100%)
        elif 14 <= action <= 17:
            size_pct = [0.25, 0.50, 0.75, 1.00][action - 14]
            reward_delta = self._open_futures_short(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Action 18: Close All Short Futures
        elif action == 18:
            reward_delta = self._close_futures_short(current_price, 1.0)
            reward += reward_delta
            info['trade'] = True

        # Actions 19-22: Add Liquidity (25%, 50%, 75%, 100%)
        elif 19 <= action <= 22:
            size_pct = [0.25, 0.50, 0.75, 1.00][action - 19]
            reward_delta = self._add_liquidity(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Action 23: Remove All Liquidity
        elif action == 23:
            reward_delta = self._remove_liquidity(current_price, 1.0)
            reward += reward_delta
            info['trade'] = True

        # Actions 24-26: Stake (25%, 50%, 100%)
        elif 24 <= action <= 26:
            size_pct = [0.25, 0.50, 1.00][action - 24]
            reward_delta = self._stake(current_price, size_pct)
            reward += reward_delta
            info['trade'] = True

        # Action 27: Unstake All
        elif action == 27:
            reward_delta = self._unstake(current_price, 1.0)
            reward += reward_delta
            info['trade'] = True

        # Action 28: Close All Positions
        elif action == 28:
            reward_delta = self._close_all_positions(current_price)
            reward += reward_delta
            info['trade'] = True
            info['emergency_exit'] = True

        # Action 29: Reduce All Positions by 50%
        elif action == 29:
            reward_delta = self._reduce_all_positions(current_price, 0.5)
            reward += reward_delta
            info['trade'] = True
            info['risk_reduction'] = True

        return reward, info

    def _open_spot_position(self, price: float, size_pct: float) -> float:
        """Open spot position"""
        if self.balance <= 0:
            return 0.0

        # Calculate order size
        order_size_usd = self.balance * size_pct

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(order_size_usd)
        if not is_sufficient:
            logger.debug(f"Liquidity check failed: {reason}")
            return -0.01  # Small penalty for failed trade

        # Apply slippage to price
        effective_price = price * (1 + slippage)

        # Calculate fees
        fee = order_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Calculate position size in coins
        coins = (order_size_usd - fee) / effective_price

        # Update balance
        self.balance -= order_size_usd

        # Create position
        position = Position(
            mode=TradingMode.SPOT,
            size=coins,
            entry_price=effective_price,
            timestamp=self.current_step
        )
        self.spot_positions.append(position)
        self.total_trades += 1

        logger.debug(f"Opened spot position: {coins:.4f} @ ${effective_price:.2f} (slippage: {slippage*100:.2f}%)")

        return -fee  # Fee is negative reward

    def _close_spot_position(self, price: float, size_pct: float) -> float:
        """Close spot position (full or partial)"""
        if not self.spot_positions:
            return 0.0

        total_coins = sum(p.size for p in self.spot_positions)
        coins_to_sell = total_coins * size_pct

        if coins_to_sell <= 0:
            return 0.0

        # Calculate order size
        order_size_usd = coins_to_sell * price

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(order_size_usd)
        if not is_sufficient:
            logger.debug(f"Liquidity check failed: {reason}")
            return -0.01

        # Apply slippage to price (negative for selling)
        effective_price = price * (1 - slippage)

        # Calculate fees
        fee = order_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Calculate P&L
        avg_entry = np.mean([p.entry_price for p in self.spot_positions])
        pnl = (effective_price - avg_entry) * coins_to_sell - fee

        # Update balance
        self.balance += order_size_usd - fee

        # Remove positions (FIFO)
        remaining = coins_to_sell
        positions_to_remove = []
        for i, pos in enumerate(self.spot_positions):
            if remaining <= 0:
                break
            if pos.size <= remaining:
                remaining -= pos.size
                positions_to_remove.append(i)
            else:
                pos.size -= remaining
                remaining = 0

        for i in reversed(positions_to_remove):
            self.spot_positions.pop(i)

        self.total_trades += 1
        logger.debug(f"Closed spot position: {coins_to_sell:.4f} @ ${effective_price:.2f}, P&L: ${pnl:.2f}")

        return pnl

    def _open_futures_long(self, price: float, size_pct: float) -> float:
        """Open futures long position with leverage"""
        if self.balance <= 0:
            return 0.0

        # Calculate position size
        margin = self.balance * size_pct
        position_size_usd = margin * self.max_leverage

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(position_size_usd)
        if not is_sufficient:
            logger.debug(f"Liquidity check failed: {reason}")
            return -0.01

        # Apply slippage
        effective_price = price * (1 + slippage)

        # Calculate fees
        fee = position_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Deduct margin + fee
        self.balance -= (margin + fee)

        # Calculate liquidation price (90% of leverage)
        liquidation_price = effective_price * (1 - 0.9 / self.max_leverage)

        # Create position
        position = Position(
            mode=TradingMode.FUTURES_LONG,
            size=position_size_usd / effective_price,
            entry_price=effective_price,
            leverage=self.max_leverage,
            liquidation_price=liquidation_price,
            timestamp=self.current_step
        )
        self.futures_long_positions.append(position)
        self.total_trades += 1

        logger.debug(f"Opened long futures: {position.size:.4f} @ ${effective_price:.2f} ({self.max_leverage}x)")

        return -fee

    def _close_futures_long(self, price: float, size_pct: float) -> float:
        """Close futures long positions"""
        if not self.futures_long_positions:
            return 0.0

        total_size = sum(p.size for p in self.futures_long_positions)
        size_to_close = total_size * size_pct

        if size_to_close <= 0:
            return 0.0

        # Calculate order size
        order_size_usd = size_to_close * price

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(order_size_usd)
        if not is_sufficient:
            return -0.01

        # Apply slippage
        effective_price = price * (1 - slippage)

        # Calculate fees
        fee = order_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Calculate P&L (with leverage)
        avg_entry = np.mean([p.entry_price for p in self.futures_long_positions])
        avg_leverage = np.mean([p.leverage for p in self.futures_long_positions])
        pnl = (effective_price - avg_entry) * size_to_close * avg_leverage - fee

        # Return margin + P&L
        margin_returned = (size_to_close * avg_entry) / avg_leverage
        self.balance += margin_returned + pnl

        # Remove positions (FIFO)
        remaining = size_to_close
        positions_to_remove = []
        for i, pos in enumerate(self.futures_long_positions):
            if remaining <= 0:
                break
            if pos.size <= remaining:
                remaining -= pos.size
                positions_to_remove.append(i)
            else:
                pos.size -= remaining
                remaining = 0

        for i in reversed(positions_to_remove):
            self.futures_long_positions.pop(i)

        self.total_trades += 1
        logger.debug(f"Closed long futures: {size_to_close:.4f} @ ${effective_price:.2f}, P&L: ${pnl:.2f}")

        return pnl

    def _open_futures_short(self, price: float, size_pct: float) -> float:
        """Open futures short position"""
        if self.balance <= 0:
            return 0.0

        # Calculate position size
        margin = self.balance * size_pct
        position_size_usd = margin * self.max_leverage

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(position_size_usd)
        if not is_sufficient:
            return -0.01

        # Apply slippage (negative for short)
        effective_price = price * (1 - slippage)

        # Calculate fees
        fee = position_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Deduct margin + fee
        self.balance -= (margin + fee)

        # Calculate liquidation price
        liquidation_price = effective_price * (1 + 0.9 / self.max_leverage)

        # Create position
        position = Position(
            mode=TradingMode.FUTURES_SHORT,
            size=position_size_usd / effective_price,
            entry_price=effective_price,
            leverage=self.max_leverage,
            liquidation_price=liquidation_price,
            timestamp=self.current_step
        )
        self.futures_short_positions.append(position)
        self.total_trades += 1

        logger.debug(f"Opened short futures: {position.size:.4f} @ ${effective_price:.2f} ({self.max_leverage}x)")

        return -fee

    def _close_futures_short(self, price: float, size_pct: float) -> float:
        """Close futures short positions"""
        if not self.futures_short_positions:
            return 0.0

        total_size = sum(p.size for p in self.futures_short_positions)
        size_to_close = total_size * size_pct

        if size_to_close <= 0:
            return 0.0

        # Calculate order size
        order_size_usd = size_to_close * price

        # Check liquidity
        is_sufficient, slippage, reason = self._check_liquidity_sufficient(order_size_usd)
        if not is_sufficient:
            return -0.01

        # Apply slippage
        effective_price = price * (1 + slippage)

        # Calculate fees
        fee = order_size_usd * self.trading_fee
        self.total_fees_paid += fee

        # Calculate P&L (with leverage)
        avg_entry = np.mean([p.entry_price for p in self.futures_short_positions])
        avg_leverage = np.mean([p.leverage for p in self.futures_short_positions])
        pnl = (avg_entry - effective_price) * size_to_close * avg_leverage - fee

        # Return margin + P&L
        margin_returned = (size_to_close * avg_entry) / avg_leverage
        self.balance += margin_returned + pnl

        # Remove positions (FIFO)
        remaining = size_to_close
        positions_to_remove = []
        for i, pos in enumerate(self.futures_short_positions):
            if remaining <= 0:
                break
            if pos.size <= remaining:
                remaining -= pos.size
                positions_to_remove.append(i)
            else:
                pos.size -= remaining
                remaining = 0

        for i in reversed(positions_to_remove):
            self.futures_short_positions.pop(i)

        self.total_trades += 1
        logger.debug(f"Closed short futures: {size_to_close:.4f} @ ${effective_price:.2f}, P&L: ${pnl:.2f}")

        return pnl

    def _add_liquidity(self, price: float, size_pct: float) -> float:
        """Add liquidity to pool"""
        if self.balance <= 0:
            return 0.0

        # Calculate amount to add
        amount = self.balance * size_pct

        # Deduct gas fee
        if amount <= self.gas_fee_usd:
            return -0.01  # Not enough for gas

        self.balance -= amount
        self.total_fees_paid += self.gas_fee_usd
        amount -= self.gas_fee_usd

        # Create LP position
        position = Position(
            mode=TradingMode.LIQUIDITY_POOL,
            size=amount,
            entry_price=price,
            timestamp=self.current_step,
            apy=self.lp_base_apy,
            impermanent_loss=0.0
        )
        self.lp_positions.append(position)
        self.total_trades += 1

        logger.debug(f"Added liquidity: ${amount:.2f} (APY: {self.lp_base_apy*100:.1f}%)")

        return -self.gas_fee_usd

    def _remove_liquidity(self, price: float, size_pct: float) -> float:
        """Remove liquidity from pool"""
        if not self.lp_positions:
            return 0.0

        total_lp_value = sum(p.size for p in self.lp_positions)
        amount_to_remove = total_lp_value * size_pct

        if amount_to_remove <= 0:
            return 0.0

        # Deduct gas fee
        self.total_fees_paid += self.gas_fee_usd

        # Calculate APY rewards
        avg_apy = np.mean([p.apy for p in self.lp_positions])
        steps_held = self.current_step - np.mean([p.timestamp for p in self.lp_positions])
        days_held = steps_held / 24  # Assuming hourly data
        apy_rewards = amount_to_remove * avg_apy * (days_held / 365)

        # Calculate impermanent loss (simplified)
        avg_entry_price = np.mean([p.entry_price for p in self.lp_positions])
        price_change_pct = abs(price - avg_entry_price) / avg_entry_price
        impermanent_loss = amount_to_remove * (price_change_pct ** 2) * 0.5

        # Net return
        net_return = apy_rewards - impermanent_loss - self.gas_fee_usd

        # Update balance
        self.balance += amount_to_remove + net_return

        # Remove positions (FIFO)
        remaining = amount_to_remove
        positions_to_remove = []
        for i, pos in enumerate(self.lp_positions):
            if remaining <= 0:
                break
            if pos.size <= remaining:
                remaining -= pos.size
                positions_to_remove.append(i)
            else:
                pos.size -= remaining
                remaining = 0

        for i in reversed(positions_to_remove):
            self.lp_positions.pop(i)

        self.total_trades += 1
        logger.debug(f"Removed liquidity: ${amount_to_remove:.2f}, net: ${net_return:.2f}")

        return net_return

    def _stake(self, price: float, size_pct: float) -> float:
        """Stake tokens"""
        if self.balance <= 0:
            return 0.0

        # Calculate amount to stake
        amount = self.balance * size_pct

        # Deduct gas fee
        if amount <= self.gas_fee_usd:
            return -0.01

        self.balance -= amount
        self.total_fees_paid += self.gas_fee_usd
        amount -= self.gas_fee_usd

        # Create staking position
        position = Position(
            mode=TradingMode.STAKING,
            size=amount,
            entry_price=price,
            timestamp=self.current_step,
            apy=self.staking_base_apy
        )
        self.staked_positions.append(position)
        self.total_trades += 1

        logger.debug(f"Staked: ${amount:.2f} (APY: {self.staking_base_apy*100:.1f}%)")

        return -self.gas_fee_usd

    def _unstake(self, price: float, size_pct: float) -> float:
        """Unstake tokens"""
        if not self.staked_positions:
            return 0.0

        total_staked = sum(p.size for p in self.staked_positions)
        amount_to_unstake = total_staked * size_pct

        if amount_to_unstake <= 0:
            return 0.0

        # Deduct gas fee
        self.total_fees_paid += self.gas_fee_usd

        # Calculate staking rewards
        steps_staked = self.current_step - np.mean([p.timestamp for p in self.staked_positions])
        days_staked = steps_staked / 24
        rewards = amount_to_unstake * self.staking_base_apy * (days_staked / 365)

        # Net return
        net_return = rewards - self.gas_fee_usd

        # Update balance
        self.balance += amount_to_unstake + net_return

        # Remove positions (FIFO)
        remaining = amount_to_unstake
        positions_to_remove = []
        for i, pos in enumerate(self.staked_positions):
            if remaining <= 0:
                break
            if pos.size <= remaining:
                remaining -= pos.size
                positions_to_remove.append(i)
            else:
                pos.size -= remaining
                remaining = 0

        for i in reversed(positions_to_remove):
            self.staked_positions.pop(i)

        self.total_trades += 1
        logger.debug(f"Unstaked: ${amount_to_unstake:.2f}, rewards: ${rewards:.2f}")

        return net_return

    def _close_all_positions(self, price: float) -> float:
        """Emergency close all positions"""
        total_pnl = 0.0

        # Close spot
        total_pnl += self._close_spot_position(price, 1.0)

        # Close futures
        total_pnl += self._close_futures_long(price, 1.0)
        total_pnl += self._close_futures_short(price, 1.0)

        # Remove LP
        total_pnl += self._remove_liquidity(price, 1.0)

        # Unstake
        total_pnl += self._unstake(price, 1.0)

        logger.info(f"Emergency exit: closed all positions, P&L: ${total_pnl:.2f}")

        return total_pnl

    def _reduce_all_positions(self, price: float, reduction_pct: float) -> float:
        """Reduce all positions by percentage (risk management)"""
        total_pnl = 0.0

        # Reduce spot
        total_pnl += self._close_spot_position(price, reduction_pct)

        # Reduce futures
        total_pnl += self._close_futures_long(price, reduction_pct)
        total_pnl += self._close_futures_short(price, reduction_pct)

        logger.debug(f"Reduced all positions by {reduction_pct*100:.0f}%, P&L: ${total_pnl:.2f}")

        return total_pnl

    def _calculate_slippage(self, order_size_usd: float) -> float:
        """
        Calculate price slippage based on order size vs market liquidity

        Args:
            order_size_usd: Order size in USD

        Returns:
            Slippage as a percentage (0.0 to 1.0)

        Formula: slippage = (order_size / liquidity_depth)^2 * base_slippage
        Larger orders cause exponentially more slippage
        """
        if self.liquidity_depth <= 0:
            return 0.0

        # Calculate slippage (quadratic function of order size)
        size_ratio = order_size_usd / self.liquidity_depth
        base_slippage = 0.001  # 0.1% base slippage for small orders
        slippage = (size_ratio ** 2) * base_slippage * 100

        return min(slippage, self.max_slippage_tolerance)

    def _check_liquidity_sufficient(self, order_size_usd: float) -> Tuple[bool, float, str]:
        """
        Check if there's sufficient liquidity for the trade

        Args:
            order_size_usd: Order size in USD

        Returns:
            (is_sufficient, slippage, reason)
        """
        slippage = self._calculate_slippage(order_size_usd)

        if slippage > self.max_slippage_tolerance:
            return False, slippage, f"Slippage too high: {slippage*100:.2f}% > {self.max_slippage_tolerance*100:.1f}%"

        # Check if order size is reasonable vs available liquidity
        if order_size_usd > self.liquidity_depth * 0.1:  # Order > 10% of liquidity
            return False, slippage, f"Order too large: {order_size_usd:.0f} > 10% of liquidity depth"

        return True, slippage, "OK"

    def _apply_periodic_fees(self):
        """Apply funding fees and margin interest"""
        # Funding fees (every 8 hours, so hourly rate)
        hourly_funding = self.funding_rate / 8

        for pos in self.futures_long_positions + self.futures_short_positions:
            fee = pos.size * pos.entry_price * pos.leverage * hourly_funding
            self.balance -= fee
            self.total_fees_paid += fee

        # Margin interest (daily, so hourly rate)
        hourly_interest = self.margin_interest / 24

        for pos in self.futures_long_positions + self.futures_short_positions:
            interest = pos.size * pos.entry_price * hourly_interest
            self.balance -= interest
            self.total_fees_paid += interest

    def _check_liquidations(self) -> bool:
        """Check for liquidations and force close if needed"""
        current_price = self.price_data[self.current_step]
        liquidated = False

        # Check long positions
        for i in reversed(range(len(self.futures_long_positions))):
            pos = self.futures_long_positions[i]
            if pos.liquidation_price and current_price <= pos.liquidation_price:
                # Liquidated - lose entire position
                logger.warning(f"LIQUIDATED long position @ ${current_price:.2f} (liq price: ${pos.liquidation_price:.2f})")
                self.futures_long_positions.pop(i)
                liquidated = True

        # Check short positions
        for i in reversed(range(len(self.futures_short_positions))):
            pos = self.futures_short_positions[i]
            if pos.liquidation_price and current_price >= pos.liquidation_price:
                # Liquidated - lose entire position
                logger.warning(f"LIQUIDATED short position @ ${current_price:.2f} (liq price: ${pos.liquidation_price:.2f})")
                self.futures_short_positions.pop(i)
                liquidated = True

        return liquidated

    def _calculate_liquidation_distance(self, positions: List[Position], current_price: float, is_short: bool = False) -> float:
        """Calculate distance to liquidation (0 = liquidated, 1 = safe)"""
        if not positions:
            return 1.0

        distances = []
        for pos in positions:
            if pos.liquidation_price:
                if is_short:
                    # Short: liquidated when price goes up
                    distance = (pos.liquidation_price - current_price) / current_price
                else:
                    # Long: liquidated when price goes down
                    distance = (current_price - pos.liquidation_price) / current_price

                distances.append(max(0, min(1, distance)))

        return np.mean(distances) if distances else 1.0

    # Market indicator methods

    def _calculate_price_change(self) -> float:
        """Calculate price change from previous step"""
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(self.price_history) < period + 1:
            return 0.5

        prices = list(self.price_history)[-period-1:]
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 1.0

        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))

        return rsi

    def _calculate_macd(self) -> float:
        """Calculate MACD indicator"""
        if len(self.price_history) < self.macd_slow:
            return 0.0

        prices = list(self.price_history)

        # Simple EMA approximation
        ema_fast = np.mean(prices[-self.macd_fast:])
        ema_slow = np.mean(prices[-self.macd_slow:])

        macd = (ema_fast - ema_slow) / ema_slow

        return macd

    def _calculate_volume(self) -> float:
        """Proxy for volume using volatility"""
        if len(self.price_history) < 10:
            return 0.5

        prices = list(self.price_history)[-10:]
        volatility = np.std(prices) / np.mean(prices)

        return min(1.0, volatility * 10)

    def _calculate_volatility(self, period: int = 20) -> float:
        """Calculate price volatility"""
        if len(self.price_history) < period:
            return 0.0

        prices = list(self.price_history)[-period:]
        returns = np.diff(prices) / prices[:-1]

        return np.std(returns)

    def _calculate_trend(self, period: int = 50) -> float:
        """Calculate long-term trend"""
        if len(self.price_history) < period:
            return 0.0

        prices = list(self.price_history)[-period:]

        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize by price
        trend = slope / np.mean(prices)

        return trend

    def _calculate_momentum(self, period: int = 20) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < period:
            return 0.0

        current = self.price_history[-1]
        past = self.price_history[-period]

        momentum = (current - past) / past

        return momentum

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        current_price = self.price_data[self.current_step]

        # Cash
        total_value = self.balance

        # Spot positions
        spot_value = sum(p.size * current_price for p in self.spot_positions)
        total_value += spot_value

        # Futures long (with P&L)
        for pos in self.futures_long_positions:
            margin = (pos.size * pos.entry_price) / pos.leverage
            pnl = (current_price - pos.entry_price) * pos.size * pos.leverage
            total_value += margin + pnl

        # Futures short (with P&L)
        for pos in self.futures_short_positions:
            margin = (pos.size * pos.entry_price) / pos.leverage
            pnl = (pos.entry_price - current_price) * pos.size * pos.leverage
            total_value += margin + pnl

        # LP positions
        lp_value = sum(p.size for p in self.lp_positions)
        total_value += lp_value

        # Staked positions
        staked_value = sum(p.size for p in self.staked_positions)
        total_value += staked_value

        return max(0, total_value)

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        portfolio_value = self.get_portfolio_value()

        return {
            'initial_balance': self.initial_balance,
            'final_value': portfolio_value,
            'total_return_%': ((portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'liquidations': self.liquidation_count,
            'num_positions': {
                'spot': len(self.spot_positions),
                'long': len(self.futures_long_positions),
                'short': len(self.futures_short_positions),
                'lp': len(self.lp_positions),
                'staked': len(self.staked_positions)
            }
        }

    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        action_names = {
            0: "hold",
            1: "buy_spot_25%", 2: "buy_spot_50%", 3: "buy_spot_75%", 4: "buy_spot_100%",
            5: "sell_spot_25%", 6: "sell_spot_50%", 7: "sell_spot_75%", 8: "sell_spot_100%",
            9: "long_futures_25%", 10: "long_futures_50%", 11: "long_futures_75%", 12: "long_futures_100%",
            13: "close_all_long",
            14: "short_futures_25%", 15: "short_futures_50%", 16: "short_futures_75%", 17: "short_futures_100%",
            18: "close_all_short",
            19: "add_liquidity_25%", 20: "add_liquidity_50%", 21: "add_liquidity_75%", 22: "add_liquidity_100%",
            23: "remove_all_liquidity",
            24: "stake_25%", 25: "stake_50%", 26: "stake_100%",
            27: "unstake_all",
            28: "close_all_positions",
            29: "reduce_positions_50%"
        }
        return action_names.get(action, f"unknown_{action}")


# Export
__all__ = ['EnhancedTradingEnvironment', 'TradingMode', 'OrderType', 'PositionSize', 'Position']
