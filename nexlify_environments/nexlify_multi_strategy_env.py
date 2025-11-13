"""
Nexlify Multi-Strategy Trading Environment
Comprehensive environment supporting all trading strategies for maximum profitability

Supported Strategies:
1. Spot Trading (multi-pair)
2. DeFi Staking
3. Yield Farming / Liquidity Provision
4. Cross-Exchange Arbitrage
5. Cross-Pair Arbitrage
6. Portfolio Rebalancing
7. Leveraged Positions (optional)

Action Space:
- BUY/SELL/HOLD for each pair
- STAKE/UNSTAKE for supported assets
- ADD_LIQUIDITY/REMOVE_LIQUIDITY for DeFi pools
- ARBITRAGE_EXECUTE for cross-exchange opportunities
- REBALANCE for portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """All possible action types"""
    # Spot trading
    BUY = 0
    SELL = 1
    HOLD = 2

    # DeFi staking
    STAKE = 3
    UNSTAKE = 4

    # Liquidity provision
    ADD_LIQUIDITY = 5
    REMOVE_LIQUIDITY = 6

    # Arbitrage
    ARBITRAGE_BUY = 7
    ARBITRAGE_SELL = 8

    # Portfolio
    REBALANCE = 9


@dataclass
class Asset:
    """Asset holdings"""
    symbol: str
    amount: float
    staked_amount: float = 0.0
    lp_tokens: float = 0.0  # Liquidity pool tokens
    avg_entry_price: float = 0.0


@dataclass
class StakingPool:
    """Staking pool configuration"""
    asset: str
    apy: float  # Annual percentage yield
    lock_period: int = 0  # Hours (0 = no lock)
    min_stake: float = 0.0


@dataclass
class LiquidityPool:
    """Liquidity pool configuration"""
    pair: str  # e.g., "BTC/ETH"
    fee_rate: float  # LP fee (e.g., 0.003 for 0.3%)
    apy: float  # Estimated APY from fees
    token0: str
    token1: str


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    pair: str
    exchange1: str
    exchange2: str
    price1: float
    price2: float
    profit_potential: float  # Percentage
    volume_limit: float  # Max volume for this opportunity


class MultiStrategyEnvironment:
    """
    Comprehensive trading environment supporting all strategies
    """

    def __init__(
        self,
        trading_pairs: List[str],
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        market_data: Dict[str, np.ndarray] = None,
        enable_staking: bool = True,
        enable_defi: bool = True,
        enable_arbitrage: bool = True,
        staking_pools: Optional[List[StakingPool]] = None,
        liquidity_pools: Optional[List[LiquidityPool]] = None
    ):
        """
        Initialize multi-strategy environment

        Args:
            trading_pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
            initial_balance: Starting balance in quote currency (USDT)
            fee_rate: Trading fee rate
            slippage: Slippage rate
            market_data: Dictionary of price arrays for each pair
            enable_staking: Enable staking strategies
            enable_defi: Enable DeFi/liquidity provision
            enable_arbitrage: Enable arbitrage detection
            staking_pools: List of available staking pools
            liquidity_pools: List of available liquidity pools
        """
        self.trading_pairs = trading_pairs
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.market_data = market_data or {}

        # Strategy enablement
        self.enable_staking = enable_staking
        self.enable_defi = enable_defi
        self.enable_arbitrage = enable_arbitrage

        # DeFi configurations
        self.staking_pools = staking_pools or self._create_default_staking_pools()
        self.liquidity_pools = liquidity_pools or self._create_default_liquidity_pools()

        # Calculate action space size
        self.action_size = self._calculate_action_space()

        # State components (will be calculated dynamically)
        self.state_size = self._calculate_state_space()

        # Portfolio state
        self.reset()

        logger.info(f"Multi-Strategy Environment initialized")
        logger.info(f"  Trading pairs: {len(trading_pairs)}")
        logger.info(f"  Staking enabled: {enable_staking}")
        logger.info(f"  DeFi enabled: {enable_defi}")
        logger.info(f"  Arbitrage enabled: {enable_arbitrage}")
        logger.info(f"  Action space: {self.action_size}")
        logger.info(f"  State space: {self.state_size}")

    def _create_default_staking_pools(self) -> List[StakingPool]:
        """Create default staking pools"""
        return [
            StakingPool(asset='BTC', apy=0.05, lock_period=0, min_stake=0.001),
            StakingPool(asset='ETH', apy=0.08, lock_period=0, min_stake=0.01),
            StakingPool(asset='SOL', apy=0.12, lock_period=0, min_stake=1.0),
            StakingPool(asset='USDT', apy=0.10, lock_period=0, min_stake=100.0),
        ]

    def _create_default_liquidity_pools(self) -> List[LiquidityPool]:
        """Create default liquidity pools"""
        return [
            LiquidityPool(pair='BTC/ETH', fee_rate=0.003, apy=0.15, token0='BTC', token1='ETH'),
            LiquidityPool(pair='ETH/USDT', fee_rate=0.003, apy=0.20, token0='ETH', token1='USDT'),
            LiquidityPool(pair='BTC/USDT', fee_rate=0.003, apy=0.18, token0='BTC', token1='USDT'),
        ]

    def _calculate_action_space(self) -> int:
        """Calculate total action space size"""
        # Base actions per pair: BUY, SELL, HOLD
        actions = len(self.trading_pairs) * 3

        # Staking actions: STAKE, UNSTAKE per asset
        if self.enable_staking:
            actions += len(self.staking_pools) * 2

        # DeFi actions: ADD_LIQUIDITY, REMOVE_LIQUIDITY per pool
        if self.enable_defi:
            actions += len(self.liquidity_pools) * 2

        # Arbitrage actions (treated as composite actions)
        if self.enable_arbitrage:
            actions += 10  # Top 10 arbitrage opportunities

        return actions

    def _calculate_state_space(self) -> int:
        """Calculate state space size"""
        # Portfolio state (per pair): balance, position, unrealized PnL
        state = len(self.trading_pairs) * 3

        # Market state (per pair): price, volume, volatility, trend
        state += len(self.trading_pairs) * 4

        # Staking state (per pool): staked amount, rewards, APY
        if self.enable_staking:
            state += len(self.staking_pools) * 3

        # DeFi state (per pool): LP tokens, rewards, APY
        if self.enable_defi:
            state += len(self.liquidity_pools) * 3

        # Arbitrage state: top opportunities
        if self.enable_arbitrage:
            state += 10  # Top 10 arbitrage profit potentials

        # Global portfolio state
        state += 5  # Total equity, portfolio diversity, risk metrics, etc.

        return state

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Initialize portfolio
        self.balance = self.initial_balance
        self.assets: Dict[str, Asset] = {}

        # Initialize assets for each trading pair
        for pair in self.trading_pairs:
            base = pair.split('/')[0]
            if base not in self.assets:
                self.assets[base] = Asset(symbol=base, amount=0.0)

        # Trading state
        self.current_step = 0
        self.total_trades = 0
        self.total_staking_rewards = 0.0
        self.total_lp_fees = 0.0
        self.total_arbitrage_profits = 0.0

        # Performance tracking
        self.equity_curve = [self.initial_balance]
        self.trade_history = []
        self.staking_history = []
        self.defi_history = []
        self.arbitrage_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        state = []

        # Portfolio state for each pair
        for pair in self.trading_pairs:
            base = pair.split('/')[0]
            asset = self.assets.get(base, Asset(symbol=base, amount=0.0))

            # Normalized values
            state.append(asset.amount / 100.0)  # Normalize by reasonable max
            state.append(asset.staked_amount / 100.0)
            state.append(asset.lp_tokens / 100.0)

        # Market state for each pair
        for pair in self.trading_pairs:
            if pair in self.market_data and self.current_step < len(self.market_data[pair]):
                price = self.market_data[pair][self.current_step]

                # Price (normalized)
                state.append(price / 100000.0)

                # Volume (simulated)
                state.append(np.random.rand())

                # Volatility (from recent price changes)
                if self.current_step >= 10:
                    recent_prices = self.market_data[pair][max(0, self.current_step-10):self.current_step]
                    volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
                else:
                    volatility = 0.0
                state.append(min(volatility, 1.0))

                # Trend (simple momentum)
                if self.current_step >= 20:
                    trend = (price - self.market_data[pair][self.current_step-20]) / self.market_data[pair][self.current_step-20]
                else:
                    trend = 0.0
                state.append(np.clip(trend, -1, 1))
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        # Staking state
        if self.enable_staking:
            for pool in self.staking_pools:
                asset = self.assets.get(pool.asset, Asset(symbol=pool.asset, amount=0.0))
                state.append(asset.staked_amount / 100.0)

                # Calculate pending rewards (simplified)
                hourly_rate = pool.apy / (365 * 24)
                rewards = asset.staked_amount * hourly_rate
                state.append(rewards / 10.0)

                state.append(pool.apy)

        # DeFi state
        if self.enable_defi:
            for pool in self.liquidity_pools:
                # Get LP tokens for this pool
                lp_amount = sum(asset.lp_tokens for asset in self.assets.values())
                state.append(lp_amount / 100.0)

                # Estimate rewards
                hourly_fee_rate = pool.apy / (365 * 24)
                rewards = lp_amount * hourly_fee_rate
                state.append(rewards / 10.0)

                state.append(pool.apy)

        # Arbitrage opportunities (simulated)
        if self.enable_arbitrage:
            for i in range(10):
                # Simulate arbitrage profit potential (0-5%)
                profit_potential = np.random.rand() * 0.05 if np.random.rand() > 0.7 else 0.0
                state.append(profit_potential)

        # Global portfolio state
        total_equity = self._calculate_total_equity()
        state.append(total_equity / self.initial_balance)  # Normalized equity
        state.append(self.balance / total_equity if total_equity > 0 else 0.0)  # Cash ratio
        state.append(len([a for a in self.assets.values() if a.amount > 0]) / len(self.trading_pairs))  # Diversity
        state.append(min(self.total_trades / 100.0, 1.0))  # Trading activity
        state.append(min((self.total_staking_rewards + self.total_lp_fees) / 1000.0, 1.0))  # Passive income

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state

        Args:
            action: Action index

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Decode action
        action_type, action_params = self._decode_action(action)

        # Execute action
        reward = 0.0
        info = {'action_type': action_type.name}

        if action_type in [ActionType.BUY, ActionType.SELL, ActionType.HOLD]:
            reward = self._execute_spot_trade(action_type, action_params)

        elif action_type in [ActionType.STAKE, ActionType.UNSTAKE]:
            reward = self._execute_staking_action(action_type, action_params)

        elif action_type in [ActionType.ADD_LIQUIDITY, ActionType.REMOVE_LIQUIDITY]:
            reward = self._execute_defi_action(action_type, action_params)

        elif action_type in [ActionType.ARBITRAGE_BUY, ActionType.ARBITRAGE_SELL]:
            reward = self._execute_arbitrage(action_type, action_params)

        # Accumulate passive rewards (staking, LP fees)
        passive_reward = self._accumulate_passive_rewards()
        reward += passive_reward

        # Update step
        self.current_step += 1

        # Check if done
        total_equity = self._calculate_total_equity()
        done = (
            self.current_step >= self._get_max_steps() or
            total_equity <= self.initial_balance * 0.1  # 90% loss = done
        )

        # Track equity
        self.equity_curve.append(total_equity)

        # Get new state
        next_state = self._get_state()

        info['total_equity'] = total_equity
        info['balance'] = self.balance
        info['passive_income'] = passive_reward

        return next_state, reward, done, info

    def _decode_action(self, action: int) -> Tuple[ActionType, Dict]:
        """Decode action index into action type and parameters"""
        # This is a simplified decoder - in production, you'd have a more sophisticated mapping

        # Spot trading actions
        spot_actions = len(self.trading_pairs) * 3
        if action < spot_actions:
            pair_idx = action // 3
            action_idx = action % 3
            return ActionType(action_idx), {'pair': self.trading_pairs[pair_idx]}

        action -= spot_actions

        # Staking actions
        if self.enable_staking:
            staking_actions = len(self.staking_pools) * 2
            if action < staking_actions:
                pool_idx = action // 2
                is_stake = action % 2 == 0
                return (ActionType.STAKE if is_stake else ActionType.UNSTAKE), {
                    'pool': self.staking_pools[pool_idx]
                }
            action -= staking_actions

        # DeFi actions
        if self.enable_defi:
            defi_actions = len(self.liquidity_pools) * 2
            if action < defi_actions:
                pool_idx = action // 2
                is_add = action % 2 == 0
                return (ActionType.ADD_LIQUIDITY if is_add else ActionType.REMOVE_LIQUIDITY), {
                    'pool': self.liquidity_pools[pool_idx]
                }
            action -= defi_actions

        # Arbitrage actions
        if self.enable_arbitrage:
            return ActionType.ARBITRAGE_BUY, {'opportunity_idx': action}

        # Default to HOLD
        return ActionType.HOLD, {}

    def _execute_spot_trade(self, action_type: ActionType, params: Dict) -> float:
        """Execute spot trade"""
        if action_type == ActionType.HOLD:
            return -0.001  # Small penalty for inaction

        pair = params['pair']
        if pair not in self.market_data or self.current_step >= len(self.market_data[pair]):
            return -0.01

        price = self.market_data[pair][self.current_step]
        base = pair.split('/')[0]

        # Simplified execution (10% of balance/position)
        if action_type == ActionType.BUY:
            amount_to_spend = self.balance * 0.1
            if amount_to_spend > 10:  # Minimum trade size
                cost = amount_to_spend * (1 + self.fee_rate + self.slippage)
                if cost <= self.balance:
                    amount_bought = amount_to_spend / price
                    self.balance -= cost
                    self.assets[base].amount += amount_bought
                    self.total_trades += 1
                    return 0.01  # Small reward for action

        elif action_type == ActionType.SELL:
            asset = self.assets.get(base, Asset(symbol=base, amount=0.0))
            if asset.amount > 0:
                amount_to_sell = asset.amount * 0.1
                revenue = (amount_to_sell * price) * (1 - self.fee_rate - self.slippage)
                asset.amount -= amount_to_sell
                self.balance += revenue
                self.total_trades += 1
                return 0.01  # Small reward for action

        return 0.0

    def _execute_staking_action(self, action_type: ActionType, params: Dict) -> float:
        """Execute staking action"""
        pool = params['pool']
        asset = self.assets.get(pool.asset, Asset(symbol=pool.asset, amount=0.0))

        if action_type == ActionType.STAKE:
            # Stake 20% of available amount
            amount_to_stake = asset.amount * 0.2
            if amount_to_stake >= pool.min_stake:
                asset.amount -= amount_to_stake
                asset.staked_amount += amount_to_stake
                self.staking_history.append({
                    'step': self.current_step,
                    'action': 'stake',
                    'asset': pool.asset,
                    'amount': amount_to_stake
                })
                return 0.02  # Reward for staking

        elif action_type == ActionType.UNSTAKE:
            # Unstake 20% of staked amount
            amount_to_unstake = asset.staked_amount * 0.2
            if amount_to_unstake > 0:
                asset.staked_amount -= amount_to_unstake
                asset.amount += amount_to_unstake
                self.staking_history.append({
                    'step': self.current_step,
                    'action': 'unstake',
                    'asset': pool.asset,
                    'amount': amount_to_unstake
                })
                return 0.01

        return 0.0

    def _execute_defi_action(self, action_type: ActionType, params: Dict) -> float:
        """Execute DeFi liquidity action"""
        pool = params['pool']

        if action_type == ActionType.ADD_LIQUIDITY:
            # Simplified: add liquidity worth 10% of balance
            liquidity_value = self.balance * 0.1
            if liquidity_value > 50:
                self.balance -= liquidity_value
                # Mint LP tokens (simplified)
                lp_tokens = liquidity_value / 100.0  # Arbitrary conversion

                # Distribute to assets (simplified)
                base = pool.token0
                if base in self.assets:
                    self.assets[base].lp_tokens += lp_tokens

                self.defi_history.append({
                    'step': self.current_step,
                    'action': 'add_liquidity',
                    'pool': pool.pair,
                    'value': liquidity_value
                })
                return 0.03  # Reward for providing liquidity

        elif action_type == ActionType.REMOVE_LIQUIDITY:
            # Remove 20% of LP tokens
            total_lp = sum(asset.lp_tokens for asset in self.assets.values())
            if total_lp > 0:
                lp_to_remove = total_lp * 0.2
                value_returned = lp_to_remove * 100.0  # Arbitrary conversion back

                self.balance += value_returned

                # Remove LP tokens
                for asset in self.assets.values():
                    if asset.lp_tokens > 0:
                        asset.lp_tokens = max(0, asset.lp_tokens - lp_to_remove)

                self.defi_history.append({
                    'step': self.current_step,
                    'action': 'remove_liquidity',
                    'pool': pool.pair,
                    'value': value_returned
                })
                return 0.01

        return 0.0

    def _execute_arbitrage(self, action_type: ActionType, params: Dict) -> float:
        """Execute arbitrage opportunity"""
        # Simplified arbitrage execution
        # In reality, this would check real price differences across exchanges

        arbitrage_amount = self.balance * 0.05  # 5% of balance for arbitrage
        if arbitrage_amount > 10:
            # Simulate profit (0-2% profit potential)
            profit_rate = np.random.rand() * 0.02 if np.random.rand() > 0.5 else -0.005
            profit = arbitrage_amount * profit_rate

            self.balance += profit
            self.total_arbitrage_profits += max(0, profit)

            self.arbitrage_history.append({
                'step': self.current_step,
                'profit': profit,
                'amount': arbitrage_amount
            })

            return profit / 100.0  # Normalized reward

        return 0.0

    def _accumulate_passive_rewards(self) -> float:
        """Accumulate staking and LP fee rewards"""
        total_passive = 0.0

        # Staking rewards
        if self.enable_staking:
            for pool in self.staking_pools:
                asset = self.assets.get(pool.asset, None)
                if asset and asset.staked_amount > 0:
                    # Calculate hourly reward
                    hourly_rate = pool.apy / (365 * 24)
                    reward = asset.staked_amount * hourly_rate
                    asset.amount += reward
                    self.total_staking_rewards += reward
                    total_passive += reward

        # LP fee rewards
        if self.enable_defi:
            for pool in self.liquidity_pools:
                total_lp = sum(asset.lp_tokens for asset in self.assets.values())
                if total_lp > 0:
                    # Calculate hourly fee income
                    hourly_rate = pool.apy / (365 * 24)
                    fee_income = total_lp * 100.0 * hourly_rate  # Convert LP tokens to value
                    self.balance += fee_income
                    self.total_lp_fees += fee_income
                    total_passive += fee_income

        return total_passive / 100.0  # Normalized reward

    def _calculate_total_equity(self) -> float:
        """Calculate total portfolio value"""
        equity = self.balance

        # Add value of holdings
        for pair in self.trading_pairs:
            base = pair.split('/')[0]
            asset = self.assets.get(base, None)

            if asset and (asset.amount > 0 or asset.staked_amount > 0):
                if pair in self.market_data and self.current_step < len(self.market_data[pair]):
                    price = self.market_data[pair][self.current_step]
                    equity += (asset.amount + asset.staked_amount) * price

        # Add value of LP tokens (simplified)
        total_lp = sum(asset.lp_tokens for asset in self.assets.values())
        equity += total_lp * 100.0

        return equity

    def _get_max_steps(self) -> int:
        """Get maximum steps based on available market data"""
        if not self.market_data:
            return 1000

        return min(len(data) for data in self.market_data.values())

    def get_episode_stats(self) -> Dict:
        """Get comprehensive episode statistics"""
        total_equity = self._calculate_total_equity()
        total_return = ((total_equity - self.initial_balance) / self.initial_balance) * 100

        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Annualized
        else:
            sharpe = 0.0

        # Calculate max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak * 100
        max_drawdown = np.min(drawdown)

        return {
            'final_equity': total_equity,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'total_staking_rewards': self.total_staking_rewards,
            'total_lp_fees': self.total_lp_fees,
            'total_arbitrage_profits': self.total_arbitrage_profits,
            'total_passive_income': self.total_staking_rewards + self.total_lp_fees,
            'steps': self.current_step
        }


if __name__ == "__main__":
    # Example usage
    print("Nexlify Multi-Strategy Environment")
    print("=" * 60)

    # Create environment with multiple pairs
    trading_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    # Simulate market data
    market_data = {
        'BTC/USDT': np.random.rand(1000) * 50000 + 30000,
        'ETH/USDT': np.random.rand(1000) * 3000 + 1500,
        'SOL/USDT': np.random.rand(1000) * 100 + 50
    }

    env = MultiStrategyEnvironment(
        trading_pairs=trading_pairs,
        market_data=market_data,
        enable_staking=True,
        enable_defi=True,
        enable_arbitrage=True
    )

    print(f"\nAction Space: {env.action_size}")
    print(f"State Space: {env.state_size}")
    print(f"\nSupported Strategies:")
    print(f"  ✓ Spot Trading ({len(trading_pairs)} pairs)")
    print(f"  ✓ Staking ({len(env.staking_pools)} pools)")
    print(f"  ✓ Liquidity Provision ({len(env.liquidity_pools)} pools)")
    print(f"  ✓ Arbitrage Detection (enabled)")

    # Run a short episode
    state = env.reset()
    total_reward = 0

    for step in range(100):
        action = np.random.randint(0, env.action_size)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    stats = env.get_episode_stats()
    print(f"\nEpisode Results:")
    print(f"  Total Return: {stats['total_return_pct']:.2f}%")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Staking Rewards: ${stats['total_staking_rewards']:.2f}")
    print(f"  LP Fees: ${stats['total_lp_fees']:.2f}")
    print(f"  Arbitrage Profits: ${stats['total_arbitrage_profits']:.2f}")
    print(f"  Total Passive Income: ${stats['total_passive_income']:.2f}")
