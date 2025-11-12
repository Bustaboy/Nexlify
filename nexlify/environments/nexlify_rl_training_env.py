#!/usr/bin/env python3
"""
Nexlify RL Training Environment

Gym-style environment for training RL agents with paper trading integration.
Supports both simulated and live market data for continuous learning.

Features:
- OpenAI Gym-compatible interface
- Paper trading integration for risk-free training
- Real market data support
- Comprehensive reward function
- State space with technical indicators
- Episode management and performance tracking
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
from nexlify.ml.nexlify_feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Statistics for a single training episode"""
    episode_num: int
    total_reward: float
    final_equity: float
    total_return: float
    total_return_percent: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    steps: int


class TradingEnvironment:
    """
    Gym-style trading environment with paper trading integration

    This environment provides a standard interface for training RL agents
    with realistic trading conditions including fees, slippage, and risk management.
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 fee_rate: float = 0.001,
                 slippage: float = 0.0005,
                 state_size: int = 8,
                 action_size: int = 3,
                 max_steps: int = 1000,
                 use_paper_trading: bool = True,
                 market_data: Optional[pd.DataFrame] = None,
                 engineer_features: bool = False):
        """
        Initialize trading environment

        Args:
            initial_balance: Starting balance
            fee_rate: Trading fee rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            state_size: Size of state vector
            action_size: Number of possible actions
            max_steps: Maximum steps per episode
            use_paper_trading: Whether to use paper trading engine
            market_data: Historical market data for training
            engineer_features: Whether to engineer features from market data
        """
        # Environment config
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps

        # Paper trading integration
        self.use_paper_trading = use_paper_trading
        if use_paper_trading:
            self.paper_engine = PaperTradingEngine({
                'paper_balance': initial_balance,
                'fee_rate': fee_rate,
                'slippage': slippage
            })
        else:
            self.paper_engine = None

        # Market data
        self.market_data = market_data
        self.current_step = 0
        self.current_price = 0.0
        self.price_history = []

        # Feature engineering
        self.engineer_features = engineer_features
        if engineer_features:
            self.feature_engineer = FeatureEngineer(enable_sentiment=False)
        else:
            self.feature_engineer = None

        # State tracking
        self.balance = initial_balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0  # Entry price for current position
        self.equity_curve = [initial_balance]

        # Episode tracking
        self.episode = 0
        self.episode_history: List[EpisodeStats] = []
        self.total_trades = 0
        self.winning_trades = 0

        # Action space: 0=Buy, 1=Sell, 2=Hold
        self.ACTION_BUY = 0
        self.ACTION_SELL = 1
        self.ACTION_HOLD = 2

        logger.info(f"🎮 Trading Environment initialized")
        logger.info(f"   Balance: ${initial_balance:,.2f}")
        logger.info(f"   State size: {state_size}, Action size: {action_size}")
        logger.info(f"   Paper trading: {'enabled' if use_paper_trading else 'disabled'}")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial state vector
        """
        # Reset paper trading engine
        if self.use_paper_trading:
            self.paper_engine = PaperTradingEngine({
                'paper_balance': self.initial_balance,
                'fee_rate': self.fee_rate,
                'slippage': self.slippage
            })

        # Reset state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.equity_curve = [self.initial_balance]
        self.price_history = []

        # Reset episode tracking
        self.total_trades = 0
        self.winning_trades = 0

        # Generate or load initial price
        if self.market_data is not None and len(self.market_data) > 0:
            self.current_price = self.market_data.iloc[0]['close']
        else:
            self.current_price = np.random.uniform(30000, 50000)  # Random BTC price

        self.price_history.append(self.current_price)

        self.episode += 1
        logger.debug(f"Episode {self.episode} started")

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state

        Args:
            action: Action to take (0=Buy, 1=Sell, 2=Hold)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute action
        trade_executed = self._execute_action(action)

        # Update market (get next price)
        self._update_market()

        # Calculate reward
        reward = self._calculate_reward(action, trade_executed)

        # Update equity
        current_equity = self._get_current_equity()
        self.equity_curve.append(current_equity)

        # Check if episode is done
        done = self._is_done()

        # Prepare info dict
        info = {
            'step': self.current_step,
            'price': self.current_price,
            'balance': self.balance,
            'position': self.position,
            'equity': current_equity,
            'total_return': current_equity - self.initial_balance,
            'total_return_percent': ((current_equity - self.initial_balance) / self.initial_balance) * 100,
            'trade_executed': trade_executed
        }

        # If episode done, record stats
        if done:
            self._record_episode_stats(current_equity)

        return self._get_state(), reward, done, info

    def _execute_action(self, action: int) -> bool:
        """
        Execute trading action

        Args:
            action: Action index

        Returns:
            True if trade was executed, False otherwise
        """
        if action == self.ACTION_BUY:
            return self._execute_buy()
        elif action == self.ACTION_SELL:
            return self._execute_sell()
        else:  # HOLD
            return False

    def _execute_buy(self) -> bool:
        """Execute buy action"""
        # Check if already in position
        if self.position > 0:
            return False

        # Calculate position size (use 95% of balance to leave room for fees)
        max_cost = self.balance * 0.95
        amount = max_cost / (self.current_price * (1 + self.fee_rate + self.slippage))

        if amount <= 0:
            return False

        # Execute in paper trading engine if enabled
        if self.use_paper_trading:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.paper_engine.place_order(
                    'BTC/USDT', 'buy', amount, self.current_price, strategy='rl_training'
                )
            )

            if not result.get('success'):
                return False

            # Update local state from paper engine
            self.balance = self.paper_engine.current_balance
            self.position = amount
            self.entry_price = self.current_price
        else:
            # Manual execution
            cost = amount * self.current_price
            fees = cost * self.fee_rate
            total_cost = cost + fees

            if total_cost > self.balance:
                return False

            self.balance -= total_cost
            self.position = amount
            self.entry_price = self.current_price

        self.total_trades += 1
        logger.debug(f"BUY executed: {amount:.4f} @ ${self.current_price:.2f}")
        return True

    def _execute_sell(self) -> bool:
        """Execute sell action"""
        # Check if in position
        if self.position <= 0:
            return False

        amount = self.position

        # Execute in paper trading engine if enabled
        if self.use_paper_trading:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.paper_engine.place_order(
                    'BTC/USDT', 'sell', amount, self.current_price, strategy='rl_training'
                )
            )

            if not result.get('success'):
                return False

            # Check if profitable
            pnl = result.get('pnl', 0)
            if pnl > 0:
                self.winning_trades += 1

            # Update local state from paper engine
            self.balance = self.paper_engine.current_balance
            self.position = 0
            self.entry_price = 0
        else:
            # Manual execution
            proceeds = amount * self.current_price
            fees = proceeds * self.fee_rate
            net_proceeds = proceeds - fees

            # Check if profitable
            cost = amount * self.entry_price
            if net_proceeds > cost:
                self.winning_trades += 1

            self.balance += net_proceeds
            self.position = 0
            self.entry_price = 0

        self.total_trades += 1
        logger.debug(f"SELL executed: {amount:.4f} @ ${self.current_price:.2f}")
        return True

    def _update_market(self):
        """Update market with next price"""
        self.current_step += 1

        if self.market_data is not None and self.current_step < len(self.market_data):
            # Use historical data
            self.current_price = self.market_data.iloc[self.current_step]['close']
        else:
            # Simulate price movement (random walk with slight upward bias)
            change_percent = np.random.normal(0.0001, 0.02)  # 0.01% mean, 2% std
            self.current_price *= (1 + change_percent)

        self.price_history.append(self.current_price)

        # Update paper trading positions if enabled
        if self.use_paper_trading and self.position > 0:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                self.paper_engine.update_positions({'BTC/USDT': self.current_price})
            )

    def _calculate_reward(self, action: int, trade_executed: bool) -> float:
        """
        Calculate reward for current step

        Reward function balances:
        - Profit/loss from trades
        - Holding cost
        - Transaction costs
        - Risk-adjusted returns

        Args:
            action: Action taken
            trade_executed: Whether trade was actually executed

        Returns:
            Reward value
        """
        reward = 0.0

        # Get current equity
        current_equity = self._get_current_equity()

        # Equity change reward (normalized)
        if len(self.equity_curve) > 1:
            equity_change = current_equity - self.equity_curve[-1]
            reward += equity_change / self.initial_balance * 100  # Scale to percentage

        # Penalize transaction costs
        if trade_executed:
            transaction_cost = self.fee_rate + self.slippage
            reward -= transaction_cost * 10  # Penalty for trading

        # Reward for holding profitable positions
        if self.position > 0 and self.current_price > self.entry_price:
            unrealized_gain = (self.current_price - self.entry_price) / self.entry_price
            reward += unrealized_gain * 5  # Reward unrealized gains

        # Penalize holding losing positions
        if self.position > 0 and self.current_price < self.entry_price:
            unrealized_loss = (self.entry_price - self.current_price) / self.entry_price
            reward -= unrealized_loss * 3  # Smaller penalty to encourage holding through dips

        # Small penalty for doing nothing when not in position
        if action == self.ACTION_HOLD and self.position == 0:
            reward -= 0.01

        return reward

    def _get_current_equity(self) -> float:
        """Calculate current total equity"""
        if self.use_paper_trading:
            return self.paper_engine.get_total_equity({'BTC/USDT': self.current_price})
        else:
            return self.balance + (self.position * self.current_price)

    def _is_done(self) -> bool:
        """Check if episode is complete"""
        # Max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Out of market data
        if self.market_data is not None and self.current_step >= len(self.market_data) - 1:
            return True

        # Catastrophic loss (lost 90% of capital)
        if self._get_current_equity() < self.initial_balance * 0.1:
            logger.warning(f"Episode {self.episode} terminated: catastrophic loss")
            return True

        return False

    def _get_state(self) -> np.ndarray:
        """
        Get current state vector

        Returns:
            State vector with normalized features
        """
        current_equity = self._get_current_equity()

        # Calculate technical indicators from price history
        rsi = self._calculate_rsi()
        macd = self._calculate_macd()
        volume_ratio = 1.0  # Placeholder

        # Calculate price change
        if len(self.price_history) > 1:
            price_change = (self.current_price - self.price_history[-2]) / self.price_history[-2]
        else:
            price_change = 0.0

        # Build state vector (8 features)
        state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position / (self.initial_balance / self.current_price),  # Normalized position
            (self.entry_price / self.current_price) if self.entry_price > 0 else 1.0,  # Relative entry price
            self.current_price / self.initial_balance,  # Normalized price
            price_change,  # Price change
            rsi / 100,  # Normalized RSI
            macd,  # MACD (already normalized)
            volume_ratio  # Volume ratio
        ])

        return state.astype(np.float32)

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI from price history"""
        if len(self.price_history) < period + 1:
            return 50.0  # Neutral RSI

        prices = np.array(self.price_history[-(period + 1):])
        deltas = np.diff(prices)

        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self) -> float:
        """Calculate MACD from price history"""
        if len(self.price_history) < 26:
            return 0.0

        prices = np.array(self.price_history[-26:])

        # Simple implementation (not EMA, but fast)
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices)

        macd = (ema_12 - ema_26) / ema_26  # Normalized

        return macd

    def _record_episode_stats(self, final_equity: float):
        """Record statistics for completed episode"""
        # Calculate metrics
        total_return = final_equity - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        stats = EpisodeStats(
            episode_num=self.episode,
            total_reward=sum(self.equity_curve) - len(self.equity_curve) * self.initial_balance,
            final_equity=final_equity,
            total_return=total_return,
            total_return_percent=total_return_percent,
            num_trades=self.total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            steps=self.current_step
        )

        self.episode_history.append(stats)

        logger.info(f"Episode {self.episode} completed: "
                   f"Return={total_return_percent:.2f}%, "
                   f"Trades={self.total_trades}, "
                   f"Win Rate={win_rate:.1f}%")

    def get_episode_stats(self) -> List[EpisodeStats]:
        """Get statistics for all episodes"""
        return self.episode_history

    def render(self, mode: str = 'human'):
        """
        Render environment state

        Args:
            mode: Render mode ('human' for console output)
        """
        if mode == 'human':
            current_equity = self._get_current_equity()
            print(f"\nStep: {self.current_step}/{self.max_steps}")
            print(f"Price: ${self.current_price:,.2f}")
            print(f"Balance: ${self.balance:,.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Equity: ${current_equity:,.2f}")
            print(f"Return: {((current_equity - self.initial_balance) / self.initial_balance * 100):.2f}%")


# Convenience function
def create_training_environment(
    initial_balance: float = 10000.0,
    use_paper_trading: bool = True,
    market_data: Optional[pd.DataFrame] = None
) -> TradingEnvironment:
    """
    Create trading environment with default settings

    Args:
        initial_balance: Starting balance
        use_paper_trading: Whether to use paper trading engine
        market_data: Historical market data

    Returns:
        Initialized TradingEnvironment
    """
    return TradingEnvironment(
        initial_balance=initial_balance,
        use_paper_trading=use_paper_trading,
        market_data=market_data
    )


__all__ = [
    'TradingEnvironment',
    'EpisodeStats',
    'create_training_environment'
]
