#!/usr/bin/env python3
"""
Nexlify Reinforcement Learning Module
Deep Q-Network (DQN) agent for autonomous trading optimization
"""

import json
import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nexlify.utils.error_handler import get_error_handler, handle_errors
from nexlify.strategies.epsilon_decay import EpsilonDecayFactory, EpsilonDecayStrategy

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class TradingEnvironment:
    """
    Gym-style trading environment for RL training
    """

    def __init__(self, price_data: np.ndarray, initial_balance: float = 10000):
        self.price_data = price_data
        self.initial_balance = initial_balance

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Amount of crypto held
        self.position_price = 0  # Entry price
        self.max_steps = len(price_data) - 1

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space_n = 3

        # State space: [balance, position, position_price, current_price,
        #               price_change, RSI, MACD, volatility, momentum, vol_clustering,
        #               drawdown, sharpe]
        self.state_space_n = 12  # Expanded for crypto-specific features

        # Tracking
        self.trade_history = []
        self.episode_reward = 0

        # Risk-adjusted reward tracking
        self.returns_history = []
        self.portfolio_values = [initial_balance]
        self.max_portfolio_value = initial_balance

        logger.info("🎮 Trading Environment initialized (Crypto-optimized)")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.trade_history = []
        self.episode_reward = 0

        # Reset risk tracking
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        self.max_portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation with crypto-specific features"""
        current_price = self.price_data[self.current_step]

        # Calculate features
        price_change = 0
        if self.current_step > 0:
            price_change = (
                current_price - self.price_data[self.current_step - 1]
            ) / current_price

        # Technical indicators
        rsi = self._calculate_rsi(14)
        macd = self._calculate_macd()
        volatility = self._calculate_volatility(10)

        # Crypto-specific features
        momentum = self._calculate_momentum(20)  # 20-period momentum
        vol_clustering = self._calculate_volatility_clustering()  # GARCH-like
        drawdown = self._calculate_drawdown()  # Current drawdown from peak
        sharpe = self._calculate_sharpe_ratio()  # Risk-adjusted returns

        state = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Position size
                (
                    self.position_price / current_price if current_price > 0 else 0
                ),  # Relative entry price
                current_price / self.initial_balance,  # Normalized price
                price_change,  # Price change
                rsi,  # RSI
                macd,  # MACD
                volatility,  # Volatility
                momentum,  # Momentum (NEW)
                vol_clustering,  # Volatility clustering (NEW)
                drawdown,  # Drawdown from peak (NEW)
                sharpe,  # Sharpe ratio (NEW)
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if self.current_step < period:
            return 0.5

        prices = self.price_data[
            max(0, self.current_step - period) : self.current_step + 1
        ]
        deltas = np.diff(prices)

        gains = deltas[deltas > 0].sum()
        losses = abs(deltas[deltas < 0].sum())

        if losses == 0:
            return 1.0

        rs = gains / losses
        rsi = 1 - (1 / (1 + rs))

        return rsi

    def _calculate_macd(self) -> float:
        """Calculate MACD indicator"""
        if self.current_step < 26:
            return 0

        prices = self.price_data[max(0, self.current_step - 26) : self.current_step + 1]

        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)

        macd = (ema_12 - ema_26) / ema_26 if ema_26 > 0 else 0

        return macd

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[-period]

        for price in prices[-period + 1 :]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_volatility(self, period: int = 10) -> float:
        """Calculate price volatility"""
        if self.current_step < period:
            return 0

        prices = self.price_data[
            max(0, self.current_step - period) : self.current_step + 1
        ]
        returns = np.diff(prices) / prices[:-1]

        return np.std(returns)

    def _calculate_momentum(self, period: int = 20) -> float:
        """
        Calculate price momentum (rate of change)
        Crypto markets show strong momentum effects
        """
        if self.current_step < period:
            return 0

        current_price = self.price_data[self.current_step]
        past_price = self.price_data[max(0, self.current_step - period)]

        if past_price == 0:
            return 0

        momentum = (current_price - past_price) / past_price
        return np.clip(momentum, -1, 1)  # Clip to [-1, 1] range

    def _calculate_volatility_clustering(self) -> float:
        """
        Calculate volatility clustering (GARCH-like effect)
        Crypto exhibits strong volatility clustering - high vol follows high vol
        """
        if self.current_step < 30:
            return 0

        # Recent volatility (10 periods)
        recent_vol = self._calculate_volatility(10)

        # Historical volatility (30 periods)
        if self.current_step < 30:
            return 0

        prices = self.price_data[max(0, self.current_step - 30): self.current_step + 1]
        returns = np.diff(prices) / prices[:-1]
        historical_vol = np.std(returns)

        if historical_vol == 0:
            return 0

        # Ratio of recent to historical volatility
        vol_ratio = recent_vol / historical_vol
        return np.clip(vol_ratio - 1, -1, 1)  # Centered at 0, clipped to [-1, 1]

    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown from peak portfolio value
        Important risk metric for crypto trading
        """
        current_value = self.balance
        if self.position > 0:
            current_price = self.price_data[self.current_step]
            current_value += self.position * current_price

        # Update max value
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)

        if self.max_portfolio_value == 0:
            return 0

        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        return np.clip(drawdown, 0, 1)

    def _calculate_sharpe_ratio(self, window: int = 50) -> float:
        """
        Calculate rolling Sharpe ratio
        Risk-adjusted return metric crucial for crypto
        Assumes hourly data (8760 periods/year) for annualization
        """
        if len(self.returns_history) < 10:
            return 0

        recent_returns = self.returns_history[-window:] if len(self.returns_history) >= window else self.returns_history

        if len(recent_returns) < 2:
            return 0

        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        if std_return == 0:
            return 0

        # Annualized Sharpe (crypto trades 24/7, assuming hourly data = 8760 periods/year)
        # This is a reasonable default for crypto trading environments
        periods_per_year = 8760  # 365 * 24 hours
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return np.clip(sharpe, -3, 3)  # Clip to reasonable range

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.price_data[self.current_step]
        reward = 0
        info = {}

        # Execute action with proper fee accounting
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:
                # Buy with 95% of balance, accounting for fees
                # Assume 0.1% fee rate (typical for crypto)
                fee_rate = 0.001
                amount_to_invest = self.balance * 0.95  # Leave room for fees
                cost = amount_to_invest / (1 + fee_rate)  # Actual amount we can buy
                fees = cost * fee_rate

                self.position = cost / current_price
                self.position_price = current_price
                self.balance -= (cost + fees)  # Deduct cost and fees

                info["action"] = "buy"
                info["price"] = current_price
                info["fees"] = fees

        elif action == 2:  # Sell
            if self.position > 0:
                # Sell entire position with fee accounting
                fee_rate = 0.001
                sell_value = self.position * current_price
                fees = sell_value * fee_rate
                net_proceeds = sell_value - fees

                # Calculate profit after fees
                cost = self.position * self.position_price
                profit = net_proceeds - cost

                self.balance = net_proceeds

                # Reward based on RETURN not absolute profit
                reward = profit / cost if cost > 0 else 0  # Return on investment

                # Track trade
                self.trade_history.append(
                    {
                        "entry_price": self.position_price,
                        "exit_price": current_price,
                        "profit": profit,
                        "return": (current_price / self.position_price - 1) * 100,
                        "fees": fees,
                    }
                )

                self.position = 0
                self.position_price = 0
                info["action"] = "sell"
                info["profit"] = profit
                info["fees"] = fees
        else:
            # Hold - no fixed penalty, let portfolio value change drive reward
            info["action"] = "hold"
            reward = 0  # Neutral, will be adjusted by equity change below

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Calculate total portfolio value
        total_value = self.balance
        if self.position > 0:
            total_value += self.position * current_price

        # Track portfolio value and returns for risk-adjusted rewards
        self.portfolio_values.append(total_value)

        # PRIMARY REWARD: Equity change (consistent with improved reward function)
        if len(self.portfolio_values) > 1:
            equity_change = total_value - self.portfolio_values[-2]
            equity_return = equity_change / self.portfolio_values[-2]
            self.returns_history.append(equity_return)

            # Only override reward if not from sell (sell already has ROI reward)
            if action != 2:  # Not sell
                reward = equity_return * 100.0  # Scale to percentage

        # Apply risk adjustment to reward (Sharpe-based) - FIXED VERSION
        # Apply as ADDITIVE bonus/penalty, not multiplicative
        if len(self.returns_history) >= 10:
            sharpe = self._calculate_sharpe_ratio()
            # Sharpe bonus/penalty (additive to avoid breaking negative rewards)
            # Clip Sharpe-based adjustment to reasonable range
            risk_adjustment = np.clip(sharpe * 0.01, -0.1, 0.1)  # ±0.1 max
            reward += risk_adjustment  # ADDITIVE, not multiplicative

            info["sharpe_ratio"] = sharpe
            info["risk_adjustment"] = risk_adjustment

        info["total_value"] = total_value
        info["equity_return"] = equity_return if len(self.portfolio_values) > 1 else 0
        info["step"] = self.current_step

        self.episode_reward += reward

        next_state = self._get_state() if not done else np.zeros(self.state_space_n)

        return next_state, reward, done, info

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        current_price = (
            self.price_data[self.current_step]
            if self.current_step < len(self.price_data)
            else 0
        )
        return self.balance + (self.position * current_price)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = None, max_size: int = None):
        # Support both capacity and max_size for backward compatibility
        size = capacity or max_size or 100000
        self.buffer = deque(maxlen=size)
        self.max_size = size  # Expose max_size for tests

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def add(self, state, action, reward, next_state, done):
        """Alias for push() for backward compatibility with tests"""
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> List:
        """Sample random batch from buffer"""
        # Handle case where batch_size > buffer size
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading
    Uses neural network to approximate Q-values
    """

    def __init__(self, state_size: int, action_size: int, config: Dict = None, **kwargs):
        # Validate inputs
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")

        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Merge kwargs into config for backward compatibility with tests
        self.config.update(kwargs)

        # Hyperparameters (optimized for 24/7 crypto trading)
        self.gamma = self.config.get("gamma", 0.89)  # Lower discount for fast crypto markets
        self.learning_rate = self.config.get("learning_rate", 0.0015)  # Aggressive learning
        self.learning_rate_decay = self.config.get("learning_rate_decay", 0.9998)
        self.batch_size = self.config.get("batch_size", 64)
        self.target_update_freq = self.config.get("target_update_freq", 1200)  # Frequent updates

        # Epsilon decay strategy (new advanced system)
        self.epsilon_decay_strategy = self._create_epsilon_strategy()
        self.epsilon = self.epsilon_decay_strategy.current_epsilon

        # Experience replay (optimized for regime adaptation)
        replay_buffer_size = self.config.get("replay_buffer_size", 60000)
        self.memory = ReplayBuffer(capacity=replay_buffer_size)

        # Neural network models
        self.model = None
        self.target_model = None

        # Try to use GPU if available
        self.device = self._get_device()

        # Build models
        self._build_model()

        # Training stats
        self.training_history = []

        logger.info(f"🤖 DQN Agent initialized (device: {self.device})")

    def _create_epsilon_strategy(self) -> EpsilonDecayStrategy:
        """Create epsilon decay strategy from config"""
        # Check if using new config format
        if "epsilon_decay_type" in self.config or "epsilon_decay_steps" in self.config:
            return EpsilonDecayFactory.create_from_config(self.config)

        # Legacy config support - convert old parameters to new system
        epsilon_start = self.config.get("epsilon", 1.0)
        epsilon_end = self.config.get("epsilon_min", 0.22)  # Crypto-optimized default (24/7 trading)

        # If old-style epsilon_decay (multiplicative) is provided, use exponential decay
        if "epsilon_decay" in self.config and "epsilon_decay_steps" not in self.config:
            decay_rate = self.config.get("epsilon_decay", 0.995)
            logger.info(f"🔄 Converting legacy epsilon_decay={decay_rate} to ExponentialEpsilonDecay")

            return EpsilonDecayFactory.create(
                "exponential",
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                decay_rate=decay_rate,
                decay_steps=10000,  # Large value for exponential decay
            )

        # Default: Use scheduled decay optimized for 24/7 crypto trading
        # Aggressive schedule for continuous trading and regime adaptation
        default_schedule = {
            0: 1.0,      # Start with full exploration
            200: 0.65,   # Learn basics quickly (24/7 = ~8 days)
            800: 0.35,   # Start exploiting patterns (~1 month)
            2000: 0.22,  # High ongoing exploration for regime changes (~2.5 months)
        }
        schedule = self.config.get("epsilon_schedule", default_schedule)

        logger.info("🚀 Using ScheduledEpsilonDecay optimized for 24/7 crypto trading")

        return EpsilonDecayFactory.create(
            "scheduled",
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            schedule=schedule,
        )

    def _get_device(self) -> str:
        """Detect available compute device"""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _build_model(self):
        """Build neural network model"""
        try:
            import torch
            import torch.nn as nn

            class DQNNetwork(nn.Module):
                def __init__(self, state_size, action_size):
                    super(DQNNetwork, self).__init__()
                    self.fc1 = nn.Linear(state_size, 128)
                    self.fc2 = nn.Linear(128, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, action_size)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.relu(self.fc3(x))
                    x = self.fc4(x)
                    return x

            self.model = DQNNetwork(self.state_size, self.action_size)
            self.target_model = DQNNetwork(self.state_size, self.action_size)

            if self.device == "cuda":
                self.model = self.model.cuda()
                self.target_model = self.target_model.cuda()

            self.update_target_model()

            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
            self.criterion = nn.MSELoss()

            logger.info("✅ PyTorch DQN model built")

        except ImportError:
            logger.warning("PyTorch not available, using NumPy fallback")
            self._build_numpy_model()

    def _build_numpy_model(self):
        """Fallback: Simple NumPy-based Q-table approximation"""
        # Initialize random weights for simple linear model
        self.weights = np.random.randn(self.state_size, self.action_size) * 0.01
        logger.info("✅ NumPy fallback model built")

    def update_target_model(self):
        """Copy weights from model to target model"""
        try:
            if hasattr(self, "model") and self.model is not None:
                self.target_model.load_state_dict(self.model.state_dict())
        except:
            pass

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)

        # Exploit: best action from Q-values
        try:
            import torch

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if self.device == "cuda":
                state_tensor = state_tensor.cuda()

            with torch.no_grad():
                q_values = self.model(state_tensor)

            return q_values.argmax().item()

        except:
            # NumPy fallback
            q_values = np.dot(state, self.weights)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size: int = None):
        """Train on batch from replay buffer"""
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return 0.0  # Return 0 loss when insufficient data (test compatibility)

        try:
            import torch

            # Sample batch
            batch = self.memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            if self.device == "cuda":
                states = states.cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                next_states = next_states.cuda()
                dones = dones.cuda()

            # Current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss
            loss = self.criterion(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        except Exception as e:
            logger.error(f"Replay error: {e}")
            return None

    def decay_epsilon(self):
        """Decay exploration rate using advanced strategy"""
        self.epsilon = self.epsilon_decay_strategy.step()

    def update_epsilon(self):
        """Alias for decay_epsilon() for backward compatibility"""
        self.decay_epsilon()

    def save(self, filepath: str):
        """Save model to file"""
        try:
            import torch

            save_data = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "epsilon_step": self.epsilon_decay_strategy.current_step,
                "training_history": self.training_history,
            }

            torch.save(save_data, filepath)
            logger.info(f"✅ Model saved to {filepath}")

            # Save epsilon history separately
            epsilon_path = (
                Path(filepath).parent / f"{Path(filepath).stem}_epsilon_history.json"
            )
            self.epsilon_decay_strategy.save_history(str(epsilon_path))

        except Exception as e:
            logger.error(f"Save error: {e}")

    def load(self, filepath: str):
        """Load model from file"""
        try:
            import torch

            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.training_history = checkpoint.get("training_history", [])

            # Restore epsilon decay strategy step
            if "epsilon_step" in checkpoint:
                self.epsilon_decay_strategy.current_step = checkpoint["epsilon_step"]
                self.epsilon_decay_strategy.current_epsilon = self.epsilon

            self.update_target_model()

            logger.info(f"✅ Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Load error: {e}")

    def get_model_summary(self) -> str:
        """Get summary of model architecture and parameters"""
        try:
            import torch

            if self.model is None:
                param_count = self.weights.size if hasattr(self, "weights") else 0
                return (
                    f"DQN Agent Model Summary\n"
                    f"{'='*50}\n"
                    f"Type: NumPy Fallback\n"
                    f"State Size: {self.state_size}\n"
                    f"Action Size: {self.action_size}\n"
                    f"Parameters: {param_count}\n"
                )

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            return (
                f"DQN Agent Model Summary\n"
                f"{'='*50}\n"
                f"Type: PyTorch DQN\n"
                f"State Size: {self.state_size}\n"
                f"Action Size: {self.action_size}\n"
                f"Total Parameters: {total_params:,}\n"
                f"Trainable Parameters: {trainable_params:,}\n"
                f"Device: {self.device}\n"
                f"Epsilon: {self.epsilon:.4f}\n"
                f"Learning Rate: {self.learning_rate}\n"
                f"Gamma: {self.gamma}\n"
            )

        except Exception as e:
            logger.error(f"Model summary error: {e}")
            return (
                f"DQN Agent Model Summary (Error)\n"
                f"{'='*50}\n"
                f"State Size: {self.state_size}\n"
                f"Action Size: {self.action_size}\n"
                f"Error: {str(e)}\n"
            )


# Export main classes
__all__ = ["TradingEnvironment", "DQNAgent", "ReplayBuffer"]
