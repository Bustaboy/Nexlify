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
        #               price_change, RSI, MACD, volume_ratio]
        self.state_space_n = 8

        # Tracking
        self.trade_history = []
        self.episode_reward = 0

        logger.info("🎮 Trading Environment initialized")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.trade_history = []
        self.episode_reward = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        current_price = self.price_data[self.current_step]

        # Calculate features
        price_change = 0
        if self.current_step > 0:
            price_change = (
                current_price - self.price_data[self.current_step - 1]
            ) / current_price

        # Simple RSI calculation (14 period)
        rsi = self._calculate_rsi(14)

        # Simple MACD
        macd = self._calculate_macd()

        # Volume ratio (simplified - using price volatility as proxy)
        volume_ratio = self._calculate_volatility(10)

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
                volume_ratio,  # Volume ratio
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

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:
                # Buy with 100% of balance (simplified)
                self.position = self.balance / current_price
                self.position_price = current_price
                self.balance = 0
                info["action"] = "buy"
                info["price"] = current_price

        elif action == 2:  # Sell
            if self.position > 0:
                # Sell entire position
                sell_value = self.position * current_price
                profit = sell_value - (self.position * self.position_price)

                self.balance = sell_value
                reward = profit / self.initial_balance  # Normalized reward

                # Track trade
                self.trade_history.append(
                    {
                        "entry_price": self.position_price,
                        "exit_price": current_price,
                        "profit": profit,
                        "return": (current_price / self.position_price - 1) * 100,
                    }
                )

                self.position = 0
                self.position_price = 0
                info["action"] = "sell"
                info["profit"] = profit
        else:
            # Hold
            info["action"] = "hold"
            # Small penalty for holding to encourage action
            reward = -0.001

        # Calculate unrealized PnL if holding position
        if self.position > 0:
            unrealized_pnl = (current_price - self.position_price) * self.position
            reward += (
                unrealized_pnl / self.initial_balance * 0.1
            )  # Small reward for good positions

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Calculate total portfolio value
        total_value = self.balance
        if self.position > 0:
            total_value += self.position * current_price

        info["total_value"] = total_value
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

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading
    Uses neural network to approximate Q-values
    """

    def __init__(self, state_size: int, action_size: int, config: Dict = None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Hyperparameters
        self.gamma = self.config.get("gamma", 0.99)  # Discount factor
        self.epsilon = self.config.get("epsilon", 1.0)  # Exploration rate
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.batch_size = self.config.get("batch_size", 64)

        # Experience replay
        self.memory = ReplayBuffer(capacity=100000)

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

    def replay(self):
        """Train on batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        try:
            import torch

            # Sample batch
            batch = self.memory.sample(self.batch_size)
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
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save model to file"""
        try:
            import torch

            save_data = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_history": self.training_history,
            }

            torch.save(save_data, filepath)
            logger.info(f"✅ Model saved to {filepath}")

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

            self.update_target_model()

            logger.info(f"✅ Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Load error: {e}")


# Export main classes
__all__ = ["TradingEnvironment", "DQNAgent", "ReplayBuffer"]
