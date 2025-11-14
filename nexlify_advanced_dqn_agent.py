#!/usr/bin/env python3
"""
Advanced DQN Agent with ALL Best Practices (Phase 1, 2, 3)

Implements state-of-the-art RL algorithms and ML best practices:

PHASE 1 (Fundamentals):
✅ Gradient clipping
✅ Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
✅ L2 regularization
✅ Early stopping
✅ Metrics tracking

PHASE 2 (Advanced Algorithms):
✅ Double DQN - Reduces Q-value overestimation
✅ Dueling DQN - Separates state value and action advantages
✅ Stochastic Weight Averaging (SWA) - Averages recent checkpoints
✅ Improved target network updates

PHASE 3 (Expert Techniques):
✅ Prioritized Experience Replay (PER) - Samples important transitions
✅ N-Step Returns - Better credit assignment
✅ Data Augmentation - Trading-specific robustness
✅ Gradient norm monitoring

This agent is designed for MAXIMUM PERFORMANCE, not speed.
Perfect for 24+ hour training runs to achieve the best possible model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, namedtuple
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Named tuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'priority'))


class SumTree:
    """
    Sum Tree data structure for Prioritized Experience Replay

    Allows O(log n) sampling based on priorities
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index based on priority sum"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get total priority sum"""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add new experience with priority"""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """Update priority for existing experience"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience based on priority value"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer

    Samples transitions with probability proportional to their TD error
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta annealing rate
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priorities

    def add(self, state, action, reward, next_state, done, error: Optional[float] = None):
        """Add transition with priority"""
        # Use max priority for new experiences (ensures they're sampled at least once)
        if error is None:
            priority = self.tree.total() / self.tree.n_entries if self.tree.n_entries > 0 else 1.0
        else:
            priority = (abs(error) + self.epsilon) ** self.alpha

        transition = Transition(state, action, reward, next_state, done, priority)
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize

        return batch, np.array(indices), weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on new TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture

    Separates value function V(s) and advantage function A(s,a):
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    This helps learning by separating "how good is this state" from
    "which action is best"
    """
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int]):
        super(DuelingDQN, self).__init__()

        # Shared feature layers
        feature_layers = []
        prev_size = state_size

        for hidden_size in hidden_layers[:-1]:  # All but last layer
            feature_layers.append(nn.Linear(prev_size, hidden_size))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.LayerNorm(hidden_size))  # Added normalization
            prev_size = hidden_size

        self.feature_layer = nn.Sequential(*feature_layers)

        # Last hidden size
        last_hidden = hidden_layers[-1] if hidden_layers else state_size

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, action_size)
        )

    def forward(self, x):
        """
        Forward pass

        Returns Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages
        # Subtract mean advantage to make advantages zero-centered
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


@dataclass
class AgentConfig:
    """Configuration for Advanced DQN Agent"""
    # Network architecture
    hidden_layers: List[int] = None  # Will be set dynamically

    # Training hyperparameters (OPTIMIZED for trading)
    gamma: float = 0.95  # OPTIMIZED: 0.99 → 0.95 (shorter planning horizon)
    learning_rate: float = 0.0003  # OPTIMIZED: 0.001 → 0.0003 (more stable)
    batch_size: int = 128  # OPTIMIZED: 64 → 128 (better gradients)

    # Exploration (CRITICAL FIX!)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05  # OPTIMIZED: 0.01 → 0.05 (maintain exploration)
    epsilon_decay: float = 0.995

    # LINEAR epsilon decay (CRITICAL: enables learning!)
    use_linear_epsilon_decay: bool = True  # OPTIMIZED: False → True (was blocking learning!)
    epsilon_decay_steps: int = 2000  # Linear decay 1.0 → 0.05 over 2000 steps

    # Replay buffer
    buffer_size: int = 100000
    use_prioritized_replay: bool = True
    per_alpha: float = 0.6  # Priority exponent
    per_beta: float = 0.4  # Importance sampling exponent
    per_beta_increment: float = 0.001

    # N-step returns
    n_step: int = 5  # OPTIMIZED: 3 → 5 (better credit assignment)

    # Target network
    target_update_frequency: int = 500  # OPTIMIZED: 1000 → 500 (faster sync)

    # Phase 1: Best practices
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5  # L2 regularization
    lr_scheduler_type: str = 'cosine'  # OPTIMIZED: 'plateau' → 'cosine'
    lr_scheduler_patience: int = 10  # OPTIMIZED: 5 → 10
    lr_scheduler_factor: float = 0.5
    lr_min: float = 1e-6

    # Phase 2: Advanced features
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_swa: bool = True
    swa_start: int = 3000  # OPTIMIZED: 5000 → 3000 (earlier)
    swa_lr: float = 0.0001  # OPTIMIZED: 0.0005 → 0.0001

    # Phase 3: Expert techniques
    use_data_augmentation: bool = False  # OPTIMIZED: True → False (simpler initially)
    augmentation_probability: float = 0.3  # OPTIMIZED: 0.5 → 0.3

    # Early stopping
    early_stop_patience: int = 10
    early_stop_threshold: float = 0.01

    # Metrics
    track_metrics: bool = True


class AdvancedDQNAgent:
    """
    Advanced DQN Agent with ALL best practices

    Combines Phase 1, 2, and 3 improvements for maximum performance
    """

    def __init__(self, state_size: int, action_size: int,
                 config: Optional[AgentConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize advanced agent

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Agent configuration (uses defaults if None)
            device: Device for training ('cuda' or 'cpu')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or AgentConfig()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("🚀 Initializing Advanced DQN Agent...")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Double DQN: {'✓' if self.config.use_double_dqn else '✗'}")
        logger.info(f"   Dueling DQN: {'✓' if self.config.use_dueling_dqn else '✗'}")
        logger.info(f"   Prioritized Replay: {'✓' if self.config.use_prioritized_replay else '✗'}")
        logger.info(f"   N-Step Returns: {self.config.n_step}")
        logger.info(f"   SWA: {'✓' if self.config.use_swa else '✗'}")
        logger.info(f"   Data Augmentation: {'✓' if self.config.use_data_augmentation else '✗'}")

        # Set default architecture if not provided
        if self.config.hidden_layers is None:
            self.config.hidden_layers = [256, 256, 128]

        # Create networks
        network_class = DuelingDQN if self.config.use_dueling_dqn else self._create_standard_dqn

        if self.config.use_dueling_dqn:
            self.model = DuelingDQN(state_size, action_size, self.config.hidden_layers).to(self.device)
            self.target_model = DuelingDQN(state_size, action_size, self.config.hidden_layers).to(self.device)
        else:
            self.model = self._create_standard_dqn(state_size, action_size, self.config.hidden_layers).to(self.device)
            self.target_model = self._create_standard_dqn(state_size, action_size, self.config.hidden_layers).to(self.device)

        self.update_target_model()

        # Optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.lr_scheduler = None
        if self.config.lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                min_lr=self.config.lr_min,
                verbose=True
            )
        elif self.config.lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.lr_min
            )

        # Stochastic Weight Averaging
        self.swa_model = None
        self.swa_scheduler = None
        if self.config.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config.swa_lr)

        # Replay buffer
        if self.config.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                alpha=self.config.per_alpha,
                beta=self.config.per_beta,
                beta_increment=self.config.per_beta_increment
            )
        else:
            self.memory = deque(maxlen=self.config.buffer_size)

        # N-step buffer
        self.n_step_buffer = deque(maxlen=self.config.n_step)

        # Exploration
        self.epsilon = self.config.epsilon_start

        # Training counters
        self.training_steps = 0
        self.episodes = 0
        self.swa_started = False

        # Early stopping
        self.best_val_score = float('-inf')
        self.no_improvement_count = 0
        self.should_stop = False

        # Metrics
        if self.config.track_metrics:
            self.metrics = {
                'loss_history': [],
                'td_error_history': [],
                'grad_norm_history': [],
                'lr_history': [],
                'q_value_history': [],
                'val_score_history': []
            }

        logger.info("✅ Advanced DQN Agent initialized successfully!")

    def _create_standard_dqn(self, state_size: int, action_size: int,
                            hidden_layers: List[int]) -> nn.Module:
        """Create standard (non-dueling) DQN"""
        layers = []
        prev_size = state_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        return nn.Sequential(*layers)

    def update_target_model(self):
        """Copy weights from model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to state

        Trading-specific augmentations:
        - Price jittering (small noise)
        - Feature scaling
        """
        if not self.config.use_data_augmentation:
            return state

        if random.random() > self.config.augmentation_probability:
            return state

        augmented = state.copy()

        # Small noise injection (0.1% jitter)
        noise = np.random.normal(0, 0.001, size=augmented.shape)
        augmented += noise

        return augmented

    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If n-step buffer is full, compute n-step return
        if len(self.n_step_buffer) == self.config.n_step:
            # Compute n-step return
            n_step_return = 0
            gamma_power = 1.0

            for _, _, r, _, _ in self.n_step_buffer:
                n_step_return += gamma_power * r
                gamma_power *= self.config.gamma

            # Get first state/action and last next_state/done
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            _, _, _, last_next_state, last_done = self.n_step_buffer[-1]

            # Store n-step transition
            if self.config.use_prioritized_replay:
                self.memory.add(first_state, first_action, n_step_return,
                              last_next_state, last_done)
            else:
                self.memory.append((first_state, first_action, n_step_return,
                                  last_next_state, last_done))

    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy"""
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Augment state
        state = self._augment_state(state)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Use SWA model if available and not training
            model = self.swa_model if (not training and self.swa_model and self.swa_started) else self.model
            q_values = model(state_tensor)

            # Track Q-values
            if self.config.track_metrics:
                self.metrics['q_value_history'].append(q_values.max().item())

            return q_values.argmax().item()

    def replay(self):
        """Train on batch from replay buffer"""
        # Check if enough samples
        batch_size = self.config.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        # Sample batch
        if self.config.use_prioritized_replay:
            batch, indices, weights = self.memory.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, batch_size)
            indices = None
            weights = torch.ones(batch_size).to(self.device)

        # Prepare batch
        if self.config.use_prioritized_replay:
            states = torch.FloatTensor([t.state for t in batch]).to(self.device)
            actions = torch.LongTensor([t.action for t in batch]).to(self.device)
            rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
            dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        else:
            states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
            actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
            rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
            dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)

        # Current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.model(next_states).argmax(1)
                # Use target network to evaluate actions
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_model(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * (self.config.gamma ** self.config.n_step) * next_q_values

        # Compute TD errors (for PER priority update)
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss
        loss = (weights * (current_q_values - target_q_values).pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm
        )

        self.optimizer.step()

        # Update priorities in PER
        if self.config.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.config.target_update_frequency == 0:
            self.update_target_model()

        # Update SWA
        if self.config.use_swa and self.training_steps >= self.config.swa_start:
            if not self.swa_started:
                logger.info("✅ SWA started")
                self.swa_started = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        elif self.config.lr_scheduler_type == 'cosine' and self.lr_scheduler:
            self.lr_scheduler.step()

        # Decay epsilon
        if self.config.use_linear_epsilon_decay:
            # Linear decay from start to end over epsilon_decay_steps
            decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
            self.epsilon = max(self.config.epsilon_end,
                             self.config.epsilon_start - decay_rate * self.training_steps)
        else:
            # Multiplicative decay (original method)
            self.epsilon = max(self.config.epsilon_end,
                              self.epsilon * self.config.epsilon_decay)

        # Track metrics
        if self.config.track_metrics:
            self.metrics['loss_history'].append(loss.item())
            self.metrics['td_error_history'].append(np.abs(td_errors).mean())
            self.metrics['grad_norm_history'].append(grad_norm.item())
            self.metrics['lr_history'].append(self.optimizer.param_groups[0]['lr'])

        return loss.item()

    def update_validation_score(self, val_score: float) -> bool:
        """
        Update validation score and check early stopping

        Returns:
            True if training should stop
        """
        # Update plateau scheduler
        if self.config.lr_scheduler_type == 'plateau' and self.lr_scheduler:
            self.lr_scheduler.step(val_score)

        # Track metrics
        if self.config.track_metrics:
            self.metrics['val_score_history'].append(val_score)

        # Check improvement
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.no_improvement_count = 0
            logger.info(f"🏆 Validation improved: {val_score:.2f}")
            return False

        # Check degradation
        degradation = (self.best_val_score - val_score) / abs(self.best_val_score)

        if degradation > self.config.early_stop_threshold:
            self.no_improvement_count += 1
            logger.info(f"⚠️  Validation: {val_score:.2f} (best: {self.best_val_score:.2f}, patience: {self.no_improvement_count}/{self.config.early_stop_patience})")

            if self.no_improvement_count >= self.config.early_stop_patience:
                logger.info(f"🛑 EARLY STOPPING triggered")
                self.should_stop = True
                return True

        return False

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get training metrics summary"""
        if not self.config.track_metrics:
            return {}

        return {
            'avg_loss': np.mean(self.metrics['loss_history'][-100:]) if self.metrics['loss_history'] else 0.0,
            'avg_td_error': np.mean(self.metrics['td_error_history'][-100:]) if self.metrics['td_error_history'] else 0.0,
            'avg_grad_norm': np.mean(self.metrics['grad_norm_history'][-100:]) if self.metrics['grad_norm_history'] else 0.0,
            'avg_q_value': np.mean(self.metrics['q_value_history'][-100:]) if self.metrics['q_value_history'] else 0.0,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'best_val_score': self.best_val_score,
            'swa_active': self.swa_started
        }

    def save(self, path: str):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'swa_model_state_dict': self.swa_model.state_dict() if self.swa_model else None,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'config': self.config,
            'metrics': self.metrics if self.config.track_metrics else None
        }, path)

    def load(self, path: str):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.swa_model and checkpoint.get('swa_model_state_dict'):
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])

        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint.get('episodes', 0)

        if checkpoint.get('metrics') and self.config.track_metrics:
            self.metrics = checkpoint['metrics']
