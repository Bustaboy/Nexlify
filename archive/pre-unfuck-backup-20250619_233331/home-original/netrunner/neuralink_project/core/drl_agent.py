# /home/netrunner/neuralink_project/core/drl_agent.py
"""
NEXUS-7 DRL Agent Core
Where decisions meet destiny in the neon sprawl
PPO-based architecture with xLSTM memory integration
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import numba as nb
from .xlstm import xLSTMLayer, sigmoid_chrome, tanh_neural

@dataclass
class AgentConfig:
    """Neural configuration for our street samurai"""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    memory_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor - how much we care about tomorrow
    gae_lambda: float = 0.95  # GAE parameter - smoothing the advantage
    clip_epsilon: float = 0.2  # PPO clipping - keeping updates street-legal
    entropy_coef: float = 0.01  # Exploration bonus - gotta take risks
    value_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping - prevent neural burnout
    batch_size: int = 64
    sequence_length: int = 32
    update_epochs: int = 4
    
class MemoryBank:
    """
    Episodic memory storage - remembers the streets
    Optimized for minimal RAM footprint
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays for speed
        self.states = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.dones = None
        self.hidden_states = None
        
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, value: float, log_prob: float, 
             done: bool, hidden_state: Optional[np.ndarray] = None):
        """Store experience in the memory bank"""
        if self.states is None:
            # Initialize on first push
            self._initialize_storage(state.shape, action.shape, 
                                   hidden_state.shape if hidden_state is not None else None)
        
        idx = self.position % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        if hidden_state is not None:
            self.hidden_states[idx] = hidden_state
        
        self.position += 1
        self.size = min(self.size + 1, self.capacity)
    
    def _initialize_storage(self, state_shape: Tuple, action_shape: Tuple, 
                          hidden_shape: Optional[Tuple]):
        """Pre-allocate memory arrays"""
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=bool)
        
        if hidden_shape:
            self.hidden_states = np.zeros((self.capacity, *hidden_shape), dtype=np.float32)
    
    def get_batch(self, indices: np.ndarray) -> Dict:
        """Extract batch for training"""
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices],
            'hidden_states': self.hidden_states[indices] if self.hidden_states is not None else None
        }
    
    def clear(self):
        """Wipe the memory - fresh start"""
        self.position = 0
        self.size = 0

class NeuralNetwork:
    """
    Base neural architecture - the chrome substrate
    Hand-crafted for maximum efficiency
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [256, 128]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Initialize layers
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            w = self._he_init(dims[i], dims[i+1])
            b = np.zeros(dims[i+1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
    
    def _he_init(self, n_in: int, n_out: int) -> np.ndarray:
        """He initialization - optimal for ReLU networks"""
        std = np.sqrt(2.0 / n_in)
        return np.random.normal(0, std, (n_in, n_out)).astype(np.float32)
    
    @nb.jit(nopython=True)
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation - simple but chrome"""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the neural maze"""
        activation = x
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            activation = activation @ self.weights[i] + self.biases[i]
            activation = np.maximum(0, activation)  # ReLU
        
        # Output layer (no activation)
        output = activation @ self.weights[-1] + self.biases[-1]
        return output

class DRLAgent:
    """
    Deep Reinforcement Learning Agent
    Combines PPO with xLSTM for enhanced memory
    Built for the mean streets of Night City
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Policy network (actor) with xLSTM
        self.policy_xlstm = xLSTMLayer(
            input_dim=config.state_dim,
            hidden_dim=config.memory_dim,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            return_sequences=False,
            stateful=True
        )
        
        self.policy_head = NeuralNetwork(
            input_dim=config.memory_dim,
            output_dim=config.action_dim,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2]
        )
        
        # Value network (critic) with xLSTM
        self.value_xlstm = xLSTMLayer(
            input_dim=config.state_dim,
            hidden_dim=config.memory_dim,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            return_sequences=False,
            stateful=True
        )
        
        self.value_head = NeuralNetwork(
            input_dim=config.memory_dim,
            output_dim=1,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2]
        )
        
        # Memory bank for experience replay
        self.memory = MemoryBank(capacity=config.sequence_length * 100)
        
        # Optimization parameters
        self.learning_rate = config.learning_rate
        self.eps = 1e-8  # Numerical stability
        
        # Running statistics for normalization
        self.state_mean = np.zeros(config.state_dim, dtype=np.float32)
        self.state_std = np.ones(config.state_dim, dtype=np.float32)
        self.n_updates = 0
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize input state - keep the signals clean"""
        return (state - self.state_mean) / (self.state_std + self.eps)
    
    def update_statistics(self, states: np.ndarray):
        """Update running statistics - adapt to the environment"""
        batch_mean = np.mean(states, axis=0)
        batch_std = np.std(states, axis=0)
        
        # Exponential moving average
        alpha = 0.01
        self.state_mean = (1 - alpha) * self.state_mean + alpha * batch_mean
        self.state_std = (1 - alpha) * self.state_std + alpha * batch_std
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action using the policy network
        Returns: action, log_prob, value
        """
        # Normalize state
        state_norm = self.normalize_state(state)
        
        # Expand dims for batch processing
        if state_norm.ndim == 1:
            state_norm = state_norm[np.newaxis, np.newaxis, :]
        
        # Get memory representation from xLSTM
        memory_rep = self.policy_xlstm.forward(state_norm)
        
        # Get action logits
        action_logits = self.policy_head.forward(memory_rep)
        
        # Get value estimate
        value_memory = self.value_xlstm.forward(state_norm)
        value = self.value_head.forward(value_memory).squeeze()
        
        # Action selection (discrete actions)
        if self.config.action_dim > 1:
            # Softmax for discrete actions
            action_probs = self._softmax(action_logits)
            
            if deterministic:
                action = np.argmax(action_probs, axis=-1)
            else:
                # Sample from distribution
                action = np.random.choice(self.config.action_dim, p=action_probs.squeeze())
            
            # Calculate log probability
            log_prob = np.log(action_probs.squeeze()[action] + self.eps)
        else:
            # Continuous action (using tanh squashing)
            mean = tanh_neural(action_logits)
            if deterministic:
                action = mean
            else:
                # Add noise for exploration
                noise = np.random.normal(0, 0.1, size=mean.shape)
                action = np.clip(mean + noise, -1, 1)
            
            # Simplified log prob for continuous (would need proper distribution in production)
            log_prob = -0.5 * np.sum((action - mean) ** 2)
        
        return action, log_prob, value
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                    dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation
        The secret sauce for stable learning
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # Work backwards through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.config.gamma * next_v * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_update(self, batch: Dict) -> Dict[str, float]:
        """
        PPO update step - where learning happens
        Returns training metrics
        """
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Normalize advantages - stability trick
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + self.eps)
        
        # Multiple epochs of updates
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy = 0
        
        for _ in range(self.config.update_epochs):
            # Get current policy predictions
            # (In production, would batch process this)
            new_log_probs = []
            new_values = []
            
            for i in range(len(states)):
                _, log_prob, value = self.get_action(states[i], deterministic=False)
                new_log_probs.append(log_prob)
                new_values.append(value)
            
            new_log_probs = np.array(new_log_probs)
            new_values = np.array(new_values)
            
            # PPO ratio
            ratio = np.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.config.clip_epsilon, 
                           1 + self.config.clip_epsilon) * advantages
            policy_loss = -np.mean(np.minimum(surr1, surr2))
            
            # Value loss (MSE)
            value_loss = np.mean((new_values - returns) ** 2)
            
            # Entropy bonus (simplified)
            entropy = -np.mean(new_log_probs)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_coef * value_loss - 
                         self.config.entropy_coef * entropy)
        
        return {
            'total_loss': float(total_loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save neural state - backup the chrome"""
        checkpoint = {
            'policy_xlstm': self.policy_xlstm.cell.__dict__,
            'value_xlstm': self.value_xlstm.cell.__dict__,
            'policy_head_weights': self.policy_head.weights,
            'policy_head_biases': self.policy_head.biases,
            'value_head_weights': self.value_head.weights,
            'value_head_biases': self.value_head.biases,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'n_updates': self.n_updates
        }
        np.savez_compressed(filepath, **checkpoint)
    
    def load_checkpoint(self, filepath: str):
        """Load neural state - restore from backup"""
        checkpoint = np.load(filepath, allow_pickle=True)
        
        # Restore xLSTM states
        for key, value in checkpoint['policy_xlstm'].item().items():
            setattr(self.policy_xlstm.cell, key, value)
        
        for key, value in checkpoint['value_xlstm'].item().items():
            setattr(self.value_xlstm.cell, key, value)
        
        # Restore network weights
        self.policy_head.weights = checkpoint['policy_head_weights'].tolist()
        self.policy_head.biases = checkpoint['policy_head_biases'].tolist()
        self.value_head.weights = checkpoint['value_head_weights'].tolist()
        self.value_head.biases = checkpoint['value_head_biases'].tolist()
        
        # Restore statistics
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.n_updates = int(checkpoint['n_updates'])
