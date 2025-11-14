# Prioritized Experience Replay (PER)

This module implements Prioritized Experience Replay (PER) with SumTree data structure for efficient sampling in Deep Reinforcement Learning.

## Overview

Prioritized Experience Replay improves learning efficiency by sampling experiences based on their importance (TD error). Important experiences are sampled more frequently, allowing the agent to learn faster from critical mistakes and significant rewards.

### Key Features

- **SumTree Data Structure**: O(log n) priority-based sampling
- **Importance Sampling Weights**: Corrects bias introduced by prioritization
- **Beta Annealing**: Gradually increases bias correction over training
- **Configurable**: Fully integrated with Nexlify configuration system
- **Visualization Tools**: Track and visualize PER statistics during training

## Architecture

```
nexlify/memory/
├── sumtree.py                      # Binary tree for O(log n) sampling
├── prioritized_replay_buffer.py   # PER buffer implementation
├── per_visualization.py            # Visualization and tracking tools
└── __init__.py                     # Package exports
```

## Usage

### Basic Usage

```python
from nexlify.memory import PrioritizedReplayBuffer
import numpy as np

# Create buffer
buffer = PrioritizedReplayBuffer(
    capacity=100000,
    alpha=0.6,          # Prioritization strength
    beta_start=0.4,     # Initial IS correction
    beta_end=1.0,       # Final IS correction
    beta_annealing_steps=10000
)

# Add experience
state = np.array([1, 2, 3])
action = 1
reward = 1.0
next_state = np.array([4, 5, 6])
done = False

buffer.push(state, action, reward, next_state, done)

# Sample batch
experiences, indices, weights = buffer.sample(batch_size=32)

# Update priorities after learning
td_errors = np.array([...])  # Calculate from TD error
buffer.update_priorities(indices, td_errors)
```

### Integration with DQNAgent

```python
from nexlify.strategies.nexlify_rl_agent import DQNAgent

# Configure agent with PER
config = {
    'use_prioritized_replay': True,
    'per_alpha': 0.6,
    'per_beta_start': 0.4,
    'per_beta_end': 1.0,
    'per_beta_annealing_steps': 10000,
    'replay_buffer_size': 60000
}

agent = DQNAgent(state_size=12, action_size=3, config=config)

# Agent automatically uses PER for training
# replay() method handles IS weights and priority updates
```

### Using Default Configuration

```python
from nexlify.config.crypto_trading_config import CRYPTO_24_7_CONFIG
from nexlify.strategies.nexlify_rl_agent import DQNAgent

# PER is enabled by default with optimized settings
agent = DQNAgent(
    state_size=12,
    action_size=3,
    config=CRYPTO_24_7_CONFIG.to_dict()
)
```

### Monitoring PER Statistics

```python
from nexlify.memory import PERStatsTracker, create_per_report

# Create tracker
tracker = PERStatsTracker()

# During training
for episode in range(num_episodes):
    # ... training code ...

    # Record PER stats
    if agent.use_per:
        stats = agent.get_per_stats()
        tracker.record(stats, episode=episode)

        # Print report every 100 episodes
        if episode % 100 == 0:
            report = create_per_report(stats, tracker)
            print(report)

# Save tracking data
tracker.save('per_history.json')

# Create visualization
tracker.plot('per_stats.png', title='PER Training Statistics')
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_prioritized_replay` | `True` | Enable/disable PER |
| `per_alpha` | `0.6` | Prioritization exponent (0=uniform, 1=full) |
| `per_beta_start` | `0.4` | Initial IS correction |
| `per_beta_end` | `1.0` | Final IS correction |
| `per_beta_annealing_steps` | `10000` | Steps to anneal beta |
| `per_epsilon` | `1e-6` | Small constant for zero priorities |
| `per_priority_clip` | `None` | Max priority value (None=no clip) |

## How It Works

### 1. Priority Assignment

Priorities are calculated from TD error:

```
priority = (|TD_error| + epsilon)^alpha
```

- Higher TD error → Higher priority → More likely to be sampled
- `alpha` controls prioritization strength (0=uniform, 1=fully prioritized)
- `epsilon` prevents zero priorities

### 2. Sampling

Experiences are sampled proportional to their priorities using a SumTree:

```
P(i) = p_i^alpha / Σ p_k^alpha
```

- SumTree enables O(log n) sampling
- Segment-based sampling ensures diversity

### 3. Importance Sampling Weights

To correct bias from non-uniform sampling:

```
w_i = (N * P(i))^(-beta) / max_w
```

- Weights are applied to loss during training
- `beta` increases from 0.4 to 1.0 over training (annealing)
- Weights are normalized by max for stability

### 4. Priority Updates

After training step, priorities are updated with new TD errors:

```python
# Calculate TD errors
td_errors = target_q - predicted_q

# Update priorities in buffer
buffer.update_priorities(indices, td_errors)
```

## Performance

### Expected Improvements

- **30-50% faster learning** on important experiences
- **Better sample efficiency** in sparse reward environments
- **Improved stability** with importance sampling correction

### Computational Complexity

| Operation | Standard Buffer | PER Buffer |
|-----------|----------------|------------|
| Push | O(1) | O(log n) |
| Sample | O(k) | O(k log n) |
| Update | - | O(k log n) |

Where:
- n = buffer capacity
- k = batch size

### Memory Usage

- **Standard Buffer**: ~8n bytes (for indices)
- **PER Buffer**: ~24n bytes (tree + data arrays)
- Example: 60,000 capacity ≈ 1.4 MB additional memory

## Validation

Run the validation script to test PER implementation:

```bash
python validate_per.py
```

Expected output:
```
✅ ALL TESTS PASSED!

PER implementation is working correctly:
  ✓ SumTree for O(log n) sampling
  ✓ Priority-based experience replay
  ✓ Importance sampling weights
  ✓ Beta annealing
  ✓ Visualization and reporting tools
```

## Testing

Comprehensive test suite available:

```bash
pytest tests/test_prioritized_replay.py -v
```

Test coverage includes:
- SumTree operations (add, update, sample)
- PrioritizedReplayBuffer functionality
- Integration with DQNAgent
- Beta annealing
- Importance sampling weights
- Visualization tools

## Research Background

Based on the paper:
> Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015).
> **Prioritized Experience Replay**
> *International Conference on Learning Representations (ICLR)*

Key insights:
- Samples with high TD error are more informative
- Importance sampling corrects for bias
- Significant improvements in Atari games and other domains

## Troubleshooting

### High Memory Usage

If memory is a concern, reduce buffer capacity:

```python
config = {
    'replay_buffer_size': 30000,  # Reduce from 60000
    'use_prioritized_replay': True
}
```

### PER Not Available

If PER is not available, agent falls back to standard replay buffer:

```
⚠️ PER requested but not available - using standard replay buffer
```

Ensure numpy is installed:
```bash
pip install numpy
```

### Disable PER

To use standard replay buffer:

```python
config = {
    'use_prioritized_replay': False
}
```

## Future Enhancements

Potential improvements:
- [ ] Multi-step prioritization
- [ ] Distributed PER for multi-agent systems
- [ ] GPU-accelerated sampling for very large buffers
- [ ] Adaptive alpha based on training progress
- [ ] Alternative priority metrics (policy gradient, curiosity)

## References

1. Schaul et al. (2015) - Prioritized Experience Replay
2. Hessel et al. (2018) - Rainbow: Combining Improvements in Deep RL
3. Horgan et al. (2018) - Distributed Prioritized Experience Replay

## License

Part of the Nexlify project. See main LICENSE file.
