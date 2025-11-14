# Nexlify Paper Trading System

Comprehensive paper trading system for training and evaluating ML/RL trading agents in a risk-free environment.

## Overview

The Paper Trading System provides a complete infrastructure for:

- **Training RL agents** with realistic market conditions
- **Evaluating multiple strategies** simultaneously
- **Comparing agent performance** with comprehensive metrics
- **Continuous learning** with real market data integration
- **Risk-free testing** before deploying with real capital

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                        │
│              (run_paper_trading.py)                     │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Paper Trading Runner                       │
│       (Coordinates training/evaluation)                 │
└────────┬───────────────────────┬────────────────────────┘
         │                       │
    ┌────▼─────┐         ┌──────▼──────┐
    │ Training │         │ Multi-Agent │
    │   Env    │         │Orchestrator │
    └────┬─────┘         └──────┬──────┘
         │                      │
         │              ┌───────┴───────┐
         │              │               │
    ┌────▼────┐    ┌───▼───┐      ┌───▼───┐
    │  Paper  │    │Agent 1│ ...  │Agent N│
    │ Trading │    │Engine │      │Engine │
    │ Engine  │    └───────┘      └───────┘
    └─────────┘
```

## Components

### 1. Paper Trading Engine (`nexlify_paper_trading.py`)

Core engine that simulates trading with realistic conditions:

- **Balance management** - Tracks cash and positions
- **Order execution** - Buy/sell with fees and slippage
- **Position tracking** - Real-time P&L calculation
- **Performance metrics** - Win rate, Sharpe ratio, drawdown, etc.

**Features:**
- Configurable fees (default: 0.1%)
- Slippage simulation (default: 0.05%)
- Multiple position support
- Complete trade history
- Equity curve tracking

### 2. RL Training Environment (`nexlify_rl_training_env.py`)

OpenAI Gym-compatible environment for training RL agents:

**State Space (12 features (crypto-optimized)):**
1. Normalized balance
2. Position size
3. Relative entry price
4. Normalized current price
5. Price change
6. RSI indicator
7. MACD indicator
8. Volume ratio

**Action Space:**
- 0: Buy
- 1: Sell
- 2: Hold

**Reward Function:**
- Equity change (normalized)
- Transaction cost penalties
- Unrealized gain rewards
- Position holding penalties

### 3. Paper Trading Orchestrator (`nexlify_paper_trading_orchestrator.py`)

Manages multiple agents simultaneously:

- **Multi-agent coordination** - Run multiple strategies in parallel
- **Performance comparison** - Real-time leaderboard
- **Isolated engines** - Each agent has independent balance
- **Comprehensive reporting** - Detailed performance analysis

### 4. Paper Trading Runner (`nexlify_paper_trading_runner.py`)

Main execution script with three modes:

**Training Mode:**
- Train single RL agent
- Automatic checkpointing
- Episode statistics
- Model saving

**Evaluation Mode:**
- Compare multiple trained models
- Statistical analysis
- Performance ranking

**Multi-Agent Mode:**
- Run multiple agents simultaneously
- Real-time comparison
- Session management

## Quick Start

### 1. Create Configuration

```bash
python run_paper_trading.py create-config
```

This creates `config/paper_trading_config.json`:

```json
{
  "paper_trading": {
    "initial_balance": 10000.0,
    "fee_rate": 0.001,
    "slippage": 0.0005,
    "update_interval": 60
  },
  "training": {
    "episodes": 100,
    "max_steps": 1000,
    "save_frequency": 10
  },
  "agents": [
    {
      "agent_id": "adaptive_rl_1",
      "agent_type": "rl_adaptive",
      "name": "Adaptive RL Agent",
      "enabled": true
    }
  ]
}
```

### 2. Train an Agent

**Train Adaptive RL Agent:**
```bash
python run_paper_trading.py train --agent-type adaptive --episodes 100
```

**Train Ultra-Optimized Agent:**
```bash
python run_paper_trading.py train --agent-type ultra --episodes 100
```

**Output:**
```
Episode 1/100 completed:
  Total Reward: 45.23
  Final Equity: $10,234.56
  Return: 2.35%
  Win Rate: 62.5%
  Trades: 8

Episode 2/100 completed:
  ...

💾 Checkpoint saved: models/paper_trading_adaptive_episode_10.pt
```

### 3. Evaluate Trained Models

```bash
python run_paper_trading.py evaluate \
    --models models/paper_trading_adaptive_episode_*.pt \
    --episodes 20
```

**Output:**
```
================================================================================
EVALUATION SUMMARY
================================================================================
1. paper_trading_adaptive_episode_100.pt
   Mean Return: 3.45% ± 1.23%
   Win Rate: 65.3%

2. paper_trading_adaptive_episode_50.pt
   Mean Return: 2.87% ± 1.45%
   Win Rate: 61.2%
```

### 4. Run Multi-Agent Session

Edit `config/paper_trading_config.json` to add multiple agents:

```json
{
  "agents": [
    {
      "agent_id": "adaptive_1",
      "agent_type": "rl_adaptive",
      "name": "Adaptive Agent",
      "enabled": true
    },
    {
      "agent_id": "ultra_1",
      "agent_type": "rl_ultra",
      "name": "Ultra-Optimized Agent",
      "model_path": "models/best_model.pt",
      "enabled": true
    }
  ]
}
```

Run session:
```bash
python run_paper_trading.py multi-agent \
    --config config/paper_trading_config.json \
    --duration 24  # 24 hours
```

## Advanced Usage

### Custom Training Loop

```python
from nexlify.env.nexlify_rl_training_env import TradingEnvironment
from nexlify.strategies.nexlify_adaptive_rl_agent import create_optimized_agent

# Create environment
env = TradingEnvironment(
    initial_balance=10000.0,
    use_paper_trading=True
)

# Create agent
agent = create_optimized_agent(
    state_size=env.state_size,
    action_size=env.action_size,
    auto_detect=True
)

# Training loop
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state, training=True)
        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    print(f"Episode {episode}: Reward={total_reward:.2f}")
```

### Custom Multi-Agent Orchestration

```python
from nexlify.backtesting.nexlify_paper_trading_orchestrator import (
    PaperTradingOrchestrator,
    AgentConfig
)

# Create orchestrator
orchestrator = PaperTradingOrchestrator({
    'initial_balance': 10000.0,
    'update_interval': 60
})

# Register agents
agents = [
    AgentConfig('agent1', 'rl_adaptive', 'Strategy A'),
    AgentConfig('agent2', 'rl_ultra', 'Strategy B'),
    AgentConfig('agent3', 'ml_ensemble', 'Strategy C')
]

for agent_config in agents:
    orchestrator.register_agent(agent_config)

    # Load and register agent instance
    agent = load_your_agent(agent_config)
    orchestrator.load_agent_model(agent_config.agent_id, agent)

# Run session
await orchestrator.start_session(duration_hours=24)

# Get results
report = orchestrator.generate_final_report()
leaderboard = orchestrator.get_leaderboard()
```

## Performance Metrics

The system tracks comprehensive performance metrics:

### Agent-Level Metrics
- **Total Return** - Absolute and percentage
- **Win Rate** - Percentage of profitable trades
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Largest peak-to-trough decline
- **Profit Factor** - Gross profit / gross loss
- **Average Win/Loss** - Mean profit per winning/losing trade
- **Total Trades** - Number of trades executed

### Real-Time Tracking
- **Equity Curve** - Balance over time
- **Unrealized P&L** - Open position value
- **Position Tracking** - Current holdings
- **Fee Analysis** - Total fees paid

### Multi-Agent Comparison
- **Leaderboard** - Agents ranked by performance
- **Relative Performance** - Head-to-head comparison
- **Risk-Adjusted Rankings** - Sharpe ratio based

## Configuration Options

### Paper Trading Settings

```json
{
  "paper_trading": {
    "initial_balance": 10000.0,      // Starting balance
    "fee_rate": 0.001,               // 0.1% trading fee
    "slippage": 0.0005,              // 0.05% slippage
    "update_interval": 60            // Update frequency (seconds)
  }
}
```

### Training Settings

```json
{
  "training": {
    "episodes": 100,                 // Training episodes
    "max_steps": 1000,               // Steps per episode
    "save_frequency": 10             // Checkpoint frequency
  }
}
```

### Agent Configuration

```json
{
  "agents": [
    {
      "agent_id": "unique_id",       // Unique identifier
      "agent_type": "rl_adaptive",   // Agent type
      "name": "Display Name",        // Human-readable name
      "enabled": true,               // Enable/disable
      "model_path": "path/to/model", // Pre-trained model (optional)
      "config": {                    // Agent-specific config
        "enable_sentiment": false
      }
    }
  ]
}
```

## Testing

Run comprehensive test suite:

```bash
# Run all tests
pytest tests/test_paper_trading_system.py -v

# Run specific test class
pytest tests/test_paper_trading_system.py::TestPaperTradingEngine -v

# Run with coverage
pytest tests/test_paper_trading_system.py --cov=nexlify.backtesting --cov=nexlify.env
```

## Examples

### Example 1: Train and Evaluate

```bash
# Train agent
python run_paper_trading.py train --agent-type adaptive --episodes 50

# Evaluate on multiple episodes
python run_paper_trading.py evaluate \
    --models models/paper_trading_adaptive_final.pt \
    --episodes 20
```

### Example 2: Compare Multiple Agents

```bash
# Train multiple agents
python run_paper_trading.py train --agent-type adaptive --episodes 100
python run_paper_trading.py train --agent-type ultra --episodes 100

# Compare performance
python run_paper_trading.py evaluate \
    --models models/paper_trading_adaptive_final.pt \
             models/paper_trading_ultra_final.pt \
    --episodes 50
```

### Example 3: Long-Running Session

```bash
# Run 7-day multi-agent session
python run_paper_trading.py multi-agent \
    --config config/paper_trading_config.json \
    --duration 168  # 7 days * 24 hours
```

## Output Files

The system generates comprehensive output:

```
paper_trading/
├── sessions/
│   └── 20250112_143052/          # Session ID
│       ├── session_summary.json   # Overall summary
│       ├── agent_adaptive_1.json  # Agent 1 data
│       ├── agent_ultra_1.json     # Agent 2 data
│       ├── performance_adaptive_1.csv  # Performance history
│       ├── performance_ultra_1.csv
│       └── final_report.txt       # Human-readable report
├── logs/
│   └── session.log                # Detailed logs
└── models/
    ├── paper_trading_adaptive_episode_10.pt
    ├── paper_trading_adaptive_episode_20.pt
    └── paper_trading_adaptive_final.pt
```

## Integration with Existing System

The paper trading system integrates seamlessly with Nexlify's existing infrastructure:

### RL Agents
- ✅ Adaptive RL Agent (`nexlify_adaptive_rl_agent.py`)
- ✅ Ultra-Optimized RL Agent (`nexlify_ultra_optimized_rl_agent.py`)
- ✅ Basic RL Agent (`nexlify_rl_agent.py`)

### ML Models
- ✅ Ensemble ML (`nexlify_ensemble_ml.py`)
- ✅ Feature Engineering (`nexlify_feature_engineering.py`)

### Infrastructure
- ✅ Paper Trading Engine (existing)
- ✅ Backtesting Framework (existing)
- ✅ Performance Tracking (existing)
- ✅ Risk Management (existing)

## Best Practices

### Training
1. **Start small** - Begin with 10-20 episodes to test
2. **Monitor progress** - Check episode stats regularly
3. **Save checkpoints** - Don't lose training progress
4. **Adjust hyperparameters** - Tune based on performance

### Evaluation
1. **Multiple episodes** - Run 20+ episodes for statistical significance
2. **Compare fairly** - Use same initial balance and fees
3. **Consider risk** - Don't just look at returns
4. **Validate on unseen data** - Test on new market conditions

### Multi-Agent Sessions
1. **Diverse strategies** - Mix different agent types
2. **Monitor resources** - Watch CPU/GPU/memory usage
3. **Set reasonable duration** - Start with short sessions
4. **Review regularly** - Check performance and adjust

## Troubleshooting

### Common Issues

**Issue: "No module named 'nexlify'"**
- **Solution:** Run from project root directory

**Issue: "Agent model not found"**
- **Solution:** Train agent first or check model path

**Issue: "PyTorch not available"**
- **Solution:** Install PyTorch: `pip install torch`

**Issue: "CUDA out of memory"**
- **Solution:** Reduce batch size or use CPU mode

## Performance Optimization

### Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: 6+ GB VRAM
- Storage: 50 GB SSD

### Optimization Tips

1. **Use GPU** - 10-50x faster training
2. **Batch processing** - Larger batches = faster training
3. **Parallel environments** - Multiple envs simultaneously
4. **Compiled models** - 30-50% speedup
5. **Mixed precision** - 2-4x faster on modern GPUs

## Future Enhancements

Planned features:

- [ ] Real-time market data integration
- [ ] Advanced visualization dashboard
- [ ] Hyperparameter optimization
- [ ] Portfolio-level paper trading
- [ ] Event simulation (flash crashes, news)
- [ ] Transfer learning support
- [ ] Distributed training across multiple machines

## Support

For issues or questions:

1. Check this documentation
2. Review test cases in `tests/test_paper_trading_system.py`
3. Check logs in `paper_trading/logs/`
4. Open an issue on GitHub

## License

Part of the Nexlify project. See main LICENSE file.
