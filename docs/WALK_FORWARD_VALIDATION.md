# Walk-Forward Validation

**Version:** 1.0.0
**Last Updated:** 2025-11-15
**Module:** `nexlify.validation.walk_forward`

## Overview

Walk-forward validation is a sophisticated backtesting technique designed for time-series trading strategies. It prevents future data leakage and provides realistic performance estimates by training on historical data and testing on sequential future periods.

## Why Walk-Forward Validation?

Traditional cross-validation methods (like k-fold) randomly split data, which can cause **future data leakage** in time-series scenarios. Walk-forward validation:

- ✅ Respects temporal order of data
- ✅ Prevents look-ahead bias
- ✅ Provides realistic out-of-sample performance
- ✅ Helps detect overfitting
- ✅ Simulates real trading conditions

## Validation Modes

### Rolling Window Mode
- **Fixed training window** that slides forward
- Maintains constant training data size
- Good for detecting regime changes
- Example:
  ```
  Fold 1: Train [0-1000]     → Test [1000-1200]
  Fold 2: Train [200-1200]   → Test [1200-1400]
  Fold 3: Train [400-1400]   → Test [1400-1600]
  ```

### Expanding Window Mode
- **Growing training window** anchored at start
- Training set expands over time
- Better for long-term trend learning
- Example:
  ```
  Fold 1: Train [0-1000]     → Test [1000-1200]
  Fold 2: Train [0-1200]     → Test [1200-1400]
  Fold 3: Train [0-1400]     → Test [1400-1600]
  ```

## Quick Start

### Basic Usage

```python
from nexlify.validation.walk_forward import WalkForwardValidator

# Initialize validator
validator = WalkForwardValidator(
    total_episodes=2000,
    train_size=1000,
    test_size=200,
    step_size=200,
    mode='rolling'
)

# Define training and evaluation functions
async def train_fn(train_start, train_end):
    # Train your model on episodes [train_start:train_end]
    model = train_model(train_start, train_end)
    return model

async def eval_fn(model, test_start, test_end):
    # Evaluate model on episodes [test_start:test_end]
    metrics = evaluate_model(model, test_start, test_end)
    return metrics  # Dict with performance metrics

# Run validation
results = await validator.validate(
    train_fn=train_fn,
    eval_fn=eval_fn,
    save_models=True
)

# View summary
print(results.summary())

# Generate visual report
validator.generate_report(results, output_dir='reports/walk_forward')
```

### Using Configuration File

Add to `config/neural_config.json`:

```json
{
  "walk_forward": {
    "enabled": true,
    "total_episodes": 2000,
    "train_size": 1000,
    "test_size": 200,
    "step_size": 200,
    "mode": "rolling",
    "min_train_size": 500,
    "save_models": true,
    "risk_free_rate": 0.02,
    "output_dir": "reports/walk_forward",
    "model_dir": "models/walk_forward"
  }
}
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_episodes` | int | - | Total number of episodes available |
| `train_size` | int | 1000 | Episodes for training window |
| `test_size` | int | 200 | Episodes for testing window |
| `step_size` | int | 200 | How far to step forward between folds |
| `mode` | str | 'rolling' | 'rolling' or 'expanding' |
| `min_train_size` | int | 500 | Minimum training episodes |
| `save_models` | bool | true | Save models from each fold |
| `risk_free_rate` | float | 0.02 | Annual risk-free rate (for Sharpe) |
| `output_dir` | str | - | Directory for reports |
| `model_dir` | str | - | Directory for saved models |

## Performance Metrics

Each fold is evaluated on comprehensive metrics:

### Return Metrics
- **Total Return**: Cumulative return over test period
- **Volatility**: Annualized volatility
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs. max drawdown

### Risk Metrics
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses

### Trade Metrics
- **Number of Trades**: Total trades executed
- **Average Trade Duration**: Mean holding period

## Results Analysis

### WalkForwardResults Object

```python
results = await validator.validate(train_fn, eval_fn)

# Access individual fold results
for fold_metric in results.fold_metrics:
    print(f"Fold {fold_metric.fold_id}: {fold_metric.total_return:.2%}")

# Access aggregated statistics
print(f"Mean Sharpe: {results.mean_metrics['sharpe_ratio']:.2f}")
print(f"Std Sharpe: {results.std_metrics['sharpe_ratio']:.2f}")

# Identify best/worst folds
best = results.fold_metrics[results.best_fold_id]
worst = results.fold_metrics[results.worst_fold_id]

# Export to JSON
with open('results.json', 'w') as f:
    json.dump(results.to_dict(), f, indent=2)
```

### Visual Reports

The `generate_report()` method creates:

1. **Fold Comparison Chart**: Bar charts comparing metrics across folds
   - Total return per fold
   - Sharpe ratio per fold
   - Win rate per fold
   - Max drawdown per fold

2. **Performance Stability Analysis**:
   - Cumulative returns across folds
   - Rolling standard deviation of returns
   - Consistency metrics

3. **Metric Distributions**:
   - Histograms of key metrics
   - Mean and standard deviation overlays
   - Performance spread visualization

4. **Text Summary**: Statistical summary with mean ± std
5. **JSON Export**: Complete results in machine-readable format

## Integration with Training

### Option 1: Run After Training

```python
# Train model normally
model = train_rl_agent(episodes=5000)

# Then validate with walk-forward
validator = WalkForwardValidator(...)
results = await validator.validate(...)
```

### Option 2: Use as Training Method

```python
# Walk-forward becomes the training loop
validator = WalkForwardValidator(
    total_episodes=5000,
    train_size=1000,
    test_size=200,
    step_size=200
)

results = await validator.validate(
    train_fn=incremental_train,
    eval_fn=evaluate,
    save_models=True
)

# Select best model based on validation metric
best_model_path = f"models/walk_forward/fold_{results.best_fold_id}_model.pt"
```

## Example: RL Agent Validation

See `examples/walk_forward_example.py` for a complete example integrating with Nexlify's RL agents.

```python
from nexlify.validation.walk_forward import WalkForwardValidator
from nexlify.strategies.nexlify_rl_agent import NexlifyRLAgent

class WalkForwardTrainer:
    async def train_fold(self, train_start, train_end):
        # Initialize fresh agent
        agent = NexlifyRLAgent(...)

        # Train on window
        for episode in range(train_start, train_end):
            # ... training loop ...
            pass

        return agent

    async def evaluate_fold(self, agent, test_start, test_end):
        # Evaluate with no exploration
        agent.epsilon = 0.0

        returns = []
        for episode in range(test_start, test_end):
            # ... evaluation loop ...
            returns.append(episode_return)

        # Calculate metrics
        from nexlify.validation.walk_forward import calculate_performance_metrics
        return calculate_performance_metrics(
            returns=np.array(returns),
            risk_free_rate=0.02
        )

# Run validation
trainer = WalkForwardTrainer(...)
validator = WalkForwardValidator(...)
results = await validator.validate(
    train_fn=trainer.train_fold,
    eval_fn=trainer.evaluate_fold
)
```

## Best Practices

### 1. Choose Appropriate Window Sizes

- **Training Window**: Should be large enough to learn patterns (e.g., 1000+ episodes)
- **Test Window**: Should be small enough to avoid overfitting metrics (e.g., 200-500 episodes)
- **Step Size**: Balance between computational cost and thoroughness
  - Smaller step = more folds = better validation but slower
  - Larger step = fewer folds = faster but less robust

### 2. Select the Right Mode

- **Rolling**: Use when market conditions change frequently
- **Expanding**: Use when long-term trends are important
- Consider running both and comparing results

### 3. Interpret Results Carefully

- **Look at standard deviations**: High std indicates unstable performance
- **Check all folds**: Don't just focus on best fold
- **Analyze failures**: Worst folds reveal weaknesses
- **Consistency > Peak Performance**: Prefer stable returns over volatile high returns

### 4. Model Selection

```python
# Option 1: Best single fold
best_fold_id = results.best_fold_id
best_model = f"models/walk_forward/fold_{best_fold_id}_model.pt"

# Option 2: Ensemble of top folds
top_folds = sorted(
    results.fold_metrics,
    key=lambda x: x.sharpe_ratio,
    reverse=True
)[:3]

# Option 3: Most consistent fold (lowest variance)
# Custom selection based on your criteria
```

## Performance Considerations

### Computational Cost

Walk-forward validation is computationally expensive:
- Training multiple models (one per fold)
- Full evaluation on each test window

**Optimization tips**:
- Use larger `step_size` to reduce number of folds
- Parallelize fold processing (future feature)
- Use GPU acceleration for RL training
- Cache market data to avoid repeated fetching

### Memory Management

- Each fold creates a new model instance
- Set `save_models=False` if disk space is limited
- Use `model_dir` to control where models are saved
- Clean up old validation runs periodically

## Troubleshooting

### No Valid Folds Generated

```python
ValueError: No valid folds could be generated
```

**Solutions**:
- Reduce `train_size` or `test_size`
- Reduce `step_size`
- Increase `total_episodes`
- Check that `train_size + test_size <= total_episodes`

### Poor Performance Across All Folds

Possible causes:
- Model architecture too simple
- Insufficient training episodes per fold
- Data quality issues
- Inappropriate hyperparameters

**Try**:
- Increase `train_size`
- Use expanding mode instead of rolling
- Review training logs for each fold
- Analyze market data quality

### High Variance Between Folds

Indicates overfitting or unstable strategy:
- Reduce model complexity
- Increase regularization
- Use ensemble methods
- Review feature engineering

## API Reference

### WalkForwardValidator

```python
class WalkForwardValidator:
    def __init__(
        total_episodes: int,
        train_size: int = 1000,
        test_size: int = 200,
        step_size: int = 200,
        mode: str = 'rolling',
        min_train_size: int = 500,
        config: Optional[Dict[str, Any]] = None
    )

    async def validate(
        train_fn: Callable[[int, int], Any],
        eval_fn: Callable[[Any, int, int], Dict[str, float]],
        save_models: bool = True,
        model_dir: Optional[Path] = None
    ) -> WalkForwardResults

    def generate_report(
        results: WalkForwardResults,
        output_dir: Optional[Path] = None
    ) -> None
```

### calculate_performance_metrics

```python
def calculate_performance_metrics(
    returns: np.ndarray,
    trades: Optional[List[Dict[str, Any]]] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]
```

## Examples

See `examples/walk_forward_example.py` for a complete working example.

## References

- [Walk-Forward Analysis on Investopedia](https://www.investopedia.com/terms/w/walk-forward-analysis.asp)
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.

## Future Enhancements

Planned features:
- Parallel fold processing for faster validation
- Purged k-fold cross-validation (embargo periods)
- Combinatorial purged cross-validation
- Walk-forward optimization (reoptimize parameters per fold)
- Integration with hyperparameter tuning
- Support for multi-asset portfolios

## Support

For issues or questions:
- Check existing tests in `tests/test_walk_forward.py`
- Review example in `examples/walk_forward_example.py`
- See main documentation in `docs/`
- Report issues on GitHub

---

**Last Updated:** 2025-11-15
**Maintainer:** Nexlify Development Team
