# ML/RL Training UI Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-15
**Module:** `nexlify.gui.training_ui`

## Overview

The Nexlify Training UI provides a comprehensive graphical interface for training and validating reinforcement learning trading agents using walk-forward validation. It offers real-time monitoring, configuration management, performance visualization, and training history tracking.

## Features

### 1. **Walk-Forward Configuration**
- Total episodes slider (100-100,000)
- Training window size control
- Test window size control
- Step size adjustment
- Mode selection (rolling/expanding)
- Risk-free rate configuration
- Model saving options

### 2. **RL Agent Parameters**
- Learning rate adjustment
- Discount factor (gamma) tuning
- Batch size configuration
- Network architecture selection (tiny/small/medium/large/xlarge)

### 3. **Real-Time Monitoring**
- **Progress Tab**: Live training progress with detailed logging
- **Results Tab**: Comprehensive metrics display with color-coded values
- **Performance Charts**: Visual performance across folds
- **Training History**: Historical training run tracking

### 4. **Performance Metrics**
- Total Return (mean ± std)
- Sharpe Ratio
- Win Rate
- Max Drawdown
- Profit Factor
- Sortino Ratio

### 5. **Visualization**
- Returns per fold bar chart
- Key metrics comparison
- Fold-by-fold performance breakdown
- Cyberpunk-themed interface matching main GUI

## Installation

### Requirements

```bash
# Core requirements
pip install PyQt5 matplotlib numpy pandas

# For full functionality (RL training)
pip install torch

# Or install all requirements
pip install -r requirements.txt
```

## Quick Start

### Launch Training UI

```bash
# Method 1: Direct launch
python launch_training_ui.py

# Method 2: From Python
python -m nexlify.gui.training_ui

# Method 3: Programmatically
from nexlify.gui.training_ui import main
main()
```

### Basic Workflow

1. **Configure Parameters**
   - Set walk-forward validation parameters in left panel
   - Adjust RL agent hyperparameters as needed
   - Optionally load a saved configuration

2. **Start Training**
   - Click "Start Training" button
   - Monitor progress in Progress tab
   - View real-time log messages

3. **Review Results**
   - Switch to Results tab to see performance metrics
   - Check Performance Charts for visualizations
   - Review fold-by-fold breakdown

4. **Save Configuration**
   - Click "Save Configuration" to export settings
   - Load previous configurations for reproducibility

## User Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  Nexlify ML/RL Training Dashboard                       │
├───────────────┬─────────────────────────────────────────┤
│               │  ┌─ Training Progress ──┐               │
│  CONFIG       │  │  Progress Bar        │               │
│  PANEL        │  │  Status Label        │               │
│               │  │  Training Log        │               │
│  Walk-Forward │  └──────────────────────┘               │
│  Parameters   │                                          │
│               │  ┌─ Results & Metrics ──┐              │
│  RL Agent     │  │  Summary Metrics     │              │
│  Parameters   │  │  Fold Results        │              │
│               │  └──────────────────────┘              │
│  Control      │                                         │
│  Buttons      │  ┌─ Performance Charts ─┐              │
│               │  │  Returns Plot        │              │
│  [Start]      │  │  Metrics Plot        │              │
│  [Stop]       │  └──────────────────────┘              │
│  [Save]       │                                         │
│  [Load]       │  ┌─ Training History ───┐              │
│               │  │  Historical Runs     │              │
│               │  │  [Load] [Clear]      │              │
└───────────────┴─────────────────────────────────────────┘
```

## Configuration Parameters

### Walk-Forward Validation

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| **Total Episodes** | Total number of training episodes | 100-100,000 | 2,000 |
| **Train Size** | Episodes per training window | 100-50,000 | 1,000 |
| **Test Size** | Episodes per test window | 50-10,000 | 200 |
| **Step Size** | Forward step between folds | 50-5,000 | 200 |
| **Mode** | Rolling or expanding window | rolling/expanding | rolling |
| **Save Models** | Save model from each fold | checkbox | True |
| **Risk-Free Rate** | Annual risk-free rate | 0.0-0.1 | 0.02 |

### RL Agent Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| **Learning Rate** | Optimizer learning rate | 0.00001-0.1 | 0.001 |
| **Discount Factor** | Gamma for future rewards | 0.9-0.999 | 0.99 |
| **Batch Size** | Training batch size | 16-512 | 64 |
| **Architecture** | Network size | tiny/small/medium/large/xlarge | medium |

## Training Process

### 1. Initialization Phase
- UI loads configuration from `config/neural_config.json`
- Walk-forward folds are generated based on parameters
- Training environment is initialized

### 2. Training Phase
For each fold:
- **Training**: Agent trains on historical episodes
- **Evaluation**: Agent evaluated on future episodes (no exploration)
- **Metrics**: Performance metrics calculated and displayed
- **Progress**: UI updates with fold completion status

### 3. Completion Phase
- Best model selected based on validation metric (default: Sharpe ratio)
- Comprehensive report generated
- Results saved to training history
- Visual reports created

## Real-Time Updates

### Progress Bar
Shows overall training progress as percentage:
```
[████████████░░░░░░░░] 60% - Training fold 4/6
```

### Status Label
Displays current operation:
- "Training fold 2/5 (1000 episodes)"
- "Evaluating fold 2/5 (200 episodes)"
- "Training complete!"

### Log Output
Detailed training messages:
```
[14:23:45] Training started at 2025-11-15 14:23:45
[14:23:46] [0.0%] Initializing walk-forward validation
[14:24:12] [16.7%] Training fold 1/6 (1000 episodes)
[14:25:38] [25.0%] Fold 1 - Episode 500/1000: Avg Reward=125.34
[14:27:05] [33.3%] Evaluating fold 1/6 (200 episodes)
[14:27:42] [40.0%] Fold 1 evaluation: Return=12.5%, Sharpe=1.82
```

## Performance Metrics Explanation

### Color Coding
- **Green** (🟢): Positive values (profits, good performance)
- **Red** (🔴): Negative values (losses, drawdowns)
- **Purple** (🟣): Neutral/informational metrics

### Metric Definitions

**Total Return**: Cumulative return over test period
- Displayed as: `12.50% ± 2.30%` (mean ± std)
- Interpretation: Higher is better

**Sharpe Ratio**: Risk-adjusted return
- Displayed as: `1.82 ± 0.15`
- Interpretation: >1 is good, >2 is excellent

**Win Rate**: Percentage of profitable trades
- Displayed as: `65.00% ± 5.20%`
- Interpretation: >50% is profitable

**Max Drawdown**: Maximum peak-to-trough decline
- Displayed as: `-5.30% ± 1.20%`
- Interpretation: Smaller (closer to 0) is better

**Profit Factor**: Gross profits / gross losses
- Displayed as: `2.50 ± 0.30`
- Interpretation: >1 is profitable, >2 is excellent

**Sortino Ratio**: Downside risk-adjusted return
- Displayed as: `2.10 ± 0.25`
- Interpretation: Similar to Sharpe, focuses on downside

## Saving and Loading

### Save Configuration
1. Click "Save Configuration"
2. Choose location and filename
3. Configuration saved as JSON

Example saved configuration:
```json
{
  "walk_forward": {
    "total_episodes": 2000,
    "train_size": 1000,
    "test_size": 200,
    "step_size": 200,
    "mode": "rolling"
  },
  "rl_agent": {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "batch_size": 64
  }
}
```

### Load Configuration
1. Click "Load Configuration"
2. Select JSON configuration file
3. UI updates with loaded parameters

## Training History

### Automatic Saving
Training history is automatically saved to:
- `training_logs/walk_forward_history.json`

### History Display
Shows last 10 training runs:
```
Training Run 1:
  Time: 2025-11-15T14:30:22
  Folds: 6
  Mean Return: 12.50%
  Mean Sharpe: 1.82

Training Run 2:
  Time: 2025-11-15T16:45:18
  Folds: 8
  Mean Return: 15.30%
  Mean Sharpe: 2.05
```

### Load History
- Click "Load History" to load previous results
- Results populate in Results and Charts tabs
- Useful for comparing different configurations

## Troubleshooting

### UI Won't Launch

**Error**: `PyQt5 not installed`
```bash
pip install PyQt5
```

**Error**: `matplotlib not available`
```bash
pip install matplotlib
```

### Training Fails

**Error**: `Train size + test size exceeds total episodes`
- **Solution**: Reduce train_size or test_size, or increase total_episodes

**Error**: `PyTorch not available`
- **Solution**: Install PyTorch: `pip install torch`

**Error**: `Configuration file not found`
- **Solution**: Copy `config/neural_config.example.json` to `config/neural_config.json`

### Performance Issues

**Slow Training**
- Reduce total_episodes
- Increase step_size (fewer folds)
- Reduce batch_size
- Use smaller architecture (tiny/small)

**High Memory Usage**
- Disable model saving
- Reduce batch_size
- Use smaller architecture
- Close unnecessary programs

## Advanced Usage

### Programmatic Access

```python
from PyQt5.QtWidgets import QApplication
from nexlify.gui.training_ui import TrainingUI

app = QApplication([])
ui = TrainingUI()

# Customize configuration
ui.total_episodes_spin.setValue(5000)
ui.train_size_spin.setValue(2000)
ui.mode_combo.setCurrentText('expanding')

ui.show()
app.exec_()
```

### Custom Progress Callback

```python
def my_progress_callback(message: str, progress: float):
    print(f"[{progress:.1f}%] {message}")
    # Send to logging system, update external dashboard, etc.

# Pass to WalkForwardTrainer
from nexlify.training.walk_forward_trainer import WalkForwardTrainer

trainer = WalkForwardTrainer(
    config=config,
    progress_callback=my_progress_callback
)
```

### Integration with Existing Systems

```python
# Launch training UI from main application
from PyQt5.QtWidgets import QPushButton

training_btn = QPushButton("Open Training UI")
training_btn.clicked.connect(lambda: os.system('python launch_training_ui.py'))
```

## Tips and Best Practices

### 1. Start Small
- Begin with fewer episodes (500-1000)
- Use smaller architectures (tiny/small)
- Test configuration before full training

### 2. Monitor Closely
- Watch the Progress tab during first few folds
- Check if metrics are improving
- Look for unusual patterns in logs

### 3. Save Configurations
- Save working configurations for reproducibility
- Name files descriptively (e.g., `config_rolling_2k_episodes.json`)
- Version control your configurations

### 4. Compare Results
- Use Training History to compare runs
- Try different modes (rolling vs expanding)
- Experiment with different step sizes

### 5. Hardware Considerations
- **CPU**: Can train, but slower
- **GPU**: Much faster, enable if available
- **RAM**: Monitor usage, reduce batch size if needed
- **Disk**: Model saving can use significant space

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save Configuration |
| `Ctrl+O` | Load Configuration |
| `Ctrl+R` | Start Training |
| `Ctrl+T` | Stop Training |
| `Ctrl+Q` | Quit Application |

## Output Files

### Reports
Generated in `reports/walk_forward/`:
- `fold_comparison_YYYYMMDD_HHMMSS.png`
- `performance_stability_YYYYMMDD_HHMMSS.png`
- `metric_distributions_YYYYMMDD_HHMMSS.png`
- `validation_results_YYYYMMDD_HHMMSS.json`
- `summary_YYYYMMDD_HHMMSS.txt`

### Models
Saved in `models/walk_forward/`:
- `fold_0_model.pt`
- `fold_1_model.pt`
- ...
- `fold_N_model.pt`

### Training Logs
Saved in `training_logs/`:
- `walk_forward_history.json`

## FAQ

**Q: Can I run multiple training sessions simultaneously?**
A: Not recommended. Training is resource-intensive. Run one at a time.

**Q: How long does training take?**
A: Depends on configuration. Example: 2000 episodes, 6 folds ≈ 20-60 minutes.

**Q: Can I pause and resume training?**
A: Not currently supported. Training must complete or be stopped.

**Q: What happens if training fails mid-way?**
A: Completed folds are saved. You can review partial results in history.

**Q: Can I export results to CSV?**
A: Yes, load the JSON results file and convert using pandas.

**Q: How do I choose between rolling and expanding?**
A: Rolling for stable/changing markets, expanding for long-term trends.

## Support

For issues or questions:
- Check the main documentation in `docs/`
- Review training examples in `examples/`
- See walk-forward validation guide: `docs/WALK_FORWARD_VALIDATION.md`
- Report bugs on GitHub issues

## Future Enhancements

Planned features:
- Parallel fold processing
- Real-time Dash/Plotly dashboard integration
- Email/Telegram notifications on completion
- Hyperparameter optimization integration
- Export to TensorBoard
- Live comparison with baseline models

---

**Last Updated:** 2025-11-15
**Maintainer:** Nexlify Development Team
