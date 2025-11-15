# Training UI Quick Start Guide

## Launch Training UI

```bash
python launch_training_ui.py
```

## 5-Minute Quick Start

### 1. Configure Training (Left Panel)

**Walk-Forward Settings:**
- Total Episodes: `2000` (default is fine)
- Train Size: `1000`
- Test Size: `200`
- Step Size: `200`
- Mode: `rolling`

**RL Agent Settings:**
- Learning Rate: `0.001` (default is fine)
- Architecture: `medium`

### 2. Start Training

Click **"Start Training"** button

### 3. Monitor Progress

Watch the **"Training Progress"** tab:
- Progress bar shows completion %
- Log shows detailed messages
- Training takes 20-60 minutes typically

### 4. View Results

Switch to **"Results & Metrics"** tab to see:
- Total Return
- Sharpe Ratio
- Win Rate
- Max Drawdown

Switch to **"Performance Charts"** tab for visualizations.

## Key Features

- ✅ **Real-time progress monitoring**
- ✅ **Walk-forward validation** (no data leakage!)
- ✅ **Comprehensive metrics** (Sharpe, Sortino, Calmar, etc.)
- ✅ **Visual performance charts**
- ✅ **Save/load configurations**
- ✅ **Training history tracking**
- ✅ **Cyberpunk-themed UI**

## Understanding Results

### Good Results
- Sharpe Ratio > 1.5
- Win Rate > 55%
- Positive Total Return
- Max Drawdown < 10%

### Results Stability
- Low standard deviations = consistent performance
- High standard deviations = unstable strategy

## Save Your Work

1. Click **"Save Configuration"** to save settings
2. Training results auto-saved to `training_logs/walk_forward_history.json`
3. Visual reports saved to `reports/walk_forward/`
4. Models saved to `models/walk_forward/`

## Common Adjustments

### Faster Training
- Reduce Total Episodes to `1000`
- Increase Step Size to `400`
- Use Architecture: `small` or `tiny`

### More Robust Validation
- Increase Total Episodes to `5000`
- Decrease Step Size to `100`
- Use Mode: `expanding`

### Better Performance
- Increase Train Size to `2000`
- Tune Learning Rate (try `0.0005` or `0.002`)
- Try larger Architecture: `large`

## Troubleshooting

### Training is slow
→ Reduce Total Episodes or increase Step Size

### Results are poor
→ Increase Train Size or try different Learning Rate

### UI won't start
→ Install requirements: `pip install PyQt5 matplotlib`

## Next Steps

1. **Read Full Guide**: See `docs/TRAINING_UI_GUIDE.md`
2. **Learn Walk-Forward**: See `docs/WALK_FORWARD_VALIDATION.md`
3. **Try Example**: Run `python examples/walk_forward_example.py`

---

**Need Help?** Check the detailed documentation in `/docs/` folder.
