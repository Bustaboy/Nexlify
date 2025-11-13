# Setup on Your Laptop - Quick Reference

## What's Already Done ✅

1. **Pre-training validation script created**: `pre_training_checklist.py`
   - Checks Python version, dependencies, directories, disk space, GPU, internet, APIs, database
   - Provides color-coded output (green=pass, yellow=warn, red=fail)

2. **Complete setup guide created**: `PRE_TRAINING_QUICK_START.md`
   - Step-by-step instructions
   - Training script options
   - Troubleshooting guide
   - Safety checklist

3. **Files committed and pushed** to branch: `claude/pre-training-setup-checks-011CV5hT3UQ8TxK9RbndyEqS`

## What You Need to Do on Your Laptop

### Step 1: Pull the Changes
```bash
cd Nexlify
git fetch origin
git checkout claude/pre-training-setup-checks-011CV5hT3UQ8TxK9RbndyEqS
```

### Step 2: Install Dependencies
```bash
# This should work cleanly on your laptop
pip install -r requirements.txt
```

**Expected time**: 5-10 minutes (downloading ~2.5GB of ML libraries)

**Key packages being installed:**
- PyTorch (670 MB)
- TensorFlow (524 MB)
- XGBoost (297 MB)
- NVIDIA CUDA libraries (~2 GB combined, if you have GPU)
- ccxt, pandas, numpy, scikit-learn, etc.

**If you encounter issues:**
- Try: `pip install --user -r requirements.txt` (installs in user directory)
- Or: Use a virtual environment (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Step 3: Validate Everything is Ready
```bash
python pre_training_checklist.py
```

This will check everything and give you a clear report of what's working and what needs attention.

### Step 4: Run Training (if validation passes)
```bash
# Recommended: Most comprehensive training system
python train_ultimate_full_pipeline.py
```

## Known Issues from Cloud Environment

During setup, we encountered these issues that **should NOT affect your laptop**:

1. ❌ **Debian system package conflicts** (pyparsing, setuptools)
   - Your laptop won't have these conflicts
   - Fresh pip install should work fine

2. ❌ **Old setuptools version** (68.1.2 → needed 80.9.0)
   - Your laptop likely has newer setuptools already

3. ⚠️ **Large downloads** (~2.5+ GB total)
   - Plan for this, especially on slower internet
   - Downloads are one-time only

## System Requirements

**Minimum:**
- Python 3.9+ (you have 3.11.14, perfect!)
- 4GB RAM (8GB+ recommended)
- 2GB free disk (5GB+ recommended)
- Internet connection (for initial setup)

**Optional but Recommended:**
- NVIDIA GPU with CUDA support (10-50x faster training)
- 8GB+ RAM for large datasets
- 10GB+ disk for extensive caching

## Quick Validation Checklist

After installing dependencies, check:
- [ ] `python pre_training_checklist.py` shows all green ✓
- [ ] Sample data exists: `data/sample_datasets/btc_usdt_raw.csv`
- [ ] Training scripts are present: `train_ultimate_full_pipeline.py`
- [ ] Database directory exists: `data/`
- [ ] Config directory exists: `config/`

## If Something Doesn't Work

1. **Check the validation output**: `python pre_training_checklist.py`
2. **Review the guide**: `cat PRE_TRAINING_QUICK_START.md`
3. **Check logs**: `tail -f logs/*.log` (during training)
4. **Run preflight checker**: `python nexlify_preflight_checker.py --automated`

## Next Steps After Successful Setup

```bash
# 1. Validate (30 seconds)
python pre_training_checklist.py

# 2. Train (30-60 minutes depending on system)
python train_ultimate_full_pipeline.py

# 3. Test with paper trading (optional)
python run_paper_trading.py

# 4. Launch GUI (if satisfied)
python scripts/nexlify_launcher.py
```

## Important Notes

- ⚠️ **Never skip paper trading** before live trading!
- 📊 Sample data is already included in `data/sample_datasets/`
- 🔑 API keys are NOT required for training (only for live trading)
- 💾 Models are saved in `models/` after training
- 📝 Logs are saved in `logs/` for debugging

## Files Created for You

1. **`pre_training_checklist.py`** - Comprehensive validation script
2. **`PRE_TRAINING_QUICK_START.md`** - Complete setup and usage guide
3. **`SETUP_ON_LAPTOP.md`** - This file (quick reference)

## Estimated Time

- **Git pull**: < 1 minute
- **Dependency installation**: 5-10 minutes
- **Validation**: 30 seconds
- **First training run**: 30-60 minutes (varies by system)

**Total setup time**: ~15-20 minutes (mostly waiting for downloads)

---

**Good luck! The setup should be straightforward on your laptop. 🚀**
