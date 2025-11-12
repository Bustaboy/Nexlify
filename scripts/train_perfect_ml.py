#!/usr/bin/env python3
"""
Nexlify Perfect ML Training System
Complete training pipeline for the comprehensive ML ensemble

This script orchestrates:
- Feature engineering (100+ features)
- Data preparation and splitting
- Multiple model training (XGBoost, RF, LSTM, Transformer, Linear)
- Cross-validation
- Ensemble creation with intelligent weighting
- Model evaluation and comparison
- Hyperparameter optimization
- Model persistence
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import json
from typing import Tuple, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Nexlify modules
try:
    from nexlify.ml.nexlify_feature_engineering import FeatureEngineer
    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure Nexlify ML modules are properly installed")
    sys.exit(1)


def fetch_market_data(symbol: str = "BTC/USDT", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical market data

    Args:
        symbol: Trading pair
        days: Number of days

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"📊 Fetching {days} days of {symbol} data...")

    try:
        import ccxt

        exchange = ccxt.binance({'enableRateLimit': True})

        # Calculate timestamps
        now = exchange.milliseconds()
        since = now - (days * 24 * 60 * 60 * 1000)

        # Fetch OHLCV (1 hour candles)
        all_candles = []
        current_since = since

        while current_since < now:
            candles = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
            if not candles:
                break

            all_candles.extend(candles)
            current_since = candles[-1][0] + 1
            logger.info(f"  Fetched {len(all_candles)} candles...")

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        logger.info(f"✅ Fetched {len(df)} price points")
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Generating synthetic data...")
        return generate_synthetic_data(days)


def generate_synthetic_data(days: int = 365) -> pd.DataFrame:
    """Generate synthetic market data"""
    num_points = days * 24
    base_price = 30000

    # Random walk with trend
    returns = np.random.normal(0.0002, 0.02, num_points)
    trend = np.linspace(0, 0.5, num_points)  # 50% uptrend
    prices = base_price * np.exp(np.cumsum(returns) + trend)

    # Generate OHLCV
    timestamps = pd.date_range(end=datetime.now(), periods=num_points, freq='1H')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })

    # Generate open, high, low
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, num_points))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, num_points))
    df['volume'] = np.random.uniform(100, 1000, num_points)

    logger.info(f"✅ Generated {len(df)} synthetic price points")
    return df


def create_labels(df: pd.DataFrame, forward_periods: int = 24, threshold: float = 0.02) -> pd.Series:
    """
    Create trading labels (buy/sell/hold)

    Args:
        df: DataFrame with price data
        forward_periods: Look ahead periods
        threshold: Minimum return to trigger buy/sell (2%)

    Returns:
        Series with labels (0=sell, 1=hold, 2=buy)
    """
    logger.info(f"Creating labels (forward_periods={forward_periods}, threshold={threshold})...")

    future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)

    # 0 = Sell, 1 = Hold, 2 = Buy
    labels = pd.Series(1, index=df.index)  # Default: hold
    labels[future_returns > threshold] = 2  # Buy
    labels[future_returns < -threshold] = 0  # Sell

    # Remove last periods (no future data)
    labels.iloc[-forward_periods:] = np.nan

    label_counts = labels.value_counts()
    logger.info(f"Label distribution:")
    logger.info(f"  Sell (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(labels.dropna())*100:.1f}%)")
    logger.info(f"  Hold (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(labels.dropna())*100:.1f}%)")
    logger.info(f"  Buy (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(labels.dropna())*100:.1f}%)")

    return labels


def prepare_data(df: pd.DataFrame, labels: pd.Series, test_size: float = 0.2, val_size: float = 0.1) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Prepare train/val/test splits

    Args:
        df: DataFrame with features
        labels: Target labels
        test_size: Test set proportion
        val_size: Validation set proportion

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    logger.info("Preparing train/validation/test splits...")

    # Remove NaN labels
    valid_idx = labels.notna()
    df_clean = df[valid_idx].copy()
    labels_clean = labels[valid_idx].copy()

    # Get feature columns (exclude original OHLCV and timestamp)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    X = df_clean[feature_cols].values
    y = labels_clean.values

    # Time-series split (no shuffling)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val

    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]

    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]

    logger.info(f"✅ Data split:")
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Val:   {len(X_val)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")
    logger.info(f"   Features: {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_clean[feature_cols].columns


def train_perfect_ml(
    symbol: str = "BTC/USDT",
    days: int = 365,
    forward_periods: int = 24,
    threshold: float = 0.02,
    output_dir: str = "models/perfect_ml",
    hardware_adaptive: bool = True
):
    """
    Train the perfect ML system

    Args:
        symbol: Trading pair
        days: Days of historical data
        forward_periods: Prediction horizon
        threshold: Buy/sell threshold
        output_dir: Output directory
        hardware_adaptive: Use hardware adaptation
    """
    logger.info("=" * 70)
    logger.info("NEXLIFY PERFECT ML TRAINING SYSTEM")
    logger.info("=" * 70)

    start_time = datetime.now()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Fetch market data
    logger.info("\n📊 STEP 1: FETCHING MARKET DATA")
    logger.info("-" * 70)
    df = fetch_market_data(symbol, days)

    # Save raw data
    raw_data_path = output_path / "raw_data.csv"
    df.to_csv(raw_data_path, index=False)
    logger.info(f"💾 Raw data saved to {raw_data_path}")

    # 2. Feature engineering
    logger.info("\n🔧 STEP 2: FEATURE ENGINEERING")
    logger.info("-" * 70)
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)

    logger.info(f"✅ Generated {len(df_features.columns) - len(df.columns)} features")

    # Save engineered data
    features_path = output_path / "engineered_features.csv"
    df_features.to_csv(features_path, index=False)
    logger.info(f"💾 Features saved to {features_path}")

    # 3. Create labels
    logger.info("\n🏷️  STEP 3: CREATING LABELS")
    logger.info("-" * 70)
    labels = create_labels(df_features, forward_periods, threshold)

    # 4. Prepare data
    logger.info("\n📦 STEP 4: PREPARING DATA")
    logger.info("-" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(
        df_features, labels
    )

    # 5. Train ensemble
    logger.info("\n🚀 STEP 5: TRAINING ENSEMBLE")
    logger.info("-" * 70)

    ensemble = EnsembleMLSystem(task='classification', hardware_adaptive=hardware_adaptive)
    ensemble.train(X_train, y_train, X_val, y_val)

    # 6. Final evaluation on test set
    logger.info("\n📊 STEP 6: FINAL EVALUATION")
    logger.info("-" * 70)
    test_performance = ensemble.evaluate(X_test, y_test)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 70)
    for model_name, metrics in test_performance.items():
        logger.info(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    # 7. Feature importance
    logger.info("\n📊 STEP 7: FEATURE IMPORTANCE")
    logger.info("-" * 70)
    importance = ensemble.get_feature_importance()

    if importance:
        for model_name, importances in importance.items():
            # Get top 20 features
            indices = np.argsort(importances)[-20:]
            logger.info(f"\n{model_name.upper()} - Top 20 Features:")
            for i in indices[::-1]:
                logger.info(f"  {feature_names[i]}: {importances[i]:.4f}")

            # Save to file
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            importance_path = output_path / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(importance_path, index=False)

    # 8. Save models
    logger.info("\n💾 STEP 8: SAVING MODELS")
    logger.info("-" * 70)
    ensemble.save(str(output_path / "ensemble"))

    # 9. Save training summary
    logger.info("\n📝 STEP 9: GENERATING REPORT")
    logger.info("-" * 70)

    duration = datetime.now() - start_time

    summary = {
        'symbol': symbol,
        'training_date': datetime.now().isoformat(),
        'duration_seconds': duration.total_seconds(),
        'data': {
            'days': days,
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': len(feature_names)
        },
        'labels': {
            'forward_periods': forward_periods,
            'threshold': threshold,
            'distribution': {
                'sell': int((y_train == 0).sum()),
                'hold': int((y_train == 1).sum()),
                'buy': int((y_train == 2).sum())
            }
        },
        'models': {
            name: {
                'trained': True,
                'performance': test_performance.get(name, {})
            }
            for name in ensemble.models.keys()
        },
        'ensemble_weights': ensemble.model_weights,
        'hardware_config': ensemble.config
    }

    summary_path = output_path / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Training summary saved to {summary_path}")

    # 10. Generate visualization
    try:
        generate_training_visualizations(
            test_performance,
            ensemble.model_weights,
            output_path
        )
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration}")
    logger.info(f"Models trained: {len(ensemble.models)}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"\nBest performing model:")

    best_model = max(test_performance.items(), key=lambda x: x[1].get('score', 0))
    logger.info(f"  {best_model[0]}: {best_model[1].get('score', 0):.4f}")

    return ensemble, summary


def generate_training_visualizations(performance: Dict, weights: Dict, output_dir: Path):
    """Generate training visualization charts"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Model Performance
        models = list(performance.keys())
        scores = [perf.get('score', 0) for perf in performance.values()]

        axes[0].barh(models, scores)
        axes[0].set_xlabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Ensemble Weights
        weight_models = list(weights.keys())
        weight_values = list(weights.values())

        axes[1].pie(weight_values, labels=weight_models, autopct='%1.1f%%')
        axes[1].set_title('Ensemble Model Weights')

        plt.tight_layout()

        viz_path = output_dir / "training_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Visualization saved to {viz_path}")

    except ImportError:
        logger.warning("matplotlib not available - skipping visualizations")
    except Exception as e:
        logger.error(f"Visualization error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Nexlify Perfect ML System"
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of historical data (default: 365)'
    )

    parser.add_argument(
        '--forward-periods',
        type=int,
        default=24,
        help='Prediction horizon in hours (default: 24)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.02,
        help='Buy/sell threshold (default: 0.02 = 2%%)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/perfect_ml',
        help='Output directory (default: models/perfect_ml)'
    )

    parser.add_argument(
        '--no-hardware-adaptive',
        action='store_true',
        help='Disable hardware adaptation'
    )

    args = parser.parse_args()

    try:
        ensemble, summary = train_perfect_ml(
            symbol=args.symbol,
            days=args.days,
            forward_periods=args.forward_periods,
            threshold=args.threshold,
            output_dir=args.output_dir,
            hardware_adaptive=not args.no_hardware_adaptive
        )

        logger.info("\n✅ Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
