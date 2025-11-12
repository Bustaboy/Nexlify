#!/usr/bin/env python3
"""
Nexlify Perfect ML System - Usage Examples
Demonstrates the comprehensive ML ensemble system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def example_1_feature_engineering():
    """Example 1: Feature Engineering"""
    print("=" * 70)
    print("EXAMPLE 1: Advanced Feature Engineering")
    print("=" * 70)

    from nexlify.ml.nexlify_feature_engineering import FeatureEngineer

    # Create sample OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(29000, 31000, 1000),
        'high': np.random.uniform(30000, 32000, 1000),
        'low': np.random.uniform(28000, 30000, 1000),
        'close': np.random.uniform(29000, 31000, 1000),
        'volume': np.random.uniform(100, 1000, 1000)
    })

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)

    print(f"\n✅ Original columns: {len(df.columns)}")
    print(f"✅ After engineering: {len(df_features.columns)}")
    print(f"✅ New features: {len(df_features.columns) - len(df.columns)}")

    # Get feature groups
    groups = engineer.get_feature_importance_groups()
    print(f"\nFeature categories:")
    for category, features in groups.items():
        print(f"  {category}: {len(features)} features")


def example_2_ensemble_training():
    """Example 2: Ensemble Model Training"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Ensemble Model Training")
    print("=" * 70)

    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # Create synthetic training data
    X_train = np.random.randn(1000, 50)  # 1000 samples, 50 features
    y_train = np.random.randint(0, 3, 1000)  # 3 classes: sell/hold/buy

    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)

    # Create ensemble
    ensemble = EnsembleMLSystem(task='classification', hardware_adaptive=True)

    print("\nTraining ensemble models...")
    ensemble.train(X_train, y_train, X_val, y_val)

    print(f"\n✅ Models trained: {list(ensemble.models.keys())}")
    print(f"\nModel weights:")
    for name, weight in ensemble.model_weights.items():
        print(f"  {name}: {weight:.4f}")


def example_3_prediction():
    """Example 3: Making Predictions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Making Predictions")
    print("=" * 70)

    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # Create and train ensemble
    X_train = np.random.randn(500, 30)
    y_train = np.random.randint(0, 3, 500)

    ensemble = EnsembleMLSystem(task='classification', hardware_adaptive=False)
    ensemble.build_models(X_train, y_train)

    # Make predictions
    X_test = np.random.randn(10, 30)

    print("\nMaking predictions with different methods...")

    # Weighted ensemble
    predictions_weighted = ensemble.predict(X_test, method='weighted')
    print(f"\nWeighted ensemble predictions: {predictions_weighted}")

    # Simple voting
    predictions_voting = ensemble.predict(X_test, method='voting')
    print(f"Voting ensemble predictions: {predictions_voting}")


def example_4_feature_importance():
    """Example 4: Feature Importance Analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Feature Importance")
    print("=" * 70)

    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # Train models
    X_train = np.random.randn(500, 20)
    y_train = np.random.randint(0, 3, 500)

    ensemble = EnsembleMLSystem(task='classification', hardware_adaptive=False)
    ensemble.build_models(X_train, y_train)

    # Get feature importance
    importance = ensemble.get_feature_importance()

    print("\nFeature importance from tree models:")
    for model_name, importances in importance.items():
        print(f"\n{model_name}:")
        # Top 5 features
        top_indices = np.argsort(importances)[-5:]
        for i in top_indices[::-1]:
            print(f"  Feature {i}: {importances[i]:.4f}")


def example_5_save_load():
    """Example 5: Save and Load Models"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Save/Load Ensemble")
    print("=" * 70)

    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # Train ensemble
    X_train = np.random.randn(300, 15)
    y_train = np.random.randint(0, 3, 300)

    ensemble = EnsembleMLSystem(task='classification')
    ensemble.build_models(X_train, y_train)

    # Save
    save_dir = "models/example_ensemble"
    ensemble.save(save_dir)
    print(f"\n✅ Ensemble saved to {save_dir}")

    # Load
    new_ensemble = EnsembleMLSystem(task='classification')
    new_ensemble.load(save_dir)
    print(f"✅ Ensemble loaded from {save_dir}")

    # Verify
    X_test = np.random.randn(5, 15)
    predictions = new_ensemble.predict(X_test)
    print(f"\nPredictions from loaded model: {predictions}")


def example_6_complete_pipeline():
    """Example 6: Complete ML Pipeline"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Complete ML Pipeline")
    print("=" * 70)

    from nexlify.ml.nexlify_feature_engineering import FeatureEngineer
    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # 1. Create sample data
    dates = pd.date_range(end=datetime.now(), periods=2000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.cumsum(np.random.randn(2000)) + 30000,
        'high': np.cumsum(np.random.randn(2000)) + 30100,
        'low': np.cumsum(np.random.randn(2000)) + 29900,
        'close': np.cumsum(np.random.randn(2000)) + 30000,
        'volume': np.random.uniform(100, 1000, 2000)
    })

    print("\n1️⃣  Feature Engineering...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)

    # 2. Create labels (buy if price goes up > 2% in next 24h)
    print("2️⃣  Creating labels...")
    future_return = df_features['close'].pct_change(24).shift(-24)
    labels = pd.Series(1, index=df.index)  # Default: hold
    labels[future_return > 0.02] = 2  # Buy
    labels[future_return < -0.02] = 0  # Sell

    # 3. Prepare data
    print("3️⃣  Preparing data...")
    valid_idx = labels.notna()
    feature_cols = [col for col in df_features.columns
                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    X = df_features.loc[valid_idx, feature_cols].values
    y = labels[valid_idx].values

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")

    # 4. Train ensemble
    print("4️⃣  Training ensemble...")
    ensemble = EnsembleMLSystem(task='classification', hardware_adaptive=True)
    ensemble.train(X_train, y_train, X_test[:100], y_test[:100])

    # 5. Evaluate
    print("5️⃣  Evaluating...")
    performance = ensemble.evaluate(X_test, y_test)

    print("\n📊 Results:")
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n✅ Complete pipeline finished!")


def example_7_regression_task():
    """Example 7: Regression Task (Price Prediction)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Regression Task")
    print("=" * 70)

    from nexlify.ml.nexlify_ensemble_ml import EnsembleMLSystem

    # Create regression data (predict next price)
    X_train = np.random.randn(1000, 30)
    y_train = np.random.randn(1000) * 1000 + 30000  # Prices around 30000

    X_test = np.random.randn(100, 30)
    y_test = np.random.randn(100) * 1000 + 30000

    # Train regression ensemble
    print("\nTraining regression models...")
    ensemble = EnsembleMLSystem(task='regression', hardware_adaptive=True)
    ensemble.train(X_train, y_train, X_test, y_test)

    # Make predictions
    predictions = ensemble.predict(X_test[:10])

    print(f"\n📈 Price Predictions:")
    for i, (pred, actual) in enumerate(zip(predictions[:10], y_test[:10])):
        error = abs(pred - actual)
        print(f"  Sample {i+1}: Predicted=${pred:.2f}, Actual=${actual:.2f}, Error=${error:.2f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("NEXLIFY PERFECT ML SYSTEM - EXAMPLES")
    print("=" * 70)

    try:
        example_1_feature_engineering()
        example_2_ensemble_training()
        example_3_prediction()
        example_4_feature_importance()
        example_5_save_load()
        example_6_complete_pipeline()
        example_7_regression_task()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED")
        print("=" * 70)
        print("\nTo train your own model:")
        print("  python scripts/train_perfect_ml.py --symbol BTC/USDT --days 365")
        print("\nFor help:")
        print("  python scripts/train_perfect_ml.py --help")
        print()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
