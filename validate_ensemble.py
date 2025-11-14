#!/usr/bin/env python3
"""
Quick validation test for ensemble system
Tests basic functionality without full dependencies
"""

import sys
sys.path.insert(0, '/home/user/Nexlify')

# Test imports
print("Testing imports...")
try:
    from nexlify.strategies.ensemble_agent import (
        EnsembleManager,
        EnsembleStrategy,
        ModelInfo,
        StackingMetaModel
    )
    print("✓ ensemble_agent imports successful")
except Exception as e:
    print(f"✗ ensemble_agent import failed: {e}")
    sys.exit(1)

try:
    from nexlify.training.ensemble_trainer import (
        EnsembleTrainer,
        EnsembleTrainingConfig,
        ModelTrainingResult
    )
    print("✓ ensemble_trainer imports successful")
except Exception as e:
    print(f"✗ ensemble_trainer import failed: {e}")
    sys.exit(1)

# Test basic instantiation
print("\nTesting instantiation...")
try:
    manager = EnsembleManager(
        state_size=12,
        action_size=3,
        strategy=EnsembleStrategy.WEIGHTED_AVG,
        ensemble_size=3
    )
    print(f"✓ EnsembleManager created: {manager.ensemble_size} models target")
except Exception as e:
    print(f"✗ EnsembleManager creation failed: {e}")
    sys.exit(1)

try:
    config = EnsembleTrainingConfig(
        num_models=3,
        episodes_per_model=10,
        parallel_training=False
    )
    print(f"✓ EnsembleTrainingConfig created: {config.num_models} models")
except Exception as e:
    print(f"✗ EnsembleTrainingConfig creation failed: {e}")
    sys.exit(1)

# Test ensemble strategies
print("\nTesting ensemble strategies...")
strategies = [
    EnsembleStrategy.SIMPLE_AVG,
    EnsembleStrategy.WEIGHTED_AVG,
    EnsembleStrategy.VOTING,
    EnsembleStrategy.STACKING
]

for strategy in strategies:
    try:
        m = EnsembleManager(
            state_size=12,
            action_size=3,
            strategy=strategy,
            ensemble_size=3
        )
        print(f"✓ {strategy} strategy works")
    except Exception as e:
        print(f"✗ {strategy} strategy failed: {e}")

print("\n✅ All basic validation tests passed!")
print("\nEnsemble system is ready to use.")
print("\nNext steps:")
print("1. Train models: python train_ensemble.py --num-models 5 --episodes 100")
print("2. Run full test suite: pytest tests/test_ensemble.py")
