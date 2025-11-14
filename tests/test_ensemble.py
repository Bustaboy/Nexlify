#!/usr/bin/env python3
"""
Unit Tests for Ensemble System

Tests for EnsembleManager and EnsembleTrainer
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nexlify.strategies.ensemble_agent import (EnsembleManager,
                                               EnsembleStrategy, ModelInfo,
                                               StackingMetaModel,
                                               create_ensemble)
from nexlify.strategies.nexlify_rl_agent import DQNAgent
from nexlify.training.ensemble_trainer import (EnsembleTrainer,
                                               EnsembleTrainingConfig,
                                               train_ensemble)


@pytest.fixture
def state_size():
    return 12


@pytest.fixture
def action_size():
    return 3


@pytest.fixture
def test_config():
    return {
        'gamma': 0.99,
        'epsilon': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32
    }


@pytest.fixture
def ensemble_manager(state_size, action_size, test_config):
    """Create ensemble manager for testing"""
    return EnsembleManager(
        state_size=state_size,
        action_size=action_size,
        strategy=EnsembleStrategy.WEIGHTED_AVG,
        ensemble_size=3,
        config=test_config
    )


@pytest.fixture
def trained_models(state_size, action_size, test_config, tmp_path):
    """Create and save multiple trained models"""
    models = []

    for i in range(3):
        # Create agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            config=test_config
        )

        # Save model
        model_path = tmp_path / f"model_{i}.pt"
        agent.save(str(model_path))

        models.append((str(model_path), np.random.uniform(0.5, 1.0)))  # (path, score)

    return models


@pytest.fixture
def sample_training_data():
    """Generate sample training data"""
    np.random.seed(42)
    num_steps = 1000
    base_price = 50000
    returns = np.random.normal(0.0001, 0.02, num_steps)
    prices = base_price * np.exp(np.cumsum(returns))
    return np.abs(prices)


@pytest.fixture
def sample_validation_data():
    """Generate sample validation data"""
    np.random.seed(123)
    num_steps = 200
    base_price = 50000
    returns = np.random.normal(0.0001, 0.02, num_steps)
    prices = base_price * np.exp(np.cumsum(returns))
    return np.abs(prices)


# =============================================================================
# EnsembleManager Tests
# =============================================================================

@pytest.mark.unit
def test_ensemble_manager_initialization(state_size, action_size, test_config):
    """Test ensemble manager initialization"""
    manager = EnsembleManager(
        state_size=state_size,
        action_size=action_size,
        strategy=EnsembleStrategy.SIMPLE_AVG,
        ensemble_size=5,
        diversity_penalty=0.2,
        config=test_config
    )

    assert manager.state_size == state_size
    assert manager.action_size == action_size
    assert manager.strategy == EnsembleStrategy.SIMPLE_AVG
    assert manager.ensemble_size == 5
    assert manager.diversity_penalty == 0.2
    assert len(manager.models) == 0


@pytest.mark.unit
def test_add_model(ensemble_manager, trained_models):
    """Test adding models to ensemble"""
    model_path, validation_score = trained_models[0]

    success = ensemble_manager.add_model(
        model_path=model_path,
        validation_score=validation_score
    )

    assert success is True
    assert len(ensemble_manager.models) == 1
    assert ensemble_manager.models[0].model_path == model_path
    assert ensemble_manager.models[0].validation_score == validation_score


@pytest.mark.unit
def test_add_multiple_models(ensemble_manager, trained_models):
    """Test adding multiple models"""
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(
            model_path=model_path,
            validation_score=validation_score
        )

    assert len(ensemble_manager.models) == len(trained_models)


@pytest.mark.unit
def test_weight_normalization(ensemble_manager, trained_models):
    """Test that weights are normalized"""
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(
            model_path=model_path,
            validation_score=validation_score
        )

    # Sum of weights should be 1.0
    total_weight = sum(m.weight for m in ensemble_manager.models)
    assert abs(total_weight - 1.0) < 1e-6


@pytest.mark.unit
def test_load_ensemble_from_directory(ensemble_manager, trained_models, tmp_path):
    """Test loading models from directory"""
    # Models are already saved in tmp_path by fixture
    validation_scores = {
        f"model_{i}": score
        for i, (_, score) in enumerate(trained_models)
    }

    loaded_count = ensemble_manager.load_ensemble_from_directory(
        models_dir=str(tmp_path),
        validation_scores=validation_scores
    )

    assert loaded_count == len(trained_models)
    assert len(ensemble_manager.models) == len(trained_models)


@pytest.mark.unit
def test_simple_avg_prediction(ensemble_manager, trained_models, state_size):
    """Test simple averaging prediction"""
    # Set strategy
    ensemble_manager.strategy = EnsembleStrategy.SIMPLE_AVG

    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Make prediction
    state = np.random.randn(state_size)
    action, uncertainty = ensemble_manager.predict(state, return_uncertainty=True)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < ensemble_manager.action_size
    assert isinstance(uncertainty, (float, np.floating))
    assert uncertainty >= 0


@pytest.mark.unit
def test_weighted_avg_prediction(ensemble_manager, trained_models, state_size):
    """Test weighted averaging prediction"""
    # Set strategy
    ensemble_manager.strategy = EnsembleStrategy.WEIGHTED_AVG

    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Make prediction
    state = np.random.randn(state_size)
    action, uncertainty = ensemble_manager.predict(state, return_uncertainty=True)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < ensemble_manager.action_size
    assert isinstance(uncertainty, (float, np.floating))
    assert uncertainty >= 0


@pytest.mark.unit
def test_voting_prediction(ensemble_manager, trained_models, state_size):
    """Test voting prediction"""
    # Set strategy
    ensemble_manager.strategy = EnsembleStrategy.VOTING

    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Make prediction
    state = np.random.randn(state_size)
    action, uncertainty = ensemble_manager.predict(state, return_uncertainty=True)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action < ensemble_manager.action_size
    assert isinstance(uncertainty, (float, np.floating))
    assert 0 <= uncertainty <= 1  # Disagreement is between 0 and 1


@pytest.mark.unit
def test_remove_model(ensemble_manager, trained_models):
    """Test removing model from ensemble"""
    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    initial_count = len(ensemble_manager.models)

    # Remove first model
    success = ensemble_manager.remove_model(0)

    assert success is True
    assert len(ensemble_manager.models) == initial_count - 1


@pytest.mark.unit
def test_get_ensemble_info(ensemble_manager, trained_models):
    """Test getting ensemble information"""
    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    info = ensemble_manager.get_ensemble_info()

    assert info['num_models'] == len(trained_models)
    assert info['strategy'] == ensemble_manager.strategy
    assert info['ensemble_size'] == ensemble_manager.ensemble_size
    assert 'models' in info
    assert 'stats' in info


@pytest.mark.unit
def test_save_load_ensemble_config(ensemble_manager, trained_models, tmp_path):
    """Test saving and loading ensemble configuration"""
    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Save config
    config_path = tmp_path / "ensemble_config.json"
    ensemble_manager.save_ensemble_config(str(config_path))

    assert config_path.exists()

    # Load config
    new_manager = EnsembleManager(
        state_size=ensemble_manager.state_size,
        action_size=ensemble_manager.action_size,
        strategy=ensemble_manager.strategy,
        ensemble_size=ensemble_manager.ensemble_size
    )

    new_manager.load_ensemble_config(str(config_path))

    assert len(new_manager.models) == len(ensemble_manager.models)


@pytest.mark.unit
def test_stacking_meta_model(state_size, action_size):
    """Test stacking meta-model"""
    num_models = 3
    meta_model = StackingMetaModel(
        num_models=num_models,
        action_size=action_size,
        hidden_size=64
    )

    # Test forward pass
    batch_size = 4
    stacked_input = torch.randn(batch_size, num_models * action_size)
    output = meta_model(stacked_input)

    assert output.shape == (batch_size, action_size)


@pytest.mark.unit
def test_get_statistics(ensemble_manager, trained_models, state_size):
    """Test getting ensemble statistics"""
    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Make some predictions to generate statistics
    for _ in range(10):
        state = np.random.randn(state_size)
        ensemble_manager.predict(state, return_uncertainty=True)

    stats = ensemble_manager.get_statistics()

    assert 'total_predictions' in stats
    assert stats['total_predictions'] >= 10
    assert 'num_models' in stats
    assert stats['num_models'] == len(trained_models)


# =============================================================================
# EnsembleTrainer Tests
# =============================================================================

@pytest.mark.unit
def test_ensemble_training_config():
    """Test ensemble training configuration"""
    config = EnsembleTrainingConfig(
        num_models=5,
        episodes_per_model=100,
        parallel_training=False,
        seed_start=42
    )

    assert config.num_models == 5
    assert config.episodes_per_model == 100
    assert config.parallel_training is False
    assert config.seed_start == 42


@pytest.mark.unit
def test_ensemble_trainer_initialization(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test ensemble trainer initialization"""
    config = EnsembleTrainingConfig(
        num_models=3,
        episodes_per_model=10,
        parallel_training=False,
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    assert trainer.state_size == state_size
    assert trainer.action_size == action_size
    assert len(trainer.training_results) == 0


@pytest.mark.integration
def test_train_single_model(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test training a single model"""
    config = EnsembleTrainingConfig(
        num_models=1,
        episodes_per_model=10,  # Short training for testing
        parallel_training=False,
        validation_episodes=10,  # Fewer validation episodes for testing
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    results = trainer.train_ensemble()

    assert len(results) == 1
    assert results[0].model_id == 0
    assert Path(results[0].model_path).exists()
    # Validation score might be 0 with minimal training, just check it's a number
    assert isinstance(results[0].validation_score, (int, float))


@pytest.mark.integration
def test_train_ensemble_sequential(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test training ensemble sequentially"""
    config = EnsembleTrainingConfig(
        num_models=2,
        episodes_per_model=10,  # Short training for testing
        parallel_training=False,
        validation_episodes=10,  # Fewer validation episodes for testing
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    results = trainer.train_ensemble()

    assert len(results) == 2
    assert all(Path(r.model_path).exists() for r in results)

    # Check that models have different seeds
    assert results[0].seed != results[1].seed


@pytest.mark.integration
def test_get_best_models(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test getting best models from ensemble"""
    config = EnsembleTrainingConfig(
        num_models=3,
        episodes_per_model=10,
        parallel_training=False,
        validation_episodes=10,  # Fewer validation episodes for testing
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    trainer.train_ensemble()

    best_models = trainer.get_best_models(top_k=2)

    assert len(best_models) == 2

    # Check that they're sorted by validation score
    assert best_models[0].validation_score >= best_models[1].validation_score


@pytest.mark.integration
def test_compare_models(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test model comparison report"""
    config = EnsembleTrainingConfig(
        num_models=2,
        episodes_per_model=10,
        parallel_training=False,
        validation_episodes=10,  # Fewer validation episodes for testing
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    trainer.train_ensemble()

    report = trainer.compare_models()

    assert isinstance(report, str)
    assert "MODEL COMPARISON" in report
    assert "Val Score" in report


@pytest.mark.integration
def test_ensemble_summary_saved(
    state_size,
    action_size,
    sample_training_data,
    sample_validation_data,
    tmp_path
):
    """Test that ensemble summary is saved"""
    config = EnsembleTrainingConfig(
        num_models=2,
        episodes_per_model=10,
        parallel_training=False,
        validation_episodes=10,  # Fewer validation episodes for testing
        output_dir=str(tmp_path)
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=sample_training_data,
        validation_data=sample_validation_data,
        config=config
    )

    trainer.train_ensemble()

    # Check that summary file exists
    summary_path = tmp_path / "ensemble_summary.json"
    assert summary_path.exists()

    # Load and validate summary
    with open(summary_path) as f:
        summary = json.load(f)

    assert 'training_config' in summary
    assert 'models' in summary
    assert 'statistics' in summary
    assert len(summary['models']) == 2


@pytest.mark.integration
def test_create_ensemble_convenience_function(
    state_size,
    action_size,
    trained_models,
    tmp_path
):
    """Test create_ensemble convenience function"""
    validation_scores = {
        f"model_{i}": score
        for i, (_, score) in enumerate(trained_models)
    }

    ensemble = create_ensemble(
        state_size=state_size,
        action_size=action_size,
        models_dir=str(tmp_path),
        strategy=EnsembleStrategy.WEIGHTED_AVG,
        ensemble_size=3,
        validation_scores=validation_scores
    )

    assert isinstance(ensemble, EnsembleManager)
    assert len(ensemble.models) == len(trained_models)


@pytest.mark.unit
def test_model_info_serialization():
    """Test ModelInfo serialization"""
    # Create a dummy agent
    agent = DQNAgent(state_size=12, action_size=3)

    model_info = ModelInfo(
        model_path="/path/to/model.pt",
        agent=agent,
        validation_score=0.75,
        weight=0.5
    )

    # Convert to dict
    info_dict = model_info.to_dict()

    assert info_dict['model_path'] == "/path/to/model.pt"
    assert info_dict['validation_score'] == 0.75
    assert info_dict['weight'] == 0.5
    assert 'prediction_count' in info_dict


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.integration
def test_ensemble_prediction_performance(
    ensemble_manager,
    trained_models,
    state_size
):
    """Test that ensemble predictions are reasonably fast"""
    import time

    # Add models
    for model_path, validation_score in trained_models:
        ensemble_manager.add_model(model_path, validation_score)

    # Time predictions
    state = np.random.randn(state_size)
    num_predictions = 100

    start_time = time.time()
    for _ in range(num_predictions):
        ensemble_manager.predict(state, return_uncertainty=True)
    elapsed_time = time.time() - start_time

    avg_time = elapsed_time / num_predictions

    # Should be fast (< 100ms per prediction)
    assert avg_time < 0.1, f"Predictions too slow: {avg_time*1000:.2f}ms per prediction"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.unit
def test_predict_without_models(ensemble_manager, state_size):
    """Test that prediction fails gracefully without models"""
    state = np.random.randn(state_size)

    with pytest.raises(ValueError, match="No models in ensemble"):
        ensemble_manager.predict(state)


@pytest.mark.unit
def test_add_invalid_model_path(ensemble_manager):
    """Test adding model with invalid path"""
    success = ensemble_manager.add_model(
        model_path="/nonexistent/model.pt",
        validation_score=0.5
    )

    assert success is False
    assert len(ensemble_manager.models) == 0


@pytest.mark.unit
def test_remove_invalid_index(ensemble_manager, trained_models):
    """Test removing model with invalid index"""
    # Add one model
    ensemble_manager.add_model(trained_models[0][0], trained_models[0][1])

    # Try to remove invalid index
    success = ensemble_manager.remove_model(999)

    assert success is False
    assert len(ensemble_manager.models) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
