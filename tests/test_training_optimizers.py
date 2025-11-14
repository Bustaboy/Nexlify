#!/usr/bin/env python3
"""
Comprehensive tests for training optimization utilities

Tests:
- GradientClipper (clipping, monitoring, statistics)
- LRSchedulerManager (cosine, plateau, step, auto)
- LRWarmup
- TrainingOptimizer (integrated)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from nexlify.training.training_optimizers import (
    GradientClipper,
    LRSchedulerManager,
    LRWarmup,
    SchedulerType,
    TrainingOptimizer,
    TrainingPhase,
)


@pytest.fixture
def simple_model():
    """Create a simple test model"""
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleNet()


@pytest.fixture
def optimizer_for_model(simple_model):
    """Create optimizer for test model"""
    return optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def sample_batch():
    """Create sample training batch"""
    return {
        'input': torch.randn(32, 10),
        'target': torch.randint(0, 5, (32,)),
    }


# ============================================================================
# GradientClipper Tests
# ============================================================================

@pytest.mark.unit
def test_gradient_clipper_initialization():
    """Test GradientClipper initialization"""
    clipper = GradientClipper(
        max_norm=1.0,
        explosion_threshold=10.0,
        vanishing_threshold=1e-6,
    )

    assert clipper.max_norm == 1.0
    assert clipper.explosion_threshold == 10.0
    assert clipper.vanishing_threshold == 1e-6
    assert clipper.step_count == 0
    assert clipper.clip_count == 0


@pytest.mark.unit
def test_gradient_clipper_no_clipping_needed(simple_model, optimizer_for_model, sample_batch):
    """Test gradient clipping when gradients are small"""
    clipper = GradientClipper(max_norm=10.0)  # High threshold
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = simple_model(sample_batch['input'])
    loss = criterion(output, sample_batch['target'])

    # Backward pass
    optimizer_for_model.zero_grad()
    loss.backward()

    # Clip gradients
    norm_before, norm_after, was_clipped = clipper.clip_gradients(simple_model)

    assert norm_before > 0
    assert norm_after > 0
    assert not was_clipped  # Should not clip with high threshold
    assert clipper.step_count == 1
    assert clipper.clip_count == 0


@pytest.mark.unit
def test_gradient_clipper_with_clipping(simple_model, optimizer_for_model, sample_batch):
    """Test gradient clipping when gradients exceed threshold"""
    clipper = GradientClipper(max_norm=0.01)  # Very low threshold
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = simple_model(sample_batch['input'])
    loss = criterion(output, sample_batch['target'])

    # Backward pass
    optimizer_for_model.zero_grad()
    loss.backward()

    # Clip gradients
    norm_before, norm_after, was_clipped = clipper.clip_gradients(simple_model)

    assert norm_before > 0
    assert norm_after > 0
    assert was_clipped  # Should clip with low threshold
    assert norm_after <= clipper.max_norm * 1.01  # Allow small numerical error
    assert clipper.step_count == 1
    assert clipper.clip_count == 1


@pytest.mark.unit
def test_gradient_clipper_statistics(simple_model, optimizer_for_model, sample_batch):
    """Test gradient statistics tracking"""
    clipper = GradientClipper(max_norm=1.0)
    criterion = nn.CrossEntropyLoss()

    # Run multiple training steps
    for _ in range(10):
        output = simple_model(sample_batch['input'])
        loss = criterion(output, sample_batch['target'])
        optimizer_for_model.zero_grad()
        loss.backward()
        clipper.clip_gradients(simple_model)
        optimizer_for_model.step()

    # Check statistics
    stats = clipper.get_statistics()

    assert stats['steps'] == 10
    assert 'clip_rate' in stats
    assert 'norm_before' in stats
    assert 'norm_after' in stats
    assert stats['norm_before']['mean'] > 0
    assert stats['norm_after']['mean'] > 0


@pytest.mark.unit
def test_gradient_clipper_save_history(simple_model, optimizer_for_model, sample_batch):
    """Test saving gradient history to file"""
    clipper = GradientClipper(max_norm=1.0)
    criterion = nn.CrossEntropyLoss()

    # Run a few training steps
    for _ in range(5):
        output = simple_model(sample_batch['input'])
        loss = criterion(output, sample_batch['target'])
        optimizer_for_model.zero_grad()
        loss.backward()
        clipper.clip_gradients(simple_model)
        optimizer_for_model.step()

    # Save history
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "gradient_history.json"
        clipper.save_history(str(filepath))

        assert filepath.exists()

        # Load and verify
        with open(filepath) as f:
            history = json.load(f)

        assert 'config' in history
        assert 'statistics' in history
        assert 'history' in history
        assert history['config']['max_norm'] == 1.0


# ============================================================================
# LRWarmup Tests
# ============================================================================

@pytest.mark.unit
def test_lr_warmup_initialization(optimizer_for_model):
    """Test LRWarmup initialization"""
    warmup = LRWarmup(
        optimizer=optimizer_for_model,
        warmup_steps=100,
        lr_start=1e-5,
        lr_target=0.001,
    )

    assert warmup.warmup_steps == 100
    assert warmup.lr_start == 1e-5
    assert warmup.lr_target == 0.001
    assert warmup.current_step == 0
    assert not warmup.is_complete()

    # Check that optimizer LR was set to start value
    assert optimizer_for_model.param_groups[0]['lr'] == 1e-5


@pytest.mark.unit
def test_lr_warmup_progression(optimizer_for_model):
    """Test LR warmup progression"""
    warmup = LRWarmup(
        optimizer=optimizer_for_model,
        warmup_steps=10,
        lr_start=1e-5,
        lr_target=1e-3,
    )

    lrs = []
    for _ in range(15):
        lr = warmup.step()
        lrs.append(lr)

    # Check that LR increases during warmup
    assert lrs[0] < lrs[5] < lrs[9]

    # Check that LR reaches target
    assert abs(lrs[9] - 1e-3) < 1e-6

    # Check that LR stays at target after warmup
    assert lrs[10] == 1e-3
    assert lrs[14] == 1e-3

    # Check that warmup is complete
    assert warmup.is_complete()


@pytest.mark.unit
def test_lr_warmup_linear_increase(optimizer_for_model):
    """Test that warmup increases linearly"""
    warmup = LRWarmup(
        optimizer=optimizer_for_model,
        warmup_steps=100,
        lr_start=0.0,
        lr_target=1.0,
    )

    # Step to 50% progress
    for _ in range(50):
        warmup.step()

    # Should be at ~0.5
    current_lr = optimizer_for_model.param_groups[0]['lr']
    assert abs(current_lr - 0.5) < 0.01


# ============================================================================
# LRSchedulerManager Tests
# ============================================================================

@pytest.mark.unit
def test_lr_scheduler_manager_cosine(optimizer_for_model):
    """Test LRSchedulerManager with cosine scheduler"""
    config = {
        'lr_cosine_T_0': 10,
        'lr_cosine_T_mult': 1,
        'lr_min': 1e-6,
    }

    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='cosine',
        config=config,
        enable_warmup=False,
    )

    assert scheduler_manager.scheduler_type == SchedulerType.COSINE

    # Run a few steps
    for _ in range(20):
        scheduler_manager.step()

    stats = scheduler_manager.get_statistics()
    assert stats['steps'] == 20
    assert 'current_lr' in stats


@pytest.mark.unit
def test_lr_scheduler_manager_plateau(optimizer_for_model):
    """Test LRSchedulerManager with plateau scheduler"""
    config = {
        'lr_plateau_factor': 0.5,
        'lr_plateau_patience': 5,
        'lr_min': 1e-6,
    }

    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='plateau',
        config=config,
        enable_warmup=False,
    )

    assert scheduler_manager.scheduler_type == SchedulerType.PLATEAU

    # Run steps with constant loss (should trigger reduction)
    initial_lr = scheduler_manager.get_current_lr()

    for _ in range(20):
        scheduler_manager.step(loss=1.0)

    final_lr = scheduler_manager.get_current_lr()

    # LR should have decreased
    assert final_lr < initial_lr


@pytest.mark.unit
def test_lr_scheduler_manager_step(optimizer_for_model):
    """Test LRSchedulerManager with step scheduler"""
    config = {
        'lr_step_size': 10,
        'lr_step_gamma': 0.5,
    }

    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='step',
        config=config,
        enable_warmup=False,
    )

    assert scheduler_manager.scheduler_type == SchedulerType.STEP

    initial_lr = scheduler_manager.get_current_lr()

    # Run 10 steps (should trigger reduction)
    for _ in range(10):
        scheduler_manager.step()

    # LR should have decreased by factor of 0.5
    final_lr = scheduler_manager.get_current_lr()
    assert abs(final_lr - initial_lr * 0.5) < 1e-8


@pytest.mark.unit
def test_lr_scheduler_manager_with_warmup(optimizer_for_model):
    """Test LRSchedulerManager with warmup enabled"""
    config = {
        'lr_warmup_start': 1e-5,
        'lr_cosine_T_0': 10,
        'lr_min': 1e-6,
    }

    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='cosine',
        config=config,
        enable_warmup=True,
        warmup_steps=10,
    )

    assert scheduler_manager.warmup is not None

    # During warmup, LR should increase
    initial_lr = scheduler_manager.get_current_lr()

    for _ in range(10):
        scheduler_manager.step()

    post_warmup_lr = scheduler_manager.get_current_lr()

    # LR should have increased during warmup
    assert post_warmup_lr > initial_lr


@pytest.mark.unit
def test_lr_scheduler_manager_auto_mode(optimizer_for_model):
    """Test LRSchedulerManager auto mode"""
    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='auto',
        enable_warmup=False,
    )

    assert scheduler_manager.auto_mode
    assert scheduler_manager.scheduler_type == SchedulerType.AUTO

    # Should start with cosine (exploration phase)
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    assert isinstance(scheduler_manager.scheduler, CosineAnnealingWarmRestarts)


@pytest.mark.unit
def test_lr_scheduler_manager_save_history(optimizer_for_model):
    """Test saving LR scheduler history"""
    scheduler_manager = LRSchedulerManager(
        optimizer=optimizer_for_model,
        scheduler_type='cosine',
        enable_warmup=False,
    )

    # Run a few steps
    for _ in range(10):
        scheduler_manager.step()

    # Save history
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "lr_history.json"
        scheduler_manager.save_history(str(filepath))

        assert filepath.exists()

        # Load and verify
        with open(filepath) as f:
            history = json.load(f)

        assert 'config' in history
        assert 'statistics' in history
        assert 'history' in history
        assert len(history['history']['learning_rates']) == 10


# ============================================================================
# TrainingOptimizer Tests (Integrated)
# ============================================================================

@pytest.mark.unit
def test_training_optimizer_initialization(simple_model, optimizer_for_model):
    """Test TrainingOptimizer initialization"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'cosine',
        'lr_warmup_enabled': True,
        'lr_warmup_steps': 100,
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    assert training_optimizer.gradient_clipper is not None
    assert training_optimizer.lr_scheduler is not None
    assert training_optimizer.step_count == 0


@pytest.mark.unit
def test_training_optimizer_step(simple_model, optimizer_for_model, sample_batch):
    """Test TrainingOptimizer step (integrated gradient clipping + LR scheduling)"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'cosine',
        'lr_warmup_enabled': True,
        'lr_warmup_steps': 10,
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = simple_model(sample_batch['input'])
    loss = criterion(output, sample_batch['target'])

    # Backward pass
    optimizer_for_model.zero_grad()
    loss.backward()

    # Training optimizer step
    step_info = training_optimizer.step(loss=loss.item())

    assert 'step' in step_info
    assert 'gradient_norm_before' in step_info
    assert 'gradient_norm_after' in step_info
    assert 'gradient_clipped' in step_info
    assert 'learning_rate' in step_info
    assert step_info['step'] == 1


@pytest.mark.unit
def test_training_optimizer_multiple_steps(simple_model, optimizer_for_model, sample_batch):
    """Test TrainingOptimizer over multiple training steps"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'step',
        'lr_step_size': 5,
        'lr_step_gamma': 0.5,
        'lr_warmup_enabled': False,
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    criterion = nn.CrossEntropyLoss()

    initial_lr = training_optimizer.lr_scheduler.get_current_lr()
    lrs = []

    # Run 10 training steps
    for _ in range(10):
        output = simple_model(sample_batch['input'])
        loss = criterion(output, sample_batch['target'])
        optimizer_for_model.zero_grad()
        loss.backward()

        step_info = training_optimizer.step(loss=loss.item())
        lrs.append(step_info['learning_rate'])

    # LR should have decreased at step 5 (0-indexed step 4)
    # So lrs[0-3] should be initial LR, lrs[4-9] should be decreased LR
    assert lrs[4] < lrs[3]  # Step 5's LR < Step 4's LR
    assert lrs[5] == lrs[4]  # Step 6's LR == Step 5's LR (same decreased value)


@pytest.mark.unit
def test_training_optimizer_statistics(simple_model, optimizer_for_model, sample_batch):
    """Test TrainingOptimizer statistics"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'cosine',
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    criterion = nn.CrossEntropyLoss()

    # Run a few steps
    for _ in range(5):
        output = simple_model(sample_batch['input'])
        loss = criterion(output, sample_batch['target'])
        optimizer_for_model.zero_grad()
        loss.backward()
        training_optimizer.step(loss=loss.item())

    # Get statistics
    stats = training_optimizer.get_statistics()

    assert 'steps' in stats
    assert 'gradient_clipper' in stats
    assert 'lr_scheduler' in stats
    assert stats['steps'] == 5


@pytest.mark.unit
def test_training_optimizer_save_histories(simple_model, optimizer_for_model, sample_batch):
    """Test saving training optimizer histories"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'cosine',
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    criterion = nn.CrossEntropyLoss()

    # Run a few steps
    for _ in range(5):
        output = simple_model(sample_batch['input'])
        loss = criterion(output, sample_batch['target'])
        optimizer_for_model.zero_grad()
        loss.backward()
        training_optimizer.step(loss=loss.item())

    # Save histories
    with tempfile.TemporaryDirectory() as tmpdir:
        training_optimizer.save_histories(tmpdir)

        gradient_path = Path(tmpdir) / "gradient_history.json"
        lr_path = Path(tmpdir) / "lr_history.json"

        assert gradient_path.exists()
        assert lr_path.exists()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
def test_gradient_clipper_empty_gradients(simple_model):
    """Test gradient clipper with no gradients"""
    clipper = GradientClipper(max_norm=1.0)

    # No backward pass, so no gradients
    norm_before, norm_after, was_clipped = clipper.clip_gradients(simple_model)

    assert norm_before == 0.0
    assert norm_after == 0.0
    assert not was_clipped


@pytest.mark.unit
def test_lr_warmup_zero_steps(optimizer_for_model):
    """Test LR warmup with zero warmup steps"""
    warmup = LRWarmup(
        optimizer=optimizer_for_model,
        warmup_steps=0,
        lr_start=1e-5,
        lr_target=1e-3,
    )

    # Should immediately be complete
    lr = warmup.step()
    assert warmup.is_complete()
    assert lr == 1e-3


@pytest.mark.unit
def test_training_optimizer_no_loss(simple_model, optimizer_for_model, sample_batch):
    """Test TrainingOptimizer step without loss (for schedulers that don't need it)"""
    config = {
        'gradient_clip_norm': 1.0,
        'lr_scheduler_type': 'step',
        'lr_warmup_enabled': False,
    }

    training_optimizer = TrainingOptimizer(
        model=simple_model,
        optimizer=optimizer_for_model,
        config=config,
    )

    criterion = nn.CrossEntropyLoss()

    # Forward + backward
    output = simple_model(sample_batch['input'])
    loss = criterion(output, sample_batch['target'])
    optimizer_for_model.zero_grad()
    loss.backward()

    # Step without loss (should still work)
    step_info = training_optimizer.step()

    assert 'step' in step_info
    assert step_info['loss'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
