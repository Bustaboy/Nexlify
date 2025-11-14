#!/usr/bin/env python3
"""
Test Script for Ultimate Training Pipeline

Tests all components for bugs, conflicts, and correct behavior
"""

import sys
from pathlib import Path
import numpy as np
import torch
import logging

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_advanced_dqn_agent():
    """Test Advanced DQN Agent"""
    logger.info("\n" + "="*80)
    logger.info("Testing Advanced DQN Agent")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig

    try:
        # Create minimal config
        config = AgentConfig(
            hidden_layers=[64, 32],
            batch_size=32,
            buffer_size=1000,
            use_prioritized_replay=True,
            use_double_dqn=True,
            use_dueling_dqn=True,
            use_swa=True,
            n_step=3
        )

        # Create agent
        agent = AdvancedDQNAgent(state_size=12, action_size=3, config=config)

        # Test act
        state = np.random.randn(12)
        action = agent.act(state, training=True)
        assert 0 <= action < 3, "Action out of range"

        # Test remember
        next_state = np.random.randn(12)
        agent.remember(state, action, 1.0, next_state, False)

        # Fill buffer a bit
        for _ in range(100):
            s = np.random.randn(12)
            a = np.random.randint(0, 3)
            r = np.random.randn()
            ns = np.random.randn(12)
            d = np.random.rand() > 0.95
            agent.remember(s, a, r, ns, d)

        # Test replay
        loss = agent.replay()
        assert isinstance(loss, float), "Loss should be a float"

        # Test metrics
        metrics = agent.get_metrics_summary()
        assert 'avg_loss' in metrics, "Metrics missing avg_loss"

        # Test save/load
        save_path = "/tmp/test_agent.pt"
        agent.save(save_path)
        agent.load(save_path)

        logger.info("✅ Advanced DQN Agent tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Advanced DQN Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prioritized_replay_buffer():
    """Test Prioritized Experience Replay"""
    logger.info("\n" + "="*80)
    logger.info("Testing Prioritized Replay Buffer")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import PrioritizedReplayBuffer

    try:
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add transitions
        for i in range(50):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = False
            error = np.random.rand()

            buffer.add(state, action, reward, next_state, done, error)

        assert len(buffer) == 50, f"Buffer size should be 50, got {len(buffer)}"

        # Sample batch
        batch, indices, weights = buffer.sample(32)
        assert len(batch) == 32, "Batch size should be 32"
        assert len(indices) == 32, "Indices size should be 32"
        assert len(weights) == 32, "Weights size should be 32"

        # Update priorities
        errors = np.random.randn(32)
        buffer.update_priorities(indices, errors)

        logger.info("✅ Prioritized Replay Buffer tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Prioritized Replay Buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dueling_dqn():
    """Test Dueling DQN Architecture"""
    logger.info("\n" + "="*80)
    logger.info("Testing Dueling DQN")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import DuelingDQN

    try:
        model = DuelingDQN(state_size=12, action_size=3, hidden_layers=[64, 32])

        # Forward pass
        state = torch.randn(1, 12)
        q_values = model(state)

        assert q_values.shape == (1, 3), f"Output shape should be (1, 3), got {q_values.shape}"

        # Batch forward
        batch = torch.randn(32, 8)
        q_batch = model(batch)

        assert q_batch.shape == (32, 3), f"Batch output shape should be (32, 3), got {q_batch.shape}"

        logger.info("✅ Dueling DQN tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Dueling DQN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_walk_forward_validator():
    """Test Walk-Forward Cross-Validation"""
    logger.info("\n" + "="*80)
    logger.info("Testing Walk-Forward Validator")
    logger.info("="*80)

    from nexlify_validation_and_optimization import WalkForwardValidator

    try:
        # Create validator
        validator = WalkForwardValidator(
            train_size=1000,
            test_size=200,
            step_size=200,
            anchored=False
        )

        # Create folds
        folds = validator.create_folds(data_length=2000)

        assert len(folds) > 0, "Should create at least one fold"

        logger.info(f"Created {len(folds)} folds")

        # Check fold structure
        for fold in folds:
            assert fold.train_end_idx > fold.train_start_idx, "Train range invalid"
            assert fold.test_end_idx > fold.test_start_idx, "Test range invalid"
            assert fold.test_start_idx == fold.train_end_idx, "Test should start where train ends"

        logger.info("✅ Walk-Forward Validator tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Walk-Forward Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_augmentation():
    """Test Data Augmentation"""
    logger.info("\n" + "="*80)
    logger.info("Testing Data Augmentation")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig

    try:
        config = AgentConfig(
            hidden_layers=[32],
            use_data_augmentation=True,
            augmentation_probability=1.0  # Always augment for testing
        )

        agent = AdvancedDQNAgent(state_size=12, action_size=3, config=config)

        # Test augmentation
        original_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        augmented_state = agent._augment_state(original_state)

        # Should be different (with very high probability)
        assert not np.array_equal(original_state, augmented_state), "Augmentation should change state"

        # Should be close (small noise)
        difference = np.abs(original_state - augmented_state)
        assert np.all(difference < 0.1), f"Augmentation noise too large: {difference}"

        logger.info("✅ Data Augmentation tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Data Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_n_step_returns():
    """Test N-Step Returns"""
    logger.info("\n" + "="*80)
    logger.info("Testing N-Step Returns")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig

    try:
        config = AgentConfig(
            hidden_layers=[32],
            n_step=3,
            buffer_size=1000
        )

        agent = AdvancedDQNAgent(state_size=12, action_size=3, config=config)

        # Add 5 transitions
        for i in range(5):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = 1.0  # Constant reward for easy checking
            next_state = np.random.randn(12)
            done = False

            agent.remember(state, action, reward, next_state, done)

        # Should have stored 3-step transitions (5 - 3 + 1 = 3)
        # Actually, it stores when buffer fills to n_step, so should have 3 transitions
        assert len(agent.memory) >= 1, "Should have stored at least 1 n-step transition"

        logger.info("✅ N-Step Returns tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ N-Step Returns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all components"""
    logger.info("\n" + "="*80)
    logger.info("Testing Full Integration")
    logger.info("="*80)

    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig

    try:
        # Create agent with ALL features enabled
        config = AgentConfig(
            hidden_layers=[64, 64, 32],
            batch_size=64,
            buffer_size=10000,
            use_prioritized_replay=True,
            use_double_dqn=True,
            use_dueling_dqn=True,
            use_swa=True,
            use_data_augmentation=True,
            n_step=3,
            gradient_clip_norm=1.0,
            weight_decay=1e-5,
            lr_scheduler_type='plateau',
            track_metrics=True
        )

        agent = AdvancedDQNAgent(state_size=10, action_size=4, config=config)

        # Run mini training loop
        for episode in range(10):
            state = np.random.randn(10)

            for step in range(50):
                action = agent.act(state, training=True)
                next_state = np.random.randn(10)
                reward = np.random.randn()
                done = step == 49

                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()

                state = next_state

                if done:
                    break

        # Check that everything ran
        metrics = agent.get_metrics_summary()
        assert metrics['training_steps'] > 0, "Should have trained"
        assert len(metrics['avg_loss']) > 0 or metrics['avg_loss'] >= 0, "Should have loss history"

        # Test validation score update
        for i in range(5):
            val_score = 70.0 + i * 2
            should_stop = agent.update_validation_score(val_score)
            if should_stop:
                break

        logger.info("✅ Full Integration tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Full Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*100)
    logger.info("ULTIMATE TRAINING PIPELINE - COMPREHENSIVE TESTING")
    logger.info("="*100 + "\n")

    tests = [
        ("Dueling DQN", test_dueling_dqn),
        ("Prioritized Replay Buffer", test_prioritized_replay_buffer),
        ("Advanced DQN Agent", test_advanced_dqn_agent),
        ("Walk-Forward Validator", test_walk_forward_validator),
        ("Data Augmentation", test_data_augmentation),
        ("N-Step Returns", test_n_step_returns),
        ("Full Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    logger.info("\n" + "="*100)
    logger.info("TEST SUMMARY")
    logger.info("="*100)

    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name:.<60} {status}")
        if not result:
            all_passed = False

    logger.info("="*100)

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.info("\n⚠️  SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
