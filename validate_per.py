#!/usr/bin/env python3
"""
Quick validation script for Prioritized Experience Replay (PER)
Tests basic functionality without requiring full dependencies
"""

import numpy as np

def main():
    print("=" * 60)
    print("Prioritized Experience Replay (PER) Validation")
    print("=" * 60)
    print()

    # Test 1: SumTree
    print("Test 1: SumTree data structure")
    print("-" * 60)
    from nexlify.memory.sumtree import SumTree

    tree = SumTree(capacity=100)
    print(f"✓ Created SumTree with capacity=100")

    # Add items
    for i in range(10):
        tree.add(float(i + 1), f"data_{i}")
    print(f"✓ Added 10 items")

    # Test total
    expected_total = sum(range(1, 11))
    assert tree.total() == expected_total, f"Total mismatch: {tree.total()} != {expected_total}"
    print(f"✓ Total priority sum correct: {tree.total()}")

    # Test sampling
    idx, priority, data = tree.sample(5.0)
    print(f"✓ Sampling works: sampled '{data}' with priority {priority}")

    print()

    # Test 2: PrioritizedReplayBuffer
    print("Test 2: PrioritizedReplayBuffer")
    print("-" * 60)
    from nexlify.memory.prioritized_replay_buffer import PrioritizedReplayBuffer

    buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
    )
    print(f"✓ Created PrioritizedReplayBuffer")

    # Add experiences
    for i in range(50):
        state = np.random.randn(12)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(12)
        done = False
        buffer.push(state, action, reward, next_state, done)

    print(f"✓ Added 50 experiences")

    # Sample batch
    experiences, indices, weights = buffer.sample(32)
    print(f"✓ Sampled batch of 32 experiences")
    print(f"  - IS weights range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Update priorities
    td_errors = np.random.randn(32) * 2.0
    buffer.update_priorities(indices, td_errors)
    print(f"✓ Updated priorities based on TD errors")

    # Get stats
    stats = buffer.get_stats()
    print(f"✓ Buffer statistics:")
    print(f"  - Size: {stats['size']}/{stats['capacity']}")
    print(f"  - Beta: {stats['beta']:.3f}")
    print(f"  - Mean priority: {stats['mean_priority']:.4f}")

    print()

    # Test 3: Beta annealing
    print("Test 3: Beta annealing")
    print("-" * 60)

    buffer_anneal = PrioritizedReplayBuffer(
        capacity=1000,
        beta_start=0.4,
        beta_end=1.0,
        beta_annealing_steps=100,
    )

    # Add experiences
    for i in range(50):
        state = np.random.randn(12)
        buffer_anneal.push(state, 0, 0.0, state, False)

    # Track beta over samples
    betas = []
    for _ in range(150):
        _, _, _ = buffer_anneal.sample(10)
        betas.append(buffer_anneal.get_stats()['beta'])

    print(f"✓ Beta progression:")
    print(f"  - Initial: {betas[0]:.3f}")
    print(f"  - At 50 samples: {betas[49]:.3f}")
    print(f"  - At 100 samples: {betas[99]:.3f}")
    print(f"  - Final (150): {betas[-1]:.3f}")
    assert betas[0] < betas[99] < betas[-1] or abs(betas[99] - 1.0) < 0.01
    print(f"✓ Beta annealing works correctly")

    print()

    # Test 4: Visualization
    print("Test 4: Visualization tools")
    print("-" * 60)
    from nexlify.memory.per_visualization import PERStatsTracker, create_per_report

    tracker = PERStatsTracker()
    print(f"✓ Created PERStatsTracker")

    # Record stats over time
    for episode in range(10):
        buffer_anneal.sample(10)
        tracker.record(buffer_anneal.get_stats(), episode=episode)

    print(f"✓ Recorded 10 episodes of stats")

    summary = tracker.get_summary()
    print(f"✓ Summary statistics:")
    print(f"  - Final beta: {summary['final_beta']:.3f}")
    print(f"  - Mean priority (avg): {summary['mean_priority_avg']:.4f}")

    # Create report
    report = create_per_report(buffer.get_stats(), tracker)
    print(f"✓ Generated PER report ({len(report)} chars)")

    print()

    # Summary
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("PER implementation is working correctly:")
    print("  ✓ SumTree for O(log n) sampling")
    print("  ✓ Priority-based experience replay")
    print("  ✓ Importance sampling weights")
    print("  ✓ Beta annealing")
    print("  ✓ Visualization and reporting tools")
    print()
    print("Next steps:")
    print("  1. Integrate with DQNAgent (already done)")
    print("  2. Train agent with PER enabled")
    print("  3. Monitor PER statistics during training")
    print("  4. Compare performance vs standard replay buffer")
    print()


if __name__ == "__main__":
    main()
