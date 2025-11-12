#!/usr/bin/env python3
"""
Comprehensive test script for multi-mode RL implementation

Tests:
1. Import verification
2. Environment creation and basic operations
3. All 30 actions
4. State space (31 features)
5. Agent integration
6. Short training loop
7. Liquidity checks
8. Fee tracking
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_section(title: str):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_imports():
    """Test 1: Verify all imports work"""
    print_section("TEST 1: IMPORT VERIFICATION")

    try:
        from nexlify.strategies.nexlify_enhanced_rl_agent import (
            EnhancedTradingEnvironment,
            TradingMode,
            OrderType,
            PositionSize,
            Position
        )
        print("✅ Enhanced RL agent imports successful")

        from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
        print("✅ Ultra optimized agent import successful")

        from nexlify.ml.nexlify_optimization_manager import OptimizationProfile
        print("✅ Optimization manager import successful")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_environment_creation():
    """Test 2: Create environment and verify parameters"""
    print_section("TEST 2: ENVIRONMENT CREATION")

    try:
        from nexlify.strategies.nexlify_enhanced_rl_agent import EnhancedTradingEnvironment

        # Generate test price data
        price_data = np.random.uniform(40000, 50000, 1000)

        # Create environment
        env = EnhancedTradingEnvironment(
            price_data=price_data,
            initial_balance=10000,
            max_leverage=10.0,
            trading_fee=0.001,
            funding_rate=0.0001,
            margin_interest=0.0002,
            gas_fee_usd=5.0,
            market_liquidity_ratio=100.0,
            max_slippage_tolerance=0.05
        )

        print(f"✅ Environment created")
        print(f"   Action space: {env.action_space_n} actions")
        print(f"   State space: {env.state_space_n} features")
        print(f"   Max steps: {env.max_steps}")
        print(f"   Liquidity depth: ${env.liquidity_depth:,.0f}")

        # Verify correct dimensions
        assert env.action_space_n == 30, f"Expected 30 actions, got {env.action_space_n}"
        assert env.state_space_n == 31, f"Expected 31 features, got {env.state_space_n}"

        print("✅ Action and state space dimensions correct")

        return env

    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_reset_and_state(env):
    """Test 3: Reset environment and verify state"""
    print_section("TEST 3: RESET AND STATE")

    try:
        state = env.reset()

        print(f"✅ Environment reset successful")
        print(f"   State shape: {state.shape}")
        print(f"   State dtype: {state.dtype}")
        print(f"   State range: [{state.min():.4f}, {state.max():.4f}]")

        # Verify state shape
        assert state.shape == (31,), f"Expected state shape (31,), got {state.shape}"

        # Verify no NaN or inf
        assert not np.any(np.isnan(state)), "State contains NaN"
        assert not np.any(np.isinf(state)), "State contains inf"

        print("✅ State verification passed")

        # Print first few features
        print("\n   First 10 state features:")
        feature_names = [
            "balance_ratio", "margin_available_ratio", "total_value_ratio",
            "equity_ratio", "margin_ratio", "price_normalized",
            "price_change", "rsi", "macd", "volume"
        ]
        for i, (name, val) in enumerate(zip(feature_names, state[:10])):
            print(f"   [{i}] {name:25s}: {val:8.4f}")

        return True

    except Exception as e:
        print(f"❌ Reset/state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_actions(env):
    """Test 4: Execute all 30 actions"""
    print_section("TEST 4: ALL ACTIONS TEST")

    try:
        action_results = []

        for action in range(30):
            # Reset environment for each action
            env.reset()

            # Get action name
            action_name = env._get_action_name(action)

            # Execute action
            next_state, reward, done, info = env.step(action)

            action_results.append({
                'action': action,
                'name': action_name,
                'reward': reward,
                'done': done,
                'trade': info.get('trade', False)
            })

            print(f"   Action {action:2d} ({action_name:25s}): reward={reward:8.4f}, trade={info.get('trade', False)}")

        print(f"\n✅ All 30 actions executed successfully")

        # Verify action names are unique
        names = [r['name'] for r in action_results]
        assert len(names) == len(set(names)), "Duplicate action names found"
        print("✅ All action names are unique")

        return True

    except Exception as e:
        print(f"❌ Actions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_liquidity_checks(env):
    """Test 5: Liquidity checks and slippage"""
    print_section("TEST 5: LIQUIDITY CHECKS")

    try:
        env.reset()

        # Test small order (should pass)
        small_order = env.initial_balance * 0.1
        is_sufficient, slippage, reason = env._check_liquidity_sufficient(small_order)
        print(f"   Small order (${small_order:,.0f}):")
        print(f"      Sufficient: {is_sufficient}")
        print(f"      Slippage: {slippage*100:.4f}%")
        print(f"      Reason: {reason}")
        assert is_sufficient, "Small order should be sufficient"

        # Test medium order
        medium_order = env.liquidity_depth * 0.05
        is_sufficient, slippage, reason = env._check_liquidity_sufficient(medium_order)
        print(f"\n   Medium order (${medium_order:,.0f}):")
        print(f"      Sufficient: {is_sufficient}")
        print(f"      Slippage: {slippage*100:.4f}%")
        print(f"      Reason: {reason}")

        # Test large order (should fail)
        large_order = env.liquidity_depth * 0.2
        is_sufficient, slippage, reason = env._check_liquidity_sufficient(large_order)
        print(f"\n   Large order (${large_order:,.0f}):")
        print(f"      Sufficient: {is_sufficient}")
        print(f"      Slippage: {slippage*100:.4f}%")
        print(f"      Reason: {reason}")
        assert not is_sufficient, "Large order should be rejected"

        print("\n✅ Liquidity checks working correctly")
        return True

    except Exception as e:
        print(f"❌ Liquidity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fee_tracking(env):
    """Test 6: Fee tracking"""
    print_section("TEST 6: FEE TRACKING")

    try:
        env.reset()
        initial_fees = env.total_fees_paid

        # Execute a buy action (should incur trading fee)
        env.step(1)  # buy_spot_25%
        after_trade_fees = env.total_fees_paid

        print(f"   Initial fees: ${initial_fees:.2f}")
        print(f"   After spot trade: ${after_trade_fees:.2f}")
        print(f"   Fee charged: ${after_trade_fees - initial_fees:.2f}")

        assert after_trade_fees > initial_fees, "Trading fee should be charged"

        # Execute a DeFi action (should incur gas fee)
        env.step(19)  # add_liquidity_25%
        after_defi_fees = env.total_fees_paid

        print(f"   After DeFi operation: ${after_defi_fees:.2f}")
        print(f"   Gas fee charged: ${after_defi_fees - after_trade_fees:.2f}")

        assert after_defi_fees > after_trade_fees, "Gas fee should be charged"

        print("\n✅ Fee tracking working correctly")
        return True

    except Exception as e:
        print(f"❌ Fee tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_integration(env):
    """Test 7: Agent creation and integration"""
    print_section("TEST 7: AGENT INTEGRATION")

    try:
        from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
        from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

        # Create agent
        agent = create_ultra_optimized_agent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            profile=OptimizationProfile.ULTRA_LOW_OVERHEAD,
            enable_sentiment=False
        )

        print(f"✅ Agent created")
        print(f"   State size: {agent.state_size}")
        print(f"   Action size: {agent.action_size}")
        print(f"   Device: {agent.device}")

        # Verify dimensions
        assert agent.state_size == 31, f"Expected state_size=31, got {agent.state_size}"
        assert agent.action_size == 30, f"Expected action_size=30, got {agent.action_size}"

        # Test agent.act()
        state = env.reset()
        action = agent.act(state, training=True)

        print(f"   Agent selected action: {action} ({env._get_action_name(action)})")
        assert 0 <= action < 30, f"Invalid action: {action}"

        print("✅ Agent integration working")

        return agent

    except Exception as e:
        print(f"❌ Agent integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_loop(env, agent):
    """Test 8: Short training loop"""
    print_section("TEST 8: SHORT TRAINING LOOP")

    try:
        episodes = 3

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0

            for step in range(100):  # Max 100 steps per episode
                # Select action
                action = agent.act(state, training=True)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train if enough samples
                if len(agent.memory) >= 32:
                    loss = agent.replay()

                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            perf = env.get_performance_summary()

            print(f"\n   Episode {episode + 1}:")
            print(f"      Steps: {steps}")
            print(f"      Reward: {episode_reward:.2f}")
            print(f"      Final value: ${perf['final_value']:,.2f}")
            print(f"      Return: {perf['total_return_%']:+.2f}%")
            print(f"      Trades: {perf['total_trades']}")
            print(f"      Fees: ${perf['total_fees']:.2f}")

        print(f"\n✅ Training loop completed successfully")

        # Cleanup
        agent.shutdown()

        return True

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  COMPREHENSIVE MULTI-MODE RL TEST SUITE")
    print("="*80)

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n❌ Critical failure: Imports failed. Aborting.")
        return 1

    # Test 2: Environment creation
    env = test_environment_creation()
    results['environment'] = env is not None
    if not results['environment']:
        print("\n❌ Critical failure: Environment creation failed. Aborting.")
        return 1

    # Test 3: Reset and state
    results['reset_state'] = test_reset_and_state(env)

    # Test 4: All actions
    results['actions'] = test_all_actions(env)

    # Test 5: Liquidity checks
    results['liquidity'] = test_liquidity_checks(env)

    # Test 6: Fee tracking
    results['fees'] = test_fee_tracking(env)

    # Test 7: Agent integration
    agent = test_agent_integration(env)
    results['agent'] = agent is not None

    # Test 8: Training loop
    if agent is not None:
        results['training'] = test_training_loop(env, agent)
    else:
        results['training'] = False

    # Summary
    print_section("TEST SUMMARY")

    total_tests = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"   {test_name:20s}: {status}")

    print(f"\n   TOTAL: {passed}/{total_tests} tests passed")

    if passed == total_tests:
        print("\n" + "="*80)
        print("  ✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        return 0
    else:
        print("\n" + "="*80)
        print(f"  ❌ {total_tests - passed} TEST(S) FAILED")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
