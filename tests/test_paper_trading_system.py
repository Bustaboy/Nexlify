#!/usr/bin/env python3
"""
Comprehensive tests for Paper Trading System

Tests cover:
- Paper Trading Engine
- RL Training Environment
- Paper Trading Orchestrator
- Multi-agent functionality
- Performance tracking
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexlify.backtesting.nexlify_paper_trading import (PaperPosition,
                                                       PaperTradingEngine)
from nexlify.backtesting.nexlify_paper_trading_orchestrator import (
    AgentConfig, PaperTradingOrchestrator)
from nexlify.environments.nexlify_rl_training_env import (EpisodeStats,
                                                          TradingEnvironment)


class TestPaperTradingEngine:
    """Test Paper Trading Engine functionality"""

    @pytest.fixture
    def engine(self):
        """Create paper trading engine"""
        return PaperTradingEngine(
            {"paper_balance": 10000.0, "fee_rate": 0.001, "slippage": 0.0005}
        )

    @pytest.mark.asyncio
    async def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.initial_balance == 10000.0
        assert engine.current_balance == 10000.0
        assert engine.fee_rate == 0.001
        assert engine.slippage == 0.0005
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 0

    @pytest.mark.asyncio
    async def test_buy_order(self, engine):
        """Test buy order execution"""
        result = await engine.place_order("BTC/USDT", "buy", 0.1, 45000)

        assert result["success"] is True
        assert "position_id" in result
        assert result["amount"] == 0.1
        assert result["price"] == 45000
        assert len(engine.positions) == 1

        # Check balance was debited
        assert engine.current_balance < 10000.0

    @pytest.mark.asyncio
    async def test_sell_order(self, engine):
        """Test sell order execution"""
        # First buy
        buy_result = await engine.place_order("BTC/USDT", "buy", 0.1, 45000)
        assert buy_result["success"]

        # Then sell at higher price
        sell_result = await engine.place_order("BTC/USDT", "sell", 0.1, 47000)

        assert sell_result["success"]
        assert sell_result["pnl"] > 0  # Should be profitable
        assert len(engine.positions) == 0  # Position closed
        assert len(engine.completed_trades) == 1

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, engine):
        """Test buy with insufficient balance"""
        # Try to buy more than balance allows
        result = await engine.place_order(
            "BTC/USDT", "buy", 1.0, 45000  # Would cost $45k
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Insufficient balance"

    @pytest.mark.asyncio
    async def test_sell_without_position(self, engine):
        """Test sell without open position"""
        result = await engine.place_order("BTC/USDT", "sell", 0.1, 45000)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_position_updates(self, engine):
        """Test position price updates"""
        # Buy position
        await engine.place_order("BTC/USDT", "buy", 0.1, 45000)

        # Update with new price
        await engine.update_positions({"BTC/USDT": 47000})

        # Check unrealized PnL
        position = list(engine.positions.values())[0]
        assert position.current_price == 47000
        assert position.unrealized_pnl > 0

    @pytest.mark.asyncio
    async def test_statistics(self, engine):
        """Test statistics calculation"""
        # Execute some trades
        await engine.place_order("BTC/USDT", "buy", 0.1, 45000)
        await engine.update_positions({"BTC/USDT": 47000})
        await engine.place_order("BTC/USDT", "sell", 0.1, 47000)

        stats = engine.get_statistics()

        assert "total_return" in stats
        assert "win_rate" in stats
        assert "total_trades" in stats
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1

    @pytest.mark.asyncio
    async def test_fees_and_slippage(self, engine):
        """Test that fees and slippage are applied"""
        initial_balance = engine.current_balance

        # Buy and immediately sell at same price
        await engine.place_order("BTC/USDT", "buy", 0.1, 45000)
        await engine.place_order("BTC/USDT", "sell", 0.1, 45000)

        # Should have less money due to fees and slippage
        assert engine.current_balance < initial_balance


class TestTradingEnvironment:
    """Test RL Trading Environment"""

    @pytest.fixture
    def env(self):
        """Create trading environment"""
        return TradingEnvironment(
            initial_balance=10000.0, max_steps=100, use_paper_trading=True
        )

    def test_initialization(self, env):
        """Test environment initialization"""
        assert env.initial_balance == 10000.0
        assert env.state_size == 8
        assert env.action_size == 3
        assert env.max_steps == 100

    def test_reset(self, env):
        """Test environment reset"""
        state = env.reset()

        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)
        assert env.current_step == 0
        assert env.balance == env.initial_balance
        assert env.position == 0.0

    def test_step_buy(self, env):
        """Test buy action"""
        env.reset()
        state, reward, done, info = env.step(env.ACTION_BUY)

        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "balance" in info
        assert "equity" in info

        # Should have entered position
        assert env.position > 0

    def test_step_sell(self, env):
        """Test sell action"""
        env.reset()

        # First buy
        env.step(env.ACTION_BUY)

        # Then sell
        state, reward, done, info = env.step(env.ACTION_SELL)

        # Position should be closed
        assert env.position == 0

    def test_step_hold(self, env):
        """Test hold action"""
        env.reset()
        initial_balance = env.balance

        state, reward, done, info = env.step(env.ACTION_HOLD)

        # Balance shouldn't change
        assert env.balance == initial_balance
        assert env.position == 0

    def test_episode_completion(self, env):
        """Test full episode"""
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            action = np.random.randint(0, 3)
            state, reward, done, info = state, reward, done, info = env.step(action)
            steps += 1

        assert done or steps == env.max_steps
        assert len(env.episode_history) > 0

    def test_state_vector(self, env):
        """Test state vector composition"""
        state = env.reset()

        # Check state is normalized
        assert np.all(state >= -10) and np.all(state <= 10)  # Reasonable bounds
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))

    def test_reward_calculation(self, env):
        """Test reward calculation"""
        env.reset()

        # Buy at current price
        _, reward_buy, _, _ = env.step(env.ACTION_BUY)

        # Price goes up (simulated)
        env.current_price *= 1.02  # 2% increase

        # Hold should give positive reward
        _, reward_hold, _, _ = env.step(env.ACTION_HOLD)

        # Rewards should be calculated
        assert isinstance(reward_buy, float)
        assert isinstance(reward_hold, float)

    def test_episode_stats(self, env):
        """Test episode statistics tracking"""
        state = env.reset()
        done = False

        while not done:
            action = np.random.randint(0, 3)
            state, reward, done, info = env.step(action)

        stats = env.get_episode_stats()
        assert len(stats) == 1

        episode_stat = stats[0]
        assert isinstance(episode_stat, EpisodeStats)
        assert hasattr(episode_stat, "total_return")
        assert hasattr(episode_stat, "win_rate")
        assert hasattr(episode_stat, "sharpe_ratio")


class TestPaperTradingOrchestrator:
    """Test Paper Trading Orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator"""
        return PaperTradingOrchestrator(
            {
                "initial_balance": 10000.0,
                "update_interval": 1,  # Fast updates for testing
            }
        )

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.initial_balance == 10000.0
        assert len(orchestrator.agents) == 0
        assert orchestrator.is_running is False

    def test_register_agent(self, orchestrator):
        """Test agent registration"""
        agent_config = AgentConfig(
            agent_id="test_agent", agent_type="rl_adaptive", name="Test Agent"
        )

        orchestrator.register_agent(agent_config)

        assert "test_agent" in orchestrator.agents
        assert "test_agent" in orchestrator.agent_engines

    def test_register_multiple_agents(self, orchestrator):
        """Test registering multiple agents"""
        configs = [
            AgentConfig("agent1", "rl_adaptive", "Agent 1"),
            AgentConfig("agent2", "rl_ultra", "Agent 2"),
            AgentConfig("agent3", "ml_ensemble", "Agent 3"),
        ]

        for config in configs:
            orchestrator.register_agent(config)

        assert len(orchestrator.agents) == 3
        assert len(orchestrator.agent_engines) == 3

    def test_agent_engines_isolated(self, orchestrator):
        """Test that agent engines are isolated"""
        agent1 = AgentConfig("agent1", "rl_adaptive", "Agent 1")
        agent2 = AgentConfig("agent2", "rl_adaptive", "Agent 2")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Execute trade for agent1
        engine1 = orchestrator.agent_engines["agent1"]
        engine2 = orchestrator.agent_engines["agent2"]

        # Initial balances should be equal
        assert engine1.current_balance == engine2.current_balance

        # Execute trade in engine1
        asyncio.run(engine1.place_order("BTC/USDT", "buy", 0.1, 45000))

        # Balances should now differ
        assert engine1.current_balance != engine2.current_balance

    def test_performance_tracking(self, orchestrator):
        """Test performance tracking"""
        agent_config = AgentConfig("test_agent", "rl_adaptive", "Test Agent")
        orchestrator.register_agent(agent_config)

        # Record snapshot
        orchestrator._record_performance_snapshot("test_agent")

        assert "test_agent" in orchestrator.performance_history
        assert len(orchestrator.performance_history["test_agent"]) == 1

    def test_comparison_metrics(self, orchestrator):
        """Test comparison metrics calculation"""
        # Register multiple agents
        for i in range(3):
            config = AgentConfig(f"agent{i}", "rl_adaptive", f"Agent {i}")
            orchestrator.register_agent(config)
            orchestrator._record_performance_snapshot(f"agent{i}")

        # Update comparison
        orchestrator._update_comparison_metrics()

        assert "agents" in orchestrator.comparison_metrics
        assert len(orchestrator.comparison_metrics["agents"]) == 3

    def test_leaderboard(self, orchestrator):
        """Test leaderboard generation"""
        # Register agents with different performance
        for i in range(3):
            config = AgentConfig(f"agent{i}", "rl_adaptive", f"Agent {i}")
            orchestrator.register_agent(config)

            # Simulate different performance
            engine = orchestrator.agent_engines[f"agent{i}"]
            engine.current_balance = 10000 + (i * 500)

            orchestrator._record_performance_snapshot(f"agent{i}")

        orchestrator._update_comparison_metrics()
        leaderboard = orchestrator.get_leaderboard()

        assert len(leaderboard) == 3
        # Should be sorted by return (highest first)
        assert leaderboard[0][1] >= leaderboard[1][1] >= leaderboard[2][1]


class TestIntegration:
    """Integration tests for complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_training(self):
        """Test complete training workflow"""
        # Create environment
        env = TradingEnvironment(
            initial_balance=10000.0,
            max_steps=50,  # Short episode for testing
            use_paper_trading=True,
        )

        # Simple mock agent
        class MockAgent:
            def act(self, state, training=True):
                return np.random.randint(0, 3)

            def remember(self, state, action, reward, next_state, done):
                pass

            def replay(self):
                return 0.0

        agent = MockAgent()

        # Run training episode
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, state, done)
            agent.replay()
            total_reward += reward

        assert done
        assert len(env.episode_history) == 1
        assert isinstance(total_reward, float)

    def test_multi_environment_training(self):
        """Test training with multiple environments"""
        envs = [
            TradingEnvironment(initial_balance=10000.0, max_steps=10) for _ in range(3)
        ]

        # Each environment should be independent
        states = [env.reset() for env in envs]

        assert len(states) == 3
        for state in states:
            assert isinstance(state, np.ndarray)

    @pytest.mark.asyncio
    async def test_orchestrator_workflow(self):
        """Test orchestrator complete workflow"""
        orchestrator = PaperTradingOrchestrator(
            {
                "initial_balance": 10000.0,
                "update_interval": 0.1,  # Very fast for testing
            }
        )

        # Register test agents
        for i in range(2):
            config = AgentConfig(
                agent_id=f"test_agent_{i}",
                agent_type="rl_adaptive",
                name=f"Test Agent {i}",
                enabled=True,
            )
            orchestrator.register_agent(config)

        # Verify setup
        assert len(orchestrator.agents) == 2
        assert all(a.enabled for a in orchestrator.agents.values())


def test_imports():
    """Test that all modules can be imported"""
    from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
    from nexlify.backtesting.nexlify_paper_trading_orchestrator import \
        PaperTradingOrchestrator
    from nexlify.environments.nexlify_rl_training_env import TradingEnvironment

    assert PaperTradingEngine is not None
    assert TradingEnvironment is not None
    assert PaperTradingOrchestrator is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
