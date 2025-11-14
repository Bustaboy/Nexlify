#!/usr/bin/env python3
"""
Multi-Gamma Parallel Training
==============================

Train multiple RL agents with different gamma values in parallel,
then swap to the best-performing agent when Sharpe parity is reached.

Key features:
- Hardware benchmarking before enabling (checks if system can handle 3x load)
- All agents train on same experiences (shared replay)
- Zero recalibration cost (swap agents, don't adjust gamma mid-training)
- Sharpe parity detection for optimal swap timing
- Automatic fallback to single agent if hardware insufficient

Author: Nexlify Team
"""

import logging
import time
import psutil
import platform
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import numpy as np

from nexlify.strategies.nexlify_rl_agent import DQNAgent, TradingEnvironment
from nexlify.strategies.gamma_selector import get_recommended_gamma, TRADING_STYLES

logger = logging.getLogger(__name__)


# ============================================================================
# HARDWARE BENCHMARKING
# ============================================================================

class HardwareBenchmark:
    """
    Benchmark system to determine if it can handle multi-gamma training

    Multi-gamma training requires 3x computational resources:
    - 3x CPU for training 3 agents
    - 3x memory for 3 models + replay buffers
    - Same data throughput (shared experiences)
    """

    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.memory_total_gb = psutil.virtual_memory().total / (1024**3)
        self.system = platform.system()

    def check_requirements(
        self,
        num_agents: int = 3,
        state_size: int = 12,
        action_size: int = 3,
        replay_buffer_size: int = 100000
    ) -> Tuple[bool, str]:
        """
        Check if system meets requirements for multi-gamma training

        Args:
            num_agents: Number of parallel agents
            state_size: State space size
            action_size: Action space size
            replay_buffer_size: Replay buffer capacity

        Returns:
            (meets_requirements, reason)
        """
        reasons = []

        # CPU Check: Need at least 4 cores for 3 agents
        min_cores = max(4, num_agents)
        if self.cpu_count < min_cores:
            reasons.append(
                f"❌ CPU: {self.cpu_count} cores (need {min_cores}+ for {num_agents} agents)"
            )
        else:
            reasons.append(
                f"✅ CPU: {self.cpu_count} cores (sufficient for {num_agents} agents)"
            )

        # Memory Check: Estimate memory requirements
        # Each agent: ~50MB (model) + replay buffer
        bytes_per_experience = (state_size + action_size + 2) * 8  # float64
        replay_memory_mb = (replay_buffer_size * bytes_per_experience) / (1024**2)
        model_memory_mb = 50  # Rough estimate for DQN model

        per_agent_mb = model_memory_mb + replay_memory_mb
        total_required_mb = per_agent_mb * num_agents
        total_required_gb = total_required_mb / 1024

        # Need at least 2GB overhead for OS
        available_gb = self.memory_total_gb - 2

        if available_gb < total_required_gb:
            reasons.append(
                f"❌ Memory: {self.memory_total_gb:.1f}GB total "
                f"(need {total_required_gb:.1f}GB + 2GB overhead)"
            )
        else:
            reasons.append(
                f"✅ Memory: {self.memory_total_gb:.1f}GB total "
                f"({total_required_gb:.1f}GB needed, {available_gb - total_required_gb:.1f}GB spare)"
            )

        # Current CPU load check
        current_cpu_percent = psutil.cpu_percent(interval=0.5)
        if current_cpu_percent > 70:
            reasons.append(
                f"⚠️  CPU Load: {current_cpu_percent:.0f}% (high load may slow training)"
            )
        else:
            reasons.append(
                f"✅ CPU Load: {current_cpu_percent:.0f}% (acceptable)"
            )

        # Current memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            reasons.append(
                f"⚠️  Memory Usage: {memory_percent:.0f}% (high usage may cause issues)"
            )
        else:
            reasons.append(
                f"✅ Memory Usage: {memory_percent:.0f}% (acceptable)"
            )

        # Check if we have GPU
        has_gpu = self._check_gpu()
        if has_gpu:
            reasons.append("✅ GPU: Available (will accelerate training)")
        else:
            reasons.append("ℹ️  GPU: Not available (CPU-only training)")

        # Final decision: Need both CPU and memory to pass
        cpu_ok = self.cpu_count >= min_cores
        memory_ok = available_gb >= total_required_gb

        meets_requirements = cpu_ok and memory_ok

        reason_str = "\n".join(reasons)
        return meets_requirements, reason_str

    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def run_quick_benchmark(self) -> Dict[str, Any]:
        """
        Run quick training benchmark to estimate performance impact

        Returns:
            Dict with benchmark results
        """
        logger.info("🔍 Running hardware benchmark...")

        # Simulate training loop
        start_time = time.time()

        # Simulate agent forward pass (matrix multiplications)
        state = np.random.randn(12).astype(np.float32)
        weights = [np.random.randn(12, 128), np.random.randn(128, 128), np.random.randn(128, 3)]

        iterations = 1000
        for _ in range(iterations):
            x = state
            for w in weights:
                x = np.maximum(0, np.dot(x, w))  # ReLU activation

        elapsed = time.time() - start_time
        iters_per_sec = iterations / elapsed

        result = {
            "iterations_per_second": iters_per_sec,
            "estimated_slowdown": 3.0,  # 3 agents = ~3x slower
            "can_handle_multi_gamma": iters_per_sec > 100,  # Need >100 iter/s
            "benchmark_time": elapsed
        }

        logger.info(
            f"✅ Benchmark complete: {iters_per_sec:.0f} iterations/sec "
            f"(3x load = {iters_per_sec/3:.0f} iter/sec per agent)"
        )

        return result


# ============================================================================
# MULTI-GAMMA TRAINER
# ============================================================================

class MultiGammaTrainer:
    """
    Train multiple agents with different gammas in parallel

    Strategy:
    1. Create 3 agents with different gammas (scalping, day trading, swing)
    2. Train all agents on same experiences
    3. Monitor Sharpe ratios
    4. When Sharpe parity reached, swap to agent with best-matching gamma
    5. Continue training with swapped agent

    Benefits:
    - Zero recalibration cost (no mid-training gamma changes)
    - Optimal gamma selection based on actual performance
    - Can compare agents objectively before swapping

    Costs:
    - 3x computational resources
    - 3x memory for models
    - Requires adequate hardware
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gammas: Optional[List[float]] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_if_hardware_sufficient: bool = True
    ):
        """
        Initialize multi-gamma trainer

        Args:
            state_size: State space size
            action_size: Action space size
            gammas: List of gamma values (default: [0.90, 0.95, 0.97])
            config: Configuration dict
            enable_if_hardware_sufficient: Only enable if hardware check passes
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Default gammas: scalping, day trading, swing trading
        self.gammas = gammas or [0.90, 0.95, 0.97]

        # Hardware check
        self.hardware_benchmark = HardwareBenchmark()
        self.hardware_sufficient = False
        self.enabled = False

        if enable_if_hardware_sufficient:
            self._check_hardware()

        # Agents
        self.agents: Dict[float, DQNAgent] = {}
        self.active_gamma: Optional[float] = None

        # Swap tracking
        self.swap_history: List[Dict] = []
        self.episodes_since_swap = 0
        self.min_episodes_before_swap = 500  # Need data before first swap
        self.sharpe_parity_threshold = 0.1  # 10% difference = parity

        # Performance tracking
        self.episode_rewards: Dict[float, List[float]] = {g: [] for g in self.gammas}
        self.episode_sharpe: Dict[float, List[float]] = {g: [] for g in self.gammas}

        # Trade duration tracking (for gamma alignment)
        self.trade_durations: List[float] = []

        if self.enabled:
            self._initialize_agents()
        else:
            # Fallback to single agent
            self._initialize_single_agent()

        logger.info(
            f"{'🚀 MultiGammaTrainer' if self.enabled else '⚡ Single Agent'} initialized:\n"
            f"   Mode: {'Parallel training' if self.enabled else 'Standard (hardware insufficient)'}\n"
            f"   Gammas: {self.gammas if self.enabled else [self.agents[self.active_gamma].gamma]}\n"
            f"   Active gamma: {self.active_gamma:.3f}"
        )

    def _check_hardware(self):
        """Check if hardware is sufficient for multi-gamma training"""
        meets_req, reason = self.hardware_benchmark.check_requirements(
            num_agents=len(self.gammas),
            state_size=self.state_size,
            action_size=self.action_size,
            replay_buffer_size=self.config.get("replay_buffer_size", 100000)
        )

        logger.info(
            f"\n{'='*70}\n"
            f"HARDWARE REQUIREMENTS CHECK\n"
            f"{'='*70}\n"
            f"{reason}\n"
            f"{'='*70}\n"
            f"Result: {'✅ ENABLED' if meets_req else '❌ DISABLED (falling back to single agent)'}\n"
            f"{'='*70}"
        )

        self.hardware_sufficient = meets_req
        self.enabled = meets_req

    def _initialize_agents(self):
        """Initialize multiple agents with different gammas"""
        logger.info(f"🤖 Initializing {len(self.gammas)} agents...")

        for gamma in self.gammas:
            # Create config for this gamma
            agent_config = self.config.copy()
            agent_config['manual_gamma'] = gamma

            # Create agent
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config=agent_config
            )

            self.agents[gamma] = agent
            logger.info(f"   ✅ Agent gamma={gamma:.2f} initialized")

        # Start with middle gamma (usually day trading = 0.95)
        self.active_gamma = sorted(self.gammas)[len(self.gammas) // 2]
        logger.info(f"   🎯 Active agent: gamma={self.active_gamma:.2f}")

    def _initialize_single_agent(self):
        """Fallback: Initialize single agent (hardware insufficient)"""
        # Use default gamma from config or 0.89
        gamma = self.config.get('gamma', 0.89)
        self.gammas = [gamma]

        agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=self.config
        )

        self.agents[gamma] = agent
        self.active_gamma = gamma

        logger.warning(
            f"⚠️  Hardware insufficient for multi-gamma training\n"
            f"   Falling back to single agent with gamma={gamma:.2f}"
        )

    def get_active_agent(self) -> DQNAgent:
        """Get currently active agent"""
        return self.agents[self.active_gamma]

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using active agent"""
        return self.get_active_agent().act(state, training)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in ALL agents' replay buffers"""
        if self.enabled:
            # Multi-gamma: All agents learn from same experiences
            for agent in self.agents.values():
                agent.remember(state, action, reward, next_state, done)
        else:
            # Single agent
            self.get_active_agent().remember(state, action, reward, next_state, done)

    def replay(self, batch_size: Optional[int] = None):
        """Train ALL agents"""
        losses = {}

        if self.enabled:
            # Train all agents
            for gamma, agent in self.agents.items():
                loss = agent.replay(batch_size)
                losses[gamma] = loss
        else:
            # Single agent
            loss = self.get_active_agent().replay(batch_size)
            losses[self.active_gamma] = loss

        return losses

    def update_epsilon(self):
        """Decay epsilon for all agents"""
        if self.enabled:
            for agent in self.agents.values():
                agent.update_epsilon()
        else:
            self.get_active_agent().update_epsilon()

    def record_trade_duration(self, duration_hours: float):
        """Record trade duration for gamma alignment analysis"""
        self.trade_durations.append(duration_hours)

        # Keep last 100 trades
        if len(self.trade_durations) > 100:
            self.trade_durations.pop(0)

    def get_optimal_gamma_from_trades(self) -> float:
        """Determine optimal gamma based on observed trade durations"""
        if len(self.trade_durations) < 10:
            return self.active_gamma  # Not enough data

        # Calculate median trade duration
        median_duration = np.median(self.trade_durations)

        # Find matching gamma
        for style in TRADING_STYLES.values():
            if style.min_duration_hours <= median_duration < style.max_duration_hours:
                return style.gamma

        # Default to highest gamma for very long trades
        return 0.99

    def maybe_swap_agent(self, episode: int) -> Tuple[bool, Optional[str]]:
        """
        Check if should swap to different gamma agent

        Returns:
            (swapped, rationale)
        """
        # Only enabled in multi-gamma mode
        if not self.enabled:
            return False, None

        # Need minimum episodes before first swap
        if episode < self.min_episodes_before_swap:
            return False, None

        # Only check every 100 episodes
        if self.episodes_since_swap < 100:
            self.episodes_since_swap += 1
            return False, None

        # Need observed trades to determine optimal gamma
        if len(self.trade_durations) < 20:
            return False, None

        # Get optimal gamma based on observed behavior
        optimal_gamma = self.get_optimal_gamma_from_trades()

        # If already using optimal gamma, no swap needed
        if abs(optimal_gamma - self.active_gamma) < 0.02:  # Within 2%
            return False, None

        # Check if candidate agent exists
        if optimal_gamma not in self.agents:
            # Find closest gamma
            optimal_gamma = min(self.agents.keys(), key=lambda g: abs(g - optimal_gamma))

        current_agent = self.agents[self.active_gamma]
        candidate_agent = self.agents[optimal_gamma]

        # Get Sharpe ratios (need recent performance data)
        current_sharpe = self._calculate_agent_sharpe(self.active_gamma)
        candidate_sharpe = self._calculate_agent_sharpe(optimal_gamma)

        if current_sharpe is None or candidate_sharpe is None:
            return False, None

        # Check Sharpe parity (within 10%)
        sharpe_diff = abs(candidate_sharpe - current_sharpe)
        if sharpe_diff > self.sharpe_parity_threshold:
            return False, None  # Not at parity yet

        # Only swap if candidate is at least as good
        if candidate_sharpe < current_sharpe - 0.05:
            return False, None

        # Perform swap
        old_gamma = self.active_gamma
        self.active_gamma = optimal_gamma

        # Build rationale
        avg_duration = np.mean(self.trade_durations)
        rationale = (
            f"🔄 AGENT SWAP: gamma {old_gamma:.2f} → {optimal_gamma:.2f}\n"
            f"   Episode: {episode}\n"
            f"   Observed avg trade duration: {avg_duration:.2f}h\n"
            f"   Optimal gamma for duration: {optimal_gamma:.2f}\n"
            f"   Current Sharpe: {current_sharpe:.2f}\n"
            f"   Candidate Sharpe: {candidate_sharpe:.2f}\n"
            f"   Sharpe difference: {sharpe_diff:.3f} (within {self.sharpe_parity_threshold} threshold)\n"
            f"   Trades analyzed: {len(self.trade_durations)}"
        )

        # Record swap
        self.swap_history.append({
            "episode": episode,
            "old_gamma": old_gamma,
            "new_gamma": optimal_gamma,
            "current_sharpe": current_sharpe,
            "candidate_sharpe": candidate_sharpe,
            "avg_trade_duration": avg_duration,
            "timestamp": datetime.now().isoformat()
        })

        self.episodes_since_swap = 0

        logger.info(rationale)
        return True, rationale

    def _calculate_agent_sharpe(self, gamma: float, window: int = 100) -> Optional[float]:
        """Calculate Sharpe ratio for an agent"""
        if gamma not in self.episode_rewards:
            return None

        rewards = self.episode_rewards[gamma]
        if len(rewards) < 30:
            return None

        recent_rewards = rewards[-window:] if len(rewards) >= window else rewards

        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        if std_reward == 0:
            return 0.0

        sharpe = mean_reward / std_reward
        return sharpe

    def record_episode_reward(self, reward: float):
        """Record episode reward for active agent"""
        self.episode_rewards[self.active_gamma].append(reward)

    def save(self, filepath: str):
        """Save active agent (or all agents in multi-gamma mode)"""
        if self.enabled:
            # Save all agents
            filepath = Path(filepath)
            for gamma, agent in self.agents.items():
                agent_path = filepath.parent / f"{filepath.stem}_gamma{gamma:.2f}{filepath.suffix}"
                agent.save(str(agent_path))

            # Save swap history
            history_path = filepath.parent / f"{filepath.stem}_swap_history.json"
            import json
            with open(history_path, 'w') as f:
                json.dump({
                    "active_gamma": self.active_gamma,
                    "swap_history": self.swap_history
                }, f, indent=2)

            logger.info(f"✅ Saved {len(self.agents)} agents to {filepath.parent}")
        else:
            # Save single agent
            self.get_active_agent().save(filepath)

    def load(self, filepath: str):
        """Load agents from files"""
        if self.enabled:
            filepath = Path(filepath)
            for gamma in self.agents.keys():
                agent_path = filepath.parent / f"{filepath.stem}_gamma{gamma:.2f}{filepath.suffix}"
                if agent_path.exists():
                    self.agents[gamma].load(str(agent_path))

            logger.info(f"✅ Loaded {len(self.agents)} agents from {filepath.parent}")
        else:
            self.get_active_agent().load(filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            "enabled": self.enabled,
            "active_gamma": self.active_gamma,
            "gammas": self.gammas,
            "hardware_sufficient": self.hardware_sufficient,
            "swaps_performed": len(self.swap_history),
            "trade_durations_recorded": len(self.trade_durations)
        }

        if self.enabled:
            # Add per-agent stats
            for gamma in self.gammas:
                rewards = self.episode_rewards[gamma]
                if len(rewards) > 0:
                    stats[f"gamma_{gamma:.2f}_episodes"] = len(rewards)
                    stats[f"gamma_{gamma:.2f}_avg_reward"] = np.mean(rewards[-100:])

                    sharpe = self._calculate_agent_sharpe(gamma)
                    if sharpe is not None:
                        stats[f"gamma_{gamma:.2f}_sharpe"] = sharpe

        return stats


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "MultiGammaTrainer",
    "HardwareBenchmark",
]


# Demo if run as script
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-GAMMA TRAINER DEMO")
    print("="*70 + "\n")

    # Run hardware benchmark
    benchmark = HardwareBenchmark()
    meets_req, reason = benchmark.check_requirements()
    print(reason)
    print(f"\nCan run multi-gamma training: {meets_req}")
