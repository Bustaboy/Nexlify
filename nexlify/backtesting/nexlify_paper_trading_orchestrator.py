#!/usr/bin/env python3
"""
Nexlify Paper Trading Orchestrator

Multi-agent paper trading system that can run and compare multiple RL/ML strategies
simultaneously with real-time performance tracking and analysis.

Features:
- Run multiple RL agents simultaneously
- Compare performance across different strategies
- Real-time performance tracking and visualization
- Continuous learning integration
- Comprehensive reporting and analytics
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nexlify.backtesting.nexlify_paper_trading import (PaperTrade,
                                                       PaperTradingEngine)
from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class AgentConfig:
    """Configuration for a single trading agent"""

    agent_id: str
    agent_type: (
        str  # 'rl_basic', 'rl_adaptive', 'rl_ultra', 'ml_ensemble', 'rule_based'
    )
    name: str
    model_path: Optional[str] = None
    config: Dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""

    timestamp: datetime
    agent_id: str
    balance: float
    equity: float
    total_return: float
    total_return_percent: float
    open_positions: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float


class PaperTradingOrchestrator:
    """
    Orchestrates multiple trading agents in paper trading mode

    Manages multiple strategies simultaneously, tracks performance,
    enables comparison, and supports continuous learning.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize orchestrator

        Args:
            config: Configuration dict with orchestrator settings
        """
        self.config = config or {}

        # Orchestrator settings
        self.initial_balance = self.config.get("initial_balance", 10000.0)
        self.fee_rate = self.config.get("fee_rate", 0.001)
        self.slippage = self.config.get("slippage", 0.0005)
        self.update_interval = self.config.get("update_interval", 60)  # seconds

        # Agents
        self.agents: Dict[str, AgentConfig] = {}
        self.agent_engines: Dict[str, PaperTradingEngine] = {}
        self.agent_instances: Dict[str, Any] = {}  # Actual RL/ML agent instances

        # Performance tracking
        self.performance_history: Dict[str, List[PerformanceSnapshot]] = defaultdict(
            list
        )
        self.comparison_metrics: Dict[str, Any] = {}

        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("🎯 Paper Trading Orchestrator initialized")
        logger.info(f"   Session ID: {self.session_id}")
        logger.info(f"   Initial balance per agent: ${self.initial_balance:,.2f}")

    def register_agent(self, agent_config: AgentConfig):
        """
        Register a new trading agent

        Args:
            agent_config: Agent configuration
        """
        if agent_config.agent_id in self.agents:
            logger.warning(
                f"Agent {agent_config.agent_id} already registered, overwriting"
            )

        self.agents[agent_config.agent_id] = agent_config

        # Create paper trading engine for this agent
        self.agent_engines[agent_config.agent_id] = PaperTradingEngine(
            {
                "paper_balance": self.initial_balance,
                "fee_rate": self.fee_rate,
                "slippage": self.slippage,
            }
        )

        logger.info(
            f"✅ Registered agent: {agent_config.name} ({agent_config.agent_type})"
        )

    def load_agent_model(self, agent_id: str, agent_instance: Any):
        """
        Load and initialize an agent's model

        Args:
            agent_id: Agent identifier
            agent_instance: Initialized agent instance (RL agent, ML model, etc.)
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        self.agent_instances[agent_id] = agent_instance
        logger.info(f"✅ Loaded model for agent: {agent_id}")

    async def start_session(self, duration_hours: Optional[float] = None):
        """
        Start paper trading session

        Args:
            duration_hours: Session duration in hours (None = run indefinitely)
        """
        if not self.agents:
            raise ValueError(
                "No agents registered. Register at least one agent before starting."
            )

        self.is_running = True
        self.start_time = datetime.now()

        end_time = None
        if duration_hours:
            end_time = self.start_time + timedelta(hours=duration_hours)
            logger.info(
                f"🚀 Starting paper trading session (duration: {duration_hours:.1f}h)"
            )
        else:
            logger.info("🚀 Starting paper trading session (indefinite)")

        logger.info(
            f"   Active agents: {len([a for a in self.agents.values() if a.enabled])}"
        )

        try:
            while self.is_running:
                # Check if session should end
                if end_time and datetime.now() >= end_time:
                    logger.info("⏰ Session duration reached")
                    break

                # Update cycle
                await self._update_cycle()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Session interrupted by user")
        except Exception as e:
            logger.error(f"❌ Session error: {e}")
            raise
        finally:
            await self.stop_session()

    async def _update_cycle(self):
        """Single update cycle for all agents"""
        try:
            # For each active agent
            for agent_id, agent_config in self.agents.items():
                if not agent_config.enabled:
                    continue

                # Get current market data (would be real in production)
                market_data = await self._get_market_data(agent_config)

                if market_data is None:
                    continue

                # Get agent instance
                agent_instance = self.agent_instances.get(agent_id)
                if agent_instance is None:
                    logger.warning(f"Agent {agent_id} has no loaded model, skipping")
                    continue

                # Agent makes decision
                decision = await self._get_agent_decision(
                    agent_id, agent_instance, market_data
                )

                # Execute decision in paper trading
                await self._execute_decision(agent_id, decision, market_data)

                # Update positions with current prices
                paper_engine = self.agent_engines[agent_id]
                await paper_engine.update_positions(market_data["prices"])

                # Record performance snapshot
                self._record_performance_snapshot(agent_id)

            # Update comparison metrics
            self._update_comparison_metrics()

            # Log progress
            self._log_progress()

        except Exception as e:
            logger.error(f"Update cycle error: {e}")

    async def _get_market_data(self, agent_config: AgentConfig) -> Optional[Dict]:
        """
        Get current market data for agent

        In production, this would fetch real-time data from exchanges.
        For now, returns mock data structure.

        Returns:
            Dict with market data: prices, volumes, indicators, etc.
        """
        # TODO: Integrate with real market data feed
        # For now, return None to indicate no data available
        return None

    async def _get_agent_decision(
        self, agent_id: str, agent_instance: Any, market_data: Dict
    ) -> Dict:
        """
        Get trading decision from agent

        Args:
            agent_id: Agent identifier
            agent_instance: Agent instance
            market_data: Current market data

        Returns:
            Decision dict with action, amount, confidence, etc.
        """
        try:
            agent_config = self.agents[agent_id]

            # Prepare state for agent
            state = self._prepare_state(agent_id, market_data)

            # Get action from agent based on type
            if agent_config.agent_type.startswith("rl_"):
                # RL agent (has act method)
                action = agent_instance.act(state, training=False)

                # Convert action to decision
                decision = self._action_to_decision(action, market_data)

            elif agent_config.agent_type == "ml_ensemble":
                # ML ensemble prediction
                prediction = agent_instance.predict(state)
                decision = self._prediction_to_decision(prediction, market_data)

            else:
                # Unknown agent type
                logger.warning(f"Unknown agent type: {agent_config.agent_type}")
                decision = {"action": "hold"}

            return decision

        except Exception as e:
            logger.error(f"Error getting decision from agent {agent_id}: {e}")
            return {"action": "hold"}

    def _prepare_state(self, agent_id: str, market_data: Dict) -> np.ndarray:
        """
        Prepare state vector for agent from market data

        Args:
            agent_id: Agent identifier
            market_data: Current market data

        Returns:
            State vector as numpy array
        """
        # Get paper trading engine
        paper_engine = self.agent_engines[agent_id]

        # Build state vector (12 features (crypto-optimized) matching RL agent expectations)
        state = np.array(
            [
                paper_engine.current_balance
                / paper_engine.initial_balance,  # Normalized balance
                len(paper_engine.positions),  # Position size
                1.0,  # Relative entry price (default)
                market_data["prices"].get("BTC/USDT", 0)
                / paper_engine.initial_balance,  # Normalized price
                market_data.get("price_change", 0),  # Price change
                market_data.get("rsi", 50) / 100,  # RSI normalized
                market_data.get("macd", 0),  # MACD
                market_data.get("volume_ratio", 1.0),  # Volume ratio
            ]
        )

        return state

    def _action_to_decision(self, action: int, market_data: Dict) -> Dict:
        """
        Convert RL action to trading decision

        Args:
            action: Action index (0=buy, 1=sell, 2=hold)
            market_data: Current market data

        Returns:
            Decision dict
        """
        if action == 0:
            return {
                "action": "buy",
                "symbol": "BTC/USDT",
                "amount": 0.01,  # Default amount
                "price": market_data["prices"]["BTC/USDT"],
            }
        elif action == 1:
            return {
                "action": "sell",
                "symbol": "BTC/USDT",
                "amount": 0.01,
                "price": market_data["prices"]["BTC/USDT"],
            }
        else:
            return {"action": "hold"}

    def _prediction_to_decision(self, prediction: float, market_data: Dict) -> Dict:
        """
        Convert ML prediction to trading decision

        Args:
            prediction: Predicted price movement
            market_data: Current market data

        Returns:
            Decision dict
        """
        if prediction > 0.02:  # 2% predicted increase
            return {
                "action": "buy",
                "symbol": "BTC/USDT",
                "amount": 0.01,
                "price": market_data["prices"]["BTC/USDT"],
            }
        elif prediction < -0.02:  # 2% predicted decrease
            return {
                "action": "sell",
                "symbol": "BTC/USDT",
                "amount": 0.01,
                "price": market_data["prices"]["BTC/USDT"],
            }
        else:
            return {"action": "hold"}

    async def _execute_decision(self, agent_id: str, decision: Dict, market_data: Dict):
        """
        Execute trading decision in paper trading engine

        Args:
            agent_id: Agent identifier
            decision: Trading decision
            market_data: Current market data
        """
        if decision["action"] == "hold":
            return

        paper_engine = self.agent_engines[agent_id]
        agent_name = self.agents[agent_id].name

        try:
            result = await paper_engine.place_order(
                symbol=decision.get("symbol", "BTC/USDT"),
                side=decision["action"],
                amount=decision.get("amount", 0.01),
                price=decision.get("price", market_data["prices"]["BTC/USDT"]),
                strategy=agent_name,
            )

            if result.get("success"):
                logger.debug(f"✅ {agent_name}: {decision['action'].upper()} executed")
            else:
                logger.debug(
                    f"❌ {agent_name}: {decision['action'].upper()} failed - {result.get('error')}"
                )

        except Exception as e:
            logger.error(f"Error executing decision for {agent_name}: {e}")

    def _record_performance_snapshot(self, agent_id: str):
        """Record current performance snapshot for agent"""
        paper_engine = self.agent_engines[agent_id]
        stats = paper_engine.get_statistics()

        # Calculate Sharpe ratio
        # Note: Equity updates are based on snapshot frequency, not fixed timeframe
        # Using hourly annualization (8760) as default since most training uses 1h data
        # For daily data, this would overestimate Sharpe by ~5.9x
        # TODO: Make this configurable based on actual update frequency
        if len(paper_engine.equity_curve) > 1:
            returns = (
                np.diff(paper_engine.equity_curve) / paper_engine.equity_curve[:-1]
            )
            periods_per_year = 8760  # Hourly updates (365 * 24)
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_curve = np.array(paper_engine.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            agent_id=agent_id,
            balance=stats["current_balance"],
            equity=stats["total_equity"],
            total_return=stats["total_return"],
            total_return_percent=stats["total_return_percent"],
            open_positions=stats["open_positions"],
            total_trades=stats["total_trades"],
            win_rate=stats["win_rate"],
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
        )

        self.performance_history[agent_id].append(snapshot)

    def _update_comparison_metrics(self):
        """Update comparison metrics across all agents"""
        if not self.performance_history:
            return

        self.comparison_metrics = {"timestamp": datetime.now(), "agents": {}}

        for agent_id, snapshots in self.performance_history.items():
            if not snapshots:
                continue

            latest = snapshots[-1]
            agent_name = self.agents[agent_id].name

            self.comparison_metrics["agents"][agent_name] = {
                "total_return_percent": latest.total_return_percent,
                "sharpe_ratio": latest.sharpe_ratio,
                "max_drawdown": latest.max_drawdown,
                "win_rate": latest.win_rate,
                "total_trades": latest.total_trades,
                "equity": latest.equity,
            }

    def _log_progress(self):
        """Log current progress of all agents"""
        if not self.comparison_metrics:
            return

        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Paper Trading Progress (Elapsed: {elapsed})")
        logger.info(f"{'='*80}")

        for agent_name, metrics in self.comparison_metrics["agents"].items():
            logger.info(f"\n{agent_name}:")
            logger.info(f"  Return: {metrics['total_return_percent']:>8.2f}%")
            logger.info(f"  Sharpe: {metrics['sharpe_ratio']:>8.2f}")
            logger.info(f"  Max DD: {metrics['max_drawdown']:>8.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']:>6.1f}%")
            logger.info(f"  Trades: {metrics['total_trades']:>4d}")
            logger.info(f"  Equity: ${metrics['equity']:>10,.2f}")

        logger.info(f"\n{'='*80}\n")

    async def stop_session(self):
        """Stop paper trading session"""
        self.is_running = False

        logger.info("🛑 Stopping paper trading session...")

        # Generate final report
        report = self.generate_final_report()

        # Save session data
        self.save_session()

        logger.info("✅ Session stopped")

        return report

    def generate_final_report(self) -> str:
        """Generate comprehensive final report"""
        if not self.start_time:
            return "No session data available"

        elapsed = datetime.now() - self.start_time

        report = f"""
{'='*80}
PAPER TRADING SESSION REPORT
{'='*80}

SESSION INFO
  Session ID:        {self.session_id}
  Duration:          {elapsed}
  Start Time:        {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
  End Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Active Agents:     {len([a for a in self.agents.values() if a.enabled])}

{'='*80}
AGENT PERFORMANCE COMPARISON
{'='*80}
"""

        # Sort agents by return
        if self.comparison_metrics and "agents" in self.comparison_metrics:
            sorted_agents = sorted(
                self.comparison_metrics["agents"].items(),
                key=lambda x: x[1]["total_return_percent"],
                reverse=True,
            )

            for rank, (agent_name, metrics) in enumerate(sorted_agents, 1):
                report += f"""
{rank}. {agent_name}
   Total Return:     {metrics['total_return_percent']:>8.2f}%
   Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}
   Max Drawdown:     {metrics['max_drawdown']:>8.2f}%
   Win Rate:         {metrics['win_rate']:>8.1f}%
   Total Trades:     {metrics['total_trades']:>8d}
   Final Equity:     ${metrics['equity']:>12,.2f}
"""

        report += f"\n{'='*80}\n"

        # Individual agent reports
        report += "\nDETAILED AGENT REPORTS\n"
        report += f"{'='*80}\n"

        for agent_id, agent_config in self.agents.items():
            if not agent_config.enabled:
                continue

            paper_engine = self.agent_engines[agent_id]
            agent_report = paper_engine.generate_report()
            report += f"\n{agent_config.name} ({agent_config.agent_type})\n"
            report += agent_report

        return report

    def save_session(self, directory: str = "paper_trading/sessions"):
        """
        Save session data to files

        Args:
            directory: Output directory for session files
        """
        try:
            output_dir = Path(directory) / self.session_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save session summary
            summary = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "config": self.config,
                "agents": {
                    agent_id: {
                        "name": config.name,
                        "type": config.agent_type,
                        "enabled": config.enabled,
                    }
                    for agent_id, config in self.agents.items()
                },
                "comparison_metrics": self.comparison_metrics,
            }

            with open(output_dir / "session_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            # Save individual agent data
            for agent_id, paper_engine in self.agent_engines.items():
                agent_file = output_dir / f"agent_{agent_id}.json"
                paper_engine.save_session(str(agent_file))

            # Save performance history
            for agent_id, snapshots in self.performance_history.items():
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": s.timestamp,
                            "balance": s.balance,
                            "equity": s.equity,
                            "total_return": s.total_return,
                            "total_return_percent": s.total_return_percent,
                            "open_positions": s.open_positions,
                            "total_trades": s.total_trades,
                            "win_rate": s.win_rate,
                            "sharpe_ratio": s.sharpe_ratio,
                            "max_drawdown": s.max_drawdown,
                        }
                        for s in snapshots
                    ]
                )

                csv_file = output_dir / f"performance_{agent_id}.csv"
                df.to_csv(csv_file, index=False)

            # Save final report
            report = self.generate_final_report()
            with open(output_dir / "final_report.txt", "w") as f:
                f.write(report)

            logger.info(f"💾 Session data saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """
        Get agent leaderboard sorted by performance

        Returns:
            List of (agent_name, return_percent) tuples
        """
        if not self.comparison_metrics or "agents" not in self.comparison_metrics:
            return []

        leaderboard = [
            (name, metrics["total_return_percent"])
            for name, metrics in self.comparison_metrics["agents"].items()
        ]

        return sorted(leaderboard, key=lambda x: x[1], reverse=True)


# Convenience function
def create_orchestrator(config: Optional[Dict] = None) -> PaperTradingOrchestrator:
    """
    Create paper trading orchestrator with default or custom config

    Args:
        config: Optional configuration dict

    Returns:
        Initialized PaperTradingOrchestrator
    """
    return PaperTradingOrchestrator(config)


__all__ = [
    "PaperTradingOrchestrator",
    "AgentConfig",
    "PerformanceSnapshot",
    "create_orchestrator",
]
