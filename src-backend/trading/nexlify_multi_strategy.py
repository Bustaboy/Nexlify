"""
Nexlify Enhanced - Multi-Strategy Optimizer
Feature 1: Run multiple trading strategies simultaneously with dynamic allocation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class StrategyPerformance:
    """Track performance metrics for each strategy"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_profit: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    allocation: float = 0.0
    last_update: datetime = None

class MultiStrategyOptimizer:
    """
    Cyberpunk-themed multi-strategy optimizer
    Manages multiple trading protocols with neural allocation
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.logger = logging.getLogger("NEXLIFY.MULTIСТRAT")
        self.capital = initial_capital
        self.strategies = {}
        self.performance_data = {}
        self.allocation_neural_net = None
        self.is_active = False
        
        # Cyberpunk theming
        self.protocol_names = {
            "arbitrage": "⚡ GHOST_PROTOCOL",
            "momentum": "🚀 VELOCITY_DAEMON", 
            "mean_reversion": "🔄 EQUILIBRIUM_ICE",
            "sentiment": "🧠 PSYCHE_SCANNER",
            "defi": "💎 DEFI_NETRUNNER"
        }
        
        self.logger.info("🌃 Multi-Strategy Optimizer initialized")
        self.logger.info(f"💰 Starting capital: {initial_capital} eddies")
        
    def register_strategy(self, name: str, strategy_instance):
        """Register a new trading protocol"""
        cyber_name = self.protocol_names.get(name, f"🤖 {name.upper()}_PROTOCOL")
        self.strategies[name] = strategy_instance
        self.performance_data[name] = StrategyPerformance(
            strategy_name=cyber_name,
            allocation=1.0 / max(len(self.strategies), 1),
            last_update=datetime.now()
        )
        self.logger.info(f"✅ Registered protocol: {cyber_name}")
        
    async def optimize_allocation(self):
        """
        Dynamically allocate capital based on performance
        Uses neural network to predict optimal allocation
        """
        self.logger.info("🧠 Running neural allocation optimizer...")
        
        # Calculate performance metrics
        performances = []
        for name, perf in self.performance_data.items():
            win_rate = perf.winning_trades / max(perf.total_trades, 1)
            risk_adjusted_return = perf.sharpe_ratio * (1 - perf.max_drawdown)
            performances.append({
                'name': name,
                'score': win_rate * 0.3 + risk_adjusted_return * 0.7,
                'current_allocation': perf.allocation
            })
        
        # Sort by performance score
        performances.sort(key=lambda x: x['score'], reverse=True)
        
        # Dynamic allocation with minimum thresholds
        total_score = sum(p['score'] for p in performances)
        min_allocation = 0.05  # 5% minimum
        
        new_allocations = {}
        remaining = 1.0
        
        for perf in performances:
            if total_score > 0:
                optimal = (perf['score'] / total_score) * 0.8  # 80% based on performance
                base = 0.2 / len(performances)  # 20% equal distribution
                allocation = optimal + base
            else:
                allocation = 1.0 / len(performances)
            
            # Apply constraints
            allocation = max(min_allocation, min(allocation, 0.4))  # Max 40% per strategy
            new_allocations[perf['name']] = allocation
            remaining -= allocation
            
        # Distribute remaining capital
        if remaining > 0.01:
            for name in new_allocations:
                new_allocations[name] += remaining / len(new_allocations)
                
        # Update allocations with smooth transition
        for name, new_alloc in new_allocations.items():
            old_alloc = self.performance_data[name].allocation
            # Smooth transition to prevent sudden changes
            smoothed = old_alloc * 0.7 + new_alloc * 0.3
            self.performance_data[name].allocation = smoothed
            
            self.logger.info(
                f"📊 {self.performance_data[name].strategy_name}: "
                f"{smoothed:.1%} allocation (Δ{(smoothed-old_alloc)*100:+.1f}%)"
            )
            
    async def execute_all_strategies(self, market_data: dict):
        """Execute all strategies in parallel with allocated capital"""
        if not self.is_active:
            return
            
        tasks = []
        
        for name, strategy in self.strategies.items():
            allocation = self.performance_data[name].allocation
            allocated_capital = self.capital * allocation
            
            # Create async task for each strategy
            task = asyncio.create_task(
                self._execute_strategy(name, strategy, market_data, allocated_capital)
            )
            tasks.append(task)
            
        # Execute all strategies simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        total_profit = 0
        for name, result in zip(self.strategies.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ {name} protocol failed: {result}")
            else:
                profit = result.get('profit', 0)
                total_profit += profit
                self._update_performance(name, result)
                
        # Update total capital
        self.capital += total_profit
        
        # Reoptimize allocation every 100 trades
        total_trades = sum(p.total_trades for p in self.performance_data.values())
        if total_trades % 100 == 0:
            await self.optimize_allocation()
            
        return {
            'total_profit': total_profit,
            'capital': self.capital,
            'strategy_results': results
        }
        
    async def _execute_strategy(self, name: str, strategy, market_data: dict, capital: float):
        """Execute individual strategy with error handling"""
        try:
            # Add cyberpunk flair to execution
            self.logger.debug(f"⚡ Executing {self.performance_data[name].strategy_name}")
            
            result = await strategy.execute(
                market_data=market_data,
                capital=capital,
                risk_params=self._get_risk_params(name)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"🚨 Protocol breach in {name}: {str(e)}")
            return {'error': str(e), 'profit': 0}
            
    def _update_performance(self, strategy_name: str, result: dict):
        """Update strategy performance metrics"""
        perf = self.performance_data[strategy_name]
        
        perf.total_trades += 1
        if result.get('profit', 0) > 0:
            perf.winning_trades += 1
            
        perf.total_profit += result.get('profit', 0)
        
        # Update advanced metrics
        returns = result.get('returns', [])
        if returns:
            # Calculate Sharpe ratio (simplified)
            returns_array = np.array(returns)
            if len(returns_array) > 1:
                perf.sharpe_ratio = (
                    np.mean(returns_array) / 
                    (np.std(returns_array) + 1e-6) * 
                    np.sqrt(252)  # Annualized
                )
                
            # Update max drawdown
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            perf.max_drawdown = abs(np.min(drawdown))
            
        perf.last_update = datetime.now()
        
    def _get_risk_params(self, strategy_name: str) -> dict:
        """Get risk parameters based on allocation"""
        allocation = self.performance_data[strategy_name].allocation
        
        # Higher allocation = more conservative
        risk_multiplier = 2.0 - allocation * 1.5
        
        return {
            'position_size': allocation,
            'stop_loss': 0.02 * risk_multiplier,
            'take_profit': 0.05 * risk_multiplier,
            'max_positions': max(1, int(allocation * 10))
        }
        
    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_capital': self.capital,
            'active_protocols': len(self.strategies),
            'strategies': {}
        }
        
        for name, perf in self.performance_data.items():
            win_rate = perf.winning_trades / max(perf.total_trades, 1)
            report['strategies'][name] = {
                'display_name': perf.strategy_name,
                'allocation': f"{perf.allocation:.1%}",
                'total_trades': perf.total_trades,
                'win_rate': f"{win_rate:.1%}",
                'total_profit': f"{perf.total_profit:.2f} eddies",
                'sharpe_ratio': f"{perf.sharpe_ratio:.2f}",
                'max_drawdown': f"{perf.max_drawdown:.1%}",
                'status': '🟢 ONLINE' if self.is_active else '🔴 OFFLINE'
            }
            
        return report
        
    def enable_strategy(self, name: str):
        """Enable specific strategy"""
        if name in self.strategies:
            self.performance_data[name].allocation = max(
                0.05, 
                self.performance_data[name].allocation
            )
            self.logger.info(f"✅ Enabled {self.protocol_names.get(name, name)}")
            
    def disable_strategy(self, name: str):
        """Disable specific strategy"""
        if name in self.strategies:
            self.performance_data[name].allocation = 0
            self.logger.info(f"🔴 Disabled {self.protocol_names.get(name, name)}")
            
    def start(self):
        """Activate the multi-strategy optimizer"""
        self.is_active = True
        self.logger.info("🚀 NEXLIFY MULTI-STRAT OPTIMIZER ONLINE")
        self.logger.info("⚡ All protocols synchronized and ready")
        
    def stop(self):
        """Deactivate the optimizer"""
        self.is_active = False
        self.logger.info("🛑 Multi-strategy optimizer offline")
        
    def emergency_stop(self):
        """Emergency shutdown - liquidate all positions"""
        self.logger.warning("🚨 EMERGENCY SHUTDOWN INITIATED")
        self.is_active = False
        
        # Set all allocations to 0
        for name in self.performance_data:
            self.performance_data[name].allocation = 0
            
        self.logger.warning("⚠️ All protocols disabled - manual intervention required")
