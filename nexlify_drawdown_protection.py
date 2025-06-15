"""
Nexlify Enhanced - Drawdown Protection System
Implements Feature 13: Advanced drawdown management and equity curve trading
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class DrawdownMetrics:
    """Drawdown measurement data"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: timedelta = timedelta()
    drawdown_start: Optional[datetime] = None
    peak_balance: float = 0.0
    recovery_time: Optional[timedelta] = None
    consecutive_losses: int = 0
    
class DrawdownProtection:
    """
    Advanced drawdown protection system with multiple safety mechanisms
    """
    
    def __init__(self, config: Dict):
        """
        Initialize drawdown protection
        
        Args:
            config: Configuration dictionary with thresholds
        """
        # Thresholds
        self.warning_threshold = config.get('warning_threshold', 0.05)  # 5%
        self.critical_threshold = config.get('critical_threshold', 0.10)  # 10%
        self.emergency_threshold = config.get('emergency_threshold', 0.20)  # 20%
        
        # Daily limits
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)  # 5% daily
        self.weekly_loss_limit = config.get('weekly_loss_limit', 0.10)  # 10% weekly
        self.monthly_loss_limit = config.get('monthly_loss_limit', 0.15)  # 15% monthly
        
        # Strategy control
        self.pause_on_drawdown = config.get('pause_on_drawdown', True)
        self.reduce_size_on_drawdown = config.get('reduce_size_on_drawdown', True)
        self.equity_curve_trading = config.get('equity_curve_trading', True)
        
        # State tracking
        self.balance_history = deque(maxlen=1000)
        self.daily_pnl = {}
        self.metrics = DrawdownMetrics()
        self.is_paused = False
        self.position_multiplier = 1.0
        
        # Equity curve MA
        self.equity_ma_period = config.get('equity_ma_period', 20)
        self.equity_curve = deque(maxlen=self.equity_ma_period)
        
    def update(self, current_balance: float, timestamp: Optional[datetime] = None) -> DrawdownLevel:
        """
        Update drawdown metrics with current balance
        
        Args:
            current_balance: Current account balance
            timestamp: Current time (defaults to now)
            
        Returns:
            Current drawdown level
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update balance history
        self.balance_history.append((timestamp, current_balance))
        
        # Update peak balance
        if current_balance > self.metrics.peak_balance:
            self.metrics.peak_balance = current_balance
            self.metrics.drawdown_start = None
            self.metrics.drawdown_duration = timedelta()
            
        # Calculate current drawdown
        if self.metrics.peak_balance > 0:
            self.metrics.current_drawdown = (
                (self.metrics.peak_balance - current_balance) / self.metrics.peak_balance
            )
        
        # Update drawdown duration
        if self.metrics.current_drawdown > 0 and self.metrics.drawdown_start is None:
            self.metrics.drawdown_start = timestamp
        elif self.metrics.current_drawdown > 0:
            self.metrics.drawdown_duration = timestamp - self.metrics.drawdown_start
            
        # Update max drawdown
        if self.metrics.current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = self.metrics.current_drawdown
            
        # Update daily P&L
        date_key = timestamp.date()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = {
                'start_balance': current_balance,
                'current_balance': current_balance,
                'pnl': 0.0,
                'pnl_percent': 0.0
            }
        else:
            self.daily_pnl[date_key]['current_balance'] = current_balance
            self.daily_pnl[date_key]['pnl'] = (
                current_balance - self.daily_pnl[date_key]['start_balance']
            )
            self.daily_pnl[date_key]['pnl_percent'] = (
                self.daily_pnl[date_key]['pnl'] / 
                self.daily_pnl[date_key]['start_balance']
            )
            
        # Update equity curve for MA
        self.equity_curve.append(current_balance)
        
        # Determine drawdown level and take action
        level = self._determine_level()
        self._take_protective_action(level)
        
        return level
        
    def _determine_level(self) -> DrawdownLevel:
        """Determine current drawdown severity level"""
        if self.metrics.current_drawdown >= self.emergency_threshold:
            return DrawdownLevel.EMERGENCY
        elif self.metrics.current_drawdown >= self.critical_threshold:
            return DrawdownLevel.CRITICAL
        elif self.metrics.current_drawdown >= self.warning_threshold:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL
            
    def _take_protective_action(self, level: DrawdownLevel):
        """Take protective action based on drawdown level"""
        if level == DrawdownLevel.EMERGENCY:
            logger.error(f"EMERGENCY: Drawdown {self.metrics.current_drawdown:.1%}")
            self.is_paused = True
            self.position_multiplier = 0.0  # No new positions
            
        elif level == DrawdownLevel.CRITICAL:
            logger.warning(f"CRITICAL: Drawdown {self.metrics.current_drawdown:.1%}")
            if self.pause_on_drawdown:
                self.is_paused = True
            if self.reduce_size_on_drawdown:
                self.position_multiplier = 0.25  # 25% position size
                
        elif level == DrawdownLevel.WARNING:
            logger.warning(f"WARNING: Drawdown {self.metrics.current_drawdown:.1%}")
            if self.reduce_size_on_drawdown:
                self.position_multiplier = 0.5  # 50% position size
                
        else:
            # Normal operation
            self.is_paused = False
            self.position_multiplier = 1.0
            
        # Apply equity curve trading filter
        if self.equity_curve_trading:
            self._apply_equity_curve_filter()
            
    def _apply_equity_curve_filter(self):
        """Apply equity curve MA filter to pause trading"""
        if len(self.equity_curve) < self.equity_ma_period:
            return
            
        current_balance = self.equity_curve[-1]
        equity_ma = np.mean(self.equity_curve)
        
        if current_balance < equity_ma * 0.98:  # 2% below MA
            logger.info("Equity curve filter: Pausing trading (below MA)")
            self.is_paused = True
            
    def check_daily_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        today = datetime.now().date()
        if today in self.daily_pnl:
            daily_loss = -self.daily_pnl[today]['pnl_percent']
            if daily_loss > self.daily_loss_limit:
                logger.warning(f"Daily loss limit exceeded: {daily_loss:.1%}")
                self.is_paused = True
                return True
        return False
        
    def check_consecutive_losses(self, trade_results: List[float]) -> bool:
        """
        Check for consecutive losses pattern
        
        Args:
            trade_results: List of recent trade P&L values
            
        Returns:
            True if concerning pattern detected
        """
        if not trade_results:
            return False
            
        # Count consecutive losses
        consecutive_losses = 0
        for result in reversed(trade_results):
            if result < 0:
                consecutive_losses += 1
            else:
                break
                
        self.metrics.consecutive_losses = consecutive_losses
        
        # Pause after 5 consecutive losses
        if consecutive_losses >= 5:
            logger.warning(f"Consecutive losses: {consecutive_losses}")
            self.is_paused = True
            return True
            
        return False
        
    def get_position_sizing_multiplier(self) -> float:
        """
        Get position sizing multiplier based on drawdown
        
        Returns:
            Multiplier for position size (0.0 to 1.0)
        """
        if self.is_paused:
            return 0.0
            
        # Progressive reduction based on drawdown
        if self.metrics.current_drawdown > 0:
            # Kelly-inspired reduction
            reduction = 1.0 - (self.metrics.current_drawdown * 2)
            return max(0.1, min(1.0, reduction * self.position_multiplier))
            
        return self.position_multiplier
        
    def calculate_recovery_metrics(self) -> Dict:
        """Calculate recovery statistics"""
        if self.metrics.current_drawdown == 0 and self.metrics.max_drawdown > 0:
            # We've recovered
            if self.metrics.drawdown_start:
                self.metrics.recovery_time = datetime.now() - self.metrics.drawdown_start
                
        recovery_stats = {
            'current_drawdown': f"{self.metrics.current_drawdown:.1%}",
            'max_drawdown': f"{self.metrics.max_drawdown:.1%}",
            'drawdown_duration': str(self.metrics.drawdown_duration),
            'recovery_time': str(self.metrics.recovery_time) if self.metrics.recovery_time else "N/A",
            'gain_to_recover': f"{self._calculate_gain_to_recover():.1%}",
            'consecutive_losses': self.metrics.consecutive_losses
        }
        
        return recovery_stats
        
    def _calculate_gain_to_recover(self) -> float:
        """Calculate gain needed to recover from current drawdown"""
        if self.metrics.current_drawdown == 0:
            return 0.0
        return self.metrics.current_drawdown / (1 - self.metrics.current_drawdown)
        
    def get_risk_adjusted_parameters(self) -> Dict:
        """Get risk-adjusted trading parameters based on drawdown"""
        params = {
            'max_positions': self._adjust_max_positions(),
            'stop_loss_multiplier': self._adjust_stop_loss(),
            'take_profit_multiplier': self._adjust_take_profit(),
            'trade_frequency': self._adjust_trade_frequency(),
            'allowed_strategies': self._filter_strategies()
        }
        
        return params
        
    def _adjust_max_positions(self) -> int:
        """Adjust maximum concurrent positions based on drawdown"""
        base_positions = 10
        
        if self.metrics.current_drawdown > 0.15:
            return 1  # Only 1 position in severe drawdown
        elif self.metrics.current_drawdown > 0.10:
            return 3
        elif self.metrics.current_drawdown > 0.05:
            return 5
            
        return base_positions
        
    def _adjust_stop_loss(self) -> float:
        """Tighten stop loss during drawdown"""
        if self.metrics.current_drawdown > 0.10:
            return 0.5  # 50% tighter stop loss
        elif self.metrics.current_drawdown > 0.05:
            return 0.75  # 25% tighter
            
        return 1.0  # Normal stop loss
        
    def _adjust_take_profit(self) -> float:
        """Adjust take profit targets during drawdown"""
        if self.metrics.current_drawdown > 0.10:
            return 0.5  # Take profits earlier
        elif self.metrics.current_drawdown > 0.05:
            return 0.75
            
        return 1.0
        
    def _adjust_trade_frequency(self) -> float:
        """Reduce trade frequency during drawdown"""
        if self.metrics.current_drawdown > 0.15:
            return 0.1  # 90% reduction
        elif self.metrics.current_drawdown > 0.10:
            return 0.3  # 70% reduction
        elif self.metrics.current_drawdown > 0.05:
            return 0.6  # 40% reduction
            
        return 1.0
        
    def _filter_strategies(self) -> List[str]:
        """Filter allowed strategies based on risk level"""
        all_strategies = ['arbitrage', 'market_making', 'momentum', 'mean_reversion']
        
        if self.metrics.current_drawdown > 0.10:
            # Only low-risk strategies
            return ['arbitrage']
        elif self.metrics.current_drawdown > 0.05:
            # No high-risk strategies
            return ['arbitrage', 'market_making']
            
        return all_strategies
        
    def export_report(self) -> str:
        """Export drawdown protection report"""
        report = f"""
# Drawdown Protection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Status
- Drawdown Level: {self._determine_level().value.upper()}
- Trading Status: {'PAUSED' if self.is_paused else 'ACTIVE'}
- Position Multiplier: {self.position_multiplier:.0%}

## Metrics
{self._format_metrics()}

## Risk Parameters
{self._format_risk_params()}

## Recent Performance
{self._format_recent_performance()}
"""
        return report
        
    def _format_metrics(self) -> str:
        """Format metrics for report"""
        metrics = self.calculate_recovery_metrics()
        return '\n'.join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in metrics.items()])
        
    def _format_risk_params(self) -> str:
        """Format risk parameters for report"""
        params = self.get_risk_adjusted_parameters()
        return '\n'.join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in params.items()])
        
    def _format_recent_performance(self) -> str:
        """Format recent performance data"""
        recent_days = list(self.daily_pnl.items())[-7:]  # Last 7 days
        lines = []
        
        for date, data in recent_days:
            lines.append(
                f"- {date}: {data['pnl_percent']:.1%} "
                f"({'${:.2f}'.format(data['pnl'])})"
            )
            
        return '\n'.join(lines) if lines else "No recent data"