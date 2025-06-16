# src/risk/nexlify_drawdown_protection.py
"""
Nexlify Enhanced - Advanced Drawdown Protection System
Comprehensive risk management with multiple safety mechanisms
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import threading
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    """Drawdown severity levels with cyberpunk naming"""
    GREEN = "green_zone"      # Normal operation
    YELLOW = "yellow_alert"   # Warning level
    ORANGE = "orange_alert"   # Critical level
    RED = "red_zone"         # Emergency level
    BLACK = "flatline"       # Total shutdown

class RecoveryMode(Enum):
    """Recovery strategies after drawdown"""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    TURTLE = "turtle"  # Ultra-conservative

@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown measurement data"""
    # Current metrics
    current_drawdown: float = 0.0
    current_drawdown_amount: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_amount: float = 0.0
    
    # Time metrics
    drawdown_start: Optional[datetime] = None
    drawdown_duration: timedelta = timedelta()
    time_underwater: timedelta = timedelta()
    longest_drawdown: timedelta = timedelta()
    
    # Peak/trough tracking
    peak_balance: float = 0.0
    peak_timestamp: Optional[datetime] = None
    trough_balance: float = 0.0
    trough_timestamp: Optional[datetime] = None
    
    # Recovery metrics
    recovery_start: Optional[datetime] = None
    recovery_time: Optional[timedelta] = None
    recovery_rate: float = 0.0
    gain_to_recover: float = 0.0
    
    # Pattern metrics
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    drawdown_count: int = 0
    recovery_count: int = 0
    
    # Risk metrics
    average_drawdown: float = 0.0
    drawdown_volatility: float = 0.0
    calmar_ratio: float = 0.0  # Annual return / Max DD
    sterling_ratio: float = 0.0  # Annual return / Average DD

@dataclass
class ProtectionRule:
    """Configurable protection rule"""
    name: str
    condition: Callable
    action: Callable
    priority: int = 0
    enabled: bool = True
    description: str = ""
    
class NexlifyDrawdownProtection:
    """
    🛡️ NEXLIFY DRAWDOWN PROTECTION MATRIX
    Advanced equity protection with neural monitoring
    """
    
    def __init__(self, config: Dict, risk_manager=None):
        """
        Initialize the drawdown protection system
        
        Args:
            config: Configuration dictionary
            risk_manager: Reference to main risk manager
        """
        # Configuration
        self.config = self._validate_config(config)
        self.risk_manager = risk_manager
        
        # Thresholds
        self.thresholds = {
            DrawdownLevel.YELLOW: config.get('yellow_threshold', 0.05),    # 5%
            DrawdownLevel.ORANGE: config.get('orange_threshold', 0.10),    # 10%
            DrawdownLevel.RED: config.get('red_threshold', 0.15),          # 15%
            DrawdownLevel.BLACK: config.get('black_threshold', 0.25)       # 25%
        }
        
        # Time-based limits
        self.time_limits = {
            'hourly': config.get('hourly_loss_limit', 0.02),    # 2%
            'daily': config.get('daily_loss_limit', 0.05),      # 5%
            'weekly': config.get('weekly_loss_limit', 0.10),    # 10%
            'monthly': config.get('monthly_loss_limit', 0.15)   # 15%
        }
        
        # Protection settings
        self.settings = {
            'pause_on_drawdown': config.get('pause_on_drawdown', True),
            'reduce_size_on_drawdown': config.get('reduce_size_on_drawdown', True),
            'equity_curve_trading': config.get('equity_curve_trading', True),
            'martingale_protection': config.get('martingale_protection', True),
            'correlation_protection': config.get('correlation_protection', True),
            'volatility_scaling': config.get('volatility_scaling', True),
            'auto_deleverage': config.get('auto_deleverage', True),
            'panic_mode_enabled': config.get('panic_mode_enabled', True)
        }
        
        # State tracking
        self.metrics = DrawdownMetrics()
        self.current_level = DrawdownLevel.GREEN
        self.recovery_mode = RecoveryMode.MODERATE
        self.is_paused = False
        self.position_multiplier = 1.0
        self.allowed_pairs = []  # Restricted trading pairs
        
        # Historical data
        self.balance_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=1000)
        self.hourly_pnl = {}
        self.daily_pnl = {}
        
        # Moving averages for equity curve
        self.equity_ma_short = config.get('equity_ma_short', 10)
        self.equity_ma_long = config.get('equity_ma_long', 30)
        self.equity_curve = pd.Series(dtype=float)
        
        # Protection rules
        self.protection_rules = self._initialize_protection_rules()
        
        # Neural monitoring
        self.neural_monitor = DrawdownNeuralMonitor(self)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Persistence
        self.persistence_file = Path("data/drawdown_state.json")
        self._load_state()
        
        # Cyberpunk logging
        logger.info("🛡️ DRAWDOWN PROTECTION MATRIX INITIALIZED")
        logger.info(f"⚡ Protection thresholds: {self._format_thresholds()}")
        
    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration"""
        defaults = {
            'yellow_threshold': 0.05,
            'orange_threshold': 0.10,
            'red_threshold': 0.15,
            'black_threshold': 0.25,
            'daily_loss_limit': 0.05,
            'pause_on_drawdown': True,
            'equity_ma_short': 10,
            'equity_ma_long': 30
        }
        
        validated = defaults.copy()
        validated.update(config)
        return validated
        
    def _initialize_protection_rules(self) -> List[ProtectionRule]:
        """Initialize all protection rules"""
        rules = [
            # Consecutive loss protection
            ProtectionRule(
                name="consecutive_loss_protection",
                condition=lambda: self.metrics.consecutive_losses >= 5,
                action=self._pause_trading,
                priority=1,
                description="Pause after 5 consecutive losses"
            ),
            
            # Daily loss limit
            ProtectionRule(
                name="daily_loss_limit",
                condition=self._check_daily_limit,
                action=self._emergency_stop,
                priority=2,
                description="Emergency stop on daily limit breach"
            ),
            
            # Volatility spike protection
            ProtectionRule(
                name="volatility_protection",
                condition=self._check_volatility_spike,
                action=self._reduce_position_size,
                priority=3,
                description="Reduce size on volatility spike"
            ),
            
            # Correlation breakdown
            ProtectionRule(
                name="correlation_protection",
                condition=self._check_correlation_breakdown,
                action=self._restrict_correlated_pairs,
                priority=4,
                description="Restrict correlated pairs"
            ),
            
            # Martingale detection
            ProtectionRule(
                name="martingale_protection",
                condition=self._detect_martingale,
                action=self._reset_position_sizes,
                priority=5,
                description="Prevent martingale behavior"
            ),
            
            # Black swan protection
            ProtectionRule(
                name="black_swan_protection",
                condition=lambda: self.metrics.current_drawdown > 0.20,
                action=self._activate_panic_mode,
                priority=10,
                description="Panic mode for extreme events"
            )
        ]
        
        return sorted(rules, key=lambda r: r.priority)
        
    def update(self, current_balance: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        🔄 Update drawdown metrics with current balance
        
        Args:
            current_balance: Current account balance
            timestamp: Current time (defaults to now)
            
        Returns:
            Dict with current status and actions taken
        """
        with self.lock:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Store previous state
            prev_level = self.current_level
            prev_multiplier = self.position_multiplier
            
            # Update balance history
            self.balance_history.append((timestamp, current_balance))
            
            # Update peak and calculate drawdown
            self._update_peak_and_drawdown(current_balance, timestamp)
            
            # Update time-based P&L
            self._update_time_based_pnl(current_balance, timestamp)
            
            # Update equity curve
            self._update_equity_curve(current_balance, timestamp)
            
            # Determine new level
            self.current_level = self._determine_level()
            
            # Apply protection rules
            actions_taken = self._apply_protection_rules()
            
            # Take level-based actions
            level_actions = self._take_level_action(self.current_level)
            actions_taken.extend(level_actions)
            
            # Neural monitoring
            neural_signals = self.neural_monitor.analyze()
            if neural_signals.get('danger_detected'):
                actions_taken.append("Neural danger signal detected")
                
            # Log state changes
            if self.current_level != prev_level:
                logger.warning(
                    f"⚡ DRAWDOWN LEVEL CHANGE: {prev_level.value} → {self.current_level.value}"
                )
                
            if self.position_multiplier != prev_multiplier:
                logger.info(
                    f"📊 Position multiplier adjusted: {prev_multiplier:.0%} → "
                    f"{self.position_multiplier:.0%}"
                )
                
            # Save state
            self._save_state()
            
            # Return comprehensive status
            return {
                'level': self.current_level,
                'drawdown': self.metrics.current_drawdown,
                'is_paused': self.is_paused,
                'position_multiplier': self.position_multiplier,
                'recovery_mode': self.recovery_mode,
                'actions_taken': actions_taken,
                'metrics': self._get_summary_metrics(),
                'neural_signals': neural_signals
            }
            
    def _update_peak_and_drawdown(self, current_balance: float, timestamp: datetime):
        """Update peak balance and drawdown metrics"""
        # Update peak
        if current_balance > self.metrics.peak_balance:
            self.metrics.peak_balance = current_balance
            self.metrics.peak_timestamp = timestamp
            
            # Reset drawdown metrics on new peak
            if self.metrics.current_drawdown > 0:
                self._record_drawdown_end(timestamp)
                
            self.metrics.drawdown_start = None
            self.metrics.current_drawdown = 0.0
            self.metrics.current_drawdown_amount = 0.0
            
        # Calculate drawdown
        if self.metrics.peak_balance > 0:
            self.metrics.current_drawdown = (
                (self.metrics.peak_balance - current_balance) / self.metrics.peak_balance
            )
            self.metrics.current_drawdown_amount = (
                self.metrics.peak_balance - current_balance
            )
            
        # Update drawdown tracking
        if self.metrics.current_drawdown > 0:
            if self.metrics.drawdown_start is None:
                self.metrics.drawdown_start = timestamp
                self.metrics.drawdown_count += 1
                
            self.metrics.drawdown_duration = timestamp - self.metrics.drawdown_start
            self.metrics.time_underwater += timedelta(minutes=1)  # Approximate
            
            # Update trough
            if current_balance < self.metrics.trough_balance or self.metrics.trough_balance == 0:
                self.metrics.trough_balance = current_balance
                self.metrics.trough_timestamp = timestamp
                
        # Update max drawdown
        if self.metrics.current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = self.metrics.current_drawdown
            self.metrics.max_drawdown_amount = self.metrics.current_drawdown_amount
            self.metrics.longest_drawdown = self.metrics.drawdown_duration
            
        # Store in history
        self.drawdown_history.append({
            'timestamp': timestamp,
            'drawdown': self.metrics.current_drawdown,
            'balance': current_balance
        })
        
    def _update_time_based_pnl(self, current_balance: float, timestamp: datetime):
        """Update hourly and daily P&L tracking"""
        # Hourly P&L
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        if hour_key not in self.hourly_pnl:
            self.hourly_pnl[hour_key] = {
                'start_balance': current_balance,
                'current_balance': current_balance,
                'pnl': 0.0,
                'pnl_percent': 0.0
            }
        else:
            self.hourly_pnl[hour_key]['current_balance'] = current_balance
            self.hourly_pnl[hour_key]['pnl'] = (
                current_balance - self.hourly_pnl[hour_key]['start_balance']
            )
            if self.hourly_pnl[hour_key]['start_balance'] > 0:
                self.hourly_pnl[hour_key]['pnl_percent'] = (
                    self.hourly_pnl[hour_key]['pnl'] / 
                    self.hourly_pnl[hour_key]['start_balance']
                )
                
        # Daily P&L
        date_key = timestamp.date()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = {
                'start_balance': current_balance,
                'current_balance': current_balance,
                'high': current_balance,
                'low': current_balance,
                'pnl': 0.0,
                'pnl_percent': 0.0
            }
        else:
            daily = self.daily_pnl[date_key]
            daily['current_balance'] = current_balance
            daily['high'] = max(daily['high'], current_balance)
            daily['low'] = min(daily['low'], current_balance)
            daily['pnl'] = current_balance - daily['start_balance']
            if daily['start_balance'] > 0:
                daily['pnl_percent'] = daily['pnl'] / daily['start_balance']
                
    def _update_equity_curve(self, current_balance: float, timestamp: datetime):
        """Update equity curve for moving average calculations"""
        # Add to series
        self.equity_curve = pd.concat([
            self.equity_curve,
            pd.Series([current_balance], index=[timestamp])
        ]).iloc[-1000:]  # Keep last 1000 points
        
    def _determine_level(self) -> DrawdownLevel:
        """Determine current drawdown severity level"""
        dd = self.metrics.current_drawdown
        
        if dd >= self.thresholds[DrawdownLevel.BLACK]:
            return DrawdownLevel.BLACK
        elif dd >= self.thresholds[DrawdownLevel.RED]:
            return DrawdownLevel.RED
        elif dd >= self.thresholds[DrawdownLevel.ORANGE]:
            return DrawdownLevel.ORANGE
        elif dd >= self.thresholds[DrawdownLevel.YELLOW]:
            return DrawdownLevel.YELLOW
        else:
            return DrawdownLevel.GREEN
            
    def _apply_protection_rules(self) -> List[str]:
        """Apply all active protection rules"""
        actions_taken = []
        
        for rule in self.protection_rules:
            if rule.enabled and rule.condition():
                try:
                    rule.action()
                    actions_taken.append(f"Applied: {rule.name}")
                    logger.info(f"🛡️ Protection rule triggered: {rule.description}")
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.name}: {e}")
                    
        return actions_taken
        
    def _take_level_action(self, level: DrawdownLevel) -> List[str]:
        """Take action based on drawdown level"""
        actions = []
        
        if level == DrawdownLevel.BLACK:
            # FLATLINE - Complete shutdown
            logger.critical("💀 FLATLINE: Emergency shutdown activated")
            self.is_paused = True
            self.position_multiplier = 0.0
            self.recovery_mode = RecoveryMode.TURTLE
            actions.append("EMERGENCY SHUTDOWN - All trading halted")
            
            # Notify risk manager
            if self.risk_manager:
                self.risk_manager.emergency_stop("Drawdown flatline")
                
        elif level == DrawdownLevel.RED:
            # RED ZONE - Severe restrictions
            logger.error(f"🔴 RED ZONE: Drawdown {self.metrics.current_drawdown:.1%}")
            self.is_paused = self.settings['pause_on_drawdown']
            self.position_multiplier = 0.1  # 10% position size
            self.recovery_mode = RecoveryMode.CONSERVATIVE
            actions.append("RED ZONE - Severe trading restrictions")
            
        elif level == DrawdownLevel.ORANGE:
            # ORANGE ALERT - Significant restrictions
            logger.warning(f"🟠 ORANGE ALERT: Drawdown {self.metrics.current_drawdown:.1%}")
            self.position_multiplier = 0.25  # 25% position size
            self.recovery_mode = RecoveryMode.CONSERVATIVE
            actions.append("ORANGE ALERT - Position sizes reduced to 25%")
            
        elif level == DrawdownLevel.YELLOW:
            # YELLOW ALERT - Caution mode
            logger.warning(f"🟡 YELLOW ALERT: Drawdown {self.metrics.current_drawdown:.1%}")
            self.position_multiplier = 0.5  # 50% position size
            self.recovery_mode = RecoveryMode.MODERATE
            actions.append("YELLOW ALERT - Position sizes reduced to 50%")
            
        else:
            # GREEN ZONE - Normal operation
            self.is_paused = False
            self.position_multiplier = 1.0
            self.recovery_mode = RecoveryMode.MODERATE
            
        # Apply equity curve filter
        if self.settings['equity_curve_trading']:
            ec_action = self._apply_equity_curve_filter()
            if ec_action:
                actions.append(ec_action)
                
        return actions
        
    def _apply_equity_curve_filter(self) -> Optional[str]:
        """Apply equity curve MA filter"""
        if len(self.equity_curve) < self.equity_ma_long:
            return None
            
        # Calculate moving averages
        ma_short = self.equity_curve.iloc[-self.equity_ma_short:].mean()
        ma_long = self.equity_curve.iloc[-self.equity_ma_long:].mean()
        current = self.equity_curve.iloc[-1]
        
        # Check conditions
        if current < ma_short * 0.98:  # 2% below short MA
            self.position_multiplier *= 0.5
            return "Equity curve filter: Below short MA"
            
        if ma_short < ma_long * 0.99:  # Short MA below long MA
            self.position_multiplier *= 0.75
            return "Equity curve filter: Bearish MA cross"
            
        return None
        
    def _check_daily_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        today = datetime.now().date()
        if today in self.daily_pnl:
            daily_loss = -self.daily_pnl[today]['pnl_percent']
            return daily_loss > self.time_limits['daily']
        return False
        
    def _check_volatility_spike(self) -> bool:
        """Check for abnormal volatility"""
        if len(self.balance_history) < 20:
            return False
            
        recent_balances = [b for _, b in list(self.balance_history)[-20:]]
        returns = pd.Series(recent_balances).pct_change().dropna()
        
        if len(returns) > 1:
            current_vol = returns.std()
            avg_vol = returns.rolling(10).std().mean()
            
            return current_vol > avg_vol * 2  # 2x normal volatility
            
        return False
        
    def _check_correlation_breakdown(self) -> bool:
        """Check if normal correlations have broken down"""
        # This would integrate with actual correlation data
        # Placeholder for now
        return False
        
    def _detect_martingale(self) -> bool:
        """Detect martingale-like position sizing"""
        if len(self.trade_history) < 5:
            return False
            
        recent_trades = list(self.trade_history)[-5:]
        sizes = [t.get('size', 0) for t in recent_trades]
        results = [t.get('pnl', 0) for t in recent_trades]
        
        # Check if position sizes increase after losses
        martingale_detected = False
        for i in range(1, len(recent_trades)):
            if results[i-1] < 0 and sizes[i] > sizes[i-1] * 1.5:
                martingale_detected = True
                break
                
        return martingale_detected
        
    def _pause_trading(self):
        """Pause all trading"""
        self.is_paused = True
        logger.warning("⏸️ Trading paused by protection system")
        
    def _emergency_stop(self):
        """Emergency stop all positions"""
        self.is_paused = True
        self.position_multiplier = 0.0
        logger.critical("🛑 EMERGENCY STOP ACTIVATED")
        
    def _reduce_position_size(self):
        """Reduce position sizes"""
        self.position_multiplier *= 0.5
        logger.warning(f"📉 Position size reduced to {self.position_multiplier:.0%}")
        
    def _restrict_correlated_pairs(self):
        """Restrict trading to uncorrelated pairs"""
        # This would integrate with correlation data
        self.allowed_pairs = ['BTC/USDT', 'ETH/USDT']  # Example
        logger.warning("🔒 Trading restricted to uncorrelated pairs")
        
    def _reset_position_sizes(self):
        """Reset position sizes to prevent martingale"""
        self.position_multiplier = min(self.position_multiplier, 0.5)
        logger.warning("🔄 Position sizes reset to prevent martingale")
        
    def _activate_panic_mode(self):
        """Activate panic mode for extreme events"""
        self.is_paused = True
        self.position_multiplier = 0.0
        self.recovery_mode = RecoveryMode.TURTLE
        logger.critical("😱 PANIC MODE ACTIVATED - BLACK SWAN DETECTED")
        
    def _record_drawdown_end(self, timestamp: datetime):
        """Record the end of a drawdown period"""
        if self.metrics.drawdown_start:
            self.metrics.recovery_time = timestamp - self.metrics.drawdown_start
            self.metrics.recovery_count += 1
            
            # Calculate recovery rate
            if self.metrics.recovery_time.total_seconds() > 0:
                self.metrics.recovery_rate = (
                    self.metrics.current_drawdown / 
                    (self.metrics.recovery_time.total_seconds() / 3600)  # Per hour
                )
                
    def get_position_sizing_multiplier(self) -> float:
        """
        Get current position sizing multiplier
        
        Returns:
            Multiplier for position sizes (0.0 to 1.0)
        """
        if self.is_paused:
            return 0.0
            
        # Base multiplier from drawdown level
        base_multiplier = self.position_multiplier
        
        # Apply recovery mode adjustment
        recovery_multipliers = {
            RecoveryMode.AGGRESSIVE: 1.2,
            RecoveryMode.MODERATE: 1.0,
            RecoveryMode.CONSERVATIVE: 0.7,
            RecoveryMode.TURTLE: 0.3
        }
        
        recovery_mult = recovery_multipliers.get(self.recovery_mode, 1.0)
        
        # Apply volatility scaling
        if self.settings['volatility_scaling']:
            vol_mult = self._get_volatility_multiplier()
        else:
            vol_mult = 1.0
            
        # Final multiplier
        final_mult = base_multiplier * recovery_mult * vol_mult
        
        return max(0.0, min(1.0, final_mult))
        
    def _get_volatility_multiplier(self) -> float:
        """Calculate volatility-based position multiplier"""
        if len(self.balance_history) < 20:
            return 1.0
            
        recent_balances = [b for _, b in list(self.balance_history)[-20:]]
        returns = pd.Series(recent_balances).pct_change().dropna()
        
        if len(returns) > 1:
            current_vol = returns.std()
            # Inverse relationship - higher vol = smaller positions
            vol_mult = 0.02 / (current_vol + 0.01)  # Target 2% vol
            return max(0.5, min(1.5, vol_mult))
            
        return 1.0
        
    def check_trade_allowed(self, symbol: str, size: float) -> Tuple[bool, str]:
        """
        Check if a trade is allowed under current conditions
        
        Args:
            symbol: Trading pair
            size: Proposed position size
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if paused
        if self.is_paused:
            return False, "Trading paused due to drawdown protection"
            
        # Check allowed pairs restriction
        if self.allowed_pairs and symbol not in self.allowed_pairs:
            return False, f"Symbol {symbol} not in allowed pairs list"
            
        # Check time-based limits
        for period, limit in self.time_limits.items():
            if self._check_time_limit(period, limit):
                return False, f"{period.capitalize()} loss limit exceeded"
                
        # Check position sizing
        adjusted_size = size * self.get_position_sizing_multiplier()
        if adjusted_size < size * 0.1:  # Less than 10% of requested
            return False, "Position size too restricted by drawdown protection"
            
        return True, "Trade allowed"
        
    def _check_time_limit(self, period: str, limit: float) -> bool:
        """Check if a time-based limit is exceeded"""
        now = datetime.now()
        
        if period == 'hourly':
            hour_key = now.replace(minute=0, second=0, microsecond=0)
            if hour_key in self.hourly_pnl:
                return -self.hourly_pnl[hour_key]['pnl_percent'] > limit
                
        elif period == 'daily':
            return self._check_daily_limit()
            
        elif period == 'weekly':
            # Calculate weekly P&L
            week_start = now - timedelta(days=now.weekday())
            weekly_pnl = sum(
                data['pnl'] for date, data in self.daily_pnl.items()
                if date >= week_start.date()
            )
            if self.metrics.peak_balance > 0:
                weekly_pnl_pct = weekly_pnl / self.metrics.peak_balance
                return -weekly_pnl_pct > limit
                
        elif period == 'monthly':
            # Calculate monthly P&L
            month_start = now.replace(day=1)
            monthly_pnl = sum(
                data['pnl'] for date, data in self.daily_pnl.items()
                if date >= month_start.date()
            )
            if self.metrics.peak_balance > 0:
                monthly_pnl_pct = monthly_pnl / self.metrics.peak_balance
                return -monthly_pnl_pct > limit
                
        return False
        
    def add_trade_result(self, trade: Dict):
        """Add a trade result for analysis"""
        self.trade_history.append(trade)
        
        # Update consecutive wins/losses
        if trade.get('pnl', 0) > 0:
            self.metrics.consecutive_wins += 1
            self.metrics.consecutive_losses = 0
        else:
            self.metrics.consecutive_losses += 1
            self.metrics.consecutive_wins = 0
            
    def calculate_recovery_metrics(self) -> Dict:
        """Calculate comprehensive recovery statistics"""
        gain_to_recover = 0.0
        if self.metrics.current_drawdown > 0:
            gain_to_recover = self.metrics.current_drawdown / (1 - self.metrics.current_drawdown)
            
        # Calculate average drawdown
        if self.drawdown_history:
            dd_values = [d['drawdown'] for d in list(self.drawdown_history)]
            self.metrics.average_drawdown = np.mean([d for d in dd_values if d > 0])
            self.metrics.drawdown_volatility = np.std(dd_values)
            
        return {
            'current_drawdown': f"{self.metrics.current_drawdown:.2%}",
            'max_drawdown': f"{self.metrics.max_drawdown:.2%}",
            'average_drawdown': f"{self.metrics.average_drawdown:.2%}",
            'drawdown_volatility': f"{self.metrics.drawdown_volatility:.2%}",
            'drawdown_duration': str(self.metrics.drawdown_duration),
            'time_underwater': str(self.metrics.time_underwater),
            'recovery_time': str(self.metrics.recovery_time) if self.metrics.recovery_time else "N/A",
            'gain_to_recover': f"{gain_to_recover:.2%}",
            'consecutive_losses': self.metrics.consecutive_losses,
            'drawdown_count': self.metrics.drawdown_count,
            'recovery_count': self.metrics.recovery_count
        }
        
    def get_risk_adjusted_parameters(self) -> Dict:
        """Get current risk-adjusted trading parameters"""
        return {
            'max_positions': self._adjust_max_positions(),
            'position_size_multiplier': self.get_position_sizing_multiplier(),
            'stop_loss_multiplier': self._adjust_stop_loss(),
            'take_profit_multiplier': self._adjust_take_profit(),
            'trade_frequency_multiplier': self._adjust_trade_frequency(),
            'allowed_strategies': self._filter_strategies(),
            'allowed_pairs': self.allowed_pairs if self.allowed_pairs else "All",
            'recovery_mode': self.recovery_mode.value
        }
        
    def _adjust_max_positions(self) -> int:
        """Adjust maximum concurrent positions based on drawdown"""
        base_positions = 10
        
        if self.current_level == DrawdownLevel.BLACK:
            return 0
        elif self.current_level == DrawdownLevel.RED:
            return 1
        elif self.current_level == DrawdownLevel.ORANGE:
            return 3
        elif self.current_level == DrawdownLevel.YELLOW:
            return 5
            
        return base_positions
        
    def _adjust_stop_loss(self) -> float:
        """Tighten stop loss during drawdown"""
        if self.current_level == DrawdownLevel.RED:
            return 0.5  # 50% tighter
        elif self.current_level == DrawdownLevel.ORANGE:
            return 0.7  # 30% tighter
        elif self.current_level == DrawdownLevel.YELLOW:
            return 0.85  # 15% tighter
            
        return 1.0
        
    def _adjust_take_profit(self) -> float:
        """Adjust take profit targets during drawdown"""
        if self.current_level == DrawdownLevel.RED:
            return 0.5  # Take profits 50% earlier
        elif self.current_level == DrawdownLevel.ORANGE:
            return 0.7
        elif self.current_level == DrawdownLevel.YELLOW:
            return 0.85
            
        return 1.0
        
    def _adjust_trade_frequency(self) -> float:
        """Reduce trade frequency during drawdown"""
        if self.current_level == DrawdownLevel.RED:
            return 0.1  # 90% reduction
        elif self.current_level == DrawdownLevel.ORANGE:
            return 0.3  # 70% reduction
        elif self.current_level == DrawdownLevel.YELLOW:
            return 0.6  # 40% reduction
            
        return 1.0
        
    def _filter_strategies(self) -> List[str]:
        """Filter allowed strategies based on risk level"""
        all_strategies = ['arbitrage', 'market_making', 'momentum', 'mean_reversion', 'grid']
        
        if self.current_level == DrawdownLevel.RED:
            return ['arbitrage']  # Only lowest risk
        elif self.current_level == DrawdownLevel.ORANGE:
            return ['arbitrage', 'market_making']
        elif self.current_level == DrawdownLevel.YELLOW:
            return ['arbitrage', 'market_making', 'grid']
            
        return all_strategies
        
    def _get_summary_metrics(self) -> Dict:
        """Get summary metrics for display"""
        return {
            'peak_balance': self.metrics.peak_balance,
            'current_drawdown_pct': self.metrics.current_drawdown,
            'current_drawdown_amt': self.metrics.current_drawdown_amount,
            'max_drawdown_pct': self.metrics.max_drawdown,
            'consecutive_losses': self.metrics.consecutive_losses,
            'daily_pnl': self._get_today_pnl(),
            'level': self.current_level.value
        }
        
    def _get_today_pnl(self) -> Dict:
        """Get today's P&L"""
        today = datetime.now().date()
        if today in self.daily_pnl:
            return {
                'amount': self.daily_pnl[today]['pnl'],
                'percent': self.daily_pnl[today]['pnl_percent']
            }
        return {'amount': 0.0, 'percent': 0.0}
        
    def _format_thresholds(self) -> str:
        """Format thresholds for logging"""
        return " | ".join([
            f"{level.value}: {threshold:.1%}"
            for level, threshold in self.thresholds.items()
        ])
        
    def export_report(self) -> str:
        """Export comprehensive drawdown protection report"""
        report = f"""
# 🛡️ NEXLIFY DRAWDOWN PROTECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Current Status
- **Protection Level**: {self.current_level.value.upper()} 
- **Trading Status**: {'⏸️ PAUSED' if self.is_paused else '▶️ ACTIVE'}
- **Position Multiplier**: {self.position_multiplier:.0%}
- **Recovery Mode**: {self.recovery_mode.value}

## 📈 Drawdown Metrics
{self._format_metrics_table()}

## ⚡ Risk Parameters
{self._format_risk_params_table()}

## 📉 Recent Performance
{self._format_recent_performance()}

## 🎯 Protection Rules Status
{self._format_protection_rules()}

## 💡 Recommendations
{self._generate_recommendations()}
"""
        return report
        
    def _format_metrics_table(self) -> str:
        """Format metrics as a table"""
        metrics = self.calculate_recovery_metrics()
        lines = []
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"- **{formatted_key}**: {value}")
        return '\n'.join(lines)
        
    def _format_risk_params_table(self) -> str:
        """Format risk parameters as a table"""
        params = self.get_risk_adjusted_parameters()
        lines = []
        for key, value in params.items():
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"- **{formatted_key}**: {value}")
        return '\n'.join(lines)
        
    def _format_recent_performance(self) -> str:
        """Format recent performance data"""
        recent_days = list(self.daily_pnl.items())[-7:]
        if not recent_days:
            return "No recent data available"
            
        lines = ["| Date | P&L | Percent | High | Low |", "|------|-----|---------|------|-----|"]
        
        for date, data in recent_days:
            lines.append(
                f"| {date} | ${data['pnl']:.2f} | {data['pnl_percent']:.2%} | "
                f"${data['high']:.2f} | ${data['low']:.2f} |"
            )
            
        return '\n'.join(lines)
        
    def _format_protection_rules(self) -> str:
        """Format protection rules status"""
        lines = []
        for rule in self.protection_rules:
            status = "✅ Active" if rule.enabled else "❌ Disabled"
            triggered = "🔴 TRIGGERED" if rule.condition() else "🟢 OK"
            lines.append(f"- **{rule.name}**: {status} | {triggered}")
            lines.append(f"  - {rule.description}")
        return '\n'.join(lines)
        
    def _generate_recommendations(self) -> str:
        """Generate AI-powered recommendations"""
        recs = []
        
        if self.current_level == DrawdownLevel.RED:
            recs.append("🚨 **CRITICAL**: Consider closing all positions and reassessing strategy")
            recs.append("📊 Reduce position sizes to minimum until recovery")
            recs.append("🎯 Focus only on highest probability setups")
            
        elif self.current_level == DrawdownLevel.ORANGE:
            recs.append("⚠️ **WARNING**: Implement strict risk controls")
            recs.append("📉 Reduce leverage and position sizes")
            recs.append("🔍 Review and optimize current strategies")
            
        elif self.current_level == DrawdownLevel.YELLOW:
            recs.append("💡 **CAUTION**: Monitor positions closely")
            recs.append("🛡️ Tighten stop losses on all positions")
            recs.append("📈 Consider taking partial profits")
            
        else:
            recs.append("✅ **HEALTHY**: Continue normal operations")
            recs.append("📊 Consider gradually increasing position sizes")
            recs.append("🚀 Look for new opportunities")
            
        return '\n'.join(recs)
        
    def _save_state(self):
        """Persist current state to disk"""
        try:
            state = {
                'metrics': asdict(self.metrics),
                'current_level': self.current_level.value,
                'recovery_mode': self.recovery_mode.value,
                'is_paused': self.is_paused,
                'position_multiplier': self.position_multiplier,
                'timestamp': datetime.now().isoformat()
            }
            
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save drawdown state: {e}")
            
    def _load_state(self):
        """Load persisted state from disk"""
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, 'r') as f:
                    state = json.load(f)
                    
                # Restore metrics
                for key, value in state.get('metrics', {}).items():
                    if hasattr(self.metrics, key):
                        if 'datetime' in str(type(getattr(self.metrics, key))):
                            setattr(self.metrics, key, 
                                   datetime.fromisoformat(value) if value else None)
                        elif 'timedelta' in str(type(getattr(self.metrics, key))):
                            # Parse timedelta from string representation
                            setattr(self.metrics, key, timedelta(seconds=0))  # Simplified
                        else:
                            setattr(self.metrics, key, value)
                            
                # Restore state
                self.current_level = DrawdownLevel(state.get('current_level', 'green_zone'))
                self.recovery_mode = RecoveryMode(state.get('recovery_mode', 'moderate'))
                self.is_paused = state.get('is_paused', False)
                self.position_multiplier = state.get('position_multiplier', 1.0)
                
                logger.info("📂 Loaded drawdown state from disk")
                
        except Exception as e:
            logger.error(f"Failed to load drawdown state: {e}")


class DrawdownNeuralMonitor:
    """Neural network monitoring for drawdown patterns"""
    
    def __init__(self, protection: NexlifyDrawdownProtection):
        self.protection = protection
        self.pattern_memory = deque(maxlen=100)
        
    def analyze(self) -> Dict:
        """Analyze patterns and predict danger"""
        # Simplified pattern recognition
        danger_score = 0.0
        
        # Check drawdown acceleration
        if len(self.protection.drawdown_history) > 10:
            recent = list(self.protection.drawdown_history)[-10:]
            dd_values = [d['drawdown'] for d in recent]
            
            # Check if drawdown is accelerating
            if len(dd_values) > 2:
                acceleration = dd_values[-1] - dd_values[-2]
                if acceleration > 0.01:  # 1% acceleration
                    danger_score += 0.3
                    
        # Check correlation breakdown patterns
        if self.protection.metrics.consecutive_losses > 3:
            danger_score += 0.2
            
        # Check time-based patterns
        if self.protection.metrics.drawdown_duration > timedelta(days=7):
            danger_score += 0.2
            
        # Neural signal
        return {
            'danger_detected': danger_score > 0.5,
            'danger_score': danger_score,
            'pattern': self._identify_pattern(),
            'prediction': self._predict_recovery()
        }
        
    def _identify_pattern(self) -> str:
        """Identify drawdown pattern type"""
        if self.protection.metrics.consecutive_losses > 5:
            return "death_spiral"
        elif self.protection.metrics.drawdown_volatility > 0.05:
            return "volatile_chop"
        elif self.protection.metrics.drawdown_duration > timedelta(days=14):
            return "slow_bleed"
        else:
            return "normal_correction"
            
    def _predict_recovery(self) -> Dict:
        """Predict recovery probability and time"""
        # Simplified prediction logic
        base_recovery_prob = 0.7
        
        # Adjust based on current level
        level_adjustments = {
            DrawdownLevel.GREEN: 0.0,
            DrawdownLevel.YELLOW: -0.1,
            DrawdownLevel.ORANGE: -0.2,
            DrawdownLevel.RED: -0.3,
            DrawdownLevel.BLACK: -0.5
        }
        
        recovery_prob = base_recovery_prob + level_adjustments.get(
            self.protection.current_level, 0
        )
        
        # Estimate recovery time
        if self.protection.metrics.average_drawdown > 0:
            est_recovery_days = (
                self.protection.metrics.current_drawdown / 
                self.protection.metrics.average_drawdown * 7  # Average 7 days
            )
        else:
            est_recovery_days = 7
            
        return {
            'probability': max(0.1, min(0.9, recovery_prob)),
            'estimated_days': int(est_recovery_days),
            'confidence': 0.6  # Medium confidence
        }


def create_drawdown_protection(config: Dict) -> NexlifyDrawdownProtection:
    """Factory function to create drawdown protection instance"""
    logger.info("🏗️ Initializing Nexlify Drawdown Protection System")
    return NexlifyDrawdownProtection(config)
