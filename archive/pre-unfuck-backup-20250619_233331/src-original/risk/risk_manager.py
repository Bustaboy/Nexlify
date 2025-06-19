# src/risk/risk_manager.py
"""
Nexlify Integrated Risk Management System
Combines all risk components including drawdown protection
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from collections import defaultdict

# Import risk components
from .nexlify_drawdown_protection import NexlifyDrawdownProtection, DrawdownLevel

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Comprehensive risk management parameters"""
    # Position sizing
    max_position_size: float = 0.1          # 10% of portfolio
    min_position_size: float = 0.001        # 0.1% minimum
    max_total_exposure: float = 0.5         # 50% max exposure
    max_single_pair_exposure: float = 0.2   # 20% per pair
    
    # Loss limits
    max_daily_loss: float = 0.05            # 5% daily loss limit
    max_weekly_loss: float = 0.10           # 10% weekly loss limit
    max_monthly_loss: float = 0.15          # 15% monthly loss limit
    
    # Risk per trade
    risk_per_trade: float = 0.01            # 1% risk per trade
    risk_reward_ratio: float = 2.0          # Minimum 2:1 RR ratio
    
    # Stop loss and take profit
    default_stop_loss: float = 0.02         # 2% default SL
    default_take_profit: float = 0.05       # 5% default TP
    use_trailing_stop: bool = True
    trailing_stop_distance: float = 0.01    # 1% trailing
    
    # Correlation limits
    max_correlated_positions: int = 3       # Max correlated positions
    correlation_threshold: float = 0.7      # Correlation threshold
    
    # Leverage
    max_leverage: float = 3.0               # 3x max leverage
    
    # Advanced features
    use_drawdown_protection: bool = True
    use_volatility_scaling: bool = True
    use_correlation_filter: bool = True
    use_ml_risk_prediction: bool = False

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float
    risk_percent: float
    time_held: timedelta
    correlation_group: Optional[str] = None

class NexlifyRiskManager:
    """
    🛡️ NEXLIFY RISK MANAGEMENT MATRIX
    Comprehensive risk control system
    """
    
    def __init__(self, config: Dict):
        """Initialize risk management system"""
        self.config = config
        self.params = self._load_risk_parameters(config)
        
        # Initialize components
        self.drawdown_protection = None
        if self.params.use_drawdown_protection:
            self.drawdown_protection = NexlifyDrawdownProtection(
                config.get('drawdown_protection', {}),
                risk_manager=self
            )
            
        # State tracking
        self.active_positions: Dict[str, PositionRisk] = {}
        self.daily_trades = defaultdict(list)
        self.performance_history = []
        
        # Correlation tracking
        self.correlation_matrix = {}
        self.correlation_groups = defaultdict(list)
        
        # Risk metrics
        self.current_exposure = 0.0
        self.open_risk = 0.0
        self.realized_pnl_today = 0.0
        
        # Persistence
        self.state_file = Path("data/risk_manager_state.json")
        self._load_state()
        
        logger.info("🛡️ RISK MANAGEMENT MATRIX INITIALIZED")
        
    def _load_risk_parameters(self, config: Dict) -> RiskParameters:
        """Load risk parameters from config"""
        risk_config = config.get('risk_management', {})
        params = RiskParameters()
        
        # Update from config
        for key, value in risk_config.items():
            if hasattr(params, key):
                setattr(params, key, value)
                
        return params
        
    async def check_trade_allowed(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Check if a trade is allowed under current risk parameters
        
        Returns:
            Tuple of (allowed, reason, adjusted_params)
        """
        # Initial parameters
        adjusted_params = {
            'size': size,
            'stop_loss': stop_loss or self._calculate_stop_loss(price, side),
            'take_profit': take_profit or self._calculate_take_profit(price, side),
            'leverage': 1.0
        }
        
        # Check drawdown protection first
        if self.drawdown_protection:
            dd_allowed, dd_reason = self.drawdown_protection.check_trade_allowed(symbol, size)
            if not dd_allowed:
                return False, f"Drawdown Protection: {dd_reason}", adjusted_params
                
            # Adjust size based on drawdown
            size_multiplier = self.drawdown_protection.get_position_sizing_multiplier()
            adjusted_params['size'] = size * size_multiplier
            
        # Check daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit exceeded", adjusted_params
            
        # Check position size limits
        position_check = self._check_position_size(symbol, adjusted_params['size'], price)
        if not position_check[0]:
            return False, position_check[1], adjusted_params
            
        # Check exposure limits
        exposure_check = self._check_exposure_limits(symbol, adjusted_params['size'], price)
        if not exposure_check[0]:
            return False, exposure_check[1], adjusted_params
            
        # Check correlation limits
        if self.params.use_correlation_filter:
            corr_check = self._check_correlation_limits(symbol)
            if not corr_check[0]:
                return False, corr_check[1], adjusted_params
                
        # Apply volatility scaling
        if self.params.use_volatility_scaling:
            vol_multiplier = await self._get_volatility_multiplier(symbol)
            adjusted_params['size'] *= vol_multiplier
            
        # Check risk/reward ratio
        rr_check = self._check_risk_reward(
            price, 
            adjusted_params['stop_loss'],
            adjusted_params['take_profit'],
            side
        )
        if not rr_check[0]:
            return False, rr_check[1], adjusted_params
            
        # Final size validation
        final_size = adjusted_params['size']
        if final_size < self.params.min_position_size:
            return False, "Position size too small after adjustments", adjusted_params
            
        # ML risk prediction (if enabled)
        if self.params.use_ml_risk_prediction:
            ml_check = await self._check_ml_risk_prediction(symbol, side, final_size, price)
            if not ml_check[0]:
                return False, ml_check[1], adjusted_params
                
        return True, "Trade approved", adjusted_params
        
    def add_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: str = 'long'
    ) -> PositionRisk:
        """Add a new position to track"""
        # Calculate risk
        if side == 'long':
            risk_amount = size * (entry_price - stop_loss)
        else:
            risk_amount = size * (stop_loss - entry_price)
            
        risk_percent = risk_amount / self._get_account_balance()
        
        position = PositionRisk(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            unrealized_pnl=0.0,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            time_held=timedelta()
        )
        
        self.active_positions[symbol] = position
        self._update_exposure()
        
        # Log trade
        self.daily_trades[datetime.now().date()].append({
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': entry_price,
            'timestamp': datetime.now()
        })
        
        logger.info(f"📊 Position added: {symbol} {side} {size} @ {entry_price}")
        return position
        
    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Update position with current price and check for actions
        
        Returns:
            Action to take (close_sl, close_tp, adjust_sl, None)
        """
        if symbol not in self.active_positions:
            return None
            
        position = self.active_positions[symbol]
        position.current_price = current_price
        
        # Update unrealized P&L
        if position.size > 0:  # Long
            position.unrealized_pnl = position.size * (current_price - position.entry_price)
        else:  # Short
            position.unrealized_pnl = -position.size * (current_price - position.entry_price)
            
        # Check stop loss
        if position.size > 0 and current_price <= position.stop_loss:
            return "close_sl"
        elif position.size < 0 and current_price >= position.stop_loss:
            return "close_sl"
            
        # Check take profit
        if position.size > 0 and current_price >= position.take_profit:
            return "close_tp"
        elif position.size < 0 and current_price <= position.take_profit:
            return "close_tp"
            
        # Update trailing stop
        if self.params.use_trailing_stop:
            new_sl = self._calculate_trailing_stop(position, current_price)
            if new_sl != position.stop_loss:
                position.stop_loss = new_sl
                return "adjust_sl"
                
        return None
        
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """Close a position and update metrics"""
        if symbol not in self.active_positions:
            return
            
        position = self.active_positions[symbol]
        
        # Calculate realized P&L
        if position.size > 0:  # Long
            realized_pnl = position.size * (close_price - position.entry_price)
        else:  # Short
            realized_pnl = -position.size * (close_price - position.entry_price)
            
        # Update daily P&L
        self.realized_pnl_today += realized_pnl
        
        # Update drawdown protection
        if self.drawdown_protection:
            self.drawdown_protection.add_trade_result({
                'symbol': symbol,
                'pnl': realized_pnl,
                'size': abs(position.size),
                'duration': position.time_held,
                'reason': reason
            })
            
        # Log performance
        self.performance_history.append({
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': close_price,
            'pnl': realized_pnl,
            'pnl_percent': realized_pnl / (abs(position.size) * position.entry_price),
            'duration': position.time_held,
            'reason': reason,
            'timestamp': datetime.now()
        })
        
        # Remove position
        del self.active_positions[symbol]
        self._update_exposure()
        
        logger.info(
            f"📊 Position closed: {symbol} @ {close_price} "
            f"P&L: ${realized_pnl:.2f} ({reason})"
        )
        
    def update_account_balance(self, balance: float):
        """Update account balance and trigger drawdown checks"""
        if self.drawdown_protection:
            update_result = self.drawdown_protection.update(balance)
            
            # Log significant changes
            if update_result['level'] != DrawdownLevel.GREEN:
                logger.warning(
                    f"⚠️ Drawdown Alert: {update_result['level'].value} - "
                    f"{update_result['drawdown']:.2%}"
                )
                
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk metrics dashboard"""
        dashboard = {
            'account': {
                'balance': self._get_account_balance(),
                'exposure': self.current_exposure,
                'open_risk': self.open_risk,
                'realized_pnl_today': self.realized_pnl_today
            },
            'positions': {
                'count': len(self.active_positions),
                'total_size': sum(abs(p.size) for p in self.active_positions.values()),
                'unrealized_pnl': sum(p.unrealized_pnl for p in self.active_positions.values()),
                'at_risk': sum(p.risk_amount for p in self.active_positions.values())
            },
            'limits': {
                'daily_loss_remaining': self._get_daily_loss_remaining(),
                'exposure_remaining': self.params.max_total_exposure - self.current_exposure,
                'positions_available': self._get_available_position_slots()
            }
        }
        
        # Add drawdown metrics if available
        if self.drawdown_protection:
            dashboard['drawdown'] = self.drawdown_protection._get_summary_metrics()
            dashboard['protection_level'] = self.drawdown_protection.current_level.value
            
        return dashboard
        
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Emergency stop all trading"""
        logger.critical(f"🛑 EMERGENCY STOP TRIGGERED: {reason}")
        
        # Close all positions
        positions_to_close = list(self.active_positions.keys())
        for symbol in positions_to_close:
            position = self.active_positions[symbol]
            self.close_position(symbol, position.current_price, f"emergency_stop: {reason}")
            
        # Activate maximum protection
        if self.drawdown_protection:
            self.drawdown_protection._activate_panic_mode()
            
        # Save state
        self._save_state()
        
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        daily_loss = -self.realized_pnl_today
        unrealized_loss = sum(
            min(0, p.unrealized_pnl) for p in self.active_positions.values()
        )
        total_loss = daily_loss + abs(unrealized_loss)
        
        balance = self._get_account_balance()
        loss_percent = total_loss / balance if balance > 0 else 0
        
        return loss_percent > self.params.max_daily_loss
        
    def _check_position_size(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """Check if position size is within limits"""
        position_value = size * price
        balance = self._get_account_balance()
        
        if balance <= 0:
            return False, "Invalid account balance"
            
        position_percent = position_value / balance
        
        if position_percent > self.params.max_position_size:
            return False, f"Position size {position_percent:.1%} exceeds limit {self.params.max_position_size:.1%}"
            
        if position_percent < self.params.min_position_size:
            return False, f"Position size {position_percent:.1%} below minimum {self.params.min_position_size:.1%}"
            
        return True, "OK"
        
    def _check_exposure_limits(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """Check total exposure limits"""
        new_exposure = self.current_exposure + (size * price / self._get_account_balance())
        
        if new_exposure > self.params.max_total_exposure:
            return False, f"Total exposure {new_exposure:.1%} would exceed limit {self.params.max_total_exposure:.1%}"
            
        # Check single pair exposure
        current_pair_exposure = 0
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            current_pair_exposure = abs(position.size) * position.current_price / self._get_account_balance()
            
        new_pair_exposure = current_pair_exposure + (size * price / self._get_account_balance())
        
        if new_pair_exposure > self.params.max_single_pair_exposure:
            return False, f"Pair exposure {new_pair_exposure:.1%} would exceed limit {self.params.max_single_pair_exposure:.1%}"
            
        return True, "OK"
        
    def _check_correlation_limits(self, symbol: str) -> Tuple[bool, str]:
        """Check correlation-based position limits"""
        # Get correlation group for symbol
        correlation_group = self._get_correlation_group(symbol)
        
        if correlation_group:
            correlated_positions = [
                s for s, p in self.active_positions.items()
                if self._get_correlation_group(s) == correlation_group
            ]
            
            if len(correlated_positions) >= self.params.max_correlated_positions:
                return False, f"Maximum {self.params.max_correlated_positions} correlated positions reached"
                
        return True, "OK"
        
    def _check_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: str
    ) -> Tuple[bool, str]:
        """Check if risk/reward ratio meets minimum requirements"""
        if side == 'long':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        if risk <= 0 or reward <= 0:
            return False, "Invalid stop loss or take profit"
            
        rr_ratio = reward / risk
        
        if rr_ratio < self.params.risk_reward_ratio:
            return False, f"Risk/Reward ratio {rr_ratio:.1f} below minimum {self.params.risk_reward_ratio:.1f}"
            
        return True, "OK"
        
    async def _get_volatility_multiplier(self, symbol: str) -> float:
        """Calculate position size multiplier based on volatility"""
        # This would integrate with actual volatility data
        # For now, return a placeholder
        return 1.0
        
    async def _check_ml_risk_prediction(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float
    ) -> Tuple[bool, str]:
        """Check ML-based risk prediction"""
        # This would integrate with ML models
        # For now, always allow
        return True, "ML check passed"
        
    def _calculate_stop_loss(self, price: float, side: str) -> float:
        """Calculate default stop loss"""
        if side == 'long':
            return price * (1 - self.params.default_stop_loss)
        else:
            return price * (1 + self.params.default_stop_loss)
            
    def _calculate_take_profit(self, price: float, side: str) -> float:
        """Calculate default take profit"""
        if side == 'long':
            return price * (1 + self.params.default_take_profit)
        else:
            return price * (1 - self.params.default_take_profit)
            
    def _calculate_trailing_stop(self, position: PositionRisk, current_price: float) -> float:
        """Calculate trailing stop loss"""
        if position.size > 0:  # Long
            # Only trail up
            new_sl = current_price * (1 - self.params.trailing_stop_distance)
            return max(position.stop_loss, new_sl)
        else:  # Short
            # Only trail down
            new_sl = current_price * (1 + self.params.trailing_stop_distance)
            return min(position.stop_loss, new_sl)
            
    def _get_correlation_group(self, symbol: str) -> Optional[str]:
        """Get correlation group for a symbol"""
        # Define correlation groups (this would use actual correlation data)
        groups = {
            'crypto_majors': ['BTC', 'ETH', 'BNB'],
            'defi': ['UNI', 'AAVE', 'SUSHI', 'COMP'],
            'layer2': ['MATIC', 'ARB', 'OP'],
            'memes': ['DOGE', 'SHIB', 'PEPE']
        }
        
        base = symbol.split('/')[0]
        for group_name, symbols in groups.items():
            if base in symbols:
                return group_name
                
        return None
        
    def _update_exposure(self):
        """Update current exposure metrics"""
        balance = self._get_account_balance()
        if balance <= 0:
            return
            
        total_exposure = 0
        total_risk = 0
        
        for position in self.active_positions.values():
            position_value = abs(position.size) * position.current_price
            total_exposure += position_value
            total_risk += position.risk_amount
            
        self.current_exposure = total_exposure / balance
        self.open_risk = total_risk / balance
        
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        # This would connect to actual account data
        # For now, return from config
        return self.config.get('initial_balance', 10000)
        
    def _get_daily_loss_remaining(self) -> float:
        """Calculate remaining daily loss allowance"""
        balance = self._get_account_balance()
        max_loss = balance * self.params.max_daily_loss
        current_loss = -self.realized_pnl_today
        
        return max(0, max_loss - current_loss)
        
    def _get_available_position_slots(self) -> int:
        """Get number of available position slots"""
        # This could be more sophisticated based on correlation, etc.
        max_positions = 10  # Example limit
        return max(0, max_positions - len(self.active_positions))
        
    def _save_state(self):
        """Save current state to disk"""
        try:
            state = {
                'active_positions': {
                    symbol: {
                        'size': p.size,
                        'entry_price': p.entry_price,
                        'stop_loss': p.stop_loss,
                        'take_profit': p.take_profit,
                        'current_price': p.current_price
                    }
                    for symbol, p in self.active_positions.items()
                },
                'realized_pnl_today': self.realized_pnl_today,
                'timestamp': datetime.now().isoformat()
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save risk manager state: {e}")
            
    def _load_state(self):
        """Load state from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                # Check if state is from today
                state_date = datetime.fromisoformat(state['timestamp']).date()
                if state_date == datetime.now().date():
                    self.realized_pnl_today = state.get('realized_pnl_today', 0)
                    
                logger.info("📂 Loaded risk manager state")
                
        except Exception as e:
            logger.error(f"Failed to load risk manager state: {e}")
            
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report"""
        report = f"""
# 🛡️ NEXLIFY RISK MANAGEMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Account Overview
- **Balance**: ${self._get_account_balance():,.2f}
- **Total Exposure**: {self.current_exposure:.1%}
- **Open Risk**: {self.open_risk:.1%}
- **Today's P&L**: ${self.realized_pnl_today:,.2f}

## 📈 Active Positions ({len(self.active_positions)})
"""
        
        for symbol, position in self.active_positions.items():
            report += f"""
### {symbol}
- Size: {position.size}
- Entry: ${position.entry_price:.2f}
- Current: ${position.current_price:.2f}
- Stop Loss: ${position.stop_loss:.2f}
- Take Profit: ${position.take_profit:.2f}
- Unrealized P&L: ${position.unrealized_pnl:.2f}
- Risk: {position.risk_percent:.1%}
"""
        
        # Add drawdown report if available
        if self.drawdown_protection:
            report += "\n" + self.drawdown_protection.export_report()
            
        return report


def create_risk_manager(config: Dict) -> NexlifyRiskManager:
    """Factory function to create risk manager instance"""
    return NexlifyRiskManager(config)
