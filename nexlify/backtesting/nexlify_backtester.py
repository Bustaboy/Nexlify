#!/usr/bin/env python3
"""
Nexlify Backtesting Framework
Test trading strategies on historical data before risking real money
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class BacktestTrade:
    """Represents a single backtest trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    side: str = "buy"
    entry_price: float = 0.0
    exit_price: float = 0.0
    amount: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    calmar_ratio: float = 0.0

    # Advanced metrics
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: timedelta = timedelta(0)

    # Time series
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)

    # All trades
    trades: List[BacktestTrade] = field(default_factory=list)


class StrategyBacktester:
    """
    Comprehensive backtesting engine for trading strategies
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.fee_rate = self.config.get('fee_rate', 0.001)  # 0.1%
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05%

        logger.info("📊 Backtesting Framework initialized")

    @handle_errors("Backtesting", reraise=False)
    async def run_backtest(
        self,
        strategy_name: str,
        historical_data: pd.DataFrame,
        initial_capital: float = 10000,
        strategy_config: Dict = None,
        timeframe: str = '1h'
    ) -> BacktestResult:
        """
        Run a complete backtest

        Args:
            strategy_name: Name of strategy to test
            historical_data: DataFrame with OHLCV data
            initial_capital: Starting capital
            strategy_config: Strategy-specific configuration
            timeframe: Timeframe of data ('1m', '5m', '15m', '1h', '4h', '1d', etc.)

        Returns:
            BacktestResult with complete performance metrics
        """
        logger.info(f"🔬 Starting backtest: {strategy_name}")
        logger.info(f"   Period: {historical_data.index[0]} to {historical_data.index[-1]}")
        logger.info(f"   Initial Capital: ${initial_capital:,.2f}")

        # Initialize backtest state
        capital = initial_capital
        position = None
        trades: List[BacktestTrade] = []
        equity_curve = [initial_capital]
        dates = [historical_data.index[0]]

        # Load strategy
        strategy = self._load_strategy(strategy_name, strategy_config or {})

        # Simulate trading
        for i in range(1, len(historical_data)):
            current_bar = historical_data.iloc[i]
            current_date = historical_data.index[i]

            # Get market data window
            window = historical_data.iloc[max(0, i-100):i+1]

            # Generate signal
            signal = strategy.generate_signal(window, position)

            # Execute trades based on signal
            if signal == 'buy' and position is None:
                # Enter long position
                entry_price = current_bar['close'] * (1 + self.slippage)
                amount = (capital * 0.95) / entry_price  # Use 95% of capital
                fees = amount * entry_price * self.fee_rate

                position = BacktestTrade(
                    entry_time=current_date,
                    symbol=current_bar.get('symbol', 'BTC/USDT'),
                    side='buy',
                    entry_price=entry_price,
                    amount=amount,
                    fees=fees,
                    strategy=strategy_name
                )

                capital -= (amount * entry_price + fees)

            elif signal == 'sell' and position is not None:
                # Exit position
                exit_price = current_bar['close'] * (1 - self.slippage)
                exit_fees = position.amount * exit_price * self.fee_rate

                # Calculate PnL
                gross_pnl = (exit_price - position.entry_price) * position.amount
                net_pnl = gross_pnl - position.fees - exit_fees
                pnl_percent = (net_pnl / (position.entry_price * position.amount)) * 100

                # Update trade
                position.exit_time = current_date
                position.exit_price = exit_price
                position.pnl = net_pnl
                position.pnl_percent = pnl_percent
                position.fees += exit_fees

                # Add proceeds back to capital
                capital += (position.amount * exit_price - exit_fees)

                trades.append(position)
                position = None

            # Update equity curve
            current_equity = capital
            if position is not None:
                # Mark to market
                current_equity += position.amount * current_bar['close']

            equity_curve.append(current_equity)
            dates.append(current_date)

        # Close any open position at end
        if position is not None:
            final_bar = historical_data.iloc[-1]
            exit_price = final_bar['close']
            exit_fees = position.amount * exit_price * self.fee_rate

            gross_pnl = (exit_price - position.entry_price) * position.amount
            net_pnl = gross_pnl - position.fees - exit_fees

            position.exit_time = historical_data.index[-1]
            position.exit_price = exit_price
            position.pnl = net_pnl
            position.pnl_percent = (net_pnl / (position.entry_price * position.amount)) * 100
            position.fees += exit_fees

            capital += (position.amount * exit_price - exit_fees)
            trades.append(position)

        # Calculate performance metrics
        result = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            dates=dates,
            initial_capital=initial_capital,
            final_capital=capital,
            timeframe=timeframe
        )

        logger.info(f"✅ Backtest complete: {strategy_name}")
        logger.info(f"   Total Return: {result.total_return_percent:.2f}%")
        logger.info(f"   Win Rate: {result.win_rate:.2f}%")
        logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"   Max Drawdown: {result.max_drawdown_percent:.2f}%")

        return result

    def _load_strategy(self, strategy_name: str, config: Dict):
        """Load trading strategy by name"""
        strategies = {
            'momentum': MomentumStrategy(config),
            'mean_reversion': MeanReversionStrategy(config),
            'breakout': BreakoutStrategy(config),
            'rl_agent': RLStrategy(config)
        }

        return strategies.get(strategy_name, MomentumStrategy(config))

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        dates: List[datetime],
        initial_capital: float,
        final_capital: float,
        timeframe: str = '1h'
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""

        # Calculate periods per year for Sharpe ratio annualization
        timeframe_to_periods = {
            '1m': 525600,   # 365 * 24 * 60
            '5m': 105120,   # 365 * 24 * 12
            '15m': 35040,   # 365 * 24 * 4
            '1h': 8760,     # 365 * 24
            '4h': 2190,     # 365 * 6
            '1d': 365,      # 365
        }
        periods_per_year = timeframe_to_periods.get(timeframe, 8760)

        # Basic returns
        total_return = final_capital - initial_capital
        total_return_percent = (total_return / initial_capital) * 100

        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Win/Loss metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = max(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Average trade duration
        durations = [
            (t.exit_time - t.entry_time)
            for t in trades if t.exit_time is not None
        ]
        avg_trade_duration = np.mean([d.total_seconds() for d in durations]) if durations else 0
        avg_trade_duration = timedelta(seconds=avg_trade_duration)

        # Calculate returns series
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Sharpe Ratio (annualized based on timeframe)
        if len(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino Ratio (uses only downside deviation, annualized based on timeframe)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(periods_per_year)
        else:
            sortino_ratio = 0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown_percent = abs(np.min(drawdown)) * 100
        max_drawdown = abs(np.min(np.array(equity_curve) - peak))

        # Calmar Ratio (return / max drawdown)
        years = (dates[-1] - dates[0]).days / 365.25
        annual_return = (total_return_percent / years) if years > 0 else 0
        calmar_ratio = (annual_return / max_drawdown_percent) if max_drawdown_percent > 0 else 0

        return BacktestResult(
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percent=total_return_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            equity_curve=equity_curve,
            drawdown_curve=(drawdown * 100).tolist(),
            dates=dates,
            trades=trades
        )

    def generate_report(self, result: BacktestResult, output_path: str = "backtests"):
        """Generate comprehensive backtest report with charts"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle(f'Backtest Report - {timestamp}', fontsize=16, fontweight='bold')

            # 1. Equity Curve
            ax1 = axes[0, 0]
            ax1.plot(result.dates, result.equity_curve, linewidth=2, color='#00ff9f')
            ax1.axhline(y=result.initial_capital, color='#666', linestyle='--', label='Initial Capital')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Capital ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. Drawdown Curve
            ax2 = axes[0, 1]
            ax2.fill_between(result.dates, result.drawdown_curve, 0, color='#ff0055', alpha=0.3)
            ax2.plot(result.dates, result.drawdown_curve, color='#ff0055', linewidth=2)
            ax2.set_title('Drawdown Curve')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # 3. Trade Distribution
            ax3 = axes[1, 0]
            pnls = [t.pnl for t in result.trades]
            colors = ['#00ff9f' if pnl > 0 else '#ff0055' for pnl in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors)
            ax3.axhline(y=0, color='#666', linestyle='-', linewidth=1)
            ax3.set_title('Trade PnL Distribution')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Profit/Loss ($)')
            ax3.grid(True, alpha=0.3)

            # 4. Win Rate & Statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            stats_text = f"""
PERFORMANCE SUMMARY

Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)
Win Rate: {result.win_rate:.2f}%
Total Trades: {result.total_trades}
Winning: {result.winning_trades} | Losing: {result.losing_trades}

RISK METRICS
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Max Drawdown: {result.max_drawdown_percent:.2f}%
Calmar Ratio: {result.calmar_ratio:.2f}

TRADE METRICS
Profit Factor: {result.profit_factor:.2f}
Avg Win: ${result.avg_win:.2f}
Avg Loss: ${result.avg_loss:.2f}
Largest Win: ${result.largest_win:.2f}
Largest Loss: ${result.largest_loss:.2f}
"""
            ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')

            # 5. Monthly Returns Heatmap (simplified)
            ax5 = axes[2, 0]
            if len(result.trades) > 0:
                # Group trades by month
                monthly_returns = {}
                for trade in result.trades:
                    if trade.exit_time:
                        month_key = trade.exit_time.strftime('%Y-%m')
                        monthly_returns[month_key] = monthly_returns.get(month_key, 0) + trade.pnl

                months = list(monthly_returns.keys())
                returns = list(monthly_returns.values())
                colors_monthly = ['#00ff9f' if r > 0 else '#ff0055' for r in returns]

                ax5.bar(range(len(returns)), returns, color=colors_monthly)
                ax5.set_title('Monthly Returns')
                ax5.set_ylabel('Return ($)')
                ax5.set_xticks(range(len(months)))
                ax5.set_xticklabels(months, rotation=45, ha='right')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No trades executed', ha='center', va='center')
                ax5.set_title('Monthly Returns')

            # 6. Trade Duration Distribution
            ax6 = axes[2, 1]
            durations_hours = [
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in result.trades if t.exit_time is not None
            ]
            if durations_hours:
                ax6.hist(durations_hours, bins=20, color='#7d5fff', alpha=0.7, edgecolor='white')
                ax6.set_title('Trade Duration Distribution')
                ax6.set_xlabel('Duration (hours)')
                ax6.set_ylabel('Frequency')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No completed trades', ha='center', va='center')
                ax6.set_title('Trade Duration Distribution')

            plt.tight_layout()

            # Save figure
            report_path = output_dir / f"backtest_report_{timestamp}.png"
            plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
            logger.info(f"📊 Report saved: {report_path}")

            # Save JSON data
            json_path = output_dir / f"backtest_data_{timestamp}.json"
            self._save_json_report(result, json_path)

            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

    def _save_json_report(self, result: BacktestResult, filepath: Path):
        """Save backtest results as JSON"""
        data = {
            'summary': {
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'total_return_percent': result.total_return_percent,
                'win_rate': result.win_rate,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_percent': result.max_drawdown_percent
            },
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'symbol': t.symbol,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent
                }
                for t in result.trades
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Data saved: {filepath}")


# Strategy implementations

class BaseStrategy:
    """Base class for trading strategies"""
    def __init__(self, config: Dict):
        self.config = config

    def generate_signal(self, data: pd.DataFrame, position) -> str:
        """Generate buy/sell/hold signal"""
        raise NotImplementedError


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy"""
    def generate_signal(self, data: pd.DataFrame, position) -> str:
        if len(data) < 20:
            return 'hold'

        # Calculate 20-period SMA
        sma = data['close'].rolling(20).mean()
        current_price = data['close'].iloc[-1]

        if position is None:
            # Enter if price > SMA (uptrend)
            if current_price > sma.iloc[-1]:
                return 'buy'
        else:
            # Exit if price < SMA (downtrend)
            if current_price < sma.iloc[-1]:
                return 'sell'

        return 'hold'


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands"""
    def generate_signal(self, data: pd.DataFrame, position) -> str:
        if len(data) < 20:
            return 'hold'

        # Calculate Bollinger Bands
        sma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        current_price = data['close'].iloc[-1]

        if position is None:
            # Buy when price touches lower band
            if current_price <= lower_band.iloc[-1]:
                return 'buy'
        else:
            # Sell when price touches upper band
            if current_price >= upper_band.iloc[-1]:
                return 'sell'

        return 'hold'


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy using 52-week high/low"""
    def generate_signal(self, data: pd.DataFrame, position) -> str:
        if len(data) < 50:
            return 'hold'

        # Calculate 50-period high/low
        high_50 = data['high'].rolling(50).max()
        low_50 = data['low'].rolling(50).min()

        current_price = data['close'].iloc[-1]

        if position is None:
            # Buy on breakout above 50-period high
            if current_price >= high_50.iloc[-2]:  # Previous high
                return 'buy'
        else:
            # Sell on breakdown below 50-period low
            if current_price <= low_50.iloc[-2]:
                return 'sell'

        return 'hold'


class RLStrategy(BaseStrategy):
    """Strategy using RL agent for signals"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rl_agent = None
        self._load_rl_agent()

    def _load_rl_agent(self):
        """Load trained RL agent"""
        try:
            from nexlify.strategies.nexlify_rl_agent import DQNAgent
            model_path = Path("models/rl_agent_trained.pth")

            if model_path.exists():
                self.rl_agent = DQNAgent(state_size=8, action_size=3)
                self.rl_agent.load(str(model_path))
                logger.info("🧠 RL agent loaded for backtesting")
        except Exception as e:
            logger.warning(f"Could not load RL agent: {e}")

    def generate_signal(self, data: pd.DataFrame, position) -> str:
        if self.rl_agent is None or len(data) < 20:
            return 'hold'

        # Prepare state
        current_price = data['close'].iloc[-1]
        price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]

        # Calculate indicators
        rsi = self._calculate_rsi(data['close'])
        macd = self._calculate_macd(data['close'])

        state = np.array([
            0.5,  # Normalized balance (placeholder)
            1.0 if position else 0.0,
            position.entry_price / 10000 if position else 0,
            current_price / 10000,
            price_change,
            rsi,
            macd,
            0.5  # Volume ratio (placeholder)
        ], dtype=np.float32)

        # Get action from RL agent
        action = self.rl_agent.act(state, training=False)

        if action == 1 and position is None:
            return 'buy'
        elif action == 2 and position is not None:
            return 'sell'

        return 'hold'

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] / 100 if not np.isnan(rsi.iloc[-1]) else 0.5

    def _calculate_macd(self, prices):
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        return (macd.iloc[-1] / prices.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0


if __name__ == "__main__":
    # Example usage
    print("Nexlify Backtesting Framework")
    print("Use via: python test_strategy.py")
