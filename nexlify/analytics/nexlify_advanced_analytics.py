#!/usr/bin/env python3
"""
Nexlify Advanced Analytics Suite
Comprehensive performance tracking and risk metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float = 0.0
    total_return_percent: float = 0.0
    annualized_return: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0

    # Trade metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Time-based
    total_days: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    positive_days: int = 0
    negative_days: int = 0

    # Advanced
    kelly_criterion: float = 0.0
    expectancy: float = 0.0
    ulcer_index: float = 0.0


class AdvancedAnalytics:
    """
    Advanced analytics engine for trading performance
    Calculates comprehensive metrics beyond basic win/loss
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual

        logger.info("📊 Advanced Analytics initialized")

    @handle_errors("Analytics Calculation", reraise=False)
    def calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[Dict],
        dates: List[datetime]
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_curve: List of portfolio values over time
            trades: List of trade dictionaries with pnl data
            dates: List of timestamps for equity curve

        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        if len(equity_curve) < 2:
            return PerformanceMetrics()

        metrics = PerformanceMetrics()

        # Convert to numpy arrays
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic returns
        metrics.total_return = equity[-1] - equity[0]
        metrics.total_return_percent = (metrics.total_return / equity[0]) * 100

        # Annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        metrics.total_days = days
        if years > 0:
            metrics.annualized_return = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100

        # Volatility
        metrics.volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized

        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        if np.std(returns) > 0:
            metrics.sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(252)

        # Sortino Ratio (uses only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            metrics.sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252)

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics.max_drawdown = abs(np.min(equity - peak))
        metrics.max_drawdown_percent = abs(np.min(drawdown)) * 100

        # Calmar Ratio
        if metrics.max_drawdown_percent > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_percent

        # Value at Risk (95% confidence)
        metrics.value_at_risk_95 = np.percentile(returns, 5) * equity[-1]

        # Conditional VaR (CVaR) - expected loss beyond VaR
        var_threshold = np.percentile(returns, 5)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) > 0:
            metrics.conditional_var_95 = np.mean(tail_losses) * equity[-1]

        # Omega Ratio (gains/losses above threshold)
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        if losses > 0:
            metrics.omega_ratio = gains / losses

        # Ulcer Index (measure of downside volatility)
        drawdown_squared = drawdown ** 2
        metrics.ulcer_index = np.sqrt(np.mean(drawdown_squared)) * 100

        # Best/Worst days
        if len(returns) > 0:
            metrics.best_day = np.max(returns) * 100
            metrics.worst_day = np.min(returns) * 100
            metrics.positive_days = np.sum(returns > 0)
            metrics.negative_days = np.sum(returns < 0)

        # Trade-specific metrics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            metrics.win_rate = (len(winning_trades) / len(trades)) * 100

            if winning_trades:
                wins = [t['pnl'] for t in winning_trades]
                metrics.avg_win = np.mean(wins)
                metrics.largest_win = np.max(wins)
                total_wins = sum(wins)
            else:
                total_wins = 0

            if losing_trades:
                losses = [abs(t['pnl']) for t in losing_trades]
                metrics.avg_loss = np.mean(losses)
                metrics.largest_loss = np.max(losses)
                total_losses = sum(losses)
            else:
                total_losses = 0

            # Profit Factor
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

            # Average Win/Loss Ratio
            if metrics.avg_loss > 0:
                metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss

            # Kelly Criterion
            if metrics.avg_loss > 0:
                win_prob = metrics.win_rate / 100
                loss_prob = 1 - win_prob
                win_loss_ratio = metrics.avg_win / metrics.avg_loss
                metrics.kelly_criterion = (win_prob / loss_prob - 1 / win_loss_ratio) * 100

            # Expectancy
            win_prob = metrics.win_rate / 100
            metrics.expectancy = (win_prob * metrics.avg_win) - ((1 - win_prob) * metrics.avg_loss)

        return metrics

    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between different assets

        Args:
            price_data: Dict mapping symbols to price lists

        Returns:
            Correlation matrix as DataFrame
        """
        df = pd.DataFrame(price_data)

        # Calculate returns
        returns = df.pct_change().dropna()

        # Calculate correlation
        correlation = returns.corr()

        return correlation

    def calculate_rolling_sharpe(
        self,
        equity_curve: List[float],
        window: int = 30
    ) -> List[float]:
        """Calculate rolling Sharpe ratio over time"""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        rolling_sharpe = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            excess_returns = window_returns - (self.risk_free_rate / 252)

            if np.std(window_returns) > 0:
                sharpe = (np.mean(excess_returns) / np.std(window_returns)) * np.sqrt(252)
                rolling_sharpe.append(sharpe)
            else:
                rolling_sharpe.append(0)

        return rolling_sharpe

    def calculate_monthly_returns(
        self,
        equity_curve: List[float],
        dates: List[datetime]
    ) -> pd.DataFrame:
        """Calculate monthly return breakdown"""
        df = pd.DataFrame({
            'equity': equity_curve,
            'date': dates
        })

        df.set_index('date', inplace=True)
        df['returns'] = df['equity'].pct_change()

        # Resample to monthly
        monthly = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table (year x month)
        monthly_df = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })

        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        pivot = pivot * 100  # Convert to percentage

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[int(m)-1] if m <= 12 else f'M{int(m)}'
                        for m in pivot.columns]

        return pivot

    def generate_analytics_report(
        self,
        metrics: PerformanceMetrics,
        equity_curve: List[float],
        dates: List[datetime],
        output_path: str = "analytics"
    ):
        """Generate comprehensive analytics report with visualizations"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create figure
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

            # 1. Equity Curve
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(dates, equity_curve, linewidth=2, color='#00ff9f', label='Portfolio Value')
            ax1.fill_between(dates, equity_curve, alpha=0.3, color='#00ff9f')
            ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. Drawdown Chart
            ax2 = fig.add_subplot(gs[1, :])
            peak = np.maximum.accumulate(equity_curve)
            drawdown = ((np.array(equity_curve) - peak) / peak) * 100
            ax2.fill_between(dates, drawdown, 0, color='#ff0055', alpha=0.3)
            ax2.plot(dates, drawdown, color='#ff0055', linewidth=2)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # 3. Returns Distribution
            ax3 = fig.add_subplot(gs[2, 0])
            returns = np.diff(equity_curve) / equity_curve[:-1]
            ax3.hist(returns * 100, bins=50, color='#7d5fff', alpha=0.7, edgecolor='white')
            ax3.axvline(x=0, color='#666', linestyle='--', linewidth=2)
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

            # 4. Rolling Sharpe
            ax4 = fig.add_subplot(gs[2, 1])
            rolling_sharpe = self.calculate_rolling_sharpe(equity_curve, window=30)
            if rolling_sharpe:
                ax4.plot(dates[30:len(rolling_sharpe)+30], rolling_sharpe,
                        linewidth=2, color='#00d4ff')
                ax4.axhline(y=0, color='#666', linestyle='--')
                ax4.set_title('Rolling Sharpe Ratio (30-day)')
                ax4.set_ylabel('Sharpe Ratio')
                ax4.grid(True, alpha=0.3)

            # 5. Performance Metrics Table
            ax5 = fig.add_subplot(gs[2, 2])
            ax5.axis('off')
            metrics_text = f"""
PERFORMANCE METRICS

Returns
• Total Return: {metrics.total_return_percent:.2f}%
• Annualized: {metrics.annualized_return:.2f}%
• Volatility: {metrics.volatility:.2f}%

Risk-Adjusted
• Sharpe Ratio: {metrics.sharpe_ratio:.2f}
• Sortino Ratio: {metrics.sortino_ratio:.2f}
• Calmar Ratio: {metrics.calmar_ratio:.2f}
• Omega Ratio: {metrics.omega_ratio:.2f}

Risk Metrics
• Max Drawdown: {metrics.max_drawdown_percent:.2f}%
• VaR (95%): ${metrics.value_at_risk_95:.2f}
• CVaR (95%): ${metrics.conditional_var_95:.2f}
• Ulcer Index: {metrics.ulcer_index:.2f}

Trade Metrics
• Win Rate: {metrics.win_rate:.2f}%
• Profit Factor: {metrics.profit_factor:.2f}
• Avg Win/Loss: {metrics.avg_win_loss_ratio:.2f}
• Expectancy: ${metrics.expectancy:.2f}
"""
            ax5.text(0.05, 0.95, metrics_text, fontsize=9, family='monospace',
                    verticalalignment='top')

            # 6. Risk Metrics Gauge
            ax6 = fig.add_subplot(gs[3, 0])
            categories = ['Sharpe', 'Sortino', 'Calmar', 'Omega']
            values = [
                min(metrics.sharpe_ratio, 3),
                min(metrics.sortino_ratio, 3),
                min(metrics.calmar_ratio, 3),
                min(metrics.omega_ratio, 3)
            ]
            colors = ['#00ff9f' if v > 1 else '#ff0055' for v in values]
            ax6.barh(categories, values, color=colors, alpha=0.7)
            ax6.axvline(x=1, color='#666', linestyle='--', label='Threshold')
            ax6.set_title('Risk-Adjusted Returns')
            ax6.set_xlabel('Ratio Value')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # 7. Win/Loss Analysis
            ax7 = fig.add_subplot(gs[3, 1])
            win_loss_data = [metrics.avg_win, -metrics.avg_loss]
            colors_wl = ['#00ff9f', '#ff0055']
            ax7.bar(['Avg Win', 'Avg Loss'], win_loss_data, color=colors_wl, alpha=0.7)
            ax7.set_title('Win/Loss Analysis')
            ax7.set_ylabel('Amount ($)')
            ax7.grid(True, alpha=0.3)

            # 8. Daily Performance
            ax8 = fig.add_subplot(gs[3, 2])
            day_data = [metrics.positive_days, metrics.negative_days]
            colors_days = ['#00ff9f', '#ff0055']
            ax8.pie(day_data, labels=['Positive', 'Negative'], colors=colors_days,
                   autopct='%1.1f%%', startangle=90)
            ax8.set_title('Daily Win Rate')

            plt.suptitle(f'Advanced Analytics Report - {timestamp}',
                        fontsize=16, fontweight='bold', y=0.995)

            # Save
            report_path = output_dir / f"analytics_report_{timestamp}.png"
            plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
            logger.info(f"📊 Analytics report saved: {report_path}")

            # Save metrics as JSON
            json_path = output_dir / f"metrics_{timestamp}.json"
            self._save_metrics_json(metrics, json_path)

            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return None

    def _save_metrics_json(self, metrics: PerformanceMetrics, filepath: Path):
        """Save metrics to JSON file"""
        data = {
            'returns': {
                'total_return': metrics.total_return,
                'total_return_percent': metrics.total_return_percent,
                'annualized_return': metrics.annualized_return
            },
            'risk_adjusted': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio
            },
            'risk_metrics': {
                'volatility': metrics.volatility,
                'max_drawdown': metrics.max_drawdown,
                'max_drawdown_percent': metrics.max_drawdown_percent,
                'value_at_risk_95': metrics.value_at_risk_95,
                'conditional_var_95': metrics.conditional_var_95,
                'ulcer_index': metrics.ulcer_index
            },
            'trade_metrics': {
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'expectancy': metrics.expectancy,
                'kelly_criterion': metrics.kelly_criterion
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Metrics saved: {filepath}")


if __name__ == "__main__":
    print("Nexlify Advanced Analytics Suite")
    print("Use via integration with trading engine")
