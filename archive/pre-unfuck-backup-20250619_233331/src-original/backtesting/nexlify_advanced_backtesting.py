"""
Nexlify Enhanced - Advanced Backtesting Engine
Implements Feature 16: Walk-forward analysis, Monte Carlo simulations, and realistic modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    
    # Commission and slippage
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = 'percentage'  # 'percentage', 'fixed', 'market_impact'
    slippage_value: float = 0.0005  # 0.05%
    
    # Walk-forward settings
    walk_forward_enabled: bool = True
    in_sample_period: int = 252  # Trading days
    out_sample_period: int = 63  # Trading days
    optimization_metric: str = 'sharpe_ratio'
    
    # Monte Carlo settings
    monte_carlo_runs: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    
    # Risk settings
    max_drawdown_limit: float = 0.20
    position_sizing: str = 'fixed'  # 'fixed', 'kelly', 'risk_parity'
    max_positions: int = 10
    
    # Data settings
    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    universe: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT'])

@dataclass
class BacktestResults:
    """Complete backtesting results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    drawdown_series: pd.Series
    
    # Walk-forward results
    walk_forward_results: Optional[Dict] = None
    
    # Monte Carlo results
    monte_carlo_results: Optional[Dict] = None
    
    # Additional analytics
    monthly_returns: pd.Series = None
    trade_analysis: pd.DataFrame = None
    parameter_sensitivity: Dict = None

class AdvancedBacktestEngine:
    """
    Advanced backtesting engine with walk-forward analysis and Monte Carlo simulation
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data = {}
        self.results = None
        
    async def run_backtest(self, 
                          strategy: Any,
                          market_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        Run complete backtest with all advanced features
        
        Args:
            strategy: Trading strategy instance
            market_data: Dictionary of DataFrames with OHLCV data
            
        Returns:
            Comprehensive backtest results
        """
        self.market_data = market_data
        
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Run standard backtest
        basic_results = await self._run_basic_backtest(strategy)
        
        # Run walk-forward analysis
        if self.config.walk_forward_enabled:
            walk_forward_results = await self._run_walk_forward_analysis(strategy)
            basic_results.walk_forward_results = walk_forward_results
            
        # Run Monte Carlo simulation
        monte_carlo_results = await self._run_monte_carlo_simulation(basic_results)
        basic_results.monte_carlo_results = monte_carlo_results
        
        # Analyze parameter sensitivity
        sensitivity_results = await self._analyze_parameter_sensitivity(strategy)
        basic_results.parameter_sensitivity = sensitivity_results
        
        # Generate additional analytics
        basic_results.monthly_returns = self._calculate_monthly_returns(basic_results.returns)
        basic_results.trade_analysis = self._analyze_trades(basic_results)
        
        self.results = basic_results
        return basic_results
        
    async def _run_basic_backtest(self, strategy: Any) -> BacktestResults:
        """Run standard backtest"""
        # Initialize portfolio
        portfolio = Portfolio(self.config.initial_capital)
        trades = []
        
        # Get combined market data
        combined_data = self._prepare_market_data()
        
        # Main backtest loop
        for timestamp, market_snapshot in combined_data.iterrows():
            # Update portfolio with market prices
            portfolio.update_market_prices(market_snapshot)
            
            # Generate signals
            signals = await strategy.generate_signals(
                market_snapshot, 
                portfolio,
                timestamp
            )
            
            # Execute trades
            for signal in signals:
                trade = self._execute_trade(signal, market_snapshot, portfolio)
                if trade:
                    trades.append(trade)
                    
            # Record equity
            portfolio.record_equity(timestamp)
            
        # Calculate results
        return self._calculate_results(portfolio, trades)
        
    def _prepare_market_data(self) -> pd.DataFrame:
        """Prepare combined market data for backtesting"""
        # Align all timeframes and symbols
        combined = pd.DataFrame()
        
        for symbol, data in self.market_data.items():
            # Ensure data is within backtest period
            mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
            filtered_data = data[mask].copy()
            
            # Add symbol prefix to columns
            filtered_data.columns = [f"{symbol}_{col}" for col in filtered_data.columns]
            
            # Merge with combined data
            if combined.empty:
                combined = filtered_data
            else:
                combined = combined.join(filtered_data, how='outer')
                
        # Forward fill missing data
        combined.fillna(method='ffill', inplace=True)
        
        return combined
        
    def _execute_trade(self, signal: Dict, market_data: pd.Series, portfolio: 'Portfolio') -> Optional[Dict]:
        """Execute trade with realistic slippage and commission"""
        symbol = signal['symbol']
        side = signal['side']
        size = signal['size']
        
        # Get execution price with slippage
        base_price = market_data[f"{symbol}_close"]
        execution_price = self._apply_slippage(base_price, side, size)
        
        # Calculate commission
        commission = abs(size * execution_price * self.config.commission_rate)
        
        # Execute trade in portfolio
        if portfolio.can_execute_trade(symbol, side, size, execution_price):
            trade = {
                'timestamp': market_data.name,
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': execution_price,
                'commission': commission,
                'slippage': abs(execution_price - base_price)
            }
            
            portfolio.execute_trade(trade)
            return trade
            
        return None
        
    def _apply_slippage(self, price: float, side: str, size: float) -> float:
        """Apply realistic slippage model"""
        if self.config.slippage_model == 'percentage':
            slippage = price * self.config.slippage_value
            
        elif self.config.slippage_model == 'fixed':
            slippage = self.config.slippage_value
            
        elif self.config.slippage_model == 'market_impact':
            # Square root market impact model
            impact = self.config.slippage_value * np.sqrt(abs(size))
            slippage = price * impact
            
        else:
            slippage = 0
            
        # Apply slippage based on trade direction
        if side == 'buy':
            return price + slippage
        else:
            return price - slippage
            
    async def _run_walk_forward_analysis(self, strategy: Any) -> Dict:
        """
        Run walk-forward optimization and validation
        """
        logger.info("Starting walk-forward analysis...")
        
        results = {
            'periods': [],
            'in_sample_performance': [],
            'out_sample_performance': [],
            'parameter_stability': [],
            'optimization_history': []
        }
        
        # Calculate walk-forward periods
        periods = self._calculate_walk_forward_periods()
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Walk-forward period {i+1}/{len(periods)}")
            
            # Optimize on in-sample data
            optimal_params = await self._optimize_parameters(
                strategy, train_start, train_end
            )
            
            # Test on out-of-sample data
            in_sample_perf = await self._test_parameters(
                strategy, optimal_params, train_start, train_end
            )
            out_sample_perf = await self._test_parameters(
                strategy, optimal_params, test_start, test_end
            )
            
            # Record results
            results['periods'].append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })
            results['in_sample_performance'].append(in_sample_perf)
            results['out_sample_performance'].append(out_sample_perf)
            results['parameter_stability'].append(self._calculate_parameter_stability(optimal_params))
            results['optimization_history'].append(optimal_params)
            
        # Calculate walk-forward efficiency
        results['walk_forward_efficiency'] = self._calculate_walk_forward_efficiency(results)
        results['parameter_robustness'] = self._analyze_parameter_robustness(results)
        
        return results
        
    def _calculate_walk_forward_periods(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Calculate walk-forward analysis periods"""
        periods = []
        
        current_date = self.config.start_date
        
        while current_date < self.config.end_date:
            # In-sample period
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.in_sample_period)
            
            # Out-of-sample period
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.out_sample_period)
            
            # Check if we have enough data
            if test_end > self.config.end_date:
                break
                
            periods.append((train_start, train_end, test_start, test_end))
            
            # Move to next period
            current_date = test_start
            
        return periods
        
    async def _optimize_parameters(self, strategy: Any, start: datetime, end: datetime) -> Dict:
        """Optimize strategy parameters using genetic algorithm or grid search"""
        # Get parameter space
        param_space = strategy.get_parameter_space()
        
        best_params = None
        best_score = -float('inf')
        
        # Grid search (can be replaced with more sophisticated optimization)
        for params in self._generate_parameter_combinations(param_space):
            # Test parameters
            score = await self._evaluate_parameters(strategy, params, start, end)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params
        
    async def _run_monte_carlo_simulation(self, base_results: BacktestResults) -> Dict:
        """
        Run Monte Carlo simulation on trade results
        """
        logger.info(f"Running {self.config.monte_carlo_runs} Monte Carlo simulations...")
        
        # Extract trade returns
        trade_returns = self._extract_trade_returns(base_results)
        
        if len(trade_returns) < 10:
            logger.warning("Insufficient trades for Monte Carlo simulation")
            return {}
            
        # Run simulations
        simulation_results = []
        
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for i in range(self.config.monte_carlo_runs):
                future = executor.submit(
                    self._run_single_monte_carlo,
                    trade_returns,
                    len(trade_returns),
                    self.config.initial_capital
                )
                futures.append(future)
                
            # Collect results
            for future in futures:
                simulation_results.append(future.result())
                
        # Analyze results
        mc_results = self._analyze_monte_carlo_results(simulation_results)
        
        return mc_results
        
    def _run_single_monte_carlo(self, trade_returns: List[float], 
                               num_trades: int, 
                               initial_capital: float) -> Dict:
        """Run single Monte Carlo simulation"""
        # Randomly sample trades with replacement
        sampled_returns = np.random.choice(trade_returns, size=num_trades, replace=True)
        
        # Calculate equity curve
        equity = initial_capital
        equity_curve = [equity]
        
        for ret in sampled_returns:
            equity *= (1 + ret)
            equity_curve.append(equity)
            
        # Calculate metrics
        total_return = (equity / initial_capital - 1)
        max_dd = self._calculate_max_drawdown(equity_curve)
        
        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'final_equity': equity,
            'equity_curve': equity_curve
        }
        
    def _analyze_monte_carlo_results(self, results: List[Dict]) -> Dict:
        """Analyze Monte Carlo simulation results"""
        # Extract metrics
        returns = [r['total_return'] for r in results]
        max_dds = [r['max_drawdown'] for r in results]
        final_equities = [r['final_equity'] for r in results]
        
        # Calculate statistics
        mc_analysis = {
            'return_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'skew': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'percentiles': {}
            },
            'drawdown_distribution': {
                'mean': np.mean(max_dds),
                'std': np.std(max_dds),
                'percentiles': {}
            },
            'risk_metrics': {
                'probability_of_loss': sum(1 for r in returns if r < 0) / len(returns),
                'var_95': np.percentile(returns, 5),
                'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
                'probability_of_ruin': sum(1 for dd in max_dds if dd > 0.5) / len(max_dds)
            },
            'confidence_intervals': {}
        }
        
        # Calculate percentiles
        for level in self.config.confidence_levels:
            percentile = level * 100
            mc_analysis['return_distribution']['percentiles'][f'p{int(percentile)}'] = np.percentile(returns, percentile)
            mc_analysis['drawdown_distribution']['percentiles'][f'p{int(percentile)}'] = np.percentile(max_dds, percentile)
            
        # Calculate confidence intervals for expected return
        for confidence in [0.90, 0.95, 0.99]:
            lower = np.percentile(returns, (1 - confidence) / 2 * 100)
            upper = np.percentile(returns, (1 + confidence) / 2 * 100)
            mc_analysis['confidence_intervals'][f'{int(confidence*100)}%'] = (lower, upper)
            
        return mc_analysis
        
    async def _analyze_parameter_sensitivity(self, strategy: Any) -> Dict:
        """Analyze how sensitive results are to parameter changes"""
        base_params = strategy.get_parameters()
        sensitivity_results = {}
        
        # Test each parameter
        for param_name, param_value in base_params.items():
            if isinstance(param_value, (int, float)):
                # Test different values
                test_values = [
                    param_value * 0.5,
                    param_value * 0.75,
                    param_value,
                    param_value * 1.25,
                    param_value * 1.5
                ]
                
                param_results = []
                
                for test_value in test_values:
                    # Create modified parameters
                    test_params = base_params.copy()
                    test_params[param_name] = test_value
                    
                    # Run backtest with modified parameters
                    strategy.set_parameters(test_params)
                    results = await self._run_basic_backtest(strategy)
                    
                    param_results.append({
                        'value': test_value,
                        'sharpe_ratio': results.sharpe_ratio,
                        'total_return': results.total_return,
                        'max_drawdown': results.max_drawdown,
                        'win_rate': results.win_rate
                    })
                    
                sensitivity_results[param_name] = param_results
                
        # Reset to original parameters
        strategy.set_parameters(base_params)
        
        return sensitivity_results
        
    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.results:
            return "No backtest results available"
            
        report = f"""
# Nexlify Advanced Backtest Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Period: {self.config.start_date.date()} to {self.config.end_date.date()}
- Initial Capital: ${self.config.initial_capital:,.2f}
- Universe: {', '.join(self.config.universe)}

## Performance Summary
{self._format_performance_summary()}

## Trade Statistics
{self._format_trade_statistics()}

## Risk Analysis
{self._format_risk_analysis()}

## Walk-Forward Analysis
{self._format_walk_forward_results()}

## Monte Carlo Simulation
{self._format_monte_carlo_results()}

## Parameter Sensitivity
{self._format_parameter_sensitivity()}

## Recommendations
{self._generate_recommendations()}
"""
        return report
        
    def plot_results(self, save_path: Optional[str] = None):
        """Generate comprehensive visualization of backtest results"""
        if not self.results:
            logger.warning("No results to plot")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Nexlify Advanced Backtest Results', fontsize=16)
        
        # 1. Equity Curve
        ax = axes[0, 0]
        self.results.equity_curve.plot(ax=ax, color='green', linewidth=2)
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        self.results.drawdown_series.plot(ax=ax, color='red', linewidth=2)
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.fill_between(self.results.drawdown_series.index, 
                       self.results.drawdown_series.values, 
                       alpha=0.3, color='red')
        ax.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        ax = axes[0, 2]
        if self.results.monthly_returns is not None:
            monthly_matrix = self.results.monthly_returns.values.reshape(-1, 12)
            sns.heatmap(monthly_matrix, 
                       annot=True, 
                       fmt='.1%', 
                       cmap='RdYlGn', 
                       center=0,
                       ax=ax)
            ax.set_title('Monthly Returns Heatmap')
            
        # 4. Return Distribution
        ax = axes[1, 0]
        self.results.returns.hist(bins=50, ax=ax, color='blue', alpha=0.7)
        ax.set_title('Return Distribution')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 5. Monte Carlo Results
        ax = axes[1, 1]
        if self.results.monte_carlo_results:
            mc_returns = self.results.monte_carlo_results['return_distribution']
            x = np.linspace(mc_returns['mean'] - 3*mc_returns['std'], 
                          mc_returns['mean'] + 3*mc_returns['std'], 100)
            y = stats.norm.pdf(x, mc_returns['mean'], mc_returns['std'])
            ax.plot(x, y, 'b-', linewidth=2, label='Normal Fit')
            ax.axvline(x=mc_returns['mean'], color='green', 
                      linestyle='--', label=f"Mean: {mc_returns['mean']:.1%}")
            ax.set_title('Monte Carlo Return Distribution')
            ax.set_xlabel('Total Return')
            ax.legend()
            
        # 6. Parameter Sensitivity
        ax = axes[1, 2]
        if self.results.parameter_sensitivity:
            # Plot first parameter sensitivity
            first_param = list(self.results.parameter_sensitivity.keys())[0]
            param_data = self.results.parameter_sensitivity[first_param]
            
            values = [d['value'] for d in param_data]
            sharpes = [d['sharpe_ratio'] for d in param_data]
            
            ax.plot(values, sharpes, 'o-', linewidth=2, markersize=8)
            ax.set_title(f'Sensitivity: {first_param}')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Sharpe Ratio')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def _calculate_results(self, portfolio: 'Portfolio', trades: List[Dict]) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        equity_curve = pd.Series(portfolio.equity_history)
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_dd = self._calculate_max_drawdown(equity_curve.values)
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                           sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        # Create results
        results = BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            returns=returns,
            drawdown_series=self._calculate_drawdown_series(equity_curve)
        )
        
        return results
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
    def _calculate_max_drawdown(self, equity_curve: np.array) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative) / cumulative
        return abs(drawdown.min())
        
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        cumulative = equity_curve.cummax()
        drawdown = (equity_curve - cumulative) / cumulative
        return drawdown

class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.positions = {}
        self.equity_history = {}
        self.initial_capital = initial_capital
        
    def update_market_prices(self, market_data: pd.Series):
        """Update position values with current market prices"""
        # Implementation here
        pass
        
    def execute_trade(self, trade: Dict):
        """Execute a trade in the portfolio"""
        # Implementation here
        pass
        
    def record_equity(self, timestamp: datetime):
        """Record current portfolio equity"""
        total_value = self.cash
        for position in self.positions.values():
            total_value += position['value']
        self.equity_history[timestamp] = total_value
        
    def can_execute_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Check if trade can be executed"""
        if side == 'buy':
            required_cash = size * price
            return self.cash >= required_cash
        else:
            # Check if we have enough position to sell
            if symbol in self.positions:
                return self.positions[symbol]['size'] >= size
            return False
