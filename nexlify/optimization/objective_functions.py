"""
Objective Functions for Hyperparameter Optimization
Defines different optimization objectives for RL agent performance
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """
    Base class for optimization objective functions

    Objective functions evaluate the performance of trained models
    and return a score to maximize (or minimize).
    """

    def __init__(self, name: str, direction: str = 'maximize'):
        """
        Initialize objective function

        Args:
            name: Name of the objective
            direction: 'maximize' or 'minimize'
        """
        if direction not in {'maximize', 'minimize'}:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction}")

        self.name = name
        self.direction = direction
        logger.debug(f"Initialized {name} objective (direction: {direction})")

    @abstractmethod
    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate objective score from training results

        Args:
            training_results: Dictionary containing training metrics
                Expected keys depend on specific objective

        Returns:
            Score value (higher is better for maximize, lower for minimize)
        """
        pass

    def __call__(self, training_results: Dict[str, Any]) -> float:
        """Allow objective to be called directly"""
        return self.calculate(training_results)


class SharpeObjective(ObjectiveFunction):
    """
    Sharpe ratio objective - risk-adjusted returns

    Maximizes Sharpe ratio = (mean_return - risk_free_rate) / std_return
    """

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: float = 252.0):
        """
        Initialize Sharpe ratio objective

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.0)
            annualization_factor: Factor for annualizing returns (default: 252 for daily)
        """
        super().__init__('sharpe_ratio', direction='maximize')
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate Sharpe ratio from training results

        Args:
            training_results: Dict with keys:
                - 'returns': List/array of returns
                OR
                - 'sharpe_ratio': Pre-calculated Sharpe ratio

        Returns:
            Sharpe ratio value
        """
        # Check if pre-calculated
        if 'sharpe_ratio' in training_results:
            return float(training_results['sharpe_ratio'])

        # Calculate from returns
        if 'returns' not in training_results:
            raise ValueError("training_results must contain 'returns' or 'sharpe_ratio'")

        returns = np.array(training_results['returns'])

        if len(returns) == 0:
            logger.warning("Empty returns array, returning -inf")
            return float('-inf')

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Check for zero or near-zero std (use tolerance for numerical stability)
        if std_return < 1e-10:
            logger.warning("Zero standard deviation in returns, returning -inf")
            return float('-inf')

        # Annualized Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return
        sharpe_annualized = sharpe * np.sqrt(self.annualization_factor)

        return float(sharpe_annualized)


class ReturnObjective(ObjectiveFunction):
    """
    Total return objective - maximize cumulative returns

    Optionally penalizes volatility
    """

    def __init__(self, volatility_penalty: float = 0.0):
        """
        Initialize return objective

        Args:
            volatility_penalty: Penalty factor for return volatility (default: 0.0)
                Higher values penalize volatile returns
        """
        super().__init__('total_return', direction='maximize')
        self.volatility_penalty = volatility_penalty

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate total return from training results

        Args:
            training_results: Dict with keys:
                - 'final_balance': Final portfolio value
                - 'initial_balance': Initial portfolio value
                OR
                - 'total_return': Pre-calculated total return
                Optional:
                - 'returns': For volatility penalty

        Returns:
            Total return (optionally adjusted for volatility)
        """
        # Check if pre-calculated
        if 'total_return' in training_results:
            total_return = float(training_results['total_return'])
        else:
            # Calculate from balances
            if 'final_balance' not in training_results or 'initial_balance' not in training_results:
                raise ValueError(
                    "training_results must contain 'final_balance' and 'initial_balance' "
                    "or 'total_return'"
                )

            final_balance = training_results['final_balance']
            initial_balance = training_results['initial_balance']

            if initial_balance <= 0:
                raise ValueError(f"initial_balance must be positive, got {initial_balance}")

            total_return = (final_balance - initial_balance) / initial_balance

        # Apply volatility penalty if configured
        if self.volatility_penalty > 0 and 'returns' in training_results:
            returns = np.array(training_results['returns'])
            if len(returns) > 0:
                volatility = np.std(returns)
                total_return -= self.volatility_penalty * volatility
                logger.debug(f"Applied volatility penalty: {self.volatility_penalty * volatility:.4f}")

        return float(total_return)


class DrawdownObjective(ObjectiveFunction):
    """
    Maximum drawdown objective - minimize worst drawdown

    Lower drawdown is better (more stable strategy)
    """

    def __init__(self):
        super().__init__('max_drawdown', direction='minimize')

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate maximum drawdown from training results

        Args:
            training_results: Dict with keys:
                - 'balance_history': List/array of portfolio balances
                OR
                - 'max_drawdown': Pre-calculated max drawdown

        Returns:
            Maximum drawdown (as negative percentage)
        """
        # Check if pre-calculated
        if 'max_drawdown' in training_results:
            return float(training_results['max_drawdown'])

        # Calculate from balance history
        if 'balance_history' not in training_results:
            raise ValueError("training_results must contain 'balance_history' or 'max_drawdown'")

        balances = np.array(training_results['balance_history'])

        if len(balances) == 0:
            logger.warning("Empty balance history, returning 0")
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(balances)

        # Calculate drawdown at each point
        drawdowns = (balances - running_max) / running_max

        # Maximum drawdown (most negative)
        max_drawdown = np.min(drawdowns)

        return float(max_drawdown)


class WinRateObjective(ObjectiveFunction):
    """
    Win rate objective - maximize percentage of profitable trades
    """

    def __init__(self):
        super().__init__('win_rate', direction='maximize')

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate win rate from training results

        Args:
            training_results: Dict with keys:
                - 'trades': List of trade results (positive = win, negative = loss)
                OR
                - 'win_rate': Pre-calculated win rate

        Returns:
            Win rate (0.0 to 1.0)
        """
        # Check if pre-calculated
        if 'win_rate' in training_results:
            return float(training_results['win_rate'])

        # Calculate from trades
        if 'trades' not in training_results:
            raise ValueError("training_results must contain 'trades' or 'win_rate'")

        trades = np.array(training_results['trades'])

        if len(trades) == 0:
            logger.warning("No trades found, returning 0")
            return 0.0

        wins = np.sum(trades > 0)
        win_rate = wins / len(trades)

        return float(win_rate)


class MultiObjective(ObjectiveFunction):
    """
    Multi-objective function combining multiple objectives

    Weighted sum of individual objectives
    """

    def __init__(
        self,
        objectives: List[Tuple[ObjectiveFunction, float]],
        name: str = 'multi_objective'
    ):
        """
        Initialize multi-objective function

        Args:
            objectives: List of (objective, weight) tuples
            name: Name for this multi-objective

        Example:
            >>> objectives = [
            ...     (SharpeObjective(), 0.5),
            ...     (ReturnObjective(), 0.3),
            ...     (DrawdownObjective(), 0.2)
            ... ]
            >>> multi_obj = MultiObjective(objectives)
        """
        # All objectives should have same direction (maximize)
        # We'll convert minimize objectives by negating their scores
        super().__init__(name, direction='maximize')

        self.objectives = objectives

        # Normalize weights
        total_weight = sum(weight for _, weight in objectives)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        self.objectives = [
            (obj, weight / total_weight)
            for obj, weight in objectives
        ]

        logger.info(f"Initialized multi-objective with {len(objectives)} components:")
        for obj, weight in self.objectives:
            logger.info(f"  - {obj.name}: {weight:.2%}")

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate weighted sum of objectives

        Args:
            training_results: Dict with metrics for all objectives

        Returns:
            Weighted sum score
        """
        total_score = 0.0

        for objective, weight in self.objectives:
            try:
                score = objective.calculate(training_results)

                # Convert minimize objectives to maximize
                if objective.direction == 'minimize':
                    score = -score

                weighted_score = score * weight
                total_score += weighted_score

                logger.debug(
                    f"{objective.name}: {score:.4f} "
                    f"(weighted: {weighted_score:.4f})"
                )

            except Exception as e:
                logger.error(
                    f"Error calculating {objective.name}: {e}. "
                    f"Skipping this objective."
                )
                # Continue with other objectives

        return float(total_score)


class ProfitFactorObjective(ObjectiveFunction):
    """
    Profit factor objective - ratio of gross profit to gross loss

    Profit factor = sum(winning_trades) / abs(sum(losing_trades))
    """

    def __init__(self):
        super().__init__('profit_factor', direction='maximize')

    def calculate(self, training_results: Dict[str, Any]) -> float:
        """
        Calculate profit factor from training results

        Args:
            training_results: Dict with keys:
                - 'trades': List of trade results
                OR
                - 'profit_factor': Pre-calculated profit factor

        Returns:
            Profit factor value
        """
        # Check if pre-calculated
        if 'profit_factor' in training_results:
            return float(training_results['profit_factor'])

        # Calculate from trades
        if 'trades' not in training_results:
            raise ValueError("training_results must contain 'trades' or 'profit_factor'")

        trades = np.array(training_results['trades'])

        if len(trades) == 0:
            logger.warning("No trades found, returning 0")
            return 0.0

        gross_profit = np.sum(trades[trades > 0])
        gross_loss = abs(np.sum(trades[trades < 0]))

        if gross_loss == 0:
            # All trades profitable or no losing trades
            return float(gross_profit) if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return float(profit_factor)


def create_objective(
    objective_type: str,
    **kwargs
) -> ObjectiveFunction:
    """
    Factory function to create objective by type

    Args:
        objective_type: Type of objective
            ('sharpe', 'return', 'drawdown', 'winrate', 'profit_factor', 'multi')
        **kwargs: Additional arguments for objective constructor

    Returns:
        ObjectiveFunction instance

    Example:
        >>> obj = create_objective('sharpe', risk_free_rate=0.02)
        >>> obj = create_objective('return', volatility_penalty=0.1)
        >>> obj = create_objective(
        ...     'multi',
        ...     objectives=[
        ...         (SharpeObjective(), 0.5),
        ...         (DrawdownObjective(), 0.5)
        ...     ]
        ... )
    """
    objective_map = {
        'sharpe': SharpeObjective,
        'return': ReturnObjective,
        'drawdown': DrawdownObjective,
        'winrate': WinRateObjective,
        'profit_factor': ProfitFactorObjective,
        'multi': MultiObjective,
    }

    if objective_type not in objective_map:
        raise ValueError(
            f"Unknown objective type '{objective_type}'. "
            f"Choose from: {list(objective_map.keys())}"
        )

    objective_class = objective_map[objective_type]
    return objective_class(**kwargs)


# Predefined multi-objective configurations
def create_balanced_objective() -> MultiObjective:
    """
    Create balanced multi-objective optimizing returns, risk, and stability

    Objectives:
    - 40% Sharpe ratio (risk-adjusted returns)
    - 30% Total return (absolute performance)
    - 30% Max drawdown (stability)
    """
    return MultiObjective([
        (SharpeObjective(), 0.4),
        (ReturnObjective(), 0.3),
        (DrawdownObjective(), 0.3),
    ], name='balanced')


def create_aggressive_objective() -> MultiObjective:
    """
    Create aggressive multi-objective prioritizing returns over risk

    Objectives:
    - 60% Total return
    - 30% Sharpe ratio
    - 10% Max drawdown
    """
    return MultiObjective([
        (ReturnObjective(), 0.6),
        (SharpeObjective(), 0.3),
        (DrawdownObjective(), 0.1),
    ], name='aggressive')


def create_conservative_objective() -> MultiObjective:
    """
    Create conservative multi-objective prioritizing stability

    Objectives:
    - 50% Max drawdown (minimize risk)
    - 30% Sharpe ratio
    - 20% Total return
    """
    return MultiObjective([
        (DrawdownObjective(), 0.5),
        (SharpeObjective(), 0.3),
        (ReturnObjective(), 0.2),
    ], name='conservative')
