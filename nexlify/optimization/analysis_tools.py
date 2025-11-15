"""
Analysis and Visualization Tools
Tools for analyzing and visualizing hyperparameter optimization results
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationAnalyzer:
    """
    Analyzer for hyperparameter optimization results

    Provides comprehensive analysis including:
    - Convergence analysis
    - Parameter sensitivity
    - Trial comparison
    - Statistical summaries
    """

    def __init__(self, study: Optional['optuna.Study'] = None, history_file: Optional[str] = None):
        """
        Initialize optimization analyzer

        Args:
            study: Optuna study object
            history_file: Path to optimization history JSON file
                (Use if study is not available)
        """
        self.study = study
        self.history = None

        if history_file:
            self.load_history(history_file)

        if not study and not history_file:
            raise ValueError("Either study or history_file must be provided")

    def load_history(self, history_file: str) -> None:
        """Load optimization history from JSON file"""
        with open(history_file, 'r') as f:
            self.history = json.load(f)
        logger.info(f"Loaded {len(self.history)} trials from {history_file}")

    def get_trial_dataframe(self) -> pd.DataFrame:
        """
        Convert trials to pandas DataFrame for analysis

        Returns:
            DataFrame with trial data
        """
        if self.study:
            # From Optuna study
            trials_data = []
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    row = {
                        'trial_number': trial.number,
                        'value': trial.value,
                        **trial.params
                    }
                    trials_data.append(row)
            return pd.DataFrame(trials_data)

        elif self.history:
            # From history file
            trials_data = []
            for trial in self.history:
                row = {
                    'trial_number': trial['trial_number'],
                    'score': trial['score'],
                    **trial['params']
                }
                trials_data.append(row)
            return pd.DataFrame(trials_data)

        return pd.DataFrame()

    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze optimization convergence

        Returns:
            Dict with convergence metrics:
                - best_trial_number: When best value was found
                - improvement_rate: Rate of improvement over trials
                - converged: Whether optimization has converged
                - convergence_trial: Trial number where convergence occurred
        """
        df = self.get_trial_dataframe()
        if df.empty:
            return {}

        value_col = 'value' if 'value' in df.columns else 'score'
        values = df[value_col].values

        # Find best trial
        best_idx = np.argmax(values)  # Assuming maximization
        best_trial_number = df.iloc[best_idx]['trial_number']

        # Calculate improvement rate
        # Use exponential moving average
        ema_span = max(10, len(values) // 10)
        ema = pd.Series(values).ewm(span=ema_span).mean()
        improvement_rate = (ema.iloc[-1] - ema.iloc[0]) / len(values)

        # Check convergence (no improvement in last 20% of trials)
        last_20_pct = int(len(values) * 0.2)
        if last_20_pct > 0:
            recent_best = np.max(values[-last_20_pct:])
            overall_best = np.max(values)
            converged = (recent_best >= overall_best * 0.995)  # Within 0.5%

            # Find convergence trial
            convergence_threshold = overall_best * 0.99
            convergence_trials = np.where(values >= convergence_threshold)[0]
            convergence_trial = int(convergence_trials[0]) if len(convergence_trials) > 0 else None
        else:
            converged = False
            convergence_trial = None

        return {
            'best_trial_number': int(best_trial_number),
            'best_value': float(values[best_idx]),
            'improvement_rate': float(improvement_rate),
            'converged': converged,
            'convergence_trial': convergence_trial,
            'total_trials': len(values)
        }

    def analyze_parameter_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter sensitivity using correlation

        Returns:
            Dict mapping parameter names to sensitivity metrics:
                - correlation: Pearson correlation with objective
                - std: Standard deviation of parameter values
                - range: (min, max) of parameter values
        """
        df = self.get_trial_dataframe()
        if df.empty:
            return {}

        value_col = 'value' if 'value' in df.columns else 'score'

        # Exclude non-parameter columns
        param_cols = [col for col in df.columns if col not in ['trial_number', 'value', 'score']]

        sensitivity = {}
        for param in param_cols:
            # Handle categorical parameters
            if df[param].dtype == 'object':
                # For categoricals, use one-hot encoding
                continue

            # Numerical parameters
            correlation = df[param].corr(df[value_col])

            sensitivity[param] = {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'std': float(df[param].std()),
                'min': float(df[param].min()),
                'max': float(df[param].max()),
                'mean': float(df[param].mean())
            }

        # Sort by absolute correlation
        sensitivity = dict(sorted(
            sensitivity.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))

        return sensitivity

    def get_top_trials(self, n: int = 10, minimize: bool = False) -> pd.DataFrame:
        """
        Get top N trials by objective value

        Args:
            n: Number of top trials to return
            minimize: If True, return trials with lowest values

        Returns:
            DataFrame of top trials
        """
        df = self.get_trial_dataframe()
        if df.empty:
            return pd.DataFrame()

        value_col = 'value' if 'value' in df.columns else 'score'

        sorted_df = df.sort_values(value_col, ascending=minimize)
        return sorted_df.head(n)

    def compare_trials(
        self,
        trial_numbers: List[int],
        params_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare specific trials

        Args:
            trial_numbers: List of trial numbers to compare
            params_to_compare: Specific parameters to include (None = all)

        Returns:
            DataFrame comparing trials
        """
        df = self.get_trial_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Filter to requested trials
        comparison_df = df[df['trial_number'].isin(trial_numbers)]

        # Filter to requested parameters
        if params_to_compare:
            cols = ['trial_number'] + params_to_compare
            cols = [c for c in cols if c in comparison_df.columns]
            comparison_df = comparison_df[cols]

        return comparison_df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of optimization

        Returns:
            Dict with summary statistics
        """
        df = self.get_trial_dataframe()
        if df.empty:
            return {}

        value_col = 'value' if 'value' in df.columns else 'score'
        values = df[value_col]

        # Exclude non-parameter columns for parameter stats
        param_cols = [col for col in df.columns if col not in ['trial_number', 'value', 'score']]

        stats = {
            'objective': {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75))
            },
            'parameters': {}
        }

        # Parameter statistics
        for param in param_cols:
            if df[param].dtype in ['int64', 'float64']:
                stats['parameters'][param] = {
                    'mean': float(df[param].mean()),
                    'std': float(df[param].std()),
                    'min': float(df[param].min()),
                    'max': float(df[param].max())
                }

        return stats


class OptimizationVisualizer:
    """
    Visualizer for hyperparameter optimization results
    """

    def __init__(self, study: Optional['optuna.Study'] = None, analyzer: Optional[OptimizationAnalyzer] = None):
        """
        Initialize visualizer

        Args:
            study: Optuna study object
            analyzer: OptimizationAnalyzer instance
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available, plotting disabled")

        self.study = study
        self.analyzer = analyzer

        if not study and not analyzer:
            raise ValueError("Either study or analyzer must be provided")

        # Set plotting style
        if PLOTTING_AVAILABLE:
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 6)

    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence

        Args:
            save_path: Path to save plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return

        df = self.analyzer.get_trial_dataframe()
        value_col = 'value' if 'value' in df.columns else 'score'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Raw values
        ax1.plot(df['trial_number'], df[value_col], alpha=0.6, marker='o', markersize=4)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)

        # Best value so far
        best_so_far = df[value_col].cummax()
        ax2.plot(df['trial_number'], best_so_far, color='green', linewidth=2)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Best Value So Far')
        ax2.set_title('Best Value Progression')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {save_path}")
        else:
            plt.show()

    def plot_parameter_correlations(
        self,
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot parameter correlations with objective

        Args:
            top_n: Number of top parameters to plot
            save_path: Path to save plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return

        sensitivity = self.analyzer.analyze_parameter_sensitivity()

        # Get top N by absolute correlation
        top_params = list(sensitivity.keys())[:top_n]
        correlations = [sensitivity[p]['correlation'] for p in top_params]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if c > 0 else 'red' for c in correlations]
        ax.barh(top_params, correlations, color=colors, alpha=0.7)
        ax.set_xlabel('Correlation with Objective')
        ax.set_title(f'Top {top_n} Parameter Correlations')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation plot to {save_path}")
        else:
            plt.show()

    def plot_parameter_distributions(
        self,
        params: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distributions of parameter values

        Args:
            params: List of parameters to plot (None = all numerical)
            save_path: Path to save plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return

        df = self.analyzer.get_trial_dataframe()

        # Get numerical parameters
        if params is None:
            params = [col for col in df.columns
                     if col not in ['trial_number', 'value', 'score']
                     and df[col].dtype in ['int64', 'float64']]

        n_params = len(params)
        if n_params == 0:
            logger.warning("No numerical parameters to plot")
            return

        # Create subplots
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(params):
            ax = axes[i]
            ax.hist(df[param], bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(param)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution: {param}')
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        else:
            plt.show()

    def create_comprehensive_report(self, output_dir: str) -> None:
        """
        Create comprehensive visualization report

        Args:
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating comprehensive report in {output_dir}...")

        # Convergence plot
        self.plot_convergence(save_path=str(output_path / 'convergence.png'))

        # Parameter correlations
        self.plot_parameter_correlations(save_path=str(output_path / 'correlations.png'))

        # Parameter distributions
        self.plot_parameter_distributions(save_path=str(output_path / 'distributions.png'))

        # Optuna-specific plots
        if OPTUNA_AVAILABLE and self.study:
            try:
                # Optimization history
                fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
                plt.savefig(output_path / 'optuna_history.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Parameter importance
                fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
                plt.savefig(output_path / 'optuna_importance.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Parallel coordinate
                fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
                plt.savefig(output_path / 'optuna_parallel.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info("Generated Optuna-specific plots")
            except Exception as e:
                logger.warning(f"Failed to generate some Optuna plots: {e}")

        logger.info(f"Comprehensive report generated in {output_dir}")


def load_and_analyze(
    study_path: Optional[str] = None,
    history_path: Optional[str] = None
) -> Tuple[OptimizationAnalyzer, OptimizationVisualizer]:
    """
    Load optimization results and create analyzer/visualizer

    Args:
        study_path: Path to pickled Optuna study
        history_path: Path to optimization history JSON

    Returns:
        Tuple of (analyzer, visualizer)

    Example:
        >>> analyzer, visualizer = load_and_analyze(
        ...     history_path='optimization_results/optimization_history_20231115.json'
        ... )
        >>> convergence = analyzer.analyze_convergence()
        >>> visualizer.plot_convergence(save_path='convergence.png')
    """
    study = None
    if study_path:
        import pickle
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        logger.info(f"Loaded study from {study_path}")

    analyzer = OptimizationAnalyzer(study=study, history_file=history_path)
    visualizer = OptimizationVisualizer(study=study, analyzer=analyzer)

    return analyzer, visualizer
