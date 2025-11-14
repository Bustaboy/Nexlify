"""
Real-Time Training Dashboard
Interactive web dashboard for monitoring RL agent training
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import time

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import pandas as pd
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Dash not available. Install with: pip install dash plotly pandas"
    )

import numpy as np

from nexlify.monitoring.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """
    Real-time training dashboard with Plotly Dash

    Features:
    - Live updating plots (profit, loss, epsilon, etc.)
    - KPI cards with color-coded alerts
    - Episode comparison tool
    - Model diagnostics
    - Experiment tracking

    Example:
        >>> dashboard = TrainingDashboard(metrics_logger=logger, port=8050)
        >>> dashboard.start(blocking=False)
        >>> # Dashboard accessible at http://localhost:8050
        >>> # ... training happens ...
        >>> dashboard.stop()
    """

    def __init__(
        self,
        metrics_logger: MetricsLogger,
        port: int = 8050,
        update_interval: int = 2000,  # ms
        theme: str = 'cyberpunk'
    ):
        """
        Initialize training dashboard

        Args:
            metrics_logger: MetricsLogger instance to read from
            port: Port to run dashboard on
            update_interval: Update interval in milliseconds
            theme: Dashboard theme ('cyberpunk', 'dark', 'light')
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash required for dashboard. "
                "Install with: pip install dash plotly pandas"
            )

        self.metrics_logger = metrics_logger
        self.port = port
        self.update_interval = update_interval
        self.theme = theme

        # Dashboard app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"TrainingDashboard initialized on port {port}")

    def _get_theme_colors(self) -> Dict[str, str]:
        """Get color scheme for theme"""
        themes = {
            'cyberpunk': {
                'bg': '#0a0e27',
                'card_bg': '#1a1f3a',
                'text': '#00ff9f',
                'accent': '#ff00ff',
                'profit': '#00ff9f',
                'loss': '#ff0080',
                'warning': '#ffaa00',
                'grid': '#2a2f4a'
            },
            'dark': {
                'bg': '#111111',
                'card_bg': '#1e1e1e',
                'text': '#ffffff',
                'accent': '#00bcd4',
                'profit': '#4caf50',
                'loss': '#f44336',
                'warning': '#ff9800',
                'grid': '#333333'
            },
            'light': {
                'bg': '#ffffff',
                'card_bg': '#f5f5f5',
                'text': '#000000',
                'accent': '#2196f3',
                'profit': '#4caf50',
                'loss': '#f44336',
                'warning': '#ff9800',
                'grid': '#e0e0e0'
            }
        }
        return themes.get(self.theme, themes['dark'])

    def _setup_layout(self) -> None:
        """Setup dashboard layout"""
        colors = self._get_theme_colors()

        self.app.layout = html.Div(
            style={
                'backgroundColor': colors['bg'],
                'color': colors['text'],
                'fontFamily': 'monospace',
                'padding': '20px'
            },
            children=[
                # Header
                html.H1(
                    f"🚀 {self.metrics_logger.experiment_name} - Training Monitor",
                    style={
                        'textAlign': 'center',
                        'color': colors['accent'],
                        'marginBottom': '20px'
                    }
                ),

                # Auto-refresh interval
                dcc.Interval(
                    id='interval-component',
                    interval=self.update_interval,
                    n_intervals=0
                ),

                # KPI Cards Row
                html.Div(
                    id='kpi-cards',
                    style={
                        'display': 'flex',
                        'justifyContent': 'space-around',
                        'marginBottom': '20px'
                    }
                ),

                # Main Charts
                html.Div([
                    # Row 1: Profit and Loss
                    html.Div([
                        dcc.Graph(id='profit-chart', style={'width': '50%', 'display': 'inline-block'}),
                        dcc.Graph(id='loss-chart', style={'width': '50%', 'display': 'inline-block'}),
                    ]),

                    # Row 2: Epsilon and Learning Rate
                    html.Div([
                        dcc.Graph(id='epsilon-chart', style={'width': '50%', 'display': 'inline-block'}),
                        dcc.Graph(id='lr-chart', style={'width': '50%', 'display': 'inline-block'}),
                    ]),

                    # Row 3: Win Rate and Sharpe
                    html.Div([
                        dcc.Graph(id='winrate-chart', style={'width': '50%', 'display': 'inline-block'}),
                        dcc.Graph(id='sharpe-chart', style={'width': '50%', 'display': 'inline-block'}),
                    ]),

                    # Row 4: Drawdown and Q-Values
                    html.Div([
                        dcc.Graph(id='drawdown-chart', style={'width': '50%', 'display': 'inline-block'}),
                        dcc.Graph(id='qvalue-chart', style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                ]),

                # Statistics Table
                html.Div(id='stats-table', style={'marginTop': '20px'}),
            ]
        )

    def _setup_callbacks(self) -> None:
        """Setup Dash callbacks for live updates"""

        @self.app.callback(
            [
                Output('kpi-cards', 'children'),
                Output('profit-chart', 'figure'),
                Output('loss-chart', 'figure'),
                Output('epsilon-chart', 'figure'),
                Output('lr-chart', 'figure'),
                Output('winrate-chart', 'figure'),
                Output('sharpe-chart', 'figure'),
                Output('drawdown-chart', 'figure'),
                Output('qvalue-chart', 'figure'),
                Output('stats-table', 'children'),
            ],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components"""
            colors = self._get_theme_colors()

            # Get latest data
            episodes = self.metrics_logger.get_episode_history()
            latest = self.metrics_logger.get_latest_episode()
            stats = self.metrics_logger.get_statistics()
            best = self.metrics_logger.get_best_episode('profit')

            if not episodes:
                # Empty state
                empty_fig = self._create_empty_figure(colors)
                return (
                    self._create_kpi_cards(colors, {}, {}, {}),
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    html.Div("No data yet...")
                )

            # Create figures
            profit_fig = self._create_profit_chart(episodes, colors)
            loss_fig = self._create_loss_chart(colors)
            epsilon_fig = self._create_epsilon_chart(episodes, colors)
            lr_fig = self._create_lr_chart(episodes, colors)
            winrate_fig = self._create_winrate_chart(episodes, colors)
            sharpe_fig = self._create_sharpe_chart(episodes, colors)
            drawdown_fig = self._create_drawdown_chart(episodes, colors)
            qvalue_fig = self._create_qvalue_chart(colors)

            # KPI cards
            kpi_cards = self._create_kpi_cards(colors, latest, best, stats)

            # Stats table
            stats_table = self._create_stats_table(colors, stats)

            return (
                kpi_cards,
                profit_fig,
                loss_fig,
                epsilon_fig,
                lr_fig,
                winrate_fig,
                sharpe_fig,
                drawdown_fig,
                qvalue_fig,
                stats_table
            )

    def _create_kpi_cards(
        self,
        colors: Dict[str, str],
        latest: Dict[str, Any],
        best: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> List:
        """Create KPI cards with color-coded alerts"""

        def create_card(title, value, status='normal'):
            status_colors = {
                'good': colors['profit'],
                'warning': colors['warning'],
                'critical': colors['loss'],
                'normal': colors['text']
            }

            return html.Div(
                style={
                    'backgroundColor': colors['card_bg'],
                    'padding': '15px',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'border': f"2px solid {status_colors[status]}",
                    'minWidth': '150px'
                },
                children=[
                    html.H4(title, style={'margin': '0', 'fontSize': '14px'}),
                    html.H2(
                        value,
                        style={
                            'margin': '10px 0 0 0',
                            'color': status_colors[status]
                        }
                    )
                ]
            )

        cards = []

        # Current Episode
        if latest:
            cards.append(
                create_card(
                    "Current Episode",
                    f"{latest.get('episode', 0)}",
                    'normal'
                )
            )

            # Latest Profit
            profit = latest.get('profit', 0.0)
            profit_status = 'good' if profit > 0 else 'critical' if profit < -100 else 'warning'
            cards.append(
                create_card(
                    "Latest Profit",
                    f"${profit:.2f}",
                    profit_status
                )
            )

        # Best Episode
        if best:
            cards.append(
                create_card(
                    "Best Profit",
                    f"${best.get('profit', 0):.2f}",
                    'good'
                )
            )

            cards.append(
                create_card(
                    "Best Episode",
                    f"#{best.get('episode', 0)}",
                    'normal'
                )
            )

        # Recent Average
        if stats:
            recent_avg = stats.get('recent_avg_profit', 0.0)
            recent_status = 'good' if recent_avg > 0 else 'warning'
            cards.append(
                create_card(
                    "Recent Avg (50)",
                    f"${recent_avg:.2f}",
                    recent_status
                )
            )

            # Sharpe Ratio
            sharpe = stats.get('recent_avg_sharpe', 0.0)
            sharpe_status = 'good' if sharpe > 1.0 else 'warning' if sharpe > 0 else 'critical'
            cards.append(
                create_card(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    sharpe_status
                )
            )

        return cards

    def _create_profit_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create profit/loss chart"""
        episode_nums = [e['episode'] for e in episodes]
        profits = [e['profit'] for e in episodes]

        # Smoothed line
        window = 10
        if len(profits) >= window:
            smoothed = pd.Series(profits).rolling(window=window, min_periods=1).mean()
        else:
            smoothed = profits

        fig = go.Figure()

        # Raw profits
        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=profits,
            mode='markers',
            name='Profit',
            marker=dict(size=4, color=colors['accent'], opacity=0.5)
        ))

        # Smoothed line
        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=smoothed,
            mode='lines',
            name=f'Smoothed ({window})',
            line=dict(color=colors['profit'], width=2)
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color=colors['grid'])

        fig.update_layout(
            title="Profit/Loss per Episode",
            xaxis_title="Episode",
            yaxis_title="Profit ($)",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid']),
            hovermode='x unified'
        )

        return fig

    def _create_loss_chart(self, colors: Dict) -> go.Figure:
        """Create loss chart"""
        model_metrics = self.metrics_logger.model_metrics

        if not model_metrics:
            return self._create_empty_figure(colors, "Model Loss (No data)")

        losses = [m['loss'] for m in model_metrics if m.get('loss') is not None]

        if not losses:
            return self._create_empty_figure(colors, "Model Loss (No data)")

        # Smoothed loss
        window = 20
        if len(losses) >= window:
            smoothed = pd.Series(losses).rolling(window=window, min_periods=1).mean()
        else:
            smoothed = losses

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=losses,
            mode='markers',
            name='Loss',
            marker=dict(size=3, color=colors['loss'], opacity=0.3)
        ))

        fig.add_trace(go.Scatter(
            y=smoothed,
            mode='lines',
            name=f'Smoothed ({window})',
            line=dict(color=colors['accent'], width=2)
        ))

        fig.update_layout(
            title="Training Loss",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid'], type='log'),
            hovermode='x unified'
        )

        return fig

    def _create_epsilon_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create epsilon decay chart"""
        episode_nums = [e['episode'] for e in episodes]
        epsilons = [e.get('epsilon', 0) for e in episodes]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=epsilons,
            mode='lines+markers',
            name='Epsilon',
            line=dict(color=colors['accent'], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title="Exploration Rate (Epsilon)",
            xaxis_title="Episode",
            yaxis_title="Epsilon",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid']),
        )

        return fig

    def _create_lr_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create learning rate chart"""
        episode_nums = [e['episode'] for e in episodes]
        lrs = [e.get('learning_rate', 0) for e in episodes]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=lrs,
            mode='lines+markers',
            name='Learning Rate',
            line=dict(color=colors['profit'], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Episode",
            yaxis_title="Learning Rate",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid'], type='log'),
        )

        return fig

    def _create_winrate_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create win rate chart"""
        episode_nums = [e['episode'] for e in episodes]
        win_rates = [e.get('win_rate', 0) * 100 for e in episodes]

        # Rolling average
        window = 20
        if len(win_rates) >= window:
            smoothed = pd.Series(win_rates).rolling(window=window, min_periods=1).mean()
        else:
            smoothed = win_rates

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=smoothed,
            mode='lines',
            name='Win Rate',
            fill='tozeroy',
            line=dict(color=colors['profit'], width=2)
        ))

        # 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color=colors['warning'])

        fig.update_layout(
            title="Win Rate Trend",
            xaxis_title="Episode",
            yaxis_title="Win Rate (%)",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid'], range=[0, 100]),
        )

        return fig

    def _create_sharpe_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create Sharpe ratio chart"""
        episode_nums = [e['episode'] for e in episodes]
        sharpes = [e.get('sharpe', 0) for e in episodes]

        # Rolling average
        window = 10
        if len(sharpes) >= window:
            smoothed = pd.Series(sharpes).rolling(window=window, min_periods=1).mean()
        else:
            smoothed = sharpes

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=smoothed,
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color=colors['accent'], width=2),
            marker=dict(size=4)
        ))

        # Reference lines
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors['profit'], annotation_text="Good")
        fig.add_hline(y=0.0, line_dash="dash", line_color=colors['grid'])

        fig.update_layout(
            title="Sharpe Ratio Evolution",
            xaxis_title="Episode",
            yaxis_title="Sharpe Ratio",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid']),
        )

        return fig

    def _create_drawdown_chart(self, episodes: List[Dict], colors: Dict) -> go.Figure:
        """Create drawdown chart"""
        episode_nums = [e['episode'] for e in episodes]
        drawdowns = [abs(e.get('drawdown', 0)) for e in episodes]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=episode_nums,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=colors['loss'], width=2)
        ))

        fig.update_layout(
            title="Maximum Drawdown",
            xaxis_title="Episode",
            yaxis_title="Drawdown ($)",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid']),
        )

        return fig

    def _create_qvalue_chart(self, colors: Dict) -> go.Figure:
        """Create Q-value distribution chart"""
        model_metrics = self.metrics_logger.model_metrics

        if not model_metrics:
            return self._create_empty_figure(colors, "Q-Values (No data)")

        # Extract Q-value stats
        q_means = []
        q_stds = []

        for m in model_metrics:
            if m.get('q_values'):
                q_means.append(m['q_values'].get('mean', 0))
                q_stds.append(m['q_values'].get('std', 0))

        if not q_means:
            return self._create_empty_figure(colors, "Q-Values (No data)")

        fig = go.Figure()

        # Mean Q-values
        fig.add_trace(go.Scatter(
            y=q_means,
            mode='lines',
            name='Mean Q-Value',
            line=dict(color=colors['accent'], width=2)
        ))

        # Standard deviation band
        upper = np.array(q_means) + np.array(q_stds)
        lower = np.array(q_means) - np.array(q_stds)

        fig.add_trace(go.Scatter(
            y=upper,
            mode='lines',
            name='Std Dev',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            y=lower,
            mode='lines',
            name='Std Dev',
            fill='tonexty',
            fillcolor=f"rgba{tuple(list(bytes.fromhex(colors['accent'][1:])) + [0.2])}",
            line=dict(width=0)
        ))

        fig.update_layout(
            title="Q-Value Distribution",
            xaxis_title="Training Step",
            yaxis_title="Q-Value",
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(gridcolor=colors['grid']),
            yaxis=dict(gridcolor=colors['grid']),
        )

        return fig

    def _create_stats_table(self, colors: Dict, stats: Dict) -> html.Div:
        """Create statistics table"""
        if not stats:
            return html.Div("No statistics available")

        rows = [
            html.Tr([
                html.Th("Metric", style={'padding': '10px', 'textAlign': 'left'}),
                html.Th("Value", style={'padding': '10px', 'textAlign': 'right'})
            ])
        ]

        metrics = [
            ("Total Episodes", stats.get('total_episodes', 0)),
            ("Training Time", f"{stats.get('training_time', 0):.1f}s"),
            ("Best Profit", f"${stats.get('best_profit', 0):.2f}"),
            ("Best Sharpe", f"{stats.get('best_sharpe', 0):.2f}"),
            ("Recent Avg Profit", f"${stats.get('recent_avg_profit', 0):.2f}"),
            ("Avg Log Time", f"{stats.get('avg_log_time_ms', 0):.3f}ms"),
        ]

        for metric, value in metrics:
            rows.append(html.Tr([
                html.Td(metric, style={'padding': '8px'}),
                html.Td(
                    value,
                    style={'padding': '8px', 'textAlign': 'right', 'color': colors['accent']}
                )
            ]))

        return html.Table(
            rows,
            style={
                'width': '100%',
                'backgroundColor': colors['card_bg'],
                'borderRadius': '8px',
                'padding': '10px'
            }
        )

    def _create_empty_figure(self, colors: Dict, title: str = "No Data") -> go.Figure:
        """Create empty placeholder figure"""
        fig = go.Figure()

        fig.update_layout(
            title=title,
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            annotations=[{
                'text': 'Waiting for data...',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20, 'color': colors['text']}
            }]
        )

        return fig

    def start(self, blocking: bool = False) -> None:
        """
        Start dashboard server

        Args:
            blocking: If True, blocks current thread (for standalone use)
        """
        if self._running:
            logger.warning("Dashboard already running")
            return

        self._running = True

        if blocking:
            logger.info(f"Starting dashboard on http://localhost:{self.port}")
            self.app.run_server(debug=False, host='0.0.0.0', port=self.port)
        else:
            # Run in background thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self._server_thread.start()
            logger.info(f"Dashboard running at http://localhost:{self.port}")

    def _run_server(self) -> None:
        """Run server in thread"""
        try:
            self.app.run_server(
                debug=False,
                host='0.0.0.0',
                port=self.port,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop dashboard server"""
        self._running = False
        logger.info("Dashboard stopped")

    def is_running(self) -> bool:
        """Check if dashboard is running"""
        return self._running
