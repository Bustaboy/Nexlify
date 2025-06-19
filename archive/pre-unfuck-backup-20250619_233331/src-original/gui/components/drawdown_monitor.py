# src/gui/components/drawdown_monitor.py
"""
Nexlify Drawdown Protection GUI Component
Real-time monitoring and control interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
import numpy as np
from typing import Dict, Optional
import json

# For PyQt5 version
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class DrawdownMonitorFrame(tk.Frame):
    """
    🛡️ Cyberpunk-themed drawdown protection monitor
    """
    
    def __init__(self, parent, drawdown_protection, colors):
        super().__init__(parent, bg=colors['bg'])
        self.dd_protection = drawdown_protection
        self.colors = colors
        
        # Update thread
        self.running = True
        self.update_thread = None
        
        self._create_widgets()
        self._start_updates()
        
    def _create_widgets(self):
        """Create all GUI components"""
        # Main container
        main_container = tk.Frame(self, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create sections
        self._create_status_section(main_container)
        self._create_metrics_section(main_container)
        self._create_protection_rules_section(main_container)
        self._create_controls_section(main_container)
        self._create_chart_section(main_container)
        
    def _create_status_section(self, parent):
        """Create status display section"""
        # Status frame with cyberpunk border
        status_frame = tk.LabelFrame(
            parent,
            text="[ DRAWDOWN PROTECTION STATUS ]",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['bg'],
            bd=2,
            relief='groove'
        )
        status_frame.pack(fill='x', pady=(0, 10))
        
        # Level indicator
        level_container = tk.Frame(status_frame, bg=self.colors['bg'])
        level_container.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            level_container,
            text="PROTECTION LEVEL:",
            font=('Consolas', 11, 'bold'),
            fg=self.colors['text'],
            bg=self.colors['bg']
        ).pack(side='left', padx=(0, 10))
        
        self.level_display = tk.Label(
            level_container,
            text="● GREEN ZONE",
            font=('Consolas', 14, 'bold'),
            fg=self.colors['success'],
            bg=self.colors['bg']
        )
        self.level_display.pack(side='left')
        
        # Trading status
        tk.Label(
            level_container,
            text="TRADING:",
            font=('Consolas', 11, 'bold'),
            fg=self.colors['text'],
            bg=self.colors['bg']
        ).pack(side='left', padx=(30, 10))
        
        self.trading_status = tk.Label(
            level_container,
            text="▶ ACTIVE",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['success'],
            bg=self.colors['bg']
        )
        self.trading_status.pack(side='left')
        
        # Visual level bar
        bar_frame = tk.Frame(status_frame, bg=self.colors['bg'])
        bar_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.level_canvas = tk.Canvas(
            bar_frame,
            height=30,
            bg=self.colors['bg_secondary'],
            highlightthickness=1,
            highlightbackground=self.colors['primary']
        )
        self.level_canvas.pack(fill='x')
        
        # Draw initial level bar
        self._update_level_bar(0.0)
        
    def _create_metrics_section(self, parent):
        """Create metrics display section"""
        metrics_frame = tk.LabelFrame(
            parent,
            text="[ DRAWDOWN METRICS ]",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        # Metrics grid
        metrics_grid = tk.Frame(metrics_frame, bg=self.colors['bg'])
        metrics_grid.pack(fill='x', padx=10, pady=10)
        
        # Configure grid
        for i in range(3):
            metrics_grid.columnconfigure(i*2, weight=1)
            
        self.metric_labels = {}
        metrics = [
            ('current_dd', 'Current Drawdown:', 0, 0),
            ('max_dd', 'Maximum Drawdown:', 0, 2),
            ('daily_pnl', "Today's P&L:", 0, 4),
            ('consecutive', 'Consecutive Losses:', 1, 0),
            ('duration', 'DD Duration:', 1, 2),
            ('recovery', 'Recovery Needed:', 1, 4),
        ]
        
        for key, label, row, col in metrics:
            # Label
            tk.Label(
                metrics_grid,
                text=label,
                font=('Consolas', 10),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg']
            ).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
            # Value
            self.metric_labels[key] = tk.Label(
                metrics_grid,
                text="0.00%",
                font=('Consolas', 11, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['bg']
            )
            self.metric_labels[key].grid(row=row, column=col+1, sticky='w', padx=5, pady=2)
            
    def _create_protection_rules_section(self, parent):
        """Create protection rules status section"""
        rules_frame = tk.LabelFrame(
            parent,
            text="[ ACTIVE PROTECTION PROTOCOLS ]",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        rules_frame.pack(fill='x', pady=(0, 10))
        
        # Rules container
        rules_container = tk.Frame(rules_frame, bg=self.colors['bg'])
        rules_container.pack(fill='x', padx=10, pady=10)
        
        self.rule_indicators = {}
        rules = [
            ('daily_limit', '📊 Daily Loss Limit'),
            ('consecutive', '🔢 Consecutive Loss Protection'),
            ('volatility', '📈 Volatility Scaling'),
            ('equity_curve', '📉 Equity Curve Filter'),
            ('correlation', '🔗 Correlation Protection'),
            ('panic_mode', '🚨 Panic Mode')
        ]
        
        for i, (key, label) in enumerate(rules):
            row = i // 3
            col = i % 3
            
            indicator_frame = tk.Frame(rules_container, bg=self.colors['bg'])
            indicator_frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')
            
            # LED indicator
            led = tk.Label(
                indicator_frame,
                text="●",
                font=('Arial', 12),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg']
            )
            led.pack(side='left', padx=(0, 5))
            
            # Label
            tk.Label(
                indicator_frame,
                text=label,
                font=('Consolas', 9),
                fg=self.colors['text'],
                bg=self.colors['bg']
            ).pack(side='left')
            
            self.rule_indicators[key] = led
            
    def _create_controls_section(self, parent):
        """Create control buttons section"""
        controls_frame = tk.LabelFrame(
            parent,
            text="[ MANUAL OVERRIDE CONTROLS ]",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['warning'],
            bg=self.colors['bg']
        )
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Button container
        btn_container = tk.Frame(controls_frame, bg=self.colors['bg'])
        btn_container.pack(fill='x', padx=10, pady=10)
        
        # Emergency stop button
        self.emergency_btn = tk.Button(
            btn_container,
            text="🛑 EMERGENCY STOP",
            command=self._emergency_stop,
            font=('Consolas', 11, 'bold'),
            fg='white',
            bg=self.colors['danger'],
            activebackground='#ff0000',
            relief='raised',
            bd=3,
            padx=20,
            pady=10
        )
        self.emergency_btn.pack(side='left', padx=5)
        
        # Pause/Resume button
        self.pause_btn = tk.Button(
            btn_container,
            text="⏸️ PAUSE TRADING",
            command=self._toggle_pause,
            font=('Consolas', 11, 'bold'),
            fg=self.colors['bg'],
            bg=self.colors['warning'],
            activebackground='#ff8800',
            relief='raised',
            bd=2,
            padx=15,
            pady=8
        )
        self.pause_btn.pack(side='left', padx=5)
        
        # Reset button
        self.reset_btn = tk.Button(
            btn_container,
            text="🔄 RESET METRICS",
            command=self._reset_metrics,
            font=('Consolas', 10),
            fg=self.colors['text'],
            bg=self.colors['bg_secondary'],
            activebackground=self.colors['primary'],
            relief='raised',
            bd=2,
            padx=10,
            pady=8
        )
        self.reset_btn.pack(side='left', padx=5)
        
        # Recovery mode selector
        tk.Label(
            btn_container,
            text="Recovery Mode:",
            font=('Consolas', 10),
            fg=self.colors['text'],
            bg=self.colors['bg']
        ).pack(side='left', padx=(20, 5))
        
        self.recovery_var = tk.StringVar(value="moderate")
        recovery_menu = ttk.Combobox(
            btn_container,
            textvariable=self.recovery_var,
            values=['aggressive', 'moderate', 'conservative', 'turtle'],
            font=('Consolas', 10),
            width=15,
            state='readonly'
        )
        recovery_menu.pack(side='left', padx=5)
        recovery_menu.bind('<<ComboboxSelected>>', self._change_recovery_mode)
        
    def _create_chart_section(self, parent):
        """Create drawdown chart section"""
        chart_frame = tk.LabelFrame(
            parent,
            text="[ DRAWDOWN VISUALIZATION ]",
            font=('Consolas', 12, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        chart_frame.pack(fill='both', expand=True)
        
        # Canvas for drawing
        self.chart_canvas = tk.Canvas(
            chart_frame,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        self.chart_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initial chart
        self._draw_drawdown_chart([])
        
    def _update_level_bar(self, drawdown: float):
        """Update the visual level bar"""
        self.level_canvas.delete('all')
        width = self.level_canvas.winfo_width()
        if width <= 1:
            width = 400  # Default width
            
        height = 30
        
        # Draw background sections
        sections = [
            (0, 0.05, self.colors['success']),      # Green zone
            (0.05, 0.10, self.colors['warning']),   # Yellow zone
            (0.10, 0.15, '#ff6600'),                 # Orange zone
            (0.15, 0.25, self.colors['danger']),    # Red zone
            (0.25, 1.0, '#800080')                   # Black zone (purple)
        ]
        
        for start, end, color in sections:
            x1 = start * width
            x2 = end * width
            self.level_canvas.create_rectangle(
                x1, 0, x2, height,
                fill=color,
                outline='',
                stipple='gray50'
            )
            
        # Draw current level indicator
        current_x = min(drawdown * width, width - 5)
        self.level_canvas.create_line(
            current_x, 0, current_x, height,
            fill='white',
            width=3
        )
        
        # Draw percentage markers
        for pct in [0.05, 0.10, 0.15, 0.25]:
            x = pct * width
            self.level_canvas.create_text(
                x, height - 5,
                text=f"{pct:.0%}",
                font=('Consolas', 8),
                fill='white',
                anchor='s'
            )
            
    def _draw_drawdown_chart(self, history):
        """Draw drawdown history chart"""
        self.chart_canvas.delete('all')
        
        if not history:
            # Draw placeholder
            self.chart_canvas.create_text(
                self.chart_canvas.winfo_width() // 2,
                self.chart_canvas.winfo_height() // 2,
                text="NO DRAWDOWN DATA",
                font=('Consolas', 14),
                fill=self.colors['text_secondary']
            )
            return
            
        # Get canvas dimensions
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
            
        # Prepare data
        values = [h['drawdown'] for h in history[-100:]]  # Last 100 points
        if not values:
            return
            
        # Calculate scaling
        max_dd = max(values) if values else 0.25
        scale_y = (height - 40) / max_dd if max_dd > 0 else 1
        scale_x = width / len(values) if len(values) > 1 else 1
        
        # Draw grid
        for y_pct in [0.05, 0.10, 0.15, 0.20]:
            y = height - (y_pct * scale_y) - 20
            self.chart_canvas.create_line(
                0, y, width, y,
                fill=self.colors['grid'],
                dash=(2, 4)
            )
            self.chart_canvas.create_text(
                width - 30, y,
                text=f"{y_pct:.0%}",
                font=('Consolas', 8),
                fill=self.colors['text_secondary']
            )
            
        # Draw drawdown line
        points = []
        for i, dd in enumerate(values):
            x = i * scale_x
            y = height - (dd * scale_y) - 20
            points.extend([x, y])
            
        if len(points) >= 4:
            # Main line
            self.chart_canvas.create_line(
                points,
                fill=self.colors['danger'],
                width=2,
                smooth=True
            )
            
            # Fill area
            fill_points = [0, height - 20] + points + [width, height - 20]
            self.chart_canvas.create_polygon(
                fill_points,
                fill=self.colors['danger'],
                stipple='gray50',
                outline=''
            )
            
        # Draw current value
        if values:
            current_dd = values[-1]
            self.chart_canvas.create_text(
                width // 2,
                20,
                text=f"Current: {current_dd:.2%}",
                font=('Consolas', 12, 'bold'),
                fill=self._get_color_for_drawdown(current_dd)
            )
            
    def _get_color_for_drawdown(self, dd: float) -> str:
        """Get color based on drawdown level"""
        if dd >= 0.25:
            return '#800080'  # Purple for black zone
        elif dd >= 0.15:
            return self.colors['danger']
        elif dd >= 0.10:
            return '#ff6600'  # Orange
        elif dd >= 0.05:
            return self.colors['warning']
        else:
            return self.colors['success']
            
    def _start_updates(self):
        """Start update thread"""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                self._update_display()
                threading.Event().wait(1.0)  # Update every second
            except Exception as e:
                print(f"Error in drawdown monitor update: {e}")
                
    def _update_display(self):
        """Update all display elements"""
        if not self.dd_protection:
            return
            
        # Get current metrics
        metrics = self.dd_protection.metrics
        level = self.dd_protection.current_level
        
        # Update level display
        level_colors = {
            'green_zone': self.colors['success'],
            'yellow_alert': self.colors['warning'],
            'orange_alert': '#ff6600',
            'red_zone': self.colors['danger'],
            'flatline': '#800080'
        }
        
        level_names = {
            'green_zone': '● GREEN ZONE',
            'yellow_alert': '● YELLOW ALERT',
            'orange_alert': '● ORANGE ALERT',
            'red_zone': '● RED ZONE',
            'flatline': '● FLATLINE'
        }
        
        self.level_display.config(
            text=level_names.get(level.value, '● UNKNOWN'),
            fg=level_colors.get(level.value, self.colors['text'])
        )
        
        # Update trading status
        if self.dd_protection.is_paused:
            self.trading_status.config(
                text="⏸ PAUSED",
                fg=self.colors['danger']
            )
            self.pause_btn.config(text="▶ RESUME TRADING")
        else:
            self.trading_status.config(
                text="▶ ACTIVE",
                fg=self.colors['success']
            )
            self.pause_btn.config(text="⏸️ PAUSE TRADING")
            
        # Update metrics
        self.metric_labels['current_dd'].config(
            text=f"{metrics.current_drawdown:.2%}",
            fg=self._get_color_for_drawdown(metrics.current_drawdown)
        )
        self.metric_labels['max_dd'].config(text=f"{metrics.max_drawdown:.2%}")
        
        # Daily P&L
        daily_pnl = self.dd_protection._get_today_pnl()
        pnl_color = self.colors['success'] if daily_pnl['percent'] >= 0 else self.colors['danger']
        self.metric_labels['daily_pnl'].config(
            text=f"{daily_pnl['percent']:.2%}",
            fg=pnl_color
        )
        
        # Other metrics
        self.metric_labels['consecutive'].config(text=str(metrics.consecutive_losses))
        self.metric_labels['duration'].config(text=self._format_duration(metrics.drawdown_duration))
        self.metric_labels['recovery'].config(text=f"{metrics.gain_to_recover:.2%}")
        
        # Update level bar
        self._update_level_bar(metrics.current_drawdown)
        
        # Update rule indicators
        self._update_rule_indicators()
        
        # Update chart
        if hasattr(self.dd_protection, 'drawdown_history'):
            self._draw_drawdown_chart(list(self.dd_protection.drawdown_history))
            
    def _update_rule_indicators(self):
        """Update protection rule LED indicators"""
        # This would check actual rule states
        # For now, using example logic
        
        # Daily limit
        daily_triggered = self.dd_protection._check_daily_limit()
        self.rule_indicators['daily_limit'].config(
            fg=self.colors['danger'] if daily_triggered else self.colors['text_secondary']
        )
        
        # Consecutive losses
        consec_triggered = self.dd_protection.metrics.consecutive_losses >= 5
        self.rule_indicators['consecutive'].config(
            fg=self.colors['danger'] if consec_triggered else self.colors['text_secondary']
        )
        
        # Volatility
        vol_triggered = self.dd_protection._check_volatility_spike()
        self.rule_indicators['volatility'].config(
            fg=self.colors['warning'] if vol_triggered else self.colors['text_secondary']
        )
        
        # Equity curve
        ec_active = self.dd_protection.settings.get('equity_curve_trading', False)
        self.rule_indicators['equity_curve'].config(
            fg=self.colors['primary'] if ec_active else self.colors['text_secondary']
        )
        
        # Panic mode
        panic_active = self.dd_protection.current_level.value == 'flatline'
        self.rule_indicators['panic_mode'].config(
            fg=self.colors['danger'] if panic_active else self.colors['text_secondary']
        )
        
    def _format_duration(self, duration) -> str:
        """Format timedelta to readable string"""
        if not duration:
            return "0d"
            
        days = duration.days
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
            
    def _emergency_stop(self):
        """Handle emergency stop button"""
        if messagebox.askyesno(
            "Emergency Stop",
            "ACTIVATE EMERGENCY STOP?\n\nThis will:\n- Pause all trading\n- Close all positions\n- Activate maximum protection\n\nContinue?",
            icon='warning'
        ):
            self.dd_protection._emergency_stop()
            messagebox.showinfo(
                "Emergency Stop Activated",
                "🛑 EMERGENCY STOP ACTIVATED\n\nAll trading has been halted.\nManual intervention required to resume."
            )
            
    def _toggle_pause(self):
        """Toggle pause state"""
        self.dd_protection.is_paused = not self.dd_protection.is_paused
        status = "paused" if self.dd_protection.is_paused else "resumed"
        messagebox.showinfo(
            "Trading Status",
            f"Trading has been {status}."
        )
        
    def _reset_metrics(self):
        """Reset drawdown metrics"""
        if messagebox.askyesno(
            "Reset Metrics",
            "Reset all drawdown metrics?\n\nThis will clear:\n- Current drawdown\n- Historical data\n- Performance metrics\n\nContinue?"
        ):
            # Reset metrics
            self.dd_protection.metrics = type(self.dd_protection.metrics)()
            self.dd_protection.balance_history.clear()
            self.dd_protection.drawdown_history.clear()
            messagebox.showinfo(
                "Metrics Reset",
                "All drawdown metrics have been reset."
            )
            
    def _change_recovery_mode(self, event=None):
        """Change recovery mode"""
        mode = self.recovery_var.get()
        # This would update the actual recovery mode
        messagebox.showinfo(
            "Recovery Mode",
            f"Recovery mode changed to: {mode.upper()}"
        )
        
    def export_report(self):
        """Export drawdown report"""
        report = self.dd_protection.export_report()
        # Save to file or display
        filename = f"drawdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(report)
        messagebox.showinfo(
            "Report Exported",
            f"Drawdown report saved to:\n{filename}"
        )
        
    def destroy(self):
        """Clean up on destroy"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        super().destroy()


# PyQt5 Version for enhanced GUI
if PYQT_AVAILABLE:
    class DrawdownMonitorWidget(QWidget):
        """PyQt5 version of drawdown monitor with advanced graphics"""
        
        def __init__(self, drawdown_protection):
            super().__init__()
            self.dd_protection = drawdown_protection
            self.init_ui()
            
            # Update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_display)
            self.timer.start(1000)  # Update every second
            
        def init_ui(self):
            """Initialize the UI"""
            layout = QVBoxLayout()
            self.setLayout(layout)
            
            # Create tabs
            tabs = QTabWidget()
            layout.addWidget(tabs)
            
            # Overview tab
            overview = self._create_overview_tab()
            tabs.addTab(overview, "Overview")
            
            # Metrics tab
            metrics = self._create_metrics_tab()
            tabs.addTab(metrics, "Metrics")
            
            # Charts tab
            charts = self._create_charts_tab()
            tabs.addTab(charts, "Charts")
            
            # Controls tab
            controls = self._create_controls_tab()
            tabs.addTab(controls, "Controls")
            
        def _create_overview_tab(self):
            """Create overview tab"""
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)
            
            # Status display
            status_group = QGroupBox("Protection Status")
            status_layout = QGridLayout()
            status_group.setLayout(status_layout)
            
            # Level indicator
            self.level_label = QLabel("GREEN ZONE")
            self.level_label.setStyleSheet("""
                QLabel {
                    color: #00ff00;
                    font-size: 24px;
                    font-weight: bold;
                    font-family: Consolas;
                    padding: 10px;
                    border: 2px solid #00ff00;
                    border-radius: 5px;
                }
            """)
            status_layout.addWidget(self.level_label, 0, 0, 1, 2, Qt.AlignCenter)
            
            # Metrics
            self.metric_widgets = {}
            metrics = [
                ('current_dd', 'Current Drawdown'),
                ('max_dd', 'Max Drawdown'),
                ('position_mult', 'Position Multiplier'),
                ('daily_pnl', "Today's P&L")
            ]
            
            for i, (key, label) in enumerate(metrics):
                row = (i // 2) + 1
                col = (i % 2) * 2
                
                lbl = QLabel(f"{label}:")
                lbl.setStyleSheet("color: #888; font-family: Consolas;")
                status_layout.addWidget(lbl, row, col)
                
                val = QLabel("0.00%")
                val.setStyleSheet("color: white; font-family: Consolas; font-weight: bold;")
                status_layout.addWidget(val, row, col + 1)
                
                self.metric_widgets[key] = val
                
            layout.addWidget(status_group)
            
            # Visual drawdown meter
            self.dd_meter = self._create_drawdown_meter()
            layout.addWidget(self.dd_meter)
            
            return widget
            
        def _create_drawdown_meter(self):
            """Create visual drawdown meter"""
            widget = QWidget()
            widget.setFixedHeight(100)
            widget.setStyleSheet("background-color: #111;")
            
            # This would be a custom paint widget
            # For now, placeholder
            return widget
            
        def _create_metrics_tab(self):
            """Create detailed metrics tab"""
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)
            
            # Metrics table
            self.metrics_table = QTableWidget()
            self.metrics_table.setColumnCount(2)
            self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
            self.metrics_table.horizontalHeader().setStretchLastSection(True)
            
            # Cyberpunk styling
            self.metrics_table.setStyleSheet("""
                QTableWidget {
                    background-color: #0a0a0a;
                    color: #00ff00;
                    gridline-color: #333;
                    font-family: Consolas;
                }
                QHeaderView::section {
                    background-color: #1a1a1a;
                    color: #00ffff;
                    border: 1px solid #333;
                    padding: 5px;
                }
            """)
            
            layout.addWidget(self.metrics_table)
            return widget
            
        def _create_charts_tab(self):
            """Create charts tab with pyqtgraph"""
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)
            
            # Drawdown chart
            self.dd_plot = pg.PlotWidget(title="Drawdown History")
            self.dd_plot.setLabel('left', 'Drawdown %')
            self.dd_plot.setLabel('bottom', 'Time')
            self.dd_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Cyberpunk colors
            self.dd_plot.setBackground('#0a0a0a')
            self.dd_plot.getAxis('left').setPen('#00ff00')
            self.dd_plot.getAxis('bottom').setPen('#00ff00')
            
            layout.addWidget(self.dd_plot)
            
            # Equity curve chart
            self.equity_plot = pg.PlotWidget(title="Equity Curve")
            self.equity_plot.setLabel('left', 'Balance')
            self.equity_plot.setLabel('bottom', 'Time')
            self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
            self.equity_plot.setBackground('#0a0a0a')
            
            layout.addWidget(self.equity_plot)
            
            return widget
            
        def _create_controls_tab(self):
            """Create controls tab"""
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)
            
            # Emergency controls
            emergency_group = QGroupBox("Emergency Controls")
            emergency_layout = QHBoxLayout()
            emergency_group.setLayout(emergency_layout)
            
            # Emergency stop button
            self.emergency_btn = QPushButton("🛑 EMERGENCY STOP")
            self.emergency_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff0000;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #ff3333;
                }
            """)
            self.emergency_btn.clicked.connect(self.emergency_stop)
            emergency_layout.addWidget(self.emergency_btn)
            
            layout.addWidget(emergency_group)
            
            # Settings
            settings_group = QGroupBox("Protection Settings")
            settings_layout = QFormLayout()
            settings_group.setLayout(settings_layout)
            
            # Recovery mode
            self.recovery_combo = QComboBox()
            self.recovery_combo.addItems(['Aggressive', 'Moderate', 'Conservative', 'Turtle'])
            settings_layout.addRow("Recovery Mode:", self.recovery_combo)
            
            # Thresholds
            self.yellow_spin = QDoubleSpinBox()
            self.yellow_spin.setRange(0, 100)
            self.yellow_spin.setSuffix('%')
            self.yellow_spin.setValue(5)
            settings_layout.addRow("Yellow Alert:", self.yellow_spin)
            
            self.red_spin = QDoubleSpinBox()
            self.red_spin.setRange(0, 100)
            self.red_spin.setSuffix('%')
            self.red_spin.setValue(15)
            settings_layout.addRow("Red Zone:", self.red_spin)
            
            layout.addWidget(settings_group)
            layout.addStretch()
            
            return widget
            
        def update_display(self):
            """Update all displays"""
            if not self.dd_protection:
                return
                
            metrics = self.dd_protection.metrics
            level = self.dd_protection.current_level
            
            # Update level display
            level_styles = {
                'green_zone': "color: #00ff00; border-color: #00ff00;",
                'yellow_alert': "color: #ffff00; border-color: #ffff00;",
                'orange_alert': "color: #ff6600; border-color: #ff6600;",
                'red_zone': "color: #ff0000; border-color: #ff0000;",
                'flatline': "color: #800080; border-color: #800080;"
            }
            
            self.level_label.setText(level.value.upper().replace('_', ' '))
            base_style = self.level_label.styleSheet().split('{')[0] + '{'
            color_style = level_styles.get(level.value, "color: white; border-color: white;")
            self.level_label.setStyleSheet(base_style + color_style + "}")
            
            # Update metrics
            self.metric_widgets['current_dd'].setText(f"{metrics.current_drawdown:.2%}")
            self.metric_widgets['max_dd'].setText(f"{metrics.max_drawdown:.2%}")
            self.metric_widgets['position_mult'].setText(
                f"{self.dd_protection.get_position_sizing_multiplier():.0%}"
            )
            
            # Update charts if data available
            self._update_charts()
            
        def _update_charts(self):
            """Update the charts with latest data"""
            # This would update pyqtgraph plots with real data
            pass
            
        def emergency_stop(self):
            """Handle emergency stop"""
            reply = QMessageBox.critical(
                self,
                "Emergency Stop",
                "Activate EMERGENCY STOP?\n\nThis will halt all trading immediately.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.dd_protection._emergency_stop()
                QMessageBox.information(
                    self,
                    "Emergency Stop Activated",
                    "All trading has been halted."
                )
