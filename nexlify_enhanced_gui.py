"""
Nexlify Enhanced - Main GUI Integration
Comprehensive cyberpunk-themed trading interface with all features
"""

import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all feature components
from gui.components.dashboard import AdvancedDashboard
from gui.components.gamification import GamificationEngine
from gui.components.ai_companion import AITradingCompanion
from gui.components.cyberpunk_effects import (
    SoundEffectsManager, CyberpunkAnimator, create_cyberpunk_theme,
    CyberpunkButton, TerminalText, NeuralNetworkVisualizer
)
from src.core.engine import TradingEngine
from src.risk.drawdown import DrawdownProtection
from src.analytics.performance import PerformanceAnalytics
from src.analytics.tax_optimizer import TaxOptimizer
from src.analytics.backtesting import AdvancedBacktestEngine
from src.analytics.audit_trail import AuditManager
from src.security.two_factor import SecurityManager
from src.ml.predictive import PredictiveEngine
from src.strategies.multi_strategy import MultiStrategyOptimizer

logger = logging.getLogger(__name__)

class NexlifyEnhancedGUI:
    """
    Main Nexlify GUI with all enhanced features integrated
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌃 Nexlify Trading Matrix - Arasaka Neural Net v3.0")
        self.root.geometry("1600x900")
        
        # Apply cyberpunk theme
        self.style = create_cyberpunk_theme()
        self.root.configure(bg='#0a0a0a')
        
        # Initialize sound manager
        self.sound_manager = SoundEffectsManager(enabled=True)
        
        # Initialize animator
        self.animator = CyberpunkAnimator(self.root)
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize core components
        self.init_core_components()
        
        # Initialize feature engines
        self.init_feature_engines()
        
        # Security check
        self.authenticated = False
        self.current_user = None
        
        # Build interface
        self.build_interface()
        
        # Play startup sound
        self.sound_manager.play('startup')
        
        # Start with login screen
        self.show_login_screen()
        
    def init_core_components(self):
        """Initialize core trading components"""
        # Trading engine
        self.trading_engine = TradingEngine(self.config)
        
        # Multi-strategy optimizer
        self.strategy_optimizer = MultiStrategyOptimizer()
        
        # Security manager
        self.security_manager = SecurityManager(self.config)
        
        # Audit manager
        self.audit_manager = AuditManager(self.config)
        
    def init_feature_engines(self):
        """Initialize all feature engines"""
        # Risk management
        self.drawdown_protection = DrawdownProtection(self.config.get('risk', {}))
        
        # Analytics
        self.performance_analytics = PerformanceAnalytics()
        self.tax_optimizer = TaxOptimizer(self.config.get('tax', {}))
        
        # Predictive engine
        self.predictive_engine = PredictiveEngine(self.config.get('ml', {}))
        
        # Gamification
        self.gamification = GamificationEngine()
        
    def build_interface(self):
        """Build the main interface"""
        # Header with Neural Network visualization
        self.create_header()
        
        # Main notebook with all features
        self.create_main_notebook()
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self):
        """Create cyberpunk header with neural network"""
        header_frame = tk.Frame(self.root, bg='#0a0a0a', height=100)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(header_frame, bg='#0a0a0a')
        title_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            title_frame,
            text="NEXLIFY",
            font=('Consolas', 32, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a'
        ).pack()
        
        tk.Label(
            title_frame,
            text="ARASAKA NEURAL-NET TRADING MATRIX",
            font=('Consolas', 12),
            fg='#00ffff',
            bg='#0a0a0a'
        ).pack()
        
        # Neural network visualization
        self.neural_viz = NeuralNetworkVisualizer(
            header_frame,
            width=300,
            height=80
        )
        self.neural_viz.pack(side=tk.RIGHT, padx=20)
        
        # User info and controls
        control_frame = tk.Frame(header_frame, bg='#0a0a0a')
        control_frame.pack(side=tk.RIGHT, padx=20)
        
        self.user_label = tk.Label(
            control_frame,
            text="User: Not Connected",
            font=('Consolas', 10),
            fg='#00ff00',
            bg='#0a0a0a'
        )
        self.user_label.pack()
        
        # Quick controls
        CyberpunkButton(
            control_frame,
            text="🚨 KILL SWITCH",
            bg='#ff0000',
            fg='#ffffff',
            command=self.emergency_stop,
            sound_manager=self.sound_manager
        ).pack(pady=5)
        
    def create_main_notebook(self):
        """Create main tabbed interface with all features"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook
        self.notebook = ttk.Notebook(main_frame, style='Cyberpunk.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs for all features
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_strategies_tab()
        self.create_risk_tab()
        self.create_analytics_tab()
        self.create_ai_companion_tab()
        self.create_achievements_tab()
        self.create_settings_tab()
        self.create_security_tab()
        self.create_audit_tab()
        
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
    def create_dashboard_tab(self):
        """Create advanced dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text='📊 Dashboard')
        
        # Use advanced dashboard component
        self.dashboard = AdvancedDashboard(dashboard_frame)
        
        # Add real-time stats
        stats_frame = tk.Frame(dashboard_frame, bg='#0a0a0a')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create stat displays
        self.stat_displays = {}
        stats = [
            ('Portfolio Value', '$50,000', '#00ff00'),
            ('Daily P&L', '+$1,250 (+2.5%)', '#00ff00'),
            ('Active Positions', '5', '#00ffff'),
            ('Win Rate', '73%', '#00ff00'),
            ('Neural Confidence', '87%', '#ffff00')
        ]
        
        for i, (label, value, color) in enumerate(stats):
            stat_frame = tk.Frame(stats_frame, bg='#1a1a1a', relief='ridge', bd=2)
            stat_frame.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            tk.Label(
                stat_frame,
                text=label,
                font=('Consolas', 9),
                fg='#888888',
                bg='#1a1a1a'
            ).pack()
            
            self.stat_displays[label] = tk.Label(
                stat_frame,
                text=value,
                font=('Consolas', 14, 'bold'),
                fg=color,
                bg='#1a1a1a'
            )
            self.stat_displays[label].pack(pady=5)
            
            # Add pulse animation
            self.animator.create_pulse_effect(stat_frame, color)
            
    def create_trading_tab(self):
        """Create trading interface tab"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text='💹 Trading Matrix')
        
        # Active positions
        positions_frame = tk.LabelFrame(
            trading_frame,
            text="ACTIVE POSITIONS",
            font=('Consolas', 12, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a',
            relief='ridge',
            bd=2
        )
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Positions treeview with cyberpunk styling
        columns = ('Symbol', 'Side', 'Size', 'Entry', 'Current', 'P&L', 'P&L %', 'Confidence')
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show='headings')
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
            
        self.positions_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add sample positions
        positions = [
            ('BTC/USDT', 'LONG', '0.5', '$45,000', '$46,500', '+$750', '+3.33%', '92%'),
            ('ETH/USDT', 'LONG', '5.0', '$3,200', '$3,350', '+$750', '+4.69%', '88%'),
            ('SOL/USDT', 'SHORT', '100', '$150', '$145', '+$500', '+3.33%', '85%')
        ]
        
        for pos in positions:
            self.positions_tree.insert('', 'end', values=pos)
            
        # Trading controls
        controls_frame = tk.Frame(trading_frame, bg='#0a0a0a')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # One-click presets
        presets_frame = tk.LabelFrame(
            controls_frame,
            text="ONE-CLICK PRESETS",
            font=('Consolas', 10, 'bold'),
            fg='#00ffff',
            bg='#0a0a0a'
        )
        presets_frame.pack(side=tk.LEFT, padx=5)
        
        presets = [
            ('🛡️ Conservative', 'conservative'),
            ('⚔️ Aggressive', 'aggressive'),
            ('🐻 Bear Market', 'bear'),
            ('🚀 Degen Mode', 'degen')
        ]
        
        for label, preset in presets:
            CyberpunkButton(
                presets_frame,
                text=label,
                command=lambda p=preset: self.apply_preset(p),
                sound_manager=self.sound_manager
            ).pack(side=tk.LEFT, padx=5, pady=5)
            
    def create_risk_tab(self):
        """Create risk management tab"""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text='🛡️ Risk Matrix')
        
        # Drawdown protection display
        drawdown_frame = tk.LabelFrame(
            risk_frame,
            text="DRAWDOWN PROTECTION",
            font=('Consolas', 12, 'bold'),
            fg='#ff6600',
            bg='#0a0a0a'
        )
        drawdown_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Drawdown metrics
        metrics = self.drawdown_protection.calculate_recovery_metrics()
        
        for key, value in metrics.items():
            metric_frame = tk.Frame(drawdown_frame, bg='#0a0a0a')
            metric_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(
                metric_frame,
                text=f"{key.replace('_', ' ').title()}:",
                font=('Consolas', 10),
                fg='#888888',
                bg='#0a0a0a',
                width=20,
                anchor='w'
            ).pack(side=tk.LEFT)
            
            tk.Label(
                metric_frame,
                text=str(value),
                font=('Consolas', 10, 'bold'),
                fg='#00ff00' if 'current' not in key else '#ff6600',
                bg='#0a0a0a'
            ).pack(side=tk.LEFT)
            
    def create_ai_companion_tab(self):
        """Create AI companion tab"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text='🤖 AI Companion')
        
        # Create AI companion
        self.ai_companion = AITradingCompanion(
            ai_frame,
            self.trading_engine,
            self.config
        )
        
    def create_achievements_tab(self):
        """Create gamification/achievements tab"""
        achievements_frame = ttk.Frame(self.notebook)
        self.notebook.add(achievements_frame, text='🏆 Achievements')
        
        # User level and XP
        level_frame = tk.Frame(achievements_frame, bg='#0a0a0a')
        level_frame.pack(fill=tk.X, padx=10, pady=10)
        
        user_level, user_title = self.gamification.get_user_level('current_user')
        user_xp = self.gamification.get_user_xp('current_user')
        
        tk.Label(
            level_frame,
            text=f"Level {user_level}: {user_title}",
            font=('Consolas', 20, 'bold'),
            fg='#00ffff',
            bg='#0a0a0a'
        ).pack()
        
        tk.Label(
            level_frame,
            text=f"XP: {user_xp}",
            font=('Consolas', 14),
            fg='#00ff00',
            bg='#0a0a0a'
        ).pack()
        
        # Achievement grid
        achievements_container = tk.Frame(achievements_frame, bg='#0a0a0a')
        achievements_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create achievement displays
        row = 0
        col = 0
        
        for achievement_id, achievement in self.gamification.achievements.items():
            ach_frame = tk.Frame(
                achievements_container,
                bg='#1a1a1a',
                relief='ridge',
                bd=2,
                width=200,
                height=100
            )
            ach_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Achievement icon
            tk.Label(
                ach_frame,
                text=achievement['badge'],
                font=('Arial', 24),
                bg='#1a1a1a'
            ).pack(pady=5)
            
            # Achievement name
            tk.Label(
                ach_frame,
                text=achievement['name'],
                font=('Consolas', 10, 'bold'),
                fg='#00ff00',
                bg='#1a1a1a'
            ).pack()
            
            # Achievement description
            tk.Label(
                ach_frame,
                text=achievement['description'],
                font=('Consolas', 8),
                fg='#888888',
                bg='#1a1a1a',
                wraplength=180
            ).pack()
            
            # XP reward
            tk.Label(
                ach_frame,
                text=f"+{achievement['xp']} XP",
                font=('Consolas', 9),
                fg='#ffff00',
                bg='#1a1a1a'
            ).pack()
            
            col += 1
            if col > 3:
                col = 0
                row += 1
                
    def create_security_tab(self):
        """Create security settings tab"""
        security_frame = ttk.Frame(self.notebook)
        self.notebook.add(security_frame, text='🔐 Security')
        
        # 2FA settings
        twofa_frame = tk.LabelFrame(
            security_frame,
            text="TWO-FACTOR AUTHENTICATION",
            font=('Consolas', 12, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a'
        )
        twofa_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 2FA status
        twofa_status = tk.Frame(twofa_frame, bg='#0a0a0a')
        twofa_status.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            twofa_status,
            text="2FA Status:",
            font=('Consolas', 10),
            fg='#888888',
            bg='#0a0a0a'
        ).pack(side=tk.LEFT)
        
        self.twofa_status_label = tk.Label(
            twofa_status,
            text="ENABLED",
            font=('Consolas', 10, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a'
        )
        self.twofa_status_label.pack(side=tk.LEFT, padx=10)
        
        # Security summary
        summary = self.security_manager.get_security_summary()
        
        summary_frame = tk.LabelFrame(
            security_frame,
            text="SECURITY SUMMARY",
            font=('Consolas', 12, 'bold'),
            fg='#00ffff',
            bg='#0a0a0a'
        )
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for key, value in summary.items():
            if isinstance(value, dict):
                continue
                
            item_frame = tk.Frame(summary_frame, bg='#0a0a0a')
            item_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(
                item_frame,
                text=f"{key.replace('_', ' ').title()}:",
                font=('Consolas', 10),
                fg='#888888',
                bg='#0a0a0a',
                width=20,
                anchor='w'
            ).pack(side=tk.LEFT)
            
            tk.Label(
                item_frame,
                text=str(value),
                font=('Consolas', 10, 'bold'),
                fg='#00ff00',
                bg='#0a0a0a'
            ).pack(side=tk.LEFT)
            
    def create_audit_tab(self):
        """Create audit trail tab"""
        audit_frame = ttk.Frame(self.notebook)
        self.notebook.add(audit_frame, text='📜 Audit Trail')
        
        # Audit status
        status_frame = tk.Frame(audit_frame, bg='#0a0a0a')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        audit_summary = self.audit_manager.get_audit_summary()
        
        tk.Label(
            status_frame,
            text=f"Blockchain Integrity: {audit_summary['integrity_status']}",
            font=('Consolas', 12, 'bold'),
            fg='#00ff00' if audit_summary['integrity_status'] == 'Valid' else '#ff0000',
            bg='#0a0a0a'
        ).pack()
        
        # Recent audit entries
        entries_frame = tk.LabelFrame(
            audit_frame,
            text="RECENT AUDIT ENTRIES",
            font=('Consolas', 12, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a'
        )
        entries_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Audit log display
        self.audit_log = TerminalText(entries_frame, height=20)
        self.audit_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Type sample entries
        self.audit_log.type_text("[2024-01-15 10:30:45] System startup initiated\n", 'system')
        self.audit_log.type_text("[2024-01-15 10:30:47] User authentication successful\n", 'success')
        self.audit_log.type_text("[2024-01-15 10:31:02] Trade executed: BTC/USDT LONG 0.5\n", 'data')
        
    def show_login_screen(self):
        """Show login screen with 2FA"""
        login_window = tk.Toplevel(self.root)
        login_window.title("🔐 Nexlify Neural-Net Access")
        login_window.geometry("400x500")
        login_window.configure(bg='#0a0a0a')
        login_window.transient(self.root)
        login_window.grab_set()
        
        # Center window
        login_window.update_idletasks()
        x = (login_window.winfo_screenwidth() // 2) - (login_window.winfo_width() // 2)
        y = (login_window.winfo_screenheight() // 2) - (login_window.winfo_height() // 2)
        login_window.geometry(f"+{x}+{y}")
        
        # Logo
        tk.Label(
            login_window,
            text="NEXLIFY",
            font=('Consolas', 24, 'bold'),
            fg='#00ff00',
            bg='#0a0a0a'
        ).pack(pady=20)
        
        tk.Label(
            login_window,
            text="NEURAL-NET ACCESS PORTAL",
            font=('Consolas', 12),
            fg='#00ffff',
            bg='#0a0a0a'
        ).pack()
        
        # Login form
        form_frame = tk.Frame(login_window, bg='#0a0a0a')
        form_frame.pack(pady=30)
        
        # Username
        tk.Label(
            form_frame,
            text="Username:",
            font=('Consolas', 10),
            fg='#00ff00',
            bg='#0a0a0a'
        ).grid(row=0, column=0, sticky='w', pady=5)
        
        self.username_entry = ttk.Entry(form_frame, font=('Consolas', 10), width=20)
        self.username_entry.grid(row=0, column=1, pady=5)
        
        # Password
        tk.Label(
            form_frame,
            text="Password:",
            font=('Consolas', 10),
            fg='#00ff00',
            bg='#0a0a0a'
        ).grid(row=1, column=0, sticky='w', pady=5)
        
        self.password_entry = ttk.Entry(form_frame, font=('Consolas', 10), width=20, show='*')
        self.password_entry.grid(row=1, column=1, pady=5)
        
        # 2FA Token
        tk.Label(
            form_frame,
            text="2FA Token:",
            font=('Consolas', 10),
            fg='#00ff00',
            bg='#0a0a0a'
        ).grid(row=2, column=0, sticky='w', pady=5)
        
        self.twofa_entry = ttk.Entry(form_frame, font=('Consolas', 10), width=20)
        self.twofa_entry.grid(row=2, column=1, pady=5)
        
        # Login button
        CyberpunkButton(
            login_window,
            text="JACK INTO THE MATRIX",
            command=lambda: self.process_login(login_window),
            sound_manager=self.sound_manager,
            width=25
        ).pack(pady=20)
        
        # Status label
        self.login_status = tk.Label(
            login_window,
            text="",
            font=('Consolas', 10),
            fg='#ff0000',
            bg='#0a0a0a'
        )
        self.login_status.pack()
        
    def process_login(self, login_window):
        """Process login with security checks"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        twofa_token = self.twofa_entry.get()
        
        # Authenticate
        success, message, session_token = self.security_manager.authenticate_user(
            username,
            password,
            twofa_token if twofa_token else None,
            "127.0.0.1"  # In production, get real IP
        )
        
        if success:
            self.authenticated = True
            self.current_user = username
            self.session_token = session_token
            
            # Update UI
            self.user_label.config(text=f"User: {username}")
            
            # Play success sound
            self.sound_manager.play('achievement')
            
            # Close login window
            login_window.destroy()
            
            # Log successful login
            self.audit_manager.audit_login(
                username,
                True,
                "127.0.0.1",
                {"method": "2FA" if twofa_token else "password"}
            )
            
            # Start main operations
            self.start_trading_operations()
            
        else:
            self.login_status.config(text=message)
            self.sound_manager.play('alert_high')
            
            # Log failed attempt
            self.audit_manager.audit_login(
                username,
                False,
                "127.0.0.1",
                {"reason": message}
            )
            
    def start_trading_operations(self):
        """Start all trading operations after successful login"""
        # Start strategy optimizer
        asyncio.create_task(self.strategy_optimizer.run_all_strategies({}))
        
        # Start predictive engine
        asyncio.create_task(self.update_predictions())
        
        # Start performance monitoring
        self.root.after(5000, self.update_performance_metrics)
        
        # Check achievements
        self.check_achievements()
        
    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = event.widget.tab('current')['text']
        self.sound_manager.play('tab_switch')
        
        # Log tab access
        self.audit_manager.blockchain_audit.add_entry(
            entry_type='navigation',
            user_id=self.current_user or 'anonymous',
            action='tab_switched',
            details={'tab': selected_tab}
        )
        
    def emergency_stop(self):
        """Execute emergency stop"""
        if messagebox.askyesno(
            "🚨 EMERGENCY STOP",
            "This will immediately halt all trading operations.\n\nAre you sure?",
            icon='warning'
        ):
            self.sound_manager.play('alert_high')
            
            # Stop all operations
            self.trading_engine.emergency_stop()
            
            # Log emergency stop
            self.audit_manager.blockchain_audit.add_entry(
                entry_type='emergency',
                user_id=self.current_user or 'anonymous',
                action='emergency_stop_activated',
                details={'timestamp': datetime.now().isoformat()}
            )
            
            # Update UI
            messagebox.showinfo(
                "System Halted",
                "All trading operations have been stopped.\n\nSystem is now in safe mode."
            )
            
    def apply_preset(self, preset: str):
        """Apply trading preset"""
        presets = {
            'conservative': {
                'risk_level': 0.01,
                'max_positions': 3,
                'strategies': ['arbitrage']
            },
            'aggressive': {
                'risk_level': 0.05,
                'max_positions': 10,
                'strategies': ['momentum', 'breakout']
            },
            'bear': {
                'risk_level': 0.02,
                'max_positions': 5,
                'strategies': ['short_bias', 'arbitrage']
            },
            'degen': {
                'risk_level': 0.10,
                'max_positions': 20,
                'strategies': ['all']
            }
        }
        
        if preset in presets:
            config = presets[preset]
            
            # Apply configuration
            self.config.update(config)
            
            # Log configuration change
            self.audit_manager.audit_config_change(
                self.current_user or 'anonymous',
                {
                    'preset_applied': preset,
                    'settings': config
                }
            )
            
            # Play sound
            self.sound_manager.play('click')
            
            # Show confirmation
            messagebox.showinfo(
                "Preset Applied",
                f"{preset.title()} trading mode activated!"
            )
            
    async def update_predictions(self):
        """Update predictive analytics"""
        while self.authenticated:
            try:
                # Get predictions for active symbols
                for symbol in ['BTC/USDT', 'ETH/USDT']:
                    # Volatility prediction
                    vol_prediction = await self.predictive_engine.predict_volatility(
                        symbol,
                        self.trading_engine.get_historical_data(symbol)
                    )
                    
                    # Update neural network activity based on predictions
                    self.neural_viz.set_activity_level(vol_prediction.confidence)
                    
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Prediction update error: {e}")
                await asyncio.sleep(300)
                
    def update_performance_metrics(self):
        """Update performance displays"""
        if not self.authenticated:
            return
            
        try:
            # Get current metrics
            metrics = self.trading_engine.get_performance_metrics()
            
            # Update displays
            self.stat_displays['Portfolio Value'].config(
                text=f"${metrics.get('portfolio_value', 0):,.2f}"
            )
            
            daily_pnl = metrics.get('daily_pnl', 0)
            daily_pnl_pct = metrics.get('daily_pnl_percent', 0)
            
            self.stat_displays['Daily P&L'].config(
                text=f"{'+' if daily_pnl >= 0 else ''}${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)",
                fg='#00ff00' if daily_pnl >= 0 else '#ff0000'
            )
            
            # Update drawdown protection
            self.drawdown_protection.update(metrics.get('portfolio_value', 0))
            
            # Schedule next update
            self.root.after(5000, self.update_performance_metrics)
            
        except Exception as e:
            logger.error(f"Performance update error: {e}")
            
    def check_achievements(self):
        """Check for new achievements"""
        if not self.authenticated:
            return
            
        # Get current stats
        stats = {
            'total_profit': 1500,  # Would come from trading engine
            'daily_profit': 150,
            'whale_trades_followed': 1,
            'longest_hold_hours': 48
        }
        
        # Check for new achievements
        new_achievements = self.gamification.check_achievements(
            self.current_user,
            stats
        )
        
        # Show notifications for new achievements
        for achievement in new_achievements:
            self.sound_manager.play('achievement')
            
            messagebox.showinfo(
                "🏆 Achievement Unlocked!",
                f"{achievement['badge']} {achievement['name']}\n\n"
                f"{achievement['description']}\n\n"
                f"+{achievement['xp']} XP"
            )
            
    def load_config(self) -> Dict:
        """Load configuration from file"""
        config_path = Path('config/enhanced_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
                
        # Return default config
        return {
            'version': '3.0.0',
            'theme': 'cyberpunk',
            'sound_enabled': True,
            'risk': {
                'warning_threshold': 0.05,
                'critical_threshold': 0.10,
                'emergency_threshold': 0.20
            },
            'ml': {
                'cache_ttl': 300,
                'prediction_horizon': 24
            },
            'users': {
                'admin': {
                    'password_hash': hashlib.sha256('admin'.encode()).hexdigest()
                }
            }
        }
        
    def save_config(self):
        """Save configuration to file"""
        config_path = Path('config/enhanced_config.json')
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def run(self):
        """Run the application"""
        # Set up async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start main loop
        self.root.mainloop()
        
        # Cleanup
        self.sound_manager.play('shutdown')
        self.animator.stop()

def main():
    """Main entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run application
    app = NexlifyEnhancedGUI()
    app.run()

if __name__ == "__main__":
    main()