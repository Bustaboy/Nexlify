#!/usr/bin/env python3
"""
Nexlify - Cyberpunk Trading Interface
Full-featured GUI for the Arasaka Neural-Net Trading Matrix
"""

import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import threading
from datetime import datetime
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import numpy as np

class CyberGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌃 NEXLIFY - ARASAKA NEURAL-NET TRADING MATRIX 🌃")
        self.root.geometry("1400x900")
        
        # Cyberpunk color scheme
        self.colors = {
            'bg': '#0a0a0a',
            'panel': '#1a1a1a',
            'accent': '#00ff00',
            'danger': '#ff0040',
            'warning': '#ffff00',
            'text': '#ffffff',
            'profit': '#00ff88',
            'loss': '#ff0044'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Configuration
        self.config = self.load_config()
        self.neural_net = None
        self.update_thread = None
        self.is_running = True
        
        # Build interface
        self.build_interface()
        
        # Start update loop
        self.start_updates()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        config_path = "config/neural_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'btc_wallet': '',
            'min_withdrawal': 100,
            'risk_level': 'medium',
            'auto_trade': True,
            'environment': {
                'debug': False,
                'log_level': 'INFO',
                'api_port': 8000,
                'database_url': 'sqlite:///data/trading.db',
                'emergency_contact': '',
                'telegram_bot_token': '',
                'telegram_chat_id': ''
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        os.makedirs('config', exist_ok=True)
        with open('config/neural_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def build_interface(self):
        """Build the cyberpunk interface"""
        # Check if first time setup needed
        if self.needs_setup():
            self.show_onboarding()
            return
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        self.build_header(main_frame)
        
        # Control Panel
        control_frame = tk.Frame(main_frame, bg=self.colors['panel'], relief='ridge', bd=2)
        control_frame.pack(fill='x', pady=(10, 5))
        self.build_controls(control_frame)
        
        # Main content area with tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, pady=5)
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.colors['bg'])
        style.configure('TNotebook.Tab', background=self.colors['panel'], 
                       foreground=self.colors['text'])
        
        # Tabs
        self.build_onboarding_tab()  # Add onboarding tab
        self.build_active_pairs_tab()
        self.build_profit_chart_tab()
        self.build_settings_tab()
        self.build_environment_tab()  # New environment settings tab
        self.build_logs_tab()
    
    def build_header(self, parent):
        """Build header with title and status"""
        header = tk.Frame(parent, bg=self.colors['bg'], height=60)
        header.pack(fill='x')
        
        # Title
        title = tk.Label(header, 
                        text="ARASAKA NEURAL-NET TRADING MATRIX",
                        font=('Courier New', 24, 'bold'),
                        bg=self.colors['bg'],
                        fg=self.colors['accent'])
        title.pack(side='left', padx=20)
        
        # Status panel
        status_frame = tk.Frame(header, bg=self.colors['panel'], relief='ridge', bd=1)
        status_frame.pack(side='right', padx=20)
        
        self.status_label = tk.Label(status_frame,
                                    text="● SYSTEM ONLINE",
                                    font=('Courier New', 12),
                                    bg=self.colors['panel'],
                                    fg=self.colors['accent'])
        self.status_label.pack(padx=10, pady=5)
    
    def build_controls(self, parent):
        """Build control panel"""
        # Left side - Trading controls
        left_frame = tk.Frame(parent, bg=self.colors['panel'])
        left_frame.pack(side='left', padx=20, pady=10)
        
        # Auto-trade toggle
        self.auto_trade_var = tk.BooleanVar(value=self.config.get('auto_trade', True))
        auto_trade_btn = tk.Checkbutton(left_frame,
                                        text="⚡ AUTO-TRADE ENABLED",
                                        font=('Courier New', 12, 'bold'),
                                        bg=self.colors['panel'],
                                        fg=self.colors['accent'],
                                        selectcolor=self.colors['panel'],
                                        variable=self.auto_trade_var,
                                        command=self.toggle_auto_trade)
        auto_trade_btn.pack(side='left', padx=10)
        
        # Emergency stop
        stop_btn = tk.Button(left_frame,
                            text="🛑 KILL SWITCH",
                            font=('Courier New', 12, 'bold'),
                            bg=self.colors['danger'],
                            fg='white',
                            command=self.emergency_stop,
                            relief='raised',
                            bd=3)
        stop_btn.pack(side='left', padx=10)
        
        # Right side - Wallet info
        right_frame = tk.Frame(parent, bg=self.colors['panel'])
        right_frame.pack(side='right', padx=20, pady=10)
        
        # BTC wallet
        wallet_label = tk.Label(right_frame,
                               text="💰 BTC WALLET:",
                               font=('Courier New', 10),
                               bg=self.colors['panel'],
                               fg=self.colors['text'])
        wallet_label.pack(side='left', padx=5)
        
        self.wallet_entry = tk.Entry(right_frame,
                                    font=('Courier New', 10),
                                    bg=self.colors['bg'],
                                    fg=self.colors['accent'],
                                    insertbackground=self.colors['accent'],
                                    width=40)
        self.wallet_entry.pack(side='left', padx=5)
        self.wallet_entry.insert(0, self.config.get('btc_wallet', ''))
        
        save_wallet_btn = tk.Button(right_frame,
                                   text="SAVE",
                                   font=('Courier New', 10),
                                   bg=self.colors['accent'],
                                   fg=self.colors['bg'],
                                   command=self.save_wallet)
        save_wallet_btn.pack(side='left', padx=5)
    
    def build_active_pairs_tab(self):
        """Build active pairs visualization tab"""
        pairs_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(pairs_frame, text="🎯 ACTIVE PAIRS")
        
        # Scrollable frame for pairs
        canvas = tk.Canvas(pairs_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(pairs_frame, orient="vertical", command=canvas.yview)
        self.pairs_container = tk.Frame(canvas, bg=self.colors['bg'])
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=self.pairs_container, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure canvas scrolling
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.pairs_container.bind("<Configure>", configure_scroll)
        
        # Initial message
        self.no_pairs_label = tk.Label(self.pairs_container,
                                      text="Scanning for profitable pairs...",
                                      font=('Courier New', 16),
                                      bg=self.colors['bg'],
                                      fg=self.colors['accent'])
        self.no_pairs_label.pack(pady=50)
    
    def build_profit_chart_tab(self):
        """Build profit visualization tab"""
        chart_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(chart_frame, text="📊 PROFIT MATRIX")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 6), facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111, facecolor=self.colors['bg'])
        
        # Style the plot
        self.ax.spines['bottom'].set_color(self.colors['accent'])
        self.ax.spines['top'].set_color(self.colors['accent'])
        self.ax.spines['left'].set_color(self.colors['accent'])
        self.ax.spines['right'].set_color(self.colors['accent'])
        self.ax.tick_params(colors=self.colors['text'])
        self.ax.xaxis.label.set_color(self.colors['text'])
        self.ax.yaxis.label.set_color(self.colors['text'])
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def build_settings_tab(self):
        """Build settings tab"""
        settings_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(settings_frame, text="⚙️ NEURAL CONFIG")
        
        # Settings container
        container = tk.Frame(settings_frame, bg=self.colors['panel'], relief='ridge', bd=2)
        container.pack(padx=50, pady=50, fill='both', expand=True)
        
        # Title
        title = tk.Label(container,
                        text="NEURAL-NET CONFIGURATION",
                        font=('Courier New', 18, 'bold'),
                        bg=self.colors['panel'],
                        fg=self.colors['accent'])
        title.pack(pady=20)
        
        # Risk level
        risk_frame = tk.Frame(container, bg=self.colors['panel'])
        risk_frame.pack(pady=10)
        
        tk.Label(risk_frame,
                text="Risk Level:",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=self.colors['text']).pack(side='left', padx=10)
        
        self.risk_var = tk.StringVar(value=self.config.get('risk_level', 'medium'))
        risk_options = ['low', 'medium', 'high', 'extreme']
        
        for risk in risk_options:
            color = {'low': self.colors['profit'], 
                    'medium': self.colors['warning'],
                    'high': '#ff8800',
                    'extreme': self.colors['danger']}[risk]
            
            tk.Radiobutton(risk_frame,
                          text=risk.upper(),
                          font=('Courier New', 10),
                          bg=self.colors['panel'],
                          fg=color,
                          selectcolor=self.colors['bg'],
                          variable=self.risk_var,
                          value=risk).pack(side='left', padx=5)
        
        # Min withdrawal
        withdrawal_frame = tk.Frame(container, bg=self.colors['panel'])
        withdrawal_frame.pack(pady=10)
        
        tk.Label(withdrawal_frame,
                text="Min Withdrawal ($):",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=self.colors['text']).pack(side='left', padx=10)
        
        self.withdrawal_entry = tk.Entry(withdrawal_frame,
                                        font=('Courier New', 12),
                                        bg=self.colors['bg'],
                                        fg=self.colors['accent'],
                                        width=10)
        self.withdrawal_entry.pack(side='left', padx=5)
        self.withdrawal_entry.insert(0, str(self.config.get('min_withdrawal', 100)))
        
        # Save button
        save_btn = tk.Button(container,
                            text="💾 SAVE CONFIGURATION",
                            font=('Courier New', 14, 'bold'),
                            bg=self.colors['accent'],
                            fg=self.colors['bg'],
                            command=self.save_settings,
                            relief='raised',
                            bd=3)
        save_btn.pack(pady=30)
    
    def build_logs_tab(self):
        """Build logs tab"""
        logs_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(logs_frame, text="📜 NEURAL LOGS")
        
        # Log text widget
        self.log_text = tk.Text(logs_frame,
                               font=('Courier New', 10),
                               bg=self.colors['bg'],
                               fg=self.colors['accent'],
                               insertbackground=self.colors['accent'])
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initial log
        self.log_text.insert('1.0', f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Neural-Net initialized\n")
        self.log_text.insert('end', f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Welcome to Night City\n")
    
    def update_active_pairs(self, pairs: List[Dict]):
        """Update active pairs display"""
        # Clear existing pairs
        for widget in self.pairs_container.winfo_children():
            widget.destroy()
        
        if not pairs:
            self.no_pairs_label = tk.Label(self.pairs_container,
                                          text="No active pairs found",
                                          font=('Courier New', 16),
                                          bg=self.colors['bg'],
                                          fg=self.colors['warning'])
            self.no_pairs_label.pack(pady=50)
            return
        
        # Display pairs
        for i, pair in enumerate(pairs):
            self.create_pair_widget(self.pairs_container, pair, i)
    
    def create_pair_widget(self, parent, pair: Dict, index: int):
        """Create a cyberpunk-styled pair widget"""
        # Main frame
        pair_frame = tk.Frame(parent, 
                             bg=self.colors['panel'],
                             relief='ridge',
                             bd=2,
                             highlightbackground=self.colors['accent'],
                             highlightthickness=1)
        pair_frame.pack(fill='x', padx=10, pady=5)
        
        # Left section - Symbol and status
        left_frame = tk.Frame(pair_frame, bg=self.colors['panel'])
        left_frame.pack(side='left', padx=15, pady=10)
        
        symbol_label = tk.Label(left_frame,
                               text=pair['symbol'],
                               font=('Courier New', 16, 'bold'),
                               bg=self.colors['panel'],
                               fg=self.colors['accent'])
        symbol_label.pack()
        
        status_label = tk.Label(left_frame,
                               text=pair['status'],
                               font=('Courier New', 10),
                               bg=self.colors['panel'])
        status_label.pack()
        
        # Middle section - Metrics
        middle_frame = tk.Frame(pair_frame, bg=self.colors['panel'])
        middle_frame.pack(side='left', expand=True, fill='x', padx=20)
        
        # Profit
        profit_color = self.colors['profit'] if float(pair['profit_score'][:-1]) > 0 else self.colors['loss']
        tk.Label(middle_frame,
                text=f"Profit: {pair['profit_score']}",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=profit_color).pack(side='left', padx=10)
        
        # Volume
        tk.Label(middle_frame,
                text=f"Vol: {pair['volume']}",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=self.colors['text']).pack(side='left', padx=10)
        
        # Volatility
        tk.Label(middle_frame,
                text=f"Volatility: {pair['volatility']}",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=self.colors['warning']).pack(side='left', padx=10)
        
        # Neural confidence
        tk.Label(middle_frame,
                text=f"AI Confidence: {pair['confidence']}",
                font=('Courier New', 12),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(side='left', padx=10)
        
        # Right section - Exchanges
        right_frame = tk.Frame(pair_frame, bg=self.colors['panel'])
        right_frame.pack(side='right', padx=15, pady=10)
        
        tk.Label(right_frame,
                text="Exchanges:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text']).pack()
        
        tk.Label(right_frame,
                text=pair['exchanges'],
                font=('Courier New', 10, 'italic'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack()
    
    def update_profit_chart(self, data: List[float]):
        """Update profit chart with new data"""
        self.ax.clear()
        
        # Generate time series
        times = list(range(len(data)))
        
        # Plot profit line
        self.ax.plot(times, data, color=self.colors['accent'], linewidth=2, label='Profit')
        
        # Fill area under curve
        self.ax.fill_between(times, data, alpha=0.3, color=self.colors['accent'])
        
        # Add grid
        self.ax.grid(True, alpha=0.2, color=self.colors['accent'])
        
        # Labels
        self.ax.set_xlabel('Time (hours)', fontsize=12)
        self.ax.set_ylabel('Profit ($)', fontsize=12)
        self.ax.set_title('NEURAL-NET PROFIT MATRIX', fontsize=16, color=self.colors['accent'])
        
        # Redraw
        self.canvas.draw()
    
    def add_log(self, message: str):
        """Add message to logs"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
    
    def toggle_auto_trade(self):
        """Toggle auto-trading"""
        self.config['auto_trade'] = self.auto_trade_var.get()
        self.save_config()
        status = "ENABLED" if self.config['auto_trade'] else "DISABLED"
        self.add_log(f"Auto-trading {status}")
    
    def save_wallet(self):
        """Save BTC wallet address"""
        wallet = self.wallet_entry.get().strip()
        if wallet:
            self.config['btc_wallet'] = wallet
            self.save_config()
            self.add_log(f"BTC wallet updated: {wallet[:8]}...")
            messagebox.showinfo("Success", "BTC wallet saved successfully!")
    
    def needs_setup(self) -> bool:
        """Check if initial setup is needed"""
        # Check if API keys are configured
        if 'exchanges' not in self.config:
            return True
        
        for exchange, settings in self.config.get('exchanges', {}).items():
            if not settings.get('api_key') or settings['api_key'] == 'YOUR_API_KEY_HERE':
                return True
        
        return False
    
    def show_onboarding(self):
        """Show initial setup wizard"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Onboarding container
        onboard_frame = tk.Frame(self.root, bg=self.colors['bg'])
        onboard_frame.pack(fill='both', expand=True)
        
        # Title
        title_frame = tk.Frame(onboard_frame, bg=self.colors['bg'])
        title_frame.pack(pady=50)
        
        tk.Label(title_frame,
                text="WELCOME TO NIGHT CITY",
                font=('Courier New', 32, 'bold'),
                bg=self.colors['bg'],
                fg=self.colors['accent']).pack()
        
        tk.Label(title_frame,
                text="Neural-Net Trading Matrix Initial Setup",
                font=('Courier New', 16),
                bg=self.colors['bg'],
                fg=self.colors['text']).pack(pady=10)
        
        # Setup form
        setup_frame = tk.Frame(onboard_frame, bg=self.colors['panel'], relief='ridge', bd=2)
        setup_frame.pack(padx=100, pady=20, fill='both', expand=True)
        
        # Instructions
        tk.Label(setup_frame,
                text="⚡ CONFIGURE YOUR TRADING CREDENTIALS ⚡",
                font=('Courier New', 18, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['warning']).pack(pady=20)
        
        # Exchange setup
        self.exchange_entries = {}
        exchanges = ['binance', 'kraken', 'coinbase']
        
        for exchange in exchanges:
            self.create_exchange_setup(setup_frame, exchange)
        
        # BTC Wallet
        wallet_frame = tk.Frame(setup_frame, bg=self.colors['panel'])
        wallet_frame.pack(pady=20)
        
        tk.Label(wallet_frame,
                text="💰 BTC Withdrawal Wallet:",
                font=('Courier New', 14),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack()
        
        self.setup_wallet_entry = tk.Entry(wallet_frame,
                                          font=('Courier New', 12),
                                          bg=self.colors['bg'],
                                          fg=self.colors['accent'],
                                          width=50)
        self.setup_wallet_entry.pack(pady=5)
        
        # Risk Level
        risk_frame = tk.Frame(setup_frame, bg=self.colors['panel'])
        risk_frame.pack(pady=10)
        
        tk.Label(risk_frame,
                text="⚠️ Risk Level:",
                font=('Courier New', 14),
                bg=self.colors['panel'],
                fg=self.colors['warning']).pack()
        
        self.setup_risk_var = tk.StringVar(value='low')
        risk_levels = [
            ('Low Risk (Recommended)', 'low', self.colors['profit']),
            ('Medium Risk', 'medium', self.colors['warning']),
            ('High Risk', 'high', self.colors['danger'])
        ]
        
        for text, value, color in risk_levels:
            tk.Radiobutton(risk_frame,
                          text=text,
                          font=('Courier New', 12),
                          bg=self.colors['panel'],
                          fg=color,
                          selectcolor=self.colors['bg'],
                          variable=self.setup_risk_var,
                          value=value).pack()
        
        # Action buttons
        button_frame = tk.Frame(setup_frame, bg=self.colors['panel'])
        button_frame.pack(pady=30)
        
        tk.Button(button_frame,
                 text="🚀 JACK INTO THE MATRIX",
                 font=('Courier New', 16, 'bold'),
                 bg=self.colors['accent'],
                 fg=self.colors['bg'],
                 command=self.complete_onboarding,
                 relief='raised',
                 bd=3).pack(side='left', padx=10)
        
        tk.Button(button_frame,
                 text="SKIP SETUP",
                 font=('Courier New', 12),
                 bg=self.colors['danger'],
                 fg='white',
                 command=self.skip_onboarding).pack(side='left', padx=10)
    
    def create_exchange_setup(self, parent, exchange: str):
        """Create exchange API setup fields"""
        frame = tk.Frame(parent, bg=self.colors['panel'])
        frame.pack(pady=15, padx=30, fill='x')
        
        # Exchange header
        header_frame = tk.Frame(frame, bg=self.colors['panel'])
        header_frame.pack(fill='x')
        
        # Exchange name with checkbox
        self.exchange_entries[exchange] = {
            'enabled': tk.BooleanVar(value=exchange == 'binance'),  # Default enable Binance
            'api_key': tk.StringVar(),
            'secret': tk.StringVar(),
            'testnet': tk.BooleanVar(value=True)
        }
        
        check = tk.Checkbutton(header_frame,
                              text=f"🔗 {exchange.upper()}",
                              font=('Courier New', 14, 'bold'),
                              bg=self.colors['panel'],
                              fg=self.colors['accent'],
                              selectcolor=self.colors['bg'],
                              variable=self.exchange_entries[exchange]['enabled'])
        check.pack(side='left')
        
        # Testnet toggle
        tk.Checkbutton(header_frame,
                      text="TESTNET",
                      font=('Courier New', 10),
                      bg=self.colors['panel'],
                      fg=self.colors['warning'],
                      selectcolor=self.colors['bg'],
                      variable=self.exchange_entries[exchange]['testnet']).pack(side='right', padx=20)
        
        # API Key
        key_frame = tk.Frame(frame, bg=self.colors['panel'])
        key_frame.pack(fill='x', pady=5)
        
        tk.Label(key_frame,
                text="API Key:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=10,
                anchor='w').pack(side='left')
        
        key_entry = tk.Entry(key_frame,
                            font=('Courier New', 10),
                            bg=self.colors['bg'],
                            fg=self.colors['accent'],
                            textvariable=self.exchange_entries[exchange]['api_key'],
                            show='*' if exchange != 'testnet' else '',
                            width=50)
        key_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        # Secret
        secret_frame = tk.Frame(frame, bg=self.colors['panel'])
        secret_frame.pack(fill='x', pady=5)
        
        tk.Label(secret_frame,
                text="Secret:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=10,
                anchor='w').pack(side='left')
        
        secret_entry = tk.Entry(secret_frame,
                               font=('Courier New', 10),
                               bg=self.colors['bg'],
                               fg=self.colors['accent'],
                               textvariable=self.exchange_entries[exchange]['secret'],
                               show='*',
                               width=50)
        secret_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        # Test connection button
        tk.Button(frame,
                 text=f"TEST {exchange.upper()} CONNECTION",
                 font=('Courier New', 8),
                 bg=self.colors['warning'],
                 fg=self.colors['bg'],
                 command=lambda: self.test_exchange_connection(exchange)).pack(pady=5)
    
    def test_exchange_connection(self, exchange: str):
        """Test exchange API connection"""
        if not self.exchange_entries[exchange]['enabled'].get():
            messagebox.showwarning("Not Enabled", f"{exchange.upper()} is not enabled")
            return
        
        api_key = self.exchange_entries[exchange]['api_key'].get()
        secret = self.exchange_entries[exchange]['secret'].get()
        
        if not api_key or not secret:
            messagebox.showerror("Missing Credentials", f"Please enter API credentials for {exchange.upper()}")
            return
        
        # Show testing message
        messagebox.showinfo("Testing", f"Testing {exchange.upper()} connection...\n\nThis would verify:\n- API key validity\n- Permissions\n- Balance access")
    
    def complete_onboarding(self):
        """Complete the onboarding process"""
        # Validate at least one exchange is configured
        configured = False
        
        new_config = {
            'exchanges': {},
            'btc_wallet': self.setup_wallet_entry.get().strip(),
            'risk_level': self.setup_risk_var.get(),
            'auto_trade': True,
            'min_withdrawal': 100,
            'environment': {  # Default environment settings
                'debug': False,
                'log_level': 'INFO',
                'api_port': 8000,
                'database_url': 'sqlite:///data/trading.db',
                'emergency_contact': '',
                'telegram_bot_token': '',
                'telegram_chat_id': ''
            }
        }
        
        for exchange, entries in self.exchange_entries.items():
            if entries['enabled'].get():
                api_key = entries['api_key'].get().strip()
                secret = entries['secret'].get().strip()
                
                if api_key and secret:
                    new_config['exchanges'][exchange] = {
                        'api_key': api_key,
                        'secret': secret,
                        'testnet': entries['testnet'].get()
                    }
                    configured = True
        
        if not configured:
            messagebox.showerror("Configuration Required", 
                               "Please configure at least one exchange with API credentials")
            return
        
        # Save configuration
        self.config.update(new_config)
        self.save_config()
        
        # Apply environment settings
        self.apply_environment_settings()
        
        # Show success and rebuild interface
        messagebox.showinfo("Welcome to Night City", 
                          "Configuration saved!\n\nThe Neural-Net is now online.")
        
        # Rebuild main interface
        for widget in self.root.winfo_children():
            widget.destroy()
        self.build_interface()
    
    def skip_onboarding(self):
        """Skip onboarding (for development/testing)"""
        if messagebox.askyesno("Skip Setup?", 
                             "Are you sure you want to skip setup?\n\n"
                             "You'll need to configure API keys manually."):
            for widget in self.root.winfo_children():
                widget.destroy()
            self.build_interface()
    
    def build_onboarding_tab(self):
        """Build API configuration tab"""
        config_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(config_frame, text="🔐 API CONFIG")
        
        # Container
        container = tk.Frame(config_frame, bg=self.colors['panel'], relief='ridge', bd=2)
        container.pack(padx=30, pady=30, fill='both', expand=True)
        
        # Title
        tk.Label(container,
                text="EXCHANGE API CONFIGURATION",
                font=('Courier New', 18, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(pady=20)
        
        # Scrollable frame for exchanges
        canvas = tk.Canvas(container, bg=self.colors['panel'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        exchange_container = tk.Frame(canvas, bg=self.colors['panel'])
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=exchange_container, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create exchange configs
        self.api_entries = {}
        for exchange in ['binance', 'kraken', 'coinbase']:
            self.create_exchange_config(exchange_container, exchange)
        
        # Configure scrolling
        exchange_container.bind("<Configure>", 
                               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    def create_exchange_config(self, parent, exchange: str):
        """Create exchange configuration in settings"""
        frame = tk.Frame(parent, bg=self.colors['panel'], relief='groove', bd=1)
        frame.pack(fill='x', padx=20, pady=10)
        
        # Header
        header = tk.Frame(frame, bg=self.colors['panel'])
        header.pack(fill='x', padx=10, pady=5)
        
        tk.Label(header,
                text=f"🔗 {exchange.upper()}",
                font=('Courier New', 14, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(side='left')
        
        # Get current config
        current_config = self.config.get('exchanges', {}).get(exchange, {})
        
        # API fields
        self.api_entries[exchange] = {
            'api_key': tk.StringVar(value=current_config.get('api_key', '')),
            'secret': tk.StringVar(value=current_config.get('secret', '')),
            'testnet': tk.BooleanVar(value=current_config.get('testnet', True))
        }
        
        # API Key
        key_frame = tk.Frame(frame, bg=self.colors['panel'])
        key_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(key_frame,
                text="API Key:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=12,
                anchor='w').pack(side='left')
        
        tk.Entry(key_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.api_entries[exchange]['api_key'],
                width=40).pack(side='left', padx=5)
        
        # Secret
        secret_frame = tk.Frame(frame, bg=self.colors['panel'])
        secret_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(secret_frame,
                text="Secret:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=12,
                anchor='w').pack(side='left')
        
        tk.Entry(secret_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.api_entries[exchange]['secret'],
                show='*',
                width=40).pack(side='left', padx=5)
        
        # Testnet toggle
        tk.Checkbutton(frame,
                      text="Use Testnet",
                      font=('Courier New', 10),
                      bg=self.colors['panel'],
                      fg=self.colors['warning'],
                      selectcolor=self.colors['bg'],
                      variable=self.api_entries[exchange]['testnet']).pack(pady=5)
        
        # Save button
        tk.Button(frame,
                 text=f"SAVE {exchange.upper()}",
                 font=('Courier New', 10),
                 bg=self.colors['accent'],
                 fg=self.colors['bg'],
                 command=lambda: self.save_exchange_config(exchange)).pack(pady=5)
    
    def save_settings(self):
        """Save all settings"""
        try:
            self.config['risk_level'] = self.risk_var.get()
            self.config['min_withdrawal'] = float(self.withdrawal_entry.get())
            self.save_config()
            self.add_log("Settings saved successfully")
            messagebox.showinfo("Success", "Neural-Net configuration saved!")
        except ValueError:
            messagebox.showerror("Error", "Invalid withdrawal amount")
    
    def save_exchange_config(self, exchange: str):
        """Save individual exchange configuration"""
        api_key = self.api_entries[exchange]['api_key'].get().strip()
        secret = self.api_entries[exchange]['secret'].get().strip()
        testnet = self.api_entries[exchange]['testnet'].get()
        
        if not api_key or not secret:
            messagebox.showerror("Invalid Configuration", 
                               f"Please provide both API key and secret for {exchange.upper()}")
            return
        
        # Update config
        if 'exchanges' not in self.config:
            self.config['exchanges'] = {}
        
        self.config['exchanges'][exchange] = {
            'api_key': api_key,
            'secret': secret,
            'testnet': testnet
        }
        
        self.save_config()
        self.add_log(f"{exchange.upper()} configuration saved")
        messagebox.showinfo("Success", f"{exchange.upper()} configuration saved successfully!")
    
    def build_environment_tab(self):
        """Build environment settings tab"""
        env_frame = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(env_frame, text="🌐 ENVIRONMENT")
        
        # Scrollable container
        canvas = tk.Canvas(env_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(env_frame, orient="vertical", command=canvas.yview)
        container = tk.Frame(canvas, bg=self.colors['bg'])
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame = canvas.create_window((0, 0), window=container, anchor="nw")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_frame = tk.Frame(container, bg=self.colors['panel'], relief='ridge', bd=2)
        title_frame.pack(fill='x', padx=30, pady=20)
        
        tk.Label(title_frame,
                text="SYSTEM ENVIRONMENT CONFIGURATION",
                font=('Courier New', 18, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(pady=10)
        
        # Get current environment settings
        env_config = self.config.get('environment', {})
        
        # Debug Mode
        debug_frame = tk.Frame(container, bg=self.colors['panel'], relief='groove', bd=1)
        debug_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(debug_frame,
                text="🐛 Debug Mode",
                font=('Courier New', 14, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(anchor='w', padx=10, pady=5)
        
        self.debug_var = tk.BooleanVar(value=env_config.get('debug', False))
        tk.Checkbutton(debug_frame,
                      text="Enable Debug Mode (verbose logging)",
                      font=('Courier New', 10),
                      bg=self.colors['panel'],
                      fg=self.colors['text'],
                      selectcolor=self.colors['bg'],
                      variable=self.debug_var).pack(anchor='w', padx=20, pady=5)
        
        # Log Level
        log_frame = tk.Frame(container, bg=self.colors['panel'], relief='groove', bd=1)
        log_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(log_frame,
                text="📊 Log Level",
                font=('Courier New', 14, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(anchor='w', padx=10, pady=5)
        
        log_level_frame = tk.Frame(log_frame, bg=self.colors['panel'])
        log_level_frame.pack(fill='x', padx=20, pady=5)
        
        self.log_level_var = tk.StringVar(value=env_config.get('log_level', 'INFO'))
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in log_levels:
            color = {
                'DEBUG': self.colors['text'],
                'INFO': self.colors['accent'],
                'WARNING': self.colors['warning'],
                'ERROR': self.colors['danger'],
                'CRITICAL': self.colors['danger']
            }[level]
            
            tk.Radiobutton(log_level_frame,
                          text=level,
                          font=('Courier New', 10),
                          bg=self.colors['panel'],
                          fg=color,
                          selectcolor=self.colors['bg'],
                          variable=self.log_level_var,
                          value=level).pack(side='left', padx=10)
        
        # API Settings
        api_frame = tk.Frame(container, bg=self.colors['panel'], relief='groove', bd=1)
        api_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(api_frame,
                text="🔌 API Configuration",
                font=('Courier New', 14, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(anchor='w', padx=10, pady=5)
        
        # API Port
        port_frame = tk.Frame(api_frame, bg=self.colors['panel'])
        port_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(port_frame,
                text="API Port:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=15,
                anchor='w').pack(side='left')
        
        self.api_port_var = tk.StringVar(value=str(env_config.get('api_port', 8000)))
        tk.Entry(port_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.api_port_var,
                width=10).pack(side='left', padx=5)
        
        tk.Label(port_frame,
                text="(Default: 8000)",
                font=('Courier New', 9, 'italic'),
                bg=self.colors['panel'],
                fg=self.colors['text']).pack(side='left', padx=10)
        
        # Database URL
        db_frame = tk.Frame(api_frame, bg=self.colors['panel'])
        db_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(db_frame,
                text="Database URL:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=15,
                anchor='w').pack(side='left')
        
        self.db_url_var = tk.StringVar(value=env_config.get('database_url', 'sqlite:///data/trading.db'))
        tk.Entry(db_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.db_url_var,
                width=40).pack(side='left', padx=5)
        
        # Notification Settings
        notif_frame = tk.Frame(container, bg=self.colors['panel'], relief='groove', bd=1)
        notif_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(notif_frame,
                text="📱 Notifications",
                font=('Courier New', 14, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(anchor='w', padx=10, pady=5)
        
        # Emergency Contact
        contact_frame = tk.Frame(notif_frame, bg=self.colors['panel'])
        contact_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(contact_frame,
                text="Emergency Email:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=15,
                anchor='w').pack(side='left')
        
        self.emergency_contact_var = tk.StringVar(value=env_config.get('emergency_contact', ''))
        tk.Entry(contact_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.emergency_contact_var,
                width=35).pack(side='left', padx=5)
        
        # Telegram Settings
        telegram_frame = tk.Frame(notif_frame, bg=self.colors['panel'])
        telegram_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(telegram_frame,
                text="🤖 Telegram Bot Integration",
                font=('Courier New', 12, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['warning']).pack(anchor='w', pady=5)
        
        # Bot Token
        token_frame = tk.Frame(telegram_frame, bg=self.colors['panel'])
        token_frame.pack(fill='x', pady=5)
        
        tk.Label(token_frame,
                text="Bot Token:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=15,
                anchor='w').pack(side='left')
        
        self.telegram_token_var = tk.StringVar(value=env_config.get('telegram_bot_token', ''))
        tk.Entry(token_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.telegram_token_var,
                show='*' if env_config.get('telegram_bot_token') else '',
                width=35).pack(side='left', padx=5)
        
        # Chat ID
        chat_frame = tk.Frame(telegram_frame, bg=self.colors['panel'])
        chat_frame.pack(fill='x', pady=5)
        
        tk.Label(chat_frame,
                text="Chat ID:",
                font=('Courier New', 10),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=15,
                anchor='w').pack(side='left')
        
        self.telegram_chat_var = tk.StringVar(value=env_config.get('telegram_chat_id', ''))
        tk.Entry(chat_frame,
                font=('Courier New', 10),
                bg=self.colors['bg'],
                fg=self.colors['accent'],
                textvariable=self.telegram_chat_var,
                width=35).pack(side='left', padx=5)
        
        # Save button
        save_frame = tk.Frame(container, bg=self.colors['bg'])
        save_frame.pack(pady=20)
        
        tk.Button(save_frame,
                 text="💾 SAVE ENVIRONMENT SETTINGS",
                 font=('Courier New', 14, 'bold'),
                 bg=self.colors['accent'],
                 fg=self.colors['bg'],
                 command=self.save_environment_settings,
                 relief='raised',
                 bd=3).pack()
        
        # Instructions
        info_frame = tk.Frame(container, bg=self.colors['panel'], relief='ridge', bd=1)
        info_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(info_frame,
                text="ℹ️ Environment Settings Info",
                font=('Courier New', 12, 'bold'),
                bg=self.colors['panel'],
                fg=self.colors['accent']).pack(anchor='w', padx=10, pady=5)
        
        info_text = """• Debug Mode: Enables verbose logging for troubleshooting
• Log Level: Controls how much information is logged
• API Port: Port for internal API server (rarely needs changing)
• Database URL: Location of trading database
• Emergency Email: Receive alerts for critical events
• Telegram Bot: Get real-time notifications on your phone"""
        
        tk.Label(info_frame,
                text=info_text,
                font=('Courier New', 9),
                bg=self.colors['panel'],
                fg=self.colors['text'],
                justify='left').pack(anchor='w', padx=20, pady=5)
        
        # Configure scrolling
        container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    def save_environment_settings(self):
        """Save environment settings"""
        try:
            # Validate port
            port = int(self.api_port_var.get())
            if port < 1 or port > 65535:
                raise ValueError("Invalid port number")
            
            # Update config
            self.config['environment'] = {
                'debug': self.debug_var.get(),
                'log_level': self.log_level_var.get(),
                'api_port': port,
                'database_url': self.db_url_var.get().strip(),
                'emergency_contact': self.emergency_contact_var.get().strip(),
                'telegram_bot_token': self.telegram_token_var.get().strip(),
                'telegram_chat_id': self.telegram_chat_var.get().strip()
            }
            
            self.save_config()
            
            # Apply settings immediately
            self.apply_environment_settings()
            
            self.add_log("Environment settings saved and applied")
            messagebox.showinfo("Success", "Environment settings saved successfully!\n\nChanges will take full effect on next restart.")
            
        except ValueError as e:
            messagebox.showerror("Invalid Settings", f"Error: {e}\n\nPort must be between 1-65535")
    
    def apply_environment_settings(self):
        """Apply environment settings to the system"""
        env = self.config.get('environment', {})
        
        # Update logging
        log_level = getattr(logging, env.get('log_level', 'INFO'))
        logging.getLogger().setLevel(log_level)
        
        # Create/update .env file for compatibility
        env_content = f"""# Auto-generated by Night City Trader
# Edit these settings in the GUI instead!
DEBUG={str(env.get('debug', False)).lower()}
LOG_LEVEL={env.get('log_level', 'INFO')}
API_PORT={env.get('api_port', 8000)}
DATABASE_URL={env.get('database_url', 'sqlite:///data/trading.db')}
EMERGENCY_CONTACT={env.get('emergency_contact', '')}
TELEGRAM_BOT_TOKEN={env.get('telegram_bot_token', '')}
TELEGRAM_CHAT_ID={env.get('telegram_chat_id', '')}
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        if messagebox.askyesno("KILL SWITCH", "Activate emergency stop?\nThis will halt ALL trading operations!"):
            self.is_running = False
            self.status_label.config(text="● SYSTEM OFFLINE", fg=self.colors['danger'])
            self.add_log("EMERGENCY STOP ACTIVATED")
            
            # Create emergency stop file
            with open('EMERGENCY_STOP_ACTIVE', 'w') as f:
                f.write(str(datetime.now()))
    
    def start_updates(self):
        """Start background update thread"""
        def update_loop():
            while self.is_running:
                try:
                    # Simulate getting data from neural net
                    # In production, this would connect to the actual neural_net instance
                    
                    # Mock active pairs data
                    mock_pairs = [
                        {
                            'symbol': 'BTC/USDT',
                            'profit_score': '2.34%',
                            'volume': '$1.2B',
                            'volatility': '3.2%',
                            'confidence': '94%',
                            'exchanges': 'Binance, Kraken',
                            'status': '🟢 ACTIVE'
                        },
                        {
                            'symbol': 'ETH/USDT',
                            'profit_score': '1.89%',
                            'volume': '$834M',
                            'volatility': '4.1%',
                            'confidence': '87%',
                            'exchanges': 'Binance, Coinbase',
                            'status': '🟢 ACTIVE'
                        }
                    ]
                    
                    # Update GUI
                    self.root.after(0, self.update_active_pairs, mock_pairs)
                    
                    # Mock profit data
                    profit_data = np.cumsum(np.random.randn(24) * 10 + 2).tolist()
                    self.root.after(0, self.update_profit_chart, profit_data)
                    
                except Exception as e:
                    self.root.after(0, self.add_log, f"Update error: {e}")
                
                # Update every 5 seconds
                threading.Event().wait(5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    gui = CyberGUI()
    gui.run()