# nexlify_cli.py
"""
Nexlify CLI - Command Line Interface for Night City's Trading Elite
Control your trading empire from the terminal like a true netrunner
"""

import click
import asyncio
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
import pandas as pd
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import print as rprint
import aiohttp
import websockets

# Initialize rich console
console = Console()

# API base URL
API_BASE = "http://localhost:8000"

class NexlifyCLI:
    """Main CLI class - your terminal trading companion"""
    
    def __init__(self):
        self.config_path = Path.home() / ".nexlify" / "config.json"
        self.config_path.parent.mkdir(exist_ok=True)
        self.api_token = None
        self.load_config()
    
    def load_config(self):
        """Load CLI configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.api_token = config.get("api_token")
    
    def save_config(self):
        """Save CLI configuration"""
        config = {
            "api_token": self.api_token,
            "last_login": datetime.now(timezone.utc).isoformat()
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make API request with authentication"""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE}{endpoint}"
            
            async with session.request(
                method,
                url,
                json=data,
                params=params,
                headers=headers
            ) as response:
                if response.status == 401:
                    raise click.ClickException("Authentication failed. Please login.")
                
                result = await response.json()
                
                if response.status >= 400:
                    raise click.ClickException(f"API Error: {result.get('error', 'Unknown error')}")
                
                return result

@click.group()
@click.pass_context
def cli(ctx):
    """
    Nexlify CLI - Trade like a netrunner from Night City
    
    Control your trading platform from the command line.
    """
    ctx.obj = NexlifyCLI()

# --- Authentication Commands ---
@cli.group()
def auth():
    """Authentication commands - jack in or out"""
    pass

@auth.command()
@click.option('--username', prompt=True, help='Your username')
@click.option('--password', prompt=True, hide_input=True, help='Your password')
@click.option('--pin', prompt=True, help='Your PIN')
@click.option('--totp', help='2FA code if enabled')
@click.pass_obj
def login(cli_obj, username, password, pin, totp):
    """Login to Nexlify - jack into the system"""
    async def do_login():
        try:
            with console.status("[cyan]Authenticating...", spinner="dots"):
                result = await cli_obj.api_request(
                    "POST",
                    "/auth/login",
                    data={
                        "username": username,
                        "password": password,
                        "pin": pin,
                        "totp_code": totp
                    }
                )
            
            cli_obj.api_token = result["access_token"]
            cli_obj.save_config()
            
            console.print("[green]✓[/green] Successfully logged in!")
            console.print(f"[cyan]Welcome back to the Net, {username}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Login failed: {e}")
    
    asyncio.run(do_login())

@auth.command()
@click.pass_obj
def logout(cli_obj):
    """Logout from Nexlify - jack out safely"""
    cli_obj.api_token = None
    cli_obj.save_config()
    console.print("[yellow]Logged out. See you in the Net, choom.[/yellow]")

@auth.command()
@click.option('--username', prompt=True, help='Choose a username')
@click.option('--email', prompt=True, help='Your email')
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)
@click.option('--enable-2fa/--no-2fa', default=True, help='Enable 2FA')
@click.pass_obj
def register(cli_obj, username, email, password, enable_2fa):
    """Register new account - join the elite traders"""
    async def do_register():
        try:
            with console.status("[cyan]Creating account...", spinner="dots"):
                result = await cli_obj.api_request(
                    "POST",
                    "/auth/register",
                    data={
                        "username": username,
                        "email": email,
                        "password": password,
                        "enable_2fa": enable_2fa
                    }
                )
            
            console.print("[green]✓[/green] Account created successfully!")
            console.print(f"\n[bold yellow]IMPORTANT - Save these details:[/bold yellow]")
            console.print(f"Username: [cyan]{username}[/cyan]")
            console.print(f"PIN: [bold red]{result['pin']}[/bold red] (This won't be shown again!)")
            
            if enable_2fa:
                console.print(f"\n[yellow]2FA Setup:[/yellow]")
                console.print(f"Scan this URI with your authenticator app:")
                console.print(f"[cyan]{result['totp_uri']}[/cyan]")
                console.print(f"\nBackup codes (save these!):")
                for code in result['backup_codes']:
                    console.print(f"  • {code}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Registration failed: {e}")
    
    asyncio.run(do_register())

# --- Portfolio Commands ---
@cli.group()
def portfolio():
    """Portfolio management - manage your digital wealth"""
    pass

@portfolio.command('list')
@click.pass_obj
def list_portfolios(cli_obj):
    """List all portfolios"""
    async def do_list():
        try:
            portfolios = await cli_obj.api_request("GET", "/portfolios")
            
            if not portfolios:
                console.print("[yellow]No portfolios found. Create one with 'nexlify portfolio create'[/yellow]")
                return
            
            table = Table(title="[bold cyan]Your Portfolios[/bold cyan]")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Initial Balance", style="yellow")
            table.add_column("Total P&L", style="green")
            table.add_column("Win Rate", style="blue")
            table.add_column("Status", style="white")
            
            for p in portfolios:
                pnl_color = "green" if float(p['total_pnl']) >= 0 else "red"
                table.add_row(
                    p['name'],
                    "Paper" if p['is_paper_trading'] else "Live",
                    f"${p['initial_balance']:,.2f}",
                    f"[{pnl_color}]${p['total_pnl']:,.2f}[/{pnl_color}]",
                    f"{p['win_rate']:.1%}",
                    "[green]Active[/green]" if p['is_active'] else "[red]Inactive[/red]"
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_list())

@portfolio.command('create')
@click.option('--name', prompt=True, help='Portfolio name')
@click.option('--balance', prompt=True, type=float, default=10000, help='Initial balance')
@click.option('--paper/--live', default=True, help='Paper trading or live')
@click.pass_obj
def create_portfolio(cli_obj, name, balance, paper):
    """Create new portfolio"""
    async def do_create():
        try:
            result = await cli_obj.api_request(
                "POST",
                "/portfolios",
                data={
                    "name": name,
                    "initial_balance": balance,
                    "is_paper_trading": paper
                }
            )
            
            console.print(f"[green]✓[/green] Portfolio '{name}' created!")
            console.print(f"ID: [cyan]{result['id']}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_create())

@portfolio.command('positions')
@click.argument('portfolio_id')
@click.option('--include-closed', is_flag=True, help='Include closed positions')
@click.pass_obj
def show_positions(cli_obj, portfolio_id, include_closed):
    """Show portfolio positions"""
    async def do_show():
        try:
            positions = await cli_obj.api_request(
                "GET",
                f"/portfolios/{portfolio_id}/positions",
                params={"include_closed": include_closed}
            )
            
            if not positions:
                console.print("[yellow]No positions found.[/yellow]")
                return
            
            table = Table(title="[bold cyan]Positions[/bold cyan]")
            table.add_column("Symbol", style="cyan")
            table.add_column("Side", style="magenta")
            table.add_column("Quantity", style="white")
            table.add_column("Entry", style="yellow")
            table.add_column("Current", style="yellow")
            table.add_column("P&L", style="green")
            table.add_column("Status", style="white")
            
            for pos in positions:
                pnl = float(pos.get('unrealized_pnl', 0) or pos.get('realized_pnl', 0))
                pnl_color = "green" if pnl >= 0 else "red"
                
                table.add_row(
                    pos['symbol'],
                    pos['side'].upper(),
                    f"{pos['quantity']:.8f}",
                    f"${pos['entry_price']:,.2f}",
                    f"${pos.get('current_price', 0):,.2f}" if pos['is_open'] else "-",
                    f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
                    "[green]Open[/green]" if pos['is_open'] else "[grey]Closed[/grey]"
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_show())

# --- Trading Commands ---
@cli.group()
def trade():
    """Trading commands - execute market operations"""
    pass

@trade.command('signal')
@click.argument('symbol')
@click.pass_obj
def get_signal(cli_obj, symbol):
    """Get AI trading signal for symbol"""
    async def do_signal():
        try:
            with console.status(f"[cyan]Consulting the neural oracle for {symbol}...", spinner="dots2"):
                signal = await cli_obj.api_request("POST", f"/ai/signal/{symbol}")
            
            # Create signal panel
            confidence_color = "green" if signal['confidence'] > 0.8 else "yellow" if signal['confidence'] > 0.6 else "red"
            action_color = "green" if signal['action'] == "buy" else "red" if signal['action'] == "sell" else "yellow"
            
            signal_text = f"""
[bold {action_color}]{signal['action'].upper()}[/bold {action_color}] Signal

[cyan]Confidence:[/cyan] [{confidence_color}]{signal['confidence']:.1%}[/{confidence_color}]
[cyan]Entry Price:[/cyan] ${signal['entry_price']:,.2f}
[cyan]Stop Loss:[/cyan] [red]${signal['stop_loss']:,.2f}[/red]
[cyan]Take Profit:[/cyan] [green]${signal['take_profit'][0]:,.2f}[/green]
[cyan]Risk/Reward:[/cyan] {signal['risk_reward_ratio']:.2f}
[cyan]Expected Return:[/cyan] {signal['expected_return']:.1%}
"""
            
            panel = Panel(
                signal_text,
                title=f"[bold magenta]AI Signal - {symbol}[/bold magenta]",
                border_style="magenta"
            )
            
            console.print(panel)
            
            # Show reasoning
            if signal.get('reasoning'):
                reason_table = Table(title="[cyan]Analysis Details[/cyan]", show_header=False)
                reason_table.add_column("Metric", style="cyan")
                reason_table.add_column("Value", style="white")
                
                if 'technical_indicators' in signal['reasoning']:
                    for indicator, value in signal['reasoning']['technical_indicators'].items():
                        if value is not None:
                            reason_table.add_row(indicator.upper(), f"{value:.2f}")
                
                console.print(reason_table)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_signal())

@trade.command('place')
@click.option('--portfolio', required=True, help='Portfolio ID')
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTC/USDT)')
@click.option('--side', type=click.Choice(['buy', 'sell']), required=True)
@click.option('--type', type=click.Choice(['market', 'limit']), default='market')
@click.option('--quantity', type=float, required=True, help='Order quantity')
@click.option('--price', type=float, help='Limit price (required for limit orders)')
@click.pass_obj
def place_order(cli_obj, portfolio, symbol, side, type, quantity, price):
    """Place a trading order"""
    async def do_place():
        try:
            if type == 'limit' and not price:
                raise click.ClickException("Price required for limit orders")
            
            order_data = {
                "portfolio_id": portfolio,
                "symbol": symbol,
                "type": type,
                "side": side,
                "quantity": quantity
            }
            
            if price:
                order_data["price"] = price
            
            with console.status("[cyan]Placing order...", spinner="dots"):
                result = await cli_obj.api_request("POST", "/orders", data=order_data)
            
            console.print(f"[green]✓[/green] Order placed successfully!")
            console.print(f"Order ID: [cyan]{result['order_id']}[/cyan]")
            console.print(f"Status: [yellow]{result['status']}[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_place())

# --- Market Data Commands ---
@cli.group()
def market():
    """Market data commands - read the matrix"""
    pass

@market.command('watch')
@click.argument('symbols', nargs=-1, required=True)
@click.option('--interval', default=2, help='Update interval in seconds')
@click.pass_obj
def watch_market(cli_obj, symbols, interval):
    """Watch live market data"""
    async def do_watch():
        try:
            console.print(f"[cyan]Watching markets: {', '.join(symbols)}[/cyan]")
            console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
            
            while True:
                # Create market table
                table = Table(title=f"[bold cyan]Market Watch - {datetime.now().strftime('%H:%M:%S')}[/bold cyan]")
                table.add_column("Symbol", style="cyan")
                table.add_column("Last", style="white")
                table.add_column("Change 24h", style="green")
                table.add_column("Volume", style="yellow")
                table.add_column("Spread", style="magenta")
                
                for symbol in symbols:
                    # In real implementation, would fetch from API
                    # For now, mock data
                    price = 50000 + (hash(symbol) % 1000)
                    change = (hash(symbol + str(datetime.now())) % 10) - 5
                    
                    change_color = "green" if change >= 0 else "red"
                    table.add_row(
                        symbol,
                        f"${price:,.2f}",
                        f"[{change_color}]{change:+.2f}%[/{change_color}]",
                        f"{abs(hash(symbol)) % 10000:,.0f}",
                        f"0.02%"
                    )
                
                console.clear()
                console.print(table)
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Market watch stopped.[/yellow]")
    
    asyncio.run(do_watch())

@market.command('candles')
@click.argument('symbol')
@click.option('--interval', default='1h', help='Candle interval')
@click.option('--limit', default=20, help='Number of candles')
@click.pass_obj
def show_candles(cli_obj, symbol, interval, limit):
    """Show historical candles"""
    async def do_candles():
        try:
            # Get symbol ID first
            markets = await cli_obj.api_request("GET", "/markets")
            symbols = await cli_obj.api_request("GET", f"/symbols/{markets[0]['id']}")
            
            symbol_obj = next((s for s in symbols if s['symbol'] == symbol), None)
            if not symbol_obj:
                raise click.ClickException(f"Symbol {symbol} not found")
            
            candles = await cli_obj.api_request(
                "GET",
                f"/market-data/{symbol_obj['id']}/candles",
                params={"interval": interval, "limit": limit}
            )
            
            # Create candle chart (text-based)
            console.print(f"\n[bold cyan]{symbol} - {interval} Candles[/bold cyan]\n")
            
            for candle in candles[-10:]:  # Show last 10
                open_price = candle['open']
                close_price = candle['close']
                high_price = candle['high']
                low_price = candle['low']
                
                # Simple ASCII candle
                is_green = close_price >= open_price
                color = "green" if is_green else "red"
                
                # Create candle visualization
                price_range = high_price - low_price
                if price_range > 0:
                    body_start = min(open_price, close_price)
                    body_end = max(open_price, close_price)
                    
                    wick_top = int(((high_price - body_end) / price_range) * 5)
                    body_size = int(((body_end - body_start) / price_range) * 5) or 1
                    wick_bottom = int(((body_start - low_price) / price_range) * 5)
                    
                    candle_str = " " * 10 + "|" * wick_top + "█" * body_size + "|" * wick_bottom
                else:
                    candle_str = " " * 10 + "─"
                
                console.print(
                    f"{candle['timestamp'][:10]} [{color}]{candle_str}[/{color}] "
                    f"O:{open_price:.2f} H:{high_price:.2f} L:{low_price:.2f} C:{close_price:.2f}"
                )
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_candles())

# --- System Commands ---
@cli.group()
def system():
    """System management commands - maintain the chrome"""
    pass

@system.command('status')
@click.pass_obj
def system_status(cli_obj):
    """Show system status"""
    async def do_status():
        try:
            status = await cli_obj.api_request("GET", "/")
            
            # Create status layout
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3)
            )
            
            # Header
            header_text = Text("NEXLIFY SYSTEM STATUS", style="bold cyan", justify="center")
            layout["header"].update(Panel(header_text))
            
            # Body - split into services and metrics
            layout["body"].split_row(
                Layout(name="services"),
                Layout(name="metrics")
            )
            
            # Services status
            services_table = Table(title="Services", show_header=False)
            services_table.add_column("Service", style="cyan")
            services_table.add_column("Status", style="green")
            
            for service, online in status['services'].items():
                status_text = "[green]ONLINE[/green]" if online else "[red]OFFLINE[/red]"
                services_table.add_row(service.upper(), status_text)
            
            layout["services"].update(Panel(services_table))
            
            # Metrics
            metrics_table = Table(title="Metrics", show_header=False)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="yellow")
            
            for metric, value in status['metrics'].items():
                if isinstance(value, float):
                    metrics_table.add_row(metric.replace('_', ' ').title(), f"{value:.1f}%")
                else:
                    metrics_table.add_row(metric.replace('_', ' ').title(), str(value))
            
            layout["metrics"].update(Panel(metrics_table))
            
            # Footer
            footer_text = f"Version: {status['version']} | {status['status'].upper()}"
            layout["footer"].update(Panel(footer_text, style="dim"))
            
            console.print(layout)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_status())

@system.command('alerts')
@click.option('--include-resolved', is_flag=True, help='Include resolved alerts')
@click.pass_obj
def show_alerts(cli_obj, include_resolved):
    """Show system alerts"""
    async def do_alerts():
        try:
            alerts = await cli_obj.api_request(
                "GET",
                "/monitoring/alerts",
                params={"include_resolved": include_resolved}
            )
            
            if not alerts:
                console.print("[green]No active alerts. System running smooth![/green]")
                return
            
            for alert in alerts:
                severity_colors = {
                    'info': 'cyan',
                    'warning': 'yellow',
                    'error': 'red',
                    'critical': 'bold red',
                    'flatline': 'bold white on red'
                }
                
                color = severity_colors.get(alert['severity'], 'white')
                
                alert_panel = Panel(
                    f"{alert['description']}\n\n"
                    f"Component: {alert['component']}\n"
                    f"Metric: {alert['metric_value']:.2f} (threshold: {alert['threshold']:.2f})\n"
                    f"Time: {alert['timestamp']}",
                    title=f"[{color}]{alert['severity'].upper()} - {alert['title']}[/{color}]",
                    border_style=color
                )
                
                console.print(alert_panel)
                console.print()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_alerts())

# --- Backtesting Commands ---
@cli.group()
def backtest():
    """Backtesting commands - test strategies in the past"""
    pass

@backtest.command('run')
@click.option('--symbol', required=True, help='Symbol to backtest')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--capital', default=10000, help='Initial capital')
@click.pass_obj
def run_backtest(cli_obj, symbol, start, end, capital):
    """Run AI backtest"""
    async def do_backtest():
        try:
            with console.status(f"[cyan]Running backtest for {symbol}...", spinner="dots2"):
                result = await cli_obj.api_request(
                    "POST",
                    "/ai/backtest",
                    data={
                        "symbol": symbol,
                        "start_date": f"{start}T00:00:00Z",
                        "end_date": f"{end}T23:59:59Z",
                        "initial_capital": capital
                    }
                )
            
            # Display results
            return_color = "green" if float(result['total_return'].rstrip('%')) >= 0 else "red"
            
            results_text = f"""
[bold]Backtest Results[/bold]

[cyan]Period:[/cyan] {start} to {end}
[cyan]Symbol:[/cyan] {symbol}
[cyan]Initial Capital:[/cyan] ${capital:,.2f}

[bold]Performance:[/bold]
[cyan]Total Return:[/cyan] [{return_color}]{result['total_return']}[/{return_color}]
[cyan]Sharpe Ratio:[/cyan] {result['sharpe_ratio']:.2f}
[cyan]Max Drawdown:[/cyan] [red]{result['max_drawdown']}[/red]
[cyan]Win Rate:[/cyan] {result['win_rate']}

[bold]Trading Stats:[/bold]
[cyan]Total Trades:[/cyan] {result['total_trades']}
[cyan]Profit Factor:[/cyan] {result['profit_factor']:.2f}
"""
            
            console.print(Panel(
                results_text,
                title="[bold magenta]Backtest Complete[/bold magenta]",
                border_style="magenta"
            ))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(do_backtest())

# --- Main Entry Point ---
if __name__ == "__main__":
    # Show banner
    banner = """
[cyan]
    _   _           _ _  __       
   | \ | |         | (_)/ _|      
   |  \| | _____  _| |_| |_ _   _ 
   | . ` |/ _ \ \/ / | |  _| | | |
   | |\  |  __/>  <| | | | | |_| |
   |_| \_|\___/_/\_\_|_|_|  \__, |
                             __/ |
   [bold]Terminal Trading Interface[/bold] |___/ 
[/cyan]
"""
    console.print(banner)
    
    # Run CLI
    cli()
