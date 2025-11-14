#!/usr/bin/env python3
"""
Nexlify Telegram Bot Integration
Remote monitoring and control via Telegram
"""

import asyncio
import logging
from typing import Dict, Optional, Callable
from datetime import datetime
import aiohttp

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class TelegramBot:
    """
    Telegram bot for remote monitoring and control
    Sends notifications and accepts commands
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.bot_token = self.config.get("telegram_bot_token", "")
        self.chat_id = self.config.get("telegram_chat_id", "")
        self.enabled = (
            self.config.get("telegram_enabled", False)
            and self.bot_token
            and self.chat_id
        )

        # Command handlers
        self.command_handlers: Dict[str, Callable] = {}

        # Statistics
        self.messages_sent = 0
        self.commands_received = 0

        if self.enabled:
            logger.info("📱 Telegram Bot initialized")
        else:
            logger.warning("⚠️ Telegram Bot disabled (missing token/chat_id)")

    async def send_message(
        self,
        message: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message to Telegram

        Args:
            message: Message text (supports Markdown)
            parse_mode: 'Markdown' or 'HTML'
            disable_notification: Silent notification

        Returns:
            Success status
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.messages_sent += 1
                        logger.debug(f"📤 Telegram message sent")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send Telegram message: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    async def send_trade_notification(
        self,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        pnl: Optional[float] = None,
    ):
        """Send trade execution notification"""
        emoji = "🟢" if action == "BUY" else "🔴"

        message = f"""
{emoji} *Trade Executed*

*Action:* {action}
*Pair:* {symbol}
*Amount:* {amount:.4f}
*Price:* ${price:,.2f}
"""

        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "📉"
            message += f"{pnl_emoji} *P&L:* ${pnl:,.2f} ({(pnl/price)*100:.2f}%)\n"

        message += f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        await self.send_message(message)

    async def send_performance_update(self, stats: Dict):
        """Send performance statistics update"""
        win_rate_emoji = "🎯" if stats.get("win_rate", 0) > 60 else "📊"
        profit_emoji = "💚" if stats.get("total_profit", 0) > 0 else "❌"

        message = f"""
📊 *Performance Update*

{profit_emoji} *Total Profit:* ${stats.get('total_profit', 0):,.2f}
{win_rate_emoji} *Win Rate:* {stats.get('win_rate', 0):.1f}%
*Total Trades:* {stats.get('total_trades', 0)}
*Active Positions:* {stats.get('active_positions', 0)}

_Last updated: {datetime.now().strftime('%H:%M:%S')}_
"""

        await self.send_message(message)

    async def send_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """
        Send alert notification

        Args:
            alert_type: 'warning', 'error', 'info'
            message: Alert message
            priority: 'normal' or 'high'
        """
        emoji_map = {"warning": "⚠️", "error": "❌", "info": "ℹ️", "success": "✅"}

        emoji = emoji_map.get(alert_type, "📢")

        formatted_message = f"""
{emoji} *{alert_type.upper()}*

{message}

*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # High priority alerts are not silent
        disable_notification = priority != "high"

        await self.send_message(
            formatted_message, disable_notification=disable_notification
        )

    async def send_opportunity_alert(self, opportunity: Dict):
        """Send trading opportunity alert"""
        message = f"""
🎯 *Trading Opportunity Detected*

*Pair:* {opportunity.get('symbol', 'N/A')}
*Profit Potential:* {opportunity.get('profit_score', 0):.2f}%
*Confidence:* {opportunity.get('neural_confidence', 0):.0%}
*Strategy:* {opportunity.get('strategy', 'N/A')}

*Entry Price:* ${opportunity.get('entry_price', 0):,.2f}
*Target:* ${opportunity.get('target_price', 0):,.2f}
*Stop Loss:* ${opportunity.get('stop_loss', 0):,.2f}

_Detected at {datetime.now().strftime('%H:%M:%S')}_
"""

        await self.send_message(message)

    async def send_risk_alert(self, alert_data: Dict):
        """Send risk management alert"""
        message = f"""
🚨 *RISK ALERT*

*Type:* {alert_data.get('type', 'Unknown')}
*Severity:* {alert_data.get('severity', 'Medium')}

*Details:*
{alert_data.get('message', 'No details available')}

*Current Exposure:*
• Daily Loss: ${alert_data.get('daily_loss', 0):,.2f}
• Open Positions: {alert_data.get('open_positions', 0)}
• Total Risk: {alert_data.get('total_risk_percent', 0):.1f}%

⚠️ *Action Required*
"""

        await self.send_message(message, disable_notification=False)

    async def send_daily_summary(self, summary: Dict):
        """Send end-of-day summary"""
        profit = summary.get("daily_profit", 0)
        profit_emoji = "🟢" if profit > 0 else "🔴"

        message = f"""
📅 *Daily Summary* - {datetime.now().strftime('%Y-%m-%d')}

{profit_emoji} *Daily P&L:* ${profit:,.2f}
*Trades Today:* {summary.get('trades_today', 0)}
*Win Rate:* {summary.get('win_rate_today', 0):.1f}%

*Best Trade:* ${summary.get('best_trade', 0):,.2f}
*Worst Trade:* ${summary.get('worst_trade', 0):,.2f}

*Total Equity:* ${summary.get('total_equity', 0):,.2f}
*Available Balance:* ${summary.get('available_balance', 0):,.2f}

*Status:* {summary.get('status', 'Active')}
"""

        await self.send_message(message)

    def register_command(self, command: str, handler: Callable):
        """Register a command handler"""
        self.command_handlers[command] = handler
        logger.info(f"Registered Telegram command: /{command}")

    async def start_polling(self, neural_net):
        """
        Start polling for commands
        Note: This is a simplified implementation
        For production, use python-telegram-bot library
        """
        if not self.enabled:
            logger.warning("Telegram bot not enabled, skipping polling")
            return

        # Register default commands
        self.register_command("status", lambda: self._handle_status(neural_net))
        self.register_command("profit", lambda: self._handle_profit(neural_net))
        self.register_command("positions", lambda: self._handle_positions(neural_net))
        self.register_command("stop", lambda: self._handle_stop(neural_net))
        self.register_command("start", lambda: self._handle_start(neural_net))

        logger.info("📱 Telegram bot started polling for commands")

        # Send startup message
        await self.send_message(
            "🤖 *Nexlify Bot Started*\n\nBot is now online and monitoring trades."
        )

    async def _handle_status(self, neural_net):
        """Handle /status command"""
        try:
            stats = (
                neural_net.get_auto_trader_stats()
                if hasattr(neural_net, "get_auto_trader_stats")
                else {}
            )

            message = f"""
📊 *Bot Status*

*Status:* Active ✅
*Auto-Trading:* {'Enabled' if stats.get('auto_trade_enabled', False) else 'Disabled'}
*Total Trades:* {stats.get('total_trades', 0)}
*Win Rate:* {stats.get('win_rate', 0):.1f}%
*Total Profit:* ${stats.get('total_profit', 0):,.2f}
*Active Positions:* {stats.get('active_positions', 0)}
"""

            await self.send_message(message)

        except Exception as e:
            await self.send_message(f"❌ Error getting status: {str(e)}")

    async def _handle_profit(self, neural_net):
        """Handle /profit command"""
        try:
            stats = (
                neural_net.get_auto_trader_stats()
                if hasattr(neural_net, "get_auto_trader_stats")
                else {}
            )

            profit = stats.get("total_profit", 0)
            emoji = "💚" if profit > 0 else "❌"

            message = f"""
{emoji} *Profit Report*

*Total Profit:* ${profit:,.2f}
*Today's Profit:* ${stats.get('daily_profit', 0):,.2f}
*Win Rate:* {stats.get('win_rate', 0):.1f}%
*Avg Profit/Trade:* ${profit / max(stats.get('total_trades', 1), 1):,.2f}

*Winning Trades:* {stats.get('winning_trades', 0)}
*Losing Trades:* {stats.get('losing_trades', 0)}
"""

            await self.send_message(message)

        except Exception as e:
            await self.send_message(f"❌ Error getting profit: {str(e)}")

    async def _handle_positions(self, neural_net):
        """Handle /positions command"""
        try:
            if hasattr(neural_net, "get_open_positions"):
                positions = await neural_net.get_open_positions()
            else:
                positions = []

            if not positions:
                await self.send_message("📭 No open positions")
                return

            message = "📊 *Open Positions*\n\n"

            for pos in positions[:5]:  # Limit to 5
                symbol = pos.get("symbol", "N/A")
                amount = pos.get("amount", 0)
                pnl = pos.get("pnl", 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"

                message += f"{pnl_emoji} *{symbol}*\n"
                message += f"Amount: {amount:.4f} | P&L: ${pnl:.2f}\n\n"

            await self.send_message(message)

        except Exception as e:
            await self.send_message(f"❌ Error getting positions: {str(e)}")

    async def _handle_stop(self, neural_net):
        """Handle /stop command"""
        try:
            if hasattr(neural_net, "toggle_auto_trading"):
                neural_net.toggle_auto_trading(False)
                await self.send_message(
                    "🛑 *Auto-trading STOPPED*\n\nAll automated trading has been disabled."
                )
            else:
                await self.send_message("❌ Auto-trading control not available")

        except Exception as e:
            await self.send_message(f"❌ Error stopping trading: {str(e)}")

    async def _handle_start(self, neural_net):
        """Handle /start command"""
        try:
            if hasattr(neural_net, "toggle_auto_trading"):
                neural_net.toggle_auto_trading(True)
                await self.send_message(
                    "✅ *Auto-trading STARTED*\n\nAutomated trading is now enabled."
                )
            else:
                await self.send_message("❌ Auto-trading control not available")

        except Exception as e:
            await self.send_message(f"❌ Error starting trading: {str(e)}")

    def get_statistics(self) -> Dict:
        """Get bot statistics"""
        return {
            "enabled": self.enabled,
            "messages_sent": self.messages_sent,
            "commands_received": self.commands_received,
            "registered_commands": list(self.command_handlers.keys()),
        }


if __name__ == "__main__":

    async def main():
        print("=" * 70)
        print("NEXLIFY TELEGRAM BOT DEMO")
        print("=" * 70)

        # Initialize (needs real token/chat_id from config)
        bot = TelegramBot(
            {
                "telegram_bot_token": "YOUR_BOT_TOKEN",
                "telegram_chat_id": "YOUR_CHAT_ID",
                "telegram_enabled": False,  # Set to True when configured
            }
        )

        if bot.enabled:
            # Send test messages
            await bot.send_message("🤖 *Test Message*\n\nTelegram bot is working!")

            await bot.send_trade_notification(
                action="BUY", symbol="BTC/USDT", amount=0.001, price=45000
            )

            await bot.send_performance_update(
                {
                    "total_profit": 1234.56,
                    "win_rate": 67.5,
                    "total_trades": 42,
                    "active_positions": 3,
                }
            )

            print("\n✅ Test messages sent!")
        else:
            print(
                "\n⚠️ Bot not configured. Set telegram_bot_token and telegram_chat_id in config."
            )

        print("\n" + "=" * 70)

    asyncio.run(main())
