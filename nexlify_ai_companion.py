#!/usr/bin/env python3
"""
Nexlify AI Trading Companion
Intelligent assistant that provides trading insights and recommendations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import random

from error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class AITradingCompanion:
    """
    AI-powered trading companion that provides insights, recommendations,
    and conversational assistance
    """

    def __init__(self, neural_net=None, gui=None, config: Dict = None):
        self.neural_net = neural_net
        self.gui = gui
        self.config = config or {}
        self.enabled = self.config.get('ai_companion_enabled', True)

        # Conversation history
        self.conversation_history: List[Dict] = []
        self.max_history = 100

        # Insight cache
        self.last_insights: List[str] = []

        # Personality settings
        self.personality = self.config.get('personality', 'professional')  # professional, friendly, cyberpunk

        logger.info("🤖 AI Trading Companion initialized")

    @handle_errors("AI Analysis", reraise=False)
    def analyze_market_conditions(self, market_data: Dict) -> str:
        """
        Analyze current market conditions and provide insight

        Args:
            market_data: Current market data

        Returns:
            Human-readable market analysis
        """
        try:
            insights = []

            # Analyze overall market trend
            if 'btc_price' in market_data:
                btc_price = market_data['btc_price']
                insights.append(f"Bitcoin is currently trading at ${btc_price:,.2f}")

            # Analyze volatility
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility < 0.02:
                    insights.append("Markets are relatively calm with low volatility")
                elif volatility < 0.05:
                    insights.append("Moderate volatility detected - good for swing trading")
                else:
                    insights.append("⚠️ High volatility - exercise caution with position sizing")

            # Analyze active opportunities
            if 'active_pairs_count' in market_data:
                count = market_data['active_pairs_count']
                if count > 15:
                    insights.append(f"Strong market activity with {count} profitable opportunities detected")
                elif count > 5:
                    insights.append(f"{count} trading opportunities identified")
                else:
                    insights.append("Limited opportunities in current market conditions")

            # Combine insights
            if insights:
                return " | ".join(insights)
            else:
                return "Analyzing market conditions..."

        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return "Unable to analyze market conditions at this time"

    def generate_trade_recommendation(self, symbol: str, analysis_data: Dict) -> Dict:
        """
        Generate a trade recommendation based on analysis

        Args:
            symbol: Trading pair symbol
            analysis_data: Analysis data from predictive engine

        Returns:
            Recommendation dictionary
        """
        try:
            recommendation = {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': [],
                'risk_level': 'unknown',
                'suggested_entry': None,
                'suggested_exit': None,
                'stop_loss': None
            }

            # Analyze prediction
            if 'prediction' in analysis_data:
                pred = analysis_data['prediction']
                direction = pred.get('direction', 'neutral')
                confidence = pred.get('confidence', 0)

                if direction == 'bullish' and confidence > 0.7:
                    recommendation['action'] = 'buy'
                    recommendation['confidence'] = confidence
                    recommendation['reasoning'].append(
                        f"Strong bullish signal with {confidence*100:.0f}% confidence"
                    )
                elif direction == 'bearish' and confidence > 0.7:
                    recommendation['action'] = 'sell'
                    recommendation['confidence'] = confidence
                    recommendation['reasoning'].append(
                        f"Strong bearish signal with {confidence*100:.0f}% confidence"
                    )

            # Analyze technical indicators
            if 'indicators' in analysis_data:
                indicators = analysis_data['indicators']

                # RSI analysis
                if 'rsi' in indicators:
                    rsi = indicators['rsi']
                    if rsi < 30:
                        recommendation['reasoning'].append("RSI indicates oversold conditions")
                        if recommendation['action'] == 'hold':
                            recommendation['action'] = 'buy'
                            recommendation['confidence'] = 0.6
                    elif rsi > 70:
                        recommendation['reasoning'].append("RSI indicates overbought conditions")
                        if recommendation['action'] == 'hold':
                            recommendation['action'] = 'sell'
                            recommendation['confidence'] = 0.6

                # MACD analysis
                if 'macd_histogram' in indicators:
                    macd_hist = indicators['macd_histogram']
                    if macd_hist > 0:
                        recommendation['reasoning'].append("MACD shows bullish momentum")
                    else:
                        recommendation['reasoning'].append("MACD shows bearish momentum")

            # Determine risk level
            if 'volatility' in analysis_data:
                volatility = analysis_data['volatility'].get('volatility', 0)
                if volatility < 0.02:
                    recommendation['risk_level'] = 'low'
                elif volatility < 0.05:
                    recommendation['risk_level'] = 'medium'
                else:
                    recommendation['risk_level'] = 'high'

            # Add price targets if we have a recommendation
            if recommendation['action'] in ['buy', 'sell']:
                current_price = analysis_data.get('current_price', 0)
                if current_price > 0:
                    if recommendation['action'] == 'buy':
                        recommendation['suggested_entry'] = current_price * 0.995  # 0.5% below
                        recommendation['suggested_exit'] = current_price * 1.02   # 2% above
                        recommendation['stop_loss'] = current_price * 0.98        # 2% below
                    else:  # sell
                        recommendation['suggested_entry'] = current_price * 1.005  # 0.5% above
                        recommendation['suggested_exit'] = current_price * 0.98    # 2% below
                        recommendation['stop_loss'] = current_price * 1.02         # 2% above

            return recommendation

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': ['Error generating recommendation'],
                'risk_level': 'unknown'
            }

    def provide_insight(self, context: str = "general") -> str:
        """
        Provide contextual insights and tips

        Args:
            context: Context for the insight (general, startup, trading, etc.)

        Returns:
            Insight string
        """
        insights_db = {
            'startup': [
                "Welcome to Nexlify! Always start with testnet mode to familiarize yourself with the platform.",
                "Pro tip: Set your risk level to 'low' when starting out.",
                "Remember to configure your BTC withdrawal address for automated profit taking.",
                "The neural net will automatically scan for the most profitable pairs across exchanges."
            ],
            'trading': [
                "Never risk more than 2-5% of your portfolio on a single trade.",
                "The best traders are patient - wait for high-confidence signals.",
                "Diversification across multiple pairs can reduce overall portfolio risk.",
                "Set stop-losses on all positions to protect against unexpected moves.",
                "Monitor the risk score - pairs below 0.3 are generally safer."
            ],
            'profit': [
                "Consider taking partial profits at predetermined levels.",
                "Consistent small gains compound into significant returns over time.",
                "Don't forget to account for fees when calculating profitability.",
                "Automated profit withdrawal to BTC can help secure your gains."
            ],
            'general': [
                "Market conditions change - stay adaptable.",
                "The AI continuously learns from market patterns.",
                "Check the audit trail regularly for complete transparency.",
                "High volatility means both higher risk and higher opportunity."
            ]
        }

        # Get insights for context
        available_insights = insights_db.get(context, insights_db['general'])

        # Return a random insight that hasn't been shown recently
        unused_insights = [i for i in available_insights if i not in self.last_insights]

        if not unused_insights:
            # Reset if all shown
            self.last_insights = []
            unused_insights = available_insights

        insight = random.choice(unused_insights)
        self.last_insights.append(insight)

        # Keep only last 5 insights in memory
        if len(self.last_insights) > 5:
            self.last_insights.pop(0)

        return self._format_message(insight)

    def _format_message(self, message: str) -> str:
        """Format message based on personality"""
        if self.personality == 'cyberpunk':
            prefixes = ["[NEURAL-LINK] ", "[CYBER-TIP] ", "[MATRIX] "]
            return random.choice(prefixes) + message
        elif self.personality == 'friendly':
            prefixes = ["💡 ", "👋 ", "✨ "]
            return random.choice(prefixes) + message
        else:  # professional
            return message

    def get_daily_summary(self, trading_data: Dict) -> str:
        """
        Generate daily trading summary

        Args:
            trading_data: Trading statistics for the day

        Returns:
            Formatted summary string
        """
        try:
            summary_parts = []

            # Trades executed
            trades_count = trading_data.get('trades_count', 0)
            summary_parts.append(f"Executed {trades_count} trades today")

            # Win rate
            win_rate = trading_data.get('win_rate', 0)
            if win_rate > 0:
                summary_parts.append(f"Win rate: {win_rate*100:.1f}%")

            # Profit/Loss
            total_profit = trading_data.get('total_profit', 0)
            if total_profit > 0:
                summary_parts.append(f"Total profit: +${total_profit:.2f}")
            elif total_profit < 0:
                summary_parts.append(f"Total loss: -${abs(total_profit):.2f}")

            # Most profitable pair
            if 'best_pair' in trading_data:
                best_pair = trading_data['best_pair']
                summary_parts.append(f"Top performer: {best_pair}")

            if summary_parts:
                return " | ".join(summary_parts)
            else:
                return "No trading activity today"

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Unable to generate summary"

    def answer_question(self, question: str) -> str:
        """
        Answer user questions about trading and the platform

        Args:
            question: User's question

        Returns:
            Answer string
        """
        # Simple keyword-based responses (could be enhanced with NLP)
        question_lower = question.lower()

        if 'risk' in question_lower:
            return ("Risk management is crucial in trading. The platform offers three risk levels: "
                   "Low (conservative), Medium (balanced), and High (aggressive). Choose based on "
                   "your risk tolerance and experience level.")

        elif 'arbitrage' in question_lower:
            return ("Arbitrage involves buying an asset on one exchange and simultaneously selling "
                   "it on another where the price is higher. The neural net automatically scans "
                   "for arbitrage opportunities across all connected exchanges.")

        elif 'fee' in question_lower or 'fees' in question_lower:
            return ("All trades account for exchange fees, gas fees (for blockchain assets), and "
                   "withdrawal fees. The displayed profit is the NET profit after all fees.")

        elif 'withdraw' in question_lower:
            return ("Profits can be automatically withdrawn to your BTC wallet. Set your BTC address "
                   "in settings and configure auto-withdrawal thresholds.")

        elif 'testnet' in question_lower:
            return ("Testnet mode allows you to test strategies with fake money on exchange test "
                   "networks. Always enable testnet when learning the platform!")

        elif 'indicator' in question_lower or 'rsi' in question_lower or 'macd' in question_lower:
            return ("The platform uses multiple technical indicators: RSI (momentum), MACD (trend), "
                   "Bollinger Bands (volatility), and more. These help identify trading opportunities.")

        else:
            return ("I can help with questions about risk management, arbitrage, fees, withdrawals, "
                   "indicators, and platform features. What would you like to know?")

    def log_interaction(self, user_message: str, ai_response: str):
        """Log conversation interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'ai': ai_response
        }

        self.conversation_history.append(interaction)

        # Limit history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
