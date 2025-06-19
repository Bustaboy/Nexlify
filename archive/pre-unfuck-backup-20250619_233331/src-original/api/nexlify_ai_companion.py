"""
Nexlify Enhanced - AI Trading Companion
Implements Feature 26: ChatGPT-style trading assistant with natural language interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import re
from dataclasses import dataclass
import openai
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Maintains conversation context and user preferences"""
    user_level: str = "intermediate"  # beginner, intermediate, expert
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    preferred_strategies: List[str] = None
    recent_topics: List[str] = None
    personality_mode: str = "cyberpunk"  # cyberpunk, professional, friendly
    
    def __post_init__(self):
        if self.preferred_strategies is None:
            self.preferred_strategies = []
        if self.recent_topics is None:
            self.recent_topics = []

class TradingKnowledgeBase:
    """Local knowledge base for trading concepts and strategies"""
    
    def __init__(self):
        self.concepts = {
            "arbitrage": {
                "definition": "Profiting from price differences of the same asset across different markets",
                "types": ["Spatial arbitrage", "Triangular arbitrage", "Statistical arbitrage"],
                "risk_level": "low",
                "requirements": ["Fast execution", "Multiple exchange accounts", "Low latency"],
                "example": "Buying BTC at $45,000 on Exchange A and selling at $45,100 on Exchange B"
            },
            "market_making": {
                "definition": "Providing liquidity by placing both buy and sell orders",
                "types": ["Passive", "Active", "Competitive"],
                "risk_level": "moderate",
                "requirements": ["Capital", "Low fees", "Risk management"],
                "example": "Placing buy at $44,950 and sell at $45,050 for BTC/USDT"
            },
            "momentum": {
                "definition": "Trading based on the strength of price trends",
                "types": ["Breakout", "Trend following", "Mean reversion fade"],
                "risk_level": "high",
                "requirements": ["Trend identification", "Stop losses", "Position sizing"],
                "example": "Buying when price breaks above 20-day high with increasing volume"
            }
        }
        
        self.market_conditions = {
            "bullish": {
                "indicators": ["Rising prices", "High volume", "Positive sentiment"],
                "strategies": ["Momentum long", "Buy dips", "Hold positions"],
                "warnings": ["Overbought conditions", "FOMO risk"]
            },
            "bearish": {
                "indicators": ["Falling prices", "Declining volume", "Negative sentiment"],
                "strategies": ["Short selling", "Cash preservation", "Arbitrage"],
                "warnings": ["Oversold bounce risk", "Capitulation events"]
            },
            "ranging": {
                "indicators": ["Sideways price action", "Decreasing volatility", "No clear trend"],
                "strategies": ["Range trading", "Market making", "Mean reversion"],
                "warnings": ["Breakout risk", "Low profit potential"]
            }
        }
        
        self.risk_management = {
            "position_sizing": {
                "kelly_criterion": "f = (bp - q) / b, where f is fraction to bet",
                "fixed_fractional": "Risk 1-2% of capital per trade",
                "volatility_based": "Size inversely proportional to ATR"
            },
            "stop_loss": {
                "percentage": "Typically 1-5% below entry",
                "atr_based": "2-3x ATR below entry",
                "support_based": "Just below key support levels"
            }
        }

class AITradingCompanion:
    """
    AI-powered trading companion with natural language interface
    """
    
    def __init__(self, parent: tk.Widget, trading_engine: Any, config: Dict):
        self.parent = parent
        self.trading_engine = trading_engine
        self.config = config
        
        # Initialize components
        self.knowledge_base = TradingKnowledgeBase()
        self.context = ConversationContext()
        self.conversation_history = []
        
        # UI elements
        self.setup_ui()
        
        # Response templates based on personality
        self.personality_templates = {
            "cyberpunk": {
                "greeting": "Welcome to the Neural Matrix, netrunner. Ready to jack into the markets?",
                "analysis": "Scanning the data streams... Neural patterns detected in {symbol}...",
                "suggestion": "The AI suggests: {action}. Confidence protocols engaged.",
                "warning": "⚠️ ALERT: Hostile market patterns detected. Defensive protocols recommended.",
                "success": "Transaction executed flawlessly. Another win for the Neural Net.",
                "error": "System glitch detected. Rerouting through backup nodes..."
            },
            "professional": {
                "greeting": "Good day. I'm your AI trading assistant. How may I help you?",
                "analysis": "Analyzing {symbol} based on current market conditions...",
                "suggestion": "Recommendation: {action}. Confidence level: {confidence}%",
                "warning": "Risk Alert: Unusual market conditions detected.",
                "success": "Trade executed successfully.",
                "error": "An error occurred. Please try again."
            },
            "friendly": {
                "greeting": "Hey there! 👋 Ready to explore the markets together?",
                "analysis": "Let me check {symbol} for you...",
                "suggestion": "Here's what I think: {action} 🎯",
                "warning": "Heads up! 🚨 The market's looking a bit risky right now.",
                "success": "Awesome! Trade went through perfectly! 🎉",
                "error": "Oops, something went wrong. Let's try that again."
            }
        }
        
        # Start greeting
        self.add_ai_message(self.get_personality_template("greeting"))
        
    def setup_ui(self):
        """Create the AI companion interface"""
        # Main container
        self.container = ttk.Frame(self.parent)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(self.container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text="🤖 AI Trading Companion",
            font=('Consolas', 16, 'bold')
        ).pack(side=tk.LEFT)
        
        # Personality selector
        ttk.Label(header_frame, text="Mode:").pack(side=tk.LEFT, padx=(20, 5))
        self.personality_var = tk.StringVar(value=self.context.personality_mode)
        personality_menu = ttk.Combobox(
            header_frame,
            textvariable=self.personality_var,
            values=["cyberpunk", "professional", "friendly"],
            width=15,
            state="readonly"
        )
        personality_menu.pack(side=tk.LEFT)
        personality_menu.bind('<<ComboboxSelected>>', self.on_personality_change)
        
        # Chat display
        self.chat_frame = ttk.Frame(self.container)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create custom text widget with styling
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            height=20,
            font=('Consolas', 10),
            bg='#1a1a1a',
            fg='#00ff00',
            insertbackground='#00ff00'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different message types
        self.chat_display.tag_config('ai', foreground='#00ffff')
        self.chat_display.tag_config('user', foreground='#ffffff')
        self.chat_display.tag_config('code', foreground='#ffff00', font=('Courier', 10))
        self.chat_display.tag_config('warning', foreground='#ff6600')
        self.chat_display.tag_config('success', foreground='#00ff00')
        self.chat_display.tag_config('error', foreground='#ff0000')
        
        # Make read-only
        self.chat_display.config(state=tk.DISABLED)
        
        # Input section
        input_frame = ttk.Frame(self.container)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Input field
        self.input_field = ttk.Entry(input_frame, font=('Consolas', 11))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind('<Return>', self.on_send_message)
        self.input_field.bind('<Up>', self.on_history_up)
        self.input_field.bind('<Down>', self.on_history_down)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.on_send_message
        )
        self.send_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Quick actions
        quick_frame = ttk.Frame(self.container)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(quick_frame, text="Quick Actions:").pack(side=tk.LEFT)
        
        quick_actions = [
            ("📊 Market Analysis", "analyze market conditions"),
            ("💰 Best Opportunities", "show best trading opportunities"),
            ("📈 My Performance", "show my trading performance"),
            ("❓ Help", "help")
        ]
        
        for label, command in quick_actions:
            btn = ttk.Button(
                quick_frame,
                text=label,
                command=lambda c=command: self.send_quick_command(c)
            )
            btn.pack(side=tk.LEFT, padx=5)
            
        # Input history
        self.input_history = []
        self.history_index = -1
        
    def on_send_message(self, event=None):
        """Handle user message"""
        message = self.input_field.get().strip()
        if not message:
            return
            
        # Add to history
        self.input_history.append(message)
        self.history_index = len(self.input_history)
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.add_user_message(message)
        
        # Process message asynchronously
        asyncio.create_task(self.process_user_message(message))
        
    async def process_user_message(self, message: str):
        """Process user message and generate response"""
        try:
            # Analyze intent
            intent = self.analyze_intent(message)
            
            # Generate response based on intent
            if intent['type'] == 'market_analysis':
                response = await self.handle_market_analysis(intent)
                
            elif intent['type'] == 'strategy_question':
                response = await self.handle_strategy_question(intent)
                
            elif intent['type'] == 'trade_execution':
                response = await self.handle_trade_execution(intent)
                
            elif intent['type'] == 'performance_query':
                response = await self.handle_performance_query(intent)
                
            elif intent['type'] == 'educational':
                response = await self.handle_educational_query(intent)
                
            elif intent['type'] == 'system_command':
                response = await self.handle_system_command(intent)
                
            else:
                response = await self.handle_general_conversation(message)
                
            # Add AI response
            self.add_ai_message(response)
            
            # Update context
            self.update_context(intent)
            
        except Exception as e:
            logger.error(f"AI companion error: {e}")
            self.add_ai_message(
                self.get_personality_template("error"),
                tag='error'
            )
            
    def analyze_intent(self, message: str) -> Dict:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        # Market analysis patterns
        if any(word in message_lower for word in ['analyze', 'analysis', 'market', 'conditions']):
            symbols = self.extract_symbols(message)
            return {
                'type': 'market_analysis',
                'symbols': symbols if symbols else ['BTC/USDT'],
                'timeframe': self.extract_timeframe(message)
            }
            
        # Strategy questions
        elif any(word in message_lower for word in ['strategy', 'how to', 'should i', 'what is']):
            return {
                'type': 'strategy_question',
                'topic': self.extract_strategy_topic(message)
            }
            
        # Trade execution
        elif any(word in message_lower for word in ['buy', 'sell', 'trade', 'execute']):
            return {
                'type': 'trade_execution',
                'action': self.extract_trade_action(message),
                'symbol': self.extract_symbols(message)[0] if self.extract_symbols(message) else None
            }
            
        # Performance queries
        elif any(word in message_lower for word in ['performance', 'profit', 'loss', 'pnl', 'results']):
            return {
                'type': 'performance_query',
                'period': self.extract_time_period(message)
            }
            
        # Educational queries
        elif any(word in message_lower for word in ['explain', 'what is', 'define', 'teach', 'learn']):
            return {
                'type': 'educational',
                'topic': self.extract_educational_topic(message)
            }
            
        # System commands
        elif message_lower.startswith('/'):
            return {
                'type': 'system_command',
                'command': message_lower[1:].split()[0],
                'args': message_lower.split()[1:]
            }
            
        # General conversation
        else:
            return {
                'type': 'general',
                'message': message
            }
            
    async def handle_market_analysis(self, intent: Dict) -> str:
        """Generate market analysis response"""
        symbols = intent['symbols']
        
        # Get market data
        analysis_results = []
        
        for symbol in symbols:
            # Get current market data from trading engine
            market_data = await self.trading_engine.get_market_data(symbol)
            
            # Analyze conditions
            condition = self.analyze_market_condition(market_data)
            
            # Get AI insights
            insights = self.generate_market_insights(market_data, condition)
            
            analysis_results.append({
                'symbol': symbol,
                'condition': condition,
                'insights': insights,
                'recommendation': self.generate_recommendation(condition, self.context.risk_tolerance)
            })
            
        # Format response
        response = self.format_market_analysis_response(analysis_results)
        
        return response
        
    def analyze_market_condition(self, market_data: Dict) -> str:
        """Determine market condition from data"""
        # Simple analysis based on price action and indicators
        price_change = market_data.get('price_change_24h', 0)
        volume_change = market_data.get('volume_change_24h', 0)
        volatility = market_data.get('volatility', 0)
        
        if price_change > 0.05 and volume_change > 0.2:
            return 'bullish'
        elif price_change < -0.05 and volume_change > 0.2:
            return 'bearish'
        elif abs(price_change) < 0.02 and volatility < 0.01:
            return 'ranging'
        else:
            return 'uncertain'
            
    def generate_market_insights(self, market_data: Dict, condition: str) -> List[str]:
        """Generate AI insights about market"""
        insights = []
        
        # Price insights
        if market_data.get('price_change_24h', 0) > 0.1:
            insights.append("Strong bullish momentum detected - potential FOMO conditions")
        elif market_data.get('price_change_24h', 0) < -0.1:
            insights.append("Heavy selling pressure - watch for oversold bounce")
            
        # Volume insights
        if market_data.get('volume_ratio', 1) > 2:
            insights.append("Unusual volume surge indicates high interest")
            
        # Support/Resistance
        if 'support_levels' in market_data:
            nearest_support = market_data['support_levels'][0]
            insights.append(f"Key support at ${nearest_support:,.2f}")
            
        # Pattern recognition
        if market_data.get('pattern'):
            insights.append(f"Pattern detected: {market_data['pattern']}")
            
        return insights
        
    def format_market_analysis_response(self, analysis_results: List[Dict]) -> str:
        """Format market analysis into readable response"""
        template = self.get_personality_template("analysis")
        
        response_parts = []
        
        for result in analysis_results:
            symbol_analysis = template.format(symbol=result['symbol'])
            
            # Add condition
            condition_emoji = {
                'bullish': '🟢',
                'bearish': '🔴',
                'ranging': '🟡',
                'uncertain': '⚪'
            }
            
            symbol_analysis += f"\n\nMarket Condition: {condition_emoji.get(result['condition'], '')} {result['condition'].upper()}"
            
            # Add insights
            if result['insights']:
                symbol_analysis += "\n\nKey Insights:"
                for insight in result['insights']:
                    symbol_analysis += f"\n• {insight}"
                    
            # Add recommendation
            symbol_analysis += f"\n\nRecommendation: {result['recommendation']}"
            
            response_parts.append(symbol_analysis)
            
        return "\n\n---\n\n".join(response_parts)
        
    async def handle_strategy_question(self, intent: Dict) -> str:
        """Handle strategy-related questions"""
        topic = intent.get('topic', 'general')
        
        # Check knowledge base
        if topic in self.knowledge_base.concepts:
            concept = self.knowledge_base.concepts[topic]
            
            response = f"Let me explain {topic} for you:\n\n"
            response += f"**Definition**: {concept['definition']}\n\n"
            
            if concept.get('types'):
                response += f"**Types**: {', '.join(concept['types'])}\n\n"
                
            response += f"**Risk Level**: {concept['risk_level'].upper()}\n\n"
            
            if concept.get('requirements'):
                response += f"**Requirements**:\n"
                for req in concept['requirements']:
                    response += f"• {req}\n"
                response += "\n"
                
            if concept.get('example'):
                response += f"**Example**: {concept['example']}\n"
                
            # Add personalized advice
            if self.context.risk_tolerance == 'conservative' and concept['risk_level'] == 'high':
                response += f"\n⚠️ Note: This strategy may be too risky for your conservative profile."
                
        else:
            response = "I'd be happy to help you with trading strategies. Could you be more specific about what you'd like to know?"
            
        return response
        
    async def handle_trade_execution(self, intent: Dict) -> str:
        """Handle trade execution requests"""
        action = intent.get('action')
        symbol = intent.get('symbol')
        
        if not symbol:
            return "Please specify which trading pair you'd like to trade. For example: 'buy BTC/USDT'"
            
        # Analyze before suggesting trade
        market_data = await self.trading_engine.get_market_data(symbol)
        condition = self.analyze_market_condition(market_data)
        
        # Risk check
        if self.should_warn_about_trade(action, condition, self.context.risk_tolerance):
            warning = self.get_personality_template("warning")
            return f"{warning}\n\nMarket condition is {condition} but you want to {action}. Are you sure?"
            
        # Generate trade suggestion
        suggestion = f"Based on current analysis, here's my suggestion for {symbol}:\n\n"
        
        # Calculate position size
        position_size = self.calculate_suggested_position_size(
            market_data,
            self.context.risk_tolerance
        )
        
        suggestion += f"• Action: {action.upper()}\n"
        suggestion += f"• Suggested Size: {position_size}% of portfolio\n"
        suggestion += f"• Entry: ${market_data['current_price']:,.2f}\n"
        
        # Calculate stops and targets
        if action == 'buy':
            stop_loss = market_data['current_price'] * 0.98
            take_profit = market_data['current_price'] * 1.03
        else:
            stop_loss = market_data['current_price'] * 1.02
            take_profit = market_data['current_price'] * 0.97
            
        suggestion += f"• Stop Loss: ${stop_loss:,.2f}\n"
        suggestion += f"• Take Profit: ${take_profit:,.2f}\n"
        suggestion += f"\nWould you like me to execute this trade?"
        
        return suggestion
        
    def should_warn_about_trade(self, action: str, condition: str, risk_tolerance: str) -> bool:
        """Check if trade goes against market conditions"""
        risky_trades = [
            (action == 'buy' and condition == 'bearish'),
            (action == 'sell' and condition == 'bullish'),
            (risk_tolerance == 'conservative' and condition == 'uncertain')
        ]
        
        return any(risky_trades)
        
    def add_user_message(self, message: str):
        """Add user message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"\n[{timestamp}] You: ", 'user')
        self.chat_display.insert(tk.END, message + "\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now()
        })
        
    def add_ai_message(self, message: str, tag: str = 'ai'):
        """Add AI message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"\n[{timestamp}] AI: ", tag)
        
        # Process message for special formatting
        formatted_message = self.format_ai_message(message)
        self.chat_display.insert(tk.END, formatted_message + "\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': message,
            'timestamp': datetime.now()
        })
        
    def format_ai_message(self, message: str) -> str:
        """Format AI message with markdown-style formatting"""
        # Bold text
        message = re.sub(r'\*\*(.*?)\*\*', r'\1', message)
        
        # Code blocks
        message = re.sub(r'```(.*?)```', r'\1', message, flags=re.DOTALL)
        
        return message
        
    def get_personality_template(self, template_type: str) -> str:
        """Get response template based on current personality"""
        personality = self.context.personality_mode
        return self.personality_templates[personality].get(
            template_type,
            "I'm not sure how to respond to that."
        )
        
    def extract_symbols(self, message: str) -> List[str]:
        """Extract trading symbols from message"""
        # Common crypto patterns
        patterns = [
            r'\b([A-Z]{2,5})/([A-Z]{2,5})\b',  # BTC/USDT
            r'\b([A-Z]{2,5})-([A-Z]{2,5})\b',  # BTC-USDT
            r'\b(BTC|ETH|BNB|SOL|ADA|DOT|AVAX|MATIC|LINK|UNI)\b'  # Common symbols
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, message.upper())
            if matches:
                if isinstance(matches[0], tuple):
                    symbols.extend([f"{m[0]}/{m[1]}" for m in matches])
                else:
                    symbols.extend([f"{m}/USDT" for m in matches])
                    
        return list(set(symbols))  # Remove duplicates
        
    def extract_timeframe(self, message: str) -> str:
        """Extract timeframe from message"""
        timeframes = {
            '1m': ['1 minute', '1min', '1m'],
            '5m': ['5 minutes', '5min', '5m'],
            '15m': ['15 minutes', '15min', '15m'],
            '1h': ['1 hour', '1hr', '1h', 'hourly'],
            '4h': ['4 hours', '4hr', '4h'],
            '1d': ['1 day', 'daily', '1d'],
            '1w': ['1 week', 'weekly', '1w']
        }
        
        message_lower = message.lower()
        
        for tf, patterns in timeframes.items():
            if any(p in message_lower for p in patterns):
                return tf
                
        return '1h'  # Default
        
    def update_context(self, intent: Dict):
        """Update conversation context based on interaction"""
        # Update recent topics
        if intent['type'] in self.context.recent_topics:
            self.context.recent_topics.remove(intent['type'])
            
        self.context.recent_topics.insert(0, intent['type'])
        
        # Keep only last 5 topics
        self.context.recent_topics = self.context.recent_topics[:5]
        
    def on_personality_change(self, event=None):
        """Handle personality mode change"""
        self.context.personality_mode = self.personality_var.get()
        
        # Announce change
        if self.context.personality_mode == "cyberpunk":
            message = "Neural pathways reconfigured. Cyberpunk mode activated. 🤖"
        elif self.context.personality_mode == "professional":
            message = "Switching to professional mode. How may I assist you?"
        else:
            message = "Hey! Friendly mode activated! Let's make some profits! 😊"
            
        self.add_ai_message(message)
        
    def send_quick_command(self, command: str):
        """Send a quick command"""
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, command)
        self.on_send_message()
        
    def on_history_up(self, event):
        """Navigate up in command history"""
        if self.input_history and self.history_index > 0:
            self.history_index -= 1
            self.input_field.delete(0, tk.END)
            self.input_field.insert(0, self.input_history[self.history_index])
            
    def on_history_down(self, event):
        """Navigate down in command history"""
        if self.input_history and self.history_index < len(self.input_history) - 1:
            self.history_index += 1
            self.input_field.delete(0, tk.END)
            self.input_field.insert(0, self.input_history[self.history_index])
        elif self.history_index == len(self.input_history) - 1:
            self.history_index = len(self.input_history)
            self.input_field.delete(0, tk.END)
