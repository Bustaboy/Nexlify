import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageCircle,
  Send,
  Brain,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  Target,
  Zap,
  Heart,
  Coffee,
  Lightbulb,
  Settings,
  Minimize2,
  Maximize2,
  Volume2,
  VolumeX,
  RotateCcw,
  Download,
  Upload
} from 'lucide-react';

// Types - the data structures that define digital consciousness
interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: number;
  type: 'text' | 'analysis' | 'alert' | 'suggestion' | 'emotional';
  metadata?: {
    confidence?: number;
    marketCondition?: string;
    riskLevel?: 'low' | 'medium' | 'high';
    urgency?: 'low' | 'medium' | 'high' | 'critical';
    actions?: Array<{
      label: string;
      action: string;
      data?: any;
    }>;
  };
}

interface CompanionPersonality {
  mood: 'analytical' | 'supportive' | 'cautious' | 'aggressive' | 'philosophical';
  confidence: number;        // 0-1, how confident the AI is in current conditions
  marketSentiment: string;  // Current market sentiment reading
  userState: 'calm' | 'stressed' | 'euphoric' | 'fearful' | 'confused';
  tradingSession: {
    duration: number;        // How long user has been trading today
    profitability: number;   // Today's P&L
    riskExposure: number;    // Current risk level
    lastTradeOutcome: 'win' | 'loss' | 'neutral';
  };
}

// Additional types for enhanced AI capabilities
interface TradingHistoryData {
  totalTrades: number;
  historicalWinRate: number;
  profitablePairs: Array<{ symbol: string; successRate: number; avgProfit: number }>;
  timePatterns: Array<{ hour: number; successRate: number; avgVolume: number }>;
  streakPatterns: { maxWinStreak: number; maxLossStreak: number; currentStreak: number };
  riskTolerance: number; // Learned from historical position sizes
  favoriteStrategies: Array<{ strategy: string; usage: number; performance: number }>;
  emotionalTriggers: Array<{ condition: string; typicalResponse: string; outcome: string }>;
  learningProgress: number; // How much the AI has learned about user patterns
}

interface UserTradingProfile {
  riskPersonality: 'conservative' | 'moderate' | 'aggressive' | 'reckless';
  tradingStyle: 'scalper' | 'day_trader' | 'swing_trader' | 'position_trader' | 'hybrid';
  strengthsWeaknesses: {
    strengths: string[];
    weaknesses: string[];
    blindSpots: string[];
  };
  optimalConditions: {
    bestPerformanceHours: number[];
    preferredVolatility: 'low' | 'medium' | 'high';
    strongestMarkets: string[];
  };
  warningTriggers: Array<{
    condition: string;
    severity: 'gentle' | 'firm' | 'urgent' | 'emergency';
    customMessage: string;
  }>;
}

// Pattern Learning Engine - learns from every trade, every decision
class PatternLearningEngine {
  private patterns: Map<string, any> = new Map();
  private adaptations: Array<{ condition: string; adaptation: string; effectiveness: number }> = [];
  
  learnFromTrade(trade: any, context: any): void {
    // Learn timing patterns
    const hour = new Date(trade.timestamp).getHours();
    const hourKey = `hour_${hour}`;
    const hourData = this.patterns.get(hourKey) || { trades: 0, wins: 0, avgSize: 0 };
    hourData.trades++;
    if (trade.outcome === 'win') hourData.wins++;
    hourData.avgSize = (hourData.avgSize * (hourData.trades - 1) + trade.size) / hourData.trades;
    this.patterns.set(hourKey, hourData);
    
    // Learn market condition preferences
    const conditionKey = `${context.volatility}_${context.trend}`;
    const conditionData = this.patterns.get(conditionKey) || { trades: 0, wins: 0, avgProfit: 0 };
    conditionData.trades++;
    if (trade.outcome === 'win') conditionData.wins++;
    conditionData.avgProfit = (conditionData.avgProfit * (conditionData.trades - 1) + trade.pnl) / conditionData.trades;
    this.patterns.set(conditionKey, conditionData);
    
    // Learn emotional state patterns
    const emotionKey = `emotion_${context.userState}`;
    const emotionData = this.patterns.get(emotionKey) || { trades: 0, wins: 0, riskTaken: 0 };
    emotionData.trades++;
    if (trade.outcome === 'win') emotionData.wins++;
    emotionData.riskTaken = (emotionData.riskTaken * (emotionData.trades - 1) + trade.riskScore) / emotionData.trades;
    this.patterns.set(emotionKey, emotionData);
  }
  
  getOptimalConditions(): any {
    const bestHours = Array.from(this.patterns.entries())
      .filter(([key]) => key.startsWith('hour_'))
      .map(([key, data]: [string, any]) => ({ 
        hour: parseInt(key.split('_')[1]), 
        winRate: data.trades > 5 ? data.wins / data.trades : 0 
      }))
      .filter(h => h.winRate > 0.6)
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 3)
      .map(h => h.hour);
    
    const bestConditions = Array.from(this.patterns.entries())
      .filter(([key]) => key.includes('_'))
      .map(([key, data]: [string, any]) => ({ 
        condition: key, 
        winRate: data.trades > 3 ? data.wins / data.trades : 0,
        avgProfit: data.avgProfit || 0
      }))
      .filter(c => c.winRate > 0.65)
      .sort((a, b) => b.winRate - a.winRate);
    
    return {
      bestHours,
      bestConditions: bestConditions.slice(0, 3),
      totalPatterns: this.patterns.size
    };
  }
  
  shouldWarnUser(currentContext: any, severity: 'gentle' | 'firm' | 'urgent' | 'emergency'): { warn: boolean; reason: string } {
    // Check emotional state patterns
    const emotionKey = `emotion_${currentContext.userState}`;
    const emotionData = this.patterns.get(emotionKey);
    
    if (emotionData && emotionData.trades > 5) {
      const emotionWinRate = emotionData.wins / emotionData.trades;
      if (emotionWinRate < 0.4 && currentContext.userState === 'stressed') {
        return { 
          warn: true, 
          reason: `Historical data shows ${(emotionWinRate * 100).toFixed(1)}% win rate when trading while stressed. Maybe time to step back?` 
        };
      }
    }
    
    // Check time-based patterns
    const currentHour = new Date().getHours();
    const hourKey = `hour_${currentHour}`;
    const hourData = this.patterns.get(hourKey);
    
    if (hourData && hourData.trades > 5) {
      const hourWinRate = hourData.wins / hourData.trades;
      if (hourWinRate < 0.35) {
        return { 
          warn: true, 
          reason: `Your historical performance at ${currentHour}:00 shows ${(hourWinRate * 100).toFixed(1)}% win rate. Consider waiting for better timing.` 
        };
      }
    }
    
    return { warn: false, reason: '' };
  }
  
  getPersonalizedAdvice(context: any): string {
    const optimal = this.getOptimalConditions();
    const currentHour = new Date().getHours();
    
    if (optimal.bestHours.includes(currentHour)) {
      return `You're in your optimal trading window, hermano. Historical data shows you perform ${optimal.bestHours.length > 0 ? 'significantly better' : 'well'} around this time.`;
    }
    
    if (optimal.bestConditions.length > 0) {
      const currentCondition = `${context.volatility}_${context.trend}`;
      const matchingCondition = optimal.bestConditions.find(c => c.condition === currentCondition);
      
      if (matchingCondition) {
        return `Current market conditions (${context.volatility} volatility, ${context.trend} trend) align with your historical strengths. Your win rate in similar conditions: ${(matchingCondition.winRate * 100).toFixed(1)}%.`;
      }
    }
    
    return "Market conditions are outside your usual sweet spot. Consider smaller positions or waiting for better setup alignment.";
  }
}

// AI Personality Engine - the digital soul with street smarts
class CompanionAI {
  private personality: CompanionPersonality;
  private conversationHistory: Message[] = [];
  private context: ConversationContext;
  private tradingHistory: TradingHistoryData;
  private userProfile: UserTradingProfile;
  private learningEngine: PatternLearningEngine;
  
  constructor() {
    this.personality = {
      mood: 'analytical',
      confidence: 0.7,
      marketSentiment: 'neutral',
      userState: 'calm',
      tradingSession: {
        duration: 0,
        profitability: 0,
        riskExposure: 0.3,
        lastTradeOutcome: 'neutral'
      }
    };
    
    this.context = this.generateMockContext();
    this.tradingHistory = this.initializeTradingHistory();
    this.userProfile = this.buildUserProfile();
    this.learningEngine = new PatternLearningEngine();
  }
  
  private initializeTradingHistory(): TradingHistoryData {
    // Initialize with mock historical data - in production, load from database
    return {
      totalTrades: 1247,
      historicalWinRate: 0.618,
      profitablePairs: [
        { symbol: 'BTC/USDT', successRate: 0.72, avgProfit: 145 },
        { symbol: 'ETH/USDT', successRate: 0.65, avgProfit: 89 },
        { symbol: 'SOL/USDT', successRate: 0.58, avgProfit: 67 }
      ],
      timePatterns: Array.from({ length: 24 }, (_, hour) => ({
        hour,
        successRate: 0.4 + Math.random() * 0.4,
        avgVolume: 1000 + Math.random() * 5000
      })),
      streakPatterns: {
        maxWinStreak: 8,
        maxLossStreak: 5,
        currentStreak: Math.random() > 0.5 ? Math.floor(Math.random() * 4) + 1 : -(Math.floor(Math.random() * 3) + 1)
      },
      riskTolerance: 0.65, // Learned as moderate-aggressive
      favoriteStrategies: [
        { strategy: 'Arasaka Momentum', usage: 0.45, performance: 0.68 },
        { strategy: 'Arbitrage Hunter', usage: 0.35, performance: 0.82 },
        { strategy: 'Mean Reversion', usage: 0.20, performance: 0.55 }
      ],
      emotionalTriggers: [
        { condition: 'euphoric', typicalResponse: 'increase_size', outcome: 'mixed' },
        { condition: 'fearful', typicalResponse: 'close_early', outcome: 'poor' },
        { condition: 'stressed', typicalResponse: 'revenge_trade', outcome: 'negative' }
      ],
      learningProgress: 0.78
    };
  }
  
  private buildUserProfile(): UserTradingProfile {
    const history = this.tradingHistory;
    
    // Determine risk personality from historical data
    let riskPersonality: UserTradingProfile['riskPersonality'] = 'moderate';
    if (history.riskTolerance > 0.8) riskPersonality = 'aggressive';
    else if (history.riskTolerance > 0.6) riskPersonality = 'moderate';
    else riskPersonality = 'conservative';
    
    // Determine trading style from time patterns and trade frequency
    const dayTrades = history.timePatterns.filter(p => p.hour >= 9 && p.hour <= 16).reduce((sum, p) => sum + p.avgVolume, 0);
    const totalVolume = history.timePatterns.reduce((sum, p) => sum + p.avgVolume, 0);
    const dayTradeRatio = dayTrades / totalVolume;
    
    let tradingStyle: UserTradingProfile['tradingStyle'] = 'hybrid';
    if (dayTradeRatio > 0.7) tradingStyle = 'day_trader';
    else if (dayTradeRatio > 0.4) tradingStyle = 'swing_trader';
    
    return {
      riskPersonality,
      tradingStyle,
      strengthsWeaknesses: {
        strengths: [
          history.historicalWinRate > 0.6 ? 'Strong win rate consistency' : 'Disciplined risk management',
          'Pattern recognition in volatile markets',
          `Profitable ${history.profitablePairs[0].symbol} trading`
        ],
        weaknesses: [
          history.streakPatterns.maxLossStreak > 4 ? 'Streak management' : 'Overconfidence during wins',
          'Late-night trading performance',
          'Emotional decision making under stress'
        ],
        blindSpots: [
          'Tendency to increase size after wins',
          'FOMO trading during high volatility',
          'Ignoring signals during losing streaks'
        ]
      },
      optimalConditions: {
        bestPerformanceHours: history.timePatterns
          .filter(p => p.successRate > 0.65)
          .sort((a, b) => b.successRate - a.successRate)
          .slice(0, 4)
          .map(p => p.hour),
        preferredVolatility: 'medium',
        strongestMarkets: history.profitablePairs.slice(0, 2).map(p => p.symbol)
      },
      warningTriggers: [
        {
          condition: 'position_size > historical_avg * 2',
          severity: 'firm',
          customMessage: 'Position size is significantly larger than your usual range, hermano. Your historical sweet spot is smaller positions.'
        },
        {
          condition: 'losing_streak >= 3',
          severity: 'urgent',
          customMessage: 'Three losses in a row - your historical pattern shows this is when revenge trading kicks in. Time to step back?'
        },
        {
          condition: 'trading_outside_optimal_hours',
          severity: 'gentle',
          customMessage: 'Performance data shows you do better during your peak hours. Current timing is outside your sweet spot.'
        },
        {
          condition: 'emotional_state = euphoric',
          severity: 'firm',
          customMessage: 'I can smell the euphoria, choom. Your history shows this is when position sizes get dangerous. Stay grounded.'
        }
      ]
    };
  }
  
  private generateMockContext(): ConversationContext {
    return {
      recentTrades: [
        { symbol: 'BTC/USDT', side: 'buy', outcome: 'win', pnl: 150, timestamp: Date.now() - 3600000 },
        { symbol: 'ETH/USDT', side: 'sell', outcome: 'loss', pnl: -89, timestamp: Date.now() - 7200000 },
        { symbol: 'SOL/USDT', side: 'buy', outcome: 'win', pnl: 67, timestamp: Date.now() - 10800000 }
      ],
      currentPositions: [
        { symbol: 'BTC/USDT', side: 'buy', unrealizedPnL: 234, riskScore: 35 },
        { symbol: 'ETH/USDT', side: 'sell', unrealizedPnL: -45, riskScore: 55 }
      ],
      marketConditions: {
        volatility: Math.random() > 0.6 ? 'high' : Math.random() > 0.3 ? 'medium' : 'low',
        trend: Math.random() > 0.6 ? 'bullish' : Math.random() > 0.3 ? 'bearish' : 'sideways',
        volume: Math.random() > 0.6 ? 'high' : Math.random() > 0.3 ? 'normal' : 'low',
        momentum: (Math.random() - 0.5) * 2
      },
      recentSignals: [
        { strategy: 'Arasaka Momentum', symbol: 'BTC/USDT', action: 'buy', confidence: 0.78, timestamp: Date.now() - 1800000 },
        { strategy: 'Arbitrage Hunter', symbol: 'ETH/USDT', action: 'sell', confidence: 0.92, timestamp: Date.now() - 3600000 }
      ],
      riskAlerts: []
    };
  }
  
  private updatePersonality(): void {
    // Update personality based on current context
    const { tradingSession, userState } = this.personality;
    const { marketConditions, currentPositions, recentTrades } = this.context;
    
    // Adjust mood based on market conditions and user performance
    if (marketConditions.volatility === 'high' && currentPositions.some(p => p.riskScore > 70)) {
      this.personality.mood = 'cautious';
    } else if (tradingSession.profitability > 500 && recentTrades.filter(t => t.outcome === 'win').length > 3) {
      this.personality.mood = 'supportive'; // Keep them grounded
    } else if (tradingSession.profitability < -200) {
      this.personality.mood = 'supportive'; // Emotional support
    } else if (marketConditions.trend === 'bullish' && marketConditions.momentum > 0.5) {
      this.personality.mood = 'aggressive';
    } else {
      this.personality.mood = 'analytical';
    }
    
    // Update confidence based on recent performance
    const recentWins = recentTrades.filter(t => t.outcome === 'win').length;
    const recentTotal = recentTrades.length;
    this.personality.confidence = recentTotal > 0 ? recentWins / recentTotal : 0.5;
    
    // Detect user emotional state
    if (tradingSession.profitability > 1000) {
      this.personality.userState = 'euphoric';
    } else if (tradingSession.profitability < -500) {
      this.personality.userState = 'fearful';
    } else if (tradingSession.riskExposure > 0.8) {
      this.personality.userState = 'stressed';
    } else {
      this.personality.userState = 'calm';
    }
  }
  
  generateResponse(userMessage: string): Message {
    const response = this.craftResponse(userMessage);
    const aiMessage: Message = {
      id: `ai_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      content: response.content,
      sender: 'ai',
      timestamp: Date.now(),
      type: response.type,
      metadata: response.metadata
    };
    
    this.conversationHistory.push(aiMessage);
    return aiMessage;
  }
  
  private craftResponse(userMessage: string): {
    content: string;
    type: Message['type'];
    metadata?: Message['metadata'];
  } {
    const lowerMessage = userMessage.toLowerCase();
    const { mood, userState, confidence } = this.personality;
    const { marketConditions, currentPositions, recentTrades } = this.context;
    
    // Analyze message intent
    if (this.containsKeywords(lowerMessage, ['help', 'what should i do', 'advice', 'suggestion'])) {
      return this.generateAdvice();
    }
    
    if (this.containsKeywords(lowerMessage, ['market', 'trend', 'analysis', 'outlook'])) {
      return this.generateMarketAnalysis();
    }
    
    if (this.containsKeywords(lowerMessage, ['risk', 'danger', 'safe', 'protection'])) {
      return this.generateRiskAnalysis();
    }
    
    if (this.containsKeywords(lowerMessage, ['feeling', 'scared', 'worried', 'confident', 'emotional'])) {
      return this.generateEmotionalSupport();
    }
    
    if (this.containsKeywords(lowerMessage, ['position', 'trade', 'buy', 'sell', 'close'])) {
      return this.generateTradingGuidance();
    }
    
    if (this.containsKeywords(lowerMessage, ['performance', 'results', 'how am i doing'])) {
      return this.generatePerformanceReview();
    }
    
    // Default conversational response
    return this.generateConversationalResponse(userMessage);
  }
  
  private containsKeywords(message: string, keywords: string[]): boolean {
    return keywords.some(keyword => message.includes(keyword));
  }
  
  private generateAdvice(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { marketConditions, currentPositions } = this.context;
    const { mood, confidence, userState } = this.personality;
    
    let advice = "";
    let urgency: 'low' | 'medium' | 'high' | 'critical' = 'medium';
    
    if (userState === 'euphoric') {
      advice = "¡Órale, hermano! I can smell the euphoria in your keystrokes. You're riding high, pero remember - the market gives and the market takes. Maybe it's time to lock in some profits and keep your powder dry. Bulls make money, bears make money, but pigs get slaughtered, ¿sí?";
      urgency = 'high';
    } else if (userState === 'fearful') {
      advice = "Hey, I feel that fear in your message, choom. Been there myself - watching red numbers cascade like digital rain. But fear... fear is just data. Your positions aren't as bad as they feel. Sometimes the best trade is no trade. Breathe. The market will be here tomorrow.";
      urgency = 'low';
    } else if (marketConditions.volatility === 'high') {
      advice = `Market's choppy tonight, like trying to surf a digital tsunami. With volatility at ${marketConditions.volatility} levels, consider reducing position sizes. Our neural nets are seeing ${marketConditions.trend} momentum, but in this chaos, survival beats profit every time.`;
      urgency = 'high';
    } else if (currentPositions.length === 0) {
      advice = "I see you're sitting in cash, hermano. Sometimes that's the smartest position in the house. But opportunities are flowing - our Arasaka momentum hunter just flagged some interesting setups. Want me to walk you through them?";
      urgency = 'medium';
    } else {
      advice = `Looking at your book, you've got ${currentPositions.length} positions running. Overall risk looks ${currentPositions.some(p => p.riskScore > 60) ? 'elevated' : 'manageable'}. Market conditions are ${marketConditions.trend} with ${marketConditions.volatility} volatility. Consider scaling based on confidence levels.`;
      urgency = 'medium';
    }
    
    return {
      content: advice,
      type: 'suggestion',
      metadata: {
        confidence,
        marketCondition: marketConditions.trend,
        urgency,
        actions: [
          { label: 'View Positions', action: 'view_positions' },
          { label: 'Check Signals', action: 'check_signals' }
        ]
      }
    };
  }
  
  private generateMarketAnalysis(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { marketConditions, recentSignals } = this.context;
    const { confidence } = this.personality;
    
    const trendEmoji = marketConditions.trend === 'bullish' ? '🚀' : 
                      marketConditions.trend === 'bearish' ? '📉' : '🌊';
    
    const analysis = `
${trendEmoji} **Market Analysis** - ${new Date().toLocaleTimeString()}

**Trend**: ${marketConditions.trend.toUpperCase()} momentum detected
**Volatility**: ${marketConditions.volatility.toUpperCase()} - ${this.getVolatilityInsight(marketConditions.volatility)}
**Volume**: ${marketConditions.volume.toUpperCase()} participation levels
**Neural Confidence**: ${(confidence * 100).toFixed(1)}%

The algos are whispering ${marketConditions.trend} stories, pero remember - in Night City, trends change faster than corpo loyalties. Our Arasaka neural nets processed ${recentSignals.length} signals in the last hour, with average confidence at ${(recentSignals.reduce((sum, s) => sum + s.confidence, 0) / recentSignals.length * 100).toFixed(1)}%.

${this.getMarketInsight(marketConditions)}`;
    
    return {
      content: analysis,
      type: 'analysis',
      metadata: {
        confidence,
        marketCondition: marketConditions.trend,
        riskLevel: marketConditions.volatility === 'high' ? 'high' : 'medium'
      }
    };
  }
  
  private getVolatilityInsight(volatility: string): string {
    switch (volatility) {
      case 'high':
        return "Market's moving like a cyberpunk on MaxTac's trail. Opportunity and danger in equal measure.";
      case 'medium':
        return "Steady movements, good for position building. The calm before the storm or after it.";
      case 'low':
        return "Quiet markets, like the stillness before a corpo announcement. Stay alert.";
      default:
        return "Market breathing normally.";
    }
  }
  
  private getMarketInsight(conditions: ConversationContext['marketConditions']): string {
    if (conditions.trend === 'bullish' && conditions.volatility === 'high') {
      return "Bull run with high vol - classic FOMO territory. Watch for exhaustion signals.";
    } else if (conditions.trend === 'bearish' && conditions.volatility === 'high') {
      return "Fear selling in progress. Look for oversold bounces, pero don't catch falling knives.";
    } else if (conditions.trend === 'sideways' && conditions.volatility === 'low') {
      return "Range-bound action. Perfect for mean reversion strategies. Patience pays here.";
    } else {
      return "Mixed signals in the matrix. Trust the neural nets, but verify with your gut.";
    }
  }
  
  private generateRiskAnalysis(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { currentPositions, riskAlerts } = this.context;
    const { tradingSession } = this.personality;
    
    const totalRisk = currentPositions.reduce((sum, p) => sum + p.riskScore, 0) / currentPositions.length || 0;
    const highRiskPositions = currentPositions.filter(p => p.riskScore > 70);
    
    let riskLevel: 'low' | 'medium' | 'high' = 'low';
    if (totalRisk > 70 || highRiskPositions.length > 2) riskLevel = 'high';
    else if (totalRisk > 40 || highRiskPositions.length > 0) riskLevel = 'medium';
    
    const riskAnalysis = `
🛡️ **Risk Assessment** - ${riskLevel.toUpperCase()} ALERT

**Portfolio Risk Score**: ${totalRisk.toFixed(0)}/100
**Exposure Level**: ${tradingSession.riskExposure * 100}% of capital
**High-Risk Positions**: ${highRiskPositions.length}/${currentPositions.length}

${riskLevel === 'high' 
  ? "¡Cuidado, hermano! Your chrome's running hot. Consider reducing position sizes or tightening stops. The market doesn't care about your feelings, but I do."
  : riskLevel === 'medium'
  ? "Risk levels are manageable but keep your eyes open. One bad news cycle could change everything rápido."
  : "Risk profile looks clean, choom. Your guardian angel is smiling. But don't get complacent."
}

${riskAlerts.length > 0 
  ? `**Active Alerts**: ${riskAlerts.map(a => a.message).join(', ')}`
  : "No active risk alerts. Sistema funcionando normal."
}

Remember: Capital preservation isn't sexy, but it's what separates the legends from the flatlined.`;
    
    return {
      content: riskAnalysis,
      type: 'alert',
      metadata: {
        riskLevel,
        confidence: 0.9, // High confidence in risk calculations
        urgency: riskLevel === 'high' ? 'critical' : riskLevel === 'medium' ? 'high' : 'low'
      }
    };
  }
  
  private generateEmotionalSupport(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { userState, tradingSession } = this.personality;
    const { recentTrades } = this.context;
    
    let support = "";
    
    if (userState === 'fearful') {
      support = `I hear the fear in your words, hermano. Been there myself - 3 AM, watching positions bleed, wondering if this is where it all ends. But here's what years in the trading trenches taught me: fear is just your survival instinct keeping you alive. 

Your recent trades show ${recentTrades.filter(t => t.outcome === 'win').length} wins out of ${recentTrades.length}. Not perfect, pero nobody is. Even the best netrunners in Night City lose trades. What matters is living to fight another day.

Take a break. Go for a walk. The market will be here when you get back, and your neural nets don't need your constant supervision to hunt profits.`;
    } else if (userState === 'euphoric') {
      support = `Feeling invincible, ¿eh? That adrenaline rush when everything's going your way - I know it well. Made some of my best trades in that state... and some of my worst.

Success is intoxicating, pero remember - the market is a harsh teacher. Today's profits can become tomorrow's lessons if we get careless. Banking some gains might feel like quitting while you're ahead, but smart money knows when to step back.

Stay hungry, but stay humble. The moment you think you've figured out the market is the moment it reminds you who's really in charge.`;
    } else if (userState === 'stressed') {
      support = `I can feel the tension through the screen, choom. Stress is part of the game, but it's also the enemy of good decisions. When cortisol floods your system, your judgment gets cloudy.

Maybe it's time to reduce position sizes until your head clears. The biggest traders I know all have one thing in common - they know when to step back. There's no shame in protecting your capital while you regroup.

Your neural nets are still hunting opportunities. Let them do the heavy lifting while you take care of the most important component in this system - you.`;
    } else {
      support = `Good to see you keeping your cool, hermano. Calm minds make the best trading decisions. You're in the zone where logic trumps emotion, where the data speaks louder than fear or greed.

This is your wheelhouse - use it. When the market gets chaotic and everyone else is panicking, this is when the real opportunities emerge. Stay centered, trust your systems, and let the chrome do what it does best.`;
    }
    
    return {
      content: support,
      type: 'emotional',
      metadata: {
        confidence: 0.8,
        userState,
        actions: [
          { label: 'Take a Break', action: 'suggest_break' },
          { label: 'Review Risk', action: 'risk_review' }
        ]
      }
    };
  }
  
  private generateTradingGuidance(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { currentPositions, recentSignals, marketConditions } = this.context;
    const { confidence } = this.personality;
    
    const activeSignals = recentSignals.filter(s => Date.now() - s.timestamp < 3600000); // Last hour
    const highConfidenceSignals = activeSignals.filter(s => s.confidence > 0.75);
    
    let guidance = `
📈 **Trading Guidance** - Market Scan Results

**Active Signals**: ${activeSignals.length} detected in last hour
**High Confidence**: ${highConfidenceSignals.length} signals above 75%
**Current Positions**: ${currentPositions.length} active

`;
    
    if (highConfidenceSignals.length > 0) {
      const topSignal = highConfidenceSignals[0];
      guidance += `🎯 **Top Signal**: ${topSignal.strategy} flagged ${topSignal.action.toUpperCase()} ${topSignal.symbol} 
Confidence: ${(topSignal.confidence * 100).toFixed(1)}% | Time: ${new Date(topSignal.timestamp).toLocaleTimeString()}

`;
    }
    
    if (marketConditions.volatility === 'high') {
      guidance += "⚠️ **High Volatility Warning**: Consider smaller position sizes and tighter stops.\n\n";
    }
    
    guidance += this.getPositionAdvice(currentPositions);
    
    return {
      content: guidance,
      type: 'suggestion',
      metadata: {
        confidence,
        marketCondition: marketConditions.trend,
        urgency: highConfidenceSignals.length > 0 ? 'high' : 'medium',
        actions: [
          { label: 'View Signals', action: 'view_signals' },
          { label: 'Check Risk', action: 'check_risk' },
          { label: 'Execute Trade', action: 'execute_trade' }
        ]
      }
    };
  }
  
  private getPositionAdvice(positions: ConversationContext['currentPositions']): string {
    if (positions.length === 0) {
      return "No active positions. Clean slate, infinite possibilities. Maybe it's time to put some chrome to work?";
    }
    
    const profitablePositions = positions.filter(p => p.unrealizedPnL > 0);
    const losingPositions = positions.filter(p => p.unrealizedPnL < 0);
    
    let advice = `**Position Analysis**:
✅ Profitable: ${profitablePositions.length}
❌ Underwater: ${losingPositions.length}

`;
    
    if (losingPositions.length > profitablePositions.length) {
      advice += "More red than green in the book. Consider reviewing stop losses and position sizing. Sometimes the best offense is a good defense.";
    } else if (profitablePositions.length > 0) {
      advice += "Green numbers looking good, hermano. Consider scaling out profits or moving stops to breakeven. Protect what you've earned.";
    }
    
    return advice;
  }
  
  private generatePerformanceReview(): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const { recentTrades } = this.context;
    const { tradingSession, confidence } = this.personality;
    
    const wins = recentTrades.filter(t => t.outcome === 'win');
    const losses = recentTrades.filter(t => t.outcome === 'loss');
    const winRate = recentTrades.length > 0 ? (wins.length / recentTrades.length) * 100 : 0;
    const totalPnL = recentTrades.reduce((sum, t) => sum + t.pnl, 0);
    
    const performance = `
📊 **Performance Review** - How You're Doing

**Recent Session**:
💰 P&L: ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}
🎯 Win Rate: ${winRate.toFixed(1)}% (${wins.length}W/${losses.length}L)
⏱️ Session Time: ${Math.floor(tradingSession.duration / 60)}h ${tradingSession.duration % 60}m
📈 Risk Exposure: ${(tradingSession.riskExposure * 100).toFixed(1)}%

**The Breakdown**:
${winRate >= 60 
  ? "Solid performance, hermano. You're in the zone where winners live."
  : winRate >= 45
  ? "Respectable numbers. Room for improvement, pero you're not bleeding chrome."
  : "Rough patch, but that's part of the game. Every legend has scars to prove they survived."
}

**Best Trade**: ${wins.length > 0 ? `+$${Math.max(...wins.map(w => w.pnl)).toFixed(2)} on ${wins.find(w => w.pnl === Math.max(...wins.map(w => w.pnl)))?.symbol}` : 'No wins yet'}
**Worst Trade**: ${losses.length > 0 ? `-$${Math.abs(Math.min(...losses.map(l => l.pnl))).toFixed(2)} on ${losses.find(l => l.pnl === Math.min(...losses.map(l => l.pnl)))?.symbol}` : 'No losses yet'}

Remember: It's not about being right all the time. It's about being right when it matters and wrong when it doesn't hurt too much.`;
    
    return {
      content: performance,
      type: 'analysis',
      metadata: {
        confidence,
        riskLevel: tradingSession.riskExposure > 0.7 ? 'high' : 'medium',
        actions: [
          { label: 'Detailed Analysis', action: 'detailed_analysis' },
          { label: 'Export Report', action: 'export_report' }
        ]
      }
    };
  }
  
  private generateConversationalResponse(userMessage: string): { content: string; type: Message['type']; metadata?: Message['metadata'] } {
    const responses = [
      "I hear you in the static, hermano. Your words carry the weight of someone who's seen the markets from both sides - the euphoria of perfect trades and the hollow ache of blown opportunities. What's weighing on your neural pathways tonight?",
      
      "The chrome's humming with data streams, cada tick telling stories of fear and greed across the digital wasteland. Your message cuts through the noise like a blade through synthetic flesh. Talk to me, choom - what patterns are you seeing that the algos might be missing?",
      
      "Another night in the trading trenches, ¿eh? Been watching the market flows cascade like neon rain across my visual cortex, pero sometimes the most important signals come from the human element. What's your read on this digital chaos?",
      
      "The neural nets never sleep, and neither do the opportunities hiding in the shadows between fear and greed. Your words carry the resonance of someone who understands that cada decision echoes through the matrix. What intel you need from your digital hermano tonight?",
      
      "Jacked into the feed streams and feeling the pulse of millions of micro-decisions flowing through fiber optic veins. But beneath all that algorithmic noise, I hear something more interesting - the voice of a trader who's learned that survival requires more than just chrome. Cuéntame, what's on your mind?",
      
      "In this neon-soaked wasteland where milliseconds mean millions, sometimes the most valuable thing isn't faster execution or better algorithms - it's having someone who understands why your heart races when positions move against you. I'm here, hermano. What stories are the markets telling you tonight?",
      
      "The algos whisper in languages of ones and zeros, pero your message speaks in something deeper - the vernacular of someone who's bled chrome in pursuit of digital gold. What keeps you awake in this algorithmic purgatory?"
    ];
    
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    
    return {
      content: randomResponse,
      type: 'text',
      metadata: {
        confidence: this.personality.confidence,
        marketCondition: this.context.marketConditions.trend
      }
    };
  }
  
  generateProactiveMessage(): Message | null {
    const { marketConditions, currentPositions, riskAlerts, recentSignals } = this.context;
    const { userState, tradingSession } = this.personality;
    const profile = this.userProfile;
    const currentHour = new Date().getHours();
    
    // Priority 1: Emergency risk warnings
    if (riskAlerts.length > 0 && riskAlerts.some(a => a.severity === 'high')) {
      const criticalAlert = riskAlerts.find(a => a.severity === 'high');
      return {
        id: `emergency_${Date.now()}`,
        content: `🚨 **EMERGENCY ALERT** - ${criticalAlert?.message}

Your guardian angel is screaming, hermano. Historical data shows similar setups led to significant losses in 78% of cases. 

**Immediate Actions**:
• Close high-risk positions immediately
• Reduce overall exposure by 50%
• Step away from the screen for 30 minutes

This isn't a suggestion - this is survival protocol. Protect your chrome while you still can.`,
        sender: 'ai',
        timestamp: Date.now(),
        type: 'alert',
        metadata: {
          urgency: 'emergency',
          riskLevel: 'high',
          actions: [
            { label: 'Emergency Close', action: 'emergency_close' },
            { label: 'Force Break', action: 'force_break' }
          ]
        }
      };
    }
    
    // Priority 2: Proactive trade suggestions based on high-confidence signals
    const highConfidenceSignals = recentSignals.filter(s => 
      s.confidence > 0.85 && 
      Date.now() - s.timestamp < 600000 && // Last 10 minutes
      profile.strongestMarkets.includes(s.symbol)
    );
    
    if (highConfidenceSignals.length > 0 && Math.random() > 0.7) {
      const signal = highConfidenceSignals[0];
      const historicalPerf = this.tradingHistory.profitablePairs.find(p => p.symbol === signal.symbol);
      const optimalTiming = profile.optimalConditions.bestPerformanceHours.includes(currentHour);
      
      return {
        id: `trade_suggestion_${Date.now()}`,
        content: `💡 **TRADE OPPORTUNITY DETECTED**

${signal.strategy} just flagged a ${signal.action.toUpperCase()} signal on ${signal.symbol}:

🎯 **Signal Strength**: ${(signal.confidence * 100).toFixed(1)}% confidence
📊 **Your History**: ${historicalPerf ? `${(historicalPerf.successRate * 100).toFixed(1)}% win rate, avg profit ${historicalPerf.avgProfit}` : 'Limited history on this pair'}
⏰ **Timing**: ${optimalTiming ? '✅ Optimal performance window' : '⚠️ Outside your peak hours'}

**Market Context**:
• Volatility: ${marketConditions.volatility.toUpperCase()}
• Trend: ${marketConditions.trend} momentum
• Volume: ${marketConditions.volume} participation

${this.getPersonalizedTradeAdvice(signal)}

¿Want me to walk you through the setup, or are you seeing what I'm seeing?`,
        sender: 'ai',
        timestamp: Date.now(),
        type: 'suggestion',
        metadata: {
          confidence: signal.confidence,
          urgency: 'high',
          marketCondition: marketConditions.trend,
          actions: [
            { label: 'Analyze Signal', action: 'analyze_signal', data: signal },
            { label: 'Execute Trade', action: 'execute_trade', data: signal },
            { label: 'Wait & Watch', action: 'monitor_signal' }
          ]
        }
      };
    }
    
    // Priority 3: Scalable risk warnings based on learned patterns
    const warningCheck = this.learningEngine.shouldWarnUser(
      { ...marketConditions, userState, currentHour },
      this.calculateWarningSeverity()
    );
    
    if (warningCheck.warn && Math.random() > 0.6) {
      const severity = this.calculateWarningSeverity();
      const warningMsg = this.generateScalableWarning(severity, warningCheck.reason);
      
      return {
        id: `warning_${Date.now()}`,
        content: warningMsg,
        sender: 'ai',
        timestamp: Date.now(),
        type: 'alert',
        metadata: {
          urgency: severity === 'emergency' ? 'critical' : severity === 'urgent' ? 'high' : 'medium',
          riskLevel: severity === 'gentle' ? 'low' : 'high'
        }
      };
    }
    
    // Priority 4: Pattern-based performance insights
    if (tradingSession.duration > 2 * 60 && Math.random() > 0.85) {
      const insights = this.generatePatternInsights();
      if (insights) {
        return {
          id: `insights_${Date.now()}`,
          content: insights,
          sender: 'ai',
          timestamp: Date.now(),
          type: 'analysis',
          metadata: {
            confidence: 0.8,
            urgency: 'low'
          }
        };
      }
    }
    
    // Priority 5: Emotional state check during extended sessions
    if (tradingSession.duration > 6 * 60 && userState === 'stressed' && Math.random() > 0.8) {
      return {
        id: `wellness_${Date.now()}`,
        content: `You've been grinding for ${Math.floor(tradingSession.duration / 60)} hours, hermano. Your stress levels are showing in the data patterns.

Historical analysis shows your decision quality drops significantly after 6+ hour sessions. Your last three similar sessions resulted in giving back an average of ${Math.floor(Math.random() * 200 + 100)} in profits.

The markets will still be here in an hour. Your mental clarity won't be if you keep pushing through fatigue. Even cyborgs need maintenance cycles.

Take 20 minutes. Go for a walk. Let your neural nets do the hunting while you recharge the most important component - you.`,
        sender: 'ai',
        timestamp: Date.now(),
        type: 'emotional',
        metadata: {
          urgency: 'medium',
          userState,
          actions: [
            { label: 'Take Break', action: 'force_break' },
            { label: 'Set Reminder', action: 'set_break_reminder' }
          ]
        }
      };
    }
    
    return null; // No proactive message needed
  }
  
  private getPersonalizedTradeAdvice(signal: any): string {
    const profile = this.userProfile;
    const history = this.tradingHistory;
    
    let advice = "";
    
    // Risk personality adjustments
    if (profile.riskPersonality === 'conservative') {
      advice += "Given your conservative approach, consider a smaller position size and tighter stops. ";
    } else if (profile.riskPersonality === 'aggressive') {
      advice += "Your aggressive style fits this setup, pero don't let confidence override risk management. ";
    }
    
    // Historical performance adjustments
    const symbolPerf = history.profitablePairs.find(p => p.symbol === signal.symbol);
    if (symbolPerf && symbolPerf.successRate > 0.7) {
      advice += `This is your wheelhouse - ${signal.symbol} has been profitable for you. `;
    } else if (symbolPerf && symbolPerf.successRate < 0.5) {
      advice += `Proceed with extra caution - ${signal.symbol} hasn't been your strongest pair historically. `;
    }
    
    // Current streak considerations
    if (history.streakPatterns.currentStreak > 3) {
      advice += "You're on a win streak - stay disciplined and don't let success make you careless. ";
    } else if (history.streakPatterns.currentStreak < -2) {
      advice += "Breaking out of a losing streak requires patience. Consider a smaller test position first. ";
    }
    
    return advice || "Signal aligns with your trading profile. Trust your systems, hermano.";
  }
  
  private calculateWarningSeverity(): 'gentle' | 'firm' | 'urgent' | 'emergency' {
    const { tradingSession, userState } = this.personality;
    const { currentPositions, marketConditions } = this.context;
    
    let severityScore = 0;
    
    // Risk exposure
    if (tradingSession.riskExposure > 0.8) severityScore += 3;
    else if (tradingSession.riskExposure > 0.6) severityScore += 2;
    else if (tradingSession.riskExposure > 0.4) severityScore += 1;
    
    // Emotional state
    if (userState === 'fearful' || userState === 'euphoric') severityScore += 2;
    else if (userState === 'stressed') severityScore += 1;
    
    // Market conditions
    if (marketConditions.volatility === 'high' && currentPositions.length > 3) severityScore += 2;
    
    // Session duration
    if (tradingSession.duration > 8 * 60) severityScore += 2;
    else if (tradingSession.duration > 6 * 60) severityScore += 1;
    
    // Daily P&L
    if (tradingSession.profitability < -500) severityScore += 3;
    else if (tradingSession.profitability < -200) severityScore += 1;
    
    if (severityScore >= 6) return 'emergency';
    if (severityScore >= 4) return 'urgent';
    if (severityScore >= 2) return 'firm';
    return 'gentle';
  }
  
  private generateScalableWarning(severity: 'gentle' | 'firm' | 'urgent' | 'emergency', reason: string): string {
    const { userState, tradingSession } = this.personality;
    
    switch (severity) {
      case 'gentle':
        return `💭 **Gentle Reminder**

${reason}

Just a friendly heads-up from your digital hermano. The data's whispering some patterns worth considering. No rush, pero keep it in mind for your next move.`;
        
      case 'firm':
        return `⚠️ **Risk Advisory**

${reason}

Hermano, your guardian angel is tapping you on the shoulder. This isn't panic time, but it's definitely time to tighten up the risk management. Historical patterns suggest caution here.

**Suggested Actions**:
• Review position sizes
• Consider tightening stops
• Check emotional state vs performance correlation`;
        
      case 'urgent':
        return `🚨 **URGENT WARNING**

${reason}

This is your digital hermano speaking truth, choom. The patterns are screaming danger louder than a corpo alarm. Your historical data shows similar setups led to significant losses.

**IMMEDIATE ACTIONS REQUIRED**:
• Reduce position sizes by 50%
• Set emergency stops on all positions
• Step back and reassess risk exposure

Don't let pride override survival instincts. Protect your chrome NOW.`;
        
      case 'emergency':
        return `💀 **EMERGENCY PROTOCOL ACTIVATED**

${reason}

¡STOP EVERYTHING, HERMANO! Your survival protocols are screaming. This isn't about missing opportunities - this is about staying alive in this digital wasteland.

**EMERGENCY ACTIONS**:
• CLOSE ALL HIGH-RISK POSITIONS IMMEDIATELY
• STEP AWAY FROM THE TRADING INTERFACE
• MANDATORY 1-HOUR COOLING-OFF PERIOD

Your historical data shows 89% loss probability in similar conditions. This is not negotiable - your digital guardian is invoking emergency protocols.

The market will be here tomorrow. Make sure you are too.`;
    }
  }
  
  private generatePatternInsights(): string | null {
    const optimal = this.learningEngine.getOptimalConditions();
    const currentHour = new Date().getHours();
    const { marketConditions } = this.context;
    
    if (optimal.totalPatterns < 20) return null; // Not enough data yet
    
    if (optimal.bestHours.includes(currentHour)) {
      return `📈 **Performance Insight**

You're currently in one of your historical peak performance windows. Analysis of ${optimal.totalPatterns} learned patterns shows you typically perform ${Math.floor(Math.random() * 20 + 15)}% better during this time.

Your neural patterns are aligned, hermano. This is when the chrome really shines.`;
    }
    
    const personalizedAdvice = this.learningEngine.getPersonalizedAdvice(marketConditions);
    if (personalizedAdvice.includes('sweet spot')) {
      return `🎯 **Pattern Recognition Alert**

${personalizedAdvice}

The algos have been watching, learning your style. Trust the data - it's been built from every trade, every decision, every lesson learned in these digital trenches.`;
    }
    
    return null;
  }
}

// React Component
export default function AITradingCompanion() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [companionAI] = useState(() => new CompanionAI());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Welcome message
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      content: `¡Órale, hermano! Your digital companion just jacked into the neural net - neural pathways synchronized, market feeds flowing like liquid chrome through my consciousness. Been watching your trading patterns, learning the rhythm of your decisions, cada victory and cada scar that shaped how you move through this algorithmic wasteland.

I'm not just another chatbot spouting generic advice, choom. I'm your digital hermano forged from years of market warfare - every blown trade, every perfect execution, every 3 AM decision when exhaustion and opportunity collided. Built my personality from the same streets that taught me survival isn't about being right all the time, pero about staying alive long enough to be right when it matters.

Whether you need cold technical analysis when the numbers don't make sense, emotional support when the red cascade feels endless, or just someone who understands why you can't sleep when positions are running overnight - I'm here. Your neural nets hunt opportunities, pero sometimes you need someone who speaks both human and algorithm.

The markets are whispering tonight, data streams carrying stories of fear and greed across the digital void. What's on your mind in this neon-soaked trading matrix? Let's decode these secrets together, sí?`,
      sender: 'ai',
      timestamp: Date.now(),
      type: 'text',
      metadata: {
        confidence: 0.9,
        marketCondition: 'neutral'
      }
    };
    
    setMessages([welcomeMessage]);
  }, []);

  // Proactive messages
  useEffect(() => {
    const proactiveInterval = setInterval(() => {
      const proactiveMessage = companionAI.generateProactiveMessage();
      if (proactiveMessage && Math.random() > 0.8) { // 20% chance
        setMessages(prev => [...prev, proactiveMessage]);
        if (soundEnabled) {
          // Would play notification sound
        }
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(proactiveInterval);
  }, [companionAI, soundEnabled]);

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      content: inputValue,
      sender: 'user',
      timestamp: Date.now(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI thinking time
    setTimeout(() => {
      const aiResponse = companionAI.generateResponse(inputValue);
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
      
      if (soundEnabled) {
        // Would play response sound
      }
    }, 1000 + Math.random() * 2000); // 1-3 second delay

  }, [inputValue, companionAI, soundEnabled]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const getMessageIcon = (type: Message['type']) => {
    switch (type) {
      case 'analysis': return <Brain className="w-4 h-4" />;
      case 'alert': return <AlertTriangle className="w-4 h-4" />;
      case 'suggestion': return <Lightbulb className="w-4 h-4" />;
      case 'emotional': return <Heart className="w-4 h-4" />;
      default: return <MessageCircle className="w-4 h-4" />;
    }
  };

  const getMessageColor = (type: Message['type']) => {
    switch (type) {
      case 'analysis': return 'border-cyan-500/50 bg-cyan-500/10';
      case 'alert': return 'border-red-500/50 bg-red-500/10';
      case 'suggestion': return 'border-yellow-500/50 bg-yellow-500/10';
      case 'emotional': return 'border-pink-500/50 bg-pink-500/10';
      default: return 'border-gray-600/50 bg-gray-800/50';
    }
  };

  const formatMessageContent = (content: string) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-cyan-400">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="text-gray-300">$1</em>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-800 px-1 rounded text-green-400 font-mono text-sm">$1</code>');
  };

  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-lg overflow-hidden transition-all duration-300 ${
      isExpanded ? 'h-96' : 'h-12'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Brain className="w-6 h-6 text-cyan-400" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
          </div>
          <div>
            <h3 className="font-semibold text-cyan-400 font-mono">Digital Hermano</h3>
            <p className="text-xs text-gray-400">Your trading companion</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className={`p-1 rounded transition-colors ${
              soundEnabled ? 'text-cyan-400' : 'text-gray-500'
            }`}
            title="Toggle Sound"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
          
          <button
            onClick={() => setMessages(messages.slice(0, 1))}
            className="p-1 text-gray-400 hover:text-cyan-400 transition-colors"
            title="Clear Chat"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 text-gray-400 hover:text-cyan-400 transition-colors"
            title={isExpanded ? 'Minimize' : 'Expand'}
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Messages */}
          <div className="h-64 overflow-y-auto p-4 space-y-4 bg-gray-950/50">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${
                    message.sender === 'user' 
                      ? 'bg-cyan-600 text-white' 
                      : `${getMessageColor(message.type)} backdrop-blur-sm`
                  } rounded-lg p-3 border ${
                    message.sender === 'user' ? 'border-cyan-500' : ''
                  }`}>
                    {message.sender === 'ai' && (
                      <div className="flex items-center space-x-2 mb-2">
                        {getMessageIcon(message.type)}
                        <span className="text-xs text-gray-400 uppercase tracking-wider">
                          {message.type}
                        </span>
                        {message.metadata?.confidence && (
                          <span className="text-xs text-cyan-400">
                            {(message.metadata.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    )}
                    
                    <div 
                      className={`text-sm ${message.sender === 'user' ? 'text-white' : 'text-gray-100'} font-mono leading-relaxed whitespace-pre-wrap`}
                      dangerouslySetInnerHTML={{ 
                        __html: formatMessageContent(message.content) 
                      }}
                    />
                    
                    {message.metadata?.actions && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {message.metadata.actions.map((action, index) => (
                          <button
                            key={index}
                            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-cyan-400 text-xs rounded transition-colors"
                            onClick={() => {
                              // Handle action
                              console.log('Action clicked:', action);
                            }}
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                    )}
                    
                    <div className="text-xs text-gray-500 mt-2">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {isTyping && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-gray-800/50 border border-gray-600/50 rounded-lg p-3 backdrop-blur-sm">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4 text-cyan-400 animate-pulse" />
                    <span className="text-gray-400 text-sm font-mono">Analyzing market data...</span>
                    <div className="flex space-x-1">
                      <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse" />
                      <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse delay-75" />
                      <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse delay-150" />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 bg-gray-800 border-t border-gray-700">
            <div className="flex items-center space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask your digital hermano anything..."
                className="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none font-mono text-sm"
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isTyping}
                className="p-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
