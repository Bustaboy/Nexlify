// frontend/src/lib/risk/RiskManagementEngine.ts
/**
 * Risk Management Engine - The Guardian Angel
 * Advanced position sizing, drawdown protection, and capital preservation
 * 
 * This is what keeps you alive when the market turns predator.
 * Been through the crash of '08, flash crashes, crypto winters...
 * Cada cicatrices on my portfolio taught me: you can make money back,
 * but you can't trade without capital.
 * 
 * The algos are ruthless hunters - this engine is your armor.
 */

import { EventEmitter } from 'events';

// Types - the rules that govern survival
interface RiskProfile {
  maxDailyLoss: number;           // Max loss per day (USD)
  maxDailyLossPercent: number;    // Max loss per day (% of equity)
  maxPositionSize: number;        // Max position size (USD)
  maxPositionPercent: number;     // Max position size (% of equity)
  maxDrawdown: number;            // Max total drawdown (USD)
  maxDrawdownPercent: number;     // Max total drawdown (%)
  maxConcurrentPositions: number; // Max open positions
  maxLeverage: number;            // Max leverage multiplier
  maxCorrelation: number;         // Max correlation between positions (0-1)
  stopLossPercent: number;        // Default stop loss (%)
  takeProfitRatio: number;        // Risk:Reward ratio for take profit
}

interface Position {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice: number;
  stopLoss?: number;
  takeProfit?: number;
  unrealizedPnL: number;
  realizedPnL: number;
  timestamp: number;
  strategy: string;
  leverage: number;
}

interface RiskMetrics {
  currentDrawdown: number;
  currentDrawdownPercent: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalEquity: number;
  availableCapital: number;
  usedCapital: number;
  portfolioVaR: number;           // Value at Risk
  sharpeRatio: number;
  maxConsecutiveLosses: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  calmarRatio: number;
}

interface RiskAlert {
  id: string;
  type: 'warning' | 'critical' | 'emergency';
  message: string;
  metric: string;
  currentValue: number;
  threshold: number;
  timestamp: number;
  positionId?: string;
  actions: string[];
}

interface OrderValidation {
  isValid: boolean;
  reason?: string;
  suggestedSize?: number;
  suggestedStopLoss?: number;
  suggestedTakeProfit?: number;
  riskScore: number; // 0-100, 100 = maximum risk
}

// Risk calculation utilities - the mathematics of survival
class RiskCalculator {
  /**
   * Calculate position size based on risk tolerance
   * Kelly Criterion with safety modifications
   */
  static calculateOptimalPositionSize(
    equity: number,
    entryPrice: number,
    stopLossPrice: number,
    winRate: number,
    avgWin: number,
    avgLoss: number,
    maxRiskPercent: number = 2
  ): number {
    // Risk per share
    const riskPerShare = Math.abs(entryPrice - stopLossPrice);
    if (riskPerShare === 0) return 0;

    // Maximum risk amount
    const maxRiskAmount = equity * (maxRiskPercent / 100);
    
    // Basic position size
    const basicPositionSize = maxRiskAmount / riskPerShare;

    // Kelly Criterion adjustment
    const winLossRatio = avgWin / Math.abs(avgLoss);
    const kellyPercent = winRate - ((1 - winRate) / winLossRatio);
    
    // Apply Kelly with 25% safety factor (never go full Kelly)
    const kellyAdjusted = Math.max(0, kellyPercent * 0.25);
    const kellyPositionValue = equity * kellyAdjusted;
    const kellyPositionSize = kellyPositionValue / entryPrice;

    // Return the more conservative of the two
    return Math.min(basicPositionSize, kellyPositionSize);
  }

  /**
   * Calculate Value at Risk (VaR) for portfolio
   */
  static calculatePortfolioVaR(
    positions: Position[],
    confidence: number = 0.95,
    timeHorizon: number = 1
  ): number {
    if (positions.length === 0) return 0;

    // Historical simulation method
    const portfolioValues = positions.map(p => p.size * p.currentPrice);
    const totalValue = portfolioValues.reduce((sum, val) => sum + val, 0);

    // Simplified VaR calculation (in production, use Monte Carlo)
    const returns = positions.map(p => (p.currentPrice - p.entryPrice) / p.entryPrice);
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);

    // Z-score for confidence level
    const zScore = confidence === 0.95 ? 1.645 : confidence === 0.99 ? 2.326 : 1.96;
    
    return totalValue * volatility * zScore * Math.sqrt(timeHorizon);
  }

  /**
   * Calculate correlation between two symbols
   */
  static calculateCorrelation(prices1: number[], prices2: number[]): number {
    if (prices1.length !== prices2.length || prices1.length < 2) return 0;

    const returns1 = this.calculateReturns(prices1);
    const returns2 = this.calculateReturns(prices2);

    const mean1 = returns1.reduce((sum, r) => sum + r, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, r) => sum + r, 0) / returns2.length;

    let numerator = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;

    for (let i = 0; i < returns1.length; i++) {
      const diff1 = returns1[i] - mean1;
      const diff2 = returns2[i] - mean2;
      numerator += diff1 * diff2;
      sumSq1 += diff1 * diff1;
      sumSq2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sumSq1 * sumSq2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Calculate returns array from prices
   */
  private static calculateReturns(prices: number[]): number[] {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
  }
}

/**
 * Advanced Risk Management Engine
 * Your personal bodyguard in the trading matrix
 */
export class RiskManagementEngine extends EventEmitter {
  private riskProfile: RiskProfile;
  private positions: Map<string, Position> = new Map();
  private dailyTrades: Array<{ timestamp: number; pnl: number }> = [];
  private equityHistory: Array<{ timestamp: number; equity: number }> = [];
  private alerts: Map<string, RiskAlert> = new Map();
  private isEmergencyMode = false;
  private dailyEquityStart = 0;

  // Performance tracking
  private consecutiveLosses = 0;
  private consecutiveWins = 0;
  private totalTrades = 0;
  private winningTrades = 0;

  constructor(initialProfile: Partial<RiskProfile> = {}) {
    super();
    
    // Default risk profile - conservative but profitable
    this.riskProfile = {
      maxDailyLoss: 1000,                 // $1000 max daily loss
      maxDailyLossPercent: 5,             // 5% max daily loss
      maxPositionSize: 5000,              // $5000 max position
      maxPositionPercent: 10,             // 10% max position size
      maxDrawdown: 5000,                  // $5000 max drawdown
      maxDrawdownPercent: 20,             // 20% max drawdown
      maxConcurrentPositions: 5,          // 5 positions max
      maxLeverage: 3,                     // 3x max leverage
      maxCorrelation: 0.7,                // 70% max correlation
      stopLossPercent: 2,                 // 2% default stop loss
      takeProfitRatio: 2,                 // 2:1 reward:risk
      ...initialProfile
    };

    console.log('🛡️ Risk management engine initialized - Your guardian angel is online');
  }

  /**
   * Validate an order before execution - the gatekeeper function
   */
  validateOrder(
    symbol: string,
    side: 'buy' | 'sell',
    size: number,
    price: number,
    stopLoss?: number,
    takeProfit?: number,
    leverage: number = 1
  ): OrderValidation {
    const equity = this.getCurrentEquity();
    const positionValue = size * price * leverage;
    const currentMetrics = this.calculateCurrentMetrics();

    // Check 1: Emergency mode - no new positions
    if (this.isEmergencyMode) {
      return {
        isValid: false,
        reason: "🚨 Emergency mode active - no new positions allowed",
        riskScore: 100
      };
    }

    // Check 2: Daily loss limit
    if (currentMetrics.dailyPnL <= -this.riskProfile.maxDailyLoss) {
      return {
        isValid: false,
        reason: `📉 Daily loss limit reached: $${Math.abs(currentMetrics.dailyPnL).toFixed(2)}`,
        riskScore: 95
      };
    }

    // Check 3: Daily loss percentage
    if (currentMetrics.dailyPnLPercent <= -this.riskProfile.maxDailyLossPercent) {
      return {
        isValid: false,
        reason: `📉 Daily loss percentage limit reached: ${Math.abs(currentMetrics.dailyPnLPercent).toFixed(2)}%`,
        riskScore: 95
      };
    }

    // Check 4: Position size limits
    if (positionValue > this.riskProfile.maxPositionSize) {
      const suggestedSize = this.riskProfile.maxPositionSize / (price * leverage);
      return {
        isValid: false,
        reason: `💰 Position too large: $${positionValue.toFixed(2)} > $${this.riskProfile.maxPositionSize}`,
        suggestedSize,
        riskScore: 80
      };
    }

    // Check 5: Position percentage limits
    const positionPercent = (positionValue / equity) * 100;
    if (positionPercent > this.riskProfile.maxPositionPercent) {
      const suggestedSize = (equity * this.riskProfile.maxPositionPercent / 100) / (price * leverage);
      return {
        isValid: false,
        reason: `📊 Position percentage too high: ${positionPercent.toFixed(2)}% > ${this.riskProfile.maxPositionPercent}%`,
        suggestedSize,
        riskScore: 75
      };
    }

    // Check 6: Maximum concurrent positions
    if (this.positions.size >= this.riskProfile.maxConcurrentPositions) {
      return {
        isValid: false,
        reason: `🔢 Too many positions: ${this.positions.size} >= ${this.riskProfile.maxConcurrentPositions}`,
        riskScore: 70
      };
    }

    // Check 7: Leverage limits
    if (leverage > this.riskProfile.maxLeverage) {
      return {
        isValid: false,
        reason: `⚡ Leverage too high: ${leverage}x > ${this.riskProfile.maxLeverage}x`,
        riskScore: 85
      };
    }

    // Check 8: Drawdown limits
    if (currentMetrics.currentDrawdown >= this.riskProfile.maxDrawdown) {
      return {
        isValid: false,
        reason: `📉 Maximum drawdown reached: $${currentMetrics.currentDrawdown.toFixed(2)}`,
        riskScore: 90
      };
    }

    // Check 9: Stop loss validation
    if (stopLoss) {
      const stopLossPercent = Math.abs((price - stopLoss) / price) * 100;
      if (stopLossPercent > this.riskProfile.stopLossPercent * 3) {
        const suggestedStopLoss = side === 'buy' 
          ? price * (1 - this.riskProfile.stopLossPercent / 100)
          : price * (1 + this.riskProfile.stopLossPercent / 100);
        
        return {
          isValid: false,
          reason: `🛑 Stop loss too wide: ${stopLossPercent.toFixed(2)}%`,
          suggestedStopLoss,
          riskScore: 60
        };
      }
    }

    // Check 10: Correlation limits (simplified)
    const correlationRisk = this.calculateCorrelationRisk(symbol);
    if (correlationRisk > this.riskProfile.maxCorrelation) {
      return {
        isValid: false,
        reason: `🔗 High correlation risk: ${(correlationRisk * 100).toFixed(1)}%`,
        riskScore: 65
      };
    }

    // Order is valid - calculate risk score
    let riskScore = 0;
    riskScore += positionPercent * 2; // Position size impact
    riskScore += Math.abs(currentMetrics.dailyPnLPercent) * 2; // Daily loss impact
    riskScore += (leverage - 1) * 10; // Leverage impact
    riskScore += correlationRisk * 30; // Correlation impact

    // Suggest optimal stop loss if not provided
    let suggestedStopLoss = stopLoss;
    if (!stopLoss) {
      suggestedStopLoss = side === 'buy'
        ? price * (1 - this.riskProfile.stopLossPercent / 100)
        : price * (1 + this.riskProfile.stopLossPercent / 100);
    }

    // Suggest take profit based on risk:reward ratio
    let suggestedTakeProfit = takeProfit;
    if (!takeProfit && suggestedStopLoss) {
      const riskAmount = Math.abs(price - suggestedStopLoss);
      suggestedTakeProfit = side === 'buy'
        ? price + (riskAmount * this.riskProfile.takeProfitRatio)
        : price - (riskAmount * this.riskProfile.takeProfitRatio);
    }

    return {
      isValid: true,
      suggestedStopLoss,
      suggestedTakeProfit,
      riskScore: Math.min(100, Math.max(0, riskScore))
    };
  }

  /**
   * Add a new position to risk monitoring
   */
  addPosition(position: Position): void {
    this.positions.set(position.id, position);
    
    console.log(`📈 Position added: ${position.symbol} ${position.side} ${position.size} @ ${position.entryPrice}`);
    
    // Check if this triggers any risk alerts
    this.checkRiskAlerts();
    
    this.emit('positionAdded', position);
  }

  /**
   * Update existing position
   */
  updatePosition(positionId: string, updates: Partial<Position>): void {
    const position = this.positions.get(positionId);
    if (!position) return;

    const updatedPosition = { ...position, ...updates };
    this.positions.set(positionId, updatedPosition);

    // Check for stop loss or take profit hits
    if (position.stopLoss && position.currentPrice <= position.stopLoss && position.side === 'buy') {
      this.triggerStopLoss(positionId);
    } else if (position.stopLoss && position.currentPrice >= position.stopLoss && position.side === 'sell') {
      this.triggerStopLoss(positionId);
    }

    if (position.takeProfit && position.currentPrice >= position.takeProfit && position.side === 'buy') {
      this.triggerTakeProfit(positionId);
    } else if (position.takeProfit && position.currentPrice <= position.takeProfit && position.side === 'sell') {
      this.triggerTakeProfit(positionId);
    }

    this.checkRiskAlerts();
    this.emit('positionUpdated', updatedPosition);
  }

  /**
   * Close a position and record the trade
   */
  closePosition(positionId: string, closingPrice: number, reason: string = 'manual'): void {
    const position = this.positions.get(positionId);
    if (!position) return;

    // Calculate final P&L
    const multiplier = position.side === 'buy' ? 1 : -1;
    const pnl = (closingPrice - position.entryPrice) * position.size * multiplier;

    // Record the trade
    this.recordTrade(pnl);

    // Remove from active positions
    this.positions.delete(positionId);

    console.log(`📊 Position closed: ${position.symbol} P&L: $${pnl.toFixed(2)} (${reason})`);

    this.emit('positionClosed', { position, pnl, reason });
  }

  /**
   * Check and trigger risk alerts
   */
  private checkRiskAlerts(): void {
    const metrics = this.calculateCurrentMetrics();
    const alerts: RiskAlert[] = [];

    // Daily loss alerts
    if (metrics.dailyPnLPercent <= -this.riskProfile.maxDailyLossPercent * 0.8) {
      alerts.push({
        id: 'daily-loss-warning',
        type: 'warning',
        message: `Approaching daily loss limit: ${Math.abs(metrics.dailyPnLPercent).toFixed(2)}%`,
        metric: 'dailyPnLPercent',
        currentValue: metrics.dailyPnLPercent,
        threshold: -this.riskProfile.maxDailyLossPercent,
        timestamp: Date.now(),
        actions: ['Consider reducing position sizes', 'Review trading strategy']
      });
    }

    // Drawdown alerts
    if (metrics.currentDrawdownPercent >= this.riskProfile.maxDrawdownPercent * 0.8) {
      alerts.push({
        id: 'drawdown-warning',
        type: 'critical',
        message: `High drawdown detected: ${metrics.currentDrawdownPercent.toFixed(2)}%`,
        metric: 'drawdown',
        currentValue: metrics.currentDrawdownPercent,
        threshold: this.riskProfile.maxDrawdownPercent,
        timestamp: Date.now(),
        actions: ['Reduce position sizes', 'Review risk management', 'Consider stopping trading']
      });
    }

    // VaR alerts
    if (metrics.portfolioVaR > metrics.totalEquity * 0.1) {
      alerts.push({
        id: 'var-warning',
        type: 'warning',
        message: `High portfolio risk detected: VaR $${metrics.portfolioVaR.toFixed(2)}`,
        metric: 'var',
        currentValue: metrics.portfolioVaR,
        threshold: metrics.totalEquity * 0.1,
        timestamp: Date.now(),
        actions: ['Diversify positions', 'Reduce correlation']
      });
    }

    // Process alerts
    for (const alert of alerts) {
      this.alerts.set(alert.id, alert);
      this.emit('riskAlert', alert);
      
      if (alert.type === 'critical') {
        console.warn(`🚨 CRITICAL RISK ALERT: ${alert.message}`);
      }
    }

    // Emergency mode check
    if (metrics.dailyPnLPercent <= -this.riskProfile.maxDailyLossPercent ||
        metrics.currentDrawdownPercent >= this.riskProfile.maxDrawdownPercent) {
      this.activateEmergencyMode();
    }
  }

  /**
   * Activate emergency mode - stop all new trades
   */
  private activateEmergencyMode(): void {
    if (this.isEmergencyMode) return;

    this.isEmergencyMode = true;
    console.error('🚨 EMERGENCY MODE ACTIVATED - All new trading halted');
    
    this.emit('emergencyMode', {
      reason: 'Risk limits exceeded',
      timestamp: Date.now(),
      metrics: this.calculateCurrentMetrics()
    });
  }

  /**
   * Deactivate emergency mode (manual override)
   */
  deactivateEmergencyMode(): void {
    this.isEmergencyMode = false;
    console.log('✅ Emergency mode deactivated - Trading resumed');
    this.emit('emergencyModeDeactivated');
  }

  /**
   * Trigger stop loss
   */
  private triggerStopLoss(positionId: string): void {
    const position = this.positions.get(positionId);
    if (!position || !position.stopLoss) return;

    console.log(`🛑 Stop loss triggered: ${position.symbol} @ ${position.stopLoss}`);
    this.closePosition(positionId, position.stopLoss, 'stop_loss');
  }

  /**
   * Trigger take profit
   */
  private triggerTakeProfit(positionId: string): void {
    const position = this.positions.get(positionId);
    if (!position || !position.takeProfit) return;

    console.log(`🎯 Take profit triggered: ${position.symbol} @ ${position.takeProfit}`);
    this.closePosition(positionId, position.takeProfit, 'take_profit');
  }

  /**
   * Record a completed trade for performance tracking
   */
  private recordTrade(pnl: number): void {
    this.dailyTrades.push({
      timestamp: Date.now(),
      pnl
    });

    this.totalTrades++;
    
    if (pnl > 0) {
      this.winningTrades++;
      this.consecutiveWins++;
      this.consecutiveLosses = 0;
    } else {
      this.consecutiveLosses++;
      this.consecutiveWins = 0;
    }

    // Clean up old trades (keep only today's)
    const todayStart = new Date().setHours(0, 0, 0, 0);
    this.dailyTrades = this.dailyTrades.filter(trade => trade.timestamp >= todayStart);
  }

  /**
   * Calculate correlation risk for a symbol
   */
  private calculateCorrelationRisk(symbol: string): number {
    // Simplified correlation calculation
    // In production, use actual price history correlation
    const existingPositions = Array.from(this.positions.values());
    const sameSymbolPositions = existingPositions.filter(p => p.symbol === symbol);
    
    if (sameSymbolPositions.length > 0) return 1.0; // Same symbol = perfect correlation
    
    // For different symbols, return a basic correlation estimate
    // This would be replaced with actual correlation calculation
    return existingPositions.length > 0 ? 0.3 : 0;
  }

  /**
   * Get current equity value
   */
  private getCurrentEquity(): number {
    // In production, this would come from the trading store
    return Array.from(this.positions.values())
      .reduce((total, pos) => total + (pos.size * pos.currentPrice), 50000); // Mock equity
  }

  /**
   * Calculate comprehensive risk metrics
   */
  calculateCurrentMetrics(): RiskMetrics {
    const positions = Array.from(this.positions.values());
    const equity = this.getCurrentEquity();
    
    // Daily P&L calculation
    const todayStart = new Date().setHours(0, 0, 0, 0);
    const todayTrades = this.dailyTrades.filter(t => t.timestamp >= todayStart);
    const dailyPnL = todayTrades.reduce((sum, trade) => sum + trade.pnl, 0);
    
    // Unrealized P&L from current positions
    const unrealizedPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
    const totalDailyPnL = dailyPnL + unrealizedPnL;
    
    // Calculate drawdown
    const equityPeak = Math.max(...this.equityHistory.map(h => h.equity), equity);
    const currentDrawdown = equityPeak - equity;
    const currentDrawdownPercent = equityPeak > 0 ? (currentDrawdown / equityPeak) * 100 : 0;
    
    // Portfolio VaR
    const portfolioVaR = RiskCalculator.calculatePortfolioVaR(positions);
    
    // Used capital
    const usedCapital = positions.reduce((sum, pos) => sum + (pos.size * pos.currentPrice), 0);
    
    // Performance metrics
    const winRate = this.totalTrades > 0 ? (this.winningTrades / this.totalTrades) * 100 : 0;
    const allTrades = this.dailyTrades.map(t => t.pnl);
    const wins = allTrades.filter(pnl => pnl > 0);
    const losses = allTrades.filter(pnl => pnl < 0);
    
    const avgWin = wins.length > 0 ? wins.reduce((sum, w) => sum + w, 0) / wins.length : 0;
    const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((sum, l) => sum + l, 0) / losses.length) : 0;
    
    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;
    
    return {
      currentDrawdown,
      currentDrawdownPercent,
      dailyPnL: totalDailyPnL,
      dailyPnLPercent: equity > 0 ? (totalDailyPnL / equity) * 100 : 0,
      totalEquity: equity,
      availableCapital: equity - usedCapital,
      usedCapital,
      portfolioVaR,
      sharpeRatio: 0, // Would calculate with actual returns
      maxConsecutiveLosses: this.consecutiveLosses,
      winRate,
      avgWin,
      avgLoss,
      profitFactor,
      calmarRatio: 0 // Would calculate with actual returns
    };
  }

  /**
   * Update risk profile
   */
  updateRiskProfile(updates: Partial<RiskProfile>): void {
    this.riskProfile = { ...this.riskProfile, ...updates };
    console.log('🛡️ Risk profile updated');
    this.emit('riskProfileUpdated', this.riskProfile);
  }

  /**
   * Get current risk profile
   */
  getRiskProfile(): RiskProfile {
    return { ...this.riskProfile };
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): RiskAlert[] {
    return Array.from(this.alerts.values());
  }

  /**
   * Clear alert
   */
  clearAlert(alertId: string): void {
    this.alerts.delete(alertId);
    this.emit('alertCleared', alertId);
  }

  /**
   * Emergency stop all positions
   */
  emergencyStopAll(): void {
    console.error('🚨 EMERGENCY STOP - Closing all positions');
    
    for (const [positionId, position] of this.positions) {
      this.closePosition(positionId, position.currentPrice, 'emergency_stop');
    }
    
    this.activateEmergencyMode();
    this.emit('emergencyStop');
  }
}

// Export singleton instance
export const riskEngine = new RiskManagementEngine();

// Utility functions for React components
export const riskUtils = {
  /**
   * Format risk score color
   */
  getRiskColor(score: number): string {
    if (score >= 80) return 'text-red-500';
    if (score >= 60) return 'text-orange-500';
    if (score >= 40) return 'text-yellow-500';
    return 'text-green-500';
  },

  /**
   * Get risk level text
   */
  getRiskLevel(score: number): string {
    if (score >= 80) return 'HIGH RISK';
    if (score >= 60) return 'Medium Risk';
    if (score >= 40) return 'Low Risk';
    return 'Safe';
  },

  /**
   * Format currency with color based on value
   */
  formatPnL(value: number): { text: string; color: string } {
    const color = value >= 0 ? 'text-green-400' : 'text-red-400';
    const text = value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
    return { text, color };
  }
};
