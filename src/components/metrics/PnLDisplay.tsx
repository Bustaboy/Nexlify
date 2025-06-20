// src/components/metrics/PnLDisplay.tsx
// NEXLIFY PNL DISPLAY - Where fortunes are measured in pixels
// Last sync: 2025-06-19 | "Green means go, red means stop, but traders never listen"

import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  Percent,
  Calendar,
  Clock,
  Target,
  AlertTriangle,
  Trophy,
  Skull,
  Zap,
  Eye,
  EyeOff,
  ChevronUp,
  ChevronDown,
  BarChart3,
  Activity
} from 'lucide-react';

import { useTradingStore } from '@/stores/tradingStore';
import { useMarketStore } from '@/stores/marketStore';

interface PnLDisplayProps {
  timeframe?: 'day' | 'week' | 'month' | 'all';
  showBreakdown?: boolean;
  compact?: boolean;
  animate?: boolean;
  hideValues?: boolean;
  onTimeframeChange?: (timeframe: string) => void;
}

interface PnLMetrics {
  realized: number;
  unrealized: number;
  total: number;
  percentage: number;
  highWaterMark: number;
  drawdown: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  bestTrade: { symbol: string; pnl: number; date: Date };
  worstTrade: { symbol: string; pnl: number; date: Date };
  streak: { type: 'win' | 'loss'; count: number };
}

/**
 * PNL DISPLAY - The truth in numbers
 * 
 * Built this after watching "Never Wrong" Nancy stare at a -$50k
 * unrealized loss for three days. She kept saying "It's not a loss
 * until you sell." The liquidation engine disagreed.
 * 
 * This component shows you the truth, whether you want to see it or not:
 * - Realized vs Unrealized (because hope isn't a strategy)
 * - Win rate and profit factor (the metrics that matter)
 * - Drawdown from peak (how much pain you've endured)
 * - Best/worst trades (learn from both)
 * 
 * Remember: The market doesn't care about your feelings, only your PnL.
 */
export const PnLDisplay = ({
  timeframe = 'day',
  showBreakdown = true,
  compact = false,
  animate = true,
  hideValues = false,
  onTimeframeChange
}: PnLDisplayProps) => {
  // State management
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [showDetails, setShowDetails] = useState(false);
  const [previousTotal, setPreviousTotal] = useState(0);
  const [flashColor, setFlashColor] = useState<'green' | 'red' | null>(null);
  
  // Store connections
  const { 
    pnlHistory, 
    positions, 
    trades,
    calculatePnLMetrics 
  } = useTradingStore();
  
  /**
   * Calculate PnL metrics for selected timeframe
   * 
   * This is where dreams meet mathematics. Every trade, every decision,
   * reduced to cold, hard numbers.
   */
  const metrics = useMemo((): PnLMetrics => {
    const now = new Date();
    let startDate: Date;
    
    switch (selectedTimeframe) {
      case 'day':
        startDate = new Date(now.setHours(0, 0, 0, 0));
        break;
      case 'week':
        startDate = new Date(now.setDate(now.getDate() - 7));
        break;
      case 'month':
        startDate = new Date(now.setMonth(now.getMonth() - 1));
        break;
      default:
        startDate = new Date(0); // All time
    }
    
    // Filter trades by timeframe
    const periodTrades = trades.filter(t => new Date(t.closeTime) >= startDate);
    
    // Calculate realized PnL
    const realized = periodTrades.reduce((sum, t) => sum + t.realizedPnL, 0);
    
    // Calculate unrealized PnL from open positions
    const unrealized = Object.values(positions).reduce((sum, p) => sum + p.unrealizedPnL, 0);
    
    // Total PnL
    const total = realized + unrealized;
    
    // Win rate calculation
    const winningTrades = periodTrades.filter(t => t.realizedPnL > 0);
    const losingTrades = periodTrades.filter(t => t.realizedPnL < 0);
    const winRate = periodTrades.length > 0 
      ? (winningTrades.length / periodTrades.length) * 100 
      : 0;
    
    // Average win/loss
    const avgWin = winningTrades.length > 0
      ? winningTrades.reduce((sum, t) => sum + t.realizedPnL, 0) / winningTrades.length
      : 0;
    const avgLoss = losingTrades.length > 0
      ? Math.abs(losingTrades.reduce((sum, t) => sum + t.realizedPnL, 0) / losingTrades.length)
      : 0;
    
    // Profit factor
    const totalWins = winningTrades.reduce((sum, t) => sum + t.realizedPnL, 0);
    const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + t.realizedPnL, 0));
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;
    
    // Find best and worst trades
    const sortedTrades = [...periodTrades].sort((a, b) => b.realizedPnL - a.realizedPnL);
    const bestTrade = sortedTrades[0] || { symbol: 'N/A', pnl: 0, date: new Date() };
    const worstTrade = sortedTrades[sortedTrades.length - 1] || { symbol: 'N/A', pnl: 0, date: new Date() };
    
    // Calculate streak
    let currentStreak = { type: 'win' as 'win' | 'loss', count: 0 };
    for (let i = periodTrades.length - 1; i >= 0; i--) {
      const trade = periodTrades[i];
      const isWin = trade.realizedPnL > 0;
      
      if (i === periodTrades.length - 1) {
        currentStreak.type = isWin ? 'win' : 'loss';
        currentStreak.count = 1;
      } else if ((isWin && currentStreak.type === 'win') || (!isWin && currentStreak.type === 'loss')) {
        currentStreak.count++;
      } else {
        break;
      }
    }
    
    // Simplified calculations for demo
    const percentage = realized > 0 ? (total / realized - 1) * 100 : 0;
    const highWaterMark = Math.max(...pnlHistory.map(p => p.total), total);
    const drawdown = highWaterMark > 0 ? ((highWaterMark - total) / highWaterMark) * 100 : 0;
    const sharpeRatio = 1.5; // Placeholder - would calculate from returns
    
    return {
      realized,
      unrealized,
      total,
      percentage,
      highWaterMark,
      drawdown,
      winRate,
      avgWin,
      avgLoss,
      profitFactor,
      sharpeRatio,
      bestTrade,
      worstTrade,
      streak: currentStreak
    };
  }, [selectedTimeframe, trades, positions, pnlHistory]);
  
  /**
   * Flash animation on PnL change
   * 
   * Visual feedback matters. Green flash = dopamine hit.
   * Red flash = cortisol spike. Use responsibly.
   */
  useEffect(() => {
    if (animate && metrics.total !== previousTotal) {
      setFlashColor(metrics.total > previousTotal ? 'green' : 'red');
      setPreviousTotal(metrics.total);
      
      const timer = setTimeout(() => setFlashColor(null), 500);
      return () => clearTimeout(timer);
    }
  }, [metrics.total, previousTotal, animate]);
  
  /**
   * Format PnL with color and sign
   */
  const formatPnL = (value: number, showSign = true): string => {
    if (hideValues) return '•••••';
    const formatted = Math.abs(value).toFixed(2);
    return showSign && value > 0 ? `+$${formatted}` : `$${formatted}`;
  };
  
  /**
   * Get color based on value
   */
  const getPnLColor = (value: number): string => {
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-400';
  };
  
  /**
   * Get emoji based on performance
   * 
   * Yes, we use emojis. Deal with it. Sometimes a 🚀 says more
   * than a thousand numbers.
   */
  const getPerformanceEmoji = (): string => {
    if (metrics.total > 1000) return '🚀';
    if (metrics.total > 500) return '💰';
    if (metrics.total > 0) return '📈';
    if (metrics.total > -500) return '📉';
    if (metrics.total > -1000) return '💸';
    return '☠️';
  };
  
  // Timeframe options
  const timeframes = [
    { value: 'day', label: 'Today', icon: Calendar },
    { value: 'week', label: '7 Days', icon: Calendar },
    { value: 'month', label: '30 Days', icon: Calendar },
    { value: 'all', label: 'All Time', icon: Clock }
  ];
  
  return (
    <div className={`
      bg-gray-900/50 border border-cyan-900/30 rounded-lg
      ${compact ? 'p-3' : 'p-4'}
      ${flashColor ? `ring-2 ring-${flashColor}-500/50` : ''}
      transition-all duration-300
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <DollarSign className="w-5 h-5" />
          P&L METRICS
          <span className="text-2xl">{getPerformanceEmoji()}</span>
        </h3>
        
        <div className="flex items-center gap-2">
          {/* Hide values toggle */}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="p-1 rounded hover:bg-gray-800 transition-colors"
            title={showDetails ? "Hide details" : "Show details"}
          >
            {showDetails ? 
              <ChevronUp className="w-4 h-4 text-cyan-400" /> : 
              <ChevronDown className="w-4 h-4 text-gray-400" />
            }
          </button>
        </div>
      </div>
      
      {/* Timeframe selector */}
      <div className="flex gap-1 mb-4">
        {timeframes.map(tf => (
          <button
            key={tf.value}
            onClick={() => {
              setSelectedTimeframe(tf.value as any);
              onTimeframeChange?.(tf.value);
            }}
            className={`
              flex-1 py-2 px-3 rounded text-sm font-medium transition-all
              ${selectedTimeframe === tf.value
                ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}
            `}
          >
            {tf.label}
          </button>
        ))}
      </div>
      
      {/* Main PnL Display */}
      <div className="text-center mb-6">
        <motion.div
          animate={flashColor ? { scale: [1, 1.05, 1] } : {}}
          transition={{ duration: 0.3 }}
        >
          <p className="text-sm text-gray-400 mb-1">Total P&L</p>
          <p className={`text-4xl font-bold font-mono ${getPnLColor(metrics.total)}`}>
            {formatPnL(metrics.total)}
          </p>
          <p className={`text-sm mt-1 ${getPnLColor(metrics.percentage)}`}>
            {!hideValues && (
              <>
                {metrics.percentage > 0 ? '+' : ''}{metrics.percentage.toFixed(2)}%
                {metrics.percentage > 0 ? 
                  <TrendingUp className="w-4 h-4 inline ml-1" /> : 
                  <TrendingDown className="w-4 h-4 inline ml-1" />
                }
              </>
            )}
          </p>
        </motion.div>
      </div>
      
      {/* Breakdown */}
      {showBreakdown && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Realized PnL */}
          <div className="bg-gray-800/50 rounded p-3">
            <p className="text-xs text-gray-400 flex items-center gap-1">
              <Trophy className="w-3 h-3" />
              Realized
            </p>
            <p className={`text-lg font-mono font-semibold ${getPnLColor(metrics.realized)}`}>
              {formatPnL(metrics.realized)}
            </p>
          </div>
          
          {/* Unrealized PnL */}
          <div className="bg-gray-800/50 rounded p-3">
            <p className="text-xs text-gray-400 flex items-center gap-1">
              <Activity className="w-3 h-3" />
              Unrealized
            </p>
            <p className={`text-lg font-mono font-semibold ${getPnLColor(metrics.unrealized)}`}>
              {formatPnL(metrics.unrealized)}
            </p>
          </div>
        </div>
      )}
      
      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        {/* Win Rate */}
        <div className="text-center">
          <p className="text-xs text-gray-400">Win Rate</p>
          <p className={`text-sm font-mono font-bold ${
            metrics.winRate >= 60 ? 'text-green-400' : 
            metrics.winRate >= 40 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {hideValues ? '••%' : `${metrics.winRate.toFixed(1)}%`}
          </p>
        </div>
        
        {/* Profit Factor */}
        <div className="text-center">
          <p className="text-xs text-gray-400">Profit Factor</p>
          <p className={`text-sm font-mono font-bold ${
            metrics.profitFactor >= 2 ? 'text-green-400' : 
            metrics.profitFactor >= 1 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {hideValues ? '•.••' : 
             metrics.profitFactor === Infinity ? '∞' : 
             metrics.profitFactor.toFixed(2)}
          </p>
        </div>
        
        {/* Drawdown */}
        <div className="text-center">
          <p className="text-xs text-gray-400">Drawdown</p>
          <p className={`text-sm font-mono font-bold ${
            metrics.drawdown <= 5 ? 'text-green-400' : 
            metrics.drawdown <= 15 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {hideValues ? '••%' : `-${metrics.drawdown.toFixed(1)}%`}
          </p>
        </div>
      </div>
      
      {/* Streak Indicator */}
      <div className={`
        text-center py-2 rounded mb-4
        ${metrics.streak.type === 'win' ? 'bg-green-900/30' : 'bg-red-900/30'}
      `}>
        <p className="text-xs text-gray-400">Current Streak</p>
        <p className={`text-sm font-bold ${
          metrics.streak.type === 'win' ? 'text-green-400' : 'text-red-400'
        }`}>
          {metrics.streak.count} {metrics.streak.type === 'win' ? 'Wins' : 'Losses'} 
          {metrics.streak.type === 'win' ? ' 🔥' : ' 💔'}
        </p>
      </div>
      
      {/* Detailed Stats (collapsible) */}
      <AnimatePresence>
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="pt-4 border-t border-gray-800"
          >
            <div className="space-y-3">
              {/* Best Trade */}
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Best Trade</span>
                <div className="text-right">
                  <span className="text-sm font-mono text-green-400">
                    {formatPnL(metrics.bestTrade.pnl)}
                  </span>
                  <span className="text-xs text-gray-500 ml-2">
                    {metrics.bestTrade.symbol}
                  </span>
                </div>
              </div>
              
              {/* Worst Trade */}
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Worst Trade</span>
                <div className="text-right">
                  <span className="text-sm font-mono text-red-400">
                    {formatPnL(metrics.worstTrade.pnl)}
                  </span>
                  <span className="text-xs text-gray-500 ml-2">
                    {metrics.worstTrade.symbol}
                  </span>
                </div>
              </div>
              
              {/* Average Win/Loss */}
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Avg Win/Loss</span>
                <div className="text-right">
                  <span className="text-sm font-mono">
                    <span className="text-green-400">{formatPnL(metrics.avgWin, false)}</span>
                    <span className="text-gray-500"> / </span>
                    <span className="text-red-400">{formatPnL(metrics.avgLoss, false)}</span>
                  </span>
                </div>
              </div>
              
              {/* Sharpe Ratio */}
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Sharpe Ratio</span>
                <span className={`text-sm font-mono font-bold ${
                  metrics.sharpeRatio >= 2 ? 'text-green-400' : 
                  metrics.sharpeRatio >= 1 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {hideValues ? '•.••' : metrics.sharpeRatio.toFixed(2)}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Warning for significant drawdown */}
      {metrics.drawdown > 20 && !hideValues && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 p-3 bg-red-900/30 border border-red-900/50 rounded
                   flex items-center gap-2"
        >
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <p className="text-xs text-red-400">
            Significant drawdown detected. Consider reducing position sizes.
          </p>
        </motion.div>
      )}
      
      {/* Motivational message based on performance */}
      {!compact && (
        <div className="mt-4 text-center">
          <p className="text-xs text-gray-500 italic">
            {metrics.total > 1000 ? 
              "You're crushing it! Keep the discipline. 🚀" :
             metrics.total > 0 ? 
              "Green is good. Stay focused. 💚" :
             metrics.total > -500 ? 
              "Minor setback. Stick to the plan. 💪" :
             metrics.total > -1000 ? 
              "Rough patch. Review your strategy. 🤔" :
              "Time to step back and reassess. 🛑"
            }
          </p>
        </div>
      )}
    </div>
  );
};

/**
 * PNL WISDOM FROM THE TRENCHES:
 * 
 * 1. The flash animation isn't just eye candy. It trains your brain
 *    to associate color with performance. Green flash = good decision.
 *    Red flash = learning opportunity.
 * 
 * 2. Win rate is overrated. A 30% win rate with 3:1 risk/reward beats
 *    a 70% win rate with 1:3 risk/reward. Profit factor tells the truth.
 * 
 * 3. Drawdown percentage matters more than dollar amount. A 50% drawdown
 *    requires a 100% gain to break even. Math is cruel like that.
 * 
 * 4. That streak counter? It's there to break psychological patterns.
 *    Humans hate breaking streaks, even losing ones.
 * 
 * 5. The hide values feature isn't for screenshots. It's for clarity.
 *    Sometimes you need to see the pattern, not the pain.
 * 
 * 6. Best/worst trade tracking teaches you about yourself. Do you cut
 *    winners too early? Let losers run too long? The data knows.
 * 
 * Remember: PnL is just a score. The real game is risk management.
 * 
 * "In trading, it's not about being right. It's about how much you
 * make when you're right and how much you lose when you're wrong."
 */
