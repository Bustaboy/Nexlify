// src/components/metrics/PnLDisplay.tsx
// Cyberpunk Trading Terminal - Neural P&L Analytics Display
// Real-time profit/loss tracking with quantum-encrypted precision

import React, { useMemo, useEffect, useState } from 'react';
import Decimal from 'decimal.js';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Zap, 
  Shield, 
  AlertTriangle,
  BarChart3,
  Target
} from 'lucide-react';

// === TYPE DEFINITIONS ===
interface TradeHistory {
  id: string;
  symbol: string;
  realizedPnL: Decimal;
  timestamp: Date;
  quantity: number;
  price: Decimal;
  side: 'buy' | 'sell';
}

interface Position {
  symbol: string;
  unrealizedPnL: Decimal;
  quantity: number;
  avgPrice: Decimal;
  currentPrice: Decimal;
}

interface PnLHistory {
  date: Date;
  total: number;
  realized: number;
  unrealized: number;
}

interface PnLMetrics {
  total: number;
  realized: number;
  unrealized: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  sharpeRatio: number;
  currentStreak: { type: 'win' | 'loss'; count: number };
  drawdown: number;
  bestTrade: { symbol: string; pnl: number; date: Date };
  worstTrade: { symbol: string; pnl: number; date: Date };
}

interface PnLDisplayProps {
  trades: TradeHistory[];
  positions: Record<string, Position>;
  pnlHistory: PnLHistory[];
  timeframe: 'day' | 'week' | 'month' | 'year' | 'all';
  riskFreeRate?: number;
  showBreakdown?: boolean;
  compact?: boolean;
  animate?: boolean;
  hideValues?: boolean;
}

// === UTILITY FUNCTIONS ===
const formatCurrency = (value: number, hide?: boolean): string => {
  if (hide) return '¥****.**';
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}¥${Math.abs(value).toLocaleString('ja-JP', { 
    minimumFractionDigits: 2,
    maximumFractionDigits: 2 
  })}`;
};

const formatPercent = (value: number, hide?: boolean): string => {
  if (hide) return '**.*%';
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}%`;
};

// === MAIN COMPONENT ===
export const PnLDisplay: React.FC<PnLDisplayProps> = ({
  trades,
  positions,
  pnlHistory,
  timeframe,
  riskFreeRate = 0.02,
  showBreakdown = true,
  compact = false,
  animate = true,
  hideValues = false
}) => {
  const [glitchEffect, setGlitchEffect] = useState(false);
  
  // Trigger glitch effect on significant P&L changes
  useEffect(() => {
    setGlitchEffect(true);
    const timer = setTimeout(() => setGlitchEffect(false), 300);
    return () => clearTimeout(timer);
  }, [trades.length]);

  const metrics = useMemo<PnLMetrics>(() => {
    // Filter trades by timeframe
    const now = new Date();
    const startDate = new Date();
    
    switch (timeframe) {
      case 'day':
        startDate.setDate(now.getDate() - 1);
        break;
      case 'week':
        startDate.setDate(now.getDate() - 7);
        break;
      case 'month':
        startDate.setMonth(now.getMonth() - 1);
        break;
      case 'year':
        startDate.setFullYear(now.getFullYear() - 1);
        break;
      case 'all':
        startDate.setFullYear(2000); // Arbitrary old date
        break;
    }

    const periodTrades = trades.filter(t => t.timestamp >= startDate);

    // Calculate realized P&L
    const realized = periodTrades.reduce((sum, t) => 
      new Decimal(sum).add(t.realizedPnL).toNumber(), 
      0
    );

    // Calculate unrealized P&L
    const unrealized = Object.values(positions).reduce((sum, p) => 
      new Decimal(sum).add(p.unrealizedPnL).toNumber(), 
      0
    );

    const total = realized + unrealized;

    // Win/Loss analysis
    const winningTrades = periodTrades.filter(t => t.realizedPnL.gt(0));
    const losingTrades = periodTrades.filter(t => t.realizedPnL.lt(0));
    const winRate = periodTrades.length > 0 
      ? (winningTrades.length / periodTrades.length) * 100 
      : 0;

    // Average win/loss
    const avgWin = winningTrades.length > 0
      ? winningTrades.reduce((sum, t) => 
          new Decimal(sum).add(t.realizedPnL).toNumber(), 
          0
        ) / winningTrades.length
      : 0;

    const avgLoss = losingTrades.length > 0
      ? Math.abs(
          losingTrades.reduce((sum, t) => 
            new Decimal(sum).add(t.realizedPnL).toNumber(), 
            0
          ) / losingTrades.length
        )
      : 0;

    // Profit factor
    const totalWins = winningTrades.reduce((sum, t) => 
      new Decimal(sum).add(t.realizedPnL).toNumber(), 
      0
    );
    const totalLosses = Math.abs(
      losingTrades.reduce((sum, t) => 
        new Decimal(sum).add(t.realizedPnL).toNumber(), 
        0
      )
    );
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;

    // Best and worst trades
    const sortedTrades = [...periodTrades].sort((a, b) => {
      const aVal = a.realizedPnL.toNumber();
      const bVal = b.realizedPnL.toNumber();
      return bVal - aVal;
    });

    const bestTrade = sortedTrades[0] 
      ? {
          symbol: sortedTrades[0].symbol,
          pnl: sortedTrades[0].realizedPnL.toNumber(),
          date: sortedTrades[0].timestamp
        }
      : { symbol: 'NULL', pnl: 0, date: new Date() };

    const worstTrade = sortedTrades[sortedTrades.length - 1]
      ? {
          symbol: sortedTrades[sortedTrades.length - 1].symbol,
          pnl: sortedTrades[sortedTrades.length - 1].realizedPnL.toNumber(),
          date: sortedTrades[sortedTrades.length - 1].timestamp
        }
      : { symbol: 'NULL', pnl: 0, date: new Date() };

    // Consecutive wins/losses and current streak
    let maxConsecutiveWins = 0;
    let maxConsecutiveLosses = 0;
    let currentWins = 0;
    let currentLosses = 0;
    let currentStreak: { type: 'win' | 'loss'; count: number } = { type: 'win', count: 0 };

    periodTrades.forEach((trade, index) => {
      const isWin = trade.realizedPnL.gt(0);
      
      if (isWin) {
        currentWins++;
        currentLosses = 0;
        maxConsecutiveWins = Math.max(maxConsecutiveWins, currentWins);
        
        if (index === periodTrades.length - 1) {
          currentStreak = { type: 'win', count: currentWins };
        }
      } else {
        currentLosses++;
        currentWins = 0;
        maxConsecutiveLosses = Math.max(maxConsecutiveLosses, currentLosses);
        
        if (index === periodTrades.length - 1) {
          currentStreak = { type: 'loss', count: currentLosses };
        }
      }
    });

    // Sharpe Ratio calculation
    const returns = pnlHistory.map(p => p.total);
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length || 0;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length || 1
    );
    const sharpeRatio = stdDev !== 0 ? (avgReturn - riskFreeRate) / stdDev : 0;

    // Drawdown calculation
    const highWaterMark = Math.max(...pnlHistory.map(p => p.total), total);
    const drawdown = highWaterMark > 0 ? ((highWaterMark - total) / highWaterMark) * 100 : 0;

    return {
      total,
      realized,
      unrealized,
      winRate,
      avgWin,
      avgLoss,
      profitFactor,
      maxConsecutiveWins,
      maxConsecutiveLosses,
      sharpeRatio,
      currentStreak,
      drawdown,
      bestTrade,
      worstTrade
    };
  }, [trades, positions, pnlHistory, timeframe, riskFreeRate]);

  // === RENDER FUNCTIONS ===
  const renderMetricCard = (
    icon: React.ReactNode,
    label: string,
    value: string | number,
    trend?: 'up' | 'down' | 'neutral',
    highlight?: boolean
  ) => (
    <motion.div
      className={`
        relative p-4 rounded border backdrop-blur-md
        ${highlight ? 'border-cyan-400 bg-cyan-900/20' : 'border-purple-600/30 bg-black/40'}
        ${glitchEffect ? 'animate-pulse' : ''}
      `}
      whileHover={animate ? { scale: 1.02, borderColor: '#00ffff' } : undefined}
      transition={{ duration: 0.2 }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`p-1.5 rounded ${highlight ? 'bg-cyan-500/20' : 'bg-purple-600/20'}`}>
            {icon}
          </div>
          <span className="text-xs text-gray-400 uppercase tracking-wider">{label}</span>
        </div>
        {trend && (
          <div className={`
            text-xs px-2 py-1 rounded
            ${trend === 'up' ? 'text-green-400 bg-green-900/30' : ''}
            ${trend === 'down' ? 'text-red-400 bg-red-900/30' : ''}
            ${trend === 'neutral' ? 'text-gray-400 bg-gray-900/30' : ''}
          `}>
            {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '—'}
          </div>
        )}
      </div>
      <div className={`
        text-2xl font-mono font-bold
        ${typeof value === 'number' && value < 0 ? 'text-red-400' : 'text-cyan-400'}
      `}>
        {typeof value === 'number' ? formatCurrency(value, hideValues) : value}
      </div>
    </motion.div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      {!compact && (
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-cyan-400 flex items-center gap-3">
            <Activity className="w-8 h-8" />
            NEURAL P&L MATRIX
          </h2>
          <div className="text-sm text-gray-400">
            TIMEFRAME: <span className="text-cyan-400">{timeframe.toUpperCase()}</span>
          </div>
        </div>
      )}

      {/* Main P&L Summary */}
      <div className={`grid ${compact ? 'grid-cols-3' : 'grid-cols-1 md:grid-cols-3'} gap-4`}>
        {renderMetricCard(
          <Zap className="w-5 h-5 text-cyan-400" />,
          'Total P&L',
          metrics.total,
          metrics.total > 0 ? 'up' : metrics.total < 0 ? 'down' : 'neutral',
          true
        )}
        {renderMetricCard(
          <Shield className="w-5 h-5 text-green-400" />,
          'Realized',
          metrics.realized,
          metrics.realized > 0 ? 'up' : metrics.realized < 0 ? 'down' : 'neutral'
        )}
        {renderMetricCard(
          <AlertTriangle className="w-5 h-5 text-yellow-400" />,
          'Unrealized',
          metrics.unrealized,
          metrics.unrealized > 0 ? 'up' : metrics.unrealized < 0 ? 'down' : 'neutral'
        )}
      </div>

      {/* Show detailed breakdown only if showBreakdown is true and not in compact mode */}
      {showBreakdown && !compact && (
        <>
          {/* Performance Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <div className="text-xs text-gray-400 uppercase mb-1">Win Rate</div>
              <div className="text-xl font-mono text-cyan-400">
                {hideValues ? '**.*%' : `${metrics.winRate.toFixed(1)}%`}
              </div>
            </div>
            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <div className="text-xs text-gray-400 uppercase mb-1">Profit Factor</div>
              <div className="text-xl font-mono text-cyan-400">
                {hideValues ? '*.**' : metrics.profitFactor === Infinity ? '∞' : metrics.profitFactor.toFixed(2)}
              </div>
            </div>
            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <div className="text-xs text-gray-400 uppercase mb-1">Sharpe Ratio</div>
              <div className="text-xl font-mono text-cyan-400">
                {hideValues ? '*.**' : metrics.sharpeRatio.toFixed(2)}
              </div>
            </div>
            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <div className="text-xs text-gray-400 uppercase mb-1">Drawdown</div>
              <div className="text-xl font-mono text-red-400">
                {hideValues ? '-**.*%' : `-${metrics.drawdown.toFixed(1)}%`}
              </div>
            </div>
          </div>

          {/* Trade Analysis */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                BEST TRADE
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Symbol:</span>
                  <span className="text-cyan-400 font-mono">{metrics.bestTrade.symbol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">P&L:</span>
                  <span className="text-green-400 font-mono">
                    {formatCurrency(metrics.bestTrade.pnl, hideValues)}
                  </span>
                </div>
              </div>
            </div>

            <div className="p-4 rounded border border-purple-600/30 bg-black/40">
              <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center gap-2">
                <TrendingDown className="w-4 h-4" />
                WORST TRADE
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Symbol:</span>
                  <span className="text-cyan-400 font-mono">{metrics.worstTrade.symbol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">P&L:</span>
                  <span className="text-red-400 font-mono">
                    {formatCurrency(metrics.worstTrade.pnl, hideValues)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Streak Information */}
          <div className="p-4 rounded border border-purple-600/30 bg-black/40">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                <span className="text-sm font-bold text-cyan-400">CURRENT STREAK</span>
              </div>
              <div className={`
                px-3 py-1 rounded font-mono text-sm
                ${metrics.currentStreak.type === 'win' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}
              `}>
                {metrics.currentStreak.count} {metrics.currentStreak.type === 'win' ? 'WINS' : 'LOSSES'}
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Max Consecutive Wins:</span>
                <span className="text-green-400 font-mono">{metrics.maxConsecutiveWins}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Max Consecutive Losses:</span>
                <span className="text-red-400 font-mono">{metrics.maxConsecutiveLosses}</span>
              </div>
            </div>
          </div>

          {/* Average Win/Loss */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded border border-green-600/30 bg-green-900/10">
              <div className="text-xs text-gray-400 uppercase mb-1">Average Win</div>
              <div className="text-xl font-mono text-green-400">
                {formatCurrency(metrics.avgWin, hideValues)}
              </div>
            </div>
            <div className="p-4 rounded border border-red-600/30 bg-red-900/10">
              <div className="text-xs text-gray-400 uppercase mb-1">Average Loss</div>
              <div className="text-xl font-mono text-red-400">
                {formatCurrency(-metrics.avgLoss, hideValues)}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};