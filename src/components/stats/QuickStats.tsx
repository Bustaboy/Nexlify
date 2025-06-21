// src/components/stats/QuickStats.tsx
// NEXLIFY QUICK STATS - Instant battlefield awareness at a glance
// Last sync: 2025-06-19 | "Numbers tell stories, but only if you're listening"

import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  Users,
  Activity,
  Zap,
  Trophy,
  Target,
  Clock,
  BarChart3,
  PieChart,
  Hash,
  ArrowUp,
  ArrowDown,
  Minus,
  AlertCircle,
  Sparkles
} from 'lucide-react';

import { useTradingStore } from '@/stores/tradingStore';
import { useMarketStore } from '@/stores/marketStore';

interface QuickStatsProps {
  layout?: 'grid' | 'row' | 'compact';
  stats?: string[]; // Which stats to show
  animate?: boolean;
  showSparklines?: boolean;
  updateInterval?: number; // ms
  theme?: 'minimal' | 'detailed' | 'neon';
}

interface StatItem {
  id: string;
  label: string;
  value: number | string;
  previousValue?: number;
  change?: number;
  changePercent?: number;
  icon: JSX.Element;
  color: string;
  suffix?: string;
  prefix?: string;
  sparkline?: number[];
  alert?: string;
  priority: number; // For sorting
}

/**
 * QUICK STATS - The HUD for digital warriors
 * 
 * Built this after "Dashboard Danny" spent 5 minutes looking for his
 * account balance during a flash crash. By the time he found it, he'd
 * lost 30%. Information delayed is opportunity destroyed.
 * 
 * This component gives you everything at a glance:
 * - Account vitals (balance, equity, margin)
 * - Performance metrics (win rate, profit factor)
 * - Market conditions (volatility, volume)
 * - Position summary (count, exposure, P&L)
 * 
 * Remember: In fast markets, the trader who sees first, acts first.
 */
export const QuickStats = ({
  layout = 'grid',
  stats: selectedStats,
  animate = true,
  showSparklines = true,
  updateInterval = 1000,
  theme = 'detailed'
}: QuickStatsProps) => {
  // State management
  const [previousValues, setPreviousValues] = useState<Map<string, number>>(new Map());
  const [sparklineData, setSparklineData] = useState<Map<string, number[]>>(new Map());
  const [pulseStats, setPulseStats] = useState<Set<string>>(new Set());
  
  // Store connections
  const { 
    accountBalance,
    positions,
    trades,
    dailyPnL,
    winRate,
    profitFactor
  } = useTradingStore();
  const { 
    marketData,
    volatilityIndex,
    totalVolume24h 
  } = useMarketStore();
  
  /**
   * Calculate all available stats
   * 
   * Each stat tells part of the story. Together, they reveal
   * your trading truth.
   */
  const allStats = useMemo((): StatItem[] => {
    const stats: StatItem[] = [];
    
    // Account Balance
    stats.push({
      id: 'balance',
      label: 'Balance',
      value: accountBalance.total,
      previousValue: previousValues.get('balance'),
      icon: <DollarSign className="w-4 h-4" />,
      color: 'text-green-400',
      prefix: '$',
      priority: 1
    });
    
    // Daily P&L
    const dailyPnLColor = dailyPnL >= 0 ? 'text-green-400' : 'text-red-400';
    stats.push({
      id: 'daily_pnl',
      label: 'Daily P&L',
      value: dailyPnL,
      previousValue: previousValues.get('daily_pnl'),
      icon: dailyPnL >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />,
      color: dailyPnLColor,
      prefix: dailyPnL >= 0 ? '+$' : '-$',
      alert: Math.abs(dailyPnL) > accountBalance.total * 0.05 ? 'Large daily move!' : undefined,
      priority: 2
    });
    
    // Open Positions
    const positionCount = Object.keys(positions).length;
    stats.push({
      id: 'positions',
      label: 'Positions',
      value: positionCount,
      icon: <Target className="w-4 h-4" />,
      color: positionCount > 10 ? 'text-yellow-400' : 'text-cyan-400',
      alert: positionCount > 15 ? 'High position count' : undefined,
      priority: 3
    });
    
    // Total Exposure
    const totalExposure = Object.values(positions).reduce(
      (sum, pos) => sum + Math.abs(pos.quantity * pos.currentPrice), 0
    );
    const exposurePercent = accountBalance.total > 0 
      ? (totalExposure / accountBalance.total) * 100 
      : 0;
    stats.push({
      id: 'exposure',
      label: 'Exposure',
      value: exposurePercent,
      icon: <Activity className="w-4 h-4" />,
      color: exposurePercent > 80 ? 'text-red-400' : 
             exposurePercent > 50 ? 'text-yellow-400' : 'text-green-400',
      suffix: '%',
      alert: exposurePercent > 100 ? 'Over-leveraged!' : undefined,
      priority: 4
    });
    
    // Win Rate
    stats.push({
      id: 'win_rate',
      label: 'Win Rate',
      value: winRate,
      previousValue: previousValues.get('win_rate'),
      icon: <Trophy className="w-4 h-4" />,
      color: winRate >= 60 ? 'text-green-400' : 
             winRate >= 40 ? 'text-yellow-400' : 'text-red-400',
      suffix: '%',
      priority: 5
    });
    
    // Profit Factor
    stats.push({
      id: 'profit_factor',
      label: 'Profit Factor',
      value: profitFactor,
      previousValue: previousValues.get('profit_factor'),
      icon: <BarChart3 className="w-4 h-4" />,
      color: profitFactor >= 2 ? 'text-green-400' : 
             profitFactor >= 1 ? 'text-yellow-400' : 'text-red-400',
      suffix: 'x',
      alert: profitFactor < 1 ? 'Losing more than winning' : undefined,
      priority: 6
    });
    
    // Market Volatility
    stats.push({
      id: 'volatility',
      label: 'Volatility',
      value: volatilityIndex,
      previousValue: previousValues.get('volatility'),
      icon: <Zap className="w-4 h-4" />,
      color: volatilityIndex > 80 ? 'text-red-400' : 
             volatilityIndex > 50 ? 'text-yellow-400' : 'text-green-400',
      suffix: '',
      alert: volatilityIndex > 80 ? 'Extreme volatility!' : undefined,
      priority: 7
    });
    
    // Active Trades Today
    const todayTrades = trades.filter(t => {
      const tradeDate = new Date(t.closeTime);
      const today = new Date();
      return tradeDate.toDateString() === today.toDateString();
    }).length;
    stats.push({
      id: 'trades_today',
      label: 'Trades Today',
      value: todayTrades,
      icon: <Hash className="w-4 h-4" />,
      color: todayTrades > 50 ? 'text-yellow-400' : 'text-cyan-400',
      alert: todayTrades > 100 ? 'High trading frequency' : undefined,
      priority: 8
    });
    
    // Average Trade Duration
    const avgDuration = trades.length > 0
      ? trades.reduce((sum, t) => {
          const duration = new Date(t.closeTime).getTime() - new Date(t.openTime).getTime();
          return sum + duration;
        }, 0) / trades.length / (60 * 1000) // Convert to minutes
      : 0;
    stats.push({
      id: 'avg_duration',
      label: 'Avg Duration',
      value: avgDuration.toFixed(0),
      icon: <Clock className="w-4 h-4" />,
      color: 'text-blue-400',
      suffix: 'm',
      priority: 9
    });
    
    // Market Volume
    const volumeInBillions = totalVolume24h / 1e9;
    stats.push({
      id: 'volume',
      label: '24h Volume',
      value: volumeInBillions.toFixed(1),
      previousValue: previousValues.get('volume'),
      icon: <PieChart className="w-4 h-4" />,
      color: 'text-purple-400',
      suffix: 'B',
      priority: 10
    });
    
    // Calculate changes
    stats.forEach(stat => {
      if (stat.previousValue !== undefined && typeof stat.value === 'number') {
        stat.change = stat.value - stat.previousValue;
        stat.changePercent = stat.previousValue !== 0 
          ? ((stat.value - stat.previousValue) / Math.abs(stat.previousValue)) * 100
          : 0;
      }
    });
    
    // Filter selected stats
    if (selectedStats?.length) {
      return stats.filter(s => selectedStats.includes(s.id))
                  .sort((a, b) => a.priority - b.priority);
    }
    
    // Default stats based on layout
    const defaultStats = layout === 'compact' 
      ? ['balance', 'daily_pnl', 'positions', 'exposure']
      : stats.map(s => s.id);
    
    return stats.filter(s => defaultStats.includes(s.id))
                .sort((a, b) => a.priority - b.priority);
  }, [
    accountBalance, 
    positions, 
    trades, 
    dailyPnL, 
    winRate, 
    profitFactor,
    volatilityIndex,
    totalVolume24h,
    previousValues,
    selectedStats,
    layout
  ]);
  
  /**
   * Update previous values for change detection
   */
  useEffect(() => {
    const timer = setInterval(() => {
      const newPrevious = new Map<string, number>();
      allStats.forEach(stat => {
        if (typeof stat.value === 'number') {
          newPrevious.set(stat.id, stat.value);
        }
      });
      setPreviousValues(newPrevious);
    }, updateInterval);
    
    return () => clearInterval(timer);
  }, [allStats, updateInterval]);
  
  /**
   * Generate sparkline data
   */
  useEffect(() => {
    if (!showSparklines) return;
    
    const timer = setInterval(() => {
      setSparklineData(prev => {
        const newData = new Map(prev);
        
        allStats.forEach(stat => {
          if (typeof stat.value === 'number') {
            const existing = newData.get(stat.id) || [];
            const updated = [...existing, stat.value].slice(-20); // Keep last 20 points
            newData.set(stat.id, updated);
          }
        });
        
        return newData;
      });
    }, updateInterval);
    
    return () => clearInterval(timer);
  }, [allStats, showSparklines, updateInterval]);
  
  /**
   * Pulse animation for significant changes
   */
  useEffect(() => {
    if (!animate) return;
    
    const newPulse = new Set<string>();
    
    allStats.forEach(stat => {
      if (stat.changePercent && Math.abs(stat.changePercent) > 5) {
        newPulse.add(stat.id);
        
        // Remove pulse after animation
        setTimeout(() => {
          setPulseStats(prev => {
            const updated = new Set(prev);
            updated.delete(stat.id);
            return updated;
          });
        }, 1000);
      }
    });
    
    setPulseStats(prev => new Set([...prev, ...newPulse]));
  }, [allStats, animate]);
  
  /**
   * Format value based on type
   */
  const formatValue = (stat: StatItem): string => {
    const val = typeof stat.value === 'number' 
      ? Math.abs(stat.value).toFixed(stat.suffix === '%' ? 1 : 2)
      : stat.value;
    
    return `${stat.prefix || ''}${val}${stat.suffix || ''}`;
  };
  
  /**
   * Get change indicator
   */
  const getChangeIndicator = (stat: StatItem) => {
    if (!stat.change || stat.change === 0) {
      return <Minus className="w-3 h-3 text-gray-500" />;
    }
    
    return stat.change > 0 
      ? <ArrowUp className="w-3 h-3 text-green-400" />
      : <ArrowDown className="w-3 h-3 text-red-400" />;
  };
  
  /**
   * Render sparkline
   */
  const renderSparkline = (data: number[]): JSX.Element => {
    if (!data || data.length < 2) return <div className="h-8" />;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
    
    const trend = data[data.length - 1] > data[0] ? 'up' : 'down';
    
    return (
      <svg className="h-8 w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={trend === 'up' ? '#10b981' : '#ef4444'}
          strokeWidth="2"
          className="opacity-50"
        />
        <circle
          cx="100"
          cy={100 - ((data[data.length - 1] - min) / range) * 100}
          r="2"
          fill={trend === 'up' ? '#10b981' : '#ef4444'}
          className="animate-pulse"
        />
      </svg>
    );
  };
  
  /**
   * Render individual stat card
   */
  const renderStatCard = (stat: StatItem) => {
    const isPulsing = pulseStats.has(stat.id);
    const sparkline = sparklineData.get(stat.id);
    
    return (
      <motion.div
        key={stat.id}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ 
          opacity: 1, 
          scale: isPulsing ? [1, 1.05, 1] : 1 
        }}
        transition={{ duration: 0.3 }}
        className={`
          bg-gray-900/50 border border-cyan-900/30 rounded-lg p-3
          ${isPulsing ? 'ring-2 ring-cyan-500/50' : ''}
          ${stat.alert ? 'border-yellow-500/50' : ''}
          hover:bg-gray-800/50 transition-all cursor-pointer
          ${theme === 'neon' ? 'shadow-lg shadow-cyan-900/20' : ''}
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className={stat.color}>{stat.icon}</div>
            <span className="text-xs text-gray-400">{stat.label}</span>
          </div>
          {animate && stat.change !== undefined && (
            <div className="flex items-center gap-1">
              {getChangeIndicator(stat)}
              {stat.changePercent !== undefined && (
                <span className={`text-xs ${
                  stat.change! > 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {Math.abs(stat.changePercent).toFixed(1)}%
                </span>
              )}
            </div>
          )}
        </div>
        
        {/* Value */}
        <div className={`text-xl font-bold font-mono ${stat.color}`}>
          {formatValue(stat)}
        </div>
        
        {/* Alert */}
        {stat.alert && (
          <div className="flex items-center gap-1 mt-1">
            <AlertCircle className="w-3 h-3 text-yellow-400" />
            <span className="text-xs text-yellow-400">{stat.alert}</span>
          </div>
        )}
        
        {/* Sparkline */}
        {showSparklines && sparkline && theme !== 'minimal' && (
          <div className="mt-2">
            {renderSparkline(sparkline)}
          </div>
        )}
      </motion.div>
    );
  };
  
  // Layout classes
  const layoutClasses = {
    grid: `grid ${layout === 'compact' ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4'} gap-3`,
    row: 'flex gap-3 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-700',
    compact: 'grid grid-cols-2 gap-2'
  };
  
  return (
    <div className={`
      ${theme === 'neon' ? 'relative' : ''}
    `}>
      {/* Neon glow effect */}
      {theme === 'neon' && (
        <div className="absolute inset-0 opacity-20 blur-xl">
          <div className="h-full w-full bg-gradient-to-r from-cyan-500 to-purple-500" />
        </div>
      )}
      
      {/* Stats grid */}
      <div className={layoutClasses[layout]}>
        <AnimatePresence mode="popLayout">
          {allStats.map(stat => renderStatCard(stat))}
        </AnimatePresence>
      </div>
      
      {/* Summary sparkle (for good performance) */}
      {theme === 'detailed' && winRate > 60 && profitFactor > 1.5 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-3 text-center"
        >
          <div className="inline-flex items-center gap-2 text-sm text-green-400">
            <Sparkles className="w-4 h-4" />
            <span>Performance is on fire! Keep it up!</span>
            <Sparkles className="w-4 h-4" />
          </div>
        </motion.div>
      )}
    </div>
  );
};

/**
 * QUICK STATS WISDOM:
 * 
 * 1. The pulse animation on change isn't just pretty - it draws
 *    attention to what matters RIGHT NOW. Movement catches the eye.
 * 
 * 2. Sparklines tell stories. That tiny chart shows trend at a 
 *    glance, no analysis required. Up good, down bad, simple.
 * 
 * 3. Color coding bypasses conscious thought. Red = danger arrives
 *    in your brain before the number is processed.
 * 
 * 4. Alert thresholds are based on real blown accounts. Every
 *    warning represents someone's financial funeral.
 * 
 * 5. The layout matters. Grid for overview, row for scanning,
 *    compact for mobile or side panels.
 * 
 * 6. Priority sorting ensures the most important stats are always
 *    visible, even on small screens.
 * 
 * Remember: Information architecture is survival architecture.
 * What you see first might be what saves you.
 * 
 * "In trading, speed of information is speed of decision. 
 * Speed of decision is the edge between profit and loss."
 */
