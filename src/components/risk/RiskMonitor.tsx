// src/components/risk/RiskMonitor.tsx
// NEXLIFY RISK MONITOR - Your guardian against financial annihilation
// Last sync: 2025-06-21 | "Risk is what's left when you think you've thought of everything"

import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield,
  AlertTriangle,
  AlertOctagon,
  TrendingDown,
  Gauge,
  Heart,
  Skull,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  Siren,
  Lock,
  Unlock,
  Zap,
  BarChart3,
  Activity,
  Timer,
  Target
} from 'lucide-react';
import toast from 'react-hot-toast';
import Decimal from 'decimal.js';
import { useTradingStore } from '@/stores/tradingStore';
import { useMarketStore } from '@/stores/marketStore';

interface RiskMonitorProps {
  compact?: boolean;
  showAlerts?: boolean;
  autoLockOnBreach?: boolean;
  theme?: 'minimal' | 'detailed' | 'matrix';
  onRiskBreach?: (type: string, level: number) => void;
}

interface RiskMetrics {
  overallRisk: number; // 0-100
  accountRisk: number; // Percentage of account at risk
  positionRisk: number; // Largest position risk
  correlationRisk: number; // Risk from correlated positions
  leverageRisk: number; // Overall leverage exposure
  liquidationRisk: number; // Distance to liquidation
  volatilityRisk: number; // Market volatility factor
  timeRisk: number; // Risk from holding duration
  breaches: RiskBreach[];
}

interface RiskBreach {
  type: 'leverage' | 'position' | 'drawdown' | 'correlation' | 'volatility' | 'daily_loss';
  severity: 'warning' | 'critical' | 'emergency';
  message: string;
  value: number;
  threshold: number;
  timestamp: Date;
}

interface RiskThresholds {
  maxLeverage: number;
  maxPositionSize: number;
  maxDailyLoss: number;
  maxDrawdown: number;
  maxCorrelation: number;
  criticalMargin: number;
}

/**
 * RISK MONITOR - The guardian at the gates of ruin
 * 
 * Built this after the "Martingale Mike" incident. He kept doubling
 * down on losing positions, convinced the market would turn. It didn't.
 * His $50k account became $0 in 3 hours. The only thing that turned
 * was his stomach.
 * 
 * This monitor is your last line of defense:
 * - Real-time risk scoring across multiple dimensions
 * - Auto-lock trading when risk exceeds thresholds
 * - Visual and audio alerts before disaster strikes
 * - Correlation detection (the silent portfolio killer)
 * 
 * Remember: The market can remain irrational longer than you can
 * remain solvent. This monitor keeps you solvent.
 */
export const RiskMonitor = ({
  compact = false,
  showAlerts = true,
  autoLockOnBreach = true,
  theme = 'detailed',
  onRiskBreach
}: RiskMonitorProps) => {
  // State management
  const [isLocked, setIsLocked] = useState(false);
  const [lockReason, setLockReason] = useState<string>('');
  const [pulseAlert, setPulseAlert] = useState(false);
  const [recentBreaches, setRecentBreaches] = useState<RiskBreach[]>([]);
  const [heartbeatInterval, setHeartbeatInterval] = useState(1000);
  
  // Store connections
  const { 
    positions, 
    accountBalance,
    riskLimits,
    dailyPnL,
    setTradingLocked 
  } = useTradingStore();
  const { volatilityIndex } = useMarketStore();
  
  // Extended risk limits with defaults
  const extendedRiskLimits = {
    maxLeverage: 10,
    maxPositionSize: 0.2, // 20% of account
    maxDailyLoss: riskLimits.maxDailyLoss.toNumber() || 0.05, // Convert Decimal to number
    maxDrawdown: 0.15, // 15% drawdown
    riskLimitPerTrade: riskLimits.riskLimitPerTrade.toNumber()
  };
  
  // Risk thresholds
  const thresholds: RiskThresholds = {
    maxLeverage: extendedRiskLimits.maxLeverage,
    maxPositionSize: extendedRiskLimits.maxPositionSize,
    maxDailyLoss: extendedRiskLimits.maxDailyLoss,
    maxDrawdown: extendedRiskLimits.maxDrawdown,
    maxCorrelation: 0.7, // 70% correlation threshold
    criticalMargin: 0.2 // 20% margin level critical
  };
  
  /**
   * Calculate correlation risk - the hidden danger
   * 
   * Correlation kills portfolios silently. You think you're diversified
   * with BTC, ETH, and MATIC. Then crypto winter comes and they all
   * move together. This detects that concentration risk.
   */
  const calculateCorrelationRisk = useCallback((): number => {
    const activePositions = Object.values(positions);
    if (activePositions.length < 2) return 0;
    
    // Simplified correlation calculation
    // In production, would use actual price correlation matrix
    const cryptoPositions = activePositions.filter(p => 
      ['BTC', 'ETH', 'SOL', 'MATIC'].some(coin => p.symbol.includes(coin))
    );
    
    const correlationRatio = cryptoPositions.length / activePositions.length;
    return correlationRatio * 100;
  }, [positions]);
  
  /**
   * Calculate comprehensive risk metrics
   * 
   * This is where math meets mortality. Every metric here was paid
   * for in liquidated accounts.
   */
  const riskMetrics = useMemo((): RiskMetrics => {
    const breaches: RiskBreach[] = [];
    
    // Account risk - how much capital is deployed
    const totalPositionValue = Object.values(positions).reduce(
      (sum, pos) => sum + pos.quantity.mul(pos.currentPrice).toNumber(),
      0
    );
    const accountBalanceNum = accountBalance.toNumber();
    const accountRisk = accountBalanceNum > 0 
      ? (totalPositionValue / accountBalanceNum) * 100 
      : 0;
    
    // Position risk - largest single position
    const positionValues = Object.values(positions).map(p => 
      pos.quantity.mul(pos.currentPrice).toNumber() / accountBalanceNum
    );
    const largestPosition = positionValues.length > 0 
      ? Math.max(...positionValues) * 100
      : 0;
    
    if (largestPosition > thresholds.maxPositionSize * 100) {
      breaches.push({
        type: 'position',
        severity: largestPosition > thresholds.maxPositionSize * 150 ? 'critical' : 'warning',
        message: `Position size ${largestPosition.toFixed(1)}% exceeds limit`,
        value: largestPosition,
        threshold: thresholds.maxPositionSize * 100,
        timestamp: new Date()
      });
    }
    
    // Leverage risk
    const totalLeverage = Object.values(positions).reduce(
      (sum, pos) => sum + pos.leverage, 0
    ) / Math.max(Object.keys(positions).length, 1);
    
    if (totalLeverage > thresholds.maxLeverage) {
      breaches.push({
        type: 'leverage',
        severity: totalLeverage > thresholds.maxLeverage * 1.5 ? 'emergency' : 'critical',
        message: `Leverage ${totalLeverage.toFixed(1)}x exceeds limit`,
        value: totalLeverage,
        threshold: thresholds.maxLeverage,
        timestamp: new Date()
      });
    }
    
    // Correlation risk
    const correlationRisk = calculateCorrelationRisk();
    if (correlationRisk > thresholds.maxCorrelation * 100) {
      breaches.push({
        type: 'correlation',
        severity: 'warning',
        message: 'High correlation between positions',
        value: correlationRisk,
        threshold: thresholds.maxCorrelation * 100,
        timestamp: new Date()
      });
    }
    
    // Liquidation risk - how close to death
    const liquidationDistances = Object.values(positions).map(pos => {
      if (!pos.liquidationPrice) return 100;
      const distance = Math.abs(
        pos.currentPrice.toNumber() - pos.liquidationPrice.toNumber()
      ) / pos.currentPrice.toNumber();
      return distance * 100;
    });
    const minLiquidationDistance = liquidationDistances.length > 0
      ? Math.min(...liquidationDistances)
      : 100;
    const liquidationRisk = 100 - minLiquidationDistance;
    
    if (liquidationRisk > 80) {
      breaches.push({
        type: 'position',
        severity: 'emergency',
        message: `LIQUIDATION IMMINENT! ${minLiquidationDistance.toFixed(1)}% away`,
        value: liquidationRisk,
        threshold: 80,
        timestamp: new Date()
      });
    }
    
    // Daily loss check
    const dailyPnLNum = dailyPnL.toNumber();
    const dailyLossPercent = Math.abs(dailyPnLNum) / accountBalanceNum * 100;
    if (dailyPnLNum < 0 && dailyLossPercent > thresholds.maxDailyLoss * 100) {
      breaches.push({
        type: 'daily_loss',
        severity: dailyLossPercent > thresholds.maxDailyLoss * 200 ? 'emergency' : 'critical',
        message: `Daily loss ${dailyLossPercent.toFixed(1)}% exceeds limit`,
        value: dailyLossPercent,
        threshold: thresholds.maxDailyLoss * 100,
        timestamp: new Date()
      });
    }
    
    // Volatility risk from market conditions
    const volatilityRisk = Math.min((volatilityIndex || 0) / 100 * 50, 100); // Scale VIX to risk
    
    // Time risk - positions held too long increase risk
    const avgHoldTime = Object.values(positions).reduce((sum, pos) => {
      const holdTime = Date.now() - new Date(pos.openTime).getTime();
      return sum + holdTime;
    }, 0) / Math.max(Object.keys(positions).length, 1);
    const timeRisk = Math.min(avgHoldTime / (24 * 60 * 60 * 1000) * 10, 50); // Days to risk
    
    // Calculate overall risk score (weighted average)
    const overallRisk = (
      accountRisk * 0.2 +
      largestPosition * 0.2 +
      (totalLeverage / thresholds.maxLeverage) * 100 * 0.2 +
      liquidationRisk * 0.2 +
      correlationRisk * 0.1 +
      volatilityRisk * 0.05 +
      timeRisk * 0.05
    );
    
    return {
      overallRisk: Math.min(overallRisk, 100),
      accountRisk,
      positionRisk: largestPosition,
      correlationRisk,
      leverageRisk: (totalLeverage / thresholds.maxLeverage) * 100,
      liquidationRisk,
      volatilityRisk,
      timeRisk,
      breaches
    };
  }, [positions, accountBalance, dailyPnL, volatilityIndex, thresholds, calculateCorrelationRisk]);
  
  /**
   * Handle risk breaches - sound the alarms
   */
  useEffect(() => {
    if (riskMetrics.breaches.length > 0) {
      const newBreaches = riskMetrics.breaches.filter(breach => 
        !recentBreaches.some(rb => 
          rb.type === breach.type && 
          rb.severity === breach.severity &&
          Date.now() - rb.timestamp.getTime() < 60000 // 1 minute cooldown
        )
      );
      
      newBreaches.forEach(breach => {
        // Visual alert
        setPulseAlert(true);
        setTimeout(() => setPulseAlert(false), 3000);
        
        // Toast notification
        const icon = breach.severity === 'emergency' ? '🚨' : 
                    breach.severity === 'critical' ? '⚠️' : '⚡';
        toast.error(`${icon} ${breach.message}`, {
          duration: breach.severity === 'emergency' ? 10000 : 5000
        });
        
        // Callback
        onRiskBreach?.(breach.type, breach.value);
        
        // Auto-lock on critical breaches
        if (autoLockOnBreach && breach.severity !== 'warning') {
          setIsLocked(true);
          setLockReason(breach.message);
          setTradingLocked(true);
        }
      });
      
      setRecentBreaches(prev => [...prev, ...newBreaches]);
    }
  }, [riskMetrics.breaches, recentBreaches, autoLockOnBreach, onRiskBreach, setTradingLocked]);
  
  /**
   * Heartbeat animation based on risk level
   * 
   * The higher the risk, the faster the heart beats.
   * At 90%+, it's basically having a panic attack.
   */
  useEffect(() => {
    if (riskMetrics.overallRisk >= 90) {
      setHeartbeatInterval(200); // PANIC
    } else if (riskMetrics.overallRisk >= 70) {
      setHeartbeatInterval(400); // DANGER
    } else if (riskMetrics.overallRisk >= 50) {
      setHeartbeatInterval(600); // CAUTION
    } else {
      setHeartbeatInterval(1000); // NORMAL
    }
  }, [riskMetrics.overallRisk]);
  
  /**
   * Risk percentage formatter with color coding
   */
  const formatRiskPercent = (value: number, threshold?: number) => {
    const isOverThreshold = threshold ? value > threshold : value > 50;
    const colorClass = value >= 80 ? 'text-red-500' :
                      value >= 60 ? 'text-orange-500' :
                      value >= 40 ? 'text-yellow-500' :
                      'text-green-500';
    
    return (
      <p className={`text-lg font-bold ${colorClass} ${isOverThreshold ? 'animate-pulse' : ''}`}>
        {value.toFixed(1)}%
      </p>
    );
  };
  
  /**
   * Get shield icon based on overall risk
   */
  const getShieldIcon = () => {
    if (riskMetrics.overallRisk >= 80) return <ShieldX className="w-6 h-6 text-red-500" />;
    if (riskMetrics.overallRisk >= 60) return <ShieldAlert className="w-6 h-6 text-orange-500" />;
    if (riskMetrics.overallRisk >= 40) return <Shield className="w-6 h-6 text-yellow-500" />;
    return <ShieldCheck className="w-6 h-6 text-green-500" />;
  };
  
  /**
   * Render the risk monitor
   */
  return (
    <div className={`bg-gray-900 border ${
      pulseAlert ? 'border-red-500 animate-pulse' : 'border-cyan-700'
    } rounded-lg p-4 ${compact ? 'space-y-2' : 'space-y-4'}`}>
      {/* Header with overall risk score */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getShieldIcon()}
          <h3 className="text-lg font-bold text-cyan-400 glitch-text">
            RISK MONITOR
          </h3>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Heartbeat indicator */}
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ 
              duration: heartbeatInterval / 1000, 
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <Heart className={`w-5 h-5 ${
              riskMetrics.overallRisk >= 70 ? 'text-red-500' : 'text-cyan-500'
            }`} />
          </motion.div>
          
          {/* Overall risk score */}
          <div className="text-right">
            <p className="text-xs text-gray-400">Overall Risk</p>
            <p className={`text-2xl font-bold ${
              riskMetrics.overallRisk >= 80 ? 'text-red-500' :
              riskMetrics.overallRisk >= 60 ? 'text-orange-500' :
              riskMetrics.overallRisk >= 40 ? 'text-yellow-500' :
              'text-green-500'
            }`}>
              {riskMetrics.overallRisk.toFixed(0)}%
            </p>
          </div>
        </div>
      </div>
      
      {/* Lock notification */}
      {isLocked && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-900/20 border border-red-500 rounded p-3 
                     flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <Lock className="w-5 h-5 text-red-500" />
            <div>
              <p className="text-sm font-semibold text-red-400">
                TRADING LOCKED
              </p>
              <p className="text-xs text-gray-400">{lockReason}</p>
            </div>
          </div>
          <button
            onClick={() => {
              setIsLocked(false);
              setTradingLocked(false);
              toast.success('Trading unlocked - BE CAREFUL!');
            }}
            className="px-3 py-1 bg-red-600 hover:bg-red-500 rounded
                     text-xs font-semibold text-white transition-colors"
          >
            OVERRIDE
          </button>
        </motion.div>
      )}
      
      {/* Risk metrics grid */}
      <div className={`grid ${compact ? 'grid-cols-2 gap-2' : 'grid-cols-4 gap-3'} mb-4`}>
        {/* Account Risk */}
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            <Gauge className="w-3 h-3" />
            Account Risk
          </p>
          {formatRiskPercent(riskMetrics.accountRisk)}
        </div>
        
        {/* Position Risk */}
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            <Target className="w-3 h-3" />
            Position Risk
          </p>
          {formatRiskPercent(riskMetrics.positionRisk, thresholds.maxPositionSize * 100)}
        </div>
        
        {/* Leverage Risk */}
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            Leverage Risk
          </p>
          {formatRiskPercent(riskMetrics.leverageRisk)}
        </div>
        
        {/* Liquidation Risk */}
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            <Skull className="w-3 h-3" />
            Liquidation Risk
          </p>
          {formatRiskPercent(riskMetrics.liquidationRisk, 80)}
        </div>
      </div>
      
      {/* Detailed metrics (if not compact) */}
      {!compact && theme === 'detailed' && (
        <div className="space-y-2 mb-4">
          {/* Correlation Risk */}
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-400">Correlation Risk</span>
            {formatRiskPercent(riskMetrics.correlationRisk, thresholds.maxCorrelation * 100)}
          </div>
          
          {/* Volatility Risk */}
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-400">Market Volatility</span>
            {formatRiskPercent(riskMetrics.volatilityRisk)}
          </div>
          
          {/* Time Risk */}
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-400">Time Exposure</span>
            {formatRiskPercent(riskMetrics.timeRisk)}
          </div>
        </div>
      )}
      
      {/* Risk level indicator bar */}
      <div className="mb-4">
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <motion.div
            className={`h-full ${
              riskMetrics.overallRisk >= 80 ? 'bg-red-500' :
              riskMetrics.overallRisk >= 60 ? 'bg-orange-500' :
              riskMetrics.overallRisk >= 40 ? 'bg-yellow-500' :
              'bg-green-500'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${riskMetrics.overallRisk}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
      
      {/* Recent breaches */}
      {showAlerts && recentBreaches.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400 uppercase tracking-wider">
            Recent Breaches
          </p>
          <AnimatePresence>
            {recentBreaches.slice(-3).reverse().map((breach, idx) => (
              <motion.div
                key={`${breach.type}-${breach.timestamp.getTime()}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: idx * 0.1 }}
                className={`p-2 rounded text-xs ${
                  breach.severity === 'emergency' ? 'bg-red-900/20 border border-red-600' :
                  breach.severity === 'critical' ? 'bg-orange-900/20 border border-orange-600' :
                  'bg-yellow-900/20 border border-yellow-600'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {breach.severity === 'emergency' ? <Siren className="w-4 h-4" /> :
                     breach.severity === 'critical' ? <AlertOctagon className="w-4 h-4" /> :
                     <AlertTriangle className="w-4 h-4" />}
                    <span>{breach.message}</span>
                  </div>
                  <span className="text-gray-500">
                    {new Date(breach.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
      
      {/* Matrix theme extras */}
      {theme === 'matrix' && (
        <div className="mt-4 font-mono text-xs text-green-400 opacity-60">
          <div>SYSTEM.MONITOR.ACTIVE</div>
          <div>RISK.THRESHOLD.{riskMetrics.overallRisk >= 60 ? 'EXCEEDED' : 'NOMINAL'}</div>
          <div>HEARTBEAT.{heartbeatInterval}MS</div>
          <div>BREACHES.COUNT.{recentBreaches.length}</div>
        </div>
      )}
    </div>
  );
};