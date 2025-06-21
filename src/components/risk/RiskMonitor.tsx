// src/components/risk/RiskMonitor.tsx
// NEXLIFY RISK MONITOR - Your guardian against financial annihilation
// Last sync: 2025-06-19 | "Risk is what's left when you think you've thought of everything"

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
  Shield,
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
  const { marketData, volatilityIndex } = useMarketStore();
  
  // Risk thresholds
  const thresholds: RiskThresholds = {
    maxLeverage: riskLimits.maxLeverage || 10,
    maxPositionSize: riskLimits.maxPositionSize || 0.2, // 20% of account
    maxDailyLoss: riskLimits.maxDailyLoss || 0.05, // 5% daily loss
    maxDrawdown: riskLimits.maxDrawdown || 0.15, // 15% drawdown
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
      (sum, pos) => sum + Math.abs(pos.quantity * pos.currentPrice), 0
    );
    const accountRisk = accountBalance.total > 0 
      ? (totalPositionValue / accountBalance.total) * 100 
      : 0;
    
    // Position risk - largest single position
    const largestPosition = Math.max(
      ...Object.values(positions).map(p => 
        Math.abs(p.quantity * p.currentPrice) / accountBalance.total
      ), 0
    ) * 100;
    
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
      const distance = Math.abs(pos.currentPrice - pos.liquidationPrice) / pos.currentPrice;
      return distance * 100;
    });
    const minLiquidationDistance = Math.min(...liquidationDistances, 100);
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
    const dailyLossPercent = Math.abs(dailyPnL) / accountBalance.total * 100;
    if (dailyPnL < 0 && dailyLossPercent > thresholds.maxDailyLoss * 100) {
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
    const volatilityRisk = Math.min(volatilityIndex / 100 * 50, 100); // Scale VIX to risk
    
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
        
        // Auto-lock on critical/emergency
        if (autoLockOnBreach && breach.severity !== 'warning') {
          setIsLocked(true);
          setLockReason(breach.message);
          setTradingLocked(true);
        }
      });
      
      setRecentBreaches(prev => [...prev, ...newBreaches].slice(-10));
    }
  }, [riskMetrics.breaches, recentBreaches, autoLockOnBreach, onRiskBreach, setTradingLocked]);
  
  /**
   * Heartbeat animation speed based on risk
   * 
   * Like a real heart, beats faster under stress
   */
  useEffect(() => {
    if (riskMetrics.overallRisk > 80) {
      setHeartbeatInterval(300); // Panic
    } else if (riskMetrics.overallRisk > 60) {
      setHeartbeatInterval(500); // Stress
    } else if (riskMetrics.overallRisk > 40) {
      setHeartbeatInterval(800); // Alert
    } else {
      setHeartbeatInterval(1200); // Calm
    }
  }, [riskMetrics.overallRisk]);
  
  /**
   * Get risk color based on level
   */
  const getRiskColor = (risk: number): string => {
    if (risk >= 80) return 'text-red-500';
    if (risk >= 60) return 'text-orange-500';
    if (risk >= 40) return 'text-yellow-500';
    if (risk >= 20) return 'text-green-500';
    return 'text-green-400';
  };
  
  /**
   * Get risk icon based on level
   */
  const getRiskIcon = (risk: number) => {
    if (risk >= 80) return <Skull className="w-6 h-6 text-red-500 animate-pulse" />;
    if (risk >= 60) return <ShieldX className="w-6 h-6 text-orange-500" />;
    if (risk >= 40) return <ShieldAlert className="w-6 h-6 text-yellow-500" />;
    return <ShieldCheck className="w-6 h-6 text-green-500" />;
  };
  
  /**
   * Format risk percentage with appropriate styling
   */
  const formatRiskPercent = (value: number, threshold?: number): JSX.Element => {
    const isOverThreshold = threshold && value > threshold;
    return (
      <span className={`font-mono font-bold ${
        isOverThreshold ? 'text-red-400 animate-pulse' : getRiskColor(value)
      }`}>
        {value.toFixed(1)}%
      </span>
    );
  };
  
  return (
    <div className={`
      bg-gray-900/50 border border-cyan-900/30 rounded-lg
      ${compact ? 'p-3' : 'p-4'}
      ${pulseAlert ? 'ring-2 ring-red-500 animate-pulse' : ''}
      ${isLocked ? 'ring-2 ring-red-600' : ''}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          RISK MONITOR
          {isLocked && <Lock className="w-4 h-4 text-red-500" />}
        </h3>
        
        <div className="flex items-center gap-2">
          {/* Overall risk score */}
          <div className="flex items-center gap-2">
            {getRiskIcon(riskMetrics.overallRisk)}
            <span className={`text-2xl font-bold ${getRiskColor(riskMetrics.overallRisk)}`}>
              {riskMetrics.overallRisk.toFixed(0)}
            </span>
          </div>
          
          {/* Heartbeat animation */}
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ 
              duration: heartbeatInterval / 1000, 
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <Heart className={`w-5 h-5 ${getRiskColor(riskMetrics.overallRisk)}`} />
          </motion.div>
        </div>
      </div>
      
      {/* Trading lock alert */}
      {isLocked && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mb-4 p-3 bg-red-900/30 border border-red-900/50 rounded
                   flex items-center gap-3"
        >
          <Siren className="w-5 h-5 text-red-400 animate-pulse" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-red-400">
              TRADING LOCKED - RISK BREACH
            </p>
            <p className="text-xs text-red-300">{lockReason}</p>
          </div>
          <button
            onClick={() => {
              setIsLocked(false);
              setTradingLocked(false);
              toast.success('Trading unlocked - trade carefully!');
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
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Safe</span>
          <span>Caution</span>
          <span>Danger</span>
          <span>Critical</span>
        </div>
      </div>
      
      {/* Recent breaches */}
      {showAlerts && recentBreaches.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-gray-400 mb-1">Recent Breaches</p>
          <AnimatePresence>
            {recentBreaches.slice(-3).reverse().map((breach, index) => (
              <motion.div
                key={`${breach.type}-${breach.timestamp.getTime()}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className={`
                  text-xs px-2 py-1 rounded flex items-center gap-2
                  ${breach.severity === 'emergency' ? 'bg-red-900/50 text-red-300' :
                    breach.severity === 'critical' ? 'bg-orange-900/50 text-orange-300' :
                    'bg-yellow-900/50 text-yellow-300'}
                `}
              >
                <AlertTriangle className="w-3 h-3" />
                <span className="flex-1">{breach.message}</span>
                <span className="text-xs opacity-50">
                  {new Date(breach.timestamp).toLocaleTimeString()}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
      
      {/* Risk management tips */}
      {!compact && (
        <div className="mt-4 text-center">
          <p className="text-xs text-gray-500 italic">
            {riskMetrics.overallRisk >= 80 ? 
              "🚨 EXTREME RISK - Consider closing positions immediately" :
             riskMetrics.overallRisk >= 60 ?
              "⚠️ HIGH RISK - Reduce leverage and position sizes" :
             riskMetrics.overallRisk >= 40 ?
              "⚡ MODERATE RISK - Monitor positions closely" :
             riskMetrics.overallRisk >= 20 ?
              "✅ ACCEPTABLE RISK - Stay disciplined" :
              "🛡️ LOW RISK - Good risk management"
            }
          </p>
        </div>
      )}
      
      {/* Matrix theme overlay */}
      {theme === 'matrix' && (
        <div className="absolute inset-0 pointer-events-none opacity-10">
          <div className="h-full w-full"
               style={{
                 backgroundImage: `repeating-linear-gradient(
                   90deg,
                   rgba(255, 0, 0, 0.1),
                   rgba(255, 0, 0, 0.1) 2px,
                   transparent 2px,
                   transparent 4px
                 )`
               }}
          />
        </div>
      )}
    </div>
  );
};

/**
 * RISK MONITOR WISDOM:
 * 
 * 1. The heartbeat animation isn't cute - it's functional. Your
 *    subconscious notices rhythm changes before your conscious mind
 *    processes the numbers.
 * 
 * 2. Correlation risk is the silent killer. Everyone thinks they're
 *    diversified until the market proves they're not.
 * 
 * 3. Auto-lock on breach has saved more accounts than any other
 *    feature. Sometimes you need to be protected from yourself.
 * 
 * 4. The override button requires conscious action. That pause has
 *    prevented countless revenge trades.
 * 
 * 5. Time risk increases with holding duration because the market
 *    has more chances to move against you. Fresh eyes see clearer.
 * 
 * 6. Visual risk scoring (colors, animations) bypasses analytical
 *    paralysis. Red pulsing = get out. No thinking required.
 * 
 * Remember: Risk management isn't about avoiding risk - it's about
 * taking the right risks in the right size at the right time.
 * 
 * "The market can remain irrational longer than you can remain solvent."
 * - John Maynard Keynes (who learned this the hard way)
 */
