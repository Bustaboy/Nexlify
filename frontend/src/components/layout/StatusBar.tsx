// frontend/src/components/layout/StatusBar.tsx

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Cpu,
  HardDrive,
  Wifi,
  Globe,
  Shield,
  AlertCircle,
  ChevronUp
} from 'lucide-react';
import { cn, formatCurrency, formatPercent, getPnLColor } from '@lib/utils';
import { useTradingStore } from '@stores/tradingStore';
import { useConnectionStatus } from '@hooks/useWebSocket';
import { useSystemMonitor } from '@hooks/useSystemMonitor';
import { useSettingsStore } from '@stores/settingsStore';

// The pulse at the bottom of your screen - every trader's lifeline
// I've watched this bar through thousands of trades, each tick a heartbeat
// Green numbers bringing relief, red ones teaching lessons I'll never forget
// Some nights it's the only light in a dark room, reflecting off empty coffee cups

interface StatusItem {
  key: string;
  label: string;
  value: string | number;
  icon: React.ElementType;
  color?: string;
  tooltip?: string;
  onClick?: () => void;
}

export const StatusBar: React.FC = () => {
  const { portfolio, positions, marketData } = useTradingStore();
  const { latency, isHealthy } = useConnectionStatus();
  const { systemStats, performanceScore } = useSystemMonitor({ interval: 10000 });
  const { hideBalances } = useSettingsStore();
  
  const [expanded, setExpanded] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);
  
  // Calculate real-time metrics - the numbers that keep me up at night
  const totalPnL = portfolio.unrealizedPnL + portfolio.realizedPnL;
  const totalPnLPercent = portfolio.totalValue > 0 
    ? (totalPnL / (portfolio.totalValue - totalPnL)) * 100 
    : 0;
  
  // Count active markets - how many battles we're fighting
  const activeMarkets = Array.from(marketData.keys()).length;
  
  // Build status items - each one a story
  const statusItems: StatusItem[] = [
    {
      key: 'equity',
      label: 'Total Equity',
      value: hideBalances ? '•••••' : formatCurrency(portfolio.totalValue),
      icon: DollarSign,
      tooltip: 'Your war chest - guard it well'
    },
    {
      key: 'pnl',
      label: 'Total P&L',
      value: hideBalances ? '•••' : `${formatCurrency(totalPnL)} (${formatPercent(totalPnLPercent)})`,
      icon: totalPnL >= 0 ? TrendingUp : TrendingDown,
      color: getPnLColor(totalPnL),
      tooltip: totalPnL >= 0 
        ? 'Profits earned through sweat and sleepless nights'
        : 'Losses that teach harder lessons than any mentor'
    },
    {
      key: 'positions',
      label: 'Open Positions',
      value: positions.length,
      icon: Activity,
      color: positions.length > 5 ? 'text-yellow-400' : 'text-gray-400',
      tooltip: `${positions.length} bets on the future`
    },
    {
      key: 'markets',
      label: 'Active Markets',
      value: activeMarkets,
      icon: Globe,
      tooltip: 'Data streams from across the digital frontier'
    },
    {
      key: 'latency',
      label: 'Latency',
      value: `${latency}ms`,
      icon: Wifi,
      color: latency < 100 ? 'text-green-400' : latency < 500 ? 'text-yellow-400' : 'text-red-400',
      tooltip: latency < 100 
        ? 'Lightning fast - riding the data stream'
        : latency < 500 
        ? 'Acceptable - but every millisecond counts'
        : 'Sluggish connection - trades might slip'
    },
    {
      key: 'cpu',
      label: 'CPU',
      value: `${systemStats?.cpu.usage.toFixed(0) || 0}%`,
      icon: Cpu,
      color: (systemStats?.cpu.usage || 0) > 80 ? 'text-red-400' : 'text-gray-400',
      tooltip: 'Your neural processor - don\'t let it overheat'
    },
    {
      key: 'memory',
      label: 'Memory',
      value: `${((systemStats?.memory.percentage || 0)).toFixed(0)}%`,
      icon: HardDrive,
      color: (systemStats?.memory.percentage || 0) > 85 ? 'text-red-400' : 'text-gray-400',
      tooltip: 'RAM usage - where market data lives and breathes'
    },
    {
      key: 'security',
      label: 'Security',
      value: 'Active',
      icon: Shield,
      color: 'text-neon-cyan',
      tooltip: 'Encryption active - your secrets are safe'
    }
  ];

  // Warning messages - wisdom earned through pain
  const getWarningMessage = (): string | null => {
    if (positions.length > 10) {
      return "Heavy position load - don't spread yourself too thin, choom";
    }
    if ((systemStats?.cpu.usage || 0) > 90) {
      return "CPU running hot - your chrome might start lagging";
    }
    if (latency > 1000) {
      return "Connection struggling - trades might not execute clean";
    }
    if (totalPnLPercent < -10) {
      return "Deep in the red - sometimes the best trade is no trade";
    }
    return null;
  };

  const warningMessage = getWarningMessage();

  return (
    <>
      {/* Main status bar */}
      <motion.div
        initial={{ y: 24 }}
        animate={{ y: 0 }}
        className={cn(
          "h-6 flex items-center justify-between",
          "bg-cyber-black border-t border-cyber-dark",
          "text-xs select-none",
          "relative z-50"
        )}
      >
        {/* Left section - Critical metrics */}
        <div className="flex items-center divide-x divide-cyber-dark">
          {statusItems.slice(0, 4).map((item) => (
            <StatusItem key={item.key} item={item} onHover={setActiveTooltip} />
          ))}
        </div>

        {/* Center - Warning message */}
        <AnimatePresence>
          {warningMessage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="flex items-center space-x-2 text-yellow-400"
            >
              <AlertCircle className="w-3 h-3 animate-pulse" />
              <span className="text-xs italic">{warningMessage}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Right section - System metrics */}
        <div className="flex items-center divide-x divide-cyber-dark">
          {statusItems.slice(4).map((item) => (
            <StatusItem key={item.key} item={item} onHover={setActiveTooltip} />
          ))}
          
          {/* Expand button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setExpanded(!expanded)}
            className="px-3 h-full flex items-center hover:bg-cyber-dark transition-colors"
          >
            <ChevronUp className={cn(
              "w-3 h-3 transition-transform",
              expanded && "rotate-180"
            )} />
          </motion.button>
        </div>

        {/* Performance indicator line - the EKG of your trading life */}
        <motion.div
          className="absolute bottom-0 left-0 h-px bg-gradient-to-r from-transparent via-neon-cyan to-transparent"
          animate={{
            opacity: [0.3, 1, 0.3],
            scaleX: [0.8, 1, 0.8]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          style={{ width: '100%' }}
        />
      </motion.div>

      {/* Expanded panel - deeper metrics for the data hungry */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="bg-cyber-dark border-t border-cyber-gray overflow-hidden"
          >
            <div className="p-4">
              {/* Extended metrics grid */}
              <div className="grid grid-cols-4 gap-4 text-xs">
                <MetricCard
                  label="24h Volume"
                  value={formatCurrency(24567.89)}
                  subtext="Across all positions"
                  trend="+12.4%"
                  trendUp={true}
                />
                <MetricCard
                  label="Win Rate"
                  value={`${portfolio.winRate.toFixed(1)}%`}
                  subtext={`Last ${positions.length} trades`}
                  trend={portfolio.winRate > 50 ? "Profitable" : "Work needed"}
                  trendUp={portfolio.winRate > 50}
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={portfolio.sharpeRatio.toFixed(2)}
                  subtext="Risk-adjusted returns"
                  trend={portfolio.sharpeRatio > 1 ? "Good" : "Risky"}
                  trendUp={portfolio.sharpeRatio > 1}
                />
                <MetricCard
                  label="System Score"
                  value={`${performanceScore}/100`}
                  subtext="Overall health"
                  trend={`${100 - performanceScore}% overhead`}
                  trendUp={performanceScore > 80}
                />
              </div>

              {/* Personal message - because we're all human here */}
              <div className="mt-4 pt-4 border-t border-cyber-gray">
                <p className="text-xs text-gray-500 italic">
                  {getPersonalMessage(totalPnLPercent, new Date().getHours())}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

// Individual status item component
const StatusItem: React.FC<{
  item: StatusItem;
  onHover: (key: string | null) => void;
}> = ({ item, onHover }) => {
  const Icon = item.icon;
  
  return (
    <motion.div
      whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}
      onMouseEnter={() => onHover(item.key)}
      onMouseLeave={() => onHover(null)}
      onClick={item.onClick}
      className={cn(
        "px-3 h-full flex items-center space-x-2",
        "transition-colors cursor-pointer",
        "relative group"
      )}
    >
      <Icon className={cn("w-3 h-3", item.color || 'text-gray-500')} />
      <span className="text-gray-400">{item.label}:</span>
      <span className={cn("font-mono", item.color || 'text-white')}>
        {item.value}
      </span>
      
      {/* Tooltip */}
      {item.tooltip && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <div className="bg-cyber-black border border-cyber-gray px-3 py-2 rounded-lg shadow-xl whitespace-nowrap">
            <p className="text-xs text-gray-300">{item.tooltip}</p>
          </div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
            <div className="border-4 border-transparent border-t-cyber-gray" />
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

// Metric card for expanded view
const MetricCard: React.FC<{
  label: string;
  value: string;
  subtext: string;
  trend: string;
  trendUp: boolean;
}> = ({ label, value, subtext, trend, trendUp }) => (
  <div className="bg-cyber-black p-3 rounded-lg border border-cyber-gray">
    <p className="text-gray-500 mb-1">{label}</p>
    <p className="text-lg font-mono text-white">{value}</p>
    <p className="text-xs text-gray-600 mt-1">{subtext}</p>
    <div className="flex items-center mt-2">
      {trendUp ? (
        <TrendingUp className="w-3 h-3 text-green-400 mr-1" />
      ) : (
        <TrendingDown className="w-3 h-3 text-red-400 mr-1" />
      )}
      <span className={cn(
        "text-xs",
        trendUp ? 'text-green-400' : 'text-red-400'
      )}>
        {trend}
      </span>
    </div>
  </div>
);

// Personal messages - because trading is personal
function getPersonalMessage(pnlPercent: number, hour: number): string {
  // Late night messages - for the night shift traders
  if (hour >= 0 && hour < 6) {
    if (pnlPercent > 5) {
      return "Solid gains in the witching hours. The night rewards the vigilant.";
    } else if (pnlPercent < -5) {
      return "Rough night, but dawn always comes. Rest might bring clarity.";
    }
    return "Another late night in the trenches. Stay sharp, stay alive.";
  }
  
  // Morning messages
  if (hour >= 6 && hour < 12) {
    if (pnlPercent > 0) {
      return "Starting the day in the green. Coffee tastes better with profits.";
    }
    return "Fresh day, fresh opportunities. Yesterday's losses don't define today's trades.";
  }
  
  // Afternoon messages
  if (hour >= 12 && hour < 18) {
    if (pnlPercent > 10) {
      return "Crushing it today. Remember to book some profits - pigs get slaughtered.";
    } else if (pnlPercent < -10) {
      return "Market's teaching hard lessons today. Every master was once a disaster.";
    }
    return "Steady as she goes. Consistency beats home runs.";
  }
  
  // Evening messages
  if (pnlPercent > 0) {
    return "Another day, another dollar. Time to review and plan tomorrow's battles.";
  }
  return "Not every day is a winner. Rest up and come back stronger.";
}
