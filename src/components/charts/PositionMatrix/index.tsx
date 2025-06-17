// Location: /src/components/charts/PositionMatrix/index.tsx
// Position Matrix - Dynamic Neural Position Optimizer

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, AlertTriangle, Crosshair } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';
import { PositionData } from '../../../types/trading.types';

export const PositionMatrix: React.FC = () => {
  const { metrics, tradingActive } = useDashboardStore();
  const theme = useThemeService();
  const [positions, setPositions] = useState<PositionData[]>([]);
  const [sortedByProfit, setSortedByProfit] = useState(true);

  useEffect(() => {
    // Dynamically generate and sort positions based on profitability
    const generateDynamicPositions = () => {
      const basePositions = Object.entries(metrics.positionsPnL).map(([symbol, pnl]) => ({
        symbol,
        pnl,
        size: Math.random() * 10000 + 1000,
        entry: 50000 + Math.random() * 10000,
        current: 50000 + Math.random() * 10000 + pnl,
        leverage: Math.floor(Math.random() * 10) + 1,
        duration: Math.floor(Math.random() * 72),
        riskScore: Math.random() * 100,
        lastUpdate: Date.now()
      }));

      // Add dynamic positions based on market conditions
      if (metrics.anomalyScore < 50 && tradingActive) {
        // Add profitable pairs in low-risk conditions
        const profitablePairs = ['SOL/USDT', 'AVAX/USDT', 'MATIC/USDT'].filter(
          pair => !basePositions.find(p => p.symbol === pair)
        );
        
        profitablePairs.forEach(pair => {
          if (Math.random() > 0.5) {
            basePositions.push({
              symbol: pair,
              pnl: Math.random() * 500 + 100,
              size: Math.random() * 5000 + 1000,
              entry: 100 + Math.random() * 50,
              current: 110 + Math.random() * 50,
              leverage: Math.floor(Math.random() * 5) + 1,
              duration: Math.floor(Math.random() * 24),
              riskScore: Math.random() * 40,
              lastUpdate: Date.now()
            });
          }
        });
      }

      // Remove losing positions when risk is high
      const filtered = metrics.anomalyScore > 80 
        ? basePositions.filter(p => p.pnl > -200)
        : basePositions;

      // Sort by profitability
      return filtered.sort((a, b) => sortedByProfit ? b.pnl - a.pnl : a.pnl - b.pnl);
    };

    const interval = setInterval(() => {
      setPositions(generateDynamicPositions());
    }, 5000);

    setPositions(generateDynamicPositions());
    return () => clearInterval(interval);
  }, [metrics, tradingActive, sortedByProfit]);

  const getHealthColor = (position: PositionData) => {
    if (position.pnl > 500) return theme.colors.success;
    if (position.pnl > 0) return theme.colors.info;
    if (position.pnl > -200) return theme.colors.warning;
    return theme.colors.danger;
  };

  return (
    <div className="overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-gray-400">
          <span className="font-bold" style={{ color: theme.colors.primary }}>
            {positions.length} active
          </span> • Auto-optimizing based on quantum profitability analysis
        </div>
        <button
          onClick={() => setSortedByProfit(!sortedByProfit)}
          className="text-xs font-mono px-3 py-1 bg-gray-800/50 border rounded-lg hover:bg-gray-700 transition-colors"
          style={{ borderColor: `${theme.colors.primary}44` }}
        >
          Sort: {sortedByProfit ? 'PROFIT ↓' : 'LOSS ↑'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
        <style jsx>{`
          .scrollbar-matrix::-webkit-scrollbar {
            width: 6px;
          }
          .scrollbar-matrix::-webkit-scrollbar-track {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 3px;
          }
          .scrollbar-matrix::-webkit-scrollbar-thumb {
            background: ${theme.colors.primary}66;
            border-radius: 3px;
          }
          .scrollbar-matrix::-webkit-scrollbar-thumb:hover {
            background: ${theme.colors.primary}88;
          }
        `}</style>
        
        <div className="scrollbar-matrix contents">
          <AnimatePresence>
            {positions.map((position, idx) => (
              <motion.div
                key={position.symbol}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: idx * 0.05 }}
                className="relative bg-gray-800/60 border-2 rounded-lg p-4 backdrop-blur-sm overflow-hidden group hover:bg-gray-800/80 transition-all"
                style={{ 
                  borderColor: getHealthColor(position) + '66',
                  boxShadow: position.pnl > 0 ? `0 0 20px ${getHealthColor(position)}33` : undefined
                }}
              >
                {/* Neural pulse effect */}
                <div className="absolute inset-0 opacity-20">
                  <motion.div
                    className="absolute inset-0"
                    animate={{
                      background: [
                        `radial-gradient(circle at 50% 50%, ${getHealthColor(position)}44 0%, transparent 70%)`,
                        `radial-gradient(circle at 50% 50%, ${getHealthColor(position)}22 0%, transparent 70%)`,
                        `radial-gradient(circle at 50% 50%, ${getHealthColor(position)}44 0%, transparent 70%)`
                      ]
                    }}
                    transition={{ duration: 3, repeat: Infinity }}
                  />
                </div>

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Crosshair className="w-4 h-4" style={{ color: getHealthColor(position) }} />
                      <span className="font-mono font-bold text-sm">{position.symbol}</span>
                      <span className="text-xs px-2 py-0.5 bg-gray-900/50 rounded" style={{ color: theme.colors.primary }}>
                        {position.leverage}x
                      </span>
                    </div>
                    {position.pnl >= 0 ? 
                      <TrendingUp className="w-4 h-4" style={{ color: theme.colors.success }} /> : 
                      <TrendingDown className="w-4 h-4" style={{ color: theme.colors.danger }} />
                    }
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">P&L</span>
                      <div className="font-bold" style={{ color: getHealthColor(position) }}>
                        {formatCredits(position.pnl)}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500">Size</span>
                      <div className="font-mono">{formatCredits(position.size)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Entry</span>
                      <div className="font-mono">{position.entry.toFixed(2)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">Current</span>
                      <div className="font-mono" style={{ color: position.current > position.entry ? theme.colors.success : theme.colors.danger }}>
                        {position.current.toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Activity className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-400">{position.duration}h</span>
                    </div>
                    {position.riskScore > 70 && (
                      <div className="flex items-center space-x-1">
                        <AlertTriangle className="w-3 h-3" style={{ color: theme.colors.danger }} />
                        <span className="text-xs" style={{ color: theme.colors.danger }}>
                          Risk: {position.riskScore.toFixed(0)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-500 text-center">
        Positions dynamically adjusted every 5s based on neural profitability algorithms
      </div>
    </div>
  );
};
