// Location: /src/components/defi/DexLiquidityPool/index.tsx
// DEX Liquidity Pool - Where Chrome Meets DeFi Chaos

import React from 'react';
import { motion } from 'framer-motion';
import { Layers, TrendingUp, AlertTriangle } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

export const DexLiquidityPool: React.FC = () => {
  const { defiPositions } = useDashboardStore();
  const theme = useThemeService();

  // Calculate pool metrics
  const totalLiquidity = defiPositions.liquidityPools.reduce((acc, pool) => acc + pool.value, 0);
  const totalImpermanentLoss = defiPositions.liquidityPools.reduce((acc, pool) => acc + (pool.impermanentLoss || 0), 0);
  const avgAPY = defiPositions.liquidityPools.length > 0
    ? defiPositions.liquidityPools.reduce((acc, pool) => acc + pool.apy, 0) / defiPositions.liquidityPools.length
    : 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Pool Overview */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-gray-800/50 border rounded-xl p-6"
        style={{ borderColor: `${theme.colors.info}33` }}
      >
        <h4 className="text-lg font-bold mb-4 flex items-center space-x-2" style={{ color: theme.colors.info }}>
          <Layers className="w-5 h-5" />
          <span>Liquidity Pool Overview</span>
        </h4>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
            <span className="text-sm text-gray-400">Total Value Locked</span>
            <span className="font-bold text-lg" style={{ color: theme.colors.primary }}>
              {formatCredits(totalLiquidity)}
            </span>
          </div>
          
          <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
            <span className="text-sm text-gray-400">Average APY</span>
            <span className="font-bold text-lg" style={{ color: theme.colors.success }}>
              {formatPercent(avgAPY)}
            </span>
          </div>
          
          <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
            <span className="text-sm text-gray-400">Impermanent Loss</span>
            <span className="font-bold text-lg" style={{ color: theme.colors.danger }}>
              {formatCredits(totalImpermanentLoss)}
            </span>
          </div>
          
          <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
            <span className="text-sm text-gray-400">Active Pools</span>
            <span className="font-bold text-lg" style={{ color: theme.colors.warning }}>
              {defiPositions.liquidityPools.length}
            </span>
          </div>
        </div>
      </motion.div>

      {/* Top Performing Pools */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-gray-800/50 border rounded-xl p-6"
        style={{ borderColor: `${theme.colors.success}33` }}
      >
        <h4 className="text-lg font-bold mb-4 flex items-center space-x-2" style={{ color: theme.colors.success }}>
          <TrendingUp className="w-5 h-5" />
          <span>Top Performing Pools</span>
        </h4>
        
        <div className="space-y-3">
          {defiPositions.liquidityPools
            .sort((a, b) => b.apy - a.apy)
            .slice(0, 4)
            .map((pool, idx) => (
              <motion.div
                key={pool.poolId}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg hover:bg-gray-900/70 transition-colors"
              >
                <div>
                  <div className="font-mono font-bold text-sm">{pool.pair}</div>
                  <div className="text-xs text-gray-500">{pool.protocol}</div>
                </div>
                <div className="text-right">
                  <div className="font-bold" style={{ color: theme.colors.success }}>
                    {formatPercent(pool.apy)} APY
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatCredits(pool.value)}
                  </div>
                </div>
              </motion.div>
            ))}
        </div>
        
        {defiPositions.liquidityPools.length === 0 && (
          <div className="text-center py-8 text-gray-500 text-sm">
            No active liquidity pools
          </div>
        )}
      </motion.div>
    </div>
  );
};
