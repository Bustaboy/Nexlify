// Location: /src/components/tabs/OverviewTab/index.tsx
// Overview Tab - Main Neural Interface Display

import React from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { MetricsGrid } from '../../dashboard/MetricsGrid';
import { TradingControls } from '../../dashboard/TradingControls';
import { EquityChart } from '../../charts/EquityChart';
import { PnLDistribution } from '../../charts/PnLDistribution';
import { AnomalyRadar } from '../../charts/AnomalyRadar';
import { PositionMatrix } from '../../charts/PositionMatrix';
import { DexLiquidityPool } from '../../defi/DexLiquidityPool';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';

export const OverviewTab: React.FC = () => {
  const { metrics, emergencyProtocol, apiConfigs, defiPositions } = useDashboardStore();
  const theme = useThemeService();

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-6 space-y-6"
    >
      {/* Emergency Protocol Alert */}
      {emergencyProtocol.isActive && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/10 border-2 border-red-500 rounded-xl p-4 flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <div>
              <p className="font-bold text-red-400">EMERGENCY PROTOCOL ACTIVE</p>
              <p className="text-sm text-gray-400">All trading operations suspended</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Trading Controls */}
      <TradingControls />

      {/* Primary Metrics */}
      <MetricsGrid />

      {/* Dynamic Position Matrix */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
        style={{ borderColor: `${theme.colors.neural}44` }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
            <GlitchText theme={theme.colors}>Dynamic Position Matrix</GlitchText>
          </h3>
          <div className="text-sm text-gray-400">
            Auto-adjusting based on profitability algorithms
          </div>
        </div>
        <PositionMatrix />
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <EquityChart />
        <PnLDistribution />
      </div>

      {/* DeFi Integration Panel */}
      {apiConfigs.some(c => c.isDex) && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.info}44` }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.info }}>
              <GlitchText theme={theme.colors}>DeFi Neural Grid</GlitchText>
            </h3>
            <div className="text-sm">
              <span className="text-gray-400">Active Pools: </span>
              <span className="font-bold" style={{ color: theme.colors.success }}>
                {defiPositions.liquidityPools.length}
              </span>
            </div>
          </div>
          <DexLiquidityPool />
        </motion.div>
      )}

      {/* Anomaly Detection */}
      <AnomalyRadar />
    </motion.div>
  );
};
