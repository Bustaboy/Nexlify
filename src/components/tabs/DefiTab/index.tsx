// Location: /src/components/tabs/DefiTab/index.tsx
// DeFi Tab - Decentralized Neural Finance Grid

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { GitMerge, Layers, Zap, Shield, TrendingUp, Flame, Coins, Network } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { NeuralCard } from '../../common/NeuralCard';
import { LiquidityPoolGrid } from '../../defi/LiquidityPoolGrid';
import { YieldFarmingPanel } from '../../defi/YieldFarmingPanel';
import { StakingPositions } from '../../defi/StakingPositions';
import { ImpermanentLossCalculator } from '../../defi/ImpermanentLossCalculator';
import { DexAggregator } from '../../defi/DexAggregator';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

export const DefiTab: React.FC = () => {
  const { defiPositions, metrics, apiConfigs } = useDashboardStore();
  const theme = useThemeService();
  const [selectedProtocol, setSelectedProtocol] = useState<string>('all');

  // Calculate DeFi metrics
  const defiMetrics = {
    totalValueLocked: defiPositions.liquidityPools.reduce((acc, pool) => acc + pool.value, 0) +
                     defiPositions.stakingPositions.reduce((acc, stake) => acc + stake.amount, 0),
    totalYield: defiPositions.yieldFarms.reduce((acc, farm) => acc + farm.pendingRewards, 0),
    avgAPY: defiPositions.liquidityPools.length > 0 
      ? defiPositions.liquidityPools.reduce((acc, pool) => acc + pool.apy, 0) / defiPositions.liquidityPools.length
      : 0,
    impermanentLoss: defiPositions.liquidityPools.reduce((acc, pool) => acc + (pool.impermanentLoss || 0), 0),
    gasSpent: Math.random() * 1000, // Simulated
    protocols: [...new Set([
      ...defiPositions.liquidityPools.map(p => p.protocol),
      ...defiPositions.stakingPositions.map(p => p.protocol),
      ...defiPositions.yieldFarms.map(p => p.protocol)
    ])]
  };

  const activeDexes = apiConfigs.filter(c => c.isDex && c.isActive);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-6 space-y-6"
    >
      {/* DeFi Overview */}
      <div className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6" 
           style={{ borderColor: `${theme.colors.info}44` }}>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold uppercase tracking-wider" style={{ color: theme.colors.info }}>
            <GlitchText theme={theme.colors}>DeFi Neural Grid</GlitchText>
          </h3>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-400">Connected DEXs:</span>
            <span className="font-bold" style={{ color: theme.colors.success }}>
              {activeDexes.length}
            </span>
          </div>
        </div>

        {/* Protocol Filter */}
        <div className="flex items-center space-x-2 mb-6 overflow-x-auto pb-2">
          <button
            onClick={() => setSelectedProtocol('all')}
            className={`px-4 py-2 rounded-lg font-mono text-sm transition-all ${
              selectedProtocol === 'all' ? 'border-2' : 'bg-gray-800/50 border border-gray-700'
            }`}
            style={{
              borderColor: selectedProtocol === 'all' ? theme.colors.primary : undefined,
              backgroundColor: selectedProtocol === 'all' ? `${theme.colors.primary}33` : undefined,
              color: selectedProtocol === 'all' ? 'white' : '#9CA3AF'
            }}
          >
            All Protocols
          </button>
          {defiMetrics.protocols.map(protocol => (
            <button
              key={protocol}
              onClick={() => setSelectedProtocol(protocol)}
              className={`px-4 py-2 rounded-lg font-mono text-sm transition-all whitespace-nowrap ${
                selectedProtocol === protocol ? 'border-2' : 'bg-gray-800/50 border border-gray-700'
              }`}
              style={{
                borderColor: selectedProtocol === protocol ? theme.colors.primary : undefined,
                backgroundColor: selectedProtocol === protocol ? `${theme.colors.primary}33` : undefined,
                color: selectedProtocol === protocol ? 'white' : '#9CA3AF'
              }}
            >
              {protocol}
            </button>
          ))}
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">TVL</span>
              <Layers className="w-4 h-4" style={{ color: theme.colors.primary }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.primary }}>
              {formatCredits(defiMetrics.totalValueLocked)}
            </div>
          </div>
          
          <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.success}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Yield Earned</span>
              <Coins className="w-4 h-4" style={{ color: theme.colors.success }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.success }}>
              {formatCredits(defiMetrics.totalYield)}
            </div>
          </div>
          
          <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.warning}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Avg APY</span>
              <TrendingUp className="w-4 h-4" style={{ color: theme.colors.warning }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.warning }}>
              {formatPercent(defiMetrics.avgAPY)}
            </div>
          </div>
          
          <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.danger}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">IL</span>
              <Shield className="w-4 h-4" style={{ color: theme.colors.danger }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.danger }}>
              {formatCredits(defiMetrics.impermanentLoss)}
            </div>
          </div>
        </div>
      </div>

      {/* DeFi Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <NeuralCard
          title="Active Pools"
          value={defiPositions.liquidityPools.length}
          icon={GitMerge}
          color={theme.colors.info}
          subtitle="Liquidity Positions"
          theme={theme.colors}
        />
        <NeuralCard
          title="Staking Positions"
          value={defiPositions.stakingPositions.length}
          icon={Zap}
          color={theme.colors.neural}
          subtitle="Active Stakes"
          theme={theme.colors}
        />
        <NeuralCard
          title="Yield Farms"
          value={defiPositions.yieldFarms.length}
          icon={Flame}
          color={theme.colors.warning}
          subtitle="Farming Positions"
          theme={theme.colors}
        />
        <NeuralCard
          title="Gas Spent"
          value={defiMetrics.gasSpent}
          icon={Network}
          color={theme.colors.danger}
          subtitle="Transaction Costs"
          theme={theme.colors}
        />
      </div>

      {/* DEX Aggregator */}
      <DexAggregator />

      {/* Liquidity Pools */}
      <LiquidityPoolGrid selectedProtocol={selectedProtocol} />

      {/* Yield Farming Panel */}
      <YieldFarmingPanel selectedProtocol={selectedProtocol} />

      {/* Staking Positions */}
      <StakingPositions selectedProtocol={selectedProtocol} />

      {/* Impermanent Loss Calculator */}
      <ImpermanentLossCalculator />
    </motion.div>
  );
};
