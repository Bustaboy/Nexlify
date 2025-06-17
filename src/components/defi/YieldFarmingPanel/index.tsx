// Location: /src/components/defi/YieldFarmingPanel/index.tsx
// Yield Farming Panel - Where Digital Dreams Meet DeFi Reality

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Flame, TrendingUp, Clock, AlertCircle, Coins } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface Props {
  selectedProtocol: string;
}

export const YieldFarmingPanel: React.FC<Props> = ({ selectedProtocol }) => {
  const { defiPositions } = useDashboardStore();
  const theme = useThemeService();
  const [selectedFarm, setSelectedFarm] = useState<string | null>(null);

  const filteredFarms = selectedProtocol === 'all'
    ? defiPositions.yieldFarms
    : defiPositions.yieldFarms.filter(farm => farm.protocol === selectedProtocol);

  const totalStaked = filteredFarms.reduce((acc, farm) => acc + farm.stakedAmount, 0);
  const totalRewards = filteredFarms.reduce((acc, farm) => acc + farm.pendingRewards, 0);
  const avgAPR = filteredFarms.length > 0
    ? filteredFarms.reduce((acc, farm) => acc + farm.apr, 0) / filteredFarms.length
    : 0;

  const getFarmRisk = (apr: number): { level: string; color: string } => {
    if (apr > 1000) return { level: 'DEGEN', color: theme.colors.danger };
    if (apr > 500) return { level: 'HIGH', color: theme.colors.warning };
    if (apr > 100) return { level: 'MEDIUM', color: theme.colors.info };
    return { level: 'LOW', color: theme.colors.success };
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.warning}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.warning }}>
          <GlitchText theme={theme.colors}>Yield Farming Operations</GlitchText>
        </h3>
        <Flame className="w-5 h-5 animate-pulse" style={{ color: theme.colors.warning }} />
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Total Staked</span>
            <Coins className="w-4 h-4" style={{ color: theme.colors.primary }} />
          </div>
          <div className="text-2xl font-bold" style={{ color: theme.colors.primary }}>
            {formatCredits(totalStaked)}
          </div>
        </div>

        <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.success}33` }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Pending Rewards</span>
            <TrendingUp className="w-4 h-4" style={{ color: theme.colors.success }} />
          </div>
          <div className="text-2xl font-bold" style={{ color: theme.colors.success }}>
            {formatCredits(totalRewards)}
          </div>
        </div>

        <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.warning}33` }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Average APR</span>
            <Flame className="w-4 h-4" style={{ color: theme.colors.warning }} />
          </div>
          <div className="text-2xl font-bold" style={{ color: theme.colors.warning }}>
            {formatPercent(avgAPR)}
          </div>
        </div>
      </div>

      {/* Farming Positions */}
      {filteredFarms.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Flame className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm">No active yield farms</p>
          <p className="text-xs mt-2">Start farming to earn rewards</p>
        </div>
      ) : (
        <div className="space-y-4">
          <AnimatePresence>
            {filteredFarms.map((farm, idx) => {
              const risk = getFarmRisk(farm.apr);
              const isSelected = selectedFarm === farm.farmId;
              
              return (
                <motion.div
                  key={farm.farmId}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ delay: idx * 0.05 }}
                  className="bg-gray-800/60 border rounded-lg p-4 cursor-pointer hover:bg-gray-800/80 transition-all"
                  style={{ 
                    borderColor: isSelected ? risk.color : `${theme.colors.primary}33`,
                    boxShadow: isSelected ? `0 0 20px ${risk.color}44` : undefined
                  }}
                  onClick={() => setSelectedFarm(isSelected ? null : farm.farmId)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="flex flex-col">
                        <span className="font-mono font-bold">{farm.pair}</span>
                        <span className="text-xs text-gray-500">{farm.protocol}</span>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded font-bold`} style={{
                        backgroundColor: `${risk.color}22`,
                        color: risk.color
                      }}>
                        {risk.level} RISK
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold" style={{ color: theme.colors.warning }}>
                        {formatPercent(farm.apr)}
                      </div>
                      <div className="text-xs text-gray-500">APR</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div>
                      <span className="text-gray-500 text-xs">Staked</span>
                      <div className="font-mono">{formatCredits(farm.stakedAmount)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">Rewards</span>
                      <div className="font-mono" style={{ color: theme.colors.success }}>
                        {formatCredits(farm.pendingRewards)}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">Duration</span>
                      <div className="font-mono flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>{farm.stakingDuration}d</span>
                      </div>
                    </div>
                  </div>

                  <AnimatePresence>
                    {isSelected && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t space-y-3"
                        style={{ borderColor: `${theme.colors.primary}22` }}
                      >
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Reward Token</span>
                          <span className="font-mono">{farm.rewardToken}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Lock Period</span>
                          <span className="font-mono">{farm.lockPeriod || 'None'}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Next Harvest</span>
                          <span className="font-mono">
                            {new Date(Date.now() + Math.random() * 86400000).toLocaleTimeString()}
                          </span>
                        </div>
                        
                        {farm.apr > 1000 && (
                          <div className="p-2 bg-red-500/10 border border-red-500/30 rounded">
                            <div className="flex items-center space-x-2 text-xs text-red-400">
                              <AlertCircle className="w-4 h-4" />
                              <span>
                                Extremely high APR - potential rug pull risk. DYOR!
                              </span>
                            </div>
                          </div>
                        )}

                        <div className="grid grid-cols-2 gap-2 mt-4">
                          <button className="py-2 bg-gray-800/80 hover:bg-gray-700 rounded text-xs font-mono transition-colors">
                            Compound
                          </button>
                          <button className="py-2 border rounded text-xs font-mono transition-colors hover:bg-green-500/20"
                                  style={{ 
                                    borderColor: theme.colors.success,
                                    color: theme.colors.success
                                  }}>
                            Harvest
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      {/* Risk Warning */}
      {filteredFarms.some(f => f.apr > 500) && (
        <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
          <div className="flex items-center space-x-2 text-sm text-yellow-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>
              High APR farms carry significant risk. Always verify contract audits and TVL before investing.
              Remember: If it seems too good to be true, it probably is.
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
};
