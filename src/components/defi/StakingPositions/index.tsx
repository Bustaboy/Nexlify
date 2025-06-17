// Location: /src/components/defi/StakingPositions/index.tsx
// Staking Positions - The Long Game in the Digital Wasteland

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Lock, Unlock, TrendingUp, Calendar, Shield } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface Props {
  selectedProtocol: string;
}

export const StakingPositions: React.FC<Props> = ({ selectedProtocol }) => {
  const { defiPositions } = useDashboardStore();
  const theme = useThemeService();
  const [expandedStake, setExpandedStake] = useState<string | null>(null);

  const filteredStakes = selectedProtocol === 'all'
    ? defiPositions.stakingPositions
    : defiPositions.stakingPositions.filter(stake => stake.protocol === selectedProtocol);

  const totalStaked = filteredStakes.reduce((acc, stake) => acc + stake.amount, 0);
  const totalRewards = filteredStakes.reduce((acc, stake) => acc + stake.rewards, 0);
  const lockedStakes = filteredStakes.filter(stake => stake.lockEndDate && stake.lockEndDate > Date.now());

  const getTimeRemaining = (endDate: number): string => {
    const now = Date.now();
    const diff = endDate - now;
    if (diff <= 0) return 'Unlocked';
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    
    if (days > 0) return `${days}d ${hours}h`;
    return `${hours}h`;
  };

  const getStakeStatus = (stake: any): { text: string; color: string; icon: React.ElementType } => {
    if (stake.lockEndDate && stake.lockEndDate > Date.now()) {
      return { text: 'LOCKED', color: theme.colors.danger, icon: Lock };
    }
    if (stake.autoCompound) {
      return { text: 'AUTO-COMPOUND', color: theme.colors.success, icon: TrendingUp };
    }
    return { text: 'FLEXIBLE', color: theme.colors.info, icon: Unlock };
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.neural}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
          <GlitchText theme={theme.colors}>Neural Staking Grid</GlitchText>
        </h3>
        <Zap className="w-5 h-5" style={{ color: theme.colors.neural }} />
      </div>

      {/* Staking Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 border rounded-lg p-3" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="text-xs text-gray-400 mb-1">Total Staked</div>
          <div className="text-xl font-bold" style={{ color: theme.colors.primary }}>
            {formatCredits(totalStaked)}
          </div>
        </div>
        <div className="bg-gray-800/50 border rounded-lg p-3" style={{ borderColor: `${theme.colors.success}33` }}>
          <div className="text-xs text-gray-400 mb-1">Total Rewards</div>
          <div className="text-xl font-bold" style={{ color: theme.colors.success }}>
            +{formatCredits(totalRewards)}
          </div>
        </div>
        <div className="bg-gray-800/50 border rounded-lg p-3" style={{ borderColor: `${theme.colors.warning}33` }}>
          <div className="text-xs text-gray-400 mb-1">Active Stakes</div>
          <div className="text-xl font-bold" style={{ color: theme.colors.warning }}>
            {filteredStakes.length}
          </div>
        </div>
        <div className="bg-gray-800/50 border rounded-lg p-3" style={{ borderColor: `${theme.colors.danger}33` }}>
          <div className="text-xs text-gray-400 mb-1">Locked</div>
          <div className="text-xl font-bold" style={{ color: theme.colors.danger }}>
            {lockedStakes.length}
          </div>
        </div>
      </div>

      {/* Staking Positions List */}
      {filteredStakes.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Shield className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm">No staking positions found</p>
          <p className="text-xs mt-2">Stake tokens to earn passive rewards</p>
        </div>
      ) : (
        <div className="space-y-3">
          <AnimatePresence>
            {filteredStakes.map((stake, idx) => {
              const status = getStakeStatus(stake);
              const StatusIcon = status.icon;
              const isExpanded = expandedStake === stake.id;
              
              return (
                <motion.div
                  key={stake.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: idx * 0.05 }}
                  className="bg-gray-800/60 border rounded-lg p-4 cursor-pointer hover:bg-gray-800/80 transition-all"
                  style={{ 
                    borderColor: `${status.color}44`,
                    boxShadow: stake.lockEndDate && stake.lockEndDate > Date.now() 
                      ? `0 0 15px ${status.color}22` 
                      : undefined
                  }}
                  onClick={() => setExpandedStake(isExpanded ? null : stake.id)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <StatusIcon className="w-5 h-5" style={{ color: status.color }} />
                      <div>
                        <div className="font-mono font-bold">{stake.token}</div>
                        <div className="text-xs text-gray-500">{stake.protocol}</div>
                      </div>
                      <span className="text-xs px-2 py-1 rounded font-mono" style={{
                        backgroundColor: `${status.color}22`,
                        color: status.color
                      }}>
                        {status.text}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold" style={{ color: theme.colors.success }}>
                        {formatPercent(stake.apy)}
                      </div>
                      <div className="text-xs text-gray-500">APY</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-sm mb-2">
                    <div>
                      <span className="text-gray-500 text-xs">Staked</span>
                      <div className="font-mono">{formatCredits(stake.amount)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">Earned</span>
                      <div className="font-mono" style={{ color: theme.colors.success }}>
                        +{formatCredits(stake.rewards)}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-500 text-xs">
                        {stake.lockEndDate ? 'Unlock In' : 'Duration'}
                      </span>
                      <div className="font-mono flex items-center space-x-1">
                        <Calendar className="w-3 h-3" />
                        <span>
                          {stake.lockEndDate 
                            ? getTimeRemaining(stake.lockEndDate)
                            : `${Math.floor((Date.now() - stake.startDate) / 86400000)}d`
                          }
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {stake.lockEndDate && (
                    <div className="mt-2">
                      <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full"
                          initial={{ width: 0 }}
                          animate={{ 
                            width: `${Math.min(100, ((Date.now() - stake.startDate) / (stake.lockEndDate - stake.startDate)) * 100)}%` 
                          }}
                          transition={{ duration: 1 }}
                          style={{ backgroundColor: status.color }}
                        />
                      </div>
                    </div>
                  )}

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t space-y-3"
                        style={{ borderColor: `${theme.colors.primary}22` }}
                      >
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Start Date</span>
                            <span className="font-mono">
                              {new Date(stake.startDate).toLocaleDateString()}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Validator</span>
                            <span className="font-mono">{stake.validator || 'Auto-select'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Total Value</span>
                            <span className="font-mono">
                              {formatCredits(stake.amount + stake.rewards)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Next Reward</span>
                            <span className="font-mono">
                              {new Date(Date.now() + 86400000).toLocaleTimeString()}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <button 
                            disabled={stake.lockEndDate && stake.lockEndDate > Date.now()}
                            className="py-2 bg-gray-800/80 hover:bg-gray-700 rounded text-xs font-mono transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {stake.lockEndDate && stake.lockEndDate > Date.now() ? 'Locked' : 'Unstake'}
                          </button>
                          <button className="py-2 border rounded text-xs font-mono transition-colors hover:bg-green-500/20"
                                  style={{ 
                                    borderColor: theme.colors.success,
                                    color: theme.colors.success
                                  }}>
                            Claim Rewards
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

      {/* Staking Tips */}
      <div className="mt-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.neural}33` }}>
        <h4 className="text-sm font-bold mb-2 flex items-center space-x-2" style={{ color: theme.colors.neural }}>
          <Shield className="w-4 h-4" />
          <span>STAKING INTEL</span>
        </h4>
        <div className="space-y-1 text-xs text-gray-400">
          <p>• Locked staking typically offers higher APY but restricts access to funds</p>
          <p>• Auto-compound maximizes returns through the power of exponential growth</p>
          <p>• Always verify validator reputation before delegating large amounts</p>
          <p>• Consider staking derivatives for liquidity while earning rewards</p>
        </div>
      </div>
    </motion.div>
  );
};
