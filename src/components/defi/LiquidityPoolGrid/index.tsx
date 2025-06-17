// Location: /src/components/defi/LiquidityPoolGrid/index.tsx
// Liquidity Pool Grid - The DeFi Battlefield Map

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GitMerge, TrendingUp, TrendingDown, AlertTriangle, Info, Zap } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface Props {
  selectedProtocol: string;
}

export const LiquidityPoolGrid: React.FC<Props> = ({ selectedProtocol }) => {
  const { defiPositions } = useDashboardStore();
  const theme = useThemeService();
  const [expandedPool, setExpandedPool] = useState<string | null>(null);

  const filteredPools = selectedProtocol === 'all' 
    ? defiPositions.liquidityPools
    : defiPositions.liquidityPools.filter(pool => pool.protocol === selectedProtocol);

  const getPoolHealth = (pool: any) => {
    const ilPercent = (pool.impermanentLoss / pool.value) * 100;
    if (ilPercent > 10) return 'critical';
    if (ilPercent > 5) return 'warning';
    return 'healthy';
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'critical': return theme.colors.danger;
      case 'warning': return theme.colors.warning;
      default: return theme.colors.success;
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.info}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.info }}>
          <GlitchText theme={theme.colors}>Liquidity Pool Matrix</GlitchText>
        </h3>
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <GitMerge className="w-4 h-4" />
          <span>{filteredPools.length} active pools</span>
        </div>
      </div>

      {filteredPools.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <GitMerge className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm">No liquidity pools found</p>
          <p className="text-xs mt-2">Add positions through the DeFi interface</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          <AnimatePresence>
            {filteredPools.map((pool, idx) => {
              const health = getPoolHealth(pool);
              const healthColor = getHealthColor(health);
              const isExpanded = expandedPool === pool.poolId;
              
              return (
                <motion.div
                  key={pool.poolId}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: idx * 0.05 }}
                  className="relative bg-gray-800/60 border-2 rounded-lg p-4 cursor-pointer hover:bg-gray-800/80 transition-all"
                  style={{ 
                    borderColor: `${healthColor}66`,
                    boxShadow: health === 'critical' ? `0 0 20px ${healthColor}44` : undefined
                  }}
                  onClick={() => setExpandedPool(isExpanded ? null : pool.poolId)}
                >
                  {/* Health Indicator */}
                  <div className="absolute top-2 right-2">
                    <motion.div
                      animate={{
                        scale: health === 'critical' ? [1, 1.2, 1] : 1,
                        opacity: health === 'critical' ? [0.5, 1, 0.5] : 1
                      }}
                      transition={{ duration: 2, repeat: health === 'critical' ? Infinity : 0 }}
                    >
                      {health === 'critical' ? (
                        <AlertTriangle className="w-4 h-4" style={{ color: healthColor }} />
                      ) : health === 'warning' ? (
                        <Info className="w-4 h-4" style={{ color: healthColor }} />
                      ) : (
                        <Zap className="w-4 h-4" style={{ color: healthColor }} />
                      )}
                    </motion.div>
                  </div>

                  {/* Pool Header */}
                  <div className="mb-3">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-mono font-bold text-sm">{pool.pair}</span>
                      <span className="text-xs px-2 py-0.5 bg-gray-900/50 rounded" style={{ color: theme.colors.primary }}>
                        {pool.protocol}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      Pool ID: {pool.poolId.slice(0, 8)}...
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-xs text-gray-500">Value</div>
                      <div className="font-bold text-sm" style={{ color: theme.colors.primary }}>
                        {formatCredits(pool.value)}
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-2">
                      <div className="text-xs text-gray-500">APY</div>
                      <div className="font-bold text-sm" style={{ color: theme.colors.success }}>
                        {formatPercent(pool.apy)}
                      </div>
                    </div>
                  </div>

                  {/* Performance Indicator */}
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">24h Performance</span>
                    <div className="flex items-center space-x-1">
                      {pool.volume24h > pool.volume24hPrev ? (
                        <TrendingUp className="w-3 h-3" style={{ color: theme.colors.success }} />
                      ) : (
                        <TrendingDown className="w-3 h-3" style={{ color: theme.colors.danger }} />
                      )}
                      <span className="text-xs font-mono" style={{ 
                        color: pool.volume24h > pool.volume24hPrev ? theme.colors.success : theme.colors.danger 
                      }}>
                        {formatPercent((pool.volume24h - pool.volume24hPrev) / pool.volume24hPrev * 100)}
                      </span>
                    </div>
                  </div>

                  {/* Impermanent Loss Warning */}
                  {pool.impermanentLoss > 0 && (
                    <div className="p-2 bg-red-500/10 border border-red-500/30 rounded text-xs">
                      <span className="text-red-400">IL: {formatCredits(pool.impermanentLoss)}</span>
                    </div>
                  )}

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-3 pt-3 border-t space-y-2"
                        style={{ borderColor: `${theme.colors.primary}22` }}
                      >
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">24h Volume</span>
                          <span className="font-mono">{formatCredits(pool.volume24h)}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Total Fees</span>
                          <span className="font-mono" style={{ color: theme.colors.success }}>
                            {formatCredits(pool.fees || 0)}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Your Share</span>
                          <span className="font-mono">{formatPercent(pool.share || 0.1)}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Token Ratio</span>
                          <span className="font-mono">{pool.tokenRatio || '50:50'}</span>
                        </div>
                        
                        <div className="mt-3 pt-3 border-t" style={{ borderColor: `${theme.colors.primary}22` }}>
                          <button className="w-full py-2 bg-gray-800/80 hover:bg-gray-700 rounded text-xs font-mono transition-colors">
                            Manage Position
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

      {/* Pool Statistics */}
      {filteredPools.length > 0 && (
        <div className="mt-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.info}33` }}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-xs text-gray-500">Healthy Pools</div>
              <div className="text-lg font-bold" style={{ color: theme.colors.success }}>
                {filteredPools.filter(p => getPoolHealth(p) === 'healthy').length}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">At Risk</div>
              <div className="text-lg font-bold" style={{ color: theme.colors.warning }}>
                {filteredPools.filter(p => getPoolHealth(p) === 'warning').length}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Critical</div>
              <div className="text-lg font-bold" style={{ color: theme.colors.danger }}>
                {filteredPools.filter(p => getPoolHealth(p) === 'critical').length}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Avg IL</div>
              <div className="text-lg font-bold" style={{ color: theme.colors.danger }}>
                {formatPercent(
                  filteredPools.reduce((acc, p) => acc + (p.impermanentLoss / p.value) * 100, 0) / filteredPools.length
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};
