// Location: /src/components/trading/AIStrategyGrid/index.tsx
// AI Strategy Grid - Neural Combat Protocol Matrix

import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Shield, TrendingUp } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface Props {
  onStrategySelect: (strategyId: string | null) => void;
  selectedStrategy: string | null;
}

export const AIStrategyGrid: React.FC<Props> = ({ onStrategySelect, selectedStrategy }) => {
  const { strategies, tradingActive, emergencyProtocol, setStrategies } = useDashboardStore();
  const theme = useThemeService();

  const toggleStrategy = (strategyId: string) => {
    if (emergencyProtocol.isActive) return;
    
    setStrategies(prev => prev.map(s => 
      s.id === strategyId ? { ...s, isActive: !s.isActive } : s
    ));
  };

  const getStrategyIcon = (index: number) => {
    const icons = [Brain, Zap, Shield];
    const Icon = icons[index % icons.length];
    return Icon;
  };

  const getStrategyColor = (index: number) => {
    return [theme.colors.success, theme.colors.info, theme.colors.warning][index % 3];
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {strategies.map((strategy, index) => {
        const Icon = getStrategyIcon(index);
        const color = getStrategyColor(index);
        const isSelected = selectedStrategy === strategy.id;
        
        return (
          <motion.div
            key={strategy.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onStrategySelect(isSelected ? null : strategy.id)}
            className={`relative bg-gray-800/60 border-2 rounded-xl p-6 cursor-pointer transition-all hover:bg-gray-800/80 ${
              isSelected ? 'ring-2' : ''
            }`}
            style={{
              borderColor: tradingActive && strategy.isActive && !emergencyProtocol.isActive
                ? `${color}88` 
                : `${theme.colors.primary}44`,
              boxShadow: tradingActive && strategy.isActive && !emergencyProtocol.isActive
                ? `0 0 30px ${color}66`
                : undefined,
              ringColor: isSelected ? color : undefined
            }}
          >
            {/* Neural Activity Indicator */}
            {tradingActive && strategy.isActive && !emergencyProtocol.isActive && (
              <div className="absolute top-2 right-2">
                <motion.div
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
              </div>
            )}

            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-lg font-bold" style={{ color }}>{strategy.codename}</h4>
                <div className="flex items-center space-x-2 mt-1">
                  <span className="text-gray-500 text-xs">Neural Load: {strategy.neuralLoad.toFixed(0)}%</span>
                  {tradingActive && strategy.isActive && !emergencyProtocol.isActive && (
                    <span className="text-xs font-mono animate-pulse" style={{ color: theme.colors.success }}>
                      • LIVE
                    </span>
                  )}
                </div>
              </div>
              <div className="p-2 rounded-lg" style={{
                backgroundColor: `${color}33`,
                color: color
              }}>
                <Icon className="w-5 h-5" />
              </div>
            </div>

            <p className="text-xs text-gray-400 mb-4">{strategy.description}</p>

            {/* Performance Metrics */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">P&L</span>
                <span className="font-bold" style={{ 
                  color: strategy.pnl >= 0 ? theme.colors.success : theme.colors.danger 
                }}>
                  {formatCredits(strategy.pnl)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Hit Rate</span>
                <span className={`font-bold`} style={{
                  color: strategy.hitRate >= 70 ? theme.colors.success : 
                        strategy.hitRate >= 50 ? theme.colors.warning : theme.colors.danger
                }}>
                  {strategy.hitRate.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Sharpe</span>
                <span className="font-bold" style={{ color: theme.colors.primary }}>
                  {strategy.sharpe.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">ML Model</span>
                <span className="font-bold uppercase text-xs" style={{ color: theme.colors.neural }}>
                  {strategy.mlModel}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Confidence</span>
                <div className="flex items-center space-x-2">
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${strategy.confidence * 100}%` }}
                      transition={{ duration: 1 }}
                      style={{ 
                        background: `linear-gradient(to right, ${color}, ${theme.colors.info})`
                      }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">
                    {(strategy.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="mt-4 pt-4 border-t" style={{ borderColor: `${theme.colors.primary}22` }}>
              <div className="grid grid-cols-2 gap-2">
                <button 
                  disabled={emergencyProtocol.isActive}
                  onClick={(e) => {
                    e.stopPropagation();
                    // Configure modal would go here
                  }}
                  className="py-2 bg-gray-800/80 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-white transition-colors text-sm font-mono disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Configure
                </button>
                <button 
                  disabled={emergencyProtocol.isActive}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleStrategy(strategy.id);
                  }}
                  className={`py-2 rounded-lg transition-colors text-sm font-mono disabled:opacity-50 disabled:cursor-not-allowed`}
                  style={{
                    backgroundColor: strategy.isActive ? `${theme.colors.danger}33` : `${theme.colors.success}33`,
                    color: strategy.isActive ? theme.colors.danger : theme.colors.success
                  }}
                >
                  {strategy.isActive ? 'Disable' : 'Enable'}
                </button>
              </div>
            </div>

            {/* Selection Indicator */}
            {isSelected && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 border-2 rounded-xl pointer-events-none"
                style={{ 
                  borderColor: color,
                  boxShadow: `inset 0 0 20px ${color}33`
                }}
              />
            )}
          </motion.div>
        );
      })}
    </div>
  );
};
