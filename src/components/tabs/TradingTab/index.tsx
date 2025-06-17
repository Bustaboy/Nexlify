// Location: /src/components/tabs/TradingTab/index.tsx
// Trading Tab - Neural Combat Operations Center

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Bot, Network, Zap, Info, GitMerge, Flame } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { AIStrategyGrid } from '../../trading/AIStrategyGrid';
import { MLControlPanel } from '../../trading/MLControlPanel';
import { StrategyPerformanceChart } from '../../charts/StrategyPerformanceChart';
import { ArbitrageMatrix } from '../../trading/ArbitrageMatrix';
import { QuantumPositionOptimizer } from '../../trading/QuantumPositionOptimizer';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

export const TradingTab: React.FC = () => {
  const { 
    tradingActive, 
    setTradingActive, 
    emergencyProtocol,
    strategies,
    apiConfigs,
    defiPositions
  } = useDashboardStore();
  
  const theme = useThemeService();
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  const activeExchanges = apiConfigs.filter(c => c.isActive);
  const activeDexes = apiConfigs.filter(c => c.isDex && c.isActive);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-6 space-y-6"
    >
      {/* Trading Control Panel */}
      <div className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6" 
           style={{ borderColor: `${theme.colors.primary}44` }}>
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 mb-6">
          <h3 className="text-2xl font-bold uppercase tracking-wider" style={{ color: theme.colors.primary }}>
            <GlitchText theme={theme.colors}>Neural Trading Control</GlitchText>
          </h3>
          
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <motion.button
              onClick={() => setTradingActive(!tradingActive)}
              disabled={emergencyProtocol.isActive}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`relative px-8 py-3 rounded-lg font-bold uppercase tracking-widest transition-all duration-300 ${
                emergencyProtocol.isActive ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              style={{
                backgroundColor: tradingActive ? `${theme.colors.danger}33` : 'rgba(75, 85, 99, 0.5)',
                borderWidth: '2px',
                borderColor: tradingActive ? theme.colors.danger : '#4B5563',
                color: tradingActive ? theme.colors.danger : '#9CA3AF',
                boxShadow: tradingActive ? `0 0 30px ${theme.colors.danger}88` : undefined
              }}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${
                  tradingActive ? 'animate-pulse' : ''
                }`} style={{
                  backgroundColor: tradingActive ? theme.colors.danger : '#4B5563',
                  boxShadow: tradingActive ? `0 0 10px ${theme.colors.danger}` : undefined
                }} />
                <span className="text-sm">
                  {tradingActive ? 'TRADING ACTIVE' : 'TRADING OFFLINE'}
                </span>
                <Zap className="w-4 h-4" />
              </div>
            </motion.button>
            
            {tradingActive && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="font-mono text-sm uppercase tracking-wider"
                style={{ color: theme.colors.danger }}
              >
                Neural protocols engaged
              </motion.div>
            )}
          </div>
        </div>

        {/* Exchange Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-800/50 border rounded-xl p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">CEX Active</span>
              <Network className="w-4 h-4" style={{ color: theme.colors.primary }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.success }}>
              {activeExchanges.filter(e => !e.isDex).length}
            </div>
          </div>
          
          <div className="bg-gray-800/50 border rounded-xl p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">DEX Active</span>
              <GitMerge className="w-4 h-4" style={{ color: theme.colors.info }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.info }}>
              {activeDexes.length}
            </div>
          </div>
          
          <div className="bg-gray-800/50 border rounded-xl p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Liquidity Pools</span>
              <Flame className="w-4 h-4" style={{ color: theme.colors.neural }} />
            </div>
            <div className="text-2xl font-bold" style={{ color: theme.colors.neural }}>
              {defiPositions.liquidityPools.length}
            </div>
          </div>
        </div>

        {/* AI Protocol Info */}
        <div className="bg-gray-800/50 border-2 rounded-xl p-4" style={{ borderColor: `${theme.colors.info}44` }}>
          <div className="flex items-center space-x-2 mb-2">
            <Info className="w-5 h-5" style={{ color: theme.colors.info }} />
            <h4 className="text-lg font-bold" style={{ color: theme.colors.info }}>AI Protocol System</h4>
          </div>
          <p className="text-sm text-gray-400">
            Each AI protocol represents a specialized neural network trained for specific market conditions. 
            Protocols operate independently but share risk management parameters. ML models are continuously 
            updated based on market performance, with reinforcement learning optimizing strategy parameters.
            <span className="block mt-2 font-bold" style={{ color: theme.colors.warning }}>
              Dynamic position sizing adjusts in real-time based on profitability calculations and quantum analysis.
            </span>
          </p>
        </div>
      </div>

      {/* ML/RL Controls */}
      <MLControlPanel />

      {/* Quantum Position Optimizer */}
      <QuantumPositionOptimizer />

      {/* Strategy Grid */}
      <AIStrategyGrid 
        onStrategySelect={setSelectedStrategy}
        selectedStrategy={selectedStrategy}
      />

      {/* Multi-Exchange Arbitrage Matrix */}
      <ArbitrageMatrix />

      {/* AI Protocol Performance Chart */}
      <StrategyPerformanceChart />
    </motion.div>
  );
};
