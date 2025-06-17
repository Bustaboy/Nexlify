// Location: /src/components/defi/ImpermanentLossCalculator/index.tsx
// Impermanent Loss Calculator - The Truth Behind the Chrome Dreams

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Calculator, AlertTriangle, TrendingDown, Info } from 'lucide-react';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface ILScenario {
  priceChange: number;
  ilPercent: number;
  holdValue: number;
  lpValue: number;
  difference: number;
}

export const ImpermanentLossCalculator: React.FC = () => {
  const theme = useThemeService();
  const [token1Amount, setToken1Amount] = useState(1000);
  const [token2Amount, setToken2Amount] = useState(1000);
  const [token1Price, setToken1Price] = useState(100);
  const [token2Price, setToken2Price] = useState(1);
  const [scenarios, setScenarios] = useState<ILScenario[]>([]);
  const [customPriceChange, setCustomPriceChange] = useState(50);

  useEffect(() => {
    calculateScenarios();
  }, [token1Amount, token2Amount, token1Price, token2Price]);

  const calculateIL = (priceRatio: number): number => {
    return 2 * Math.sqrt(priceRatio) / (1 + priceRatio) - 1;
  };

  const calculateScenarios = () => {
    const initialValue = token1Amount * token1Price + token2Amount * token2Price;
    const priceChanges = [-50, -25, 0, 25, 50, 100, 200];
    
    const newScenarios = priceChanges.map(change => {
      const newToken1Price = token1Price * (1 + change / 100);
      const priceRatio = newToken1Price / token1Price;
      const ilFactor = calculateIL(priceRatio);
      const ilPercent = ilFactor * 100;
      
      // Calculate new LP position
      const k = token1Amount * token2Amount;
      const newToken1Amount = Math.sqrt(k * token2Price / newToken1Price);
      const newToken2Amount = k / newToken1Amount;
      const lpValue = newToken1Amount * newToken1Price + newToken2Amount * token2Price;
      
      // Calculate HODL value
      const holdValue = token1Amount * newToken1Price + token2Amount * token2Price;
      
      return {
        priceChange: change,
        ilPercent: Math.abs(ilPercent),
        holdValue,
        lpValue,
        difference: lpValue - holdValue
      };
    });
    
    setScenarios(newScenarios);
  };

  const getILSeverity = (ilPercent: number): { color: string; text: string } => {
    if (ilPercent > 20) return { color: theme.colors.danger, text: 'SEVERE' };
    if (ilPercent > 10) return { color: theme.colors.warning, text: 'HIGH' };
    if (ilPercent > 5) return { color: theme.colors.info, text: 'MODERATE' };
    return { color: theme.colors.success, text: 'LOW' };
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.danger}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
          <GlitchText theme={theme.colors}>Impermanent Loss Matrix</GlitchText>
        </h3>
        <Calculator className="w-5 h-5" style={{ color: theme.colors.danger }} />
      </div>

      {/* Warning Banner */}
      <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
        <div className="flex items-start space-x-2">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: theme.colors.danger }} />
          <div className="text-sm text-gray-300">
            <p className="font-bold mb-1" style={{ color: theme.colors.danger }}>Reality Check, Choom</p>
            <p className="text-xs text-gray-400">
              Impermanent Loss is the silent killer of LP dreams. It's not a maybe - it's a when. 
              Every price movement bleeds your position, and the only question is how much chrome you'll lose.
            </p>
          </div>
        </div>
      </div>

      {/* Input Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <h4 className="text-sm font-bold mb-3" style={{ color: theme.colors.primary }}>Initial Position</h4>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Token 1 Amount</label>
              <input
                type="number"
                value={token1Amount}
                onChange={(e) => setToken1Amount(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800/80 border rounded-lg text-sm font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">Token 1 Price (€)</label>
              <input
                type="number"
                value={token1Price}
                onChange={(e) => setToken1Price(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800/80 border rounded-lg text-sm font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              />
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold mb-3" style={{ color: theme.colors.primary }}>&nbsp;</h4>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Token 2 Amount</label>
              <input
                type="number"
                value={token2Amount}
                onChange={(e) => setToken2Amount(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800/80 border rounded-lg text-sm font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">Token 2 Price (€)</label>
              <input
                type="number"
                value={token2Price}
                onChange={(e) => setToken2Price(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800/80 border rounded-lg text-sm font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Initial Value Display */}
      <div className="mb-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">Initial Position Value</span>
          <span className="text-2xl font-bold" style={{ color: theme.colors.primary }}>
            {formatCredits(token1Amount * token1Price + token2Amount * token2Price)}
          </span>
        </div>
      </div>

      {/* IL Scenarios */}
      <div className="space-y-3">
        <h4 className="text-sm font-bold" style={{ color: theme.colors.warning }}>
          Impermanent Loss Scenarios
        </h4>
        
        {scenarios.map((scenario, idx) => {
          const severity = getILSeverity(scenario.ilPercent);
          const isProfit = scenario.priceChange > 0;
          
          return (
            <motion.div
              key={scenario.priceChange}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="p-3 bg-gray-800/50 border rounded-lg hover:bg-gray-800/70 transition-all"
              style={{ borderColor: `${severity.color}33` }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <TrendingDown className="w-4 h-4" style={{ color: severity.color }} />
                  <span className="font-mono font-bold text-sm">
                    {scenario.priceChange > 0 ? '+' : ''}{scenario.priceChange}% Price Change
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded" style={{
                    backgroundColor: `${severity.color}22`,
                    color: severity.color
                  }}>
                    {severity.text} IL
                  </span>
                </div>
                <span className="font-bold" style={{ color: theme.colors.danger }}>
                  -{scenario.ilPercent.toFixed(2)}% IL
                </span>
              </div>
              
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div>
                  <span className="text-gray-500">HODL Value</span>
                  <div className="font-mono">{formatCredits(scenario.holdValue)}</div>
                </div>
                <div>
                  <span className="text-gray-500">LP Value</span>
                  <div className="font-mono">{formatCredits(scenario.lpValue)}</div>
                </div>
                <div>
                  <span className="text-gray-500">Difference</span>
                  <div className="font-mono font-bold" style={{ color: theme.colors.danger }}>
                    {formatCredits(scenario.difference)}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* The Hard Truth */}
      <div className="mt-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.info}33` }}>
        <div className="flex items-center space-x-2 mb-2">
          <Info className="w-4 h-4" style={{ color: theme.colors.info }} />
          <span className="text-sm font-bold" style={{ color: theme.colors.info }}>The Hard Truth</span>
        </div>
        <div className="space-y-1 text-xs text-gray-400">
          <p>• IL is permanent if you withdraw when prices have diverged</p>
          <p>• Trading fees and rewards might compensate, but often don't</p>
          <p>• The more volatile the pair, the higher the IL risk</p>
          <p>• Stablecoin pairs minimize IL but offer lower yields</p>
        </div>
      </div>
    </motion.div>
  );
};
