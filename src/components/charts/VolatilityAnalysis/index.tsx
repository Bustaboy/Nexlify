// Location: /src/components/charts/VolatilityAnalysis/index.tsx
// Volatility Analysis - Reading the Market's Psychosis

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Activity, Zap, AlertCircle } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatPercent } from '../../../utils/dashboard.utils';

interface VolatilityData {
  timestamp: number;
  realized: number;
  implied: number;
  spread: number;
  regime: 'calm' | 'normal' | 'elevated' | 'extreme';
}

export const VolatilityAnalysis: React.FC = () => {
  const { timeSeriesData, metrics } = useDashboardStore();
  const theme = useThemeService();

  const volatilityData = useMemo(() => {
    const windowSize = 20;
    return timeSeriesData.slice(-100).map((point, idx, arr) => {
      // Calculate realized volatility (simplified)
      const startIdx = Math.max(0, idx - windowSize);
      const window = arr.slice(startIdx, idx + 1);
      
      if (window.length < 2) {
        return {
          timestamp: point.timestamp,
          realized: 15,
          implied: 18,
          spread: 3,
          regime: 'normal' as const
        };
      }

      const returns = window.slice(1).map((p, i) => 
        Math.log(p.equity / window[i].equity)
      );
      
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
      const realized = Math.sqrt(variance * 252) * 100; // Annualized
      
      // Simulate implied volatility
      const marketStress = metrics.anomalyScore / 100;
      const implied = realized * (1 + marketStress * 0.5) + Math.random() * 5;
      const spread = implied - realized;
      
      // Determine volatility regime
      let regime: VolatilityData['regime'] = 'normal';
      if (realized > 50) regime = 'extreme';
      else if (realized > 30) regime = 'elevated';
      else if (realized < 10) regime = 'calm';
      
      return {
        timestamp: point.timestamp,
        realized: Math.max(5, Math.min(100, realized)),
        implied: Math.max(5, Math.min(100, implied)),
        spread,
        regime
      };
    });
  }, [timeSeriesData, metrics.anomalyScore]);

  const currentVol = volatilityData[volatilityData.length - 1] || { realized: 0, implied: 0, spread: 0, regime: 'normal' };
  
  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'calm': return theme.colors.success;
      case 'normal': return theme.colors.info;
      case 'elevated': return theme.colors.warning;
      case 'extreme': return theme.colors.danger;
      default: return theme.colors.primary;
    }
  };

  const getRegimeDescription = (regime: string) => {
    switch (regime) {
      case 'calm': return 'Market sleeping - perfect for accumulation';
      case 'normal': return 'Standard chop - business as usual';
      case 'elevated': return 'Storm brewing - tighten stops';
      case 'extreme': return 'CHAOS MODE - reduce exposure NOW';
      default: return 'Unknown state';
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.warning}66` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.warning }}>
          <GlitchText theme={theme.colors}>Volatility Neural Scanner</GlitchText>
        </h3>
        <Activity className="w-5 h-5" style={{ color: theme.colors.warning }} />
      </div>

      {/* Current Status */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Realized Vol</span>
            <Zap className="w-4 h-4" style={{ color: theme.colors.info }} />
          </div>
          <div className="text-xl font-bold" style={{ color: theme.colors.info }}>
            {currentVol.realized.toFixed(1)}%
          </div>
        </div>
        
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Implied Vol</span>
            <Activity className="w-4 h-4" style={{ color: theme.colors.warning }} />
          </div>
          <div className="text-xl font-bold" style={{ color: theme.colors.warning }}>
            {currentVol.implied.toFixed(1)}%
          </div>
        </div>
        
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ 
          borderColor: `${getRegimeColor(currentVol.regime)}44`,
          backgroundColor: `${getRegimeColor(currentVol.regime)}11`
        }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Regime</span>
            <AlertCircle className="w-4 h-4" style={{ color: getRegimeColor(currentVol.regime) }} />
          </div>
          <div className="text-sm font-bold uppercase" style={{ color: getRegimeColor(currentVol.regime) }}>
            {currentVol.regime}
          </div>
        </div>
      </div>

      {/* Volatility Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={volatilityData}>
            <defs>
              <linearGradient id="volSpreadGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={theme.colors.neural} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={theme.colors.neural} stopOpacity={0.05}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.grid} />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={(time) => new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: theme.colors.grid }}
            />
            <YAxis 
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: theme.colors.grid }}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as VolatilityData;
                  return (
                    <div className="bg-gray-900/95 border-2 rounded-lg p-3 font-mono text-sm backdrop-blur-md" 
                         style={{ borderColor: `${theme.colors.primary}88` }}>
                      <div className="border-b pb-2 mb-2" style={{ borderColor: `${theme.colors.primary}44` }}>
                        <p className="text-xs" style={{ color: theme.colors.primary }}>
                          {new Date(label).toLocaleString()}
                        </p>
                      </div>
                      <p style={{ color: theme.colors.info }}>
                        Realized: {data.realized.toFixed(1)}%
                      </p>
                      <p style={{ color: theme.colors.warning }}>
                        Implied: {data.implied.toFixed(1)}%
                      </p>
                      <p style={{ color: theme.colors.neural }}>
                        Spread: {data.spread.toFixed(1)}%
                      </p>
                      <p className="mt-2" style={{ color: getRegimeColor(data.regime) }}>
                        Regime: {data.regime.toUpperCase()}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Line 
              type="monotone" 
              dataKey="realized" 
              stroke={theme.colors.info} 
              strokeWidth={2}
              dot={false}
              name="Realized"
            />
            <Line 
              type="monotone" 
              dataKey="implied" 
              stroke={theme.colors.warning} 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Implied"
            />
            <Bar 
              dataKey="spread" 
              fill="url(#volSpreadGradient)"
              name="IV-RV Spread"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volatility Regime Analysis */}
      <div className="mt-4 p-4 bg-gray-800/50 border rounded-lg" style={{ 
        borderColor: `${getRegimeColor(currentVol.regime)}44` 
      }}>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-bold flex items-center space-x-2" style={{ color: getRegimeColor(currentVol.regime) }}>
            <Zap className="w-4 h-4" />
            <span>VOLATILITY REGIME: {currentVol.regime.toUpperCase()}</span>
          </h4>
          <span className="text-xs text-gray-400">
            Vol Spread: {currentVol.spread > 0 ? '+' : ''}{currentVol.spread.toFixed(1)}%
          </span>
        </div>
        <p className="text-xs text-gray-400 mb-3">
          {getRegimeDescription(currentVol.regime)}
        </p>
        
        {/* Trading Recommendations */}
        <div className="space-y-2">
          <div className="text-xs font-bold text-gray-400 mb-1">TACTICAL ADJUSTMENTS:</div>
          {currentVol.regime === 'extreme' && (
            <>
              <div className="flex items-center space-x-2 text-xs text-red-400">
                <AlertCircle className="w-3 h-3" />
                <span>• Reduce leverage to {Math.max(1, metrics.leverage / 2)}x immediately</span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-red-400">
                <AlertCircle className="w-3 h-3" />
                <span>• Tighten stops to 1% max loss per position</span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-red-400">
                <AlertCircle className="w-3 h-3" />
                <span>• Consider hedging with options or inverse positions</span>
              </div>
            </>
          )}
          {currentVol.regime === 'elevated' && (
            <>
              <div className="flex items-center space-x-2 text-xs text-yellow-400">
                <AlertCircle className="w-3 h-3" />
                <span>• Monitor positions closely - volatility expanding</span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-yellow-400">
                <AlertCircle className="w-3 h-3" />
                <span>• Reduce position sizes by 25-50%</span>
              </div>
            </>
          )}
          {currentVol.regime === 'calm' && (
            <>
              <div className="flex items-center space-x-2 text-xs text-green-400">
                <Zap className="w-3 h-3" />
                <span>• Optimal conditions for accumulation</span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-green-400">
                <Zap className="w-3 h-3" />
                <span>• Consider selling volatility for income</span>
              </div>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
};
