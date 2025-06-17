// Location: /src/components/charts/DrawdownChart/index.tsx
// Drawdown Chart - The Abyss Stares Back

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { TrendingDown, AlertTriangle } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatPercent } from '../../../utils/dashboard.utils';

export const DrawdownChart: React.FC = () => {
  const { timeSeriesData, metrics } = useDashboardStore();
  const theme = useThemeService();

  const drawdownData = useMemo(() => {
    let peak = 0;
    return timeSeriesData.map(point => {
      if (point.equity > peak) peak = point.equity;
      const drawdown = peak > 0 ? ((peak - point.equity) / peak) * 100 : 0;
      return {
        ...point,
        drawdown: -drawdown, // Negative for visual impact
        drawdownPercent: drawdown
      };
    });
  }, [timeSeriesData]);

  const maxDrawdown = Math.max(...drawdownData.map(d => d.drawdownPercent));
  const currentDrawdown = drawdownData[drawdownData.length - 1]?.drawdownPercent || 0;

  const getDrawdownSeverity = (dd: number) => {
    if (dd > 20) return { color: theme.colors.danger, text: 'CRITICAL', description: 'System failure imminent' };
    if (dd > 15) return { color: theme.colors.danger, text: 'SEVERE', description: 'Emergency protocols advised' };
    if (dd > 10) return { color: theme.colors.warning, text: 'HIGH', description: 'Risk parameters exceeded' };
    if (dd > 5) return { color: theme.colors.info, text: 'MODERATE', description: 'Within acceptable range' };
    return { color: theme.colors.success, text: 'LOW', description: 'Optimal performance' };
  };

  const severity = getDrawdownSeverity(currentDrawdown);

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.danger}66` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
          <GlitchText theme={theme.colors}>Drawdown Analysis</GlitchText>
        </h3>
        <TrendingDown className="w-5 h-5" style={{ color: theme.colors.danger }} />
      </div>

      {/* Current Status */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${severity.color}44` }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">Current Drawdown</span>
            <AlertTriangle className="w-4 h-4" style={{ color: severity.color }} />
          </div>
          <div className="text-2xl font-bold" style={{ color: severity.color }}>
            {formatPercent(currentDrawdown)}
          </div>
          <div className="text-xs mt-1" style={{ color: severity.color }}>
            {severity.text} - {severity.description}
          </div>
        </div>
        
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.danger}33` }}>
          <div className="text-xs text-gray-400 mb-1">Max Drawdown</div>
          <div className="text-2xl font-bold" style={{ color: theme.colors.danger }}>
            {formatPercent(maxDrawdown)}
          </div>
          <div className="text-xs text-gray-500 mt-1">Historical worst</div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={drawdownData.slice(-100)}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={theme.colors.danger} stopOpacity={0.1}/>
                <stop offset="95%" stopColor={theme.colors.danger} stopOpacity={0.8}/>
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
              tickFormatter={(value) => `${value}%`}
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: theme.colors.grid }}
              domain={[Math.min(-maxDrawdown * 1.1, -25), 0]}
            />
            <Tooltip 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const dd = Math.abs(payload[0].value as number);
                  const sev = getDrawdownSeverity(dd);
                  return (
                    <div className="bg-gray-900/95 border-2 rounded-lg p-3 font-mono text-sm backdrop-blur-md" 
                         style={{ borderColor: `${sev.color}88` }}>
                      <div className="border-b pb-2 mb-2" style={{ borderColor: `${theme.colors.primary}44` }}>
                        <p className="text-xs" style={{ color: theme.colors.primary }}>
                          {new Date(label).toLocaleString()}
                        </p>
                      </div>
                      <p style={{ color: sev.color }}>
                        Drawdown: {formatPercent(dd)}
                      </p>
                      <p className="text-xs mt-1" style={{ color: sev.color }}>
                        {sev.text}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke={theme.colors.danger}
              fill="url(#drawdownGradient)"
              strokeWidth={2}
              name="Drawdown"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Zones */}
      <div className="mt-4 space-y-2">
        <div className="text-xs font-bold text-gray-400 mb-2">RISK ZONES</div>
        <div className="grid grid-cols-4 gap-2">
          <div className="p-2 bg-green-500/10 border border-green-500/30 rounded text-center">
            <div className="text-xs text-green-400">0-5%</div>
            <div className="text-xs text-gray-500">Safe</div>
          </div>
          <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded text-center">
            <div className="text-xs text-blue-400">5-10%</div>
            <div className="text-xs text-gray-500">Caution</div>
          </div>
          <div className="p-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-center">
            <div className="text-xs text-yellow-400">10-20%</div>
            <div className="text-xs text-gray-500">Danger</div>
          </div>
          <div className="p-2 bg-red-500/10 border border-red-500/30 rounded text-center">
            <div className="text-xs text-red-400">&gt;20%</div>
            <div className="text-xs text-gray-500">Critical</div>
          </div>
        </div>
      </div>

      {/* The Abyss Speaks */}
      {currentDrawdown > 15 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg"
        >
          <p className="text-xs text-red-400 font-mono">
            "When you stare into the drawdown, the drawdown stares back. 
            Every percentage point is a lesson written in lost chrome. 
            Cut your losses before they cut you."
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};
