// Location: /src/components/charts/PnLDistribution/index.tsx
// P&L Distribution Chart - Neural Profit Matrix Visualization

import React from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { TimeRangeSelector } from '../../common/TimeRangeSelector';
import { formatCredits } from '../../../utils/dashboard.utils';

export const PnLDistribution: React.FC = () => {
  const { timeSeriesData } = useDashboardStore();
  const theme = useThemeService();

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.1 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.primary}44` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.primary }}>
          <GlitchText theme={theme.colors}>P&L Matrix</GlitchText>
        </h3>
        <TimeRangeSelector theme={theme.colors} />
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={timeSeriesData.slice(-50)}>
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
            />
            <Tooltip content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-gray-900/95 border-2 rounded-lg p-3 font-mono text-sm backdrop-blur-md" style={{ borderColor: `${theme.colors.primary}88` }}>
                    <div className="border-b pb-2 mb-2" style={{ borderColor: `${theme.colors.primary}44`, color: theme.colors.primary }}>
                      <p>{new Date(label).toLocaleString()}</p>
                    </div>
                    <p style={{ color: payload[0].value >= 0 ? theme.colors.success : theme.colors.danger }}>
                      P&L: {formatCredits(payload[0].value)}
                    </p>
                  </div>
                );
              }
              return null;
            }} />
            <Bar 
              dataKey="pnl"
              shape={(props: any) => {
                const fill = props.payload.pnl >= 0 ? theme.colors.success : theme.colors.danger;
                return <rect {...props} fill={fill} />;
              }}
              name="P&L"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};
