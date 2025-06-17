// Location: /src/components/charts/StrategyPerformanceChart/index.tsx
// Strategy Performance Chart - Neural Combat Analytics

import React from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits } from '../../../utils/dashboard.utils';

export const StrategyPerformanceChart: React.FC = () => {
  const { strategies } = useDashboardStore();
  const theme = useThemeService();

  const activeStrategies = strategies.filter(s => s.isActive);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.neural}66` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
          <GlitchText theme={theme.colors}>AI Protocol Performance</GlitchText>
        </h3>
      </div>
      
      <div className="h-64 flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={activeStrategies.map((s, idx) => ({
                name: s.codename,
                value: Math.abs(s.pnl),
                pnl: s.pnl,
                fill: [theme.colors.success, theme.colors.info, theme.colors.warning][idx] || theme.colors.neural
              }))}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={90}
              paddingAngle={3}
              dataKey="value"
            >
              {activeStrategies.map((_, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={[theme.colors.success, theme.colors.info, theme.colors.warning][index] || theme.colors.neural} 
                  style={{ filter: `drop-shadow(0 0 20px ${[theme.colors.success, theme.colors.info, theme.colors.warning][index] || theme.colors.neural}88)` }}
                />
              ))}
            </Pie>
            <Tooltip 
              content={({ active, payload }: any) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-gray-900/95 border-2 rounded-lg p-3 font-mono text-sm backdrop-blur-md" 
                         style={{ borderColor: `${theme.colors.neural}88` }}>
                      <p className="font-bold" style={{ color: theme.colors.neural }}>{data.name}</p>
                      <p style={{ color: data.pnl >= 0 ? theme.colors.success : theme.colors.danger }}>
                        {formatCredits(data.pnl)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      
      {activeStrategies.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-gray-500 text-sm font-mono">No active AI protocols</p>
        </div>
      )}
    </motion.div>
  );
};
