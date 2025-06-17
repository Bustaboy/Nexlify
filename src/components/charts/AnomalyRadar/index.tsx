// Location: /src/components/charts/AnomalyRadar/index.tsx
// Anomaly Radar - Neural Pattern Detection System

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Sparkles } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';

export const AnomalyRadar: React.FC = () => {
  const { metrics } = useDashboardStore();
  const theme = useThemeService();
  const [radarData, setRadarData] = useState<any[]>([]);

  useEffect(() => {
    // Generate dynamic radar data based on market conditions
    const data = [
      { metric: 'Market Vol', value: Math.random() * 100, fullMark: 100 },
      { metric: 'Order Flow', value: Math.random() * 100, fullMark: 100 },
      { metric: 'Spread', value: Math.random() * 100, fullMark: 100 },
      { metric: 'Sentiment', value: Math.random() * 100, fullMark: 100 },
      { metric: 'Correlation', value: Math.random() * 100, fullMark: 100 },
      { metric: 'Liquidity', value: Math.random() * 100, fullMark: 100 }
    ];
    setRadarData(data);
  }, [metrics.lastSync]);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.neural}44` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
          <GlitchText theme={theme.colors}>Predictive Anomaly Detection</GlitchText>
        </h3>
        <Sparkles className="w-5 h-5" style={{ color: theme.colors.neural }} />
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke={theme.colors.grid} />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#6B7280', fontSize: 10 }} />
            <Radar 
              name="Current" 
              dataKey="value" 
              stroke={theme.colors.neural} 
              fill={theme.colors.neural} 
              fillOpacity={0.3} 
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="text-xs text-gray-400 mb-1">Anomaly Score</div>
          <div className="text-2xl font-bold" style={{ color: metrics.anomalyScore > 80 ? theme.colors.danger : theme.colors.success }}>
            {metrics.anomalyScore.toFixed(0)}
          </div>
        </div>
        <div className="p-3 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="text-xs text-gray-400 mb-1">Risk Level</div>
          <div className="text-2xl font-bold" style={{ 
            color: metrics.anomalyScore > 80 ? theme.colors.danger : 
                   metrics.anomalyScore > 50 ? theme.colors.warning : 
                   theme.colors.success 
          }}>
            {metrics.anomalyScore > 80 ? 'HIGH' : metrics.anomalyScore > 50 ? 'MEDIUM' : 'LOW'}
          </div>
        </div>
      </div>
    </motion.div>
  );
};
