// Location: /src/components/charts/EquityChart/index.tsx
// Equity curve chart with anomaly detection overlay

import React, { useMemo } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { BaseCard } from '../../../styles/styled';
import { GlitchText } from '../../common/GlitchText';
import { TimeRangeSelector, TimeRange } from '../../common/TimeRangeSelector';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { formatCredits } from '../../../utils/dashboard.utils';

const ChartCard = styled(BaseCard)`
  padding: 1.5rem;
  height: 400px;
`;

const ChartHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const ChartTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--color-primary);
`;

const ChartContainer = styled.div`
  height: calc(100% - 4rem);
  width: 100%;
`;

const CustomTooltip = styled.div`
  background: rgba(17, 24, 39, 0.95);
  border: 2px solid var(--color-primary-80);
  border-radius: var(--radius-lg);
  padding: 0.75rem;
  font-family: monospace;
  font-size: 0.875rem;
  backdrop-filter: blur(12px);
`;

const TooltipHeader = styled.div`
  border-bottom: 1px solid var(--color-primary-40);
  padding-bottom: 0.5rem;
  margin-bottom: 0.5rem;
  color: var(--color-primary);
`;

const TooltipRow = styled.p<{ $color?: string }>`
  margin: 0.25rem 0;
  color: ${({ $color }) => $color || '#E5E7EB'};
`;

export const EquityChart: React.FC = () => {
  const { timeSeriesData } = useDashboardStore();
  const [timeRange, setTimeRange] = React.useState<TimeRange>('1d');
  
  // Filter data based on time range
  const filteredData = useMemo(() => {
    const now = Date.now();
    const ranges = {
      '1h': 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };
    
    const cutoff = now - ranges[timeRange];
    return timeSeriesData.filter(d => d.timestamp >= cutoff).slice(-100);
  }, [timeSeriesData, timeRange]);
  
  const renderTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    
    return (
      <CustomTooltip>
        <TooltipHeader>
          {new Date(label).toLocaleString()}
        </TooltipHeader>
        {payload.map((entry: any, index: number) => (
          <TooltipRow key={index} $color={entry.color}>
            {entry.name}: {
              entry.name === 'Anomaly' 
                ? entry.value.toFixed(0) 
                : formatCredits(entry.value)
            }
          </TooltipRow>
        ))}
      </CustomTooltip>
    );
  };
  
  return (
    <ChartCard
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.1 }}
    >
      <ChartHeader>
        <ChartTitle>
          <GlitchText>Neural Equity Stream</GlitchText>
        </ChartTitle>
        <TimeRangeSelector 
          defaultValue={timeRange}
          onChange={setTimeRange}
        />
      </ChartHeader>
      
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filteredData}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-success)" stopOpacity={0.4} />
                <stop offset="95%" stopColor="var(--color-success)" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="anomalyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-danger)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--color-danger)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="var(--color-grid)" 
              opacity={0.5}
            />
            
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={(time) => new Date(time).toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
              })}
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'var(--color-grid)' }}
            />
            
            <YAxis 
              yAxisId="left"
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'var(--color-grid)' }}
              tickFormatter={(value) => `€${(value / 1000).toFixed(0)}k`}
            />
            
            <YAxis 
              yAxisId="right"
              orientation="right"
              tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={{ stroke: 'var(--color-grid)' }}
              domain={[0, 100]}
            />
            
            <Tooltip content={renderTooltip} />
            
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="equity"
              stroke="var(--color-success)"
              fill="url(#equityGradient)"
              strokeWidth={2}
              name="Equity"
            />
            
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="anomaly"
              stroke="var(--color-danger)"
              fill="url(#anomalyGradient)"
              strokeWidth={1}
              strokeDasharray="5 5"
              name="Anomaly"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
};

export default EquityChart;
