// Location: /src/components/dashboard/MetricsGrid/index.tsx
// Grid layout for displaying key trading metrics

import React from 'react';
import styled from 'styled-components';
import { 
  Database, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  BarChart3,
  CreditCard,
  Percent,
  BookOpen,
  Clock,
  Flame,
  Crosshair
} from 'lucide-react';
import { NeuralCard } from '../../common/NeuralCard';
import { Grid } from '../../../styles/styled';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { formatPercent } from '../../../utils/dashboard.utils';
import { METRIC_DESCRIPTIONS } from '../../../constants/dashboard.constants';

const Section = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #9CA3AF;
  margin-bottom: 1rem;
  padding-left: 0.5rem;
  border-left: 3px solid var(--color-primary-60);
`;

export const MetricsGrid: React.FC = () => {
  const { metrics, themeSettings } = useDashboardStore();
  const theme = themeSettings.currentTheme;
  
  // Get theme colors for dynamic styling
  const getMetricColor = (metric: string, value: number) => {
    switch (metric) {
      case 'hitRate':
        return value >= 60 ? 'var(--color-success)' : 
               value >= 50 ? 'var(--color-warning)' : 
               'var(--color-danger)';
      case 'sharpeIndex':
        return value >= 1.5 ? 'var(--color-success)' : 
               value >= 1.0 ? 'var(--color-warning)' : 
               'var(--color-danger)';
      case 'anomalyScore':
        return value > 80 ? 'var(--color-danger)' : 
               value > 50 ? 'var(--color-warning)' : 
               'var(--color-success)';
      case 'fundingRate':
        return value > 0 ? 'var(--color-danger)' : 'var(--color-success)';
      default:
        return 'var(--color-primary)';
    }
  };
  
  return (
    <>
      {/* Primary Metrics */}
      <Section>
        <SectionTitle>Core Metrics</SectionTitle>
        <Grid $cols={4} $gap="1.5rem">
          <NeuralCard
            title="Total Equity"
            value={metrics.totalEquity}
            icon={Database}
            color="var(--color-primary)"
            subtitle={`${formatPercent((metrics.totalEquity - 50000) / 50000 * 100)} from genesis`}
          />
          
          <NeuralCard
            title="Daily P&L"
            value={metrics.dailyPnL}
            change={(metrics.dailyPnL / metrics.totalEquity) * 100}
            icon={metrics.dailyPnL >= 0 ? TrendingUp : TrendingDown}
            color={metrics.dailyPnL >= 0 ? 'var(--color-success)' : 'var(--color-danger)'}
          />
          
          <NeuralCard
            title="Hit Rate"
            value={`${metrics.hitRate.toFixed(1)}%`}
            icon={Target}
            color={getMetricColor('hitRate', metrics.hitRate)}
            subtitle={`${metrics.successfulOps}/${metrics.totalOps} ops`}
            description={METRIC_DESCRIPTIONS.hitRate}
          />
          
          <NeuralCard
            title="Sharpe Index"
            value={metrics.sharpeIndex.toFixed(2)}
            icon={BarChart3}
            color={getMetricColor('sharpeIndex', metrics.sharpeIndex)}
            description={METRIC_DESCRIPTIONS.sharpeIndex}
          />
        </Grid>
      </Section>
      
      {/* Position Metrics */}
      <Section>
        <SectionTitle>Position Management</SectionTitle>
        <Grid $cols={6} $gap="1rem">
          <NeuralCard
            title="Margin Used"
            value={metrics.marginUsed}
            icon={CreditCard}
            color="var(--color-info)"
            subtitle={`${((metrics.marginUsed / metrics.totalEquity) * 100).toFixed(1)}% of equity`}
          />
          
          <NeuralCard
            title="Available Margin"
            value={metrics.marginAvailable}
            icon={Percent}
            color="var(--color-success)"
          />
          
          <NeuralCard
            title="Open Orders"
            value={metrics.openOrdersCount.toString()}
            icon={BookOpen}
            color="var(--color-warning)"
            subtitle={`€${metrics.openOrdersValue.toLocaleString()}`}
          />
          
          <NeuralCard
            title="Funding Rate"
            value={`${(metrics.fundingRate * 100).toFixed(4)}%`}
            icon={Clock}
            color={getMetricColor('fundingRate', metrics.fundingRate)}
            subtitle={`Next: ${new Date(metrics.nextFundingTime).toLocaleTimeString()}`}
            description={METRIC_DESCRIPTIONS.fundingRate}
          />
          
          <NeuralCard
            title="Anomaly Score"
            value={metrics.anomalyScore.toFixed(0)}
            icon={Flame}
            color={getMetricColor('anomalyScore', metrics.anomalyScore)}
            description={METRIC_DESCRIPTIONS.anomalyScore}
          />
          
          <NeuralCard
            title="Unrealized P&L"
            value={metrics.unrealizedPnL}
            icon={Crosshair}
            color={metrics.unrealizedPnL >= 0 ? 'var(--color-success)' : 'var(--color-danger)'}
          />
        </Grid>
      </Section>
    </>
  );
};

export default MetricsGrid;
