// Location: /src/components/risk/RiskMetricsPanel/index.tsx
// Risk Metrics Panel - Neural Defense Grid

import React from 'react';
import { motion } from 'framer-motion';
import { Shield, HelpCircle } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { METRIC_DESCRIPTIONS } from '../../../constants/dashboard.constants';
import { formatPercent } from '../../../utils/dashboard.utils';

interface RiskMetric {
  label: string;
  value: number;
  threshold?: { warning: number; critical: number };
  isRatio?: boolean;
  description?: string;
}

export const RiskMetricsPanel: React.FC = () => {
  const { metrics } = useDashboardStore();
  const theme = useThemeService();

  const riskMetrics: RiskMetric[] = [
    { 
      label: 'Current Drawdown', 
      value: (metrics.currentDrawdown / metrics.totalEquity) * 100, 
      threshold: { warning: 5, critical: 10 } 
    },
    { 
      label: 'Sharpe Ratio', 
      value: metrics.sharpeIndex, 
      isRatio: true, 
      description: METRIC_DESCRIPTIONS.sharpeIndex 
    },
    { 
      label: 'Sortino Ratio', 
      value: metrics.sortinoIndex, 
      isRatio: true, 
      description: METRIC_DESCRIPTIONS.sortinoIndex 
    },
    { 
      label: 'Calmar Ratio', 
      value: metrics.calmarIndex, 
      isRatio: true, 
      description: METRIC_DESCRIPTIONS.calmarIndex 
    },
    { 
      label: 'Avg Slippage', 
      value: metrics.slippage * 100, 
      threshold: { warning: 0.1, critical: 0.5 }, 
      description: METRIC_DESCRIPTIONS.slippage 
    },
    {
      label: 'Margin Level',
      value: metrics.marginLevel,
      threshold: { warning: 150, critical: 100 },
      description: METRIC_DESCRIPTIONS.marginLevel
    }
  ];

  const getMetricColor = (metric: RiskMetric): string => {
    if (metric.isRatio) {
      return metric.value >= 1.5 ? theme.colors.success : 
             metric.value >= 1.0 ? theme.colors.warning : 
             theme.colors.danger;
    }
    
    if (metric.threshold) {
      if (metric.label === 'Margin Level') {
        // Inverted logic for margin level
        return metric.value < metric.threshold.critical ? theme.colors.danger :
               metric.value < metric.threshold.warning ? theme.colors.warning :
               theme.colors.success;
      }
      return metric.value >= metric.threshold.critical ? theme.colors.danger : 
             metric.value >= metric.threshold.warning ? theme.colors.warning : 
             theme.colors.success;
    }
    
    return theme.colors.primary;
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.danger}66` }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
          <GlitchText theme={theme.colors}>Risk Matrix</GlitchText>
        </h3>
        <Shield className="w-5 h-5" style={{ color: theme.colors.danger }} />
      </div>

      <div className="space-y-4">
        {riskMetrics.map((metric, idx) => {
          const color = getMetricColor(metric);
          
          return (
            <motion.div 
              key={metric.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="flex justify-between items-center p-3 rounded-lg bg-gray-800/50 border hover:bg-gray-800/70 transition-all group" 
              style={{ borderColor: `${theme.colors.primary}22` }}
            >
              <div className="flex items-center space-x-2">
                <span className="text-gray-400 text-sm font-semibold">{metric.label}</span>
                {metric.description && (
                  <div className="relative">
                    <HelpCircle className="w-3 h-3 text-gray-500 cursor-help opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-gray-900/95 border-2 rounded-lg text-xs text-gray-300 opacity-0 hover:opacity-100 transition-opacity z-50 pointer-events-none shadow-2xl" 
                         style={{ borderColor: `${theme.colors.primary}88` }}>
                      {metric.description}
                    </div>
                  </div>
                )}
              </div>
              
              <div className="flex items-center space-x-3">
                {!metric.isRatio && metric.threshold && (
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <span>⚠ {metric.threshold.warning}</span>
                    <span>⛔ {metric.threshold.critical}</span>
                  </div>
                )}
                <span className="font-bold min-w-[60px] text-right" style={{ color }}>
                  {metric.isRatio ? metric.value.toFixed(2) : formatPercent(metric.value)}
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Risk Summary */}
      <div className="mt-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.warning}33` }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold" style={{ color: theme.colors.warning }}>Overall Risk Assessment</span>
          <div className={`px-3 py-1 rounded text-xs font-mono font-bold ${
            metrics.riskExposure / metrics.totalEquity > 0.05 ? 'bg-red-500/20 text-red-400' :
            metrics.riskExposure / metrics.totalEquity > 0.03 ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-green-500/20 text-green-400'
          }`}>
            {metrics.riskExposure / metrics.totalEquity > 0.05 ? 'HIGH RISK' :
             metrics.riskExposure / metrics.totalEquity > 0.03 ? 'MODERATE' :
             'LOW RISK'}
          </div>
        </div>
        <p className="text-xs text-gray-400">
          VaR at 95% confidence: {formatPercent((metrics.riskExposure / metrics.totalEquity) * 100)} of portfolio. 
          Current leverage multiplies both profits and losses by {metrics.leverage}x.
        </p>
      </div>
    </motion.div>
  );
};
