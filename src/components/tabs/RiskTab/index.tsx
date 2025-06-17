// Location: /src/components/tabs/RiskTab/index.tsx
// Risk Tab - Neural Risk Analysis Center

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, AlertTriangle, BarChart3, ShieldAlert, HelpCircle, CheckCircle, ChevronRight, ChevronLeft } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { NeuralCard } from '../../common/NeuralCard';
import { GlitchText } from '../../common/GlitchText';
import { RiskMetricsPanel } from '../../risk/RiskMetricsPanel';
import { SystemAlerts } from '../../risk/SystemAlerts';
import { DrawdownChart } from '../../charts/DrawdownChart';
import { RiskHeatmap } from '../../charts/RiskHeatmap';
import { VolatilityAnalysis } from '../../charts/VolatilityAnalysis';
import { METRIC_DESCRIPTIONS } from '../../../constants/dashboard.constants';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

export const RiskTab: React.FC = () => {
  const { metrics, alerts, defiPositions } = useDashboardStore();
  const theme = useThemeService();
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null);

  // Calculate DeFi specific risks
  const defiRiskMetrics = {
    impermanentLoss: defiPositions.liquidityPools.reduce((acc, pool) => 
      acc + (pool.impermanentLoss || 0), 0
    ),
    protocolRisk: defiPositions.stakingPositions.length * 0.05, // 5% per protocol
    smartContractRisk: (defiPositions.yieldFarms.length + defiPositions.liquidityPools.length) * 0.03
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-6 space-y-6"
    >
      {/* Risk Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <NeuralCard
          title="Max Drawdown"
          value={`${metrics.maxDrawdownPercent.toFixed(2)}%`}
          icon={Shield}
          color={
            metrics.maxDrawdownPercent >= 20 ? theme.colors.danger :
            metrics.maxDrawdownPercent >= 10 ? theme.colors.warning :
            theme.colors.success
          }
          subtitle={formatCredits(metrics.maxDrawdown)}
          description={METRIC_DESCRIPTIONS.maxDrawdown}
          theme={theme.colors}
        />
        <NeuralCard
          title="Risk Exposure"
          value={metrics.riskExposure}
          icon={AlertTriangle}
          color={theme.colors.danger}
          subtitle={`${((metrics.riskExposure / metrics.totalEquity) * 100).toFixed(2)}% of equity`}
          description={METRIC_DESCRIPTIONS.riskExposure}
          theme={theme.colors}
        />
        <NeuralCard
          title="Profit Factor"
          value={metrics.profitRatio.toFixed(2)}
          icon={BarChart3}
          color={
            metrics.profitRatio >= 1.5 ? theme.colors.success :
            metrics.profitRatio >= 1.0 ? theme.colors.warning :
            theme.colors.danger
          }
          description={METRIC_DESCRIPTIONS.profitRatio}
          theme={theme.colors}
        />
      </div>

      {/* DeFi Risk Metrics */}
      {(defiPositions.liquidityPools.length > 0 || defiPositions.stakingPositions.length > 0) && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.neural}66` }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
              <GlitchText theme={theme.colors}>DeFi Risk Analysis</GlitchText>
            </h3>
            <ShieldAlert className="w-5 h-5" style={{ color: theme.colors.neural }} />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
              <div className="text-sm text-gray-400 mb-1">Impermanent Loss</div>
              <div className="text-2xl font-bold" style={{ 
                color: defiRiskMetrics.impermanentLoss > 1000 ? theme.colors.danger : theme.colors.warning 
              }}>
                {formatCredits(defiRiskMetrics.impermanentLoss)}
              </div>
            </div>
            <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
              <div className="text-sm text-gray-400 mb-1">Protocol Risk</div>
              <div className="text-2xl font-bold" style={{ color: theme.colors.warning }}>
                {formatPercent(defiRiskMetrics.protocolRisk * 100)}
              </div>
            </div>
            <div className="bg-gray-800/50 border rounded-lg p-4" style={{ borderColor: `${theme.colors.primary}33` }}>
              <div className="text-sm text-gray-400 mb-1">Smart Contract Risk</div>
              <div className="text-2xl font-bold" style={{ color: theme.colors.info }}>
                {formatPercent(defiRiskMetrics.smartContractRisk * 100)}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Risk Metrics Panel */}
        <RiskMetricsPanel />
        
        {/* System Alerts */}
        <SystemAlerts 
          alerts={alerts}
          expandedAlert={expandedAlert}
          setExpandedAlert={setExpandedAlert}
        />
      </div>

      {/* Risk Analysis Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <DrawdownChart />
        <RiskHeatmap />
        <VolatilityAnalysis />
      </div>

      {/* Position Risk Matrix */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
        style={{ borderColor: `${theme.colors.danger}66` }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
            <GlitchText theme={theme.colors}>Position Risk Matrix</GlitchText>
          </h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b" style={{ borderColor: `${theme.colors.primary}33` }}>
                <th className="text-left py-2 text-sm text-gray-400">Position</th>
                <th className="text-right py-2 text-sm text-gray-400">Size</th>
                <th className="text-right py-2 text-sm text-gray-400">P&L</th>
                <th className="text-right py-2 text-sm text-gray-400">Risk Score</th>
                <th className="text-right py-2 text-sm text-gray-400">VaR</th>
                <th className="text-right py-2 text-sm text-gray-400">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.positionsPnL).map(([symbol, pnl]) => {
                const riskScore = Math.random() * 100;
                const var95 = Math.abs(pnl) * 0.05;
                return (
                  <tr key={symbol} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                    <td className="py-3 font-mono">{symbol}</td>
                    <td className="text-right py-3">{formatCredits(Math.random() * 10000)}</td>
                    <td className="text-right py-3" style={{ color: pnl >= 0 ? theme.colors.success : theme.colors.danger }}>
                      {formatCredits(pnl)}
                    </td>
                    <td className="text-right py-3">
                      <span className="px-2 py-1 rounded text-xs font-bold" style={{
                        backgroundColor: riskScore > 70 ? `${theme.colors.danger}33` : 
                                       riskScore > 40 ? `${theme.colors.warning}33` : 
                                       `${theme.colors.success}33`,
                        color: riskScore > 70 ? theme.colors.danger : 
                               riskScore > 40 ? theme.colors.warning : 
                               theme.colors.success
                      }}>
                        {riskScore.toFixed(0)}
                      </span>
                    </td>
                    <td className="text-right py-3 text-red-400">{formatCredits(-var95)}</td>
                    <td className="text-right py-3">
                      {riskScore > 70 ? (
                        <span className="text-xs text-red-400 font-bold">HIGH RISK</span>
                      ) : riskScore > 40 ? (
                        <span className="text-xs text-yellow-400 font-bold">MONITOR</span>
                      ) : (
                        <span className="text-xs text-green-400 font-bold">STABLE</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
};
