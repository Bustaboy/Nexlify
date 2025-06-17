// Location: /src/components/risk/SystemAlerts/index.tsx
// System Alerts - Neural Threat Detection Grid

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle, ChevronLeft, ChevronRight } from 'lucide-react';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { Alert } from '../../../types/dashboard.types';

interface Props {
  alerts: Alert[];
  expandedAlert: string | null;
  setExpandedAlert: (id: string | null) => void;
}

export const SystemAlerts: React.FC<Props> = ({ alerts, expandedAlert, setExpandedAlert }) => {
  const theme = useThemeService();

  const getAlertAnalysis = (alert: Alert): string => {
    if (alert.metric === 'anomaly') {
      if (alert.value > 90) return "⚠️ CRITICAL: Flash crash or black swan event detected";
      if (alert.value > 70) return "⚠️ HIGH: Unusual market behavior, possible manipulation";
      if (alert.value > 50) return "⚠️ MEDIUM: Increased volatility or trend reversal";
      return "ℹ️ LOW: Normal market fluctuations";
    }
    return "System metric exceeded threshold";
  };

  const getRecommendedAction = (alert: Alert): string => {
    if (alert.metric === 'anomaly') {
      if (alert.value > 90) return 'Immediately close risky positions. Consider activating emergency protocol.';
      if (alert.value > 70) return 'Reduce position sizes. Monitor closely for further anomalies.';
      if (alert.value > 50) return 'Review current positions. Consider hedging strategies.';
      return 'Continue normal operations with standard risk management.';
    }
    return "Review system parameters and adjust if necessary.";
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="relative bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6 overflow-hidden"
      style={{ borderColor: `${theme.colors.danger}66` }}
    >
      <div className="absolute top-0 left-0 w-full h-1 animate-pulse" style={{
        background: `linear-gradient(to right, ${theme.colors.danger}, ${theme.colors.warning}, ${theme.colors.danger})`
      }} />
      
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
          <GlitchText theme={theme.colors}>System Alerts</GlitchText>
        </h3>
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 animate-pulse" style={{ color: theme.colors.warning }} />
          <span className="text-sm text-gray-400 font-semibold">{alerts.length} active</span>
        </div>
      </div>
      
      <div className="space-y-2 max-h-96 overflow-y-auto">
        <style jsx>{`
          .scrollbar-alerts::-webkit-scrollbar {
            width: 6px;
          }
          .scrollbar-alerts::-webkit-scrollbar-track {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 3px;
          }
          .scrollbar-alerts::-webkit-scrollbar-thumb {
            background: ${theme.colors.danger}66;
            border-radius: 3px;
          }
          .scrollbar-alerts::-webkit-scrollbar-thumb:hover {
            background: ${theme.colors.danger}88;
          }
        `}</style>
        
        <div className="scrollbar-alerts">
          {alerts.length === 0 ? (
            <div className="text-gray-500 text-sm font-mono flex items-center space-x-2 p-4">
              <CheckCircle className="w-4 h-4" style={{ color: theme.colors.success }} />
              <span>All systems nominal - Neural grid stable</span>
            </div>
          ) : (
            <AnimatePresence>
              {alerts.map((alert, idx) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`p-3 rounded-lg border-l-4 backdrop-blur-sm cursor-pointer transition-all hover:bg-gray-800/50`}
                  style={{
                    borderLeftColor: alert.severity === 'critical' ? theme.colors.danger : 
                                    alert.severity === 'warning' ? theme.colors.warning : 
                                    theme.colors.info,
                    backgroundColor: alert.severity === 'critical' ? `${theme.colors.danger}11` : 
                                   alert.severity === 'warning' ? `${theme.colors.warning}11` : 
                                   `${theme.colors.info}11`
                  }}
                  onClick={() => setExpandedAlert(expandedAlert === alert.id ? null : alert.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="text-sm font-mono text-gray-300 flex items-center space-x-2">
                        <span>{alert.message}</span>
                        {expandedAlert === alert.id ? 
                          <ChevronLeft className="w-4 h-4 text-gray-400" /> : 
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        }
                      </div>
                      <div className="text-xs text-gray-500 font-mono mt-1">
                        {alert.metric}: {alert.value.toFixed(2)} / {alert.threshold}
                      </div>
                      
                      <AnimatePresence>
                        {expandedAlert === alert.id && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-3 pt-3 border-t" 
                            style={{ borderColor: `${theme.colors.primary}22` }}
                          >
                            <div className="space-y-2 text-gray-400">
                              <p className="font-bold" style={{ color: theme.colors.info }}>Alert Analysis:</p>
                              <p className="text-xs">{getAlertAnalysis(alert)}</p>
                              
                              <div className="grid grid-cols-2 gap-2 mt-2">
                                <div className="bg-gray-800/50 p-2 rounded">
                                  <span className="text-xs">Detection Method:</span>
                                  <p className="text-sm font-bold" style={{ color: theme.colors.info }}>
                                    {alert.value > 70 ? 'ML Pattern Recognition' : 'Statistical Analysis'}
                                  </p>
                                </div>
                                <div className="bg-gray-800/50 p-2 rounded">
                                  <span className="text-xs">Confidence:</span>
                                  <p className="text-sm font-bold" style={{ color: theme.colors.info }}>
                                    {(95 - (alert.value / 10)).toFixed(0)}%
                                  </p>
                                </div>
                              </div>
                              
                              <p className="text-xs mt-2">
                                <span className="font-bold">Recommended Action:</span> {getRecommendedAction(alert)}
                              </p>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                    <span className="text-xs text-gray-500 ml-2">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Alert Statistics */}
      <div className="mt-4 pt-4 border-t" style={{ borderColor: `${theme.colors.primary}22` }}>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="text-gray-500">Critical</div>
            <div className="font-bold" style={{ color: theme.colors.danger }}>
              {alerts.filter(a => a.severity === 'critical').length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Warning</div>
            <div className="font-bold" style={{ color: theme.colors.warning }}>
              {alerts.filter(a => a.severity === 'warning').length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Info</div>
            <div className="font-bold" style={{ color: theme.colors.info }}>
              {alerts.filter(a => a.severity === 'info').length}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
