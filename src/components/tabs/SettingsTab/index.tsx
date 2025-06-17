// Location: /src/components/tabs/SettingsTab/index.tsx
// Settings Tab - Neural Configuration Matrix

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings, Save, RotateCcw, AlertTriangle, Sliders, Bot, Shield, Bell } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';

export const SettingsTab: React.FC = () => {
  const { settings, updateSettings } = useDashboardStore();
  const theme = useThemeService();
  const [hasChanges, setHasChanges] = useState(false);
  const [localSettings, setLocalSettings] = useState(settings);

  const handleSettingChange = (key: string, value: any) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    updateSettings(localSettings);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalSettings(settings);
    setHasChanges(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold uppercase tracking-wider" style={{ color: theme.colors.primary }}>
          <GlitchText theme={theme.colors}>Neural Configuration</GlitchText>
        </h2>
        <div className="flex items-center space-x-3">
          {hasChanges && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-sm font-mono"
              style={{ color: theme.colors.warning }}
            >
              • Unsaved changes
            </motion.div>
          )}
          <button
            onClick={handleReset}
            disabled={!hasChanges}
            className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg font-mono text-sm transition-colors hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RotateCcw className="w-4 h-4 inline mr-2" />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges}
            className="px-4 py-2 border-2 rounded-lg font-mono text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              backgroundColor: hasChanges ? `${theme.colors.success}33` : 'rgba(75, 85, 99, 0.5)',
              borderColor: hasChanges ? theme.colors.success : '#4B5563',
              color: hasChanges ? theme.colors.success : '#9CA3AF'
            }}
          >
            <Save className="w-4 h-4 inline mr-2" />
            Save Configuration
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Risk Management Settings */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.danger}44` }}
        >
          <div className="flex items-center space-x-2 mb-6">
            <Shield className="w-5 h-5" style={{ color: theme.colors.danger }} />
            <h3 className="text-lg font-bold uppercase tracking-wider" style={{ color: theme.colors.danger }}>
              Risk Management
            </h3>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">Max Drawdown Limit (%)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={localSettings.maxDrawdownLimit}
                  onChange={(e) => handleSettingChange('maxDrawdownLimit', Number(e.target.value))}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, ${theme.colors.danger} 0%, ${theme.colors.danger} ${(localSettings.maxDrawdownLimit - 5) / 45 * 100}%, #374151 ${(localSettings.maxDrawdownLimit - 5) / 45 * 100}%, #374151 100%)`
                  }}
                />
                <span className="font-mono w-12 text-right font-bold" style={{ color: theme.colors.danger }}>
                  {localSettings.maxDrawdownLimit}%
                </span>
              </div>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Max Position Size (€)</label>
              <input
                type="number"
                value={localSettings.maxPositionSize}
                onChange={(e) => handleSettingChange('maxPositionSize', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none transition-colors"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-gray-400 block mb-2">Stop Loss (%)</label>
                <input
                  type="number"
                  value={localSettings.stopLossPercent}
                  onChange={(e) => handleSettingChange('stopLossPercent', Number(e.target.value))}
                  className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none transition-colors"
                  style={{ 
                    borderColor: `${theme.colors.danger}44`,
                    color: theme.colors.danger
                  }}
                />
              </div>
              <div>
                <label className="text-sm text-gray-400 block mb-2">Take Profit (%)</label>
                <input
                  type="number"
                  value={localSettings.takeProfitPercent}
                  onChange={(e) => handleSettingChange('takeProfitPercent', Number(e.target.value))}
                  className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none transition-colors"
                  style={{ 
                    borderColor: `${theme.colors.success}44`,
                    color: theme.colors.success
                  }}
                />
              </div>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Max Leverage</label>
              <select
                value={localSettings.maxLeverage}
                onChange={(e) => handleSettingChange('maxLeverage', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              >
                <option value="1">1x</option>
                <option value="2">2x</option>
                <option value="5">5x</option>
                <option value="10">10x</option>
                <option value="20">20x</option>
                <option value="50">50x</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* ML/AI Settings */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.neural}44` }}
        >
          <div className="flex items-center space-x-2 mb-6">
            <Bot className="w-5 h-5" style={{ color: theme.colors.neural }} />
            <h3 className="text-lg font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
              Neural AI Configuration
            </h3>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-800/50 border-2 rounded-lg" 
                 style={{ borderColor: `${theme.colors.primary}33` }}>
              <span className="text-sm text-gray-400 font-semibold">Enable Machine Learning</span>
              <button
                onClick={() => handleSettingChange('enableML', !localSettings.enableML)}
                className={`relative w-12 h-6 rounded-full transition-colors`}
                style={{ backgroundColor: localSettings.enableML ? theme.colors.primary : '#374151' }}
              >
                <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                  localSettings.enableML ? 'translate-x-6' : ''
                }`} />
              </button>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">ML Update Frequency</label>
              <select
                value={localSettings.mlUpdateFrequency}
                onChange={(e) => handleSettingChange('mlUpdateFrequency', Number(e.target.value))}
                disabled={!localSettings.enableML}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none disabled:opacity-50"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              >
                <option value="300">5 minutes</option>
                <option value="900">15 minutes</option>
                <option value="3600">1 hour</option>
                <option value="86400">Daily</option>
              </select>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-800/50 border-2 rounded-lg" 
                 style={{ borderColor: `${theme.colors.primary}33` }}>
              <span className="text-sm text-gray-400 font-semibold">Reinforcement Learning</span>
              <button
                onClick={() => handleSettingChange('reinforcementLearning', !localSettings.reinforcementLearning)}
                disabled={!localSettings.enableML}
                className={`relative w-12 h-6 rounded-full transition-colors disabled:opacity-50`}
                style={{ backgroundColor: localSettings.reinforcementLearning ? theme.colors.primary : '#374151' }}
              >
                <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                  localSettings.reinforcementLearning ? 'translate-x-6' : ''
                }`} />
              </button>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Exploration Rate (%)</label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={localSettings.explorationRate * 100}
                  onChange={(e) => handleSettingChange('explorationRate', Number(e.target.value) / 100)}
                  disabled={!localSettings.reinforcementLearning || !localSettings.enableML}
                  className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                  style={{
                    background: `linear-gradient(to right, ${theme.colors.neural} 0%, ${theme.colors.neural} ${localSettings.explorationRate * 2 * 100}%, #374151 ${localSettings.explorationRate * 2 * 100}%, #374151 100%)`
                  }}
                />
                <span className="font-mono w-12 text-right font-bold" style={{ color: theme.colors.neural }}>
                  {(localSettings.explorationRate * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Reward Function</label>
              <select
                value={localSettings.rewardFunction}
                onChange={(e) => handleSettingChange('rewardFunction', e.target.value)}
                disabled={!localSettings.reinforcementLearning || !localSettings.enableML}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none disabled:opacity-50"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              >
                <option value="sharpe">Sharpe Ratio</option>
                <option value="sortino">Sortino Ratio</option>
                <option value="calmar">Calmar Ratio</option>
                <option value="profit">Pure Profit</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* Alert Settings */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.warning}44` }}
        >
          <div className="flex items-center space-x-2 mb-6">
            <Bell className="w-5 h-5" style={{ color: theme.colors.warning }} />
            <h3 className="text-lg font-bold uppercase tracking-wider" style={{ color: theme.colors.warning }}>
              Alert Configuration
            </h3>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-800/50 border-2 rounded-lg" 
                 style={{ borderColor: `${theme.colors.primary}33` }}>
              <span className="text-sm text-gray-400 font-semibold">Sound Alerts</span>
              <button
                onClick={() => handleSettingChange('soundAlerts', !localSettings.soundAlerts)}
                className={`relative w-12 h-6 rounded-full transition-colors`}
                style={{ backgroundColor: localSettings.soundAlerts ? theme.colors.primary : '#374151' }}
              >
                <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                  localSettings.soundAlerts ? 'translate-x-6' : ''
                }`} />
              </button>
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Drawdown Alert (%)</label>
              <input
                type="number"
                value={localSettings.drawdownAlert}
                onChange={(e) => handleSettingChange('drawdownAlert', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.warning}44`,
                  color: theme.colors.warning
                }}
              />
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Latency Alert (ms)</label>
              <input
                type="number"
                value={localSettings.latencyAlert}
                onChange={(e) => handleSettingChange('latencyAlert', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.warning}44`,
                  color: theme.colors.warning
                }}
              />
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">Win Rate Alert (%)</label>
              <input
                type="number"
                value={localSettings.winRateAlert}
                onChange={(e) => handleSettingChange('winRateAlert', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.warning}44`,
                  color: theme.colors.warning
                }}
              />
            </div>
          </div>
        </motion.div>

        {/* System Settings */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
          style={{ borderColor: `${theme.colors.info}44` }}
        >
          <div className="flex items-center space-x-2 mb-6">
            <Sliders className="w-5 h-5" style={{ color: theme.colors.info }} />
            <h3 className="text-lg font-bold uppercase tracking-wider" style={{ color: theme.colors.info }}>
              System Configuration
            </h3>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">Refresh Rate (ms)</label>
              <select
                value={localSettings.refreshRate}
                onChange={(e) => handleSettingChange('refreshRate', Number(e.target.value))}
                className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none"
                style={{ 
                  borderColor: `${theme.colors.primary}44`,
                  color: theme.colors.primary
                }}
              >
                <option value="1000">1 second</option>
                <option value="2000">2 seconds</option>
                <option value="5000">5 seconds</option>
                <option value="10000">10 seconds</option>
              </select>
            </div>

            <div className="p-4 bg-gray-800/50 border-2 rounded-lg" style={{ borderColor: `${theme.colors.warning}33` }}>
              <div className="flex items-center space-x-2 mb-2">
                <AlertTriangle className="w-4 h-4" style={{ color: theme.colors.warning }} />
                <span className="text-sm font-bold" style={{ color: theme.colors.warning }}>
                  Performance Notice
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Lower refresh rates increase system load. Recommended: 2s for optimal performance.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};
