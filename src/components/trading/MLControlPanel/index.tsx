// Location: /src/components/trading/MLControlPanel/index.tsx
// ML Control Panel - Where Chrome Meets Consciousness

import React from 'react';
import { motion } from 'framer-motion';
import { Bot, Brain, Activity, Cpu } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';

export const MLControlPanel: React.FC = () => {
  const { settings, updateSettings, emergencyProtocol } = useDashboardStore();
  const theme = useThemeService();

  const handleSettingChange = (key: string, value: any) => {
    updateSettings({ ...settings, [key]: value });
  };

  return (
    <div className="bg-gray-800/50 border-2 rounded-xl p-6" 
         style={{ borderColor: `${theme.colors.neural}44` }}>
      <h4 className="text-lg font-bold mb-4 uppercase tracking-wider flex items-center space-x-2" 
          style={{ color: theme.colors.neural }}>
        <Bot className="w-5 h-5" />
        <span>Machine Learning Controls</span>
      </h4>

      {/* Neural Status Display */}
      <div className="mb-6 p-4 bg-gray-900/50 border rounded-lg" 
           style={{ borderColor: `${theme.colors.primary}33` }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Neural Network Status</span>
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 animate-pulse" style={{ color: settings.enableML ? theme.colors.success : theme.colors.danger }} />
            <span className="text-xs font-mono uppercase" style={{ 
              color: settings.enableML ? theme.colors.success : theme.colors.danger 
            }}>
              {settings.enableML ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="text-gray-500">Neurons</div>
            <div className="font-bold" style={{ color: theme.colors.info }}>1,337</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Layers</div>
            <div className="font-bold" style={{ color: theme.colors.warning }}>12</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Iterations</div>
            <div className="font-bold" style={{ color: theme.colors.success }}>∞</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div>
          <label className="text-sm text-gray-400 block mb-2 font-semibold">ML Status</label>
          <button
            onClick={() => handleSettingChange('enableML', !settings.enableML)}
            disabled={emergencyProtocol.isActive}
            className={`w-full px-4 py-2 rounded-lg font-mono font-bold transition-all ${
              emergencyProtocol.isActive ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            style={{
              backgroundColor: settings.enableML ? `${theme.colors.success}33` : 'rgba(75, 85, 99, 0.7)',
              borderWidth: '2px',
              borderColor: settings.enableML ? theme.colors.success : '#4B5563',
              color: settings.enableML ? theme.colors.success : '#9CA3AF',
              boxShadow: settings.enableML ? `0 0 20px ${theme.colors.success}44` : undefined
            }}
          >
            {settings.enableML ? 'ENABLED' : 'DISABLED'}
          </button>
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-2 font-semibold">Update Frequency</label>
          <select
            value={settings.mlUpdateFrequency}
            onChange={(e) => handleSettingChange('mlUpdateFrequency', Number(e.target.value))}
            disabled={emergencyProtocol.isActive || !settings.enableML}
            className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none disabled:opacity-50"
            style={{ 
              borderColor: `${theme.colors.primary}44`,
              color: theme.colors.primary
            }}
          >
            <option value="300">5 min (Aggressive)</option>
            <option value="900">15 min (Balanced)</option>
            <option value="3600">1 hour (Conservative)</option>
            <option value="86400">Daily (Safe)</option>
          </select>
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-2 font-semibold">Reward Function</label>
          <select
            value={settings.rewardFunction}
            onChange={(e) => handleSettingChange('rewardFunction', e.target.value)}
            disabled={emergencyProtocol.isActive || !settings.enableML}
            className="w-full px-4 py-2 bg-gray-800/80 border-2 rounded-lg font-mono focus:outline-none disabled:opacity-50"
            style={{ 
              borderColor: `${theme.colors.primary}44`,
              color: theme.colors.primary
            }}
          >
            <option value="sharpe">Sharpe Ratio</option>
            <option value="sortino">Sortino Ratio</option>
            <option value="calmar">Calmar Ratio</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
        <div className="flex items-center justify-between p-3 bg-gray-900/80 border-2 rounded-lg" 
             style={{ borderColor: `${theme.colors.primary}33` }}>
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4" style={{ color: theme.colors.neural }} />
            <span className="text-sm text-gray-400 font-semibold">Reinforcement Learning</span>
          </div>
          <button
            onClick={() => handleSettingChange('reinforcementLearning', !settings.reinforcementLearning)}
            disabled={emergencyProtocol.isActive || !settings.enableML}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              emergencyProtocol.isActive || !settings.enableML ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            style={{ backgroundColor: settings.reinforcementLearning ? theme.colors.primary : '#374151' }}
          >
            <motion.div 
              className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full"
              animate={{ x: settings.reinforcementLearning ? 24 : 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
            />
          </button>
        </div>

        <div>
          <label className="text-sm text-gray-400 block mb-2 font-semibold">Exploration Rate</label>
          <div className="flex items-center space-x-3">
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.01"
              value={settings.explorationRate}
              onChange={(e) => handleSettingChange('explorationRate', Number(e.target.value))}
              disabled={emergencyProtocol.isActive || !settings.reinforcementLearning || !settings.enableML}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              style={{
                background: `linear-gradient(to right, ${theme.colors.primary} 0%, ${theme.colors.primary} ${settings.explorationRate * 200}%, #374151 ${settings.explorationRate * 200}%, #374151 100%)`
              }}
            />
            <span className="font-mono w-16 text-right font-bold" style={{ color: theme.colors.primary }}>
              {(settings.explorationRate * 100).toFixed(0)}%
            </span>
          </div>
          <div className="mt-1 text-xs text-gray-500">
            Higher = more experimental trades, lower = safer patterns
          </div>
        </div>
      </div>

      {/* Neural Health Monitor */}
      <div className="mt-6 p-4 bg-gray-900/50 border rounded-lg" 
           style={{ borderColor: `${theme.colors.neural}33` }}>
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-bold flex items-center space-x-2" style={{ color: theme.colors.neural }}>
            <Cpu className="w-4 h-4" />
            <span>Neural Health Monitor</span>
          </span>
        </div>
        <div className="space-y-2">
          {['Pattern Recognition', 'Market Prediction', 'Risk Assessment', 'Anomaly Detection'].map((metric, idx) => {
            const health = 70 + Math.random() * 30;
            return (
              <div key={metric} className="flex items-center justify-between">
                <span className="text-xs text-gray-400">{metric}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${health}%` }}
                      transition={{ duration: 1, delay: idx * 0.1 }}
                      style={{
                        backgroundColor: health > 90 ? theme.colors.success : 
                                       health > 75 ? theme.colors.warning : 
                                       theme.colors.danger
                      }}
                    />
                  </div>
                  <span className="text-xs font-mono" style={{
                    color: health > 90 ? theme.colors.success : 
                          health > 75 ? theme.colors.warning : 
                          theme.colors.danger
                  }}>
                    {health.toFixed(0)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
