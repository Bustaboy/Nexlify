// Location: /src/pages/Dashboard/index.tsx
// Main Dashboard Assembly - Neural Command Center

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../../stores/dashboardStore';
import { DashboardHeader } from '../../components/layout/DashboardHeader';
import { TabNavigation } from '../../components/layout/TabNavigation';
import { OverviewTab } from '../../components/tabs/OverviewTab';
import { TradingTab } from '../../components/tabs/TradingTab';
import { RiskTab } from '../../components/tabs/RiskTab';
import { SettingsTab } from '../../components/tabs/SettingsTab';
import { DefiTab } from '../../components/tabs/DefiTab';
import { NeuralLock } from '../../components/security/NeuralLock';
import { CyberwareProtocol } from '../../systems/onboarding/CyberwareProtocol';
import { useWebSocketService } from '../../services/websocket.service';
import { useAuthService } from '../../services/auth.service';
import { TabType } from '../../types/dashboard.types';

export const Dashboard: React.FC = () => {
  const {
    activeTab,
    metrics,
    emergencyProtocol,
    isFullscreen,
    autoRefresh,
    isLocked,
    hasCompletedOnboarding,
    setActiveTab,
    toggleFullscreen,
    setIsLocked
  } = useDashboardStore();

  const { isAuthenticated, checkSession } = useAuthService();
  const { connect, disconnect } = useWebSocketService();
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Initialize systems
  useEffect(() => {
    const initializeSystems = async () => {
      const sessionValid = await checkSession();
      if (!sessionValid) {
        setIsLocked(true);
        return;
      }

      if (!hasCompletedOnboarding) {
        setShowOnboarding(true);
        return;
      }

      // Connect to live data feeds
      if (autoRefresh && !emergencyProtocol.isActive) {
        connect();
      }
    };

    initializeSystems();

    return () => {
      disconnect();
    };
  }, [autoRefresh, emergencyProtocol.isActive, hasCompletedOnboarding]);

  // Handle lock screen
  if (isLocked || !isAuthenticated) {
    return (
      <NeuralLock 
        onUnlock={() => {
          setIsLocked(false);
          if (!hasCompletedOnboarding) {
            setShowOnboarding(true);
          }
        }}
      />
    );
  }

  // Handle onboarding
  if (showOnboarding || !hasCompletedOnboarding) {
    return (
      <CyberwareProtocol 
        onComplete={() => {
          setShowOnboarding(false);
          useDashboardStore.setState({ hasCompletedOnboarding: true });
        }}
      />
    );
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'trading':
        return <TradingTab />;
      case 'risk':
        return <RiskTab />;
      case 'defi':
        return <DefiTab />;
      case 'settings':
        return <SettingsTab />;
      default:
        return <OverviewTab />;
    }
  };

  return (
    <div className={`min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white font-mono overflow-hidden ${
      isFullscreen ? 'fixed inset-0 z-50' : ''
    }`}>
      {/* Background Neural Grid */}
      <div className="fixed inset-0 opacity-20 pointer-events-none">
        <div className="absolute inset-0 bg-cyber-grid" />
      </div>

      {/* Main Interface */}
      <div className="relative z-10 flex flex-col h-screen">
        <DashboardHeader />
        <TabNavigation />
        
        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {renderTabContent()}
          </AnimatePresence>
        </div>

        {/* Footer Status Bar */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="border-t-2 border-cyber-primary/30 px-6 py-3 bg-gray-950/90 backdrop-blur-xl"
        >
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-4">
              <span className="text-gray-500">
                Last sync: <span className="text-cyber-primary font-bold">
                  {new Date(metrics.lastSync).toLocaleTimeString()}
                </span>
              </span>
              {emergencyProtocol.isActive && (
                <span className="text-red-400 font-bold animate-pulse">
                  • EMERGENCY PROTOCOL ACTIVE
                </span>
              )}
            </div>
            <div className="font-mono text-gray-600">
              NEXLIFY NEURAL CHROME v7.1.0 • "Welcome to the future of trading, choom"
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
