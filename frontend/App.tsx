/**
 * Nexlify Trading Matrix - Main App
 * Cyberpunk-themed trading interface with real-time data
 */

import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';

// Store and hooks
import { useAuthStore } from './stores/authStore';
import { useTradingStore } from './stores/tradingStore';
import { useSettingsStore } from './stores/settingsStore';
import { useWebSocket } from './hooks/useWebSocket';
import { useSystemMonitor } from './hooks/useSystemMonitor';

// Components
import { TitleBar } from './components/layout/TitleBar';
import { Sidebar } from './components/layout/Sidebar';
import { StatusBar } from './components/layout/StatusBar';
import { PinAuth } from './components/auth/PinAuth';
import { TwoFactorSetup } from './components/auth/TwoFactorSetup';
import { CyberpunkBackground } from './components/effects/CyberpunkBackground';
import { LoadingScreen } from './components/common/LoadingScreen';
import { ErrorBoundary } from './components/common/ErrorBoundary';

// Pages
import { Dashboard } from './pages/Dashboard';
import { TradingMatrix } from './pages/TradingMatrix';
import { RiskMatrix } from './pages/RiskMatrix';
import { Analytics } from './pages/Analytics';
import { AICompanion } from './pages/AICompanion';
import { Achievements } from './pages/Achievements';
import { Security } from './pages/Security';
import { AuditTrail } from './pages/AuditTrail';
import { Settings } from './pages/Settings';

// Styles
import './styles/globals.css';
import './styles/cyberpunk.css';

// Create query client with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5 seconds
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 3,
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always'
    },
    mutations: {
      retry: 2,
      retryDelay: 1000
    }
  }
});

// Main App Component
export const App: React.FC = () => {
  const { isAuthenticated, requires2FA, checkSession } = useAuthStore();
  const { theme, soundEnabled, loadSettings } = useSettingsStore();
  const [isLoading, setIsLoading] = useState(true);
  const [isAPIReady, setIsAPIReady] = useState(false);

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Load saved settings
        await loadSettings();
        
        // Check if API is running
        const apiStatus = await checkAPIHealth();
        setIsAPIReady(apiStatus);
        
        // Restore session if exists
        await checkSession();
        
        // Apply theme
        document.documentElement.className = theme;
        
        // Enable sound if configured
        if (soundEnabled) {
          await initializeSounds();
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('App initialization failed:', error);
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  // System monitoring
  useSystemMonitor();

  // WebSocket connection (only when authenticated)
  useWebSocket(isAuthenticated);

  // Loading state
  if (isLoading) {
    return <LoadingScreen />;
  }

  // API not ready
  if (!isAPIReady) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-center">
          <div className="text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-red-500 mb-2">API Offline</h1>
          <p className="text-gray-400">Please ensure the backend is running</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-cyan-500 text-black rounded hover:bg-cyan-400"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="app-container h-screen flex flex-col overflow-hidden">
            {/* Cyberpunk background effects */}
            <CyberpunkBackground />
            
            {/* Custom title bar for frameless window */}
            <TitleBar />
            
            {/* Main content */}
            <div className="flex-1 flex overflow-hidden">
              <AnimatePresence mode="wait">
                {!isAuthenticated ? (
                  <motion.div
                    key="auth"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full"
                  >
                    {requires2FA ? <TwoFactorSetup /> : <PinAuth />}
                  </motion.div>
                ) : (
                  <motion.div
                    key="main"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex w-full"
                  >
                    {/* Sidebar navigation */}
                    <Sidebar />
                    
                    {/* Main content area */}
                    <main className="flex-1 overflow-hidden bg-gray-900/95">
                      <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/trading" element={<TradingMatrix />} />
                        <Route path="/risk" element={<RiskMatrix />} />
                        <Route path="/analytics" element={<Analytics />} />
                        <Route path="/ai" element={<AICompanion />} />
                        <Route path="/achievements" element={<Achievements />} />
                        <Route path="/security" element={<Security />} />
                        <Route path="/audit" element={<AuditTrail />} />
                        <Route path="/settings" element={<Settings />} />
                      </Routes>
                    </main>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            
            {/* Status bar */}
            <StatusBar />
            
            {/* Toast notifications */}
            <Toaster
              position="bottom-right"
              toastOptions={{
                className: 'cyberpunk-toast',
                duration: 4000,
                style: {
                  background: '#1a1a1a',
                  color: '#fff',
                  border: '1px solid #00ffff',
                  borderRadius: '4px',
                },
                success: {
                  iconTheme: {
                    primary: '#00ff00',
                    secondary: '#000',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ff0000',
                    secondary: '#000',
                  },
                },
              }}
            />
          </div>
        </Router>
        
        {/* React Query Devtools */}
        {process.env.NODE_ENV === 'development' && (
          <ReactQueryDevtools initialIsOpen={false} />
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

// Helper functions
async function checkAPIHealth(): Promise<boolean> {
  try {
    const apiEndpoint = await window.nexlify.config.get('apiEndpoint') || 'http://localhost:8000';
    const response = await fetch(`${apiEndpoint}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
}

async function initializeSounds(): Promise<void> {
  try {
    // Preload sound effects
    const sounds = [
      'startup.mp3',
      'click.mp3',
      'notification.mp3',
      'error.mp3',
      'success.mp3',
      'trade_execute.mp3'
    ];
    
    // This would load actual sound files
    // For now, just log
    console.log('Sound system initialized');
  } catch (error) {
    console.error('Failed to initialize sounds:', error);
  }
}

// Export for index.tsx
export default App;
