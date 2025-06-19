// Location: /src/stores/dashboardStore.ts
// Zustand store for dashboard state management

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { 
  NeuralMetrics, 
  APIConfig, 
  EmergencyProtocol, 
  ThemeSettings, 
  AIStrategy, 
  TimeSeriesDataPoint, 
  Alert, 
  DashboardSettings,
  DashboardTab 
} from '../types/dashboard.types';
import { DEFAULT_SETTINGS } from '../constants/dashboard.constants';
import { generateMockMetrics, generateMockStrategies, generateMockTimeSeriesData } from '../services/mockData.service';

interface DashboardState {
  // Core Data
  metrics: NeuralMetrics;
  strategies: AIStrategy[];
  timeSeriesData: TimeSeriesDataPoint[];
  alerts: Alert[];
  
  // Configuration
  apiConfigs: APIConfig[];
  emergencyProtocol: EmergencyProtocol;
  themeSettings: ThemeSettings;
  settings: DashboardSettings;
  
  // UI State
  activeTab: DashboardTab;
  autoRefresh: boolean;
  tradingActive: boolean;
  isFullscreen: boolean;
  
  // Selected states
  selectedExchange: string;
  expandedAlert: string | null;
  
  // Actions
  updateMetrics: (metrics: Partial<NeuralMetrics>) => void;
  setMetrics: (metrics: NeuralMetrics) => void;
  addTimeSeriesPoint: (point: TimeSeriesDataPoint) => void;
  setStrategies: (strategies: AIStrategy[]) => void;
  toggleStrategyActive: (strategyId: string) => void;
  
  // API Config Actions
  setApiConfigs: (configs: APIConfig[]) => void;
  updateApiConfig: (exchange: string, config: Partial<APIConfig>) => void;
  
  // Emergency Protocol Actions
  activateEmergencyProtocol: (password: string, reason: string) => void;
  deactivateEmergencyProtocol: () => void;
  
  // Theme Actions
  setThemeSettings: (settings: ThemeSettings) => void;
  
  // Settings Actions
  updateSettings: (settings: Partial<DashboardSettings>) => void;
  
  // UI Actions
  setActiveTab: (tab: DashboardTab) => void;
  setAutoRefresh: (enabled: boolean) => void;
  setTradingActive: (active: boolean) => void;
  setFullscreen: (fullscreen: boolean) => void;
  setSelectedExchange: (exchange: string) => void;
  setExpandedAlert: (alertId: string | null) => void;
  
  // Alert Actions
  addAlert: (alert: Alert) => void;
  clearAlerts: () => void;
  
  // Initialize with mock data
  initializeMockData: () => void;
}

export const useDashboardStore = create<DashboardState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        metrics: generateMockMetrics(),
        strategies: generateMockStrategies(),
        timeSeriesData: generateMockTimeSeriesData(),
        alerts: [],
        
        apiConfigs: [],
        emergencyProtocol: {
          isActive: false,
          closedPositions: []
        },
        themeSettings: {
          currentTheme: 'nexlify',
          animations: true,
          glowEffects: true,
          soundEnabled: true
        },
        settings: DEFAULT_SETTINGS,
        
        activeTab: 'overview',
        autoRefresh: true,
        tradingActive: false,
        isFullscreen: false,
        
        selectedExchange: 'binance',
        expandedAlert: null,
        
        // Metrics Actions
        updateMetrics: (metrics) => set((state) => ({
          metrics: { ...state.metrics, ...metrics }
        })),
        
        setMetrics: (metrics) => set({ metrics }),
        
        addTimeSeriesPoint: (point) => set((state) => ({
          timeSeriesData: [...state.timeSeriesData.slice(-719), point]
        })),
        
        // Strategy Actions
        setStrategies: (strategies) => set({ strategies }),
        
        toggleStrategyActive: (strategyId) => set((state) => ({
          strategies: state.strategies.map(s => 
            s.id === strategyId ? { ...s, isActive: !s.isActive } : s
          )
        })),
        
        // API Config Actions
        setApiConfigs: (configs) => set({ apiConfigs: configs }),
        
        updateApiConfig: (exchange, config) => set((state) => {
          const existing = state.apiConfigs.find(c => c.exchange === exchange);
          if (existing) {
            return {
              apiConfigs: state.apiConfigs.map(c => 
                c.exchange === exchange ? { ...c, ...config } : c
              )
            };
          } else {
            return {
              apiConfigs: [...state.apiConfigs, {
                exchange,
                endpoint: '',
                apiKey: '',
                apiSecret: '',
                testnet: false,
                rateLimit: 1200,
                isActive: false,
                ...config
              }]
            };
          }
        }),
        
        // Emergency Protocol Actions
        activateEmergencyProtocol: (passwordHash, reason) => set((state) => ({
          emergencyProtocol: {
            isActive: true,
            triggeredAt: Date.now(),
            reason,
            passwordHash,
            closedPositions: Object.entries(state.metrics.positionsPnL)
              .filter(([_, pnl]) => pnl < 0)
              .sort((a, b) => a[1] - b[1])
              .map(([symbol, loss]) => ({
                symbol,
                loss,
                closedAt: Date.now()
              }))
          },
          tradingActive: false,
          autoRefresh: false
        })),
        
        deactivateEmergencyProtocol: () => set({
          emergencyProtocol: {
            isActive: false,
            closedPositions: []
          }
        }),
        
        // Theme Actions
        setThemeSettings: (settings) => set({ themeSettings: settings }),
        
        // Settings Actions
        updateSettings: (settings) => set((state) => ({
          settings: { ...state.settings, ...settings }
        })),
        
        // UI Actions
        setActiveTab: (tab) => set({ activeTab: tab }),
        setAutoRefresh: (enabled) => set({ autoRefresh: enabled }),
        setTradingActive: (active) => set({ tradingActive: active }),
        setFullscreen: (fullscreen) => set({ isFullscreen: fullscreen }),
        setSelectedExchange: (exchange) => set({ selectedExchange: exchange }),
        setExpandedAlert: (alertId) => set({ expandedAlert: alertId }),
        
        // Alert Actions
        addAlert: (alert) => set((state) => ({
          alerts: [alert, ...state.alerts].slice(0, 10)
        })),
        
        clearAlerts: () => set({ alerts: [] }),
        
        // Initialize Mock Data
        initializeMockData: () => {
          const { metrics, strategies, timeSeriesData } = get();
          // MOCK_DATA: This initializes with mock data for demo purposes
          // TODO: Replace with real data service layer
          if (timeSeriesData.length === 0) {
            set({
              metrics: generateMockMetrics(),
              strategies: generateMockStrategies(),
              timeSeriesData: generateMockTimeSeriesData()
            });
          }
        }
      }),
      {
        name: 'nexlify-dashboard-storage',
        partialize: (state) => ({
          apiConfigs: state.apiConfigs,
          themeSettings: state.themeSettings,
          settings: state.settings
        })
      }
    ),
    {
      name: 'NexlifyDashboard'
    }
  )
);
