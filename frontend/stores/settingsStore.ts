/**
 * Nexlify Settings Store
 * Personal config, themes, and preferences - make the matrix yours
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import toast from 'react-hot-toast';

// Types
interface SettingsState {
  // Appearance
  theme: 'cyberpunk' | 'corpo' | 'street' | 'netrunner';
  accentColor: string;
  fontSize: 'small' | 'normal' | 'large';
  highContrast: boolean;
  reduceMotion: boolean;
  matrixRainEffect: boolean;
  
  // Sound & Notifications
  soundEnabled: boolean;
  soundVolume: number;
  notificationsEnabled: boolean;
  tradingAlerts: boolean;
  priceAlertSound: string;
  criticalAlertSound: string;
  
  // Trading Preferences
  defaultExchange: string;
  favoriteSymbols: string[];
  defaultOrderType: 'market' | 'limit';
  confirmOrders: boolean;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
  maxPositionSize: number;
  stopLossDefault: number;
  takeProfitDefault: number;
  
  // Display Settings
  chartType: 'candlestick' | 'line' | 'heikin-ashi';
  chartInterval: string;
  showOrderBook: boolean;
  showTradeHistory: boolean;
  compactMode: boolean;
  
  // System
  autoStart: boolean;
  minimizeToTray: boolean;
  checkUpdates: boolean;
  language: string;
  timezone: string;
  
  // Privacy & Security
  hideBalances: boolean;
  requirePinForTrades: boolean;
  sessionTimeout: number; // minutes
  clearDataOnExit: boolean;
  
  // API Settings
  apiEndpoint: string;
  websocketReconnect: boolean;
  requestTimeout: number;
  
  // Actions
  loadSettings: () => Promise<void>;
  updateSettings: (settings: Partial<SettingsState>) => Promise<void>;
  resetToDefaults: () => void;
  exportSettings: () => Promise<string>;
  importSettings: (data: string) => Promise<boolean>;
  toggleTheme: () => void;
  toggleSound: () => void;
  addFavoriteSymbol: (symbol: string) => void;
  removeFavoriteSymbol: (symbol: string) => void;
}

// Default settings - Night City standard issue
const DEFAULT_SETTINGS: Omit<SettingsState, 'loadSettings' | 'updateSettings' | 'resetToDefaults' | 'exportSettings' | 'importSettings' | 'toggleTheme' | 'toggleSound' | 'addFavoriteSymbol' | 'removeFavoriteSymbol'> = {
  // Appearance
  theme: 'cyberpunk',
  accentColor: '#00ffff',
  fontSize: 'normal',
  highContrast: false,
  reduceMotion: false,
  matrixRainEffect: true,
  
  // Sound & Notifications
  soundEnabled: true,
  soundVolume: 0.7,
  notificationsEnabled: true,
  tradingAlerts: true,
  priceAlertSound: 'notification',
  criticalAlertSound: 'alert_high',
  
  // Trading Preferences
  defaultExchange: 'binance',
  favoriteSymbols: ['BTC/USDT', 'ETH/USDT'],
  defaultOrderType: 'limit',
  confirmOrders: true,
  riskLevel: 'moderate',
  maxPositionSize: 0.1, // 10% of portfolio
  stopLossDefault: 2, // 2%
  takeProfitDefault: 5, // 5%
  
  // Display Settings
  chartType: 'candlestick',
  chartInterval: '15m',
  showOrderBook: true,
  showTradeHistory: true,
  compactMode: false,
  
  // System
  autoStart: false,
  minimizeToTray: true,
  checkUpdates: true,
  language: 'en',
  timezone: 'local',
  
  // Privacy & Security
  hideBalances: false,
  requirePinForTrades: false,
  sessionTimeout: 60, // 1 hour
  clearDataOnExit: false,
  
  // API Settings
  apiEndpoint: 'http://localhost:8000',
  websocketReconnect: true,
  requestTimeout: 30000 // 30 seconds
};

// Theme configurations - different chrome for different streets
const THEMES = {
  cyberpunk: {
    name: 'Cyberpunk',
    primary: '#0a0a0a',
    secondary: '#151515',
    accent: '#00ffff',
    success: '#00ff00',
    warning: '#ffff00',
    error: '#ff0000',
    text: '#ffffff'
  },
  corpo: {
    name: 'Corpo Elite',
    primary: '#0d1117',
    secondary: '#161b22',
    accent: '#58a6ff',
    success: '#3fb950',
    warning: '#d29922',
    error: '#f85149',
    text: '#c9d1d9'
  },
  street: {
    name: 'Street Kid',
    primary: '#1a1a1a',
    secondary: '#2d2d2d',
    accent: '#ff00ff',
    success: '#39ff14',
    warning: '#ff9f00',
    error: '#ff0040',
    text: '#e0e0e0'
  },
  netrunner: {
    name: 'Netrunner',
    primary: '#000000',
    secondary: '#0a0a0a',
    accent: '#00ff00',
    success: '#00ff00',
    warning: '#00ff00',
    error: '#00ff00',
    text: '#00ff00'
  }
};

export const useSettingsStore = create<SettingsState>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...DEFAULT_SETTINGS,

        // Load settings from Electron store
        loadSettings: async () => {
          try {
            const electronSettings = await window.nexlify.config.get();
            
            if (electronSettings) {
              set((draft) => {
                Object.assign(draft, electronSettings);
              });
            }
            
            // Apply theme
            const theme = get().theme;
            document.documentElement.setAttribute('data-theme', theme);
            
            // Apply font size
            const fontSize = get().fontSize;
            document.documentElement.style.fontSize = 
              fontSize === 'small' ? '14px' : 
              fontSize === 'large' ? '18px' : '16px';
              
            // Apply accent color as CSS variable
            document.documentElement.style.setProperty('--accent-color', get().accentColor);
            
          } catch (error) {
            console.error('Failed to load settings:', error);
            toast.error('Failed to load settings');
          }
        },

        // Update settings - both local and Electron store
        updateSettings: async (newSettings: Partial<SettingsState>) => {
          try {
            set((draft) => {
              Object.assign(draft, newSettings);
            });
            
            // Save to Electron store
            for (const [key, value] of Object.entries(newSettings)) {
              await window.nexlify.config.set(key, value);
            }
            
            // Apply theme changes immediately
            if (newSettings.theme) {
              document.documentElement.setAttribute('data-theme', newSettings.theme);
              
              // Update CSS variables
              const themeConfig = THEMES[newSettings.theme];
              Object.entries(themeConfig).forEach(([key, value]) => {
                if (key !== 'name') {
                  document.documentElement.style.setProperty(`--color-${key}`, value);
                }
              });
            }
            
            // Apply font size changes
            if (newSettings.fontSize) {
              document.documentElement.style.fontSize = 
                newSettings.fontSize === 'small' ? '14px' : 
                newSettings.fontSize === 'large' ? '18px' : '16px';
            }
            
            // Apply accent color
            if (newSettings.accentColor) {
              document.documentElement.style.setProperty('--accent-color', newSettings.accentColor);
            }
            
            toast.success('Settings updated', {
              icon: '⚙️',
              duration: 2000
            });
            
          } catch (error) {
            console.error('Failed to update settings:', error);
            toast.error('Failed to save settings');
          }
        },

        // Reset to factory defaults - clean slate
        resetToDefaults: () => {
          set((draft) => {
            Object.assign(draft, DEFAULT_SETTINGS);
          });
          
          // Apply default theme
          document.documentElement.setAttribute('data-theme', 'cyberpunk');
          document.documentElement.style.fontSize = '16px';
          
          toast.success('Settings reset to defaults', {
            icon: '🔄'
          });
        },

        // Export settings - backup your config
        exportSettings: async (): Promise<string> => {
          try {
            const state = get();
            const exportData = {
              version: '1.0',
              timestamp: new Date().toISOString(),
              settings: {
                theme: state.theme,
                accentColor: state.accentColor,
                fontSize: state.fontSize,
                favoriteSymbols: state.favoriteSymbols,
                defaultExchange: state.defaultExchange,
                riskLevel: state.riskLevel,
                chartType: state.chartType,
                chartInterval: state.chartInterval,
                soundEnabled: state.soundEnabled,
                notificationsEnabled: state.notificationsEnabled
                // Add other non-sensitive settings
              }
            };
            
            const json = JSON.stringify(exportData, null, 2);
            
            // Save to file
            const result = await window.nexlify.dialog.showSaveDialog({
              title: 'Export Settings',
              defaultPath: `nexlify-settings-${Date.now()}.json`,
              filters: [
                { name: 'JSON Files', extensions: ['json'] }
              ]
            });
            
            if (!result.canceled && result.filePath) {
              await window.nexlify.fs.writeFile(result.filePath, json);
              toast.success('Settings exported successfully');
            }
            
            return json;
            
          } catch (error) {
            toast.error('Failed to export settings');
            throw error;
          }
        },

        // Import settings - load someone else's chrome
        importSettings: async (data: string): Promise<boolean> => {
          try {
            const parsed = JSON.parse(data);
            
            if (!parsed.version || !parsed.settings) {
              throw new Error('Invalid settings file');
            }
            
            // Validate and apply settings
            const validSettings: Partial<SettingsState> = {};
            
            // Only import safe settings
            const safeKeys = [
              'theme', 'accentColor', 'fontSize', 'favoriteSymbols',
              'defaultExchange', 'riskLevel', 'chartType', 'chartInterval',
              'soundEnabled', 'notificationsEnabled'
            ];
            
            for (const key of safeKeys) {
              if (key in parsed.settings) {
                validSettings[key as keyof SettingsState] = parsed.settings[key];
              }
            }
            
            await get().updateSettings(validSettings);
            
            toast.success('Settings imported successfully', {
              icon: '📥'
            });
            
            return true;
            
          } catch (error) {
            toast.error('Failed to import settings');
            return false;
          }
        },

        // Quick theme toggle - cycle through the styles
        toggleTheme: () => {
          const themes = Object.keys(THEMES) as Array<keyof typeof THEMES>;
          const currentIndex = themes.indexOf(get().theme);
          const nextIndex = (currentIndex + 1) % themes.length;
          const nextTheme = themes[nextIndex];
          
          get().updateSettings({ theme: nextTheme });
          
          toast(`Theme: ${THEMES[nextTheme].name}`, {
            icon: '🎨',
            duration: 2000
          });
        },

        // Toggle sound - silence the chrome
        toggleSound: () => {
          const newState = !get().soundEnabled;
          get().updateSettings({ soundEnabled: newState });
          
          toast(newState ? 'Sound enabled' : 'Sound disabled', {
            icon: newState ? '🔊' : '🔇',
            duration: 2000
          });
        },

        // Add favorite symbol - keep your eyes on it
        addFavoriteSymbol: (symbol: string) => {
          const favorites = get().favoriteSymbols;
          
          if (!favorites.includes(symbol)) {
            set((draft) => {
              draft.favoriteSymbols.push(symbol);
            });
            
            toast(`${symbol} added to favorites`, {
              icon: '⭐',
              duration: 2000
            });
          }
        },

        // Remove favorite symbol
        removeFavoriteSymbol: (symbol: string) => {
          set((draft) => {
            draft.favoriteSymbols = draft.favoriteSymbols.filter(s => s !== symbol);
          });
          
          toast(`${symbol} removed from favorites`, {
            icon: '💔',
            duration: 2000
          });
        }
      })),
      {
        name: 'nexlify-settings-store',
        partialize: (state) => ({
          // Persist all settings except actions
          theme: state.theme,
          accentColor: state.accentColor,
          fontSize: state.fontSize,
          highContrast: state.highContrast,
          reduceMotion: state.reduceMotion,
          matrixRainEffect: state.matrixRainEffect,
          soundEnabled: state.soundEnabled,
          soundVolume: state.soundVolume,
          notificationsEnabled: state.notificationsEnabled,
          tradingAlerts: state.tradingAlerts,
          defaultExchange: state.defaultExchange,
          favoriteSymbols: state.favoriteSymbols,
          defaultOrderType: state.defaultOrderType,
          confirmOrders: state.confirmOrders,
          riskLevel: state.riskLevel,
          chartType: state.chartType,
          chartInterval: state.chartInterval,
          showOrderBook: state.showOrderBook,
          showTradeHistory: state.showTradeHistory,
          compactMode: state.compactMode,
          hideBalances: state.hideBalances,
          requirePinForTrades: state.requirePinForTrades,
          sessionTimeout: state.sessionTimeout
        })
      }
    ),
    {
      name: 'NexlifySettings'
    }
  )
);

// Auto-apply critical settings on store creation
const unsubscribe = useSettingsStore.subscribe(
  (state) => state.theme,
  (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
  }
);
