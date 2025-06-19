// Location: /src/services/theme.service.ts
// Theme management service for CSS variables and styled-components

import { Theme, ThemeSettings } from '../types/dashboard.types';
import { THEMES } from '../constants/dashboard.constants';

export class ThemeService {
  private static instance: ThemeService;
  
  private constructor() {}
  
  static getInstance(): ThemeService {
    if (!ThemeService.instance) {
      ThemeService.instance = new ThemeService();
    }
    return ThemeService.instance;
  }
  
  /**
   * Initialize CSS variables from theme
   */
  initializeTheme(themeName: keyof typeof THEMES): void {
    const theme = THEMES[themeName];
    if (!theme) {
      console.error(`Theme ${themeName} not found`);
      return;
    }
    
    this.applyTheme(theme);
  }
  
  /**
   * Apply theme colors to CSS variables
   */
  applyTheme(theme: Theme): void {
    const root = document.documentElement;
    
    // Set CSS variables for each color
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
      
      // Create opacity variants
      root.style.setProperty(`--color-${key}-10`, `${value}1A`); // 10% opacity
      root.style.setProperty(`--color-${key}-20`, `${value}33`); // 20% opacity
      root.style.setProperty(`--color-${key}-30`, `${value}4D`); // 30% opacity
      root.style.setProperty(`--color-${key}-40`, `${value}66`); // 40% opacity
      root.style.setProperty(`--color-${key}-50`, `${value}80`); // 50% opacity
      root.style.setProperty(`--color-${key}-60`, `${value}99`); // 60% opacity
      root.style.setProperty(`--color-${key}-70`, `${value}B3`); // 70% opacity
      root.style.setProperty(`--color-${key}-80`, `${value}CC`); // 80% opacity
      root.style.setProperty(`--color-${key}-90`, `${value}E6`); // 90% opacity
    });
    
    // Set theme name
    root.setAttribute('data-theme', theme.name.toLowerCase().replace(/\s+/g, '-'));
  }
  
  /**
   * Get current theme from localStorage or default
   */
  getCurrentTheme(): Theme {
    try {
      const savedSettings = localStorage.getItem('nexlify_theme_settings');
      if (savedSettings) {
        const settings: ThemeSettings = JSON.parse(savedSettings);
        return THEMES[settings.currentTheme] || THEMES.nexlify;
      }
    } catch (error) {
      console.error('Failed to load theme settings:', error);
    }
    
    return THEMES.nexlify;
  }
  
  /**
   * Save theme settings
   */
  saveThemeSettings(settings: ThemeSettings): void {
    try {
      localStorage.setItem('nexlify_theme_settings', JSON.stringify(settings));
      this.initializeTheme(settings.currentTheme);
    } catch (error) {
      console.error('Failed to save theme settings:', error);
    }
  }
  
  /**
   * Create global styles for styled-components
   */
  getGlobalStyles(): string {
    return `
      :root {
        /* Base colors will be set dynamically */
        --transition-fast: 150ms ease-in-out;
        --transition-normal: 300ms ease-in-out;
        --transition-slow: 500ms ease-in-out;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --shadow-glow: 0 0 20px var(--color-primary-50);
        --shadow-glow-strong: 0 0 40px var(--color-primary-70);
        
        /* Border radius */
        --radius-sm: 0.25rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        --radius-full: 9999px;
      }
      
      * {
        box-sizing: border-box;
      }
      
      body {
        margin: 0;
        padding: 0;
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: var(--color-dark);
        color: white;
        overflow-x: hidden;
      }
      
      /* Scrollbar styling */
      ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      
      ::-webkit-scrollbar-track {
        background: rgba(31, 41, 55, 0.5);
        border-radius: 4px;
      }
      
      ::-webkit-scrollbar-thumb {
        background: var(--color-primary-60);
        border-radius: 4px;
        transition: background var(--transition-fast);
      }
      
      ::-webkit-scrollbar-thumb:hover {
        background: var(--color-primary-80);
      }
      
      /* Animations */
      @keyframes pulse {
        0%, 100% {
          opacity: 1;
        }
        50% {
          opacity: 0.5;
        }
      }
      
      @keyframes spin {
        from {
          transform: rotate(0deg);
        }
        to {
          transform: rotate(360deg);
        }
      }
      
      @keyframes glow {
        0%, 100% {
          box-shadow: 0 0 20px var(--color-primary-50);
        }
        50% {
          box-shadow: 0 0 40px var(--color-primary-70), 0 0 60px var(--color-primary-40);
        }
      }
      
      /* Utility classes */
      .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
      }
      
      .animate-spin {
        animation: spin 1s linear infinite;
      }
      
      .animate-glow {
        animation: glow 2s ease-in-out infinite;
      }
    `;
  }
}
