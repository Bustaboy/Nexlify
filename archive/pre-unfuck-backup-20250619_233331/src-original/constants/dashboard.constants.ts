// Location: /src/constants/dashboard.constants.ts
// Constants for the Nexlify Neural Chrome Dashboard

import { Theme } from '../types/dashboard.types';

export const THEMES: Record<string, Theme> = {
  nexlify: {
    name: 'Nexlify Classic',
    colors: {
      primary: '#00FFFF',    // Cyan
      success: '#00FF41',    // Green
      danger: '#FF1744',     // Red
      warning: '#FFAA00',    // Amber
      info: '#0088FF',       // Blue
      neural: '#FF00FF',     // Magenta
      dark: '#0A0F1B',       // Deep dark blue
      grid: 'rgba(0, 255, 255, 0.1)',
      accent: '#00BFFF'      // Deep sky blue
    }
  },
  arasaka: {
    name: 'Arasaka Corp',
    colors: {
      primary: '#FF0040',    // Arasaka red
      success: '#00FF00',    // Matrix green
      danger: '#CC0000',     // Dark red
      warning: '#FF6600',    // Orange
      info: '#0080FF',       // Corporate blue
      neural: '#FF0080',     // Pink
      dark: '#0A0A0A',       // Black
      grid: 'rgba(255, 0, 64, 0.1)',
      accent: '#FF3366'      // Light red
    }
  },
  militech: {
    name: 'Militech Industries',
    colors: {
      primary: '#FFD700',    // Gold
      success: '#32CD32',    // Lime green
      danger: '#DC143C',     // Crimson
      warning: '#FFA500',    // Orange
      info: '#4169E1',       // Royal blue
      neural: '#9370DB',     // Medium purple
      dark: '#1C1C1C',       // Charcoal
      grid: 'rgba(255, 215, 0, 0.1)',
      accent: '#FFEB3B'      // Yellow
    }
  }
};

export const DEFAULT_ENDPOINTS = {
  binance: {
    mainnet: 'https://api.binance.com',
    testnet: 'https://testnet.binance.vision',
    ws: 'wss://stream.binance.com:9443/ws'
  },
  kraken: {
    mainnet: 'https://api.kraken.com',
    testnet: 'https://api.kraken.com', // No testnet
    ws: 'wss://ws.kraken.com'
  },
  coinbase: {
    mainnet: 'https://api.pro.coinbase.com',
    testnet: 'https://api-public.sandbox.pro.coinbase.com',
    ws: 'wss://ws-feed.pro.coinbase.com'
  }
};

export const METRIC_DESCRIPTIONS = {
  sharpeIndex: "Risk-adjusted returns. Higher is better. >1.0 good, >1.5 excellent, >2.0 exceptional.",
  sortinoIndex: "Downside risk-adjusted returns. Focuses only on negative volatility.",
  calmarIndex: "Return over maximum drawdown. Measures return per unit of downside risk.",
  hitRate: "Percentage of profitable trades. Above 50% with good risk/reward is profitable.",
  profitRatio: "Average win divided by average loss. >1.5 is good, >2.0 is excellent.",
  maxDrawdown: "Largest peak-to-trough decline. Lower is better. <10% excellent, <20% acceptable.",
  riskExposure: "Value at Risk (VaR) - potential loss in worst 5% of scenarios.",
  leverage: "Capital multiplication factor. Higher leverage = higher risk and potential return.",
  slippage: "Difference between expected and actual execution price. <0.1% is excellent.",
  marginLevel: "Ratio of equity to used margin. >150% safe, <100% danger zone.",
  fundingRate: "Periodic payment between long and short positions. Positive = longs pay shorts.",
  anomalyScore: "ML-detected market anomaly score. Higher = more unusual market behavior."
};

export const DEFAULT_SETTINGS = {
  maxDrawdownLimit: 20,
  maxPositionSize: 10000,
  stopLossPercent: 2,
  takeProfitPercent: 5,
  maxLeverage: 10,
  drawdownAlert: 15,
  latencyAlert: 200,
  winRateAlert: 45,
  refreshRate: 2000,
  soundAlerts: true,
  enableML: true,
  mlUpdateFrequency: 3600,
  reinforcementLearning: true,
  explorationRate: 0.1,
  rewardFunction: 'sharpe' as const
};

export const ANIMATION_VARIANTS = {
  fadeIn: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  },
  slideIn: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 }
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 }
  }
};
