// Location: /src/types/dashboard.types.ts
// Type definitions for the Nexlify Neural Chrome Dashboard

export interface NeuralMetrics {
  // Financial Core
  totalEquity: number;
  totalPnL: number;
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  unrealizedPnL: number;
  realizedPnL: number;
  
  // Position Management
  marginUsed: number;
  marginAvailable: number;
  marginLevel: number;
  openOrdersCount: number;
  openOrdersValue: number;
  positionsPnL: Record<string, number>;
  fundingRate: number;
  nextFundingTime: number;
  
  // Combat Stats
  totalOps: number;
  successfulOps: number;
  failedOps: number;
  hitRate: number;
  avgProfit: number;
  avgLoss: number;
  maxProfit: number;
  maxLoss: number;
  profitRatio: number;
  
  // Risk Matrix
  maxDrawdown: number;
  maxDrawdownPercent: number;
  currentDrawdown: number;
  sharpeIndex: number;
  sortinoIndex: number;
  calmarIndex: number;
  riskExposure: number;
  
  // Neural Performance
  avgLatency: number;
  slippage: number;
  totalFees: number;
  feeRatio: number;
  
  // AI Performance
  signalPrecision: number;
  arbitrageHits: number;
  arbitrageSuccess: number;
  anomalyScore: number;
  
  // System Status
  systemUptime: number;
  connectionStatus: 'online' | 'degraded' | 'offline';
  latency: number;
  dataSync: number;
  
  // Trading Controls
  leverage: number;
  maxPositionSize: number;
  openPositions: number;
  
  lastSync: number;
}

export interface APIConfig {
  exchange: string;
  endpoint: string;
  apiKey: string;
  apiSecret: string;
  testnet: boolean;
  rateLimit: number;
  isActive: boolean;
  lastUpdate?: number;
}

export interface EmergencyProtocol {
  isActive: boolean;
  triggeredAt?: number;
  reason?: string;
  passwordHash?: string;
  closedPositions: Array<{
    symbol: string;
    loss: number;
    closedAt: number;
  }>;
}

export interface ThemeSettings {
  currentTheme: 'nexlify' | 'arasaka' | 'militech';
  customColors?: Partial<ThemeColors>;
  animations: boolean;
  glowEffects: boolean;
  soundEnabled: boolean;
}

export interface ThemeColors {
  primary: string;
  success: string;
  danger: string;
  warning: string;
  info: string;
  neural: string;
  dark: string;
  grid: string;
  accent: string;
}

export interface Theme {
  name: string;
  colors: ThemeColors;
}

export interface AIStrategy {
  id: string;
  codename: string;
  description: string;
  pnl: number;
  operations: number;
  hitRate: number;
  sharpe: number;
  maxDrawdown: number;
  isActive: boolean;
  confidence: number;
  neuralLoad: number;
  parameters: {
    learningRate: number;
    epochs: number;
    batchSize: number;
    riskLimit: number;
  };
  mlModel: 'lstm' | 'transformer' | 'ensemble';
}

export interface TimeSeriesDataPoint {
  timestamp: number;
  equity: number;
  pnl: number;
  drawdown: number;
  operations: number;
  signals: number;
  arbitrageOps: number;
  anomaly: number;
}

export interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  metric: string;
  value: number;
  threshold: number;
  timestamp: number;
}

export interface DashboardSettings {
  maxDrawdownLimit: number;
  maxPositionSize: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxLeverage: number;
  drawdownAlert: number;
  latencyAlert: number;
  winRateAlert: number;
  refreshRate: number;
  soundAlerts: boolean;
  enableML: boolean;
  mlUpdateFrequency: number;
  reinforcementLearning: boolean;
  explorationRate: number;
  rewardFunction: 'sharpe' | 'sortino' | 'calmar';
}

export type DashboardTab = 'overview' | 'trading' | 'risk' | 'settings';
