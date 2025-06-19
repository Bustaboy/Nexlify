// Location: /src/services/mockData.service.ts
// Mock data generators for demo purposes
// MOCK_DATA: All functions here generate demo data
// TODO: Replace with real API calls when backend is connected

import { 
  NeuralMetrics, 
  AIStrategy, 
  TimeSeriesDataPoint 
} from '../types/dashboard.types';

/**
 * Generate mock neural metrics
 * MOCK_DATA: Simulates real-time trading metrics
 */
export const generateMockMetrics = (): NeuralMetrics => {
  const equity = 50000 + Math.random() * 25000;
  const dailyPnL = (Math.random() - 0.45) * 2000;
  const totalOps = Math.floor(Math.random() * 1000) + 500;
  const successfulOps = Math.floor(totalOps * (0.55 + Math.random() * 0.2));
  
  return {
    // Financial Core
    totalEquity: equity,
    totalPnL: dailyPnL * 30 + (Math.random() - 0.3) * 5000,
    dailyPnL,
    weeklyPnL: dailyPnL * 7 + (Math.random() - 0.5) * 3000,
    monthlyPnL: dailyPnL * 30 + (Math.random() - 0.5) * 8000,
    unrealizedPnL: (Math.random() - 0.5) * 1000,
    realizedPnL: dailyPnL - (Math.random() - 0.5) * 200,
    
    // Position Management
    marginUsed: equity * (0.2 + Math.random() * 0.3),
    marginAvailable: equity * (0.5 + Math.random() * 0.3),
    marginLevel: 100 + Math.random() * 200,
    openOrdersCount: Math.floor(Math.random() * 10),
    openOrdersValue: Math.random() * 10000,
    positionsPnL: {
      'BTC/USDT': (Math.random() - 0.5) * 1000,
      'ETH/USDT': (Math.random() - 0.5) * 500,
      'SOL/USDT': (Math.random() - 0.5) * 300
    },
    fundingRate: (Math.random() - 0.5) * 0.01,
    nextFundingTime: Date.now() + 3600000 * Math.floor(Math.random() * 8),
    
    // Combat Stats
    totalOps,
    successfulOps,
    failedOps: totalOps - successfulOps,
    hitRate: (successfulOps / totalOps) * 100,
    avgProfit: 50 + Math.random() * 100,
    avgLoss: -(30 + Math.random() * 60),
    maxProfit: 500 + Math.random() * 1000,
    maxLoss: -(200 + Math.random() * 800),
    profitRatio: 1.1 + Math.random() * 0.8,
    
    // Risk Matrix
    maxDrawdown: 2000 + Math.random() * 3000,
    maxDrawdownPercent: 5 + Math.random() * 10,
    currentDrawdown: Math.random() * 1000,
    sharpeIndex: 0.8 + Math.random() * 1.2,
    sortinoIndex: 1.0 + Math.random() * 1.5,
    calmarIndex: 0.5 + Math.random() * 1.0,
    riskExposure: equity * (0.02 + Math.random() * 0.03),
    
    // Neural Performance
    avgLatency: 50 + Math.random() * 200,
    slippage: Math.random() * 0.005,
    totalFees: Math.random() * 1000,
    feeRatio: Math.random() * 0.5,
    
    // AI Performance
    signalPrecision: 60 + Math.random() * 25,
    arbitrageHits: Math.floor(Math.random() * 50),
    arbitrageSuccess: 70 + Math.random() * 25,
    anomalyScore: Math.random() * 100,
    
    // System Status
    systemUptime: 95 + Math.random() * 5,
    connectionStatus: Math.random() > 0.9 ? 'degraded' : 'online',
    latency: 20 + Math.random() * 80,
    dataSync: Math.random() * 1000,
    
    // Trading Controls
    leverage: 1 + Math.floor(Math.random() * 10),
    maxPositionSize: 10000 + Math.random() * 40000,
    openPositions: Math.floor(Math.random() * 20),
    
    lastSync: Date.now()
  };
};

/**
 * Generate mock AI strategies
 * MOCK_DATA: Simulates different trading algorithms
 */
export const generateMockStrategies = (): AIStrategy[] => [
  {
    id: 'ghost_runner',
    codename: 'Ghost Runner MK-IV',
    description: 'High-frequency momentum strategy using LSTM neural networks',
    pnl: 2500 + Math.random() * 2000,
    operations: 150 + Math.floor(Math.random() * 100),
    hitRate: 65 + Math.random() * 15,
    sharpe: 1.2 + Math.random() * 0.8,
    maxDrawdown: 800 + Math.random() * 500,
    isActive: true,
    confidence: 0.75 + Math.random() * 0.2,
    neuralLoad: 45 + Math.random() * 30,
    parameters: {
      learningRate: 0.001,
      epochs: 100,
      batchSize: 32,
      riskLimit: 0.02
    },
    mlModel: 'lstm'
  },
  {
    id: 'netwatch_hunter',
    codename: 'Netwatch Hunter',
    description: 'Cross-exchange arbitrage detector with transformer architecture',
    pnl: 1800 + Math.random() * 1500,
    operations: 89 + Math.floor(Math.random() * 50),
    hitRate: 78 + Math.random() * 15,
    sharpe: 2.1 + Math.random() * 0.5,
    maxDrawdown: 300 + Math.random() * 200,
    isActive: true,
    confidence: 0.85 + Math.random() * 0.1,
    neuralLoad: 60 + Math.random() * 20,
    parameters: {
      learningRate: 0.0005,
      epochs: 200,
      batchSize: 64,
      riskLimit: 0.01
    },
    mlModel: 'transformer'
  },
  {
    id: 'daemon_reversal',
    codename: 'Daemon Reversal',
    description: 'Mean reversion strategy using ensemble learning',
    pnl: 1200 + Math.random() * 1000,
    operations: 67 + Math.floor(Math.random() * 40),
    hitRate: 58 + Math.random() * 20,
    sharpe: 0.9 + Math.random() * 0.6,
    maxDrawdown: 600 + Math.random() * 400,
    isActive: true,
    confidence: 0.68 + Math.random() * 0.25,
    neuralLoad: 35 + Math.random() * 25,
    parameters: {
      learningRate: 0.002,
      epochs: 150,
      batchSize: 16,
      riskLimit: 0.03
    },
    mlModel: 'ensemble'
  }
];

/**
 * Generate mock time series data
 * MOCK_DATA: Simulates historical price/equity data
 */
export const generateMockTimeSeriesData = (days: number = 30): TimeSeriesDataPoint[] => {
  const data: TimeSeriesDataPoint[] = [];
  let equity = 50000;
  const now = Date.now();
  
  for (let i = 0; i < days * 24; i++) {
    const change = (Math.random() - 0.45) * 200;
    equity += change;
    
    data.push({
      timestamp: now - (days * 24 - i) * 60 * 60 * 1000,
      equity,
      pnl: change,
      drawdown: Math.max(0, (Math.max(...data.map(d => d.equity || 0), equity) - equity)),
      operations: Math.floor(Math.random() * 5),
      signals: Math.floor(Math.random() * 10),
      arbitrageOps: Math.floor(Math.random() * 3),
      anomaly: Math.random() * 100
    });
  }
  
  return data;
};

/**
 * Generate mock alert
 * MOCK_DATA: Simulates system alerts
 */
export const generateMockAlert = () => ({
  id: `alert_${Date.now()}`,
  severity: Math.random() > 0.7 ? 'critical' as const : 
            Math.random() > 0.4 ? 'warning' as const : 'info' as const,
  message: 'Anomaly detected in market structure',
  metric: 'anomaly',
  value: Math.random() * 100,
  threshold: 80,
  timestamp: Date.now()
});
