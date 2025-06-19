// Location: /src/types/trading.types.ts
// Nexlify Trading System Type Definitions

export type PositionState = 
  | 'initializing'
  | 'scaling_in'
  | 'optimal'
  | 'scaling_out'
  | 'closing'
  | 'closed';

export interface Position {
  id: string;
  symbol: string;
  exchange: string;
  chain?: string;
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  pnl: number;
  pnlPercent: number;
  state: PositionState;
  timestamp: number;
  metadata?: {
    strategy?: string;
    riskScore?: number;
    correlationGroup?: string | null;
    leverage?: number;
  };
}

export interface QuantumScore {
  overall: number; // 0-100
  components: {
    neural: number;
    profitability: number;
    risk: number;
    momentum: number;
    correlation: number;
    timeDecay: number;
  };
  recommendation: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';
}

export interface ClusterAnalysis {
  id: string;
  positions: Position[];
  totalPnL: number;
  averageScore: number;
  riskConcentration: number; // 0-1
  recommendation: 'increase' | 'hold' | 'reduce' | 'exit';
}

export interface RebalanceStrategy {
  timestamp: number;
  actions: Array<{
    positionId: string;
    action: 'increase' | 'reduce';
    percentage: number;
    reason: 'risk_concentration' | 'high_performance' | 'portfolio_balance';
  }>;
  estimatedImpact: number;
}

export interface CrossChainPosition {
  protocol: string;
  chain: string;
  poolAddress: string;
  tokenA: string;
  tokenB: string;
  liquidity: number;
  fees24h: number;
  apr: number;
}

export interface RiskProfile {
  type: 'conservative' | 'balanced' | 'aggressive' | 'degen';
  maxDrawdown: number;
  maxLeverage: number;
  preferredTimeframe: 'scalping' | 'intraday' | 'swing' | 'position';
  riskPerTrade: number;
  correlationLimit: number;
}

export interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  mlModel: 'lstm' | 'transformer' | 'ensemble' | 'reinforcement';
  requiredCapital: number;
  expectedSharpe: number;
  maxDrawdown: number;
  parameters: Record<string, any>;
  backtestResults?: {
    period: string;
    trades: number;
    winRate: number;
    sharpe: number;
    sortino: number;
    maxDrawdown: number;
    totalReturn: number;
  };
}

export interface TradingSignal {
  id: string;
  timestamp: number;
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number; // 0-1
  source: 'neural' | 'technical' | 'sentiment' | 'arbitrage';
  metadata: {
    indicators?: Record<string, number>;
    mlPrediction?: number;
    targetPrice?: number;
    stopLoss?: number;
    timeframe?: string;
  };
}

export interface OrderExecution {
  id: string;
  positionId: string;
  exchange: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected';
  filledQuantity: number;
  averagePrice: number;
  fees: number;
  timestamp: number;
  metadata?: {
    slippage?: number;
    latency?: number;
    route?: string[];
  };
}
