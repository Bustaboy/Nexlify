// Location: /src/types/defi.types.ts
// Nexlify DeFi System Type Definitions

export interface DeFiProtocol {
  id: string;
  name: string;
  type: 'dex' | 'lending' | 'yield' | 'derivative' | 'bridge' | 'aggregator';
  chains: string[];
  tvl: number; // Total Value Locked in USD
  apy?: number;
  gasEfficiency: number; // 0-100 score
  riskScore: number; // 0-100 (lower is safer)
  features: {
    flashLoans?: boolean;
    yieldFarming?: boolean;
    leveragedPositions?: boolean;
    crossChain?: boolean;
  };
}

export interface YieldStrategy {
  protocol: string;
  type: 'single' | 'lp' | 'lending' | 'vault' | 'leveraged';
  estimatedAPY: number;
  risk: 'low' | 'medium' | 'high' | 'degen';
  gasEstimate: number; // in USD
  requirements: {
    minAmount: number;
    lockPeriod?: number; // in days
    tokens?: string[];
  };
  rewards?: Array<{
    token: string;
    apy: number;
    vestingPeriod?: number;
  }>;
}

export interface GasOptimizationStrategy {
  method: 'batching' | 'flashloan' | 'multicall' | 'gastoken';
  estimatedSavings: number; // percentage
  implementation: string;
  risks: string[];
}

export interface MEVProtection {
  type: 'sandwich' | 'frontrun' | 'backrun';
  severity: 'low' | 'medium' | 'high' | 'critical';
  transaction: string;
  mitigation: string;
  estimatedLoss: number;
}

export interface LiquidityPosition {
  id: string;
  protocol: string;
  chain: string;
  tokenA: string;
  tokenB: string;
  amountA: number;
  amountB: number;
  valueUSD: number;
  apy: number;
  rewards: Array<{
    token: string;
    amount: number;
    valueUSD: number;
    claimable: boolean;
  }>;
  impermanentLoss: number; // percentage
  entryTimestamp: number;
}

export interface VaultStrategy {
  id: string;
  name: string;
  protocol: string;
  description: string;
  tvl: number;
  apy: {
    current: number;
    average30d: number;
    max: number;
  };
  risk: {
    score: number; // 0-100
    factors: string[];
    audited: boolean;
    timelock: number; // hours
  };
  fees: {
    deposit: number;
    withdrawal: number;
    management: number;
    performance: number;
  };
  underlying: string[];
  strategies: string[];
}

export interface FlashLoanOpportunity {
  id: string;
  type: 'arbitrage' | 'liquidation' | 'collateral_swap';
  profit: number; // in USD
  requiredCapital: number;
  gasEstimate: number;
  confidence: number; // 0-1
  route: Array<{
    protocol: string;
    action: string;
    params: any;
  }>;
  expiresAt: number;
}

export interface CrossChainRoute {
  bridge: string;
  fromChain: string;
  toChain: string;
  estimatedTime: number; // seconds
  fee: number; // in USD
  slippage: number; // percentage
  gasEstimate: number; // in USD
  securityScore?: number; // 0-100
  steps?: Array<{
    protocol: string;
    chain: string;
    action: string;
  }>;
}

export interface DeFiPosition {
  id: string;
  type: 'lending' | 'borrowing' | 'liquidity' | 'staking' | 'farming';
  protocol: string;
  chain: string;
  assets: Array<{
    token: string;
    amount: number;
    valueUSD: number;
  }>;
  debt?: Array<{
    token: string;
    amount: number;
    valueUSD: number;
    apy: number;
  }>;
  collateralRatio?: number;
  liquidationPrice?: number;
  healthFactor?: number;
  earnings: {
    total: number;
    unclaimed: number;
    apr: number;
  };
  entryTimestamp: number;
  lastUpdate: number;
}

export interface GasEstimate {
  action: string;
  chain: string;
  standard: number; // in gwei
  fast: number;
  instant: number;
  baseFee: number;
  priorityFee: number;
  estimatedCost: {
    usd: number;
    native: number;
  };
  congestion: 'low' | 'medium' | 'high';
}

export interface ProtocolRisk {
  protocol: string;
  riskScore: number; // 0-100
  factors: {
    smart_contract: number;
    oracle: number;
    liquidity: number;
    centralization: number;
    regulatory: number;
  };
  audits: Array<{
    auditor: string;
    date: string;
    severity: 'clean' | 'low' | 'medium' | 'high';
    findings: number;
  }>;
  incidents: Array<{
    date: string;
    type: string;
    impact: number; // USD lost
    resolved: boolean;
  }>;
  insurance: {
    available: boolean;
    providers: string[];
    coverage: number;
    premium: number;
  };
}

export interface YieldAggregatorStrategy {
  id: string;
  name: string;
  description: string;
  protocols: string[]; // Protocols involved
  expectedAPY: number;
  riskLevel: 'conservative' | 'moderate' | 'aggressive' | 'degen';
  capitalRequired: number;
  steps: Array<{
    protocol: string;
    action: 'deposit' | 'borrow' | 'swap' | 'stake' | 'farm';
    asset: string;
    amount: string; // Can be percentage or fixed
    params?: any;
  }>;
  gasEstimate: number;
  breakEvenDays: number;
}
