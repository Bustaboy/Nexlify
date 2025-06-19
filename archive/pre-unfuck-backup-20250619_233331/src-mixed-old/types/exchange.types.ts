// Location: /src/types/exchange.types.ts
// Nexlify Exchange System Type Definitions

export interface Exchange {
  id: string;
  name: string;
  status: 'online' | 'degraded' | 'offline';
  markets: number;
  fees: {
    maker: number;
    taker: number;
  };
  features: {
    spot?: boolean;
    futures?: boolean;
    margin?: boolean;
    staking?: boolean;
    websocket?: boolean;
  };
  limits: any;
  tier: 'primary' | 'secondary' | 'experimental';
}

export interface OrderBook {
  exchange: string;
  symbol: string;
  bids: Array<[number, number]>; // [price, volume]
  asks: Array<[number, number]>;
  timestamp: number;
}

export interface Trade {
  id: string;
  exchange: string;
  symbol: string;
  price: number;
  amount: number;
  side: 'buy' | 'sell';
  timestamp: number;
  fee?: number;
  feeCurrency?: string;
}

export interface Balance {
  exchange: string;
  currency: string;
  free: number;
  used: number;
  total: number;
  usdValue?: number;
}

export interface Market {
  id: string;
  symbol: string;
  base: string;
  quote: string;
  active: boolean;
  type: 'spot' | 'future' | 'option' | 'perpetual';
  limits: {
    amount: { min: number; max: number };
    price: { min: number; max: number };
    cost: { min: number; max: number };
  };
  precision: {
    amount: number;
    price: number;
  };
  info: any;
}

export interface Ticker {
  symbol: string;
  timestamp: number;
  datetime: string;
  high: number;
  low: number;
  bid: number;
  bidVolume?: number;
  ask: number;
  askVolume?: number;
  vwap?: number;
  open: number;
  close: number;
  last: number;
  previousClose?: number;
  change: number;
  percentage: number;
  average?: number;
  baseVolume: number;
  quoteVolume: number;
}

export interface ExchangeStatus {
  exchange: string;
  status: 'operational' | 'degraded' | 'maintenance' | 'offline';
  latency: number;
  lastUpdate: number;
  message?: string;
}

export interface DarkPoolOrder {
  id: string;
  pool: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'iceberg' | 'hidden' | 'sweep';
  totalQuantity: number;
  displayQuantity?: number;
  executedQuantity: number;
  price?: number;
  timeInForce: 'IOC' | 'FOK' | 'GTC';
  minCounterpartyRating?: number;
  metadata: {
    venue?: string;
    anonymityLevel?: 'full' | 'partial';
    settlementTime?: string;
  };
}

export interface CrossExchangeArbitrage {
  id: string;
  pair: string;
  buyExchange: string;
  sellExchange: string;
  buyPrice: number;
  sellPrice: number;
  spread: number; // percentage
  estimatedProfit: number; // percentage after fees
  volume: number; // recommended volume
  confidence: number; // 0-1
  expiresAt: number;
}

export interface SmartOrderRouting {
  orderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  totalQuantity: number;
  splits: Array<{
    exchange: string;
    quantity: number;
    price: number;
    status: 'pending' | 'executed' | 'failed';
    executedPrice?: number;
    slippage?: number;
  }>;
  averagePrice: number;
  totalSlippage: number;
  savingsVsBestPrice: number;
}

export interface ExchangeCredentials {
  exchange: string;
  apiKey: string;
  apiSecret: string;
  passphrase?: string;
  subaccount?: string;
  testnet: boolean;
  permissions: string[];
  ipWhitelist?: string[];
  created: number;
  lastUsed?: number;
}
