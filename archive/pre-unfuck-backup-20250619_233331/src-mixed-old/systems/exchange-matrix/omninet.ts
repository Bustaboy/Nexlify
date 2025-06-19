// Location: /src/systems/exchange-matrix/omninet.ts
// Nexlify Omninet Exchange Matrix - Elite multi-exchange integration

import { BehaviorSubject, Subject, merge, interval, combineLatest } from 'rxjs';
import { filter, map, throttleTime, catchError, retry, timeout } from 'rxjs/operators';
import { WebSocket } from 'ws';
import ccxt from 'ccxt';
import { ethers } from 'ethers';
import {
  Exchange,
  OrderBook,
  Trade,
  Balance,
  Market,
  Ticker,
  ExchangeStatus,
  DarkPoolOrder,
  CrossExchangeArbitrage
} from '@/types/exchange.types';

export interface OmninetConfig {
  exchanges: {
    cex: ExchangeCredentials[];
    dex: DexConfig[];
    darkPools: DarkPoolConfig[];
    otc: OTCProviderConfig[];
  };
  routing: {
    smartOrderRouting: boolean;
    aggregationEnabled: boolean;
    latencyOptimization: boolean;
    privateMempool: boolean;
  };
  riskLimits: {
    maxSlippage: number;
    maxExposurePerExchange: number;
    minLiquidity: number;
    banList: string[]; // Banned exchanges
  };
  monitoring: {
    healthCheckInterval: number;
    latencyThreshold: number;
    volumeThreshold: number;
  };
}

interface ExchangeCredentials {
  id: string;
  name: string;
  apiKey: string;
  apiSecret: string;
  passphrase?: string;
  testnet: boolean;
  tier: 'primary' | 'secondary' | 'experimental';
  rateLimit: number;
  features: string[];
}

interface DexConfig {
  chainId: number;
  protocol: string;
  routerAddress: string;
  factoryAddress: string;
  subgraph?: string;
}

interface DarkPoolConfig {
  id: string;
  name: string;
  endpoint: string;
  authToken: string;
  minOrderSize: number;
}

interface OTCProviderConfig {
  id: string;
  name: string;
  apiEndpoint: string;
  credentials: any;
  supportedPairs: string[];
}

interface ExchangeHealth {
  exchangeId: string;
  status: 'healthy' | 'degraded' | 'offline';
  latency: number;
  lastUpdate: number;
  errorRate: number;
  volume24h: number;
  reliability: number; // 0-100 score
}

export class OmninetExchangeMatrix {
  // Core observables
  private exchanges$ = new BehaviorSubject<Map<string, Exchange>>(new Map());
  private orderBooks$ = new BehaviorSubject<Map<string, OrderBook>>(new Map());
  private healthStatus$ = new BehaviorSubject<Map<string, ExchangeHealth>>(new Map());
  private arbitrageOpportunities$ = new Subject<CrossExchangeArbitrage>();
  private darkPoolOrders$ = new Subject<DarkPoolOrder[]>();
  
  // Exchange instances
  private ccxtExchanges = new Map<string, ccxt.Exchange>();
  private dexConnections = new Map<string, ethers.Contract>();
  private wsConnections = new Map<string, WebSocket>();
  
  // Performance tracking
  private latencyTracker = new Map<string, number[]>();
  private volumeTracker = new Map<string, number>();
  private errorTracker = new Map<string, number>();
  
  private config: OmninetConfig;
  private isInitialized = false;

  // Supported exchanges (only reliable ones as requested)
  private readonly SUPPORTED_EXCHANGES = {
    // Tier 1 - Most reliable
    primary: [
      'binance',
      'coinbase',
      'kraken',
      'okx',
      'bybit',
      'kucoin',
      'gate',
      'bitfinex'
    ],
    // Tier 2 - Good but with caveats
    secondary: [
      'huobi',
      'bitget',
      'mexc',
      'crypto.com',
      'bitstamp',
      'gemini'
    ],
    // Tier 3 - Use with caution
    experimental: [
      'deribit',
      'phemex',
      'bitmart',
      'lbank'
    ],
    // Banned - Never use
    banned: [
      'ftx', // RIP
      'hotbit', // Scam
      'coinsbit', // Fake volume
      'yobit' // Unreliable
    ]
  };

  // DEX protocols
  private readonly DEX_PROTOCOLS = {
    ethereum: ['uniswapV3', 'sushiswap', 'curve', 'balancer'],
    bsc: ['pancakeswap', 'biswap', 'apeswap'],
    polygon: ['quickswap', 'sushiswap', 'curve'],
    arbitrum: ['uniswapV3', 'sushiswap', 'gmx'],
    optimism: ['uniswapV3', 'velodrome'],
    avalanche: ['traderJoe', 'pangolin'],
    fantom: ['spookyswap', 'spiritswap'],
    base: ['baseswap', 'aerodrome'],
    zkSync: ['syncswap', 'mute'],
    linea: ['lynex', 'velocore']
  };

  constructor(config: OmninetConfig) {
    this.config = config;
    this.validateConfiguration();
  }

  private validateConfiguration(): void {
    // Filter out banned exchanges
    this.config.exchanges.cex = this.config.exchanges.cex.filter(
      ex => !this.SUPPORTED_EXCHANGES.banned.includes(ex.id) &&
           !this.config.riskLimits.banList.includes(ex.id)
    );
    
    // Validate credentials
    this.config.exchanges.cex.forEach(ex => {
      if (!ex.apiKey || !ex.apiSecret) {
        console.warn(`Missing credentials for ${ex.name}, removing from active list`);
        this.config.exchanges.cex = this.config.exchanges.cex.filter(e => e.id !== ex.id);
      }
    });
  }

  public async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    console.log('[OMNINET] Initializing Exchange Matrix...');
    
    try {
      // Initialize CEX connections
      await this.initializeCEXConnections();
      
      // Initialize DEX connections
      await this.initializeDEXConnections();
      
      // Initialize dark pools
      await this.initializeDarkPools();
      
      // Start monitoring
      this.startHealthMonitoring();
      this.startArbitrageScanner();
      this.startVolumeTracking();
      
      this.isInitialized = true;
      console.log('[OMNINET] Exchange Matrix online. Connected to', this.ccxtExchanges.size, 'exchanges');
      
    } catch (error) {
      console.error('[OMNINET] Initialization failed:', error);
      throw error;
    }
  }

  private async initializeCEXConnections(): Promise<void> {
    const initPromises = this.config.exchanges.cex.map(async (exchangeConfig) => {
      try {
        // Create CCXT instance
        const ExchangeClass = ccxt[exchangeConfig.id as keyof typeof ccxt] as any;
        if (!ExchangeClass) {
          console.warn(`Exchange ${exchangeConfig.id} not supported by CCXT`);
          return;
        }
        
        const exchange = new ExchangeClass({
          apiKey: exchangeConfig.apiKey,
          secret: exchangeConfig.apiSecret,
          password: exchangeConfig.passphrase,
          enableRateLimit: true,
          rateLimit: exchangeConfig.rateLimit,
          options: {
            defaultType: 'spot', // Can be 'future', 'margin', etc.
            adjustForTimeDifference: true,
            recvWindow: 60000
          }
        });
        
        // Test connection
        await exchange.loadMarkets();
        
        this.ccxtExchanges.set(exchangeConfig.id, exchange);
        
        // Initialize WebSocket if available
        if (exchange.has.ws) {
          await this.initializeWebSocket(exchangeConfig.id, exchange);
        }
        
        // Set up exchange info
        const exchangeInfo: Exchange = {
          id: exchangeConfig.id,
          name: exchangeConfig.name,
          status: 'online',
          markets: Object.keys(exchange.markets).length,
          fees: {
            maker: exchange.fees?.trading?.maker || 0.001,
            taker: exchange.fees?.trading?.taker || 0.001
          },
          features: {
            spot: exchange.has.spot,
            futures: exchange.has.future,
            margin: exchange.has.margin,
            staking: exchange.has.staking,
            websocket: exchange.has.ws
          },
          limits: exchange.limits,
          tier: exchangeConfig.tier
        };
        
        const exchanges = this.exchanges$.value;
        exchanges.set(exchangeConfig.id, exchangeInfo);
        this.exchanges$.next(exchanges);
        
        console.log(`[OMNINET] Connected to ${exchangeConfig.name}`);
        
      } catch (error) {
        console.error(`[OMNINET] Failed to connect to ${exchangeConfig.name}:`, error);
        
        // Mark as offline
        const exchanges = this.exchanges$.value;
        exchanges.set(exchangeConfig.id, {
          id: exchangeConfig.id,
          name: exchangeConfig.name,
          status: 'offline',
          markets: 0,
          fees: { maker: 0, taker: 0 },
          features: {},
          limits: {},
          tier: exchangeConfig.tier
        });
        this.exchanges$.next(exchanges);
      }
    });
    
    await Promise.allSettled(initPromises);
  }

  private async initializeWebSocket(exchangeId: string, exchange: ccxt.Exchange): Promise<void> {
    try {
      // Exchange-specific WebSocket setup
      let wsUrl: string;
      
      switch (exchangeId) {
        case 'binance':
          wsUrl = exchange.urls.api.ws + '/ws';
          break;
        case 'kraken':
          wsUrl = 'wss://ws.kraken.com';
          break;
        case 'coinbase':
          wsUrl = 'wss://ws-feed.exchange.coinbase.com';
          break;
        default:
          return; // Skip if no known WebSocket endpoint
      }
      
      const ws = new WebSocket(wsUrl);
      
      ws.on('open', () => {
        console.log(`[OMNINET] WebSocket connected to ${exchangeId}`);
        
        // Subscribe to order book updates for top pairs
        const topPairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'];
        this.subscribeToOrderBooks(exchangeId, ws, topPairs);
      });
      
      ws.on('message', (data: any) => {
        this.handleWebSocketMessage(exchangeId, data);
      });
      
      ws.on('error', (error) => {
        console.error(`[OMNINET] WebSocket error for ${exchangeId}:`, error);
      });
      
      ws.on('close', () => {
        console.log(`[OMNINET] WebSocket disconnected from ${exchangeId}`);
        // Attempt reconnection after 5 seconds
        setTimeout(() => this.initializeWebSocket(exchangeId, exchange), 5000);
      });
      
      this.wsConnections.set(exchangeId, ws);
      
    } catch (error) {
      console.error(`[OMNINET] WebSocket initialization failed for ${exchangeId}:`, error);
    }
  }

  private async initializeDEXConnections(): Promise<void> {
    for (const dexConfig of this.config.exchanges.dex) {
      try {
        const provider = new ethers.providers.JsonRpcProvider(
          this.getChainRPC(dexConfig.chainId)
        );
        
        // Load router contract ABI based on protocol
        const routerABI = this.getRouterABI(dexConfig.protocol);
        const router = new ethers.Contract(
          dexConfig.routerAddress,
          routerABI,
          provider
        );
        
        this.dexConnections.set(
          `${dexConfig.protocol}_${dexConfig.chainId}`,
          router
        );
        
        console.log(`[OMNINET] Connected to ${dexConfig.protocol} on chain ${dexConfig.chainId}`);
        
      } catch (error) {
        console.error(`[OMNINET] Failed to connect to DEX ${dexConfig.protocol}:`, error);
      }
    }
  }

  private async initializeDarkPools(): Promise<void> {
    // Dark pool connections are more complex and often require special access
    for (const poolConfig of this.config.exchanges.darkPools) {
      try {
        // Placeholder for dark pool initialization
        console.log(`[OMNINET] Connecting to dark pool: ${poolConfig.name}`);
        
        // In production, this would involve:
        // 1. Authenticated WebSocket connections
        // 2. Special order types (iceberg, hidden)
        // 3. Minimum order size enforcement
        // 4. Counterparty verification
        
      } catch (error) {
        console.error(`[OMNINET] Dark pool connection failed for ${poolConfig.name}:`, error);
      }
    }
  }

  private startHealthMonitoring(): void {
    interval(this.config.monitoring.healthCheckInterval).subscribe(async () => {
      const healthChecks = new Map<string, ExchangeHealth>();
      
      for (const [exchangeId, exchange] of this.ccxtExchanges) {
        try {
          const startTime = Date.now();
          
          // Ping exchange
          await exchange.fetchTicker('BTC/USDT');
          
          const latency = Date.now() - startTime;
          
          // Update latency tracking
          if (!this.latencyTracker.has(exchangeId)) {
            this.latencyTracker.set(exchangeId, []);
          }
          const latencies = this.latencyTracker.get(exchangeId)!;
          latencies.push(latency);
          if (latencies.length > 100) latencies.shift();
          
          // Calculate health metrics
          const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
          const errorRate = (this.errorTracker.get(exchangeId) || 0) / 100;
          const volume24h = this.volumeTracker.get(exchangeId) || 0;
          
          const health: ExchangeHealth = {
            exchangeId,
            status: latency < this.config.monitoring.latencyThreshold ? 'healthy' : 
                   latency < this.config.monitoring.latencyThreshold * 2 ? 'degraded' : 'offline',
            latency: avgLatency,
            lastUpdate: Date.now(),
            errorRate,
            volume24h,
            reliability: this.calculateReliability(avgLatency, errorRate, volume24h)
          };
          
          healthChecks.set(exchangeId, health);
          
        } catch (error) {
          console.error(`[OMNINET] Health check failed for ${exchangeId}:`, error);
          
          healthChecks.set(exchangeId, {
            exchangeId,
            status: 'offline',
            latency: Infinity,
            lastUpdate: Date.now(),
            errorRate: 1,
            volume24h: 0,
            reliability: 0
          });
          
          // Increment error counter
          this.errorTracker.set(exchangeId, (this.errorTracker.get(exchangeId) || 0) + 1);
        }
      }
      
      this.healthStatus$.next(healthChecks);
    });
  }

  private startArbitrageScanner(): void {
    // Scan for arbitrage opportunities across exchanges
    interval(1000).pipe(
      throttleTime(500) // Prevent overwhelming the system
    ).subscribe(async () => {
      const opportunities: CrossExchangeArbitrage[] = [];
      const pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MATIC/USDT'];
      
      for (const pair of pairs) {
        const prices = new Map<string, number>();
        
        // Collect prices from all healthy exchanges
        for (const [exchangeId, exchange] of this.ccxtExchanges) {
          const health = this.healthStatus$.value.get(exchangeId);
          if (health?.status !== 'healthy') continue;
          
          try {
            const ticker = await exchange.fetchTicker(pair);
            prices.set(exchangeId, ticker.bid || 0);
          } catch (error) {
            // Silent fail for individual price fetches
          }
        }
        
        // Find arbitrage opportunities
        const priceArray = Array.from(prices.entries());
        for (let i = 0; i < priceArray.length; i++) {
          for (let j = i + 1; j < priceArray.length; j++) {
            const [buyExchange, buyPrice] = priceArray[i];
            const [sellExchange, sellPrice] = priceArray[j];
            
            const spread = Math.abs(sellPrice - buyPrice) / buyPrice;
            
            // Check if profitable after fees
            const buyFee = this.exchanges$.value.get(buyExchange)?.fees.taker || 0.001;
            const sellFee = this.exchanges$.value.get(sellExchange)?.fees.maker || 0.001;
            const totalFees = buyFee + sellFee;
            
            if (spread > totalFees + 0.001) { // 0.1% minimum profit
              opportunities.push({
                id: `arb_${Date.now()}_${pair}`,
                pair,
                buyExchange: buyPrice < sellPrice ? buyExchange : sellExchange,
                sellExchange: buyPrice < sellPrice ? sellExchange : buyExchange,
                buyPrice: Math.min(buyPrice, sellPrice),
                sellPrice: Math.max(buyPrice, sellPrice),
                spread: spread * 100,
                estimatedProfit: spread - totalFees,
                volume: this.estimateArbitrageVolume(pair, Math.min(buyPrice, sellPrice)),
                confidence: this.calculateArbitrageConfidence(spread, totalFees),
                expiresAt: Date.now() + 5000
              });
            }
          }
        }
      }
      
      // Emit high-confidence opportunities
      opportunities
        .filter(opp => opp.confidence > 0.7)
        .forEach(opp => this.arbitrageOpportunities$.next(opp));
    });
  }

  private startVolumeTracking(): void {
    // Track 24h volume for each exchange
    interval(300000).subscribe(async () => { // Every 5 minutes
      for (const [exchangeId, exchange] of this.ccxtExchanges) {
        try {
          const tickers = await exchange.fetchTickers();
          const totalVolume = Object.values(tickers).reduce(
            (sum: number, ticker: any) => sum + (ticker.quoteVolume || 0), 
            0
          );
          
          this.volumeTracker.set(exchangeId, totalVolume);
        } catch (error) {
          console.error(`[OMNINET] Volume tracking failed for ${exchangeId}:`, error);
        }
      }
    });
  }

  // Public methods for order execution
  public async executeOrder(params: {
    exchange: string;
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit';
    amount: number;
    price?: number;
    params?: any;
  }): Promise<any> {
    const exchange = this.ccxtExchanges.get(params.exchange);
    if (!exchange) {
      throw new Error(`Exchange ${params.exchange} not connected`);
    }
    
    // Validate exchange health
    const health = this.healthStatus$.value.get(params.exchange);
    if (health?.status === 'offline') {
      throw new Error(`Exchange ${params.exchange} is offline`);
    }
    
    try {
      const order = await exchange.createOrder(
        params.symbol,
        params.type,
        params.side,
        params.amount,
        params.price,
        params.params
      );
      
      console.log(`[OMNINET] Order executed on ${params.exchange}:`, order.id);
      return order;
      
    } catch (error) {
      console.error(`[OMNINET] Order execution failed on ${params.exchange}:`, error);
      throw error;
    }
  }

  public async executeSmartOrder(params: {
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    maxSlippage: number;
    preferredExchanges?: string[];
    darkPoolEnabled?: boolean;
  }): Promise<any> {
    if (!this.config.routing.smartOrderRouting) {
      throw new Error('Smart order routing is disabled');
    }
    
    // Get best execution venues
    const venues = await this.findBestExecutionVenues(
      params.symbol,
      params.side,
      params.amount,
      params.preferredExchanges
    );
    
    // Split order across venues for best execution
    const orderSplits = this.calculateOrderSplits(
      params.amount,
      venues,
      params.maxSlippage
    );
    
    // Execute orders in parallel
    const executionPromises = orderSplits.map(split => 
      this.executeOrder({
        exchange: split.exchange,
        symbol: params.symbol,
        side: params.side,
        type: 'limit',
        amount: split.amount,
        price: split.price
      })
    );
    
    const results = await Promise.allSettled(executionPromises);
    
    return {
      totalAmount: params.amount,
      executedAmount: results
        .filter(r => r.status === 'fulfilled')
        .reduce((sum, r: any) => sum + r.value.filled, 0),
      orders: results,
      averagePrice: this.calculateAveragePrice(results)
    };
  }

  // Helper methods
  private calculateReliability(latency: number, errorRate: number, volume: number): number {
    const latencyScore = Math.max(0, 100 - (latency / 10));
    const errorScore = Math.max(0, 100 - (errorRate * 100));
    const volumeScore = Math.min(100, volume / 1000000);
    
    return (latencyScore * 0.3 + errorScore * 0.5 + volumeScore * 0.2);
  }

  private estimateArbitrageVolume(pair: string, price: number): number {
    // Estimate safe arbitrage volume based on liquidity
    const baseVolume = 10000; // $10k base
    const priceMultiplier = 50000 / price; // Adjust for asset price
    
    return baseVolume * priceMultiplier;
  }

  private calculateArbitrageConfidence(spread: number, fees: number): number {
    const profitMargin = spread - fees;
    const baseConfidence = Math.min(1, profitMargin * 100);
    
    // Adjust for market conditions
    const volatilityPenalty = 0.1; // Would calculate from real volatility
    
    return Math.max(0, baseConfidence - volatilityPenalty);
  }

  private async findBestExecutionVenues(
    symbol: string,
    side: 'buy' | 'sell',
    amount: number,
    preferredExchanges?: string[]
  ): Promise<any[]> {
    const venues = [];
    
    for (const [exchangeId, exchange] of this.ccxtExchanges) {
      try {
        const orderBook = await exchange.fetchOrderBook(symbol);
        const liquidity = side === 'buy' 
          ? orderBook.asks.reduce((sum, [price, vol]) => sum + vol, 0)
          : orderBook.bids.reduce((sum, [price, vol]) => sum + vol, 0);
        
        if (liquidity > amount * 0.1) { // At least 10% of order size
          venues.push({
            exchange: exchangeId,
            liquidity,
            bestPrice: side === 'buy' ? orderBook.asks[0][0] : orderBook.bids[0][0],
            depth: orderBook
          });
        }
      } catch (error) {
        // Skip failed exchanges
      }
    }
    
    // Sort by best price
    return venues.sort((a, b) => 
      side === 'buy' 
        ? a.bestPrice - b.bestPrice 
        : b.bestPrice - a.bestPrice
    );
  }

  private calculateOrderSplits(
    totalAmount: number,
    venues: any[],
    maxSlippage: number
  ): any[] {
    const splits = [];
    let remainingAmount = totalAmount;
    
    for (const venue of venues) {
      if (remainingAmount <= 0) break;
      
      // Calculate amount that can be filled without exceeding slippage
      const venueAmount = Math.min(
        remainingAmount,
        venue.liquidity * 0.5 // Don't take more than 50% of available liquidity
      );
      
      splits.push({
        exchange: venue.exchange,
        amount: venueAmount,
        price: venue.bestPrice * (1 + maxSlippage)
      });
      
      remainingAmount -= venueAmount;
    }
    
    return splits;
  }

  private calculateAveragePrice(results: PromiseSettledResult<any>[]): number {
    let totalValue = 0;
    let totalAmount = 0;
    
    results.forEach(result => {
      if (result.status === 'fulfilled' && result.value) {
        totalValue += result.value.cost || 0;
        totalAmount += result.value.filled || 0;
      }
    });
    
    return totalAmount > 0 ? totalValue / totalAmount : 0;
  }

  private subscribeToOrderBooks(exchangeId: string, ws: WebSocket, pairs: string[]): void {
    // Exchange-specific subscription messages
    switch (exchangeId) {
      case 'binance':
        pairs.forEach(pair => {
          const symbol = pair.replace('/', '').toLowerCase();
          ws.send(JSON.stringify({
            method: 'SUBSCRIBE',
            params: [`${symbol}@depth20`],
            id: Date.now()
          }));
        });
        break;
        
      case 'kraken':
        ws.send(JSON.stringify({
          event: 'subscribe',
          pair: pairs,
          subscription: { name: 'book', depth: 25 }
        }));
        break;
        
      case 'coinbase':
        ws.send(JSON.stringify({
          type: 'subscribe',
          product_ids: pairs.map(p => p.replace('/', '-')),
          channels: ['level2']
        }));
        break;
    }
  }

  private handleWebSocketMessage(exchangeId: string, data: any): void {
    try {
      const message = JSON.parse(data.toString());
      
      // Parse message based on exchange format
      switch (exchangeId) {
        case 'binance':
          if (message.e === 'depthUpdate') {
            this.updateOrderBook(exchangeId, message.s, {
              bids: message.b,
              asks: message.a
            });
          }
          break;
          
        case 'kraken':
          if (Array.isArray(message) && message[2] === 'book-25') {
            // Kraken order book update
          }
          break;
          
        case 'coinbase':
          if (message.type === 'l2update') {
            // Coinbase level 2 update
          }
          break;
      }
    } catch (error) {
      console.error(`[OMNINET] Failed to parse WebSocket message from ${exchangeId}:`, error);
    }
  }

  private updateOrderBook(exchangeId: string, symbol: string, data: any): void {
    const key = `${exchangeId}_${symbol}`;
    const orderBooks = this.orderBooks$.value;
    
    orderBooks.set(key, {
      exchange: exchangeId,
      symbol,
      bids: data.bids,
      asks: data.asks,
      timestamp: Date.now()
    });
    
    this.orderBooks$.next(orderBooks);
  }

  private getChainRPC(chainId: number): string {
    const rpcs: Record<number, string> = {
      1: 'https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY',
      56: 'https://bsc-dataseed.binance.org/',
      137: 'https://polygon-rpc.com/',
      42161: 'https://arb1.arbitrum.io/rpc',
      10: 'https://mainnet.optimism.io',
      43114: 'https://api.avax.network/ext/bc/C/rpc'
    };
    
    return rpcs[chainId] || '';
  }

  private getRouterABI(protocol: string): any[] {
    // Return appropriate ABI based on protocol
    // This would contain the actual ABIs in production
    return [];
  }

  // Public observables
  public getExchanges$() {
    return this.exchanges$.asObservable();
  }

  public getHealthStatus$() {
    return this.healthStatus$.asObservable();
  }

  public getArbitrageOpportunities$() {
    return this.arbitrageOpportunities$.asObservable();
  }

  public getOrderBooks$() {
    return this.orderBooks$.asObservable();
  }

  public getDarkPoolOrders$() {
    return this.darkPoolOrders$.asObservable();
  }

  // Cleanup
  public async disconnect(): Promise<void> {
    // Close all WebSocket connections
    this.wsConnections.forEach(ws => ws.close());
    
    // Clear all subscriptions
    this.exchanges$.complete();
    this.orderBooks$.complete();
    this.healthStatus$.complete();
    this.arbitrageOpportunities$.complete();
    this.darkPoolOrders$.complete();
    
    console.log('[OMNINET] Exchange Matrix disconnected');
  }
}
