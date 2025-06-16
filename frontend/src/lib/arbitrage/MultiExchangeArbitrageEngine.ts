// frontend/src/lib/arbitrage/MultiExchangeArbitrageEngine.ts
/**
 * Multi-Exchange Arbitrage Engine - Digital Predator
 * Hunt profit in the gaps between exchange prices
 * 
 * This is the chrome that hunts while you sleep, hermano. The digital
 * coyote that slips between market inefficiencies faster than a corpo
 * can blink. Been perfecting this for years - cada price discrepancy
 * is a heartbeat of opportunity, cada successful arb a small victory
 * against the system.
 * 
 * In Night City, information flows like blood through digital veins.
 * And we're about to tap that vein, drain every eddy from the price
 * differences that the big players are too slow to catch.
 * 
 * Remember: In arbitrage, speed isn't everything - it's the ONLY thing.
 * Hesitate for a nanosecond, and someone else's eating your profit
 * while you're still thinking about it.
 */

import { EventEmitter } from 'events';
import { executionPipeline } from '../execution/OrderExecutionPipeline';
import { riskEngine } from '../risk/RiskManagementEngine';
import { nexlifyCache } from '../cache/IndexedDBCache';

// Types - the data structures that define profit opportunities
interface ArbitrageOpportunity {
  id: string;
  type: 'simple' | 'triangular' | 'cross_currency' | 'funding_rate';
  
  // Basic opportunity data
  symbol: string;
  buyExchange: string;
  sellExchange: string;
  buyPrice: number;
  sellPrice: number;
  spreadPercent: number;
  spreadAbsolute: number;
  
  // Profitability analysis
  grossProfit: number;
  netProfit: number;
  profitPercent: number;
  roi: number;                    // Return on investment
  
  // Execution parameters
  maxSize: number;               // Maximum profitable size
  optimalSize: number;           // Size that maximizes profit
  requiredCapital: number;       // Capital needed for execution
  
  // Risk metrics
  riskScore: number;             // 0-100 risk assessment
  executionRisk: number;         // Risk of partial execution
  latencyRisk: number;           // Risk due to price movement
  liquidityRisk: number;         // Risk due to order book depth
  
  // Timing data
  detectedAt: number;
  validUntil: number;
  estimatedDuration: number;     // How long opportunity might last
  
  // Execution tracking
  status: 'detected' | 'analyzing' | 'executing' | 'completed' | 'failed' | 'expired';
  executionStarted?: number;
  executionCompleted?: number;
  actualProfit?: number;
  
  // Metadata
  confidence: number;            // 0-1 confidence in opportunity
  historicalSuccess: number;     // Historical success rate for similar ops
  marketConditions: {
    volatility: number;
    volume: number;
    momentum: string;
    sentiment: string;
  };
}

interface TriangularArbitrage extends ArbitrageOpportunity {
  type: 'triangular';
  path: string[];               // e.g., ['BTC', 'ETH', 'USDT', 'BTC']
  exchanges: string[];          // Exchange for each leg
  prices: number[];             // Price for each leg
  legs: Array<{
    from: string;
    to: string;
    exchange: string;
    symbol: string;
    price: number;
    side: 'buy' | 'sell';
    estimatedFee: number;
  }>;
}

interface ExchangePriceData {
  exchange: string;
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  volume24h: number;
  timestamp: number;
  orderBookDepth: {
    bidDepth: number;          // Total bid volume at top 10 levels
    askDepth: number;          // Total ask volume at top 10 levels
  };
  latency: number;             // Time to receive this data
}

interface ArbitrageMetrics {
  totalOpportunities: number;
  successfulTrades: number;
  failedTrades: number;
  totalProfit: number;
  totalVolume: number;
  
  // Performance metrics
  avgProfitPercent: number;
  avgExecutionTime: number;
  successRate: number;
  
  // Risk metrics
  maxDrawdown: number;
  sharpeRatio: number;
  
  // Opportunity analysis
  avgSpread: number;
  medianSpread: number;
  largestSpread: number;
  
  // By exchange pair
  exchangePairStats: Map<string, {
    opportunities: number;
    avgSpread: number;
    successRate: number;
    totalProfit: number;
  }>;
  
  // By time periods
  hourlyStats: Array<{
    hour: number;
    opportunities: number;
    avgSpread: number;
    volume: number;
  }>;
  
  lastUpdated: number;
}

interface ArbitrageConfig {
  // Opportunity detection
  minSpreadPercent: number;      // Minimum spread to consider
  maxSpreadPercent: number;      // Maximum spread (too good to be true)
  minProfitUSD: number;          // Minimum profit in USD
  maxPositionSize: number;       // Maximum position size per opportunity
  
  // Risk management
  maxConcurrentArbs: number;     // Maximum simultaneous arbitrages
  maxExposurePercent: number;    // Max portfolio exposure to arbitrage
  stopLossPercent: number;       // Stop loss for failed arbitrages
  
  // Execution settings
  maxLatencyMs: number;          // Maximum acceptable latency
  slippageBuffer: number;        // Extra slippage buffer
  executionTimeoutMs: number;    // Timeout for opportunity execution
  
  // Filtering
  enabledExchanges: string[];    // Which exchanges to monitor
  enabledSymbols: string[];      // Which symbols to arbitrage
  blacklistPairs: string[];      // Exchange pairs to avoid
  
  // Strategy settings
  enableSimpleArbitrage: boolean;
  enableTriangularArbitrage: boolean;
  enableCrossCurrencyArbitrage: boolean;
  enableFundingRateArbitrage: boolean;
}

// Price Monitor - watches exchange prices for discrepancies
class PriceMonitor extends EventEmitter {
  private priceFeeds: Map<string, Map<string, ExchangePriceData>> = new Map();
  private lastUpdateTimes: Map<string, number> = new Map();
  private subscriptions: Set<string> = new Set();
  private monitoringInterval: NodeJS.Timeout | null = null;
  
  constructor() {
    super();
    this.startMonitoring();
  }
  
  /**
   * Start price monitoring across exchanges
   */
  private startMonitoring(): void {
    this.monitoringInterval = setInterval(() => {
      this.updatePriceFeeds();
      this.detectStaleData();
    }, 100); // Update every 100ms for maximum responsiveness
    
    console.log('👁️ Price monitor online - Watching for discrepancies across the digital wasteland');
  }
  
  /**
   * Subscribe to price updates for a symbol
   */
  subscribeSymbol(symbol: string, exchanges: string[]): void {
    for (const exchange of exchanges) {
      const key = `${exchange}:${symbol}`;
      this.subscriptions.add(key);
      
      // Initialize price feed if not exists
      if (!this.priceFeeds.has(exchange)) {
        this.priceFeeds.set(exchange, new Map());
      }
    }
    
    console.log(`📡 Subscribed to ${symbol} on ${exchanges.join(', ')} - Data streams flowing`);
  }
  
  /**
   * Update price feeds from exchanges
   */
  private async updatePriceFeeds(): Promise<void> {
    const updatePromises: Promise<void>[] = [];
    
    for (const subscription of this.subscriptions) {
      const [exchange, symbol] = subscription.split(':');
      updatePromises.push(this.fetchPriceData(exchange, symbol));
    }
    
    await Promise.allSettled(updatePromises);
  }
  
  /**
   * Fetch price data from a specific exchange
   */
  private async fetchPriceData(exchange: string, symbol: string): Promise<void> {
    const startTime = performance.now();
    
    try {
      // In production, this would make real API calls
      // For now, simulate realistic price data with small variations
      const basePrice = await this.getBasePriceFromCache(symbol);
      const priceVariation = (Math.random() - 0.5) * 0.002; // ±0.1% variation
      const latencyVariation = Math.random() * 50; // 0-50ms latency variation
      
      // Exchange-specific price adjustments
      let exchangeAdjustment = 0;
      switch (exchange) {
        case 'binance':
          exchangeAdjustment = (Math.random() - 0.5) * 0.0005; // Tight spreads
          break;
        case 'coinbase':
          exchangeAdjustment = (Math.random() - 0.5) * 0.001;  // Medium spreads
          break;
        case 'kraken':
          exchangeAdjustment = (Math.random() - 0.5) * 0.0015; // Wider spreads
          break;
        case 'okx':
          exchangeAdjustment = (Math.random() - 0.5) * 0.0008; // Competitive spreads
          break;
      }
      
      const adjustedPrice = basePrice * (1 + priceVariation + exchangeAdjustment);
      const spread = adjustedPrice * (0.0001 + Math.random() * 0.0005); // 0.01-0.06% spread
      
      const priceData: ExchangePriceData = {
        exchange,
        symbol,
        bid: adjustedPrice - spread / 2,
        ask: adjustedPrice + spread / 2,
        spread: spread / adjustedPrice,
        volume24h: 1000000 + Math.random() * 5000000, // Random volume
        timestamp: Date.now(),
        orderBookDepth: {
          bidDepth: 50000 + Math.random() * 200000,
          askDepth: 50000 + Math.random() * 200000
        },
        latency: performance.now() - startTime + latencyVariation
      };
      
      // Store the price data
      const exchangeMap = this.priceFeeds.get(exchange);
      if (exchangeMap) {
        exchangeMap.set(symbol, priceData);
        this.lastUpdateTimes.set(`${exchange}:${symbol}`, Date.now());
      }
      
      // Emit price update
      this.emit('priceUpdate', priceData);
      
    } catch (error) {
      console.error(`Price feed error for ${exchange}:${symbol}:`, error);
    }
  }
  
  /**
   * Get base price from cache or generate realistic price
   */
  private async getBasePriceFromCache(symbol: string): Promise<number> {
    // Try to get from cache first
    const cached = await nexlifyCache.getMarketData(symbol);
    if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute freshness
      return cached.price;
    }
    
    // Generate realistic base prices for common symbols
    const basePrices: Record<string, number> = {
      'BTC/USDT': 50000 + Math.random() * 20000,
      'ETH/USDT': 2500 + Math.random() * 1000,
      'SOL/USDT': 80 + Math.random() * 40,
      'ADA/USDT': 0.3 + Math.random() * 0.2,
      'DOT/USDT': 5 + Math.random() * 3,
      'MATIC/USDT': 0.8 + Math.random() * 0.4,
      'LINK/USDT': 12 + Math.random() * 6,
      'UNI/USDT': 6 + Math.random() * 3,
      'AAVE/USDT': 150 + Math.random() * 50,
      'AVAX/USDT': 25 + Math.random() * 15
    };
    
    return basePrices[symbol] || 100 + Math.random() * 50;
  }
  
  /**
   * Detect and handle stale price data
   */
  private detectStaleData(): void {
    const now = Date.now();
    const staleThreshold = 5000; // 5 seconds
    
    for (const [key, lastUpdate] of this.lastUpdateTimes) {
      if (now - lastUpdate > staleThreshold) {
        console.warn(`📡 Stale data detected for ${key} - last update ${now - lastUpdate}ms ago`);
        this.emit('staleData', key);
      }
    }
  }
  
  /**
   * Get current price data for all exchanges and symbol
   */
  getPriceData(symbol: string): ExchangePriceData[] {
    const prices: ExchangePriceData[] = [];
    
    for (const [exchange, symbolMap] of this.priceFeeds) {
      const priceData = symbolMap.get(symbol);
      if (priceData) {
        prices.push(priceData);
      }
    }
    
    return prices.sort((a, b) => b.timestamp - a.timestamp); // Most recent first
  }
  
  /**
   * Get all symbols being monitored
   */
  getMonitoredSymbols(): string[] {
    const symbols = new Set<string>();
    
    for (const subscription of this.subscriptions) {
      const [, symbol] = subscription.split(':');
      symbols.add(symbol);
    }
    
    return Array.from(symbols);
  }
  
  /**
   * Shutdown monitoring
   */
  shutdown(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    this.priceFeeds.clear();
    this.subscriptions.clear();
    console.log('💀 Price monitor shutdown');
  }
}

// Opportunity Detector - identifies profitable arbitrage opportunities
class OpportunityDetector extends EventEmitter {
  private config: ArbitrageConfig;
  private detectionHistory: Array<{ opportunity: ArbitrageOpportunity; timestamp: number }> = [];
  
  constructor(config: ArbitrageConfig) {
    super();
    this.config = config;
  }
  
  /**
   * Analyze price data for arbitrage opportunities
   */
  analyzePriceData(symbol: string, priceData: ExchangePriceData[]): ArbitrageOpportunity[] {
    const opportunities: ArbitrageOpportunity[] = [];
    
    if (priceData.length < 2) return opportunities;
    
    // Simple arbitrage detection
    if (this.config.enableSimpleArbitrage) {
      opportunities.push(...this.detectSimpleArbitrage(symbol, priceData));
    }
    
    // Triangular arbitrage detection
    if (this.config.enableTriangularArbitrage) {
      opportunities.push(...this.detectTriangularArbitrage(symbol, priceData));
    }
    
    // Filter opportunities by profitability and risk
    const filteredOpportunities = opportunities.filter(opp => this.isOpportunityViable(opp));
    
    // Cache detected opportunities
    for (const opp of filteredOpportunities) {
      this.detectionHistory.push({ opportunity: opp, timestamp: Date.now() });
    }
    
    // Keep history manageable
    if (this.detectionHistory.length > 10000) {
      this.detectionHistory = this.detectionHistory.slice(-5000);
    }
    
    return filteredOpportunities;
  }
  
  /**
   * Detect simple buy-low-sell-high arbitrage opportunities
   */
  private detectSimpleArbitrage(symbol: string, priceData: ExchangePriceData[]): ArbitrageOpportunity[] {
    const opportunities: ArbitrageOpportunity[] = [];
    
    // Sort by bid price (highest first) and ask price (lowest first)
    const sortedByBid = [...priceData].sort((a, b) => b.bid - a.bid);
    const sortedByAsk = [...priceData].sort((a, b) => a.ask - b.ask);
    
    // Check each combination of buy low, sell high
    for (const sellExchange of sortedByBid) {
      for (const buyExchange of sortedByAsk) {
        if (sellExchange.exchange === buyExchange.exchange) continue;
        
        // Calculate spread
        const spread = sellExchange.bid - buyExchange.ask;
        const spreadPercent = (spread / buyExchange.ask) * 100;
        
        // Check if spread meets minimum threshold
        if (spreadPercent < this.config.minSpreadPercent) continue;
        if (spreadPercent > this.config.maxSpreadPercent) continue; // Too good to be true
        
        // Calculate opportunity details
        const opportunity = this.calculateSimpleArbitrageOpportunity(
          symbol,
          buyExchange,
          sellExchange,
          spread,
          spreadPercent
        );
        
        if (opportunity.netProfit >= this.config.minProfitUSD) {
          opportunities.push(opportunity);
        }
      }
    }
    
    return opportunities;
  }
  
  /**
   * Calculate simple arbitrage opportunity details
   */
  private calculateSimpleArbitrageOpportunity(
    symbol: string,
    buyExchange: ExchangePriceData,
    sellExchange: ExchangePriceData,
    spread: number,
    spreadPercent: number
  ): ArbitrageOpportunity {
    
    // Calculate maximum size based on order book depth
    const maxBuySize = Math.min(buyExchange.orderBookDepth.askDepth, this.config.maxPositionSize);
    const maxSellSize = Math.min(sellExchange.orderBookDepth.bidDepth, this.config.maxPositionSize);
    const maxSize = Math.min(maxBuySize, maxSellSize);
    
    // Estimate fees
    const buyFee = this.estimateTradingFee(buyExchange.exchange, maxSize * buyExchange.ask);
    const sellFee = this.estimateTradingFee(sellExchange.exchange, maxSize * sellExchange.bid);
    const totalFees = buyFee + sellFee;
    
    // Calculate profits
    const grossProfit = spread * maxSize;
    const netProfit = grossProfit - totalFees;
    const profitPercent = (netProfit / (maxSize * buyExchange.ask)) * 100;
    const requiredCapital = maxSize * buyExchange.ask;
    const roi = (netProfit / requiredCapital) * 100;
    
    // Calculate optimal size (maximize profit while considering fees)
    const optimalSize = this.calculateOptimalSize(
      buyExchange.ask,
      sellExchange.bid,
      buyExchange.exchange,
      sellExchange.exchange,
      maxSize
    );
    
    // Risk assessment
    const riskScore = this.calculateRiskScore({
      spread: spreadPercent,
      buyLatency: buyExchange.latency,
      sellLatency: sellExchange.latency,
      buyDepth: buyExchange.orderBookDepth.askDepth,
      sellDepth: sellExchange.orderBookDepth.bidDepth,
      volume: Math.min(buyExchange.volume24h, sellExchange.volume24h)
    });
    
    // Market conditions analysis
    const marketConditions = this.analyzeMarketConditions(symbol, [buyExchange, sellExchange]);
    
    // Opportunity duration estimate
    const estimatedDuration = this.estimateOpportunityDuration(spreadPercent, riskScore);
    
    return {
      id: `arb_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      type: 'simple',
      symbol,
      buyExchange: buyExchange.exchange,
      sellExchange: sellExchange.exchange,
      buyPrice: buyExchange.ask,
      sellPrice: sellExchange.bid,
      spreadPercent,
      spreadAbsolute: spread,
      
      grossProfit,
      netProfit,
      profitPercent,
      roi,
      
      maxSize,
      optimalSize,
      requiredCapital,
      
      riskScore,
      executionRisk: this.calculateExecutionRisk(buyExchange, sellExchange),
      latencyRisk: this.calculateLatencyRisk(buyExchange.latency, sellExchange.latency),
      liquidityRisk: this.calculateLiquidityRisk(buyExchange.orderBookDepth, sellExchange.orderBookDepth),
      
      detectedAt: Date.now(),
      validUntil: Date.now() + estimatedDuration,
      estimatedDuration,
      
      status: 'detected',
      confidence: this.calculateConfidence(spreadPercent, riskScore, marketConditions),
      historicalSuccess: this.getHistoricalSuccessRate(buyExchange.exchange, sellExchange.exchange),
      marketConditions
    };
  }
  
  /**
   * Detect triangular arbitrage opportunities
   */
  private detectTriangularArbitrage(symbol: string, priceData: ExchangePriceData[]): TriangularArbitrage[] {
    const opportunities: TriangularArbitrage[] = [];
    
    // For triangular arbitrage, we need at least 3 currency pairs
    // Example: BTC/USDT -> ETH/BTC -> ETH/USDT -> USDT
    
    const baseSymbol = symbol.split('/')[0];
    const quoteSymbol = symbol.split('/')[1];
    
    // Common intermediate currencies for triangular arbitrage
    const intermediateCurrencies = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB'];
    
    for (const intermediate of intermediateCurrencies) {
      if (intermediate === baseSymbol || intermediate === quoteSymbol) continue;
      
      // Try to find a triangular path
      const triangularPath = this.findTriangularPath(baseSymbol, intermediate, quoteSymbol, priceData);
      
      if (triangularPath && triangularPath.netProfit > this.config.minProfitUSD) {
        opportunities.push(triangularPath);
      }
    }
    
    return opportunities;
  }
  
  /**
   * Find triangular arbitrage path
   */
  private findTriangularPath(
    base: string,
    intermediate: string,
    quote: string,
    priceData: ExchangePriceData[]
  ): TriangularArbitrage | null {
    
    // This is a simplified triangular arbitrage detection
    // In production, would need more sophisticated path finding
    
    const path = [base, intermediate, quote, base];
    const exchanges = priceData.map(p => p.exchange);
    const prices: number[] = [];
    const legs: any[] = [];
    
    // For now, return null as triangular arbitrage requires more complex implementation
    // Would need to analyze cross-currency pairs across multiple exchanges
    return null;
  }
  
  /**
   * Calculate optimal position size
   */
  private calculateOptimalSize(
    buyPrice: number,
    sellPrice: number,
    buyExchange: string,
    sellExchange: string,
    maxSize: number
  ): number {
    
    // Use Kelly criterion modified for arbitrage
    const spread = sellPrice - buyPrice;
    const spreadPercent = spread / buyPrice;
    
    // Conservative approach: use 50% of max size for high-confidence opportunities
    const conservativeFactor = 0.5;
    const riskAdjustedFactor = Math.min(1, spreadPercent * 10); // Higher spread = higher confidence
    
    return maxSize * conservativeFactor * riskAdjustedFactor;
  }
  
  /**
   * Calculate comprehensive risk score
   */
  private calculateRiskScore(factors: {
    spread: number;
    buyLatency: number;
    sellLatency: number;
    buyDepth: number;
    sellDepth: number;
    volume: number;
  }): number {
    let risk = 0;
    
    // Spread risk (lower spread = higher risk)
    if (factors.spread < 0.1) risk += 30;
    else if (factors.spread < 0.2) risk += 15;
    else if (factors.spread < 0.5) risk += 5;
    
    // Latency risk
    const maxLatency = Math.max(factors.buyLatency, factors.sellLatency);
    if (maxLatency > 500) risk += 25;
    else if (maxLatency > 200) risk += 15;
    else if (maxLatency > 100) risk += 5;
    
    // Liquidity risk
    const minDepth = Math.min(factors.buyDepth, factors.sellDepth);
    if (minDepth < 10000) risk += 20;
    else if (minDepth < 50000) risk += 10;
    else if (minDepth < 100000) risk += 5;
    
    // Volume risk
    if (factors.volume < 100000) risk += 15;
    else if (factors.volume < 500000) risk += 8;
    else if (factors.volume < 1000000) risk += 3;
    
    return Math.max(0, Math.min(100, risk));
  }
  
  /**
   * Calculate execution risk
   */
  private calculateExecutionRisk(buy: ExchangePriceData, sell: ExchangePriceData): number {
    // Risk that one leg fails to execute
    const latencyDiff = Math.abs(buy.latency - sell.latency);
    const spreadDiff = Math.abs(buy.spread - sell.spread);
    
    let risk = 0;
    
    if (latencyDiff > 100) risk += 20;
    if (spreadDiff > 0.001) risk += 15;
    
    return Math.max(0, Math.min(100, risk));
  }
  
  /**
   * Calculate latency risk
   */
  private calculateLatencyRisk(buyLatency: number, sellLatency: number): number {
    const totalLatency = buyLatency + sellLatency;
    const maxAcceptable = this.config.maxLatencyMs;
    
    if (totalLatency > maxAcceptable) {
      return Math.min(100, (totalLatency / maxAcceptable) * 50);
    }
    
    return (totalLatency / maxAcceptable) * 25;
  }
  
  /**
   * Calculate liquidity risk
   */
  private calculateLiquidityRisk(buyDepth: any, sellDepth: any): number {
    const minLiquidity = Math.min(buyDepth.askDepth, sellDepth.bidDepth);
    const targetLiquidity = 100000; // Target minimum liquidity
    
    if (minLiquidity < targetLiquidity) {
      return Math.min(100, ((targetLiquidity - minLiquidity) / targetLiquidity) * 80);
    }
    
    return 0;
  }
  
  /**
   * Analyze market conditions
   */
  private analyzeMarketConditions(symbol: string, exchanges: ExchangePriceData[]): any {
    const avgVolume = exchanges.reduce((sum, e) => sum + e.volume24h, 0) / exchanges.length;
    const avgSpread = exchanges.reduce((sum, e) => sum + e.spread, 0) / exchanges.length;
    
    return {
      volatility: avgSpread > 0.001 ? 'high' : avgSpread > 0.0005 ? 'medium' : 'low',
      volume: avgVolume,
      momentum: 'neutral', // Would analyze price trends
      sentiment: 'neutral'  // Would analyze market sentiment
    };
  }
  
  /**
   * Estimate how long an opportunity might last
   */
  private estimateOpportunityDuration(spread: number, riskScore: number): number {
    // Higher spreads typically last longer, but higher risk reduces duration
    const baseRisk = spread * 1000; // Base duration in ms
    const riskReduction = (riskScore / 100) * 0.5; // Risk reduces duration
    
    return Math.max(1000, baseRisk * (1 - riskReduction)); // At least 1 second
  }
  
  /**
   * Calculate confidence in opportunity
   */
  private calculateConfidence(spread: number, riskScore: number, marketConditions: any): number {
    let confidence = 0.5; // Base confidence
    
    // Spread boost
    confidence += Math.min(0.3, spread * 0.1);
    
    // Risk penalty
    confidence -= (riskScore / 100) * 0.4;
    
    // Market conditions adjustment
    if (marketConditions.volatility === 'low') confidence += 0.1;
    if (marketConditions.volume > 1000000) confidence += 0.1;
    
    return Math.max(0.1, Math.min(1, confidence));
  }
  
  /**
   * Get historical success rate for exchange pair
   */
  private getHistoricalSuccessRate(buyExchange: string, sellExchange: string): number {
    // In production, would query historical database
    // For now, return estimated success rates based on exchange reliability
    const exchangeReliability: Record<string, number> = {
      'binance': 0.95,
      'coinbase': 0.92,
      'kraken': 0.88,
      'okx': 0.90,
      'huobi': 0.85
    };
    
    const buyReliability = exchangeReliability[buyExchange] || 0.8;
    const sellReliability = exchangeReliability[sellExchange] || 0.8;
    
    return buyReliability * sellReliability;
  }
  
  /**
   * Estimate trading fee for exchange
   */
  private estimateTradingFee(exchange: string, notionalValue: number): number {
    const feeRates: Record<string, number> = {
      'binance': 0.001,    // 0.1%
      'coinbase': 0.005,   // 0.5%
      'kraken': 0.0026,    // 0.26%
      'okx': 0.001,        // 0.1%
      'huobi': 0.002       // 0.2%
    };
    
    const feeRate = feeRates[exchange] || 0.003; // Default 0.3%
    return notionalValue * feeRate;
  }
  
  /**
   * Check if opportunity is viable for execution
   */
  private isOpportunityViable(opportunity: ArbitrageOpportunity): boolean {
    // Basic viability checks
    if (opportunity.netProfit < this.config.minProfitUSD) return false;
    if (opportunity.riskScore > 80) return false; // Too risky
    if (opportunity.confidence < 0.3) return false; // Low confidence
    
    // Check if exchanges are enabled
    if (!this.config.enabledExchanges.includes(opportunity.buyExchange)) return false;
    if (!this.config.enabledExchanges.includes(opportunity.sellExchange)) return false;
    
    // Check if symbol is enabled
    if (!this.config.enabledSymbols.includes(opportunity.symbol)) return false;
    
    // Check blacklist
    const pairKey = `${opportunity.buyExchange}-${opportunity.sellExchange}`;
    if (this.config.blacklistPairs.includes(pairKey)) return false;
    
    return true;
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ArbitrageConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('🔧 Arbitrage detector configuration updated');
  }
  
  /**
   * Get detection statistics
   */
  getDetectionStats(): any {
    const recent = this.detectionHistory.filter(h => Date.now() - h.timestamp < 3600000); // Last hour
    
    return {
      totalDetected: this.detectionHistory.length,
      recentDetected: recent.length,
      avgSpread: recent.length > 0
        ? recent.reduce((sum, h) => sum + h.opportunity.spreadPercent, 0) / recent.length
        : 0,
      avgProfit: recent.length > 0
        ? recent.reduce((sum, h) => sum + h.opportunity.netProfit, 0) / recent.length
        : 0
    };
  }
}

/**
 * Main Multi-Exchange Arbitrage Engine
 * The digital predator that hunts profit in market inefficiencies
 */
export class MultiExchangeArbitrageEngine extends EventEmitter {
  private priceMonitor: PriceMonitor;
  private opportunityDetector: OpportunityDetector;
  private config: ArbitrageConfig;
  private activeOpportunities: Map<string, ArbitrageOpportunity> = new Map();
  private executionQueue: ArbitrageOpportunity[] = [];
  private metrics: ArbitrageMetrics;
  private isActive = false;
  private processingInterval: NodeJS.Timeout | null = null;
  
  constructor(config?: Partial<ArbitrageConfig>) {
    super();
    
    // Default configuration
    this.config = {
      minSpreadPercent: 0.1,          // 0.1% minimum spread
      maxSpreadPercent: 5.0,          // 5% maximum spread
      minProfitUSD: 1,                // $1 minimum profit
      maxPositionSize: 10000,         // $10k maximum position
      
      maxConcurrentArbs: 5,           // 5 concurrent arbitrages max
      maxExposurePercent: 20,         // 20% max portfolio exposure
      stopLossPercent: 0.5,           // 0.5% stop loss
      
      maxLatencyMs: 500,              // 500ms max latency
      slippageBuffer: 0.05,           // 0.05% extra slippage buffer
      executionTimeoutMs: 10000,      // 10 second timeout
      
      enabledExchanges: ['binance', 'coinbase', 'kraken', 'okx'],
      enabledSymbols: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT'],
      blacklistPairs: [],
      
      enableSimpleArbitrage: true,
      enableTriangularArbitrage: false, // More complex, disabled by default
      enableCrossCurrencyArbitrage: false,
      enableFundingRateArbitrage: false,
      
      ...config
    };
    
    this.priceMonitor = new PriceMonitor();
    this.opportunityDetector = new OpportunityDetector(this.config);
    
    this.metrics = {
      totalOpportunities: 0,
      successfulTrades: 0,
      failedTrades: 0,
      totalProfit: 0,
      totalVolume: 0,
      avgProfitPercent: 0,
      avgExecutionTime: 0,
      successRate: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      avgSpread: 0,
      medianSpread: 0,
      largestSpread: 0,
      exchangePairStats: new Map(),
      hourlyStats: [],
      lastUpdated: Date.now()
    };
    
    this.initializeEventHandlers();
    
    console.log('🦅 Multi-Exchange Arbitrage Engine initializing - The digital predator awakens');
  }
  
  /**
   * Initialize event handlers
   */
  private initializeEventHandlers(): void {
    // Price update handler
    this.priceMonitor.on('priceUpdate', (priceData: ExchangePriceData) => {
      this.handlePriceUpdate(priceData);
    });
    
    // Execution pipeline handlers
    executionPipeline.on('orderExecuted', (data) => {
      this.handleOrderExecuted(data);
    });
    
    executionPipeline.on('orderFailed', (data) => {
      this.handleOrderFailed(data);
    });
  }
  
  /**
   * Start the arbitrage engine
   */
  async start(): Promise<void> {
    if (this.isActive) {
      console.warn('⚠️ Arbitrage engine already active');
      return;
    }
    
    try {
      console.log('🚀 Starting arbitrage engine - Jacking into exchange feeds');
      
      // Subscribe to price feeds for enabled symbols
      for (const symbol of this.config.enabledSymbols) {
        this.priceMonitor.subscribeSymbol(symbol, this.config.enabledExchanges);
      }
      
      // Start opportunity processing
      this.startOpportunityProcessing();
      
      this.isActive = true;
      
      console.log('✅ Arbitrage engine online - Hunting for profit in the digital wasteland');
      this.emit('started');
      
    } catch (error) {
      console.error('🚨 Failed to start arbitrage engine:', error);
      throw error;
    }
  }
  
  /**
   * Stop the arbitrage engine
   */
  async stop(): Promise<void> {
    console.log('🛑 Stopping arbitrage engine');
    
    this.isActive = false;
    
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
    
    // Cancel active opportunities
    for (const [id, opportunity] of this.activeOpportunities) {
      if (opportunity.status === 'executing') {
        opportunity.status = 'failed';
        console.log(`❌ Cancelled active arbitrage: ${id}`);
      }
    }
    
    this.activeOpportunities.clear();
    this.executionQueue = [];
    
    console.log('💀 Arbitrage engine stopped');
    this.emit('stopped');
  }
  
  /**
   * Handle price updates from exchanges
   */
  private handlePriceUpdate(priceData: ExchangePriceData): void {
    if (!this.isActive) return;
    
    // Get all prices for this symbol
    const allPrices = this.priceMonitor.getPriceData(priceData.symbol);
    
    // Detect opportunities
    const opportunities = this.opportunityDetector.analyzePriceData(priceData.symbol, allPrices);
    
    // Process new opportunities
    for (const opportunity of opportunities) {
      this.processNewOpportunity(opportunity);
    }
  }
  
  /**
   * Process a newly detected opportunity
   */
  private processNewOpportunity(opportunity: ArbitrageOpportunity): void {
    // Check if we already have too many concurrent arbitrages
    const activeCount = Array.from(this.activeOpportunities.values())
      .filter(opp => opp.status === 'executing').length;
    
    if (activeCount >= this.config.maxConcurrentArbs) {
      console.log(`⏸️ Opportunity queued: ${opportunity.id} (too many active: ${activeCount})`);
      this.executionQueue.push(opportunity);
      return;
    }
    
    // Risk check
    const riskValidation = riskEngine.validateOrder(
      opportunity.symbol,
      'buy', // Simplified for arbitrage
      opportunity.optimalSize,
      opportunity.buyPrice,
      undefined,
      undefined,
      1
    );
    
    if (!riskValidation.isValid) {
      console.log(`❌ Opportunity rejected by risk engine: ${opportunity.id} - ${riskValidation.reason}`);
      return;
    }
    
    // Execute the opportunity
    this.executeArbitrageOpportunity(opportunity);
  }
  
  /**
   * Execute an arbitrage opportunity
   */
  private async executeArbitrageOpportunity(opportunity: ArbitrageOpportunity): Promise<void> {
    console.log(`⚡ Executing arbitrage: ${opportunity.id} - ${opportunity.spreadPercent.toFixed(3)}% spread, $${opportunity.netProfit.toFixed(2)} profit`);
    
    opportunity.status = 'executing';
    opportunity.executionStarted = Date.now();
    this.activeOpportunities.set(opportunity.id, opportunity);
    
    try {
      // Create buy order
      const buyOrderId = await executionPipeline.submitOrder({
        exchange: opportunity.buyExchange,
        symbol: opportunity.symbol,
        side: 'buy',
        type: 'market',
        amount: opportunity.optimalSize,
        urgency: 'critical',
        source: 'arbitrage',
        maxSlippage: opportunity.spreadPercent * 0.5, // Use half the spread as max slippage
        maxLatency: this.config.maxLatencyMs / 2
      });
      
      // Create sell order
      const sellOrderId = await executionPipeline.submitOrder({
        exchange: opportunity.sellExchange,
        symbol: opportunity.symbol,
        side: 'sell',
        type: 'market',
        amount: opportunity.optimalSize,
        urgency: 'critical',
        source: 'arbitrage',
        maxSlippage: opportunity.spreadPercent * 0.5,
        maxLatency: this.config.maxLatencyMs / 2
      });
      
      console.log(`📈 Arbitrage orders submitted: buy ${buyOrderId}, sell ${sellOrderId}`);
      
      this.metrics.totalOpportunities++;
      this.emit('opportunityExecuted', opportunity);
      
    } catch (error) {
      console.error(`🚨 Arbitrage execution failed: ${opportunity.id} - ${error.message}`);
      opportunity.status = 'failed';
      this.metrics.failedTrades++;
      this.emit('opportunityFailed', { opportunity, error: error.message });
    }
  }
  
  /**
   * Start opportunity processing loop
   */
  private startOpportunityProcessing(): void {
    this.processingInterval = setInterval(() => {
      this.processExecutionQueue();
      this.cleanupExpiredOpportunities();
      this.updateMetrics();
    }, 100); // Process every 100ms
  }
  
  /**
   * Process queued opportunities
   */
  private processExecutionQueue(): void {
    if (this.executionQueue.length === 0) return;
    
    const activeCount = Array.from(this.activeOpportunities.values())
      .filter(opp => opp.status === 'executing').length;
    
    const availableSlots = this.config.maxConcurrentArbs - activeCount;
    
    if (availableSlots > 0) {
      // Sort queue by profit potential
      this.executionQueue.sort((a, b) => b.netProfit - a.netProfit);
      
      // Execute opportunities up to available slots
      const toExecute = this.executionQueue.splice(0, availableSlots);
      
      for (const opportunity of toExecute) {
        // Check if opportunity is still valid
        if (Date.now() > opportunity.validUntil) {
          console.log(`⏰ Opportunity expired: ${opportunity.id}`);
          continue;
        }
        
        this.executeArbitrageOpportunity(opportunity);
      }
    }
  }
  
  /**
   * Clean up expired opportunities
   */
  private cleanupExpiredOpportunities(): void {
    const now = Date.now();
    
    for (const [id, opportunity] of this.activeOpportunities) {
      if (opportunity.status === 'executing' && now > opportunity.validUntil) {
        console.log(`⏰ Active opportunity expired: ${id}`);
        opportunity.status = 'expired';
        this.activeOpportunities.delete(id);
      }
    }
    
    // Clean up execution queue
    this.executionQueue = this.executionQueue.filter(opp => now <= opp.validUntil);
  }
  
  /**
   * Handle successful order execution
   */
  private handleOrderExecuted(data: any): void {
    const { order } = data;
    
    if (order.source !== 'arbitrage') return;
    
    // Find the corresponding arbitrage opportunity
    for (const [id, opportunity] of this.activeOpportunities) {
      if (opportunity.status === 'executing') {
        // Simplified: assume order belongs to this opportunity
        // In production, would need better order-to-opportunity mapping
        
        opportunity.status = 'completed';
        opportunity.executionCompleted = Date.now();
        opportunity.actualProfit = opportunity.netProfit; // Simplified calculation
        
        this.metrics.successfulTrades++;
        this.metrics.totalProfit += opportunity.actualProfit;
        this.metrics.totalVolume += opportunity.optimalSize * opportunity.buyPrice;
        
        console.log(`✅ Arbitrage completed: ${id} - Profit: $${opportunity.actualProfit.toFixed(2)}`);
        
        this.emit('opportunityCompleted', opportunity);
        this.activeOpportunities.delete(id);
        break;
      }
    }
  }
  
  /**
   * Handle failed order execution
   */
  private handleOrderFailed(data: any): void {
    const { order, error } = data;
    
    if (order.source !== 'arbitrage') return;
    
    // Find and mark corresponding opportunity as failed
    for (const [id, opportunity] of this.activeOpportunities) {
      if (opportunity.status === 'executing') {
        opportunity.status = 'failed';
        this.metrics.failedTrades++;
        
        console.error(`❌ Arbitrage failed: ${id} - ${error}`);
        
        this.emit('opportunityFailed', { opportunity, error });
        this.activeOpportunities.delete(id);
        break;
      }
    }
  }
  
  /**
   * Update engine metrics
   */
  private updateMetrics(): void {
    const totalTrades = this.metrics.successfulTrades + this.metrics.failedTrades;
    
    this.metrics.successRate = totalTrades > 0 ? (this.metrics.successfulTrades / totalTrades) * 100 : 0;
    
    this.metrics.avgProfitPercent = this.metrics.successfulTrades > 0
      ? (this.metrics.totalProfit / this.metrics.totalVolume) * 100
      : 0;
    
    // Calculate execution time for active opportunities
    const activeOpportunities = Array.from(this.activeOpportunities.values())
      .filter(opp => opp.executionStarted);
    
    if (activeOpportunities.length > 0) {
      const totalExecutionTime = activeOpportunities.reduce((sum, opp) => {
        const endTime = opp.executionCompleted || Date.now();
        return sum + (endTime - opp.executionStarted!);
      }, 0);
      
      this.metrics.avgExecutionTime = totalExecutionTime / activeOpportunities.length;
    }
    
    this.metrics.lastUpdated = Date.now();
  }
  
  /**
   * Get current metrics
   */
  getMetrics(): ArbitrageMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }
  
  /**
   * Get active opportunities
   */
  getActiveOpportunities(): ArbitrageOpportunity[] {
    return Array.from(this.activeOpportunities.values());
  }
  
  /**
   * Get opportunity queue
   */
  getOpportunityQueue(): ArbitrageOpportunity[] {
    return [...this.executionQueue];
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ArbitrageConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.opportunityDetector.updateConfig(this.config);
    
    console.log('🔧 Arbitrage engine configuration updated');
    this.emit('configUpdated', this.config);
  }
  
  /**
   * Manual opportunity execution (for testing)
   */
  async executeManualOpportunity(
    symbol: string,
    buyExchange: string,
    sellExchange: string,
    amount: number
  ): Promise<string> {
    
    const priceData = this.priceMonitor.getPriceData(symbol);
    const buyData = priceData.find(p => p.exchange === buyExchange);
    const sellData = priceData.find(p => p.exchange === sellExchange);
    
    if (!buyData || !sellData) {
      throw new Error('Price data not available for specified exchanges');
    }
    
    const spread = sellData.bid - buyData.ask;
    const spreadPercent = (spread / buyData.ask) * 100;
    
    const manualOpportunity: ArbitrageOpportunity = {
      id: `manual_${Date.now()}`,
      type: 'simple',
      symbol,
      buyExchange,
      sellExchange,
      buyPrice: buyData.ask,
      sellPrice: sellData.bid,
      spreadPercent,
      spreadAbsolute: spread,
      grossProfit: spread * amount,
      netProfit: (spread * amount) - (amount * buyData.ask * 0.002), // Estimate fees
      profitPercent: spreadPercent,
      roi: spreadPercent,
      maxSize: amount,
      optimalSize: amount,
      requiredCapital: amount * buyData.ask,
      riskScore: 50, // Medium risk for manual
      executionRisk: 25,
      latencyRisk: 25,
      liquidityRisk: 25,
      detectedAt: Date.now(),
      validUntil: Date.now() + 30000, // 30 second validity
      estimatedDuration: 30000,
      status: 'detected',
      confidence: 0.7,
      historicalSuccess: 0.8,
      marketConditions: {
        volatility: 0.5,
        volume: Math.min(buyData.volume24h, sellData.volume24h),
        momentum: 'neutral',
        sentiment: 'neutral'
      }
    };
    
    await this.executeArbitrageOpportunity(manualOpportunity);
    return manualOpportunity.id;
  }
  
  /**
   * Emergency stop all arbitrage activities
   */
  emergencyStop(): void {
    console.error('🚨 EMERGENCY STOP - Halting all arbitrage activities');
    
    this.isActive = false;
    
    // Clear execution queue
    const queuedCount = this.executionQueue.length;
    this.executionQueue = [];
    
    // Mark all active opportunities as failed
    let activeCount = 0;
    for (const [id, opportunity] of this.activeOpportunities) {
      if (opportunity.status === 'executing') {
        opportunity.status = 'failed';
        activeCount++;
      }
    }
    
    console.error(`🛑 Emergency stop complete: ${queuedCount} queued opportunities cleared, ${activeCount} active opportunities stopped`);
    
    this.emit('emergencyStop', {
      queuedOpportunitiesCleared: queuedCount,
      activeOpportunitiesStopped: activeCount
    });
  }
  
  /**
   * Shutdown the engine
   */
  shutdown(): void {
    this.stop();
    this.priceMonitor.shutdown();
    
    console.log('💀 Multi-Exchange Arbitrage Engine shutdown complete');
  }
}

// Export singleton instance
export const arbitrageEngine = new MultiExchangeArbitrageEngine();

// Utility functions for React components
export const arbitrageUtils = {
  /**
   * Format spread percentage with color
   */
  formatSpread(spread: number): { text: string; color: string } {
    const text = `${spread.toFixed(3)}%`;
    let color = 'text-gray-400';
    
    if (spread >= 1.0) color = 'text-green-400';
    else if (spread >= 0.5) color = 'text-yellow-400';
    else if (spread >= 0.2) color = 'text-orange-400';
    
    return { text, color };
  },
  
  /**
   * Get opportunity urgency level
   */
  getOpportunityUrgency(opportunity: ArbitrageOpportunity): 'low' | 'medium' | 'high' | 'critical' {
    const timeLeft = opportunity.validUntil - Date.now();
    const spread = opportunity.spreadPercent;
    
    if (timeLeft < 5000 && spread > 0.5) return 'critical';
    if (timeLeft < 10000 || spread > 1.0) return 'high';
    if (spread > 0.3) return 'medium';
    return 'low';
  },
  
  /**
   * Calculate opportunity score for ranking
   */
  calculateOpportunityScore(opportunity: ArbitrageOpportunity): number {
    return (
      opportunity.netProfit * 0.4 +
      opportunity.confidence * 100 * 0.3 +
      (100 - opportunity.riskScore) * 0.3
    );
  },
  
  /**
   * Format profit with emoji indicators
   */
  formatProfit(profit: number): string {
    if (profit >= 100) return `💰 $${profit.toFixed(2)}`;
    if (profit >= 50) return `💵 $${profit.toFixed(2)}`;
    if (profit >= 10) return `💸 $${profit.toFixed(2)}`;
    return `💱 $${profit.toFixed(2)}`;
  }
};
