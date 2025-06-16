// frontend/src/lib/execution/OrderExecutionPipeline.ts
/**
 * Order Execution Pipeline - Nanosecond Chrome
 * Lightning-fast order routing and execution across multiple exchanges
 * 
 * This is where dreams become reality, hermano. Where those beautiful signals
 * from our stolen neural nets get transformed into cold, hard eddies.
 * Been through enough blown executions to know - speed isn't everything,
 * it's the ONLY thing that matters when the algos are hunting.
 * 
 * Every line here is optimized for nanosecond execution. Every function
 * call measured, every memory allocation considered. Because in the matrix
 * of high-frequency trading, lag kills more dreams than corpo betrayal.
 * 
 * Esta no es solo código - it's digital survival instinct refined into
 * pure profit potential.
 */

import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';
import { riskEngine } from '../risk/RiskManagementEngine';
import { nexlifyCache } from '../cache/IndexedDBCache';

// Types - the data structures that define execution reality
interface Order {
  id: string;
  clientOrderId: string;
  exchange: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit' | 'iceberg' | 'twap';
  amount: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'GTD';
  postOnly?: boolean;
  reduceOnly?: boolean;
  
  // Execution parameters
  maxSlippage: number;          // Maximum acceptable slippage %
  partialFillAcceptable: boolean;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  
  // Risk parameters
  maxLatency: number;           // Max execution time in ms
  failoverExchanges: string[];  // Alternative exchanges
  
  // Status tracking
  status: 'pending' | 'routing' | 'submitted' | 'partial' | 'filled' | 'cancelled' | 'rejected' | 'failed';
  filledAmount: number;
  avgFillPrice: number;
  fees: number;
  
  // Timing data
  createdAt: number;
  submittedAt?: number;
  firstFillAt?: number;
  completedAt?: number;
  
  // Metadata
  source: 'signal' | 'manual' | 'arbitrage' | 'hedge';
  strategyId?: string;
  signalId?: string;
  parentOrderId?: string;
  
  // Risk override flags
  riskCheckPassed: boolean;
  emergencyOverride: boolean;
}

interface ExecutionReport {
  orderId: string;
  fillId: string;
  symbol: string;
  side: 'buy' | 'sell';
  amount: number;
  price: number;
  fee: number;
  feeAsset: string;
  timestamp: number;
  exchange: string;
  tradeId?: string;
  liquidity: 'maker' | 'taker';
  executionLatency: number;
}

interface ExchangeConfig {
  id: string;
  name: string;
  isActive: boolean;
  priority: number;              // 1 = highest priority
  maxOrderSize: number;
  minOrderSize: number;
  makerFee: number;
  takerFee: number;
  avgLatency: number;           // Historical average latency
  reliability: number;          // 0-1 reliability score
  supportedOrderTypes: string[];
  rateLimits: {
    ordersPerSecond: number;
    ordersPerMinute: number;
    requestsPerSecond: number;
  };
  
  // Connection details
  apiEndpoint: string;
  wsEndpoint: string;
  isConnected: boolean;
  lastPing: number;
  
  // Performance metrics
  successRate: number;
  avgFillTime: number;
  slippageHistory: number[];
}

interface RoutingDecision {
  exchange: string;
  reason: string;
  expectedLatency: number;
  expectedFees: number;
  expectedSlippage: number;
  confidence: number;           // 0-1 confidence in routing decision
  alternativeRoutes: Array<{
    exchange: string;
    score: number;
    reason: string;
  }>;
}

interface ExecutionStrategy {
  id: string;
  name: string;
  description: string;
  isActive: boolean;
  applicableOrderTypes: string[];
  minOrderSize: number;
  maxOrderSize: number;
  
  // Strategy parameters
  maxSliceSize: number;         // For TWAP/Iceberg orders
  sliceInterval: number;        // Time between slices (ms)
  adaptiveSlicing: boolean;     // Adjust slice size based on market conditions
  liquidityThreshold: number;   // Min liquidity requirement
  
  // Performance thresholds
  maxExecutionTime: number;     // Max time to complete order
  maxSlippage: number;          // Max acceptable slippage
  urgencyMultiplier: number;    // Speed vs cost trade-off
}

interface PipelineMetrics {
  totalOrders: number;
  successfulOrders: number;
  failedOrders: number;
  cancelledOrders: number;
  avgExecutionTime: number;
  avgSlippage: number;
  totalVolume: number;
  totalFees: number;
  
  // Real-time metrics
  ordersPerSecond: number;
  currentQueueSize: number;
  activeConnections: number;
  
  // Performance breakdown by exchange
  exchangeMetrics: Map<string, {
    orderCount: number;
    successRate: number;
    avgLatency: number;
    avgSlippage: number;
    totalVolume: number;
  }>;
  
  lastUpdated: number;
}

// Smart Order Router - finds the optimal execution path
class SmartOrderRouter {
  private exchangeConfigs: Map<string, ExchangeConfig> = new Map();
  private liquidityCache: Map<string, any> = new Map();
  private routingHistory: Array<{ order: Order; routing: RoutingDecision; outcome: string }> = [];
  
  constructor() {
    this.initializeExchangeConfigs();
  }
  
  /**
   * Initialize exchange configurations
   */
  private initializeExchangeConfigs(): void {
    const exchanges: ExchangeConfig[] = [
      {
        id: 'binance',
        name: 'Binance',
        isActive: true,
        priority: 1,
        maxOrderSize: 1000000,
        minOrderSize: 1,
        makerFee: 0.001,
        takerFee: 0.001,
        avgLatency: 50,
        reliability: 0.98,
        supportedOrderTypes: ['market', 'limit', 'stop', 'stop_limit', 'iceberg'],
        rateLimits: {
          ordersPerSecond: 10,
          ordersPerMinute: 1200,
          requestsPerSecond: 20
        },
        apiEndpoint: process.env.BINANCE_API_URL || 'https://api.binance.com',
        wsEndpoint: process.env.BINANCE_WS_URL || 'wss://stream.binance.com',
        isConnected: false,
        lastPing: 0,
        successRate: 0.97,
        avgFillTime: 150,
        slippageHistory: []
      },
      {
        id: 'coinbase',
        name: 'Coinbase Pro',
        isActive: true,
        priority: 2,
        maxOrderSize: 500000,
        minOrderSize: 1,
        makerFee: 0.0025,
        takerFee: 0.0035,
        avgLatency: 80,
        reliability: 0.95,
        supportedOrderTypes: ['market', 'limit', 'stop'],
        rateLimits: {
          ordersPerSecond: 5,
          ordersPerMinute: 300,
          requestsPerSecond: 10
        },
        apiEndpoint: process.env.COINBASE_API_URL || 'https://api.exchange.coinbase.com',
        wsEndpoint: process.env.COINBASE_WS_URL || 'wss://ws-feed.exchange.coinbase.com',
        isConnected: false,
        lastPing: 0,
        successRate: 0.94,
        avgFillTime: 200,
        slippageHistory: []
      },
      {
        id: 'kraken',
        name: 'Kraken',
        isActive: true,
        priority: 3,
        maxOrderSize: 250000,
        minOrderSize: 1,
        makerFee: 0.0016,
        takerFee: 0.0026,
        avgLatency: 120,
        reliability: 0.92,
        supportedOrderTypes: ['market', 'limit', 'stop'],
        rateLimits: {
          ordersPerSecond: 3,
          ordersPerMinute: 180,
          requestsPerSecond: 5
        },
        apiEndpoint: process.env.KRAKEN_API_URL || 'https://api.kraken.com',
        wsEndpoint: process.env.KRAKEN_WS_URL || 'wss://ws.kraken.com',
        isConnected: false,
        lastPing: 0,
        successRate: 0.91,
        avgFillTime: 300,
        slippageHistory: []
      }
    ];
    
    for (const config of exchanges) {
      this.exchangeConfigs.set(config.id, config);
    }
    
    console.log(`🎯 Smart router initialized with ${exchanges.length} exchanges`);
  }
  
  /**
   * Route order to optimal exchange based on multiple factors
   */
  async routeOrder(order: Order): Promise<RoutingDecision> {
    const startTime = performance.now();
    
    try {
      // Get available exchanges for this order
      const availableExchanges = this.getAvailableExchanges(order);
      
      if (availableExchanges.length === 0) {
        throw new Error('No exchanges available for order execution');
      }
      
      // Score each exchange
      const exchangeScores = await Promise.all(
        availableExchanges.map(exchange => this.scoreExchange(exchange, order))
      );
      
      // Sort by score (highest first)
      exchangeScores.sort((a, b) => b.score - a.score);
      
      const bestExchange = exchangeScores[0];
      const alternatives = exchangeScores.slice(1).map(score => ({
        exchange: score.exchange.id,
        score: score.score,
        reason: score.reason
      }));
      
      const decision: RoutingDecision = {
        exchange: bestExchange.exchange.id,
        reason: bestExchange.reason,
        expectedLatency: bestExchange.expectedLatency,
        expectedFees: bestExchange.expectedFees,
        expectedSlippage: bestExchange.expectedSlippage,
        confidence: bestExchange.score,
        alternativeRoutes: alternatives
      };
      
      // Cache the routing decision
      this.routingHistory.push({
        order,
        routing: decision,
        outcome: 'pending'
      });
      
      // Keep history manageable
      if (this.routingHistory.length > 1000) {
        this.routingHistory = this.routingHistory.slice(-500);
      }
      
      const routingTime = performance.now() - startTime;
      console.log(`🎯 Order routed to ${decision.exchange} in ${routingTime.toFixed(2)}ms (confidence: ${(decision.confidence * 100).toFixed(1)}%)`);
      
      return decision;
      
    } catch (error) {
      console.error('🚨 Order routing failed:', error);
      throw error;
    }
  }
  
  /**
   * Get exchanges available for a specific order
   */
  private getAvailableExchanges(order: Order): ExchangeConfig[] {
    return Array.from(this.exchangeConfigs.values()).filter(exchange => {
      // Basic availability checks
      if (!exchange.isActive || !exchange.isConnected) return false;
      
      // Size limits
      if (order.amount < exchange.minOrderSize || order.amount > exchange.maxOrderSize) return false;
      
      // Order type support
      if (!exchange.supportedOrderTypes.includes(order.type)) return false;
      
      // Rate limit check (simplified)
      const now = Date.now();
      if (now - exchange.lastPing < 100) return false; // Too recent
      
      return true;
    });
  }
  
  /**
   * Score an exchange for order execution
   */
  private async scoreExchange(exchange: ExchangeConfig, order: Order): Promise<{
    exchange: ExchangeConfig;
    score: number;
    reason: string;
    expectedLatency: number;
    expectedFees: number;
    expectedSlippage: number;
  }> {
    let score = 0;
    const reasons = [];
    
    // Reliability score (40% weight)
    const reliabilityScore = exchange.reliability * 40;
    score += reliabilityScore;
    reasons.push(`reliability: ${(exchange.reliability * 100).toFixed(1)}%`);
    
    // Latency score (25% weight)
    const maxAcceptableLatency = order.maxLatency || 1000;
    const latencyScore = Math.max(0, (maxAcceptableLatency - exchange.avgLatency) / maxAcceptableLatency) * 25;
    score += latencyScore;
    reasons.push(`latency: ${exchange.avgLatency}ms`);
    
    // Fee score (20% weight)
    const estimatedFee = order.type === 'market' ? exchange.takerFee : exchange.makerFee;
    const feeScore = (1 - estimatedFee) * 20;
    score += feeScore;
    reasons.push(`fees: ${(estimatedFee * 100).toFixed(3)}%`);
    
    // Liquidity score (10% weight)
    const liquidityScore = await this.getLiquidityScore(exchange.id, order.symbol, order.amount);
    score += liquidityScore * 10;
    reasons.push(`liquidity: ${(liquidityScore * 100).toFixed(1)}%`);
    
    // Priority boost (5% weight)
    const priorityScore = (10 - exchange.priority) / 10 * 5;
    score += priorityScore;
    
    // Urgency adjustments
    if (order.urgency === 'critical') {
      // Heavily favor low latency for critical orders
      score = score * 0.7 + latencyScore * 0.3;
      reasons.push('critical urgency boost');
    }
    
    const expectedLatency = exchange.avgLatency;
    const expectedFees = estimatedFee * order.amount * (order.price || 0);
    const expectedSlippage = await this.estimateSlippage(exchange.id, order);
    
    return {
      exchange,
      score: Math.max(0, Math.min(100, score)) / 100, // Normalize to 0-1
      reason: reasons.join(', '),
      expectedLatency,
      expectedFees,
      expectedSlippage
    };
  }
  
  /**
   * Get liquidity score for symbol on exchange
   */
  private async getLiquidityScore(exchangeId: string, symbol: string, amount: number): Promise<number> {
    // Check cache first
    const cacheKey = `liquidity:${exchangeId}:${symbol}`;
    const cached = this.liquidityCache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < 10000) { // 10 second cache
      return cached.score;
    }
    
    // In production, this would query real order book data
    // For now, estimate based on exchange priority and typical volumes
    let score = 0.5; // Base score
    
    switch (exchangeId) {
      case 'binance':
        score = 0.9; // Highest liquidity
        break;
      case 'coinbase':
        score = 0.7; // Good liquidity
        break;
      case 'kraken':
        score = 0.6; // Moderate liquidity
        break;
    }
    
    // Adjust for order size
    if (amount > 100000) score *= 0.8; // Large orders get lower liquidity score
    if (amount < 1000) score *= 1.1;   // Small orders get higher score
    
    // Cache the result
    this.liquidityCache.set(cacheKey, {
      score: Math.max(0, Math.min(1, score)),
      timestamp: Date.now()
    });
    
    return score;
  }
  
  /**
   * Estimate slippage for order
   */
  private async estimateSlippage(exchangeId: string, order: Order): Promise<number> {
    const exchange = this.exchangeConfigs.get(exchangeId);
    if (!exchange) return 0.01; // 1% default
    
    // Base slippage from historical data
    const avgHistoricalSlippage = exchange.slippageHistory.length > 0
      ? exchange.slippageHistory.reduce((sum, s) => sum + s, 0) / exchange.slippageHistory.length
      : 0.001; // 0.1% default
    
    // Adjust for order type
    let estimatedSlippage = avgHistoricalSlippage;
    
    if (order.type === 'market') {
      estimatedSlippage *= 1.5; // Market orders have higher slippage
    }
    
    // Adjust for order size
    if (order.amount > 50000) {
      estimatedSlippage *= 1.3; // Large orders have higher slippage
    }
    
    // Adjust for urgency
    if (order.urgency === 'critical') {
      estimatedSlippage *= 1.2; // Urgent orders accept higher slippage
    }
    
    return Math.max(0.0001, estimatedSlippage); // Minimum 0.01% slippage
  }
  
  /**
   * Update exchange performance metrics
   */
  updateExchangeMetrics(exchangeId: string, latency: number, slippage: number, success: boolean): void {
    const exchange = this.exchangeConfigs.get(exchangeId);
    if (!exchange) return;
    
    // Update latency (exponential moving average)
    exchange.avgLatency = exchange.avgLatency * 0.9 + latency * 0.1;
    
    // Update slippage history
    exchange.slippageHistory.push(slippage);
    if (exchange.slippageHistory.length > 100) {
      exchange.slippageHistory.shift();
    }
    
    // Update success rate
    const currentRate = exchange.successRate;
    exchange.successRate = success
      ? currentRate * 0.99 + 0.01
      : currentRate * 0.99;
    
    console.log(`📊 Exchange metrics updated: ${exchangeId} - latency: ${latency.toFixed(1)}ms, slippage: ${(slippage * 100).toFixed(3)}%`);
  }
  
  /**
   * Get exchange configuration
   */
  getExchangeConfig(exchangeId: string): ExchangeConfig | undefined {
    return this.exchangeConfigs.get(exchangeId);
  }
  
  /**
   * Update exchange connection status
   */
  updateConnectionStatus(exchangeId: string, isConnected: boolean): void {
    const exchange = this.exchangeConfigs.get(exchangeId);
    if (exchange) {
      exchange.isConnected = isConnected;
      exchange.lastPing = Date.now();
    }
  }
}

// Execution Engine - handles the actual order submission and monitoring
class ExecutionEngine extends EventEmitter {
  private activeOrders: Map<string, Order> = new Map();
  private executionReports: Array<ExecutionReport> = [];
  private strategies: Map<string, ExecutionStrategy> = new Map();
  
  constructor() {
    super();
    this.initializeStrategies();
  }
  
  /**
   * Initialize execution strategies
   */
  private initializeStrategies(): void {
    const strategies: ExecutionStrategy[] = [
      {
        id: 'aggressive',
        name: 'Aggressive Execution',
        description: 'Fast execution prioritizing speed over cost',
        isActive: true,
        applicableOrderTypes: ['market'],
        minOrderSize: 1,
        maxOrderSize: 100000,
        maxSliceSize: 10000,
        sliceInterval: 100,
        adaptiveSlicing: false,
        liquidityThreshold: 0.1,
        maxExecutionTime: 1000,
        maxSlippage: 0.005,
        urgencyMultiplier: 2.0
      },
      {
        id: 'balanced',
        name: 'Balanced Execution',
        description: 'Optimal balance between speed and cost',
        isActive: true,
        applicableOrderTypes: ['market', 'limit'],
        minOrderSize: 1,
        maxOrderSize: 500000,
        maxSliceSize: 25000,
        sliceInterval: 500,
        adaptiveSlicing: true,
        liquidityThreshold: 0.3,
        maxExecutionTime: 5000,
        maxSlippage: 0.003,
        urgencyMultiplier: 1.0
      },
      {
        id: 'stealth',
        name: 'Stealth Execution',
        description: 'Minimize market impact for large orders',
        isActive: true,
        applicableOrderTypes: ['limit', 'iceberg', 'twap'],
        minOrderSize: 10000,
        maxOrderSize: 10000000,
        maxSliceSize: 5000,
        sliceInterval: 2000,
        adaptiveSlicing: true,
        liquidityThreshold: 0.5,
        maxExecutionTime: 30000,
        maxSlippage: 0.001,
        urgencyMultiplier: 0.5
      }
    ];
    
    for (const strategy of strategies) {
      this.strategies.set(strategy.id, strategy);
    }
    
    console.log(`⚡ Execution engine initialized with ${strategies.length} strategies`);
  }
  
  /**
   * Execute order using optimal strategy
   */
  async executeOrder(order: Order, routing: RoutingDecision): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Update order status
      order.status = 'routing';
      order.submittedAt = Date.now();
      this.activeOrders.set(order.id, order);
      
      console.log(`⚡ Executing order ${order.id}: ${order.side} ${order.amount} ${order.symbol} on ${routing.exchange}`);
      
      // Select execution strategy
      const strategy = this.selectExecutionStrategy(order);
      
      // Execute based on order type and strategy
      switch (order.type) {
        case 'market':
          await this.executeMarketOrder(order, routing, strategy);
          break;
        case 'limit':
          await this.executeLimitOrder(order, routing, strategy);
          break;
        case 'iceberg':
          await this.executeIcebergOrder(order, routing, strategy);
          break;
        case 'twap':
          await this.executeTWAPOrder(order, routing, strategy);
          break;
        default:
          throw new Error(`Unsupported order type: ${order.type}`);
      }
      
      const executionTime = performance.now() - startTime;
      console.log(`✅ Order execution initiated in ${executionTime.toFixed(2)}ms`);
      
      this.emit('orderExecuted', { order, routing, executionTime });
      
    } catch (error) {
      console.error(`🚨 Order execution failed: ${error.message}`);
      order.status = 'failed';
      this.emit('orderFailed', { order, error: error.message });
      throw error;
    }
  }
  
  /**
   * Select optimal execution strategy for order
   */
  private selectExecutionStrategy(order: Order): ExecutionStrategy {
    const applicableStrategies = Array.from(this.strategies.values()).filter(strategy => {
      return strategy.isActive &&
             strategy.applicableOrderTypes.includes(order.type) &&
             order.amount >= strategy.minOrderSize &&
             order.amount <= strategy.maxOrderSize;
    });
    
    if (applicableStrategies.length === 0) {
      throw new Error('No applicable execution strategy found');
    }
    
    // Select based on order urgency
    switch (order.urgency) {
      case 'critical':
        return applicableStrategies.find(s => s.id === 'aggressive') || applicableStrategies[0];
      case 'high':
        return applicableStrategies.find(s => s.id === 'balanced') || applicableStrategies[0];
      case 'medium':
        return applicableStrategies.find(s => s.id === 'balanced') || applicableStrategies[0];
      case 'low':
        return applicableStrategies.find(s => s.id === 'stealth') || applicableStrategies[0];
      default:
        return applicableStrategies[0];
    }
  }
  
  /**
   * Execute market order
   */
  private async executeMarketOrder(order: Order, routing: RoutingDecision, strategy: ExecutionStrategy): Promise<void> {
    const startTime = performance.now();
    
    try {
      // For market orders, we submit immediately
      const exchangeOrder = await this.submitToExchange(order, routing.exchange);
      
      // Simulate immediate fill for market orders
      const fillPrice = order.price || 0; // Would get from exchange response
      const executionReport: ExecutionReport = {
        orderId: order.id,
        fillId: `fill_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        symbol: order.symbol,
        side: order.side,
        amount: order.amount,
        price: fillPrice,
        fee: this.calculateFee(order.amount, fillPrice, routing.exchange, 'taker'),
        feeAsset: 'USDT',
        timestamp: Date.now(),
        exchange: routing.exchange,
        liquidity: 'taker',
        executionLatency: performance.now() - startTime
      };
      
      this.processExecutionReport(executionReport);
      
    } catch (error) {
      console.error('Market order execution failed:', error);
      throw error;
    }
  }
  
  /**
   * Execute limit order
   */
  private async executeLimitOrder(order: Order, routing: RoutingDecision, strategy: ExecutionStrategy): Promise<void> {
    try {
      // Submit limit order to exchange
      const exchangeOrder = await this.submitToExchange(order, routing.exchange);
      
      // For limit orders, we wait for fills
      order.status = 'submitted';
      
      // Set up monitoring for partial fills
      this.monitorOrderFills(order, strategy);
      
    } catch (error) {
      console.error('Limit order execution failed:', error);
      throw error;
    }
  }
  
  /**
   * Execute iceberg order (hidden large order)
   */
  private async executeIcebergOrder(order: Order, routing: RoutingDecision, strategy: ExecutionStrategy): Promise<void> {
    const sliceSize = Math.min(strategy.maxSliceSize, order.amount / 10);
    const slices = Math.ceil(order.amount / sliceSize);
    
    console.log(`🧊 Executing iceberg order: ${slices} slices of ${sliceSize} each`);
    
    let remainingAmount = order.amount;
    let sliceIndex = 0;
    
    while (remainingAmount > 0 && sliceIndex < slices) {
      const currentSliceSize = Math.min(sliceSize, remainingAmount);
      
      // Create slice order
      const sliceOrder: Order = {
        ...order,
        id: `${order.id}_slice_${sliceIndex}`,
        amount: currentSliceSize,
        parentOrderId: order.id
      };
      
      try {
        await this.executeMarketOrder(sliceOrder, routing, strategy);
        remainingAmount -= currentSliceSize;
        sliceIndex++;
        
        // Wait between slices if there's more to execute
        if (remainingAmount > 0) {
          await this.sleep(strategy.sliceInterval);
        }
        
      } catch (error) {
        console.error(`Iceberg slice ${sliceIndex} failed:`, error);
        break;
      }
    }
    
    // Update parent order status
    order.filledAmount = order.amount - remainingAmount;
    order.status = remainingAmount === 0 ? 'filled' : 'partial';
  }
  
  /**
   * Execute TWAP order (Time-Weighted Average Price)
   */
  private async executeTWAPOrder(order: Order, routing: RoutingDecision, strategy: ExecutionStrategy): Promise<void> {
    const executionWindow = strategy.maxExecutionTime;
    const sliceInterval = strategy.sliceInterval;
    const numberOfSlices = Math.floor(executionWindow / sliceInterval);
    const sliceSize = order.amount / numberOfSlices;
    
    console.log(`⏰ Executing TWAP order: ${numberOfSlices} slices over ${executionWindow}ms`);
    
    let remainingAmount = order.amount;
    let totalFilled = 0;
    let weightedPriceSum = 0;
    
    for (let i = 0; i < numberOfSlices && remainingAmount > 0; i++) {
      const currentSliceSize = Math.min(sliceSize, remainingAmount);
      
      // Adapt slice size based on market conditions if enabled
      const adaptiveSize = strategy.adaptiveSlicing 
        ? await this.calculateAdaptiveSliceSize(currentSliceSize, order.symbol, routing.exchange)
        : currentSliceSize;
      
      const sliceOrder: Order = {
        ...order,
        id: `${order.id}_twap_${i}`,
        amount: adaptiveSize,
        type: 'market', // TWAP usually uses market orders for immediate execution
        parentOrderId: order.id
      };
      
      try {
        await this.executeMarketOrder(sliceOrder, routing, strategy);
        
        // Update TWAP calculation
        const fillPrice = order.price || 0; // Would get actual fill price
        totalFilled += adaptiveSize;
        weightedPriceSum += fillPrice * adaptiveSize;
        remainingAmount -= adaptiveSize;
        
        // Wait for next slice
        if (i < numberOfSlices - 1 && remainingAmount > 0) {
          await this.sleep(sliceInterval);
        }
        
      } catch (error) {
        console.error(`TWAP slice ${i} failed:`, error);
        break;
      }
    }
    
    // Update order with TWAP results
    order.filledAmount = totalFilled;
    order.avgFillPrice = totalFilled > 0 ? weightedPriceSum / totalFilled : 0;
    order.status = remainingAmount === 0 ? 'filled' : 'partial';
    
    console.log(`⏰ TWAP execution completed: filled ${totalFilled}/${order.amount} at avg price ${order.avgFillPrice}`);
  }
  
  /**
   * Calculate adaptive slice size based on market conditions
   */
  private async calculateAdaptiveSliceSize(baseSize: number, symbol: string, exchange: string): Promise<number> {
    // In production, this would analyze current market conditions
    // For now, add some randomization to avoid predictable patterns
    const randomFactor = 0.8 + Math.random() * 0.4; // 80% to 120% of base size
    const adaptiveSize = baseSize * randomFactor;
    
    return Math.max(1, adaptiveSize);
  }
  
  /**
   * Submit order to exchange
   */
  private async submitToExchange(order: Order, exchangeId: string): Promise<any> {
    const startTime = performance.now();
    
    try {
      // In production, this would make actual API calls to exchanges
      // For now, simulate the exchange submission
      
      const latency = 50 + Math.random() * 100; // Simulate network latency
      await this.sleep(latency);
      
      const exchangeOrderId = `${exchangeId}_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
      
      // Simulate success/failure based on exchange reliability
      const exchange = this.getExchangeConfig(exchangeId);
      const success = Math.random() < (exchange?.reliability || 0.95);
      
      if (!success) {
        throw new Error(`Exchange ${exchangeId} rejected order: simulated failure`);
      }
      
      const submissionLatency = performance.now() - startTime;
      
      console.log(`📤 Order submitted to ${exchangeId} in ${submissionLatency.toFixed(2)}ms - ID: ${exchangeOrderId}`);
      
      return {
        exchangeOrderId,
        status: 'submitted',
        timestamp: Date.now()
      };
      
    } catch (error) {
      console.error(`Exchange submission failed:`, error);
      throw error;
    }
  }
  
  /**
   * Process execution report from exchange
   */
  private processExecutionReport(report: ExecutionReport): void {
    this.executionReports.push(report);
    
    // Update order status
    const order = this.activeOrders.get(report.orderId);
    if (order) {
      order.filledAmount += report.amount;
      order.avgFillPrice = ((order.avgFillPrice * (order.filledAmount - report.amount)) + (report.price * report.amount)) / order.filledAmount;
      order.fees += report.fee;
      
      if (!order.firstFillAt) {
        order.firstFillAt = report.timestamp;
      }
      
      if (order.filledAmount >= order.amount) {
        order.status = 'filled';
        order.completedAt = report.timestamp;
        console.log(`✅ Order ${order.id} fully filled at avg price ${order.avgFillPrice.toFixed(6)}`);
      } else {
        order.status = 'partial';
      }
    }
    
    // Keep execution reports manageable
    if (this.executionReports.length > 10000) {
      this.executionReports = this.executionReports.slice(-5000);
    }
    
    this.emit('executionReport', report);
  }
  
  /**
   * Monitor order for fills
   */
  private monitorOrderFills(order: Order, strategy: ExecutionStrategy): void {
    const startTime = Date.now();
    const maxWaitTime = strategy.maxExecutionTime;
    
    const checkFills = setInterval(() => {
      const elapsed = Date.now() - startTime;
      
      // Check if order should timeout
      if (elapsed > maxWaitTime) {
        clearInterval(checkFills);
        
        if (order.status === 'submitted') {
          order.status = 'cancelled';
          console.log(`⏰ Order ${order.id} timed out after ${maxWaitTime}ms`);
          this.emit('orderTimeout', order);
        }
        return;
      }
      
      // In production, would check actual order status from exchange
      // For now, simulate random fills
      if (Math.random() < 0.1) { // 10% chance per check
        const fillAmount = Math.min(order.amount - order.filledAmount, order.amount * 0.3);
        
        if (fillAmount > 0) {
          const fillPrice = order.price || 0;
          const report: ExecutionReport = {
            orderId: order.id,
            fillId: `fill_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
            symbol: order.symbol,
            side: order.side,
            amount: fillAmount,
            price: fillPrice,
            fee: this.calculateFee(fillAmount, fillPrice, 'binance', 'maker'),
            feeAsset: 'USDT',
            timestamp: Date.now(),
            exchange: 'binance',
            liquidity: 'maker',
            executionLatency: elapsed
          };
          
          this.processExecutionReport(report);
          
          if (order.status === 'filled') {
            clearInterval(checkFills);
          }
        }
      }
    }, 1000); // Check every second
  }
  
  /**
   * Calculate trading fee
   */
  private calculateFee(amount: number, price: number, exchangeId: string, liquidity: 'maker' | 'taker'): number {
    const exchange = this.getExchangeConfig(exchangeId);
    if (!exchange) return 0;
    
    const feeRate = liquidity === 'maker' ? exchange.makerFee : exchange.takerFee;
    return amount * price * feeRate;
  }
  
  /**
   * Get exchange configuration (placeholder for router integration)
   */
  private getExchangeConfig(exchangeId: string): ExchangeConfig | undefined {
    // This would be integrated with the SmartOrderRouter
    return undefined;
  }
  
  /**
   * Utility function for async delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Get active orders
   */
  getActiveOrders(): Order[] {
    return Array.from(this.activeOrders.values());
  }
  
  /**
   * Get order by ID
   */
  getOrder(orderId: string): Order | undefined {
    return this.activeOrders.get(orderId);
  }
  
  /**
   * Cancel order
   */
  async cancelOrder(orderId: string): Promise<boolean> {
    const order = this.activeOrders.get(orderId);
    if (!order) return false;
    
    try {
      // In production, would cancel on exchange
      order.status = 'cancelled';
      console.log(`❌ Order ${orderId} cancelled`);
      this.emit('orderCancelled', order);
      return true;
    } catch (error) {
      console.error(`Failed to cancel order ${orderId}:`, error);
      return false;
    }
  }
}

/**
 * Main Order Execution Pipeline
 * The chrome that turns signals into reality
 */
export class OrderExecutionPipeline extends EventEmitter {
  private router: SmartOrderRouter;
  private engine: ExecutionEngine;
  private metrics: PipelineMetrics;
  private isActive = true;
  private orderQueue: Array<{ order: Order; priority: number; timestamp: number }> = [];
  private processingInterval: NodeJS.Timeout | null = null;
  
  constructor() {
    super();
    
    this.router = new SmartOrderRouter();
    this.engine = new ExecutionEngine();
    
    this.metrics = {
      totalOrders: 0,
      successfulOrders: 0,
      failedOrders: 0,
      cancelledOrders: 0,
      avgExecutionTime: 0,
      avgSlippage: 0,
      totalVolume: 0,
      totalFees: 0,
      ordersPerSecond: 0,
      currentQueueSize: 0,
      activeConnections: 0,
      exchangeMetrics: new Map(),
      lastUpdated: Date.now()
    };
    
    this.initializeEventHandlers();
    this.startProcessing();
    
    console.log('⚡ Order Execution Pipeline online - Ready to turn signals into eddies');
  }
  
  /**
   * Initialize event handlers
   */
  private initializeEventHandlers(): void {
    this.engine.on('orderExecuted', (data) => {
      this.metrics.successfulOrders++;
      this.updateExecutionMetrics(data.order, data.executionTime, true);
      this.emit('orderExecuted', data);
    });
    
    this.engine.on('orderFailed', (data) => {
      this.metrics.failedOrders++;
      this.updateExecutionMetrics(data.order, 0, false);
      this.emit('orderFailed', data);
    });
    
    this.engine.on('orderCancelled', (order) => {
      this.metrics.cancelledOrders++;
      this.emit('orderCancelled', order);
    });
    
    this.engine.on('executionReport', (report) => {
      this.updateExecutionReport(report);
      this.emit('executionReport', report);
    });
  }
  
  /**
   * Submit order for execution
   */
  async submitOrder(order: Partial<Order>): Promise<string> {
    if (!this.isActive) {
      throw new Error('Execution pipeline is not active');
    }
    
    try {
      // Create full order object with defaults
      const fullOrder: Order = {
        id: order.id || `order_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`,
        clientOrderId: order.clientOrderId || '',
        exchange: order.exchange || '',
        symbol: order.symbol || '',
        side: order.side || 'buy',
        type: order.type || 'market',
        amount: order.amount || 0,
        price: order.price,
        stopPrice: order.stopPrice,
        timeInForce: order.timeInForce || 'GTC',
        postOnly: order.postOnly || false,
        reduceOnly: order.reduceOnly || false,
        
        maxSlippage: order.maxSlippage || 0.005,
        partialFillAcceptable: order.partialFillAcceptable !== false,
        urgency: order.urgency || 'medium',
        
        maxLatency: order.maxLatency || 1000,
        failoverExchanges: order.failoverExchanges || [],
        
        status: 'pending',
        filledAmount: 0,
        avgFillPrice: 0,
        fees: 0,
        
        createdAt: Date.now(),
        
        source: order.source || 'manual',
        strategyId: order.strategyId,
        signalId: order.signalId,
        parentOrderId: order.parentOrderId,
        
        riskCheckPassed: false,
        emergencyOverride: order.emergencyOverride || false
      };
      
      // Validate order
      this.validateOrder(fullOrder);
      
      // Risk check unless emergency override
      if (!fullOrder.emergencyOverride) {
        const riskValidation = riskEngine.validateOrder(
          fullOrder.symbol,
          fullOrder.side,
          fullOrder.amount,
          fullOrder.price || 0,
          fullOrder.stopPrice,
          undefined, // take profit
          1 // leverage
        );
        
        if (!riskValidation.isValid) {
          throw new Error(`Risk check failed: ${riskValidation.reason}`);
        }
        
        fullOrder.riskCheckPassed = true;
      }
      
      // Calculate priority
      const priority = this.calculateOrderPriority(fullOrder);
      
      // Add to queue
      this.orderQueue.push({
        order: fullOrder,
        priority,
        timestamp: Date.now()
      });
      
      // Sort queue by priority (higher priority first)
      this.orderQueue.sort((a, b) => b.priority - a.priority);
      
      this.metrics.totalOrders++;
      this.metrics.currentQueueSize = this.orderQueue.length;
      
      console.log(`📥 Order queued: ${fullOrder.id} (priority: ${priority})`);
      
      this.emit('orderQueued', fullOrder);
      
      return fullOrder.id;
      
    } catch (error) {
      console.error('Order submission failed:', error);
      throw error;
    }
  }
  
  /**
   * Validate order parameters
   */
  private validateOrder(order: Order): void {
    if (!order.symbol) throw new Error('Symbol is required');
    if (!order.side) throw new Error('Side is required');
    if (order.amount <= 0) throw new Error('Amount must be positive');
    
    if (order.type === 'limit' && !order.price) {
      throw new Error('Price is required for limit orders');
    }
    
    if ((order.type === 'stop' || order.type === 'stop_limit') && !order.stopPrice) {
      throw new Error('Stop price is required for stop orders');
    }
    
    if (order.maxSlippage < 0 || order.maxSlippage > 1) {
      throw new Error('Max slippage must be between 0 and 1');
    }
  }
  
  /**
   * Calculate order priority for queue ordering
   */
  private calculateOrderPriority(order: Order): number {
    let priority = 50; // Base priority
    
    // Urgency boost
    switch (order.urgency) {
      case 'critical':
        priority += 40;
        break;
      case 'high':
        priority += 20;
        break;
      case 'medium':
        priority += 0;
        break;
      case 'low':
        priority -= 20;
        break;
    }
    
    // Source boost
    switch (order.source) {
      case 'signal':
        priority += 15; // AI signals get priority
        break;
      case 'arbitrage':
        priority += 25; // Arbitrage is time-sensitive
        break;
      case 'hedge':
        priority += 10; // Risk management orders
        break;
      case 'manual':
        priority += 0;
        break;
    }
    
    // Order type adjustment
    if (order.type === 'market') {
      priority += 10; // Market orders need fast execution
    }
    
    // Size adjustment
    if (order.amount > 100000) {
      priority += 5; // Large orders get slight priority
    }
    
    return Math.max(0, Math.min(100, priority));
  }
  
  /**
   * Start order processing
   */
  private startProcessing(): void {
    this.processingInterval = setInterval(async () => {
      await this.processOrderQueue();
    }, 10); // Process every 10ms for maximum throughput
  }
  
  /**
   * Process order queue
   */
  private async processOrderQueue(): Promise<void> {
    if (this.orderQueue.length === 0) return;
    
    const queueItem = this.orderQueue.shift();
    if (!queueItem) return;
    
    const { order } = queueItem;
    
    try {
      // Route the order
      const routing = await this.router.routeOrder(order);
      
      // Execute the order
      await this.engine.executeOrder(order, routing);
      
      // Update metrics
      this.metrics.currentQueueSize = this.orderQueue.length;
      
    } catch (error) {
      console.error(`Order processing failed for ${order.id}:`, error);
      
      // Try failover exchanges if available
      if (order.failoverExchanges.length > 0) {
        console.log(`🔄 Attempting failover for order ${order.id}`);
        // Would implement failover logic here
      }
    }
  }
  
  /**
   * Update execution metrics
   */
  private updateExecutionMetrics(order: Order, executionTime: number, success: boolean): void {
    // Update average execution time
    const currentAvg = this.metrics.avgExecutionTime;
    const totalSuccessful = this.metrics.successfulOrders;
    
    if (success && totalSuccessful > 0) {
      this.metrics.avgExecutionTime = ((currentAvg * (totalSuccessful - 1)) + executionTime) / totalSuccessful;
    }
    
    // Update volume
    this.metrics.totalVolume += order.amount * (order.avgFillPrice || order.price || 0);
    
    this.metrics.lastUpdated = Date.now();
  }
  
  /**
   * Update execution report metrics
   */
  private updateExecutionReport(report: ExecutionReport): void {
    this.metrics.totalFees += report.fee;
    
    // Update exchange-specific metrics
    const exchangeMetric = this.metrics.exchangeMetrics.get(report.exchange) || {
      orderCount: 0,
      successRate: 0,
      avgLatency: 0,
      avgSlippage: 0,
      totalVolume: 0
    };
    
    exchangeMetric.orderCount++;
    exchangeMetric.avgLatency = ((exchangeMetric.avgLatency * (exchangeMetric.orderCount - 1)) + report.executionLatency) / exchangeMetric.orderCount;
    exchangeMetric.totalVolume += report.amount * report.price;
    
    this.metrics.exchangeMetrics.set(report.exchange, exchangeMetric);
  }
  
  /**
   * Get pipeline metrics
   */
  getMetrics(): PipelineMetrics {
    // Calculate orders per second
    const now = Date.now();
    const timeWindow = 60000; // 1 minute
    const recentOrders = this.metrics.successfulOrders; // Simplified calculation
    
    this.metrics.ordersPerSecond = recentOrders / 60;
    this.metrics.lastUpdated = now;
    
    return { ...this.metrics };
  }
  
  /**
   * Get active orders
   */
  getActiveOrders(): Order[] {
    return this.engine.getActiveOrders();
  }
  
  /**
   * Cancel order
   */
  async cancelOrder(orderId: string): Promise<boolean> {
    // Try to remove from queue first
    const queueIndex = this.orderQueue.findIndex(item => item.order.id === orderId);
    if (queueIndex !== -1) {
      this.orderQueue.splice(queueIndex, 1);
      this.metrics.currentQueueSize = this.orderQueue.length;
      console.log(`❌ Order ${orderId} removed from queue`);
      return true;
    }
    
    // Try to cancel active order
    return await this.engine.cancelOrder(orderId);
  }
  
  /**
   * Emergency stop all orders
   */
  emergencyStop(): void {
    console.error('🚨 EMERGENCY STOP - Halting all order processing');
    
    this.isActive = false;
    
    // Clear the queue
    const queuedOrders = this.orderQueue.length;
    this.orderQueue = [];
    
    // Cancel all active orders
    const activeOrders = this.engine.getActiveOrders();
    for (const order of activeOrders) {
      this.engine.cancelOrder(order.id);
    }
    
    console.error(`🛑 Emergency stop complete: ${queuedOrders} queued orders cleared, ${activeOrders.length} active orders cancelled`);
    
    this.emit('emergencyStop', {
      queuedOrdersCleared: queuedOrders,
      activeOrdersCancelled: activeOrders.length
    });
  }
  
  /**
   * Resume operations after emergency stop
   */
  resume(): void {
    this.isActive = true;
    console.log('✅ Order execution pipeline resumed');
    this.emit('resumed');
  }
  
  /**
   * Shutdown the pipeline
   */
  shutdown(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
    
    this.isActive = false;
    console.log('💀 Order execution pipeline shutdown');
  }
}

// Export singleton instance
export const executionPipeline = new OrderExecutionPipeline();

// Utility functions for React components
export const executionUtils = {
  /**
   * Format order status with color
   */
  getStatusColor(status: string): string {
    switch (status) {
      case 'filled':
        return 'text-green-400';
      case 'partial':
        return 'text-yellow-400';
      case 'cancelled':
      case 'rejected':
      case 'failed':
        return 'text-red-400';
      case 'pending':
      case 'routing':
      case 'submitted':
        return 'text-blue-400';
      default:
        return 'text-gray-400';
    }
  },
  
  /**
   * Calculate execution efficiency score
   */
  calculateEfficiencyScore(order: Order): number {
    if (!order.completedAt || !order.createdAt) return 0;
    
    const executionTime = order.completedAt - order.createdAt;
    const targetTime = 1000; // 1 second target
    
    const timeScore = Math.max(0, (targetTime - executionTime) / targetTime * 50);
    const fillScore = (order.filledAmount / order.amount) * 30;
    const slippageScore = Math.max(0, (0.005 - order.maxSlippage) / 0.005 * 20);
    
    return Math.min(100, timeScore + fillScore + slippageScore);
  },
  
  /**
   * Format execution time
   */
  formatExecutionTime(startTime: number, endTime?: number): string {
    if (!endTime) return 'Pending...';
    
    const duration = endTime - startTime;
    
    if (duration < 1000) return `${duration.toFixed(0)}ms`;
    if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`;
    return `${(duration / 60000).toFixed(1)}m`;
  }
};
