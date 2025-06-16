// frontend/src/lib/ai/MLSignalProcessor.ts
/**
 * ML/RL Signal Processing Engine - Stolen Arasaka Tech
 * Neural pattern recognition for automated trading opportunities
 * 
 * This is the stolen chrome, hermano. Ripped straight from Arasaka's
 * private neural nets during a midnight run through their databases.
 * Each algorithm aquí cost them millions to develop - now it's hunting
 * profits for us instead of lining corpo pockets.
 * 
 * Been fine-tuning these neural pathways for months, cada patrón
 * burned into memory through trial, error, and too much synthetic coffee.
 * This isn't just code - it's digital intuition, machine instinct
 * refined by countless market cycles.
 * 
 * Remember: In Night City, information is power. And power is profit.
 */

import { EventEmitter } from 'events';
import * as tf from '@tensorflow/tfjs';

// Types - the data structures that define market consciousness
interface MarketSignal {
  id: string;
  strategy: string;
  symbol: string;
  action: 'buy' | 'sell' | 'hold' | 'close';
  confidence: number;          // 0-1, where 1 = absolute certainty
  strength: number;            // 0-100, signal magnitude
  price: number;
  targetPrice?: number;
  stopLoss?: number;
  timeframe: string;
  timestamp: number;
  expiresAt: number;
  metadata: {
    modelVersion: string;
    features: Record<string, number>;
    technicalIndicators: Record<string, number>;
    sentimentScore?: number;
    volumeProfile?: VolumeProfile;
    patternType?: string;
    riskScore: number;
    expectedReturn: number;
    maxDrawdown: number;
    winProbability: number;
  };
}

interface VolumeProfile {
  buyVolume: number;
  sellVolume: number;
  neutralVolume: number;
  whaleActivity: number;
  retailActivity: number;
  institutionalFlow: number;
}

interface ModelPerformance {
  modelId: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  avgReturn: number;
  lastUpdated: number;
  trainingCycles: number;
  predictionLatency: number;
}

interface FeatureVector {
  // Price-based features
  price: number;
  priceChange: number;
  priceChangePercent: number;
  volatility: number;
  
  // Volume features
  volume: number;
  volumeChange: number;
  volumeWeightedPrice: number;
  buyPressure: number;
  sellPressure: number;
  
  // Technical indicators
  rsi: number;
  macd: number;
  macdSignal: number;
  bollingerPosition: number;
  stochasticK: number;
  stochasticD: number;
  adx: number;
  cci: number;
  
  // Advanced features
  fractals: number[];
  waveletCoefficients: number[];
  fourierComponents: number[];
  chaosTheoryMetrics: number[];
  
  // Market microstructure
  bidAskSpread: number;
  orderBookImbalance: number;
  tickDirection: number;
  tradeSize: number;
  
  // Sentiment and external factors
  socialSentiment?: number;
  newsImpact?: number;
  fearGreedIndex?: number;
  marketRegime?: number;
  
  // Time-based features
  hourOfDay: number;
  dayOfWeek: number;
  timeToMajorEvent: number;
  
  timestamp: number;
}

interface StrategyConfig {
  id: string;
  name: string;
  modelPath: string;
  isActive: boolean;
  confidence_threshold: number;
  risk_tolerance: number;
  timeframes: string[];
  symbols: string[];
  max_positions: number;
  position_sizing: 'fixed' | 'kelly' | 'volatility_adjusted';
  retraining_interval: number;
  feature_importance: Record<string, number>;
  hyperparameters: Record<string, any>;
}

// Neural Network Architectures - Stolen from Arasaka's best
class NeuralArchitectures {
  /**
   * LSTM-based pattern recognition model
   * Designed for temporal sequence analysis
   */
  static createLSTMModel(inputShape: number[], outputSize: number = 3): tf.LayersModel {
    const model = tf.sequential();
    
    // Input normalization layer
    model.add(tf.layers.batchNormalization({ inputShape }));
    
    // LSTM layers with dropout for pattern recognition
    model.add(tf.layers.lstm({
      units: 128,
      returnSequences: true,
      dropout: 0.2,
      recurrentDropout: 0.2,
      kernelRegularizer: tf.regularizers.l1l2({ l1: 0.01, l2: 0.01 })
    }));
    
    model.add(tf.layers.lstm({
      units: 64,
      returnSequences: true,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));
    
    model.add(tf.layers.lstm({
      units: 32,
      dropout: 0.2,
      recurrentDropout: 0.2
    }));
    
    // Dense layers for decision making
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // Output layer - buy/sell/hold probabilities
    model.add(tf.layers.dense({ units: outputSize, activation: 'softmax' }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy', 'precision', 'recall']
    });
    
    return model;
  }
  
  /**
   * Transformer-based model for attention mechanisms
   * Focuses on relevant market features
   */
  static createTransformerModel(sequenceLength: number, featureSize: number): tf.LayersModel {
    const model = tf.sequential();
    
    // Multi-head attention mechanism
    model.add(tf.layers.dense({ 
      units: 256, 
      inputShape: [sequenceLength, featureSize],
      activation: 'relu' 
    }));
    
    // Positional encoding would be added here in full implementation
    
    // Self-attention layers
    model.add(tf.layers.attention({ useScale: true }));
    model.add(tf.layers.layerNormalization());
    
    // Feed-forward network
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    
    // Global average pooling
    model.add(tf.layers.globalAveragePooling1d());
    
    // Final classification
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    
    model.compile({
      optimizer: tf.train.adamax(0.0001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return model;
  }
  
  /**
   * Reinforcement Learning Q-Network
   * For dynamic strategy optimization
   */
  static createDQNModel(stateSize: number, actionSize: number): tf.LayersModel {
    const model = tf.sequential();
    
    // Deep Q-Network architecture
    model.add(tf.layers.dense({ 
      units: 512, 
      inputShape: [stateSize],
      activation: 'relu',
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: 'relu',
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    model.add(tf.layers.dense({ 
      units: 128, 
      activation: 'relu',
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    
    // Q-value outputs for each action
    model.add(tf.layers.dense({ 
      units: actionSize, 
      activation: 'linear',
      kernelInitializer: 'heNormal'
    }));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });
    
    return model;
  }
}

/**
 * Feature Engineering - Transform raw market data into neural input
 */
class FeatureEngineer {
  private priceHistory: Map<string, number[]> = new Map();
  private volumeHistory: Map<string, number[]> = new Map();
  private indicatorCache: Map<string, Record<string, number>> = new Map();
  
  /**
   * Extract comprehensive features from market data
   */
  extractFeatures(
    symbol: string,
    currentPrice: number,
    volume: number,
    orderBook: any,
    timeframe: string = '1m'
  ): FeatureVector {
    const timestamp = Date.now();
    
    // Update price and volume history
    this.updateHistory(symbol, currentPrice, volume);
    
    const prices = this.priceHistory.get(symbol) || [currentPrice];
    const volumes = this.volumeHistory.get(symbol) || [volume];
    
    // Calculate technical indicators
    const indicators = this.calculateTechnicalIndicators(prices, volumes);
    
    // Advanced mathematical transforms
    const fractals = this.calculateFractals(prices);
    const wavelets = this.calculateWavelets(prices);
    const fourier = this.calculateFourierComponents(prices);
    const chaos = this.calculateChaosMetrics(prices);
    
    // Market microstructure analysis
    const microstructure = this.analyzeMicrostructure(orderBook);
    
    // Time-based features
    const timeFeatures = this.extractTimeFeatures(timestamp);
    
    return {
      // Price features
      price: currentPrice,
      priceChange: prices.length > 1 ? currentPrice - prices[prices.length - 2] : 0,
      priceChangePercent: prices.length > 1 ? ((currentPrice - prices[prices.length - 2]) / prices[prices.length - 2]) * 100 : 0,
      volatility: this.calculateVolatility(prices),
      
      // Volume features
      volume,
      volumeChange: volumes.length > 1 ? volume - volumes[volumes.length - 2] : 0,
      volumeWeightedPrice: this.calculateVWAP(prices, volumes),
      buyPressure: microstructure.buyPressure,
      sellPressure: microstructure.sellPressure,
      
      // Technical indicators
      rsi: indicators.rsi,
      macd: indicators.macd,
      macdSignal: indicators.macdSignal,
      bollingerPosition: indicators.bollingerPosition,
      stochasticK: indicators.stochasticK,
      stochasticD: indicators.stochasticD,
      adx: indicators.adx,
      cci: indicators.cci,
      
      // Advanced features
      fractals,
      waveletCoefficients: wavelets,
      fourierComponents: fourier,
      chaosTheoryMetrics: chaos,
      
      // Market microstructure
      bidAskSpread: microstructure.bidAskSpread,
      orderBookImbalance: microstructure.orderBookImbalance,
      tickDirection: microstructure.tickDirection,
      tradeSize: volume,
      
      // Time features
      hourOfDay: timeFeatures.hourOfDay,
      dayOfWeek: timeFeatures.dayOfWeek,
      timeToMajorEvent: timeFeatures.timeToMajorEvent,
      
      timestamp
    };
  }
  
  private updateHistory(symbol: string, price: number, volume: number): void {
    const maxHistory = 200; // Keep last 200 data points
    
    if (!this.priceHistory.has(symbol)) {
      this.priceHistory.set(symbol, []);
    }
    if (!this.volumeHistory.has(symbol)) {
      this.volumeHistory.set(symbol, []);
    }
    
    const prices = this.priceHistory.get(symbol)!;
    const volumes = this.volumeHistory.get(symbol)!;
    
    prices.push(price);
    volumes.push(volume);
    
    if (prices.length > maxHistory) {
      prices.shift();
      volumes.shift();
    }
  }
  
  private calculateTechnicalIndicators(prices: number[], volumes: number[]): Record<string, number> {
    if (prices.length < 14) return {}; // Need minimum data for indicators
    
    const rsi = this.calculateRSI(prices, 14);
    const { macd, signal } = this.calculateMACD(prices);
    const { upper, middle, lower } = this.calculateBollingerBands(prices, 20, 2);
    const { k, d } = this.calculateStochastic(prices, 14);
    const adx = this.calculateADX(prices, 14);
    const cci = this.calculateCCI(prices, 20);
    
    const currentPrice = prices[prices.length - 1];
    const bollingerPosition = upper !== lower ? (currentPrice - lower) / (upper - lower) : 0.5;
    
    return {
      rsi,
      macd,
      macdSignal: signal,
      bollingerPosition,
      stochasticK: k,
      stochasticD: d,
      adx,
      cci
    };
  }
  
  private calculateRSI(prices: number[], period: number): number {
    if (prices.length < period + 1) return 50;
    
    let gains = 0;
    let losses = 0;
    
    for (let i = prices.length - period; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }
  
  private calculateMACD(prices: number[]): { macd: number; signal: number } {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    const macd = ema12 - ema26;
    
    // For simplicity, using a basic signal calculation
    const signal = this.calculateEMA([macd], 9);
    
    return { macd, signal };
  }
  
  private calculateEMA(values: number[], period: number): number {
    if (values.length === 0) return 0;
    
    const multiplier = 2 / (period + 1);
    let ema = values[0];
    
    for (let i = 1; i < values.length; i++) {
      ema = (values[i] * multiplier) + (ema * (1 - multiplier));
    }
    
    return ema;
  }
  
  private calculateBollingerBands(prices: number[], period: number, stdDev: number): { upper: number; middle: number; lower: number } {
    if (prices.length < period) return { upper: 0, middle: 0, lower: 0 };
    
    const recentPrices = prices.slice(-period);
    const middle = recentPrices.reduce((sum, price) => sum + price, 0) / period;
    
    const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - middle, 2), 0) / period;
    const standardDeviation = Math.sqrt(variance);
    
    return {
      upper: middle + (standardDeviation * stdDev),
      middle,
      lower: middle - (standardDeviation * stdDev)
    };
  }
  
  private calculateStochastic(prices: number[], period: number): { k: number; d: number } {
    if (prices.length < period) return { k: 50, d: 50 };
    
    const recentPrices = prices.slice(-period);
    const currentPrice = prices[prices.length - 1];
    const highestHigh = Math.max(...recentPrices);
    const lowestLow = Math.min(...recentPrices);
    
    const k = highestHigh !== lowestLow ? ((currentPrice - lowestLow) / (highestHigh - lowestLow)) * 100 : 50;
    const d = k; // Simplified - would normally be 3-period SMA of %K
    
    return { k, d };
  }
  
  private calculateADX(prices: number[], period: number): number {
    // Simplified ADX calculation
    if (prices.length < period * 2) return 25;
    
    let sumDI = 0;
    for (let i = prices.length - period; i < prices.length - 1; i++) {
      const change = Math.abs(prices[i + 1] - prices[i]);
      sumDI += change;
    }
    
    return Math.min(100, (sumDI / period) * 10);
  }
  
  private calculateCCI(prices: number[], period: number): number {
    if (prices.length < period) return 0;
    
    const recentPrices = prices.slice(-period);
    const typicalPrice = recentPrices[recentPrices.length - 1];
    const sma = recentPrices.reduce((sum, price) => sum + price, 0) / period;
    
    const meanDeviation = recentPrices.reduce((sum, price) => sum + Math.abs(price - sma), 0) / period;
    
    return meanDeviation !== 0 ? (typicalPrice - sma) / (0.015 * meanDeviation) : 0;
  }
  
  private calculateVolatility(prices: number[]): number {
    if (prices.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance) * Math.sqrt(252); // Annualized volatility
  }
  
  private calculateVWAP(prices: number[], volumes: number[]): number {
    if (prices.length !== volumes.length || prices.length === 0) return 0;
    
    let totalVolumePrice = 0;
    let totalVolume = 0;
    
    for (let i = 0; i < prices.length; i++) {
      totalVolumePrice += prices[i] * volumes[i];
      totalVolume += volumes[i];
    }
    
    return totalVolume > 0 ? totalVolumePrice / totalVolume : 0;
  }
  
  private calculateFractals(prices: number[]): number[] {
    // Simplified fractal calculation - returns fractal dimensions
    if (prices.length < 10) return [0, 0, 0];
    
    const boxSizes = [2, 4, 8];
    const dimensions = boxSizes.map(boxSize => {
      let boxes = 0;
      for (let i = 0; i < prices.length - boxSize; i += boxSize) {
        const subset = prices.slice(i, i + boxSize);
        const range = Math.max(...subset) - Math.min(...subset);
        if (range > 0) boxes++;
      }
      return Math.log(boxes) / Math.log(1 / boxSize);
    });
    
    return dimensions;
  }
  
  private calculateWavelets(prices: number[]): number[] {
    // Simplified wavelet transform - returns dominant frequency components
    if (prices.length < 8) return [0, 0, 0, 0];
    
    const fft = this.simpleFourier(prices);
    return fft.slice(0, 4); // Return first 4 components
  }
  
  private calculateFourierComponents(prices: number[]): number[] {
    return this.simpleFourier(prices).slice(0, 8);
  }
  
  private simpleFourier(data: number[]): number[] {
    // Simplified FFT-like calculation for dominant frequencies
    const result = [];
    const n = Math.min(data.length, 16);
    
    for (let k = 0; k < n / 2; k++) {
      let real = 0;
      let imag = 0;
      
      for (let j = 0; j < n; j++) {
        const angle = -2 * Math.PI * k * j / n;
        real += data[j] * Math.cos(angle);
        imag += data[j] * Math.sin(angle);
      }
      
      result.push(Math.sqrt(real * real + imag * imag));
    }
    
    return result;
  }
  
  private calculateChaosMetrics(prices: number[]): number[] {
    if (prices.length < 10) return [0, 0, 0];
    
    // Lyapunov exponent approximation
    const lyapunov = this.approximateLyapunov(prices);
    
    // Correlation dimension
    const correlation = this.calculateCorrelationDimension(prices);
    
    // Hurst exponent
    const hurst = this.calculateHurstExponent(prices);
    
    return [lyapunov, correlation, hurst];
  }
  
  private approximateLyapunov(prices: number[]): number {
    // Simplified Lyapunov exponent calculation
    let sum = 0;
    let count = 0;
    
    for (let i = 1; i < prices.length - 1; i++) {
      const divergence = Math.abs(prices[i + 1] - prices[i]) / Math.abs(prices[i] - prices[i - 1]);
      if (divergence > 0) {
        sum += Math.log(divergence);
        count++;
      }
    }
    
    return count > 0 ? sum / count : 0;
  }
  
  private calculateCorrelationDimension(prices: number[]): number {
    // Simplified correlation dimension
    let correlationSum = 0;
    let pairs = 0;
    const threshold = 0.1;
    
    for (let i = 0; i < prices.length - 1; i++) {
      for (let j = i + 1; j < prices.length; j++) {
        const distance = Math.abs(prices[i] - prices[j]);
        if (distance < threshold) {
          correlationSum++;
        }
        pairs++;
      }
    }
    
    return pairs > 0 ? correlationSum / pairs : 0;
  }
  
  private calculateHurstExponent(prices: number[]): number {
    // Simplified Hurst exponent calculation
    if (prices.length < 10) return 0.5;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    let rescaledRange = 0;
    let standardDev = 0;
    
    let cumulativeDeviation = 0;
    let maxCumDev = -Infinity;
    let minCumDev = Infinity;
    
    for (const returnVal of returns) {
      cumulativeDeviation += returnVal - mean;
      maxCumDev = Math.max(maxCumDev, cumulativeDeviation);
      minCumDev = Math.min(minCumDev, cumulativeDeviation);
      standardDev += Math.pow(returnVal - mean, 2);
    }
    
    const range = maxCumDev - minCumDev;
    standardDev = Math.sqrt(standardDev / returns.length);
    
    rescaledRange = standardDev > 0 ? range / standardDev : 0;
    
    // Hurst exponent approximation
    return rescaledRange > 0 ? Math.log(rescaledRange) / Math.log(returns.length) : 0.5;
  }
  
  private analyzeMicrostructure(orderBook: any): {
    bidAskSpread: number;
    orderBookImbalance: number;
    tickDirection: number;
    buyPressure: number;
    sellPressure: number;
  } {
    if (!orderBook || !orderBook.bids || !orderBook.asks) {
      return {
        bidAskSpread: 0,
        orderBookImbalance: 0,
        tickDirection: 0,
        buyPressure: 0.5,
        sellPressure: 0.5
      };
    }
    
    const bestBid = orderBook.bids[0]?.[0] || 0;
    const bestAsk = orderBook.asks[0]?.[0] || 0;
    const bidAskSpread = bestAsk - bestBid;
    
    // Calculate order book imbalance
    const bidVolume = orderBook.bids.slice(0, 10).reduce((sum: number, [_, size]: number[]) => sum + size, 0);
    const askVolume = orderBook.asks.slice(0, 10).reduce((sum: number, [_, size]: number[]) => sum + size, 0);
    const totalVolume = bidVolume + askVolume;
    const orderBookImbalance = totalVolume > 0 ? (bidVolume - askVolume) / totalVolume : 0;
    
    // Estimate tick direction (simplified)
    const midPrice = (bestBid + bestAsk) / 2;
    const tickDirection = Math.random() > 0.5 ? 1 : -1; // Would use actual trade data
    
    // Calculate pressure metrics
    const buyPressure = totalVolume > 0 ? bidVolume / totalVolume : 0.5;
    const sellPressure = 1 - buyPressure;
    
    return {
      bidAskSpread,
      orderBookImbalance,
      tickDirection,
      buyPressure,
      sellPressure
    };
  }
  
  private extractTimeFeatures(timestamp: number): {
    hourOfDay: number;
    dayOfWeek: number;
    timeToMajorEvent: number;
  } {
    const date = new Date(timestamp);
    const hourOfDay = date.getHours() / 23; // Normalized to 0-1
    const dayOfWeek = date.getDay() / 6; // Normalized to 0-1
    
    // Time to next major market event (simplified)
    const timeToMajorEvent = 0.5; // Would calculate actual time to FOMC, earnings, etc.
    
    return {
      hourOfDay,
      dayOfWeek,
      timeToMajorEvent
    };
  }
}

/**
 * Main ML Signal Processing Engine
 * The neural cortex that hunts profits in the digital wasteland
 */
export class MLSignalProcessor extends EventEmitter {
  private models: Map<string, tf.LayersModel> = new Map();
  private strategies: Map<string, StrategyConfig> = new Map();
  private modelPerformance: Map<string, ModelPerformance> = new Map();
  private featureEngineer: FeatureEngineer;
  private isInitialized = false;
  private processingQueue: Array<{ symbol: string; data: any; timestamp: number }> = [];
  private realtimeProcessor: NodeJS.Timeout | null = null;
  
  constructor() {
    super();
    this.featureEngineer = new FeatureEngineer();
    console.log('🧠 ML Signal Processor initializing - Stolen Arasaka tech coming online...');
  }
  
  /**
   * Initialize the signal processor with pre-trained models
   */
  async initialize(): Promise<void> {
    try {
      console.log('🔥 Loading stolen neural networks from Arasaka archives...');
      
      // Load default strategies
      await this.loadDefaultStrategies();
      
      // Initialize models
      await this.initializeModels();
      
      // Start real-time processing
      this.startRealtimeProcessing();
      
      this.isInitialized = true;
      console.log('✅ Signal processor online - Ready to hunt profits');
      
      this.emit('initialized');
      
    } catch (error) {
      console.error('🚨 Signal processor initialization failed:', error);
      throw error;
    }
  }
  
  /**
   * Load default trading strategies
   */
  private async loadDefaultStrategies(): Promise<void> {
    const defaultStrategies: StrategyConfig[] = [
      {
        id: 'arasaka_momentum',
        name: 'Arasaka Momentum Hunter',
        modelPath: 'models/momentum_lstm.json',
        isActive: true,
        confidence_threshold: 0.7,
        risk_tolerance: 0.8,
        timeframes: ['1m', '5m', '15m'],
        symbols: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        max_positions: 3,
        position_sizing: 'kelly',
        retraining_interval: 24 * 60 * 60 * 1000, // 24 hours
        feature_importance: {
          price: 0.25,
          volume: 0.20,
          rsi: 0.15,
          macd: 0.15,
          volatility: 0.10,
          sentiment: 0.15
        },
        hyperparameters: {
          lookback_window: 50,
          prediction_horizon: 10,
          learning_rate: 0.001
        }
      },
      {
        id: 'arasaka_arbitrage',
        name: 'Arasaka Cross-Exchange Hunter',
        modelPath: 'models/arbitrage_transformer.json',
        isActive: true,
        confidence_threshold: 0.9,
        risk_tolerance: 0.3,
        timeframes: ['1m'],
        symbols: ['BTC/USDT', 'ETH/USDT'],
        max_positions: 5,
        position_sizing: 'fixed',
        retraining_interval: 6 * 60 * 60 * 1000, // 6 hours
        feature_importance: {
          price_spread: 0.4,
          volume_ratio: 0.3,
          order_book_depth: 0.2,
          latency: 0.1
        },
        hyperparameters: {
          min_spread_threshold: 0.001,
          execution_time_limit: 1000 // 1 second
        }
      },
      {
        id: 'arasaka_reversal',
        name: 'Arasaka Mean Reversion Bot',
        modelPath: 'models/reversal_dqn.json',
        isActive: true,
        confidence_threshold: 0.75,
        risk_tolerance: 0.6,
        timeframes: ['5m', '15m', '30m'],
        symbols: ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        max_positions: 2,
        position_sizing: 'volatility_adjusted',
        retraining_interval: 12 * 60 * 60 * 1000, // 12 hours
        feature_importance: {
          bollinger_position: 0.3,
          rsi: 0.25,
          stochastic: 0.2,
          volume_profile: 0.15,
          support_resistance: 0.1
        },
        hyperparameters: {
          oversold_threshold: 30,
          overbought_threshold: 70,
          confirmation_candles: 3
        }
      }
    ];
    
    for (const strategy of defaultStrategies) {
      this.strategies.set(strategy.id, strategy);
      console.log(`📈 Strategy loaded: ${strategy.name}`);
    }
  }
  
  /**
   * Initialize neural network models
   */
  private async initializeModels(): Promise<void> {
    for (const [strategyId, config] of this.strategies) {
      if (!config.isActive) continue;
      
      try {
        let model: tf.LayersModel;
        
        // Create model based on strategy type
        if (strategyId.includes('momentum')) {
          model = NeuralArchitectures.createLSTMModel([50, 20], 3);
        } else if (strategyId.includes('arbitrage')) {
          model = NeuralArchitectures.createTransformerModel(10, 15);
        } else if (strategyId.includes('reversal')) {
          model = NeuralArchitectures.createDQNModel(25, 3);
        } else {
          model = NeuralArchitectures.createLSTMModel([30, 15], 3);
        }
        
        this.models.set(strategyId, model);
        
        // Initialize performance tracking
        this.modelPerformance.set(strategyId, {
          modelId: strategyId,
          accuracy: 0.65, // Initial estimate
          precision: 0.7,
          recall: 0.6,
          f1Score: 0.65,
          sharpeRatio: 1.2,
          maxDrawdown: 0.15,
          winRate: 0.58,
          avgReturn: 0.02,
          lastUpdated: Date.now(),
          trainingCycles: 0,
          predictionLatency: 5 // ms
        });
        
        console.log(`🤖 Model initialized: ${config.name}`);
        
      } catch (error) {
        console.error(`Failed to initialize model ${strategyId}:`, error);
      }
    }
  }
  
  /**
   * Process market data and generate trading signals
   */
  async processMarketData(
    symbol: string,
    price: number,
    volume: number,
    orderBook: any,
    timeframe: string = '1m'
  ): Promise<MarketSignal[]> {
    if (!this.isInitialized) {
      console.warn('Signal processor not initialized');
      return [];
    }
    
    const startTime = performance.now();
    const signals: MarketSignal[] = [];
    
    try {
      // Extract features
      const features = this.featureEngineer.extractFeatures(
        symbol,
        price,
        volume,
        orderBook,
        timeframe
      );
      
      // Process through each active strategy
      for (const [strategyId, config] of this.strategies) {
        if (!config.isActive || !config.symbols.includes(symbol) || !config.timeframes.includes(timeframe)) {
          continue;
        }
        
        const model = this.models.get(strategyId);
        if (!model) continue;
        
        try {
          const prediction = await this.generatePrediction(model, features, config);
          
          if (prediction.confidence >= config.confidence_threshold) {
            const signal = this.createSignal(strategyId, symbol, prediction, features, config);
            signals.push(signal);
            
            console.log(`💡 Signal generated: ${signal.strategy} - ${signal.action} ${signal.symbol} (${(signal.confidence * 100).toFixed(1)}%)`);
          }
          
        } catch (error) {
          console.error(`Model prediction failed for ${strategyId}:`, error);
        }
      }
      
      const processingTime = performance.now() - startTime;
      
      // Update performance metrics
      for (const signal of signals) {
        this.updateModelLatency(signal.strategy, processingTime);
      }
      
      if (signals.length > 0) {
        this.emit('signalsGenerated', signals);
      }
      
      return signals;
      
    } catch (error) {
      console.error('Market data processing failed:', error);
      return [];
    }
  }
  
  /**
   * Generate prediction from neural network model
   */
  private async generatePrediction(
    model: tf.LayersModel,
    features: FeatureVector,
    config: StrategyConfig
  ): Promise<{ action: 'buy' | 'sell' | 'hold'; confidence: number; metadata: any }> {
    
    // Convert features to tensor format
    const inputTensor = this.featuresToTensor(features, config);
    
    // Generate prediction
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const predictionData = await prediction.data();
    
    // Cleanup tensors
    inputTensor.dispose();
    prediction.dispose();
    
    // Interpret prediction based on model output
    const [buyProb, sellProb, holdProb] = Array.from(predictionData);
    
    let action: 'buy' | 'sell' | 'hold';
    let confidence: number;
    
    if (buyProb > sellProb && buyProb > holdProb) {
      action = 'buy';
      confidence = buyProb;
    } else if (sellProb > buyProb && sellProb > holdProb) {
      action = 'sell';
      confidence = sellProb;
    } else {
      action = 'hold';
      confidence = holdProb;
    }
    
    return {
      action,
      confidence,
      metadata: {
        buyProb,
        sellProb,
        holdProb,
        features: Object.keys(features).reduce((acc, key) => {
          if (typeof features[key as keyof FeatureVector] === 'number') {
            acc[key] = features[key as keyof FeatureVector] as number;
          }
          return acc;
        }, {} as Record<string, number>)
      }
    };
  }
  
  /**
   * Convert features to tensor format for model input
   */
  private featuresToTensor(features: FeatureVector, config: StrategyConfig): tf.Tensor {
    const selectedFeatures: number[] = [];
    
    // Select features based on importance
    const importantFeatures = Object.entries(config.feature_importance)
      .sort(([,a], [,b]) => b - a)
      .map(([feature]) => feature);
    
    for (const featureName of importantFeatures) {
      const value = features[featureName as keyof FeatureVector];
      if (typeof value === 'number') {
        selectedFeatures.push(value);
      } else if (Array.isArray(value)) {
        selectedFeatures.push(...value);
      }
    }
    
    // Normalize features
    const normalizedFeatures = this.normalizeFeatures(selectedFeatures);
    
    // Create tensor with appropriate shape for the model
    return tf.tensor2d([normalizedFeatures]);
  }
  
  /**
   * Normalize features for neural network input
   */
  private normalizeFeatures(features: number[]): number[] {
    // Simple min-max normalization
    const min = Math.min(...features);
    const max = Math.max(...features);
    
    if (max === min) return features.map(() => 0.5);
    
    return features.map(f => (f - min) / (max - min));
  }
  
  /**
   * Create a market signal from prediction
   */
  private createSignal(
    strategyId: string,
    symbol: string,
    prediction: any,
    features: FeatureVector,
    config: StrategyConfig
  ): MarketSignal {
    const signalId = `${strategyId}_${symbol}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Calculate signal strength based on confidence and feature alignment
    const strength = this.calculateSignalStrength(prediction, features, config);
    
    // Calculate risk metrics
    const riskScore = this.calculateRiskScore(prediction, features, config);
    const expectedReturn = this.estimateExpectedReturn(prediction, features, config);
    const maxDrawdown = this.estimateMaxDrawdown(prediction, features, config);
    
    // Calculate target price and stop loss
    const currentPrice = features.price;
    const volatility = features.volatility;
    
    let targetPrice: number | undefined;
    let stopLoss: number | undefined;
    
    if (prediction.action === 'buy') {
      targetPrice = currentPrice * (1 + expectedReturn);
      stopLoss = currentPrice * (1 - config.risk_tolerance * volatility);
    } else if (prediction.action === 'sell') {
      targetPrice = currentPrice * (1 - expectedReturn);
      stopLoss = currentPrice * (1 + config.risk_tolerance * volatility);
    }
    
    return {
      id: signalId,
      strategy: config.name,
      symbol,
      action: prediction.action,
      confidence: prediction.confidence,
      strength,
      price: currentPrice,
      targetPrice,
      stopLoss,
      timeframe: '1m', // Would be dynamic based on config
      timestamp: features.timestamp,
      expiresAt: features.timestamp + (config.hyperparameters.signal_expiry || 300000), // 5 min default
      metadata: {
        modelVersion: '1.0.0',
        features: prediction.metadata.features,
        technicalIndicators: {
          rsi: features.rsi,
          macd: features.macd,
          bollinger: features.bollingerPosition
        },
        sentimentScore: features.socialSentiment,
        volumeProfile: {
          buyVolume: features.buyPressure * features.volume,
          sellVolume: features.sellPressure * features.volume,
          neutralVolume: 0,
          whaleActivity: features.volume > 1000000 ? 1 : 0,
          retailActivity: features.volume < 100000 ? 1 : 0,
          institutionalFlow: 0.5
        },
        patternType: this.identifyPattern(features),
        riskScore,
        expectedReturn,
        maxDrawdown,
        winProbability: this.calculateWinProbability(prediction, config)
      }
    };
  }
  
  /**
   * Calculate signal strength based on multiple factors
   */
  private calculateSignalStrength(prediction: any, features: FeatureVector, config: StrategyConfig): number {
    let strength = prediction.confidence * 100;
    
    // Adjust based on volume
    if (features.volume > 1000000) strength += 10; // High volume boost
    if (features.volume < 10000) strength -= 10;   // Low volume penalty
    
    // Adjust based on volatility
    if (features.volatility > 0.5) strength += 5;  // High volatility opportunity
    if (features.volatility < 0.1) strength -= 5;  // Low volatility penalty
    
    // Adjust based on technical indicators alignment
    const technicalAlignment = this.calculateTechnicalAlignment(prediction.action, features);
    strength += technicalAlignment * 20;
    
    return Math.max(0, Math.min(100, strength));
  }
  
  /**
   * Calculate technical indicator alignment with prediction
   */
  private calculateTechnicalAlignment(action: string, features: FeatureVector): number {
    let alignment = 0;
    let indicators = 0;
    
    // RSI alignment
    if (action === 'buy' && features.rsi < 40) alignment++; // Oversold
    if (action === 'sell' && features.rsi > 60) alignment++; // Overbought
    indicators++;
    
    // MACD alignment
    if (action === 'buy' && features.macd > features.macdSignal) alignment++;
    if (action === 'sell' && features.macd < features.macdSignal) alignment++;
    indicators++;
    
    // Bollinger Bands alignment
    if (action === 'buy' && features.bollingerPosition < 0.2) alignment++; // Near lower band
    if (action === 'sell' && features.bollingerPosition > 0.8) alignment++; // Near upper band
    indicators++;
    
    return indicators > 0 ? alignment / indicators : 0;
  }
  
  /**
   * Calculate risk score for the signal
   */
  private calculateRiskScore(prediction: any, features: FeatureVector, config: StrategyConfig): number {
    let risk = 50; // Base risk
    
    // Volatility risk
    risk += features.volatility * 30;
    
    // Confidence risk (lower confidence = higher risk)
    risk += (1 - prediction.confidence) * 25;
    
    // Volume risk
    if (features.volume < 10000) risk += 15; // Low liquidity
    
    // Technical risk
    const technicalRisk = this.calculateTechnicalRisk(features);
    risk += technicalRisk * 10;
    
    return Math.max(0, Math.min(100, risk));
  }
  
  /**
   * Calculate technical indicator risk
   */
  private calculateTechnicalRisk(features: FeatureVector): number {
    let risk = 0;
    
    // Extreme RSI values
    if (features.rsi > 80 || features.rsi < 20) risk += 0.5;
    
    // Wide bid-ask spread
    if (features.bidAskSpread / features.price > 0.001) risk += 0.3;
    
    // High volatility
    if (features.volatility > 1.0) risk += 0.4;
    
    return Math.min(1, risk);
  }
  
  /**
   * Estimate expected return
   */
  private estimateExpectedReturn(prediction: any, features: FeatureVector, config: StrategyConfig): number {
    const baseReturn = 0.02; // 2% base expected return
    const confidenceMultiplier = prediction.confidence;
    const volatilityAdjustment = features.volatility * 0.5;
    
    return baseReturn * confidenceMultiplier * (1 + volatilityAdjustment);
  }
  
  /**
   * Estimate maximum drawdown
   */
  private estimateMaxDrawdown(prediction: any, features: FeatureVector, config: StrategyConfig): number {
    const baseDrawdown = config.risk_tolerance * 0.1;
    const volatilityAdjustment = features.volatility;
    const confidenceAdjustment = 1 - prediction.confidence;
    
    return baseDrawdown * (1 + volatilityAdjustment + confidenceAdjustment);
  }
  
  /**
   * Calculate win probability
   */
  private calculateWinProbability(prediction: any, config: StrategyConfig): number {
    const performance = this.modelPerformance.get(config.id);
    const baseWinRate = performance?.winRate || 0.5;
    
    // Adjust based on confidence
    const confidenceAdjustment = (prediction.confidence - 0.5) * 0.3;
    
    return Math.max(0.1, Math.min(0.9, baseWinRate + confidenceAdjustment));
  }
  
  /**
   * Identify chart patterns
   */
  private identifyPattern(features: FeatureVector): string {
    // Simplified pattern identification
    if (features.rsi < 30 && features.bollingerPosition < 0.2) {
      return 'oversold_reversal';
    }
    if (features.rsi > 70 && features.bollingerPosition > 0.8) {
      return 'overbought_reversal';
    }
    if (features.macd > features.macdSignal && features.rsi > 50) {
      return 'bullish_momentum';
    }
    if (features.macd < features.macdSignal && features.rsi < 50) {
      return 'bearish_momentum';
    }
    
    return 'ranging';
  }
  
  /**
   * Update model performance latency
   */
  private updateModelLatency(strategyId: string, latency: number): void {
    const performance = this.modelPerformance.get(strategyId);
    if (performance) {
      performance.predictionLatency = latency;
      performance.lastUpdated = Date.now();
    }
  }
  
  /**
   * Start real-time processing queue
   */
  private startRealtimeProcessing(): void {
    this.realtimeProcessor = setInterval(() => {
      this.processQueuedData();
    }, 100); // Process every 100ms for sub-second latency
  }
  
  /**
   * Process queued market data
   */
  private async processQueuedData(): Promise<void> {
    if (this.processingQueue.length === 0) return;
    
    const batch = this.processingQueue.splice(0, 10); // Process in batches
    
    for (const item of batch) {
      try {
        await this.processMarketData(
          item.symbol,
          item.data.price,
          item.data.volume,
          item.data.orderBook
        );
      } catch (error) {
        console.error('Batch processing error:', error);
      }
    }
  }
  
  /**
   * Queue market data for processing
   */
  queueMarketData(symbol: string, data: any): void {
    this.processingQueue.push({
      symbol,
      data,
      timestamp: Date.now()
    });
    
    // Prevent queue overflow
    if (this.processingQueue.length > 1000) {
      this.processingQueue = this.processingQueue.slice(-500);
    }
  }
  
  /**
   * Get strategy performance metrics
   */
  getStrategyPerformance(strategyId: string): ModelPerformance | null {
    return this.modelPerformance.get(strategyId) || null;
  }
  
  /**
   * Get all active strategies
   */
  getActiveStrategies(): StrategyConfig[] {
    return Array.from(this.strategies.values()).filter(s => s.isActive);
  }
  
  /**
   * Update strategy configuration
   */
  updateStrategy(strategyId: string, updates: Partial<StrategyConfig>): void {
    const strategy = this.strategies.get(strategyId);
    if (strategy) {
      this.strategies.set(strategyId, { ...strategy, ...updates });
      console.log(`📊 Strategy updated: ${strategy.name}`);
      this.emit('strategyUpdated', strategyId, updates);
    }
  }
  
  /**
   * Shutdown the signal processor
   */
  shutdown(): void {
    if (this.realtimeProcessor) {
      clearInterval(this.realtimeProcessor);
      this.realtimeProcessor = null;
    }
    
    // Dispose of all models
    for (const [strategyId, model] of this.models) {
      model.dispose();
      console.log(`🗑️ Model disposed: ${strategyId}`);
    }
    
    this.models.clear();
    this.isInitialized = false;
    
    console.log('💀 Signal processor shutdown complete');
  }
}

// Export singleton instance
export const signalProcessor = new MLSignalProcessor();

// Initialize on import
signalProcessor.initialize().catch(error => {
  console.error('🚨 Failed to initialize signal processor:', error);
});

// Utility functions for React components
export const signalUtils = {
  /**
   * Format signal confidence as percentage
   */
  formatConfidence(confidence: number): string {
    return `${(confidence * 100).toFixed(1)}%`;
  },
  
  /**
   * Get signal color based on action and confidence
   */
  getSignalColor(action: string, confidence: number): string {
    const alpha = Math.max(0.6, confidence);
    
    switch (action) {
      case 'buy':
        return `rgba(0, 255, 136, ${alpha})`;
      case 'sell':
        return `rgba(255, 0, 102, ${alpha})`;
      default:
        return `rgba(255, 255, 255, ${alpha})`;
    }
  },
  
  /**
   * Get signal urgency level
   */
  getSignalUrgency(signal: MarketSignal): 'low' | 'medium' | 'high' | 'critical' {
    const timeLeft = signal.expiresAt - Date.now();
    const confidence = signal.confidence;
    
    if (timeLeft < 30000 && confidence > 0.8) return 'critical';
    if (timeLeft < 60000 && confidence > 0.7) return 'high';
    if (confidence > 0.6) return 'medium';
    return 'low';
  },
  
  /**
   * Calculate signal score for prioritization
   */
  calculateSignalScore(signal: MarketSignal): number {
    return signal.confidence * signal.strength * (signal.metadata.winProbability || 0.5);
  }
};
