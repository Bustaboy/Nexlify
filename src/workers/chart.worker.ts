// src/workers/chart.worker.ts
// NEXLIFY CHART WORKER - The shadow thread that never sleeps
// Last sync: 2025-06-19 | "Main thread for trading, worker thread for thinking"

import { binaryIPC, MessageType } from '@/lib/binary-ipc';

// Worker message types
export interface WorkerMessage {
  id: string;
  type: 'process_candles' | 'calculate_indicators' | 'render_heatmap' | 
        'compress_data' | 'detect_patterns' | 'stream_update';
  data: any;
  timestamp: number;
}

export interface WorkerResponse {
  id: string;
  type: WorkerMessage['type'];
  result?: any;
  error?: string;
  processingTime: number;
  memoryUsed?: number;
}

// Candle data structure
interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Pattern types we detect
interface Pattern {
  type: 'breakout' | 'breakdown' | 'triangle' | 'flag' | 'channel' | 
        'double_top' | 'double_bottom' | 'head_shoulders';
  confidence: number;
  startIndex: number;
  endIndex: number;
  targetPrice?: number;
  stopLoss?: number;
}

/**
 * CHART WORKER - The unsung hero of smooth trading
 * 
 * December 2021. Bitcoin flash crash. 10,000 candles updating per second.
 * Main thread frozen. UI unresponsive. Trying to close positions but
 * clicks weren't registering. Chart calculations were blocking everything.
 * 
 * Watched $30k evaporate because the FUCKING CHART was too busy drawing
 * pretty lines to let me click the sell button.
 * 
 * Never. Again.
 * 
 * This worker handles all heavy lifting in a separate thread:
 * - Indicator calculations (RSI, MACD, etc.)
 * - Pattern detection (because humans see patterns, machines find them)
 * - Data compression (10,000 candles → 1,000 visual points)
 * - Heatmap generation (liquidation levels, volume profiles)
 * 
 * Main thread stays free. Clicks always work. Orders always execute.
 */

// Performance monitoring
let lastGC = Date.now();
const performanceStats = {
  tasksProcessed: 0,
  totalProcessingTime: 0,
  peakMemoryUsage: 0,
  errors: 0
};

/**
 * Process incoming messages - the command center
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const startTime = performance.now();
  const { id, type, data } = event.data;
  
  try {
    let result: any;
    
    switch (type) {
      case 'process_candles':
        result = await processCandles(data);
        break;
        
      case 'calculate_indicators':
        result = calculateIndicators(data);
        break;
        
      case 'detect_patterns':
        result = detectPatterns(data);
        break;
        
      case 'render_heatmap':
        result = renderHeatmap(data);
        break;
        
      case 'compress_data':
        result = compressData(data);
        break;
        
      case 'stream_update':
        result = handleStreamUpdate(data);
        break;
        
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
    
    const processingTime = performance.now() - startTime;
    performanceStats.tasksProcessed++;
    performanceStats.totalProcessingTime += processingTime;
    
    // Check memory usage
    if ('memory' in performance) {
      const memoryUsed = (performance as any).memory.usedJSHeapSize;
      performanceStats.peakMemoryUsage = Math.max(
        performanceStats.peakMemoryUsage, 
        memoryUsed
      );
      
      // Garbage collection hint if memory is high
      if (Date.now() - lastGC > 30000 && memoryUsed > 100 * 1024 * 1024) {
        if (global.gc) global.gc();
        lastGC = Date.now();
      }
    }
    
    const response: WorkerResponse = {
      id,
      type,
      result,
      processingTime,
      memoryUsed: 'memory' in performance ? 
        (performance as any).memory.usedJSHeapSize : undefined
    };
    
    self.postMessage(response);
    
  } catch (error: any) {
    performanceStats.errors++;
    
    const response: WorkerResponse = {
      id,
      type,
      error: error.message || 'Unknown error',
      processingTime: performance.now() - startTime
    };
    
    self.postMessage(response);
  }
};

/**
 * Process candles - the raw data massage
 * 
 * Exchanges send candles in all sorts of fucked up formats.
 * This normalizes everything into clean, tradeable data.
 */
async function processCandles(data: {
  candles: any[],
  timeframe: string,
  symbol: string
}): Promise<{ processed: OHLCV[], stats: any }> {
  const processed: OHLCV[] = [];
  const stats = {
    gaps: 0,
    outliers: 0,
    volumeSpikes: 0,
    priceRange: { min: Infinity, max: -Infinity }
  };
  
  // Calculate expected interval
  const intervals: { [key: string]: number } = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
  };
  const expectedInterval = intervals[data.timeframe] || 60000;
  
  // Process each candle
  for (let i = 0; i < data.candles.length; i++) {
    const candle = data.candles[i];
    
    // Normalize the candle structure
    const normalized: OHLCV = {
      time: typeof candle.time === 'string' ? Date.parse(candle.time) : candle.time,
      open: parseFloat(candle.open || candle.o),
      high: parseFloat(candle.high || candle.h),
      low: parseFloat(candle.low || candle.l),
      close: parseFloat(candle.close || candle.c),
      volume: parseFloat(candle.volume || candle.v || 0)
    };
    
    // Validation - because exchanges lie
    if (normalized.high < normalized.low) {
      [normalized.high, normalized.low] = [normalized.low, normalized.high];
    }
    
    if (normalized.open > normalized.high) normalized.open = normalized.high;
    if (normalized.open < normalized.low) normalized.open = normalized.low;
    if (normalized.close > normalized.high) normalized.close = normalized.high;
    if (normalized.close < normalized.low) normalized.close = normalized.low;
    
    // Gap detection
    if (i > 0) {
      const timeDiff = normalized.time - processed[i - 1].time;
      if (timeDiff > expectedInterval * 1.5) {
        stats.gaps++;
        
        // Fill gaps with interpolated candles
        const gapCandles = Math.floor(timeDiff / expectedInterval) - 1;
        for (let j = 1; j <= gapCandles && j < 100; j++) { // Max 100 to prevent memory explosion
          const ratio = j / (gapCandles + 1);
          const interpolated: OHLCV = {
            time: processed[i - 1].time + expectedInterval * j,
            open: processed[i - 1].close,
            high: processed[i - 1].close + (normalized.high - processed[i - 1].close) * ratio,
            low: processed[i - 1].close + (normalized.low - processed[i - 1].close) * ratio,
            close: processed[i - 1].close + (normalized.open - processed[i - 1].close) * ratio,
            volume: 0 // No volume for interpolated candles
          };
          processed.push(interpolated);
        }
      }
    }
    
    // Outlier detection - the "fat finger" filter
    if (i > 0) {
      const priceChange = Math.abs(normalized.close - processed[i - 1].close) / processed[i - 1].close;
      if (priceChange > 0.1) { // 10% in one candle = suspicious
        stats.outliers++;
      }
    }
    
    // Volume spike detection
    if (i > 20) {
      const recentVolumes = processed.slice(-20).map(c => c.volume);
      const avgVolume = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length;
      if (normalized.volume > avgVolume * 5) {
        stats.volumeSpikes++;
      }
    }
    
    // Update price range
    stats.priceRange.min = Math.min(stats.priceRange.min, normalized.low);
    stats.priceRange.max = Math.max(stats.priceRange.max, normalized.high);
    
    processed.push(normalized);
  }
  
  return { processed, stats };
}

/**
 * Calculate indicators - where math meets money
 * 
 * All the classics, optimized for speed. Remember: indicators
 * don't predict the future, they just make the past look smart.
 */
function calculateIndicators(data: {
  candles: OHLCV[],
  indicators: string[],
  params?: any
}): any {
  const results: any = {};
  const { candles, indicators, params = {} } = data;
  
  // Pre-calculate common values
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const volumes = candles.map(c => c.volume);
  
  indicators.forEach(indicator => {
    switch (indicator) {
      case 'sma':
        results.sma = calculateSMA(closes, params.sma?.period || 20);
        break;
        
      case 'ema':
        results.ema = calculateEMA(closes, params.ema?.period || 20);
        break;
        
      case 'rsi':
        results.rsi = calculateRSI(closes, params.rsi?.period || 14);
        break;
        
      case 'macd':
        results.macd = calculateMACD(
          closes,
          params.macd?.fast || 12,
          params.macd?.slow || 26,
          params.macd?.signal || 9
        );
        break;
        
      case 'bollinger':
        results.bollinger = calculateBollingerBands(
          closes,
          params.bollinger?.period || 20,
          params.bollinger?.stdDev || 2
        );
        break;
        
      case 'volume_profile':
        results.volumeProfile = calculateVolumeProfile(
          candles,
          params.volumeProfile?.bins || 24
        );
        break;
        
      case 'support_resistance':
        results.supportResistance = findSupportResistance(
          candles,
          params.supportResistance?.sensitivity || 0.02
        );
        break;
    }
  });
  
  return results;
}

/**
 * Pattern detection - finding order in chaos
 * 
 * The market loves patterns. Triangles, flags, channels...
 * Sometimes they work, sometimes they're just pretty shapes.
 * But when they work? *chef's kiss*
 */
function detectPatterns(data: {
  candles: OHLCV[],
  minConfidence?: number,
  lookback?: number
}): Pattern[] {
  const { candles, minConfidence = 0.7, lookback = 100 } = data;
  const patterns: Pattern[] = [];
  
  // Only analyze recent data for performance
  const startIndex = Math.max(0, candles.length - lookback);
  const recentCandles = candles.slice(startIndex);
  
  // Triangle pattern detection
  const triangles = detectTrianglePatterns(recentCandles);
  patterns.push(...triangles.filter(p => p.confidence >= minConfidence));
  
  // Channel detection (parallel support/resistance)
  const channels = detectChannels(recentCandles);
  patterns.push(...channels.filter(p => p.confidence >= minConfidence));
  
  // Double top/bottom
  const doubles = detectDoubleTopsBottoms(recentCandles);
  patterns.push(...doubles.filter(p => p.confidence >= minConfidence));
  
  // Breakout detection - the money makers
  const breakouts = detectBreakouts(recentCandles);
  patterns.push(...breakouts.filter(p => p.confidence >= minConfidence));
  
  // Adjust indices to account for slice
  patterns.forEach(p => {
    p.startIndex += startIndex;
    p.endIndex += startIndex;
  });
  
  return patterns;
}

/**
 * Render heatmap - visualizing the invisible
 * 
 * Liquidation levels, volume clusters, order density...
 * The stuff that moves markets but isn't on the chart.
 */
function renderHeatmap(data: {
  candles: OHLCV[],
  type: 'liquidation' | 'volume' | 'volatility',
  resolution: number,
  params?: any
}): Float32Array {
  const { candles, type, resolution, params = {} } = data;
  
  // Determine price range
  const prices = candles.map(c => [c.high, c.low]).flat();
  const minPrice = Math.min(...prices) * 0.95;
  const maxPrice = Math.max(...prices) * 1.05;
  const priceStep = (maxPrice - minPrice) / resolution;
  
  // Initialize heatmap
  const heatmap = new Float32Array(resolution);
  
  switch (type) {
    case 'liquidation':
      // Estimate liquidation levels based on common leverages
      const leverages = [2, 3, 5, 10, 20, 50, 100];
      const currentPrice = candles[candles.length - 1].close;
      
      leverages.forEach(leverage => {
        // Long liquidation
        const longLiqPrice = currentPrice * (1 - 0.9 / leverage);
        const longIndex = Math.floor((longLiqPrice - minPrice) / priceStep);
        if (longIndex >= 0 && longIndex < resolution) {
          heatmap[longIndex] += 100 / leverage; // Higher leverage = more heat
        }
        
        // Short liquidation
        const shortLiqPrice = currentPrice * (1 + 0.9 / leverage);
        const shortIndex = Math.floor((shortLiqPrice - minPrice) / priceStep);
        if (shortIndex >= 0 && shortIndex < resolution) {
          heatmap[shortIndex] += 100 / leverage;
        }
      });
      break;
      
    case 'volume':
      // Volume at price levels
      candles.forEach(candle => {
        const priceRange = candle.high - candle.low;
        const volumePerPrice = candle.volume / Math.max(1, priceRange / priceStep);
        
        for (let price = candle.low; price <= candle.high; price += priceStep) {
          const index = Math.floor((price - minPrice) / priceStep);
          if (index >= 0 && index < resolution) {
            heatmap[index] += volumePerPrice;
          }
        }
      });
      break;
      
    case 'volatility':
      // Price visit frequency = volatility proxy
      candles.forEach(candle => {
        const touches = [candle.open, candle.high, candle.low, candle.close];
        touches.forEach(price => {
          const index = Math.floor((price - minPrice) / priceStep);
          if (index >= 0 && index < resolution) {
            heatmap[index] += 1;
          }
        });
      });
      break;
  }
  
  // Normalize
  const maxHeat = Math.max(...heatmap);
  if (maxHeat > 0) {
    for (let i = 0; i < heatmap.length; i++) {
      heatmap[i] = (heatmap[i] / maxHeat) * 100;
    }
  }
  
  return heatmap;
}

/**
 * Data compression - because 100k candles will melt your GPU
 * 
 * Learned this the hard way during the 2017 bull run. Tried to
 * display 6 months of 1-minute candles. Chrome just said "nope"
 * and took the whole system with it.
 */
function compressData(data: {
  candles: OHLCV[],
  targetPoints: number,
  method: 'lttb' | 'simple' | 'peaks'
}): OHLCV[] {
  const { candles, targetPoints, method } = data;
  
  if (candles.length <= targetPoints) {
    return candles; // No compression needed
  }
  
  switch (method) {
    case 'lttb':
      // Largest Triangle Three Buckets - preserves visual shape
      return compressLTTB(candles, targetPoints);
      
    case 'peaks':
      // Keep local maxima/minima - preserves extremes
      return compressPeaks(candles, targetPoints);
      
    case 'simple':
    default:
      // Simple downsampling - fast but lossy
      const step = Math.floor(candles.length / targetPoints);
      return candles.filter((_, i) => i % step === 0);
  }
}

/**
 * LTTB compression - the smart way to downsample
 * 
 * Keeps the visual shape intact while reducing points.
 * Your eyes can't tell the difference, but your GPU can.
 */
function compressLTTB(candles: OHLCV[], targetPoints: number): OHLCV[] {
  if (targetPoints >= candles.length || targetPoints === 0) {
    return candles;
  }
  
  const compressed: OHLCV[] = [];
  const bucketSize = (candles.length - 2) / (targetPoints - 2);
  
  // Always keep first point
  compressed.push(candles[0]);
  
  let prevSelectedIndex = 0;
  
  for (let i = 1; i < targetPoints - 1; i++) {
    const bucketStart = Math.floor((i - 1) * bucketSize) + 1;
    const bucketEnd = Math.floor(i * bucketSize) + 1;
    
    // Calculate average for next bucket (for triangle area calculation)
    const nextBucketStart = Math.floor(i * bucketSize) + 1;
    const nextBucketEnd = Math.floor((i + 1) * bucketSize) + 1;
    
    let avgX = 0;
    let avgY = 0;
    let avgRangeLength = 0;
    
    for (let j = nextBucketStart; j < nextBucketEnd && j < candles.length; j++) {
      avgX += candles[j].time;
      avgY += candles[j].close;
      avgRangeLength++;
    }
    
    avgX /= avgRangeLength;
    avgY /= avgRangeLength;
    
    // Find point in bucket with largest triangle area
    let maxArea = -1;
    let selectedIndex = bucketStart;
    
    for (let j = bucketStart; j < bucketEnd && j < candles.length; j++) {
      // Triangle area with previous selected point and average next point
      const area = Math.abs(
        (candles[prevSelectedIndex].time - avgX) * (candles[j].close - candles[prevSelectedIndex].close) -
        (candles[prevSelectedIndex].time - candles[j].time) * (avgY - candles[prevSelectedIndex].close)
      );
      
      if (area > maxArea) {
        maxArea = area;
        selectedIndex = j;
      }
    }
    
    compressed.push(candles[selectedIndex]);
    prevSelectedIndex = selectedIndex;
  }
  
  // Always keep last point
  compressed.push(candles[candles.length - 1]);
  
  return compressed;
}

/**
 * Peak compression - keep the mountains and valleys
 */
function compressPeaks(candles: OHLCV[], targetPoints: number): OHLCV[] {
  const peaks: Array<{ index: number; importance: number }> = [];
  
  // Find all local extrema
  for (let i = 1; i < candles.length - 1; i++) {
    const prev = candles[i - 1].close;
    const curr = candles[i].close;
    const next = candles[i + 1].close;
    
    if ((curr > prev && curr > next) || (curr < prev && curr < next)) {
      // Calculate importance as price change magnitude
      const importance = Math.abs(curr - prev) + Math.abs(curr - next);
      peaks.push({ index: i, importance });
    }
  }
  
  // Sort by importance and keep top N
  peaks.sort((a, b) => b.importance - a.importance);
  const selectedIndices = peaks
    .slice(0, targetPoints - 2)
    .map(p => p.index)
    .sort((a, b) => a - b);
  
  // Build result with first, selected peaks, and last
  const compressed: OHLCV[] = [candles[0]];
  selectedIndices.forEach(i => compressed.push(candles[i]));
  compressed.push(candles[candles.length - 1]);
  
  return compressed;
}

/**
 * Stream update handler - real-time data flow
 * 
 * This is where we handle the firehose. Market data streaming
 * at the speed of light, processed without blocking the UI.
 */
function handleStreamUpdate(data: {
  update: any,
  currentData: OHLCV[],
  indicators?: string[]
}): any {
  const { update, currentData, indicators = [] } = data;
  
  // Apply update based on type
  let updated = [...currentData];
  
  if (update.type === 'new_candle') {
    updated.push(update.candle);
    
    // Maintain sliding window
    if (updated.length > 10000) {
      updated = updated.slice(-10000);
    }
  } else if (update.type === 'update_last') {
    if (updated.length > 0) {
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        ...update.changes
      };
    }
  }
  
  // Recalculate only affected indicators
  const results: any = { candles: updated };
  
  if (indicators.length > 0) {
    // Only calculate last N values for efficiency
    const lookback = 100;
    const recentCandles = updated.slice(-lookback);
    
    indicators.forEach(indicator => {
      // Quick calculation for streaming updates
      switch (indicator) {
        case 'sma':
          const closes = recentCandles.map(c => c.close);
          const sma = closes.slice(-20).reduce((a, b) => a + b, 0) / 20;
          results[indicator] = sma;
          break;
          
        case 'rsi':
          // Simplified RSI for last value only
          results[indicator] = quickRSI(recentCandles);
          break;
      }
    });
  }
  
  return results;
}

// Helper functions for indicators
function calculateSMA(values: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }
  
  return result;
}

function calculateEMA(values: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  const multiplier = 2 / (period + 1);
  
  // Start with SMA
  let ema = 0;
  for (let i = 0; i < period; i++) {
    ema += values[i];
  }
  ema /= period;
  
  // Fill with nulls
  for (let i = 0; i < period - 1; i++) {
    result.push(null);
  }
  result.push(ema);
  
  // Calculate EMA
  for (let i = period; i < values.length; i++) {
    ema = (values[i] - ema) * multiplier + ema;
    result.push(ema);
  }
  
  return result;
}

function calculateRSI(closes: number[], period = 14): (number | null)[] {
  const changes: number[] = [];
  const gains: number[] = [];
  const losses: number[] = [];
  
  // Calculate price changes
  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    changes.push(change);
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  
  const avgGains = calculateSMA(gains, period);
  const avgLosses = calculateSMA(losses, period);
  
  const result: (number | null)[] = [null]; // First value has no change
  
  for (let i = 0; i < avgGains.length; i++) {
    if (avgGains[i] === null || avgLosses[i] === null) {
      result.push(null);
    } else {
      const rs = avgGains[i]! / (avgLosses[i]! || 0.0001);
      const rsi = 100 - (100 / (1 + rs));
      result.push(rsi);
    }
  }
  
  return result;
}

function quickRSI(candles: OHLCV[], period = 14): number {
  if (candles.length < period + 1) return 50; // Neutral
  
  const closes = candles.slice(-period - 1).map(c => c.close);
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  const rs = avgGain / (avgLoss || 0.0001);
  
  return 100 - (100 / (1 + rs));
}

function calculateMACD(
  closes: number[],
  fastPeriod = 12,
  slowPeriod = 26,
  signalPeriod = 9
): { macd: (number | null)[], signal: (number | null)[], histogram: (number | null)[] } {
  const emaFast = calculateEMA(closes, fastPeriod);
  const emaSlow = calculateEMA(closes, slowPeriod);
  
  const macd: (number | null)[] = [];
  for (let i = 0; i < closes.length; i++) {
    if (emaFast[i] === null || emaSlow[i] === null) {
      macd.push(null);
    } else {
      macd.push(emaFast[i]! - emaSlow[i]!);
    }
  }
  
  const signal = calculateEMA(macd.filter(v => v !== null) as number[], signalPeriod);
  
  // Align signal with MACD
  const alignedSignal: (number | null)[] = [];
  let signalIndex = 0;
  for (let i = 0; i < macd.length; i++) {
    if (macd[i] === null) {
      alignedSignal.push(null);
    } else {
      alignedSignal.push(signal[signalIndex++] || null);
    }
  }
  
  const histogram: (number | null)[] = [];
  for (let i = 0; i < macd.length; i++) {
    if (macd[i] === null || alignedSignal[i] === null) {
      histogram.push(null);
    } else {
      histogram.push(macd[i]! - alignedSignal[i]!);
    }
  }
  
  return { macd, signal: alignedSignal, histogram };
}

function calculateBollingerBands(
  closes: number[],
  period = 20,
  stdDev = 2
): { upper: (number | null)[], middle: (number | null)[], lower: (number | null)[] } {
  const middle = calculateSMA(closes, period);
  const upper: (number | null)[] = [];
  const lower: (number | null)[] = [];
  
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      upper.push(null);
      lower.push(null);
    } else {
      const slice = closes.slice(i - period + 1, i + 1);
      const mean = middle[i]!;
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const std = Math.sqrt(variance);
      
      upper.push(mean + std * stdDev);
      lower.push(mean - std * stdDev);
    }
  }
  
  return { upper, middle, lower };
}

function calculateVolumeProfile(candles: OHLCV[], bins = 24): {
  levels: number[],
  volumes: number[],
  poc: number // Point of Control
} {
  const prices = candles.map(c => [c.high, c.low]).flat();
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const binSize = (maxPrice - minPrice) / bins;
  
  const profile = new Array(bins).fill(0);
  const levels = new Array(bins).fill(0).map((_, i) => minPrice + binSize * (i + 0.5));
  
  candles.forEach(candle => {
    const avgPrice = (candle.high + candle.low + candle.close) / 3;
    const binIndex = Math.floor((avgPrice - minPrice) / binSize);
    if (binIndex >= 0 && binIndex < bins) {
      profile[binIndex] += candle.volume;
    }
  });
  
  const maxVolumeIndex = profile.indexOf(Math.max(...profile));
  const poc = levels[maxVolumeIndex];
  
  return { levels, volumes: profile, poc };
}

function findSupportResistance(
  candles: OHLCV[],
  sensitivity = 0.02
): { support: number[], resistance: number[] } {
  const support: number[] = [];
  const resistance: number[] = [];
  
  // Find local minima (support) and maxima (resistance)
  for (let i = 10; i < candles.length - 10; i++) {
    const current = candles[i];
    const window = 10;
    
    // Check if local maximum
    let isResistance = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j !== i && candles[j].high > current.high) {
        isResistance = false;
        break;
      }
    }
    
    if (isResistance) {
      // Check if close to existing resistance
      const exists = resistance.some(r => Math.abs(r - current.high) / r < sensitivity);
      if (!exists) {
        resistance.push(current.high);
      }
    }
    
    // Check if local minimum
    let isSupport = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j !== i && candles[j].low < current.low) {
        isSupport = false;
        break;
      }
    }
    
    if (isSupport) {
      // Check if close to existing support
      const exists = support.some(s => Math.abs(s - current.low) / s < sensitivity);
      if (!exists) {
        support.push(current.low);
      }
    }
  }
  
  return { support, resistance };
}

// Pattern detection helpers
function detectTrianglePatterns(candles: OHLCV[]): Pattern[] {
  const patterns: Pattern[] = [];
  const minLength = 20;
  
  if (candles.length < minLength) return patterns;
  
  // Simplified triangle detection - look for converging highs and lows
  for (let start = 0; start < candles.length - minLength; start++) {
    const window = candles.slice(start, start + minLength);
    
    // Find peaks and troughs
    const peaks: number[] = [];
    const troughs: number[] = [];
    
    for (let i = 1; i < window.length - 1; i++) {
      if (window[i].high > window[i - 1].high && window[i].high > window[i + 1].high) {
        peaks.push(i);
      }
      if (window[i].low < window[i - 1].low && window[i].low < window[i + 1].low) {
        troughs.push(i);
      }
    }
    
    if (peaks.length >= 2 && troughs.length >= 2) {
      // Check if peaks are descending and troughs are ascending
      const peakSlope = (window[peaks[peaks.length - 1]].high - window[peaks[0]].high) / 
                       (peaks[peaks.length - 1] - peaks[0]);
      const troughSlope = (window[troughs[troughs.length - 1]].low - window[troughs[0]].low) / 
                         (troughs[troughs.length - 1] - troughs[0]);
      
      if (peakSlope < 0 && troughSlope > 0) {
        // Converging triangle detected
        const confidence = Math.min(
          1,
          (Math.abs(peakSlope) + Math.abs(troughSlope)) / 0.01
        );
        
        patterns.push({
          type: 'triangle',
          confidence,
          startIndex: start,
          endIndex: start + minLength - 1,
          targetPrice: window[window.length - 1].close * 1.05 // Simplified target
        });
      }
    }
  }
  
  return patterns;
}

function detectChannels(candles: OHLCV[]): Pattern[] {
  const patterns: Pattern[] = [];
  // Simplified channel detection
  // In production, would use linear regression for precise channels
  return patterns;
}

function detectDoubleTopsBottoms(candles: OHLCV[]): Pattern[] {
  const patterns: Pattern[] = [];
  const window = 50;
  const tolerance = 0.02; // 2% price similarity
  
  for (let i = window; i < candles.length - window; i++) {
    // Look for two similar peaks
    const firstPeak = Math.max(...candles.slice(i - window, i).map(c => c.high));
    const secondPeak = Math.max(...candles.slice(i, i + window).map(c => c.high));
    
    if (Math.abs(firstPeak - secondPeak) / firstPeak < tolerance) {
      // Potential double top
      patterns.push({
        type: 'double_top',
        confidence: 0.7,
        startIndex: i - window,
        endIndex: i + window,
        targetPrice: candles[i].close * 0.95
      });
    }
  }
  
  return patterns;
}

function detectBreakouts(candles: OHLCV[]): Pattern[] {
  const patterns: Pattern[] = [];
  const lookback = 20;
  
  for (let i = lookback; i < candles.length; i++) {
    const recent = candles.slice(i - lookback, i);
    const current = candles[i];
    
    const recentHigh = Math.max(...recent.map(c => c.high));
    const recentLow = Math.min(...recent.map(c => c.low));
    const avgVolume = recent.reduce((sum, c) => sum + c.volume, 0) / lookback;
    
    // Breakout detection
    if (current.close > recentHigh && current.volume > avgVolume * 2) {
      patterns.push({
        type: 'breakout',
        confidence: Math.min(1, current.volume / (avgVolume * 3)),
        startIndex: i - lookback,
        endIndex: i,
        targetPrice: current.close * 1.05,
        stopLoss: recentHigh
      });
    }
    
    // Breakdown detection
    if (current.close < recentLow && current.volume > avgVolume * 2) {
      patterns.push({
        type: 'breakdown',
        confidence: Math.min(1, current.volume / (avgVolume * 3)),
        startIndex: i - lookback,
        endIndex: i,
        targetPrice: current.close * 0.95,
        stopLoss: recentLow
      });
    }
  }
  
  return patterns;
}

// Performance monitoring
setInterval(() => {
  if (performanceStats.tasksProcessed > 0) {
    const avgProcessingTime = performanceStats.totalProcessingTime / performanceStats.tasksProcessed;
    console.log('Worker Performance:', {
      tasksProcessed: performanceStats.tasksProcessed,
      avgProcessingTime: avgProcessingTime.toFixed(2) + 'ms',
      peakMemory: (performanceStats.peakMemoryUsage / 1024 / 1024).toFixed(2) + 'MB',
      errorRate: ((performanceStats.errors / performanceStats.tasksProcessed) * 100).toFixed(2) + '%'
    });
  }
}, 30000); // Every 30 seconds

/**
 * CHART WORKER WISDOM:
 * 
 * 1. Web Workers saved my sanity during the 2021 bull run. Main thread
 *    for trading, worker thread for thinking. Never mix them.
 * 
 * 2. Pattern detection is 50% math, 50% art. The patterns that work
 *    in textbooks rarely work in crypto. Adapt or get rekt.
 * 
 * 3. Data compression is not optional when dealing with years of
 *    1-minute candles. LTTB is magic - looks the same, 90% less data.
 * 
 * 4. Indicators lag. By definition. By the time RSI says "overbought",
 *    the pump might be over. Or just beginning. Context matters.
 * 
 * 5. Memory management in workers is crucial. Garbage collection pauses
 *    can cause calculation delays. Monitor and clean proactively.
 * 
 * 6. Always validate exchange data. They send garbage more often than
 *    you'd think. Bad data = bad decisions = bad outcomes.
 * 
 * Remember: This worker is your analytical brain, separated from your
 * reactive brain (main thread). Let each do what it does best.
 * 
 * "The market is a parallel process. Your code should be too."
 * - Written during a 16-hour debugging session, 2022
 */
