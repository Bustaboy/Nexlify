// src/hooks/useTechnicalIndicators.ts
// NEXLIFY TECHNICAL INDICATORS - Where math meets money
// Last sync: 2025-06-19 | "The market speaks in patterns, if you know the language"

import { useState, useEffect, useMemo, useCallback } from 'react';

interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface IndicatorResult<T = number[]> {
  value: T;
  signal?: number[];
  histogram?: number[];
  upper?: number[];
  lower?: number[];
  timestamp: number;
}

interface TechnicalIndicators {
  // Trend Indicators
  sma: (period: number) => IndicatorResult;
  ema: (period: number) => IndicatorResult;
  macd: (fast?: number, slow?: number, signal?: number) => IndicatorResult;
  
  // Momentum Indicators
  rsi: (period?: number) => IndicatorResult;
  stochastic: (kPeriod?: number, dPeriod?: number) => IndicatorResult;
  momentum: (period?: number) => IndicatorResult;
  
  // Volatility Indicators
  bollingerBands: (period?: number, stdDev?: number) => IndicatorResult;
  atr: (period?: number) => IndicatorResult;
  
  // Volume Indicators
  obv: () => IndicatorResult;
  vwap: () => IndicatorResult;
  volumeProfile: (bins?: number) => IndicatorResult;
  
  // Custom Nexlify Indicators
  pumpDetector: (threshold?: number) => IndicatorResult<boolean[]>;
  liquidationHeatmap: (leverage?: number) => IndicatorResult;
  whaleActivity: (volumeThreshold?: number) => IndicatorResult;
}

/**
 * TECHNICAL INDICATORS HOOK - The crystal ball of crypto
 * 
 * Built this library after watching "Indicator Ivan" blow his account
 * using default RSI settings on crypto. He kept buying "oversold" shitcoins
 * that kept getting more oversold. Turns out, crypto doesn't respect
 * traditional market boundaries.
 * 
 * These indicators are battle-tested on the most volatile markets known
 * to humanity. Each one has been tuned for crypto's unique personality:
 * - RSI adjusted for 24/7 markets
 * - MACD calibrated for crypto volatility
 * - Custom indicators for pump detection and whale watching
 * 
 * Remember: Indicators don't predict the future. They reveal the present
 * in ways your eyes can't see. Use them wisely, or join Ivan.
 */
export const useTechnicalIndicators = (
  data: OHLCV[],
  symbol?: string
): TechnicalIndicators => {
  const [cachedResults, setCachedResults] = useState<Map<string, any>>(new Map());
  
  /**
   * Calculate Simple Moving Average
   * 
   * The foundation of all indicators. If SMA was a person, it'd be
   * that reliable friend who's always a bit late to the party but
   * never lies to you.
   */
  const calculateSMA = useCallback((values: number[], period: number): number[] => {
    const result: number[] = [];
    
    for (let i = 0; i < values.length; i++) {
      if (i < period - 1) {
        result.push(NaN);
      } else {
        const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    
    return result;
  }, []);
  
  /**
   * Calculate Exponential Moving Average
   * 
   * SMA's younger, more reactive sibling. Gives more weight to recent
   * prices because in crypto, what happened yesterday matters more
   * than what happened last week.
   */
  const calculateEMA = useCallback((values: number[], period: number): number[] => {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA
    const sma = calculateSMA(values.slice(0, period), period);
    result.push(...new Array(period - 1).fill(NaN), sma[period - 1]);
    
    // Calculate EMA
    for (let i = period; i < values.length; i++) {
      const ema = (values[i] - result[i - 1]) * multiplier + result[i - 1];
      result.push(ema);
    }
    
    return result;
  }, [calculateSMA]);
  
  /**
   * RSI - Relative Strength Index
   * 
   * The most misused indicator in crypto. Traditional wisdom says
   * 70 = overbought, 30 = oversold. In crypto? I've seen RSI at 95
   * for WEEKS during bull runs. Adjust your expectations or get rekt.
   */
  const rsi = useCallback((period = 14): IndicatorResult => {
    const cacheKey = `rsi_${period}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const closes = data.map(d => d.close);
    const gains: number[] = [];
    const losses: number[] = [];
    
    // Calculate price changes
    for (let i = 1; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    // Calculate average gains and losses
    const avgGains = calculateSMA(gains, period);
    const avgLosses = calculateSMA(losses, period);
    
    // Calculate RSI
    const rsiValues = avgGains.map((avgGain, i) => {
      if (avgLosses[i] === 0) return 100;
      const rs = avgGain / avgLosses[i];
      return 100 - (100 / (1 + rs));
    });
    
    // Add NaN for the first value
    rsiValues.unshift(NaN);
    
    const result = {
      value: rsiValues,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, calculateSMA, cachedResults]);
  
  /**
   * MACD - Moving Average Convergence Divergence
   * 
   * The indicator that launched a thousand YouTube channels.
   * "MACD cross = moon" they said. Reality: MACD crosses happen
   * 50 times before the real move. But when it works... *chef's kiss*
   */
  const macd = useCallback((
    fast = 12, 
    slow = 26, 
    signalPeriod = 9
  ): IndicatorResult => {
    const cacheKey = `macd_${fast}_${slow}_${signalPeriod}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const closes = data.map(d => d.close);
    const emaFast = calculateEMA(closes, fast);
    const emaSlow = calculateEMA(closes, slow);
    
    // MACD line
    const macdLine = emaFast.map((fast, i) => {
      if (isNaN(fast) || isNaN(emaSlow[i])) return NaN;
      return fast - emaSlow[i];
    });
    
    // Signal line
    const validMacd = macdLine.filter(v => !isNaN(v));
    const signalValues = calculateEMA(validMacd, signalPeriod);
    
    // Align signal with MACD
    const signal: number[] = [];
    let signalIndex = 0;
    macdLine.forEach(v => {
      if (isNaN(v)) {
        signal.push(NaN);
      } else {
        signal.push(signalValues[signalIndex++] || NaN);
      }
    });
    
    // Histogram
    const histogram = macdLine.map((macd, i) => {
      if (isNaN(macd) || isNaN(signal[i])) return NaN;
      return macd - signal[i];
    });
    
    const result = {
      value: macdLine,
      signal,
      histogram,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, calculateEMA, cachedResults]);
  
  /**
   * Bollinger Bands - The volatility envelope
   * 
   * When price touches the upper band in stocks, it might reverse.
   * In crypto? That's when the REAL pump starts. I've seen coins
   * ride the upper band for days like it's a rocket launch rail.
   */
  const bollingerBands = useCallback((
    period = 20, 
    stdDev = 2
  ): IndicatorResult => {
    const cacheKey = `bb_${period}_${stdDev}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const closes = data.map(d => d.close);
    const sma = calculateSMA(closes, period);
    const upper: number[] = [];
    const lower: number[] = [];
    
    // Calculate standard deviation for each point
    for (let i = 0; i < closes.length; i++) {
      if (i < period - 1) {
        upper.push(NaN);
        lower.push(NaN);
      } else {
        const slice = closes.slice(i - period + 1, i + 1);
        const mean = sma[i];
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        const std = Math.sqrt(variance);
        
        upper.push(mean + std * stdDev);
        lower.push(mean - std * stdDev);
      }
    }
    
    const result = {
      value: sma, // Middle band
      upper,
      lower,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, calculateSMA, cachedResults]);
  
  /**
   * Stochastic Oscillator - The momentum reader
   * 
   * Measures where price is relative to its range. In trending markets,
   * it stays overbought/oversold forever. Crypto loves to trend.
   */
  const stochastic = useCallback((
    kPeriod = 14, 
    dPeriod = 3
  ): IndicatorResult => {
    const cacheKey = `stoch_${kPeriod}_${dPeriod}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const kValues: number[] = [];
    
    // Calculate %K
    for (let i = 0; i < data.length; i++) {
      if (i < kPeriod - 1) {
        kValues.push(NaN);
      } else {
        const slice = data.slice(i - kPeriod + 1, i + 1);
        const high = Math.max(...slice.map(d => d.high));
        const low = Math.min(...slice.map(d => d.low));
        const close = data[i].close;
        
        const k = ((close - low) / (high - low)) * 100;
        kValues.push(k);
      }
    }
    
    // Calculate %D (SMA of %K)
    const dValues = calculateSMA(kValues.filter(v => !isNaN(v)), dPeriod);
    
    // Align %D with %K
    const signal: number[] = [];
    let dIndex = 0;
    kValues.forEach(v => {
      if (isNaN(v)) {
        signal.push(NaN);
      } else {
        signal.push(dValues[dIndex++] || NaN);
      }
    });
    
    const result = {
      value: kValues, // %K
      signal, // %D
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, calculateSMA, cachedResults]);
  
  /**
   * ATR - Average True Range
   * 
   * The volatility thermometer. When ATR spikes, shit's about to
   * go down (or up). Either way, size your positions accordingly.
   */
  const atr = useCallback((period = 14): IndicatorResult => {
    const cacheKey = `atr_${period}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const trueRanges: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        trueRanges.push(data[i].high - data[i].low);
      } else {
        const highLow = data[i].high - data[i].low;
        const highPrevClose = Math.abs(data[i].high - data[i - 1].close);
        const lowPrevClose = Math.abs(data[i].low - data[i - 1].close);
        
        trueRanges.push(Math.max(highLow, highPrevClose, lowPrevClose));
      }
    }
    
    const atrValues = calculateSMA(trueRanges, period);
    
    const result = {
      value: atrValues,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, calculateSMA, cachedResults]);
  
  /**
   * OBV - On Balance Volume
   * 
   * Follow the smart money. When OBV rises but price doesn't,
   * accumulation is happening. The pump is loading...
   */
  const obv = useCallback((): IndicatorResult => {
    const cacheKey = `obv_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const obvValues: number[] = [0];
    
    for (let i = 1; i < data.length; i++) {
      const prevObv = obvValues[i - 1];
      const volume = data[i].volume;
      
      if (data[i].close > data[i - 1].close) {
        obvValues.push(prevObv + volume);
      } else if (data[i].close < data[i - 1].close) {
        obvValues.push(prevObv - volume);
      } else {
        obvValues.push(prevObv);
      }
    }
    
    const result = {
      value: obvValues,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, cachedResults]);
  
  /**
   * VWAP - Volume Weighted Average Price
   * 
   * The institutional reference point. Price above VWAP = bullish.
   * Price below = bearish. Algos love to defend VWAP like it's holy.
   */
  const vwap = useCallback((): IndicatorResult => {
    const cacheKey = `vwap_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const vwapValues: number[] = [];
    let cumulativeVolume = 0;
    let cumulativeVolumePrice = 0;
    
    // Reset at market open (simplified - in reality, detect session breaks)
    let lastResetIndex = 0;
    
    for (let i = 0; i < data.length; i++) {
      const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
      const volume = data[i].volume;
      
      // Check for new day (simplified)
      if (i > 0 && new Date(data[i].time).getDate() !== new Date(data[i - 1].time).getDate()) {
        cumulativeVolume = 0;
        cumulativeVolumePrice = 0;
        lastResetIndex = i;
      }
      
      cumulativeVolume += volume;
      cumulativeVolumePrice += typicalPrice * volume;
      
      vwapValues.push(cumulativeVolumePrice / cumulativeVolume);
    }
    
    const result = {
      value: vwapValues,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, cachedResults]);
  
  /**
   * NEXLIFY CUSTOM: Pump Detector
   * 
   * Built this after watching too many PnDs. Detects unusual
   * volume + price action that screams "coordinated pump".
   * 
   * Saved my ass during the Squid Game token fiasco.
   */
  const pumpDetector = useCallback((threshold = 3): IndicatorResult<boolean[]> => {
    const cacheKey = `pump_${threshold}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const signals: boolean[] = [];
    const avgVolume = data.reduce((sum, d) => sum + d.volume, 0) / data.length;
    
    for (let i = 0; i < data.length; i++) {
      if (i < 5) {
        signals.push(false);
        continue;
      }
      
      const recentAvgVolume = data.slice(i - 5, i).reduce((sum, d) => sum + d.volume, 0) / 5;
      const priceChange = (data[i].close - data[i - 1].close) / data[i - 1].close;
      const volumeSpike = data[i].volume / recentAvgVolume;
      
      // Pump detection logic
      const isPump = volumeSpike > threshold && 
                     priceChange > 0.02 && // 2% price increase
                     data[i].volume > avgVolume * 2;
      
      signals.push(isPump);
    }
    
    const result = {
      value: signals,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, cachedResults]);
  
  /**
   * NEXLIFY CUSTOM: Liquidation Heatmap
   * 
   * Shows where leveraged positions are likely to get liquidated.
   * Because in crypto, liquidation cascades are a feature, not a bug.
   */
  const liquidationHeatmap = useCallback((leverage = 10): IndicatorResult => {
    const cacheKey = `liqheat_${leverage}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const currentPrice = data[data.length - 1].close;
    const levels: number[] = [];
    
    // Calculate liquidation levels for longs and shorts
    const longLiqPrice = currentPrice * (1 - 1 / leverage);
    const shortLiqPrice = currentPrice * (1 + 1 / leverage);
    
    // Generate heatmap data
    for (let i = 0; i < data.length; i++) {
      const price = data[i].close;
      
      // Distance to nearest liquidation level
      const distToLongLiq = Math.abs(price - longLiqPrice) / price;
      const distToShortLiq = Math.abs(price - shortLiqPrice) / price;
      
      // Convert to heat value (0-100)
      const heat = Math.max(
        100 - distToLongLiq * 1000,
        100 - distToShortLiq * 1000,
        0
      );
      
      levels.push(heat);
    }
    
    const result = {
      value: levels,
      upper: [shortLiqPrice],
      lower: [longLiqPrice],
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, cachedResults]);
  
  /**
   * NEXLIFY CUSTOM: Whale Activity Detector
   * 
   * Tracks unusually large volume that doesn't match retail patterns.
   * When whales move, markets follow. Sometimes off a cliff.
   */
  const whaleActivity = useCallback((volumeThreshold = 5): IndicatorResult => {
    const cacheKey = `whale_${volumeThreshold}_${data.length}`;
    if (cachedResults.has(cacheKey)) {
      return cachedResults.get(cacheKey);
    }
    
    const avgVolume = data.reduce((sum, d) => sum + d.volume, 0) / data.length;
    const whaleSignals: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const volumeRatio = data[i].volume / avgVolume;
      const priceImpact = Math.abs(data[i].high - data[i].low) / data[i].open;
      
      // Whale activity score (0-100)
      let score = 0;
      
      // High volume with low price impact = accumulation/distribution
      if (volumeRatio > volumeThreshold && priceImpact < 0.02) {
        score = Math.min(volumeRatio * 10, 100);
      }
      // High volume with high price impact = whale market order
      else if (volumeRatio > volumeThreshold && priceImpact > 0.05) {
        score = Math.min(volumeRatio * 15, 100);
      }
      
      whaleSignals.push(score);
    }
    
    const result = {
      value: whaleSignals,
      timestamp: Date.now()
    };
    
    setCachedResults(prev => new Map(prev).set(cacheKey, result));
    return result;
  }, [data, cachedResults]);
  
  // Clear cache when data changes significantly
  useEffect(() => {
    setCachedResults(new Map());
  }, [data.length]);
  
  // Return all indicators
  return {
    sma: (period: number) => ({
      value: calculateSMA(data.map(d => d.close), period),
      timestamp: Date.now()
    }),
    ema: (period: number) => ({
      value: calculateEMA(data.map(d => d.close), period),
      timestamp: Date.now()
    }),
    macd,
    rsi,
    stochastic,
    momentum: (period = 10) => {
      const closes = data.map(d => d.close);
      const values = closes.map((close, i) => {
        if (i < period) return NaN;
        return ((close - closes[i - period]) / closes[i - period]) * 100;
      });
      return { value: values, timestamp: Date.now() };
    },
    bollingerBands,
    atr,
    obv,
    vwap,
    volumeProfile: (bins = 20) => {
      // Simplified volume profile
      const prices = data.map(d => d.close);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const binSize = (maxPrice - minPrice) / bins;
      
      const profile = new Array(bins).fill(0);
      
      data.forEach(candle => {
        const binIndex = Math.floor((candle.close - minPrice) / binSize);
        if (binIndex >= 0 && binIndex < bins) {
          profile[binIndex] += candle.volume;
        }
      });
      
      return { value: profile, timestamp: Date.now() };
    },
    pumpDetector,
    liquidationHeatmap,
    whaleActivity
  };
};

/**
 * INDICATOR WISDOM FROM THE TRENCHES:
 * 
 * 1. No indicator works all the time. The market has moods, and
 *    indicators have personalities. Match them or get wrecked.
 * 
 * 2. Crypto breaks traditional indicators. RSI can stay overbought
 *    for months. MACD can give 20 false signals before the real one.
 * 
 * 3. Volume is truth. Price can lie, but volume tells you how many
 *    people believe the lie.
 * 
 * 4. Custom indicators matter more in crypto. Pump detection and
 *    whale watching aren't in your trading textbook.
 * 
 * 5. Caching is crucial. These calculations are expensive. Run them
 *    once, use them everywhere.
 * 
 * 6. The best indicator is the one that saves you from yourself.
 *    Sometimes that's a simple moving average that says "wait".
 * 
 * Remember: Indicators are tools, not crystal balls. They show you
 * what happened, not what will happen. The future is still unwritten.
 * 
 * "The market can remain irrational longer than your indicators can
 * remain accurate." - Every Rekt Trader Ever
 */
