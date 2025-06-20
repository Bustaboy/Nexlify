// src/components/charts/ChartContainer.tsx
// NEXLIFY WEBGL CHART ENGINE - Where patterns become prophecy
// Last sync: 2025-06-19 | "The market speaks in candlesticks and volume bars"

import { useEffect, useRef, useState, useCallback, memo } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  ColorType,
  CrosshairMode,
  LineStyle,
  PriceScaleMode
} from 'lightweight-charts';
import { motion } from 'framer-motion';
import { 
  Maximize2, 
  Minimize2, 
  TrendingUp, 
  Activity,
  Layers,
  Cpu,
  Zap
} from 'lucide-react';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore } from '@/stores/tradingStore';
import { useTechnicalIndicators } from '@/hooks/useTechnicalIndicators';
import { ChartWebGLRenderer } from './ChartWebGLRenderer';

interface ChartContainerProps {
  symbol: string;
  timeframe: string;
  height: string | number;
  theme?: 'dark' | 'neon' | 'matrix';
  fullscreen?: boolean;
  showIndicators?: boolean;
  onPriceClick?: (price: number) => void;
}

/**
 * CHART CONTAINER - The crystal ball of the digital age
 * 
 * This component? It's seen more price action than a Bangkok street market.
 * Built it after watching traders squint at TradingView on 13" laptops,
 * missing signals because they couldn't see shit.
 * 
 * WebGL rendering because Canvas is for amateurs. When BTC drops 10%
 * in 5 minutes, you need every frame. Every. Single. Frame.
 */
export const ChartContainer = memo(({
  symbol,
  timeframe,
  height,
  theme = 'neon',
  fullscreen = false,
  showIndicators = true,
  onPriceClick
}: ChartContainerProps) => {
  // Refs - our anchors to the DOM
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  
  // WebGL renderer for advanced graphics
  const webglRendererRef = useRef<ChartWebGLRenderer | null>(null);
  
  // State
  const [isLoading, setIsLoading] = useState(true);
  const [chartStats, setChartStats] = useState({
    fps: 60,
    dataPoints: 0,
    renderTime: 0
  });
  
  // Store connections
  const { candles, subscribeToMarket } = useMarketStore();
  const { positions, activeOrders } = useTradingStore();
  
  // Technical indicators hook
  const { 
    ema20, 
    ema50, 
    rsi, 
    macd, 
    bollinger,
    volume 
  } = useTechnicalIndicators(symbol, timeframe);
  
  /**
   * Theme configurations - because aesthetics matter in the sprawl
   */
  const themes = {
    dark: {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#d1d5db',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, monospace'
      },
      grid: {
        vertLines: { color: '#1f2937', style: LineStyle.Dashed },
        horzLines: { color: '#1f2937', style: LineStyle.Dashed }
      },
      candle: {
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444'
      }
    },
    neon: {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0f' },
        textColor: '#0dd9ff',
        fontSize: 11,
        fontFamily: 'Orbitron, monospace'
      },
      grid: {
        vertLines: { color: '#0dd9ff15', style: LineStyle.Solid },
        horzLines: { color: '#0dd9ff15', style: LineStyle.Solid }
      },
      candle: {
        upColor: '#00ff88',
        downColor: '#ff0066',
        borderUpColor: '#00ffaa',
        borderDownColor: '#ff0088',
        wickUpColor: '#00ff8855',
        wickDownColor: '#ff006655'
      }
    },
    matrix: {
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#00ff00',
        fontSize: 10,
        fontFamily: 'Courier New, monospace'
      },
      grid: {
        vertLines: { color: '#00ff0020', style: LineStyle.Dotted },
        horzLines: { color: '#00ff0020', style: LineStyle.Dotted }
      },
      candle: {
        upColor: '#00ff00',
        downColor: '#ff0000',
        borderUpColor: '#00ff00',
        borderDownColor: '#ff0000',
        wickUpColor: '#00ff0080',
        wickDownColor: '#ff000080'
      }
    }
  };
  
  /**
   * Initialize chart - birth of a visual oracle
   */
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    console.log(`📊 Initializing ${theme} chart for ${symbol}`);
    
    const currentTheme = themes[theme];
    
    // Create chart with WebGL renderer
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: currentTheme.layout,
      grid: currentTheme.grid,
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: {
          width: 1,
          color: currentTheme.layout.textColor,
          style: LineStyle.Dashed,
          labelBackgroundColor: theme === 'neon' ? '#0dd9ff' : currentTheme.candle.upColor
        },
        horzLine: {
          width: 1,
          color: currentTheme.layout.textColor,
          style: LineStyle.Dashed,
          labelBackgroundColor: theme === 'neon' ? '#0dd9ff' : currentTheme.candle.upColor
        }
      },
      rightPriceScale: {
        borderColor: currentTheme.grid.vertLines.color,
        scaleMargins: { top: 0.1, bottom: 0.2 }
      },
      timeScale: {
        borderColor: currentTheme.grid.horzLines.color,
        timeVisible: true,
        secondsVisible: false,
        barSpacing: 6,
        minBarSpacing: 3,
        fixLeftEdge: true,
        fixRightEdge: true,
        lockVisibleTimeRangeOnResize: true
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true
      }
    });
    
    // Create series
    const candleSeries = chart.addCandlestickSeries({
      upColor: currentTheme.candle.upColor,
      downColor: currentTheme.candle.downColor,
      borderUpColor: currentTheme.candle.borderUpColor,
      borderDownColor: currentTheme.candle.borderDownColor,
      wickUpColor: currentTheme.candle.wickUpColor,
      wickDownColor: currentTheme.candle.wickDownColor
    });
    
    // Volume series
    const volumeSeries = chart.addHistogramSeries({
      color: theme === 'neon' ? '#0dd9ff40' : '#6b728040',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      scaleMargins: { top: 0.8, bottom: 0 }
    });
    
    // Store refs
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    
    // Initialize WebGL renderer for advanced features
    if (window.WebGLRenderingContext) {
      webglRendererRef.current = new ChartWebGLRenderer(
        chartContainerRef.current,
        theme
      );
    }
    
    // Handle resize - responsive like a street fighter
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    // Click handler for price levels
    chart.subscribeClick((param) => {
      if (param.point && param.seriesPrices.size > 0) {
        const price = param.seriesPrices.values().next().value;
        if (onPriceClick && typeof price === 'number') {
          onPriceClick(price);
          
          // Visual feedback - the chrome responds
          if (webglRendererRef.current) {
            webglRendererRef.current.pulseAtPrice(price);
          }
        }
      }
    });
    
    setIsLoading(false);
    
    // Cleanup - leave no trace
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      webglRendererRef.current?.destroy();
    };
  }, [symbol, theme, onPriceClick]);
  
  /**
   * Update chart data - feeding the beast
   */
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
    
    // Get candle data for this symbol/timeframe
    const key = `${symbol}:${timeframe}`;
    const candleData = candles[key] || [];
    
    if (candleData.length === 0) {
      console.log(`📊 No candle data for ${key}, subscribing...`);
      subscribeToMarket(symbol, ['candles']);
      return;
    }
    
    // Transform data for TradingView format
    const tvData = candleData.map(candle => ({
      time: candle.timestamp.getTime() / 1000,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close
    }));
    
    const volumeData = candleData.map(candle => ({
      time: candle.timestamp.getTime() / 1000,
      value: candle.volume,
      color: candle.close >= candle.open 
        ? themes[theme].candle.upColor + '60'
        : themes[theme].candle.downColor + '60'
    }));
    
    // Update series
    candleSeriesRef.current.setData(tvData);
    volumeSeriesRef.current.setData(volumeData);
    
    // Update stats
    setChartStats(prev => ({
      ...prev,
      dataPoints: tvData.length
    }));
    
    console.log(`📊 Updated chart with ${tvData.length} candles`);
    
  }, [candles, symbol, timeframe, theme, subscribeToMarket]);
  
  /**
   * Draw positions and orders - marking our territory
   */
  useEffect(() => {
    if (!chartRef.current || !candleSeriesRef.current) return;
    
    const position = positions[symbol];
    const orders = Object.values(activeOrders).filter(o => o.symbol === symbol);
    
    // Clear existing lines
    candleSeriesRef.current.removeAllPriceLines();
    
    // Draw position line - where we stand
    if (position) {
      const profitColor = position.unrealizedPnl.gte(0) 
        ? themes[theme].candle.upColor 
        : themes[theme].candle.downColor;
      
      candleSeriesRef.current.createPriceLine({
        price: position.entryPrice.toNumber(),
        color: profitColor,
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Position: ${position.quantity} @ ${position.entryPrice}`
      });
      
      // Liquidation price if exists - the danger zone
      if (position.liquidationPrice) {
        candleSeriesRef.current.createPriceLine({
          price: position.liquidationPrice.toNumber(),
          color: '#ff0000',
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: 'LIQUIDATION'
        });
      }
    }
    
    // Draw order lines - our intentions
    orders.forEach(order => {
      if (order.price) {
        const color = order.side === 'buy' 
          ? themes[theme].candle.upColor 
          : themes[theme].candle.downColor;
        
        candleSeriesRef.current!.createPriceLine({
          price: order.price.toNumber(),
          color: color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: `${order.side.toUpperCase()} ${order.quantity}`
        });
      }
    });
    
  }, [positions, activeOrders, symbol, theme]);
  
  /**
   * Performance monitoring - because lag kills
   */
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    
    const measurePerformance = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        const renderTime = (currentTime - lastTime) / frameCount;
        
        setChartStats(prev => ({
          ...prev,
          fps,
          renderTime
        }));
        
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measurePerformance);
    };
    
    const handle = requestAnimationFrame(measurePerformance);
    return () => cancelAnimationFrame(handle);
  }, []);
  
  return (
    <div className="relative h-full w-full">
      {/* Loading state - the anticipation */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-50">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Cpu className="w-8 h-8 text-cyan-400" />
          </motion.div>
        </div>
      )}
      
      {/* Chart container */}
      <div 
        ref={chartContainerRef} 
        className="h-full w-full"
        style={{ opacity: isLoading ? 0 : 1, transition: 'opacity 0.3s' }}
      />
      
      {/* Overlay controls */}
      <div className="absolute top-4 left-4 flex items-center gap-2">
        {/* Timeframe selector */}
        <select
          value={timeframe}
          onChange={(e) => console.log('Timeframe change:', e.target.value)}
          className="bg-black/60 backdrop-blur-sm border border-cyan-800/50 rounded px-2 py-1 text-xs text-cyan-400 font-mono"
        >
          <option value="1m">1m</option>
          <option value="5m">5m</option>
          <option value="15m">15m</option>
          <option value="1h">1h</option>
          <option value="4h">4h</option>
          <option value="1d">1D</option>
        </select>
        
        {/* Indicator toggle */}
        <button
          onClick={() => console.log('Toggle indicators')}
          className="bg-black/60 backdrop-blur-sm border border-cyan-800/50 rounded p-1"
          title="Toggle Indicators"
        >
          <Layers className="w-4 h-4 text-cyan-400" />
        </button>
        
        {/* Fullscreen toggle */}
        <button
          onClick={() => console.log('Toggle fullscreen')}
          className="bg-black/60 backdrop-blur-sm border border-cyan-800/50 rounded p-1"
          title="Fullscreen"
        >
          {fullscreen ? 
            <Minimize2 className="w-4 h-4 text-cyan-400" /> : 
            <Maximize2 className="w-4 h-4 text-cyan-400" />
          }
        </button>
      </div>
      
      {/* Performance stats - for the data junkies */}
      <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm border border-cyan-800/50 rounded px-2 py-1 text-xs font-mono">
        <div className="flex items-center gap-3 text-cyan-400">
          <div className="flex items-center gap-1">
            <Activity className="w-3 h-3" />
            <span>{chartStats.fps} FPS</span>
          </div>
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span>{chartStats.renderTime.toFixed(1)}ms</span>
          </div>
          <div>
            {chartStats.dataPoints} candles
          </div>
        </div>
      </div>
      
      {/* Price click hint */}
      {onPriceClick && (
        <div className="absolute bottom-4 left-4 text-xs text-gray-500">
          Click on chart to set price levels
        </div>
      )}
    </div>
  );
});

ChartContainer.displayName = 'ChartContainer';

/**
 * CHART WISDOM (earned through screen-burned retinas):
 * 
 * 1. WebGL > Canvas. When shit hits the fan and everyone's
 *    refreshing charts, WebGL keeps you ahead of the pack.
 * 
 * 2. Those FPS stats? Not just for show. Below 30 FPS and
 *    you're trading on delayed information. Might as well
 *    use carrier pigeons.
 * 
 * 3. Price lines for positions are CRITICAL. I've seen
 *    traders forget they had positions because the chart
 *    didn't show them. Now they're reminded every tick.
 * 
 * 4. The liquidation line? That's not a warning, it's a
 *    promise. Respect it or the market will teach you.
 * 
 * 5. Click-to-set-price seems simple but it's saved more
 *    bad orders than any other feature. Fat fingers on
 *    number pads have cost fortunes.
 * 
 * Remember: The chart doesn't lie, but it doesn't tell
 * the whole truth either. It shows what was, not what
 * will be. Trade accordingly.
 */
