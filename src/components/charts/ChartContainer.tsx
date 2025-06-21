// src/components/charts/ChartContainer.tsx
// NEXLIFY WEBGL CHART ENGINE - Where patterns become prophecy
// Last sync: 2025-06-21 | "Time is money, and we've fixed both"

import { useEffect, useRef, useState, useCallback, memo } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  ColorType,
  CrosshairMode,
  LineStyle,
  PriceScaleMode,
  Time,
  UTCTimestamp,
  IPriceLine
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
 * Fixed the time paradox. TradingView wanted seconds, we were giving
 * milliseconds. Like showing up to a street race in a DeLorean set
 * to the wrong year. Now we're synced with the matrix.
 * 
 * Price lines? We track them ourselves now. Every position, every order,
 * stored in refs like ammunition clips. When the market moves, we
 * reload instantly.
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
  
  // Price line management - manual tracking because the API won't do it for us
  const priceLinesRef = useRef<IPriceLine[]>([]);
  
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
        textColor: '#9ca3af',
        fontSize: 11
      },
      grid: {
        vertLines: { color: '#1f2937', style: LineStyle.Solid },
        horzLines: { color: '#1f2937', style: LineStyle.Solid }
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
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#0dd9ff',
        fontSize: 11
      },
      grid: {
        vertLines: { color: '#0dd9ff20', style: LineStyle.Dotted },
        horzLines: { color: '#0dd9ff20', style: LineStyle.Dotted }
      },
      candle: {
        upColor: '#0dd9ff',
        downColor: '#ff0080',
        borderUpColor: '#0dd9ff',
        borderDownColor: '#ff0080',
        wickUpColor: '#0dd9ff80',
        wickDownColor: '#ff008080'
      }
    },
    matrix: {
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#00ff00',
        fontSize: 10
      },
      grid: {
        vertLines: { color: '#00ff0020', style: LineStyle.Solid },
        horzLines: { color: '#00ff0020', style: LineStyle.Solid }
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
   * Clear all price lines - nuclear option for chart cleanup
   */
  const clearAllPriceLines = useCallback(() => {
    priceLinesRef.current.forEach(line => {
      candleSeriesRef.current?.removePriceLine(line);
    });
    priceLinesRef.current = [];
  }, []);
  
  /**
   * Initialize chart - birth of the visualization
   */
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    const currentTheme = themes[theme];
    
    // Create the chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: currentTheme.layout,
      grid: currentTheme.grid,
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: theme === 'neon' ? '#0dd9ff' : currentTheme.candle.upColor,
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
      priceScaleId: 'volume'
    });
    
    // Store refs
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    
    // Initialize WebGL renderer for advanced features
    if (window.WebGLRenderingContext && chartContainerRef.current) {
      // Create a canvas element for WebGL renderer
      const canvas = document.createElement('canvas');
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.pointerEvents = 'none';
      canvas.style.zIndex = '10';
      chartContainerRef.current.appendChild(canvas);
      
      webglRendererRef.current = new ChartWebGLRenderer(
        canvas,
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
      if (param.point && param.seriesData && param.seriesData.size > 0) {
        // Get the first series' data
        const seriesData = param.seriesData.values().next().value;
        // Extract price from the data (handles both line and candlestick data)
        const price = seriesData?.value || seriesData?.close;
        
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
      clearAllPriceLines();
      chart.remove();
      webglRendererRef.current?.destroy();
    };
  }, [symbol, theme, onPriceClick, clearAllPriceLines]);
  
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
    
    // Transform data for TradingView format - THE FIX IS HERE
    const tvData = candleData.map(candle => ({
      time: Math.floor(candle.timestamp.getTime() / 1000) as UTCTimestamp, // Convert to seconds and cast
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close
    }));
    
    const volumeData = candleData.map(candle => ({
      time: Math.floor(candle.timestamp.getTime() / 1000) as UTCTimestamp, // Same fix for volume
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
    
    // Clear existing lines - FIXED METHOD
    clearAllPriceLines();
    
    // Draw position line - where we stand
    if (position) {
      const profitColor = position.unrealizedPnL.gte(0) 
        ? themes[theme].candle.upColor 
        : themes[theme].candle.downColor;
      
      const positionLine = candleSeriesRef.current.createPriceLine({
        price: position.entryPrice.toNumber(),
        color: profitColor,
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Position: ${position.quantity} @ ${position.entryPrice}`
      });
      
      priceLinesRef.current.push(positionLine);
      
      // Liquidation price if exists - the danger zone
      if (position.liquidationPrice) {
        const liquidationLine = candleSeriesRef.current.createPriceLine({
          price: position.liquidationPrice.toNumber(),
          color: '#ff0000',
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: 'LIQUIDATION'
        });
        
        priceLinesRef.current.push(liquidationLine);
      }
    }
    
    // Draw order lines - our intentions
    orders.forEach(order => {
      if (order.price) {
        const color = order.side === 'buy' 
          ? themes[theme].candle.upColor 
          : themes[theme].candle.downColor;
        
        const orderLine = candleSeriesRef.current!.createPriceLine({
          price: order.price.toNumber(),
          color: color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: `${order.side.toUpperCase()} ${order.quantity}`
        });
        
        priceLinesRef.current.push(orderLine);
      }
    });
    
  }, [positions, activeOrders, symbol, theme, clearAllPriceLines]);
  
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
 * CHART WISDOM - CYBERPUNK EDITION:
 * 
 * 1. Time is a flat circle, but TradingView wants it in seconds.
 *    We convert milliseconds to seconds because that's how the
 *    matrix measures decay.
 * 
 * 2. Price lines are like memories - you have to track them yourself
 *    or they vanish into the digital void. Store refs, manage state,
 *    never trust the API to remember.
 * 
 * 3. TypeScript is your netrunner - it catches bugs before they
 *    crash your chrome. Cast types explicitly, the compiler is
 *    your friend in the dark.
 * 
 * 4. When the market glitches and candles update 1000x/second,
 *    your chart needs to keep up. WebGL isn't optional, it's
 *    survival gear.
 * 
 * 5. Every millisecond of lag is credits lost. Optimize render
 *    loops, batch updates, and always monitor FPS. In the sprawl,
 *    speed is the only currency that matters.
 * 
 * Remember: The chart shows the past, the trader sees the future,
 * and the code bridges the gap between them.
 */