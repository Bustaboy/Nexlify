// frontend/src/components/trading/TradingChart.tsx
/**
 * TradingView Lightweight Charts Integration - The Visual Cortex
 * High-performance price visualization optimized for algorithmic trading
 * 
 * This is where pattern recognition meets profit potential.
 * Every candlestick tells a story, every volume spike whispers secrets.
 * Been staring at charts so long they're burned into my retinas.
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi,
  ChartOptions,
  DeepPartial,
  Time,
  UTCTimestamp,
  ColorType,
  LineStyle,
  CrosshairMode
} from 'lightweight-charts';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Maximize2, 
  Minimize2, 
  Settings, 
  TrendingUp, 
  Volume2,
  Target,
  AlertTriangle,
  Zap
} from 'lucide-react';

// Stores and hooks
import { useTradingStore } from '@stores/tradingStore';
import { useSettingsStore } from '@stores/settingsStore';
import { useWebSocket } from '@hooks/useWebSocket';

// Utils
import { formatCurrency, formatVolume } from '@lib/utils';
import { playSound } from '@lib/sounds';

// Types - the data structures that flow through our neural pathways
interface ChartData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface TradingLevel {
  price: number;
  type: 'support' | 'resistance' | 'signal' | 'order';
  label: string;
  color: string;
  style?: LineStyle;
}

interface ChartConfig {
  symbol: string;
  timeframe: string;
  showVolume: boolean;
  showOrders: boolean;
  showSignals: boolean;
  showLevels: boolean;
  theme: 'cyberpunk' | 'classic' | 'minimal';
}

interface TradingChartProps {
  symbol: string;
  timeframe?: string;
  height?: number;
  showControls?: boolean;
  onCrosshairMove?: (price: number, time: Time) => void;
  onChartClick?: (price: number, time: Time) => void;
}

// Cyberpunk theme configuration - because aesthetics matter when you're hunting profits
const CYBERPUNK_THEME: DeepPartial<ChartOptions> = {
  layout: {
    background: { type: ColorType.Solid, color: '#0a0a0f' },
    textColor: '#00ff88',
    fontSize: 12,
    fontFamily: 'JetBrains Mono, monospace'
  },
  grid: {
    vertLines: { color: '#1a1a2e', style: LineStyle.Solid },
    horzLines: { color: '#1a1a2e', style: LineStyle.Solid }
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#00ff88',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#00ff88'
    },
    horzLine: {
      color: '#00ff88',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#00ff88'
    }
  },
  rightPriceScale: {
    borderColor: '#2a2a4e',
    textColor: '#00ff88',
    entireTextOnly: false
  },
  timeScale: {
    borderColor: '#2a2a4e',
    textColor: '#00ff88',
    timeVisible: true,
    secondsVisible: true
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: true
  },
  handleScale: {
    axisPressedMouseMove: true,
    mouseWheel: true,
    pinch: true
  }
};

export const TradingChart: React.FC<TradingChartProps> = ({
  symbol,
  timeframe = '1m',
  height = 500,
  showControls = true,
  onCrosshairMove,
  onChartClick
}) => {
  // Refs for chart API access
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  // Store connections
  const { marketData, positions, orders, signals } = useTradingStore();
  const { theme } = useSettingsStore();
  const { subscribe, unsubscribe } = useWebSocket();

  // Component state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState<ChartConfig>({
    symbol,
    timeframe,
    showVolume: true,
    showOrders: true,
    showSignals: true,
    showLevels: true,
    theme: 'cyberpunk'
  });

  // Chart data management - the heart that pumps price data through our veins
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [tradingLevels, setTradingLevels] = useState<TradingLevel[]>([]);

  // Memoized chart options - optimization for the speed demons
  const chartOptions = useMemo<DeepPartial<ChartOptions>>(() => ({
    ...CYBERPUNK_THEME,
    width: isFullscreen ? window.innerWidth : undefined,
    height: isFullscreen ? window.innerHeight : height,
    autoSize: true
  }), [isFullscreen, height, theme]);

  // Initialize chart - birth of the visual cortex
  const initChart = useCallback(() => {
    if (!chartContainerRef.current) return;

    console.log(`🎯 Initializing chart for ${symbol} - Visual cortex coming online`);

    // Create the chart instance
    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;

    // Create candlestick series - where price action lives and breathes
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',      // Profit green - the color of success
      downColor: '#ff0066',    // Loss red - lessons learned the hard way
      borderUpColor: '#00ff88',
      borderDownColor: '#ff0066',
      wickUpColor: '#00ff88',
      wickDownColor: '#ff0066',
      priceFormat: {
        type: 'price',
        precision: 8,
        minMove: 0.00000001
      }
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series if enabled - market participation matters
    if (config.showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#404040',
        priceFormat: {
          type: 'volume'
        },
        priceScaleId: 'volume_scale'
      });
      volumeSeriesRef.current = volumeSeries;
      
      // Position volume series at bottom
      chart.priceScale('volume_scale').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0
        }
      });
    }

    // Crosshair tracking - know exactly where your cursor points to opportunity
    chart.subscribeCrosshairMove((param) => {
      if (param.time && param.point) {
        const price = candlestickSeries.coordinateToPrice(param.point.y);
        if (price && onCrosshairMove) {
          onCrosshairMove(price, param.time);
        }
      }
    });

    // Click handling - interaction with the price universe
    chart.subscribeClick((param) => {
      if (param.time && param.point) {
        const price = candlestickSeries.coordinateToPrice(param.point.y);
        if (price && onChartClick) {
          onChartClick(price, param.time);
          playSound('chart_click');
        }
      }
    });

    // Setup resize observer - adapt to every screen change
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect();
    }
    
    resizeObserverRef.current = new ResizeObserver(() => {
      if (chartRef.current) {
        chartRef.current.applyOptions({ autoSize: true });
      }
    });
    
    resizeObserverRef.current.observe(chartContainerRef.current);

  }, [chartOptions, config.showVolume, onCrosshairMove, onChartClick, symbol]);

  // Update chart data - feed the beast with fresh market data
  const updateChartData = useCallback((newData: ChartData[]) => {
    if (!candlestickSeriesRef.current || !newData.length) return;

    try {
      // Sort data by time - chronological order is sacred
      const sortedData = [...newData].sort((a, b) => (a.time as number) - (b.time as number));
      
      // Update candlestick series
      candlestickSeriesRef.current.setData(sortedData);
      
      // Update volume series if present
      if (volumeSeriesRef.current && config.showVolume) {
        const volumeData = sortedData
          .filter(d => d.volume !== undefined)
          .map(d => ({
            time: d.time,
            value: d.volume!,
            color: d.close > d.open ? '#00ff8844' : '#ff006644'
          }));
        
        volumeSeriesRef.current.setData(volumeData);
      }

      // Auto-fit the view - always show the full picture
      setTimeout(() => {
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent();
        }
      }, 100);

    } catch (error) {
      console.error('Chart data update failed:', error);
    }
  }, [config.showVolume]);

  // Add trading levels - visualize support, resistance, and signals
  const addTradingLevel = useCallback((level: TradingLevel) => {
    if (!candlestickSeriesRef.current) return;

    // Create price line
    const priceLine = candlestickSeriesRef.current.createPriceLine({
      price: level.price,
      color: level.color,
      lineWidth: 2,
      lineStyle: level.style || LineStyle.Solid,
      axisLabelVisible: true,
      title: level.label
    });

    return priceLine;
  }, []);

  // Handle real-time market data updates
  useEffect(() => {
    const symbolData = marketData.get(symbol);
    if (!symbolData) return;

    // Convert market data to chart format
    const newCandle: ChartData = {
      time: (symbolData.timestamp.getTime() / 1000) as UTCTimestamp,
      open: symbolData.open || symbolData.price,
      high: symbolData.high || symbolData.price,
      low: symbolData.low || symbolData.price,
      close: symbolData.price,
      volume: symbolData.volume
    };

    // Update or add the latest candle
    if (candlestickSeriesRef.current) {
      candlestickSeriesRef.current.update(newCandle);
      
      if (volumeSeriesRef.current && newCandle.volume) {
        volumeSeriesRef.current.update({
          time: newCandle.time,
          value: newCandle.volume,
          color: newCandle.close > newCandle.open ? '#00ff8844' : '#ff006644'
        });
      }
    }

  }, [marketData, symbol]);

  // Initialize chart on mount
  useEffect(() => {
    initChart();
    
    return () => {
      // Cleanup - clean disconnection from the matrix
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      }
    };
  }, [initChart]);

  // Subscribe to symbol data
  useEffect(() => {
    subscribe([symbol]);
    
    return () => {
      unsubscribe([symbol]);
    };
  }, [symbol, subscribe, unsubscribe]);

  // Keyboard shortcuts - speed of thought trading
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target === document.body) {
        switch (e.key) {
          case 'f':
          case 'F':
            setIsFullscreen(!isFullscreen);
            break;
          case 'v':
          case 'V':
            setConfig(prev => ({ ...prev, showVolume: !prev.showVolume }));
            break;
          case 's':
          case 'S':
            setShowSettings(!showSettings);
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isFullscreen, showSettings]);

  // Build trading levels from current positions and orders
  useEffect(() => {
    const levels: TradingLevel[] = [];

    // Add open positions
    positions
      .filter(p => p.symbol === symbol && p.status === 'open')
      .forEach(position => {
        levels.push({
          price: position.entry_price,
          type: 'order',
          label: `${position.side.toUpperCase()} ${position.size} @ ${formatCurrency(position.entry_price)}`,
          color: position.side === 'buy' ? '#00ff88' : '#ff0066'
        });

        // Add stop loss and take profit if set
        if (position.stop_loss) {
          levels.push({
            price: position.stop_loss,
            type: 'order',
            label: `Stop Loss @ ${formatCurrency(position.stop_loss)}`,
            color: '#ff6600',
            style: LineStyle.Dashed
          });
        }

        if (position.take_profit) {
          levels.push({
            price: position.take_profit,
            type: 'order',
            label: `Take Profit @ ${formatCurrency(position.take_profit)}`,
            color: '#00ff88',
            style: LineStyle.Dashed
          });
        }
      });

    // Add pending orders
    orders
      .filter(o => o.symbol === symbol && o.status === 'pending')
      .forEach(order => {
        levels.push({
          price: order.price,
          type: 'order',
          label: `${order.side.toUpperCase()} Order @ ${formatCurrency(order.price)}`,
          color: order.side === 'buy' ? '#0088ff' : '#ff8800'
        });
      });

    // Add recent signals
    signals
      .filter(s => s.symbol === symbol && Date.now() - new Date(s.timestamp).getTime() < 300000) // 5 min
      .forEach(signal => {
        levels.push({
          price: signal.price,
          type: 'signal',
          label: `AI Signal: ${signal.action.toUpperCase()} (${(signal.confidence * 100).toFixed(0)}%)`,
          color: signal.action === 'buy' ? '#00ffff' : '#ff00ff',
          style: LineStyle.Dotted
        });
      });

    setTradingLevels(levels);
  }, [positions, orders, signals, symbol]);

  // Render trading levels on chart
  useEffect(() => {
    if (!candlestickSeriesRef.current || !config.showLevels) return;

    // Clear existing levels (simplified - in production, track and update individually)
    tradingLevels.forEach(level => {
      addTradingLevel(level);
    });

  }, [tradingLevels, addTradingLevel, config.showLevels]);

  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden border border-gray-700 ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Chart Header - the control center */}
      {showControls && (
        <div className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <h3 className="text-lg font-bold text-cyan-400 font-mono">
              {symbol} <span className="text-sm text-gray-400">({timeframe})</span>
            </h3>
            
            {/* Current price display */}
            {marketData.get(symbol) && (
              <div className="flex items-center space-x-2">
                <span className="text-xl font-mono text-white">
                  {formatCurrency(marketData.get(symbol)!.price)}
                </span>
                <span className={`text-sm font-mono ${
                  (marketData.get(symbol)!.change || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(marketData.get(symbol)!.change || 0) >= 0 ? '+' : ''}{((marketData.get(symbol)!.change || 0) * 100).toFixed(2)}%
                </span>
              </div>
            )}
          </div>

          {/* Chart controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setConfig(prev => ({ ...prev, showVolume: !prev.showVolume }))}
              className={`p-2 rounded transition-colors ${
                config.showVolume ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
              title="Toggle Volume (V)"
            >
              <Volume2 size={16} />
            </button>
            
            <button
              onClick={() => setConfig(prev => ({ ...prev, showLevels: !prev.showLevels }))}
              className={`p-2 rounded transition-colors ${
                config.showLevels ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
              title="Toggle Trading Levels"
            >
              <Target size={16} />
            </button>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 bg-gray-700 text-gray-400 rounded hover:bg-gray-600 transition-colors"
              title="Chart Settings (S)"
            >
              <Settings size={16} />
            </button>

            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 bg-gray-700 text-gray-400 rounded hover:bg-gray-600 transition-colors"
              title="Fullscreen (F)"
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          </div>
        </div>
      )}

      {/* Chart container - where the magic happens */}
      <div 
        ref={chartContainerRef}
        className="w-full"
        style={{ height: isFullscreen ? 'calc(100vh - 60px)' : height }}
      />

      {/* Trading levels legend */}
      {config.showLevels && tradingLevels.length > 0 && (
        <div className="absolute top-16 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 max-w-xs">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2">Active Levels</h4>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {tradingLevels.slice(0, 8).map((level, index) => (
              <div key={index} className="flex items-center space-x-2 text-xs">
                <div 
                  className="w-3 h-0.5 rounded"
                  style={{ backgroundColor: level.color }}
                />
                <span className="text-gray-300 truncate">{level.label}</span>
              </div>
            ))}
            {tradingLevels.length > 8 && (
              <div className="text-xs text-gray-500">
                +{tradingLevels.length - 8} more levels
              </div>
            )}
          </div>
        </div>
      )}

      {/* Settings panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="absolute top-0 right-0 w-80 h-full bg-gray-900/95 backdrop-blur-md border-l border-gray-700 p-4"
          >
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">Chart Settings</h3>
            
            {/* Settings content would go here */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Timeframe</label>
                <select 
                  value={config.timeframe}
                  onChange={(e) => setConfig(prev => ({ ...prev, timeframe: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
                >
                  <option value="1m">1 Minute</option>
                  <option value="5m">5 Minutes</option>
                  <option value="15m">15 Minutes</option>
                  <option value="1h">1 Hour</option>
                  <option value="4h">4 Hours</option>
                  <option value="1d">1 Day</option>
                </select>
              </div>
              
              {/* More settings would be added here */}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
