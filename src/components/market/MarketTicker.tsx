// src/components/market/MarketTicker.tsx
// NEXLIFY MARKET TICKER - The pulse of the digital economy
// Last sync: 2025-06-19 | "The ticker never stops, neither should you"

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown,
  Activity,
  Zap,
  AlertCircle,
  Volume2,
  VolumeX,
  Pause,
  Play,
  ChevronLeft,
  ChevronRight,
  Star,
  Flame,
  Skull,
  Rocket
} from 'lucide-react';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore } from '@/stores/tradingStore';

interface MarketTickerProps {
  symbols?: string[];
  speed?: 'slow' | 'normal' | 'fast' | 'ludicrous';
  showVolume?: boolean;
  showChange24h?: boolean;
  alertOnPriceChange?: number; // Percentage threshold
  theme?: 'minimal' | 'detailed' | 'matrix';
  height?: number;
}

interface TickerItem {
  symbol: string;
  price: number;
  previousPrice: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  trend: 'up' | 'down' | 'neutral';
  velocity: number; // Rate of change
  alert?: {
    type: 'pump' | 'dump' | 'volume' | 'volatility';
    message: string;
  };
}

/**
 * MARKET TICKER - The endless scroll of opportunity and disaster
 * 
 * Built this after the infamous "Ticker Blindness" incident of 2021.
 * Trader stared at BTC pumping for 3 hours, forgot to actually buy.
 * By the time he snapped out of it, 20% gains had passed him by.
 * 
 * This ticker doesn't just show prices - it SCREAMS opportunity:
 * - Velocity indicators for momentum
 * - Audio alerts for significant moves
 * - Color-coded urgency levels
 * - Auto-pause on hover (no more missed entries)
 * 
 * Remember: The ticker is like the matrix code - once you learn
 * to read it, you see the market's true nature.
 */
export const MarketTicker = ({
  symbols: propSymbols,
  speed = 'normal',
  showVolume = true,
  showChange24h = true,
  alertOnPriceChange = 5, // 5% default alert threshold
  theme = 'detailed',
  height = 60
}: MarketTickerProps) => {
  // State management
  const [isPaused, setIsPaused] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<Map<string, Date>>(new Map());
  const tickerRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Store connections
  const { marketData, topMovers, subscribeToMarket } = useMarketStore();
  const { watchlist, positions } = useTradingStore();
  
  // Determine symbols to display
  const displaySymbols = useMemo(() => {
    if (propSymbols?.length) return propSymbols;
    
    // Default: watchlist + positions + top movers
    const positionSymbols = Object.keys(positions);
    const allSymbols = new Set([...watchlist, ...positionSymbols]);
    
    // Add top gainers/losers
    topMovers.gainers.slice(0, 3).forEach(m => allSymbols.add(m.symbol));
    topMovers.losers.slice(0, 3).forEach(m => allSymbols.add(m.symbol));
    
    return Array.from(allSymbols);
  }, [propSymbols, watchlist, positions, topMovers]);
  
  /**
   * Process market data into ticker items
   * 
   * This is where we detect pumps, dumps, and other anomalies.
   * The velocity calculation helps spot momentum shifts before
   * they become obvious.
   */
  const tickerItems = useMemo((): TickerItem[] => {
    return displaySymbols.map(symbol => {
      const data = marketData[symbol];
      if (!data) {
        return {
          symbol,
          price: 0,
          previousPrice: 0,
          change24h: 0,
          changePercent24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          trend: 'neutral',
          velocity: 0
        };
      }
      
      // Calculate velocity (rate of change)
      const priceChange = data.price - data.previousPrice;
      const timeElapsed = Date.now() - data.lastUpdate.getTime();
      const velocity = timeElapsed > 0 ? (priceChange / timeElapsed) * 1000 * 60 : 0; // Per minute
      
      // Determine trend
      const trend = data.changePercent24h > 0.5 ? 'up' : 
                   data.changePercent24h < -0.5 ? 'down' : 'neutral';
      
      // Check for alerts
      let alert: TickerItem['alert'];
      const lastAlert = recentAlerts.get(symbol);
      const alertCooldown = 5 * 60 * 1000; // 5 minutes
      
      if (!lastAlert || Date.now() - lastAlert.getTime() > alertCooldown) {
        // Pump detection
        if (data.changePercent24h > alertOnPriceChange) {
          alert = {
            type: 'pump',
            message: `${symbol} pumping! +${data.changePercent24h.toFixed(1)}%`
          };
        }
        // Dump detection
        else if (data.changePercent24h < -alertOnPriceChange) {
          alert = {
            type: 'dump',
            message: `${symbol} dumping! ${data.changePercent24h.toFixed(1)}%`
          };
        }
        // Volume spike
        else if (data.volume24h > data.avgVolume * 3) {
          alert = {
            type: 'volume',
            message: `${symbol} volume spike! ${(data.volume24h / data.avgVolume).toFixed(1)}x avg`
          };
        }
        // High volatility
        else if ((data.high24h - data.low24h) / data.low24h > 0.15) {
          alert = {
            type: 'volatility',
            message: `${symbol} high volatility!`
          };
        }
      }
      
      return {
        symbol,
        price: data.price,
        previousPrice: data.previousPrice,
        change24h: data.change24h,
        changePercent24h: data.changePercent24h,
        volume24h: data.volume24h,
        high24h: data.high24h,
        low24h: data.low24h,
        trend,
        velocity,
        alert
      };
    });
  }, [displaySymbols, marketData, alertOnPriceChange, recentAlerts]);
  
  /**
   * Play alert sound - audio feedback for critical moves
   * 
   * Different sounds for different alerts. Your ears learn faster
   * than your eyes in a fast market.
   */
  const playAlertSound = useCallback((type: string) => {
    if (isMuted) return;
    
    // In real implementation, would use Web Audio API
    console.log(`🔊 Alert sound: ${type}`);
  }, [isMuted]);
  
  /**
   * Handle alerts
   */
  useEffect(() => {
    tickerItems.forEach(item => {
      if (item.alert) {
        // Update recent alerts
        setRecentAlerts(prev => new Map(prev).set(item.symbol, new Date()));
        
        // Play sound
        playAlertSound(item.alert.type);
        
        // Log alert
        console.log(`🚨 ALERT: ${item.alert.message}`);
      }
    });
  }, [tickerItems, playAlertSound]);
  
  /**
   * Auto-scroll animation
   */
  useEffect(() => {
    if (isPaused || !scrollRef.current) return;
    
    const scrollSpeed = {
      slow: 30,
      normal: 50,
      fast: 80,
      ludicrous: 150
    }[speed];
    
    let animationId: number;
    let scrollPosition = 0;
    
    const animate = () => {
      if (!scrollRef.current) return;
      
      scrollPosition += scrollSpeed / 60; // 60 FPS
      
      // Reset when reaching end
      if (scrollPosition >= scrollRef.current.scrollWidth / 2) {
        scrollPosition = 0;
      }
      
      scrollRef.current.style.transform = `translateX(-${scrollPosition}px)`;
      animationId = requestAnimationFrame(animate);
    };
    
    animationId = requestAnimationFrame(animate);
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isPaused, speed]);
  
  /**
   * Subscribe to market data for displayed symbols
   */
  useEffect(() => {
    displaySymbols.forEach(symbol => {
      subscribeToMarket(symbol, ['ticker']);
    });
  }, [displaySymbols, subscribeToMarket]);
  
  /**
   * Format price based on value
   */
  const formatPrice = (price: number): string => {
    if (price >= 10000) return price.toFixed(0);
    if (price >= 1000) return price.toFixed(1);
    if (price >= 1) return price.toFixed(2);
    if (price >= 0.01) return price.toFixed(4);
    return price.toFixed(6);
  };
  
  /**
   * Format volume
   */
  const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toFixed(0);
  };
  
  /**
   * Get price color based on change
   */
  const getPriceColor = (change: number): string => {
    if (change > 5) return 'text-green-300';
    if (change > 0) return 'text-green-400';
    if (change < -5) return 'text-red-300';
    if (change < 0) return 'text-red-400';
    return 'text-gray-400';
  };
  
  /**
   * Get trend icon
   */
  const getTrendIcon = (item: TickerItem) => {
    if (item.changePercent24h > 10) return <Rocket className="w-4 h-4 text-green-300" />;
    if (item.changePercent24h > 5) return <Flame className="w-4 h-4 text-green-400" />;
    if (item.changePercent24h > 0) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (item.changePercent24h < -10) return <Skull className="w-4 h-4 text-red-300" />;
    if (item.changePercent24h < -5) return <AlertCircle className="w-4 h-4 text-red-400" />;
    if (item.changePercent24h < 0) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Activity className="w-4 h-4 text-gray-500" />;
  };
  
  return (
    <div 
      className="relative bg-gray-900/80 border-y border-cyan-900/30 overflow-hidden"
      style={{ height }}
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      {/* Controls */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 z-10 flex items-center gap-1">
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="p-1 bg-gray-800/80 rounded hover:bg-gray-700 transition-colors"
          title={isPaused ? "Resume" : "Pause"}
        >
          {isPaused ? 
            <Play className="w-3 h-3 text-cyan-400" /> : 
            <Pause className="w-3 h-3 text-cyan-400" />
          }
        </button>
        
        <button
          onClick={() => setIsMuted(!isMuted)}
          className="p-1 bg-gray-800/80 rounded hover:bg-gray-700 transition-colors"
          title={isMuted ? "Unmute alerts" : "Mute alerts"}
        >
          {isMuted ? 
            <VolumeX className="w-3 h-3 text-gray-400" /> : 
            <Volume2 className="w-3 h-3 text-cyan-400" />
          }
        </button>
      </div>
      
      {/* Speed indicator */}
      <div className="absolute right-2 top-1/2 -translate-y-1/2 z-10">
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Zap className="w-3 h-3" />
          <span className="uppercase">{speed}</span>
        </div>
      </div>
      
      {/* Ticker content */}
      <div className="h-full flex items-center">
        <div 
          ref={scrollRef}
          className="flex items-center gap-6 whitespace-nowrap"
          style={{ paddingLeft: '100%' }}
        >
          {/* Duplicate items for seamless loop */}
          {[...tickerItems, ...tickerItems].map((item, index) => (
            <motion.div
              key={`${item.symbol}-${index}`}
              className={`
                flex items-center gap-3 px-4 py-2 rounded-lg cursor-pointer
                transition-all duration-300
                ${selectedSymbol === item.symbol ? 
                  'bg-cyan-900/30 ring-1 ring-cyan-500' : 
                  'hover:bg-gray-800/50'
                }
                ${item.alert ? 'animate-pulse' : ''}
              `}
              onClick={() => setSelectedSymbol(
                selectedSymbol === item.symbol ? null : item.symbol
              )}
              whileHover={{ scale: 1.05 }}
            >
              {/* Symbol */}
              <div className="flex items-center gap-2">
                {getTrendIcon(item)}
                <span className="font-bold text-white">
                  {item.symbol}
                </span>
                {positions[item.symbol] && (
                  <Star className="w-3 h-3 text-yellow-400" title="Position open" />
                )}
              </div>
              
              {/* Price */}
              <div className="flex items-center gap-2">
                <span className={`font-mono font-semibold ${
                  getPriceColor(item.changePercent24h)
                }`}>
                  ${formatPrice(item.price)}
                </span>
                
                {/* Flash animation for price changes */}
                {Math.abs(item.velocity) > 0.01 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0 }}
                    className={`
                      w-2 h-2 rounded-full
                      ${item.velocity > 0 ? 'bg-green-400' : 'bg-red-400'}
                    `}
                  />
                )}
              </div>
              
              {/* 24h Change */}
              {showChange24h && (
                <div className={`flex items-center gap-1 text-sm ${
                  getPriceColor(item.changePercent24h)
                }`}>
                  <span className="font-mono">
                    {item.changePercent24h > 0 ? '+' : ''}
                    {item.changePercent24h.toFixed(2)}%
                  </span>
                </div>
              )}
              
              {/* Volume */}
              {showVolume && theme === 'detailed' && (
                <div className="text-xs text-gray-400">
                  Vol: {formatVolume(item.volume24h)}
                </div>
              )}
              
              {/* Alert indicator */}
              {item.alert && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                  className={`
                    w-2 h-2 rounded-full
                    ${item.alert.type === 'pump' ? 'bg-green-400' :
                      item.alert.type === 'dump' ? 'bg-red-400' :
                      item.alert.type === 'volume' ? 'bg-yellow-400' :
                      'bg-purple-400'}
                  `}
                  title={item.alert.message}
                />
              )}
            </motion.div>
          ))}
        </div>
      </div>
      
      {/* Selected symbol details */}
      <AnimatePresence>
        {selectedSymbol && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                     bg-gray-800 border border-cyan-900/50 rounded-lg p-3
                     shadow-lg shadow-cyan-900/20 z-20"
          >
            <div className="text-xs space-y-1">
              <div className="font-bold text-cyan-400 mb-2">
                {selectedSymbol} Quick Stats
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <span className="text-gray-400">24h High:</span>
                <span className="text-white font-mono">
                  ${formatPrice(marketData[selectedSymbol]?.high24h || 0)}
                </span>
                <span className="text-gray-400">24h Low:</span>
                <span className="text-white font-mono">
                  ${formatPrice(marketData[selectedSymbol]?.low24h || 0)}
                </span>
                <span className="text-gray-400">Volume:</span>
                <span className="text-white font-mono">
                  {formatVolume(marketData[selectedSymbol]?.volume24h || 0)}
                </span>
                <span className="text-gray-400">Velocity:</span>
                <span className={`font-mono ${
                  tickerItems.find(i => i.symbol === selectedSymbol)?.velocity || 0 > 0
                    ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(tickerItems.find(i => i.symbol === selectedSymbol)?.velocity || 0).toFixed(4)}/min
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Matrix theme overlay */}
      {theme === 'matrix' && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="h-full w-full opacity-10"
               style={{
                 backgroundImage: `repeating-linear-gradient(
                   0deg,
                   transparent,
                   transparent 2px,
                   rgba(0, 255, 255, 0.1) 2px,
                   rgba(0, 255, 255, 0.1) 4px
                 )`
               }}
          />
        </div>
      )}
    </div>
  );
};

/**
 * TICKER WISDOM FROM THE FRONT LINES:
 * 
 * 1. The pause-on-hover feature was added after countless missed
 *    entries. When you see opportunity, you need time to click.
 * 
 * 2. Velocity tracking (rate of change) often signals moves before
 *    they show in percentage terms. Fast velocity = momentum building.
 * 
 * 3. Audio alerts aren't annoying - they're profitable. Your brain
 *    processes sound faster than visual changes in peripheral vision.
 * 
 * 4. The duplicate ticker items create a seamless loop. No jarring
 *    resets that might make you miss a price spike.
 * 
 * 5. Speed settings matter based on market conditions. Slow for
 *    stable markets, ludicrous during volatility events.
 * 
 * 6. Position indicators (stars) remind you what you own. Amazing
 *    how many traders forget their positions during busy sessions.
 * 
 * Remember: The ticker is your market radar. It shows you everything
 * happening at once. Master it, and you'll never miss opportunity.
 * 
 * "The market speaks to those who listen. The ticker is its voice."
 */
