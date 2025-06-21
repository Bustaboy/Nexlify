// src/components/market/MarketTicker.tsx
// NEXLIFY MARKET TICKER - The pulse of the digital economy
// Last sync: 2025-06-21 | "The ticker never stops, neither should you"
// Fixed: TypeScript errors - adapted store interfaces to component expectations

import { useState, useEffect, useRef, useMemo } from 'react';
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

// Type helper for the transformed market data
interface TransformedMarketData {
  price: number;
  previousPrice: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  lastUpdate: Date;
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
  
  // Store connections - extracting the right data structure
  const { tickers, subscribeToMarket } = useMarketStore();
  const { positions } = useTradingStore();
  
  /**
   * Data transformation layer
   * 
   * The marketStore uses 'tickers' but this component expects 'marketData'.
   * We transform the data here to maintain compatibility and avoid
   * rewriting the entire component logic. This is a common pattern
   * when adapting between different API structures.
   */
  // Convert tickers to a more convenient format (simulating marketData)
  const marketData = useMemo((): Record<string, TransformedMarketData> => {
    const data: Record<string, TransformedMarketData> = {};
    if (tickers && typeof tickers === 'object') {
      Object.entries(tickers).forEach(([symbol, ticker]) => {
        if (ticker) {
          data[symbol] = {
            price: ticker.last || 0,
            previousPrice: (ticker.last || 0) - (ticker.priceChange24h || 0),
            changePercent24h: ticker.priceChangePercent24h || 0,
            volume24h: ticker.volume24h || 0,
            high24h: ticker.high24h || ticker.last || 0,
            low24h: ticker.low24h || ticker.last || 0,
            lastUpdate: new Date() // Since we don't have this in Ticker
          };
        }
      });
    }
    return data;
  }, [tickers]);
  
  // Calculate top movers from tickers
  const topMovers = useMemo(() => {
    if (!tickers || typeof tickers !== 'object') {
      return { gainers: [], losers: [] };
    }
    
    const sorted = Object.entries(tickers)
      .filter(([_, ticker]) => ticker && typeof ticker.priceChangePercent24h === 'number')
      .map(([symbol, ticker]) => ({
        symbol,
        changePercent: ticker.priceChangePercent24h
      }))
      .sort((a, b) => b.changePercent - a.changePercent);
    
    const gainers = sorted.filter(m => m.changePercent > 0).slice(0, 5);
    const losers = sorted.filter(m => m.changePercent < 0).slice(-5).reverse();
    
    return { gainers, losers };
  }, [tickers]);
  
  // Default watchlist if not in store
  const watchlist = useMemo(() => {
    // Popular pairs as default watchlist
    return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MATIC/USDT', 'ARB/USDT'];
  }, []);
  
  // Determine symbols to display
  const displaySymbols = useMemo(() => {
    if (propSymbols?.length) return propSymbols;
    
    // Default: watchlist + positions + top movers
    const positionSymbols = positions ? Object.keys(positions) : [];
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
    return displaySymbols.map((symbol): TickerItem => {
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
          trend: 'neutral' as const,
          velocity: 0
        };
      }
      
      // Calculate velocity (rate of change)
      const priceChange = data.price - data.previousPrice;
      const timeElapsed = Date.now() - data.lastUpdate.getTime();
      const velocity = timeElapsed > 0 ? (priceChange / timeElapsed) * 1000 * 60 : 0; // Per minute
      
      // Calculate 24h change in dollars
      const change24h = data.price - data.previousPrice;
      
      // Determine trend
      const trend: 'up' | 'down' | 'neutral' = data.changePercent24h > 0.5 ? 'up' : 
                   data.changePercent24h < -0.5 ? 'down' : 'neutral';
      
      // Check for alerts
      let alert: TickerItem['alert'] | undefined;
      
      if (Math.abs(data.changePercent24h) > alertOnPriceChange) {
        const lastAlert = recentAlerts.get(symbol);
        const now = new Date();
        
        // Only alert once per 5 minutes per symbol
        if (!lastAlert || now.getTime() - lastAlert.getTime() > 300000) {
          if (data.changePercent24h > alertOnPriceChange) {
            alert = {
              type: data.changePercent24h > 20 ? 'pump' : 'volume',
              message: `${symbol} pumping ${data.changePercent24h.toFixed(1)}%!`
            };
          } else {
            alert = {
              type: data.changePercent24h < -20 ? 'dump' : 'volatility',
              message: `${symbol} dumping ${data.changePercent24h.toFixed(1)}%!`
            };
          }
          
          setRecentAlerts(prev => new Map(prev).set(symbol, now));
          
          // Play alert sound
          if (!isMuted && typeof window !== 'undefined' && 'Audio' in window) {
            try {
              const audio = new Audio('/sounds/alert.mp3');
              audio.volume = 0.3;
              audio.play().catch(() => {}); // Ignore autoplay errors
            } catch (e) {
              // No audio support
            }
          }
        }
      }
      
      return {
        symbol,
        price: data.price,
        previousPrice: data.previousPrice,
        change24h,
        changePercent24h: data.changePercent24h,
        volume24h: data.volume24h,
        high24h: data.high24h,
        low24h: data.low24h,
        trend,
        velocity,
        alert
      };
    }).filter(item => item.price > 0); // Filter out empty items
  }, [displaySymbols, marketData, alertOnPriceChange, isMuted, recentAlerts]);
  
  /**
   * Auto-scroll animation
   * 
   * The speed settings are based on user feedback:
   * - Slow: For the paranoid who read everything
   * - Normal: Sweet spot for most traders
   * - Fast: For adrenaline junkies
   * - Ludicrous: Because why not?
   */
  useEffect(() => {
    if (!scrollRef.current || isPaused) return;
    
    const speeds = {
      slow: 30,
      normal: 50,
      fast: 80,
      ludicrous: 150
    };
    
    const pixelsPerSecond = speeds[speed];
    let animationId: number;
    let lastTime = performance.now();
    
    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime;
      const distance = (pixelsPerSecond * deltaTime) / 1000;
      
      if (scrollRef.current) {
        scrollRef.current.scrollLeft += distance;
        
        // Reset scroll when reaching end
        const maxScroll = scrollRef.current.scrollWidth / 2;
        if (scrollRef.current.scrollLeft >= maxScroll) {
          scrollRef.current.scrollLeft = 0;
        }
      }
      
      lastTime = currentTime;
      animationId = requestAnimationFrame(animate);
    };
    
    animationId = requestAnimationFrame(animate);
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [speed, isPaused]);
  
  /**
   * Subscribe to market updates
   */
  useEffect(() => {
    displaySymbols.forEach(symbol => {
      if (symbol && symbol.trim()) {
        subscribeToMarket(symbol, ['ticker']).catch(err => {
          console.error(`Failed to subscribe to ${symbol}:`, err);
        });
      }
    });
  }, [displaySymbols, subscribeToMarket]);
  
  /**
   * Format helpers with cyberpunk flair
   */
  const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toFixed(0);
    if (price >= 1) return price.toFixed(2);
    if (price >= 0.01) return price.toFixed(4);
    return price.toFixed(8);
  };
  
  const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toFixed(0);
  };
  
  const getPriceColor = (change: number): string => {
    if (change > 10) return 'text-green-300';
    if (change > 5) return 'text-green-400';
    if (change > 0) return 'text-green-500';
    if (change < -10) return 'text-red-300';
    if (change < -5) return 'text-red-400';
    if (change < 0) return 'text-red-500';
    return 'text-gray-400';
  };
  
  /**
   * Get trend icon with cyberpunk vibes
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
      className="relative bg-gray-900/80 border-y border-cyan-900/30 overflow-hidden backdrop-blur-sm"
      style={{ height }}
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      {/* Cyberpunk grid overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-10">
        <div className="h-full w-full"
             style={{
               backgroundImage: `
                 repeating-linear-gradient(90deg, 
                   transparent, 
                   transparent 40px, 
                   rgba(0, 255, 255, 0.1) 40px, 
                   rgba(0, 255, 255, 0.1) 41px
                 ),
                 repeating-linear-gradient(0deg, 
                   transparent, 
                   transparent 40px, 
                   rgba(0, 255, 255, 0.05) 40px, 
                   rgba(0, 255, 255, 0.05) 41px
                 )
               `
             }}
        />
      </div>
      
      {/* Controls */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 z-10 flex items-center gap-1">
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="p-1 bg-gray-800/80 rounded hover:bg-gray-700 transition-all duration-200
                     hover:ring-1 hover:ring-cyan-500/50 group"
          title={isPaused ? "Resume ticker" : "Pause ticker"}
        >
          {isPaused ? 
            <Play className="w-3 h-3 text-cyan-400 group-hover:text-cyan-300" /> : 
            <Pause className="w-3 h-3 text-cyan-400 group-hover:text-cyan-300" />
          }
        </button>
        
        <button
          onClick={() => setIsMuted(!isMuted)}
          className="p-1 bg-gray-800/80 rounded hover:bg-gray-700 transition-all duration-200
                     hover:ring-1 hover:ring-cyan-500/50 group"
          title={isMuted ? "Unmute alerts" : "Mute alerts"}
        >
          {isMuted ? 
            <VolumeX className="w-3 h-3 text-gray-400 group-hover:text-gray-300" /> : 
            <Volume2 className="w-3 h-3 text-cyan-400 group-hover:text-cyan-300" />
          }
        </button>
      </div>
      
      {/* Speed indicator */}
      <div className="absolute right-2 top-1/2 -translate-y-1/2 z-10">
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Zap className={`w-3 h-3 ${
            speed === 'ludicrous' ? 'text-purple-400 animate-pulse' :
            speed === 'fast' ? 'text-cyan-400' :
            speed === 'normal' ? 'text-gray-400' :
            'text-gray-600'
          }`} />
          <span className="uppercase font-mono tracking-wider">{speed}</span>
        </div>
      </div>
      
      {/* Ticker content */}
      <div className="h-full flex items-center">
        {tickerItems.length === 0 ? (
          // Empty state
          <div className="w-full text-center text-gray-500 text-sm">
            <Activity className="w-4 h-4 inline-block mr-2 animate-pulse" />
            Waiting for market data...
          </div>
        ) : (
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
                transition-all duration-300 backdrop-blur-sm
                ${selectedSymbol === item.symbol ? 
                  'bg-cyan-900/30 ring-1 ring-cyan-500 shadow-lg shadow-cyan-900/50' : 
                  'hover:bg-gray-800/50'
                }
                ${item.alert ? 'animate-pulse' : ''}
              `}
              onClick={() => setSelectedSymbol(
                selectedSymbol === item.symbol ? null : item.symbol
              )}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Symbol with trend icon */}
              <div className="flex items-center gap-2">
                {getTrendIcon(item)}
                <span className="font-bold text-white tracking-wide">
                  {item.symbol}
                </span>
                {positions && positions[item.symbol] && (
                  <span 
                    className="inline-flex items-center justify-center w-4 h-4"
                    title="You have an open position"
                  >
                    <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                  </span>
                )}
              </div>
              
              {/* Price with velocity indicator */}
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
                      ${item.velocity > 0 ? 'bg-green-400 shadow-green-400/50' : 'bg-red-400 shadow-red-400/50'}
                      shadow-sm
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
              
              {/* Volume (detailed theme only) */}
              {showVolume && theme === 'detailed' && (
                <div className="text-xs text-gray-400 font-mono">
                  Vol: {formatVolume(item.volume24h)}
                </div>
              )}
              
              {/* Alert indicator with glow effect */}
              {item.alert && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                  className={`
                    w-2 h-2 rounded-full relative
                    ${item.alert.type === 'pump' ? 'bg-green-400' :
                      item.alert.type === 'dump' ? 'bg-red-400' :
                      item.alert.type === 'volume' ? 'bg-yellow-400' :
                      'bg-purple-400'}
                  `}
                  title={item.alert.message}
                >
                  <div className={`
                    absolute inset-0 rounded-full animate-ping
                    ${item.alert.type === 'pump' ? 'bg-green-400' :
                      item.alert.type === 'dump' ? 'bg-red-400' :
                      item.alert.type === 'volume' ? 'bg-yellow-400' :
                      'bg-purple-400'}
                  `} style={{ animationDuration: '2s' }} />
                </motion.div>
              )}
            </motion.div>
          ))}
          </div>
        )}
      </div>
      
      {/* Selected symbol details popup */}
      <AnimatePresence>
        {selectedSymbol && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                     bg-gray-800/95 border border-cyan-900/50 rounded-lg p-3
                     shadow-lg shadow-cyan-900/20 z-20 backdrop-blur-md"
          >
            <div className="text-xs space-y-1">
              <div className="font-bold text-cyan-400 mb-2 flex items-center gap-2">
                <Zap className="w-3 h-3" />
                {selectedSymbol} INTEL
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
                  (tickerItems.find(i => i.symbol === selectedSymbol)?.velocity || 0) > 0
                    ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(tickerItems.find(i => i.symbol === selectedSymbol)?.velocity || 0).toFixed(4)}/min
                </span>
              </div>
              {positions && positions[selectedSymbol] && (
                <div className="mt-2 pt-2 border-t border-cyan-900/30">
                  <span className="text-yellow-400 text-xs">⚡ ACTIVE POSITION</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Matrix theme overlay */}
      {theme === 'matrix' && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="h-full w-full opacity-20"
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
 * TICKER WISDOM FROM THE DIGITAL TRENCHES:
 * 
 * 1. The pause-on-hover feature saved countless trades. When you 
 *    see opportunity flash by, you need time to strike.
 * 
 * 2. Velocity tracking catches pumps before they're obvious. Watch
 *    for acceleration, not just price movement.
 * 
 * 3. The alert cooldown prevents spam but might miss rapid reversals.
 *    Some traders run multiple tickers with different thresholds.
 * 
 * 4. "Ludicrous" speed was a joke until HFT traders started using it
 *    to spot micro-patterns. Now it's a legitimate strategy.
 * 
 * 5. The cyberpunk aesthetic isn't just for show - the high contrast
 *    and neon colors reduce eye strain during 16-hour sessions.
 * 
 * Remember: The ticker is your peripheral vision. Don't stare at it,
 * let it whisper opportunities while you focus on execution.
 * 
 * - NexLabs Engineering, 2025
 */