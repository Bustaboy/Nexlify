// frontend/src/components/trading/OrderBook.tsx
/**
 * OrderBook Component - Market Depth Scanner
 * Real-time visualization of market liquidity and whale positions
 * 
 * This is where you see the invisible - the walls of buy and sell orders
 * that shape every price movement. Been watching these numbers dance
 * for years, cada nivel tells a story of greed, fear, and opportunity.
 * 
 * The algos hunt in these depths, and if you're not watching,
 * you're just another fish in their net.
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Target, 
  AlertTriangle,
  Layers,
  Activity,
  Zap
} from 'lucide-react';

// Stores and hooks
import { useTradingStore } from '@stores/tradingStore';
import { useWebSocket } from '@hooks/useWebSocket';
import { useSettingsStore } from '@stores/settingsStore';

// Utils
import { formatCurrency, formatVolume, calculateMarketDepth } from '@lib/utils';
import { playSound } from '@lib/sounds';

// Types - the data structures that reveal market intentions
interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
  orders: number;
  timestamp: number;
}

interface OrderBookData {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  spreadPercent: number;
  lastUpdate: number;
  sequence: number;
}

interface MarketDepthMetrics {
  bidVolume: number;
  askVolume: number;
  imbalance: number;
  resistance: number;
  support: number;
  liquidity: number;
}

interface OrderBookProps {
  symbol: string;
  maxLevels?: number;
  showSpread?: boolean;
  showMetrics?: boolean;
  precision?: number;
  groupByPrice?: number;
  highlightLargeOrders?: boolean;
  compactMode?: boolean;
}

// Constants - tuned for market reality
const ORDERBOOK_CONSTANTS = {
  DEFAULT_LEVELS: 15,
  MAX_LEVELS: 50,
  LARGE_ORDER_THRESHOLD: 10000,     // USD value that catches attention
  WHALE_ORDER_THRESHOLD: 100000,    // When the big players move
  UPDATE_THROTTLE: 50,              // 50ms throttle for smooth updates
  SPREAD_WARNING_THRESHOLD: 0.5,    // 0.5% spread triggers warning
  IMBALANCE_THRESHOLD: 70,          // 70% imbalance is significant
  FLASH_DURATION: 1000              // Order flash animation duration
} as const;

export const OrderBook: React.FC<OrderBookProps> = ({
  symbol,
  maxLevels = ORDERBOOK_CONSTANTS.DEFAULT_LEVELS,
  showSpread = true,
  showMetrics = true,
  precision = 2,
  groupByPrice = 0,
  highlightLargeOrders = true,
  compactMode = false
}) => {
  // Store connections
  const { marketData } = useTradingStore();
  const { soundEnabled } = useSettingsStore();
  const { subscribe, unsubscribe, socket } = useWebSocket();

  // Component state - the memory of market movements
  const [orderBookData, setOrderBookData] = useState<OrderBookData>({
    symbol,
    bids: [],
    asks: [],
    spread: 0,
    spreadPercent: 0,
    lastUpdate: Date.now(),
    sequence: 0
  });

  const [flashingOrders, setFlashingOrders] = useState<Set<string>>(new Set());
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'depth' | 'grouped' | 'raw'>('depth');

  // Refs for performance optimization
  const lastUpdateRef = useRef<number>(0);
  const updateThrottleRef = useRef<NodeJS.Timeout>();

  // Market depth metrics calculation - the intelligence behind the numbers
  const depthMetrics = useMemo<MarketDepthMetrics>(() => {
    const { bids, asks } = orderBookData;
    
    if (!bids.length || !asks.length) {
      return {
        bidVolume: 0,
        askVolume: 0,
        imbalance: 0,
        resistance: 0,
        support: 0,
        liquidity: 0
      };
    }

    const bidVolume = bids.reduce((sum, level) => sum + level.size, 0);
    const askVolume = asks.reduce((sum, level) => sum + level.size, 0);
    const totalVolume = bidVolume + askVolume;
    
    const imbalance = totalVolume > 0 ? (bidVolume / totalVolume) * 100 : 50;
    
    // Find significant support/resistance levels
    const maxBidSize = Math.max(...bids.map(b => b.size));
    const maxAskSize = Math.max(...asks.map(a => a.size));
    
    const support = bids.find(b => b.size === maxBidSize)?.price || 0;
    const resistance = asks.find(a => a.size === maxAskSize)?.price || 0;
    
    return {
      bidVolume,
      askVolume,
      imbalance,
      resistance,
      support,
      liquidity: totalVolume
    };
  }, [orderBookData]);

  // Price grouping for cleaner display
  const groupedOrderBook = useMemo(() => {
    if (groupByPrice <= 0) return orderBookData;

    const groupLevels = (levels: OrderBookLevel[], roundUp: boolean) => {
      const grouped = new Map<number, OrderBookLevel>();
      
      levels.forEach(level => {
        const groupedPrice = roundUp 
          ? Math.ceil(level.price / groupByPrice) * groupByPrice
          : Math.floor(level.price / groupByPrice) * groupByPrice;
        
        const existing = grouped.get(groupedPrice);
        if (existing) {
          existing.size += level.size;
          existing.orders += level.orders;
          existing.total += level.total;
        } else {
          grouped.set(groupedPrice, {
            price: groupedPrice,
            size: level.size,
            total: level.total,
            orders: level.orders,
            timestamp: level.timestamp
          });
        }
      });
      
      return Array.from(grouped.values());
    };

    return {
      ...orderBookData,
      bids: groupLevels(orderBookData.bids, false)
        .sort((a, b) => b.price - a.price)
        .slice(0, maxLevels),
      asks: groupLevels(orderBookData.asks, true)
        .sort((a, b) => a.price - b.price)
        .slice(0, maxLevels)
    };
  }, [orderBookData, groupByPrice, maxLevels]);

  // Handle order book updates with throttling - smooth as silk, fast as lightning
  const handleOrderBookUpdate = useCallback((data: any) => {
    const now = Date.now();
    
    // Throttle updates for smooth performance
    if (now - lastUpdateRef.current < ORDERBOOK_CONSTANTS.UPDATE_THROTTLE) {
      if (updateThrottleRef.current) {
        clearTimeout(updateThrottleRef.current);
      }
      
      updateThrottleRef.current = setTimeout(() => {
        handleOrderBookUpdate(data);
      }, ORDERBOOK_CONSTANTS.UPDATE_THROTTLE);
      
      return;
    }
    
    lastUpdateRef.current = now;

    try {
      // Process the raw order book data
      const { bids, asks, symbol: dataSymbol, sequence } = data;
      
      if (dataSymbol !== symbol) return;

      // Convert to our format with totals calculated
      const processLevels = (levels: any[], reverse: boolean = false) => {
        let runningTotal = 0;
        
        const processed = levels
          .slice(0, maxLevels)
          .map(([price, size, orders = 1]) => {
            runningTotal += size;
            return {
              price: parseFloat(price),
              size: parseFloat(size),
              total: runningTotal,
              orders,
              timestamp: now
            };
          });
        
        return reverse ? processed.reverse() : processed;
      };

      const processedBids = processLevels(bids.slice(0, maxLevels), false);
      const processedAsks = processLevels(asks.slice(0, maxLevels), false);

      // Calculate spread
      const bestBid = processedBids[0]?.price || 0;
      const bestAsk = processedAsks[0]?.price || 0;
      const spread = bestAsk - bestBid;
      const spreadPercent = bestBid > 0 ? (spread / bestBid) * 100 : 0;

      // Detect significant order changes for flashing animation
      const newFlashing = new Set<string>();
      
      processedBids.forEach(bid => {
        if (bid.size > ORDERBOOK_CONSTANTS.LARGE_ORDER_THRESHOLD) {
          newFlashing.add(`bid-${bid.price}`);
        }
      });
      
      processedAsks.forEach(ask => {
        if (ask.size > ORDERBOOK_CONSTANTS.LARGE_ORDER_THRESHOLD) {
          newFlashing.add(`ask-${ask.price}`);
        }
      });

      // Sound alerts for significant changes
      if (soundEnabled) {
        if (spreadPercent > ORDERBOOK_CONSTANTS.SPREAD_WARNING_THRESHOLD) {
          playSound('wide_spread');
        }
        
        if (Math.abs(depthMetrics.imbalance - 50) > ORDERBOOK_CONSTANTS.IMBALANCE_THRESHOLD) {
          playSound('order_imbalance');
        }
      }

      setOrderBookData({
        symbol,
        bids: processedBids,
        asks: processedAsks,
        spread,
        spreadPercent,
        lastUpdate: now,
        sequence: sequence || 0
      });

      if (newFlashing.size > 0) {
        setFlashingOrders(newFlashing);
        setTimeout(() => setFlashingOrders(new Set()), ORDERBOOK_CONSTANTS.FLASH_DURATION);
      }

    } catch (error) {
      console.error('OrderBook update error:', error);
    }
  }, [symbol, maxLevels, soundEnabled, depthMetrics.imbalance]);

  // Subscribe to order book updates
  useEffect(() => {
    if (!socket) return;

    console.log(`📊 Subscribing to order book: ${symbol} - Watching the whale movements`);
    
    socket.emit('subscribe_orderbook', { symbol, levels: maxLevels });
    socket.on('orderbook_update', handleOrderBookUpdate);
    
    return () => {
      socket.emit('unsubscribe_orderbook', { symbol });
      socket.off('orderbook_update', handleOrderBookUpdate);
    };
  }, [socket, symbol, maxLevels, handleOrderBookUpdate]);

  // Cleanup throttle on unmount
  useEffect(() => {
    return () => {
      if (updateThrottleRef.current) {
        clearTimeout(updateThrottleRef.current);
      }
    };
  }, []);

  // Order level click handler - interact with the market depth
  const handleLevelClick = useCallback((price: number, side: 'bid' | 'ask') => {
    setSelectedLevel(price);
    playSound('level_select');
    
    // Emit selected price for order form integration
    window.dispatchEvent(new CustomEvent('orderbook:priceSelect', {
      detail: { price, side, symbol }
    }));
  }, [symbol]);

  // Render individual order level - each one a story of supply and demand
  const renderOrderLevel = (level: OrderBookLevel, side: 'bid' | 'ask', index: number) => {
    const isLarge = level.size * level.price > ORDERBOOK_CONSTANTS.LARGE_ORDER_THRESHOLD;
    const isWhale = level.size * level.price > ORDERBOOK_CONSTANTS.WHALE_ORDER_THRESHOLD;
    const isFlashing = flashingOrders.has(`${side}-${level.price}`);
    const isSelected = selectedLevel === level.price;
    
    const maxTotal = Math.max(
      ...groupedOrderBook.bids.map(b => b.total),
      ...groupedOrderBook.asks.map(a => a.total)
    );
    
    const fillPercent = maxTotal > 0 ? (level.total / maxTotal) * 100 : 0;
    
    return (
      <motion.div
        key={`${side}-${level.price}-${index}`}
        initial={{ opacity: 0, x: side === 'bid' ? -20 : 20 }}
        animate={{ 
          opacity: 1, 
          x: 0,
          backgroundColor: isFlashing ? '#00ff8820' : 'transparent'
        }}
        className={`
          relative cursor-pointer transition-all duration-200 group
          ${compactMode ? 'py-0.5' : 'py-1'} px-2 rounded-sm
          ${isSelected ? 'bg-cyan-500/20 border border-cyan-500/50' : 'hover:bg-gray-700/50'}
          ${isWhale ? 'border-l-2 border-l-yellow-400' : isLarge ? 'border-l-2 border-l-orange-400' : ''}
        `}
        onClick={() => handleLevelClick(level.price, side)}
      >
        {/* Background fill bar - visual representation of order depth */}
        <div
          className={`
            absolute inset-y-0 ${side === 'bid' ? 'right-0' : 'left-0'}
            ${side === 'bid' ? 'bg-green-500/10' : 'bg-red-500/10'}
            transition-all duration-300
          `}
          style={{ width: `${fillPercent}%` }}
        />
        
        {/* Order level content */}
        <div className={`relative z-10 grid grid-cols-3 gap-2 text-xs font-mono ${compactMode ? 'text-xs' : 'text-sm'}`}>
          <div className={`text-right ${side === 'bid' ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(level.price, precision)}
          </div>
          
          <div className="text-center text-gray-300">
            {formatVolume(level.size)}
          </div>
          
          <div className="text-left text-gray-400">
            {formatVolume(level.total)}
          </div>
        </div>
        
        {/* Whale indicator */}
        {isWhale && (
          <div className="absolute right-1 top-1/2 transform -translate-y-1/2">
            <div className="w-1 h-1 bg-yellow-400 rounded-full animate-pulse" />
          </div>
        )}
        
        {/* Order count tooltip on hover */}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1 px-2 py-1 bg-gray-900 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
          {level.orders} orders • ${formatCurrency(level.size * level.price, 0)} total
        </div>
      </motion.div>
    );
  };

  // Render spread indicator - the gap that market makers live in
  const renderSpread = () => {
    const bestBid = groupedOrderBook.bids[0]?.price || 0;
    const bestAsk = groupedOrderBook.asks[0]?.price || 0;
    const midPrice = (bestBid + bestAsk) / 2;
    
    const isWideSpread = orderBookData.spreadPercent > ORDERBOOK_CONSTANTS.SPREAD_WARNING_THRESHOLD;
    
    return (
      <div className={`
        flex items-center justify-between p-3 my-2 rounded-lg border
        ${isWideSpread ? 'bg-orange-500/10 border-orange-500/30' : 'bg-gray-800/50 border-gray-600/50'}
      `}>
        <div className="text-center flex-1">
          <div className="text-xs text-gray-400 mb-1">Spread</div>
          <div className={`font-mono ${isWideSpread ? 'text-orange-400' : 'text-gray-300'}`}>
            {formatCurrency(orderBookData.spread, precision)}
          </div>
          <div className={`text-xs ${isWideSpread ? 'text-orange-300' : 'text-gray-500'}`}>
            {orderBookData.spreadPercent.toFixed(3)}%
          </div>
        </div>
        
        <div className="text-center flex-1">
          <div className="text-xs text-gray-400 mb-1">Mid Price</div>
          <div className="font-mono text-cyan-400">
            {formatCurrency(midPrice, precision)}
          </div>
        </div>
        
        {isWideSpread && (
          <div className="flex items-center text-orange-400">
            <AlertTriangle size={16} />
          </div>
        )}
      </div>
    );
  };

  // Render market depth metrics - the intelligence layer
  const renderMetrics = () => {
    const imbalanceColor = depthMetrics.imbalance > 60 ? 'text-green-400' : 
                          depthMetrics.imbalance < 40 ? 'text-red-400' : 'text-gray-400';
    
    return (
      <div className="p-3 bg-gray-800/30 rounded-lg mt-2 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Market Depth</span>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode(viewMode === 'depth' ? 'grouped' : 'depth')}
              className="p-1 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
              title="Toggle view mode"
            >
              <Layers size={12} />
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-gray-400">Bid Volume</div>
            <div className="font-mono text-green-400">{formatVolume(depthMetrics.bidVolume)}</div>
          </div>
          
          <div>
            <div className="text-gray-400">Ask Volume</div>
            <div className="font-mono text-red-400">{formatVolume(depthMetrics.askVolume)}</div>
          </div>
          
          <div>
            <div className="text-gray-400">Imbalance</div>
            <div className={`font-mono ${imbalanceColor}`}>
              {depthMetrics.imbalance.toFixed(1)}%
            </div>
          </div>
          
          <div>
            <div className="text-gray-400">Liquidity</div>
            <div className="font-mono text-cyan-400">{formatVolume(depthMetrics.liquidity)}</div>
          </div>
        </div>
        
        {/* Imbalance bar visualization */}
        <div className="mt-2">
          <div className="flex h-2 rounded-full overflow-hidden bg-gray-700">
            <div 
              className="bg-green-500 transition-all duration-500"
              style={{ width: `${depthMetrics.imbalance}%` }}
            />
            <div 
              className="bg-red-500 transition-all duration-500"
              style={{ width: `${100 - depthMetrics.imbalance}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Buy Pressure</span>
            <span>Sell Pressure</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <BarChart3 size={16} className="text-cyan-400" />
          <h3 className="font-semibold text-white">Order Book</h3>
          <span className="text-xs text-gray-400 font-mono">{symbol}</span>
        </div>
        
        <div className="flex items-center space-x-1 text-xs text-gray-400">
          <Activity size={12} />
          <span>{orderBookData.sequence}</span>
        </div>
      </div>

      {/* Column headers */}
      <div className="px-2 py-2 bg-gray-800/50 grid grid-cols-3 gap-2 text-xs text-gray-400 font-mono border-b border-gray-700">
        <div className="text-right">Price</div>
        <div className="text-center">Size</div>
        <div className="text-left">Total</div>
      </div>

      {/* Order book content */}
      <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600">
        {/* Asks (sell orders) - shown in descending price order */}
        <div className="space-y-0.5 p-1">
          {groupedOrderBook.asks
            .slice()
            .reverse()
            .map((ask, index) => renderOrderLevel(ask, 'ask', index))}
        </div>

        {/* Spread indicator */}
        {showSpread && renderSpread()}

        {/* Bids (buy orders) - shown in descending price order */}
        <div className="space-y-0.5 p-1">
          {groupedOrderBook.bids.map((bid, index) => renderOrderLevel(bid, 'bid', index))}
        </div>
      </div>

      {/* Market metrics */}
      {showMetrics && renderMetrics()}
      
      {/* Connection status indicator */}
      <div className="px-3 py-1 bg-gray-800/30 border-t border-gray-700 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2 text-gray-400">
          <div className={`w-2 h-2 rounded-full ${socket?.connected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span>
            {socket?.connected ? 'Connected' : 'Disconnected'} • 
            Last: {new Date(orderBookData.lastUpdate).toLocaleTimeString()}
          </span>
        </div>
        
        {selectedLevel && (
          <div className="text-cyan-400 font-mono">
            Selected: {formatCurrency(selectedLevel, precision)}
          </div>
        )}
      </div>
    </div>
  );
};
