// src/components/orderbook/OrderBook.tsx
// NEXLIFY ORDERBOOK - Where supply meets demand in the digital bazaar
// Last sync: 2025-06-21 | "The orderbook never lies, but it doesn't tell the whole truth"

import { useState, useEffect, useMemo, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  AlertTriangle,
  Zap,
  Eye,
  EyeOff
} from 'lucide-react';
import Decimal from 'decimal.js';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore } from '@/stores/tradingStore';

interface OrderBookProps {
  symbol: string;
  orderbook?: {
    bids: Array<{ price: number; quantity: number; orderCount?: number }>;
    asks: Array<{ price: number; quantity: number; orderCount?: number }>;
    lastUpdate: Date;
    spread: number;
    midPrice: number;
  };
  maxLevels?: number;
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void;
  showDepthChart?: boolean;
  theme?: 'minimal' | 'detailed' | 'heatmap';
}

/**
 * ORDERBOOK COMPONENT - The market's heartbeat visualized
 * 
 * Built this after watching a whale manipulate BTC orderbooks in '21.
 * Fake walls everywhere, spoofing like it was an art form. That's when
 * I learned - you need to see not just the numbers, but the PATTERNS.
 * 
 * This component shows you the matrix behind the market. Each level
 * tells a story. Big walls might be fake. Thin books spell danger.
 * And that spread? That's where dreams go to die.
 */
export const OrderBook = memo(({
  symbol,
  orderbook,
  maxLevels = 20,
  onPriceClick,
  showDepthChart = false,
  theme = 'detailed'
}: OrderBookProps) => {
  // State - our window into chaos
  const [showFullBook, setShowFullBook] = useState(false);
  const [grouping, setGrouping] = useState<number>(0.01); // Price grouping
  const [highlightLarge, setHighlightLarge] = useState(true);
  const [animateChanges, setAnimateChanges] = useState(true);
  
  // Store connections
  const { subscribeToMarket } = useMarketStore();
  const { positions } = useTradingStore();
  
  // Current position for this symbol
  const currentPosition = positions[symbol];
  
  // Subscribe to orderbook updates if not provided
  useEffect(() => {
    if (!orderbook && symbol) {
      subscribeToMarket(symbol, ['orderbook']);
    }
  }, [symbol, orderbook, subscribeToMarket]);
  
  /**
   * Group orders by price level - because sometimes less detail is more clarity
   * 
   * Learned this from a Korean HFT trader. He said "Grouping is like
   * adjusting your glasses - too much detail blinds you, too little
   * and you miss the moves." 
   */
  const groupOrdersByPrice = useCallback((
    orders: Array<{ price: number; quantity: number; orderCount?: number }>,
    groupingSize: number
  ) => {
    const grouped = new Map<number, { quantity: number; orderCount: number }>();
    
    orders.forEach(order => {
      const groupedPrice = Math.floor(order.price / groupingSize) * groupingSize;
      const existing = grouped.get(groupedPrice) || { quantity: 0, orderCount: 0 };
      
      grouped.set(groupedPrice, {
        quantity: existing.quantity + order.quantity,
        orderCount: existing.orderCount + (order.orderCount || 1)
      });
    });
    
    return Array.from(grouped.entries())
      .map(([price, data]) => ({
        price,
        quantity: data.quantity,
        orderCount: data.orderCount
      }))
      .sort((a, b) => b.price - a.price); // Descending for display
  }, []);
  
  // Process orderbook data
  const processedOrderbook = useMemo(() => {
    if (!orderbook) return null;
    
    const displayLevels = showFullBook ? orderbook.bids.length : maxLevels;
    
    // Group if needed
    const bids = grouping > 0 
      ? groupOrdersByPrice(orderbook.bids, grouping)
      : orderbook.bids;
      
    const asks = grouping > 0
      ? groupOrdersByPrice(orderbook.asks, grouping)
      : orderbook.asks;
    
    // Calculate max quantities for visualization
    const maxBidQty = Math.max(...bids.slice(0, displayLevels).map(b => b.quantity));
    const maxAskQty = Math.max(...asks.slice(0, displayLevels).map(a => a.quantity));
    const maxQuantity = Math.max(maxBidQty, maxAskQty);
    
    // Calculate cumulative quantities for depth
    let bidCumulative = 0;
    const bidsWithCumulative = bids.slice(0, displayLevels).map(bid => {
      bidCumulative += bid.quantity;
      return { ...bid, cumulative: bidCumulative };
    });
    
    let askCumulative = 0;
    const asksWithCumulative = asks.slice(0, displayLevels).map(ask => {
      askCumulative += ask.quantity;
      return { ...ask, cumulative: askCumulative };
    });
    
    // Detect potential spoofing - large orders far from mid
    const detectSpoofing = (orders: typeof bidsWithCumulative, isBid: boolean) => {
      return orders.map((order, index) => {
        const distanceFromTop = index;
        const sizeRatio = order.quantity / maxQuantity;
        const isLarge = sizeRatio > 0.3;
        const isFar = distanceFromTop > 5;
        const isSpoofCandidate = isLarge && isFar;
        
        return { ...order, isSpoofCandidate };
      });
    };
    
    return {
      bids: detectSpoofing(bidsWithCumulative, true),
      asks: detectSpoofing(asksWithCumulative.reverse(), false).reverse(), // Reverse for correct order
      spread: orderbook.spread,
      midPrice: orderbook.midPrice,
      maxQuantity,
      lastUpdate: orderbook.lastUpdate
    };
  }, [orderbook, maxLevels, showFullBook, grouping, groupOrdersByPrice]);
  
  /**
   * Render individual order level
   */
  const renderOrderLevel = useCallback((
    order: any,
    side: 'bid' | 'ask',
    index: number
  ) => {
    const percentage = (order.quantity / processedOrderbook!.maxQuantity) * 100;
    const isLargeOrder = percentage > 30;
    const isPriceNearPosition = currentPosition && 
      Math.abs(order.price - currentPosition.entryPrice.toNumber()) < 10;
    
    // Create motion props conditionally
    const motionProps = animateChanges ? {
      initial: { opacity: 0, x: side === 'bid' ? -20 : 20 },
      animate: { opacity: 1, x: 0 },
      exit: { opacity: 0, x: side === 'bid' ? -20 : 20 },
      transition: { duration: 0.2, delay: index * 0.01 }
    } : {
      initial: false,
      animate: { opacity: 1, x: 0 }
    };
    
    return (
      <motion.div
        key={`${side}-${order.price}`}
        {...motionProps}
        className={`
          relative flex items-center justify-between px-2 py-1 
          cursor-pointer transition-all duration-200
          hover:bg-gray-800/50 hover:border-l-2
          ${side === 'bid' ? 'hover:border-l-green-400' : 'hover:border-l-red-400'}
          ${isPriceNearPosition ? 'ring-1 ring-yellow-500/50' : ''}
          ${order.isSpoofCandidate ? 'opacity-60' : ''}
        `}
        onClick={() => onPriceClick?.(order.price, side)}
      >
        {/* Background fill - the visual weight */}
        <div
          className={`
            absolute inset-0 opacity-20
            ${side === 'bid' ? 'bg-gradient-to-r from-transparent to-green-500' : 'bg-gradient-to-r from-transparent to-red-500'}
          `}
          style={{ width: `${percentage}%` }}
        />
        
        {/* Price */}
        <div className={`
          relative z-10 font-mono text-sm
          ${side === 'bid' ? 'text-green-400' : 'text-red-400'}
          ${isLargeOrder && highlightLarge ? 'font-bold text-glow' : ''}
        `}>
          {order.price.toFixed(2)}
        </div>
        
        {/* Quantity */}
        <div className="relative z-10 font-mono text-sm text-gray-300">
          {order.quantity.toFixed(4)}
        </div>
        
        {/* Order count (if available) */}
        {theme === 'detailed' && order.orderCount && (
          <div className="relative z-10 text-xs text-gray-500">
            ({order.orderCount})
          </div>
        )}
        
        {/* Spoof warning - Fixed with wrapper span */}
        {order.isSpoofCandidate && (
          <span 
            className="relative z-10 ml-1 inline-flex items-center"
            title="Potential spoof - Large order far from market"
          >
            <AlertTriangle className="w-3 h-3 text-yellow-500 animate-pulse" />
          </span>
        )}
      </motion.div>
    );
  }, [
    processedOrderbook,
    currentPosition,
    animateChanges,
    highlightLarge,
    theme,
    onPriceClick
  ]);
  
  // Loading state - cyberpunk style
  if (!processedOrderbook) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900/50 rounded-lg border border-cyan-900/30">
        <div className="flex flex-col items-center gap-2">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Activity className="w-6 h-6 text-cyan-400" />
          </motion.div>
          <span className="text-xs text-cyan-400/60 font-mono">SYNCING ORDERBOOK...</span>
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex flex-col h-full bg-gray-900/50 rounded-lg border border-cyan-900/30 shadow-lg shadow-cyan-900/10">
      {/* Header - Cyberpunk enhanced */}
      <div className="flex items-center justify-between p-3 border-b border-cyan-900/30 bg-gradient-to-r from-gray-900/80 to-gray-800/50">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-bold text-cyan-400 tracking-wider">ORDER BOOK</h3>
          <span className="text-xs text-gray-500 font-mono">{symbol}</span>
          <span className="text-xs text-gray-600 font-mono animate-pulse">●</span>
        </div>
        
        {/* Controls - Enhanced UI */}
        <div className="flex items-center gap-2">
          {/* Grouping selector */}
          <select
            value={grouping}
            onChange={(e) => setGrouping(parseFloat(e.target.value))}
            className="bg-gray-800 text-xs px-2 py-1 rounded border border-gray-700 
                     focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30
                     transition-all duration-200"
          >
            <option value="0">RAW DATA</option>
            <option value="0.01">GROUP: 0.01</option>
            <option value="0.1">GROUP: 0.10</option>
            <option value="1">GROUP: 1.00</option>
            <option value="10">GROUP: 10.00</option>
          </select>
          
          {/* Toggle animations */}
          <button
            onClick={() => setAnimateChanges(!animateChanges)}
            className={`p-1.5 rounded transition-all duration-200 ${
              animateChanges 
                ? 'text-cyan-400 bg-cyan-900/20 shadow-inner shadow-cyan-400/20' 
                : 'text-gray-500 hover:text-gray-400'
            }`}
            title={animateChanges ? "Animations ON" : "Animations OFF"}
          >
            <Zap className="w-4 h-4" />
          </button>
          
          {/* Toggle full book */}
          <button
            onClick={() => setShowFullBook(!showFullBook)}
            className={`p-1.5 rounded transition-all duration-200 ${
              showFullBook 
                ? 'text-cyan-400 bg-cyan-900/20 shadow-inner shadow-cyan-400/20' 
                : 'text-gray-500 hover:text-gray-400'
            }`}
            title={showFullBook ? "Full book view" : "Limited view"}
          >
            {showFullBook ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>
      
      {/* Spread indicator - the gap where fortunes fall */}
      <div className="px-3 py-2 bg-gray-800/50 flex items-center justify-between border-b border-gray-800">
        <span className="text-xs text-gray-400 font-mono uppercase">Spread</span>
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-yellow-400">
            ${processedOrderbook.spread.toFixed(2)}
          </span>
          <span className="text-xs font-mono text-yellow-400/60">
            ({((processedOrderbook.spread / processedOrderbook.midPrice) * 100).toFixed(3)}%)
          </span>
        </div>
      </div>
      
      {/* Order levels */}
      <div className="flex-1 overflow-hidden flex">
        {/* Bids - Buy side */}
        <div className="flex-1 flex flex-col border-r border-gray-800">
          <div className="px-3 py-1.5 text-xs text-gray-500 flex justify-between bg-green-900/10">
            <span className="font-semibold">BID PRICE</span>
            <span className="font-semibold">SIZE</span>
            {theme === 'detailed' && <span className="font-semibold">ORDERS</span>}
          </div>
          
          <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            <AnimatePresence mode="popLayout">
              {processedOrderbook.bids.map((bid, index) => 
                renderOrderLevel(bid, 'bid', index)
              )}
            </AnimatePresence>
          </div>
        </div>
        
        {/* Mid price divider - Enhanced cyberpunk style */}
        <div className="w-px bg-gradient-to-b from-cyan-900/0 via-cyan-900/50 to-cyan-900/0 relative">
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 
                        bg-gray-900 border border-cyan-900/50 px-3 py-1.5 rounded-full
                        shadow-lg shadow-cyan-900/20">
            <span className="text-sm font-mono font-bold text-cyan-400">
              ${processedOrderbook.midPrice.toFixed(2)}
            </span>
          </div>
        </div>
        
        {/* Asks - Sell side */}
        <div className="flex-1 flex flex-col">
          <div className="px-3 py-1.5 text-xs text-gray-500 flex justify-between bg-red-900/10">
            <span className="font-semibold">ASK PRICE</span>
            <span className="font-semibold">SIZE</span>
            {theme === 'detailed' && <span className="font-semibold">ORDERS</span>}
          </div>
          
          <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            <AnimatePresence mode="popLayout">
              {processedOrderbook.asks.map((ask, index) => 
                renderOrderLevel(ask, 'ask', index)
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      
      {/* Footer stats - Enhanced metrics */}
      <div className="px-3 py-2 border-t border-cyan-900/30 bg-gradient-to-r from-gray-900/80 to-gray-800/50">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3 text-green-400" />
              <span className="text-gray-400">Bid Vol:</span>
              <span className="font-mono text-green-400">
                {processedOrderbook.bids.reduce((sum, b) => sum + b.quantity, 0).toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <TrendingDown className="w-3 h-3 text-red-400" />
              <span className="text-gray-400">Ask Vol:</span>
              <span className="font-mono text-red-400">
                {processedOrderbook.asks.reduce((sum, a) => sum + a.quantity, 0).toFixed(2)}
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-1 text-gray-500">
            <Activity className="w-3 h-3" />
            <span className="font-mono">
              {new Date(processedOrderbook.lastUpdate).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

OrderBook.displayName = 'OrderBook';

// Add CSS for glow effect (add to your global styles)
const glowStyles = `
  .text-glow {
    text-shadow: 0 0 8px currentColor, 0 0 16px currentColor;
  }
`;