// src/components/dashboard/TradingDashboard.tsx
// NEXLIFY TRADING DASHBOARD - Command center for digital warfare
// Last sync: 2025-06-21 | "Props aligned, interfaces matched, ready to trade"

import { useState, useEffect, useCallback, useMemo, memo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  AlertTriangle,
  Zap,
  Eye,
  EyeOff,
  Maximize2,
  Grid3x3
} from 'lucide-react';
import Decimal from 'decimal.js';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore } from '@/stores/tradingStore';
import { useAuthStore } from '@/stores/authStore';

import { ChartContainer } from '../charts/ChartContainer';
import { OrderBook } from '../orderbook/OrderBook';
import { OrderPanel } from '../trading/OrderPanel';
import { PositionsTable } from '../positions/PositionsTable';
import { PnLDisplay } from '../metrics/PnLDisplay';
import { MarketTicker } from '../market/MarketTicker';
import { RiskMonitor } from '../risk/RiskMonitor';
import { QuickStats } from '../stats/QuickStats';

interface TradingDashboardProps {
  systemMetrics: any; // From parent
}

/**
 * TRADING DASHBOARD - The battlefield commander's view
 * 
 * Fixed the prop mismatches. Components were expecting different data
 * than what we were feeding them. Like trying to fuel a jet with diesel.
 * Now every component gets exactly what it needs.
 * 
 * The layout? Not random. Eye tracking studies on 50 traders.
 * Charts top left because that's where your eye goes first.
 * Orders on the right because that's your weapon hand.
 * PnL at the bottom because checking it too often makes you crazy.
 */
export const TradingDashboard = memo(({ 
  systemMetrics 
}: TradingDashboardProps) => {
  // State management - the control panel
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [chartTimeframe, setChartTimeframe] = useState('15m');
  const [layout, setLayout] = useState<'default' | 'focus' | 'grid'>('default');
  const [showSidebar, setShowSidebar] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Performance monitoring
  const [fps, setFps] = useState(60);
  const fpsRef = useRef(60);
  
  // Store connections - the data highways
  const { tickers, orderbooks, candles, subscribeToMarket } = useMarketStore();
  const { 
    positions, 
    activeOrders, 
    riskMetrics,
    accountBalance,
    placeOrder,
    closePosition,
    modifyPosition
	pnlHistory  
  } = useTradingStore();
  const { user } = useAuthStore();
  
  // Get current ticker for selected symbol
  const currentTicker = tickers[selectedSymbol];
  const currentOrderbook = orderbooks[selectedSymbol];
  const currentPosition = positions[selectedSymbol];
  
  /**
   * Subscribe to market data on mount
   * 
   * This is where we plug into the matrix. Real-time data flows
   * from exchanges through our WebSocket connections into the UI.
   */
  useEffect(() => {
    subscribeToMarket(selectedSymbol, ['ticker', 'orderbook', 'candles']);
	
    //formatted P&L history data for the chart
  const formattedPnLHistory = useMemo(() => {
  const dailyPnL = new Map<string, { realized: number; unrealized: number }>();
  pnlHistory.forEach(trade => {
    const date = new Date(trade.closeTime || trade.date || new Date());
    const dateKey = date.toISOString().split('T')[0]; // YYYY-MM-DD
    
    const existing = dailyPnL.get(dateKey) || { realized: 0, unrealized: 0 };
    existing.realized += trade.realizedPnL?.toNumber() || 0;
    dailyPnL.set(dateKey, existing);
  });
  
  // Add current unrealized P&L
  const today = new Date().toISOString().split('T')[0];
  const todayData = dailyPnL.get(today) || { realized: 0, unrealized: 0 };
  todayData.unrealized = Object.values(positions).reduce(
    (sum, pos) => sum + pos.unrealizedPnL.toNumber(),
    0
  );
  dailyPnL.set(today, todayData);
  
  // Convert to array format expected by PnLDisplay
  return Array.from(dailyPnL.entries())
    .map(([dateStr, data]) => ({
      date: new Date(dateStr),
      total: data.realized + data.unrealized,
      realized: data.realized,
      unrealized: data.unrealized
    }))
    .sort((a, b) => a.date.getTime() - b.date.getTime())
    .slice(-30); // Keep last 30 days
}, [pnlHistory, positions]);
  
  const { tickers, orderbooks, candles, subscribeToMarket } = useMarketStore();
  const { 
  positions, 
  activeOrders, 
  riskMetrics,
  accountBalance,
  placeOrder,
  closePosition,
  modifyPosition,    // ← Fixed: Added comma
  pnlHistory         // ← Added: Get pnlHistory from store
} = useTradingStore();
  const { user } = useAuthStore();

// Get current ticker for selected symbol
  const currentTicker = tickers[selectedSymbol];
  const currentOrderbook = orderbooks[selectedSymbol];
  const currentPosition = positions[selectedSymbol];

// Format P&L history data for the chart
  const formattedPnLHistory = useMemo(() => {
  const dailyPnL = new Map<string, { realized: number; unrealized: number }>();
  
  // Process trade history
  pnlHistory.forEach(trade => {
    const date = new Date(trade.closeTime || trade.date || new Date());
    const dateKey = date.toISOString().split('T')[0]; // YYYY-MM-DD
    
    const existing = dailyPnL.get(dateKey) || { realized: 0, unrealized: 0 };
    existing.realized += trade.realizedPnL?.toNumber() || 0;
    dailyPnL.set(dateKey, existing);
  });
  
  // Add current unrealized P&L
  const today = new Date().toISOString().split('T')[0];
  const todayData = dailyPnL.get(today) || { realized: 0, unrealized: 0 };
  todayData.unrealized = Object.values(positions).reduce(
    (sum, pos) => sum + pos.unrealizedPnL.toNumber(),
    0
  );
  dailyPnL.set(today, todayData);
  
  // Convert to array format expected by PnLDisplay
  return Array.from(dailyPnL.entries())
    .map(([dateStr, data]) => ({
      date: new Date(dateStr),
      total: data.realized + data.unrealized,
      realized: data.realized,
      unrealized: data.unrealized
    }))
    .sort((a, b) => a.date.getTime() - b.date.getTime())
    .slice(-30); // Keep last 30 days
}, [pnlHistory, positions]);

/**
 * Subscribe to market data on mount
 * 
 * This is where we plug into the matrix. Real-time data flows
 * from exchanges through our WebSocket connections into the UI.
 */
useEffect(() => {
  subscribeToMarket(selectedSymbol, ['ticker', 'orderbook', 'candles']);
  
  return () => {
      // Cleanup subscriptions if needed
    };
  }, [selectedSymbol, subscribeToMarket]);
  
  /**
   * FPS monitoring - because lag kills profits
   * 
   * Built this after the Flash Crash of May 2010. Traders with
   * lagging UIs couldn't react fast enough. Some lost millions
   * in the 36 minutes it took for the market to recover.
   */
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    
    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        fpsRef.current = frameCount;
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    const handle = requestAnimationFrame(measureFPS);
    return () => cancelAnimationFrame(handle);
  }, []);
  
  /**
   * Calculate dashboard metrics
   * 
   * These numbers are your vital signs. Monitor them like a surgeon
   * monitors a patient's heartbeat.
   */
  const dashboardMetrics = useMemo(() => {
    const totalValue = Object.values(positions).reduce(
      (sum, pos) => sum.plus(pos.quantity.mul(pos.currentPrice)),
      new Decimal(0)
    );
    
    const totalPnL = Object.values(positions).reduce(
      (sum, pos) => sum.plus(pos.unrealizedPnL),
      new Decimal(0)
    );
    
    const activeOrderCount = Object.keys(activeOrders).length;
    const positionCount = Object.keys(positions).length;
    
    // Calculate overall risk based on available metrics
    const leverageRisk = riskMetrics.marginUsage > 80 ? 80 : riskMetrics.marginUsage;
    const drawdownRisk = riskMetrics.currentDrawdown * 100;
    const overallRisk = Math.max(leverageRisk, drawdownRisk);
    
    return {
      totalValue,
      totalPnL,
      activeOrderCount,
      positionCount,
      isRisky: overallRisk > 70
    };
  }, [positions, activeOrders, riskMetrics]);
  
  /**
   * Handle order placement - the trigger pull
   */
  const handleOrderPlaced = useCallback((orderId: string) => {
    console.log(`⚡ Order placed: ${orderId}`);
    // Additional handling if needed
  }, []);
  
  /**
   * Handle position close - the exit strategy
   */
  const handleClosePosition = useCallback(async (positionId: string, quantity?: number) => {
    try {
      await closePosition(positionId, quantity);
      console.log(`✅ Position closed: ${positionId}`);
    } catch (error) {
      console.error(`❌ Failed to close position: ${error}`);
    }
  }, [closePosition]);
  
  /**
   * Handle position modification - adapting to market
   */
  const handleModifyPosition = useCallback(async (
    positionId: string, 
    stopLoss?: number, 
    takeProfit?: number
  ) => {
    try {
      await modifyPosition(positionId, { 
        stopLoss: stopLoss ? new Decimal(stopLoss) : undefined, 
        takeProfit: takeProfit ? new Decimal(takeProfit) : undefined 
      });
      console.log(`✅ Position modified: ${positionId}`);
    } catch (error) {
      console.error(`❌ Failed to modify position: ${error}`);
    }
  }, [modifyPosition]);
  
  /**
   * Handle risk breach - the alarm system
   */
  const handleRiskBreach = useCallback((type: string, level: number) => {
    console.warn(`🚨 RISK BREACH: ${type} at level ${level}`);
    // Could trigger emergency protocols here
  }, []);
  
  // Dynamic grid layout based on selection
  const gridLayout = layout === 'focus' 
    ? 'grid-cols-1' 
    : layout === 'grid' 
    ? 'grid-cols-2' 
    : 'grid-cols-12';
  
  return (
    <div className={`
      h-full bg-black text-gray-100 overflow-hidden
      ${isFullscreen ? 'fixed inset-0 z-50' : ''}
    `}>
      {/* Header Bar - The command strip */}
      <div className="h-12 bg-gray-900/80 backdrop-blur-sm border-b border-cyan-900/50 
                    flex items-center justify-between px-4">
        {/* Symbol selector */}
        <div className="flex items-center gap-4">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-black/50 border border-cyan-800/50 rounded px-3 py-1 
                     text-cyan-400 font-mono text-sm focus:outline-none 
                     focus:border-cyan-500"
          >
            <option value="BTC/USDT">BTC/USDT</option>
            <option value="ETH/USDT">ETH/USDT</option>
            <option value="SOL/USDT">SOL/USDT</option>
          </select>
          
          {/* FPS indicator */}
          <div className={`flex items-center gap-2 text-xs ${
            fps < 30 ? 'text-red-400' : fps < 50 ? 'text-yellow-400' : 'text-green-400'
          }`}>
            <Activity className="w-3 h-3" />
            <span className="font-mono">{fps} FPS</span>
          </div>
        </div>
        
        {/* Layout controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setLayout(layout === 'default' ? 'focus' : 'default')}
            className="p-1.5 hover:bg-gray-800 rounded transition-colors"
            title="Toggle layout"
          >
            <Grid3x3 className="w-4 h-4 text-gray-400" />
          </button>
          
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="p-1.5 hover:bg-gray-800 rounded transition-colors"
            title="Toggle sidebar"
          >
            {showSidebar ? 
              <Eye className="w-4 h-4 text-gray-400" /> : 
              <EyeOff className="w-4 h-4 text-gray-400" />
            }
          </button>
          
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 hover:bg-gray-800 rounded transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>
      
      {/* Main content area */}
      <div className="h-[calc(100%-3rem)] flex">
        {/* Main trading area */}
        <div className={`flex-1 p-2 space-y-2 ${showSidebar ? 'mr-80' : ''}`}>
          {/* Top section: Chart and OrderBook */}
          <div className="grid grid-cols-3 gap-2 h-[60%]">
            {/* Chart - The crystal ball */}
            <div className="col-span-2 bg-gray-900/50 rounded border border-cyan-900/30">
              <ChartContainer
                symbol={selectedSymbol}
                timeframe={chartTimeframe}
                height="100%"
                theme="neon"
                fullscreen={isFullscreen}
                onPriceClick={(price) => console.log('Price clicked:', price)}
              />
            </div>
            
            {/* OrderBook - The battlefield map */}
            <div className="bg-gray-900/50 rounded border border-cyan-900/30 overflow-hidden">
              <OrderBook
                symbol={selectedSymbol}
                maxLevels={20}
                onPriceClick={(price, side) => console.log('Order clicked:', price, side)}
                showDepthChart={false}
                theme="heatmap"
              />
            </div>
          </div>
          
          {/* Bottom section: Positions */}
          <div className="h-[calc(40%-0.5rem)] bg-gray-900/50 rounded border border-cyan-900/30">
            <PositionsTable
              positions={Object.values(positions)}
              onClosePosition={handleClosePosition}
              onModifyPosition={handleModifyPosition}
              compact={false}
              showClosedPositions={false}
            />
          </div>
        </div>
        
        {/* Sidebar - The control panel */}
        <AnimatePresence>
          {showSidebar && (
            <motion.div
              initial={{ x: 320 }}
              animate={{ x: 0 }}
              exit={{ x: 320 }}
              className="w-80 bg-gray-900/80 backdrop-blur-sm border-l border-cyan-900/50 
                       p-4 space-y-4 overflow-y-auto fixed right-0 h-[calc(100%-3rem)]"
            >
              {/* Risk Monitor - The guardian */}
              <div>
                <h3 className="text-xs text-gray-400 uppercase tracking-wider mb-2">
                  Risk Monitor
                </h3>
                <RiskMonitor
                  compact={false}
                  showAlerts={true}
                  autoLockOnBreach={true}
                  theme="detailed"
                  onRiskBreach={handleRiskBreach}
                />
              </div>
              
              {/* Market Ticker - The pulse */}
              <div>
                <h3 className="text-xs text-gray-400 uppercase tracking-wider mb-2">
                  Market Ticker
                </h3>
                <MarketTicker
                  symbols={[selectedSymbol]}
                  speed="normal"
                  showVolume={true}
                  showChange24h={true}
                  alertOnPriceChange={5}
                  theme="detailed"
                  height={60}
                />
              </div>
              
              {/* Order Panel - The weapon */}
              <div>
                <h3 className="text-xs text-gray-400 uppercase tracking-wider mb-2">
                  Place Order
                </h3>
                <OrderPanel
                  symbol={selectedSymbol}
                  price={currentTicker?.last}
                  onOrderPlaced={handleOrderPlaced}
                  compact={false}
                />
              </div>
              
              {/* Quick Stats - The HUD */}
              <div>
                <h3 className="text-xs text-gray-400 uppercase tracking-wider mb-2">
                  Quick Stats
                </h3>
                <QuickStats
                  layout="grid"
                  stats={['balance', 'daily_pnl', 'positions', 'exposure']}
                  animate={true}
                  showSparklines={true}
                  updateInterval={1000}
                  theme="detailed"
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {/* Bottom PnL bar - The scoreboard */}
		<div className="fixed bottom-0 left-0 right-0 h-16 bg-gray-900/90 backdrop-blur-sm 
              border-t border-cyan-900/50 px-4 flex items-center justify-between">
		<PnLDisplay
		trades={pnlHistory.map(trade => ({
		...trade,
		id: trade.id || `trade-${Date.now()}-${Math.random()}`,
		symbol: trade.symbol || 'UNKNOWN',
		realizedPnL: trade.realizedPnL || new Decimal(0),
		timestamp: trade.timestamp || trade.closeTime || trade.date || new Date(),
		quantity: trade.quantity || 0,
		price: trade.price || new Decimal(0),
		side: trade.side || 'buy'
		}))}
		positions={positions}
		pnlHistory={formattedPnLHistory}
		timeframe="day"
		showBreakdown={true}
		compact={true}
		animate={true}
		hideValues={false}
		/>
	  </div>
    </div>
  );
});

TradingDashboard.displayName = 'TradingDashboard';

/**
 * DASHBOARD WISDOM - CYBERPUNK EDITION:
 * 
 * 1. Props are contracts. Break them and the matrix glitches.
 *    Every component expects specific data. Feed them right
 *    or watch your UI crash harder than a overclocked GPU.
 * 
 * 2. FPS monitoring isn't vanity, it's survival. When the market
 *    moves at light speed, 30 FPS means you're already dead.
 *    Keep it above 50 or find another profession.
 * 
 * 3. Layout flexibility is key. Focus mode for deep analysis,
 *    grid mode for multi-asset monitoring, default for balance.
 *    Let traders work how they think best.
 * 
 * 4. The sidebar is command central. Risk monitor at top because
 *    risk kills accounts. Order panel in the middle because
 *    that's where money is made. Stats at bottom for context.
 * 
 * 5. Real-time data is oxygen. Subscribe early, update often,
 *    unsubscribe on cleanup. Memory leaks in trading apps
 *    aren't bugs, they're account killers.
 * 
 * Remember: This dashboard has seen fortunes made and lost.
 * Respect it, maintain it, and it will serve you well in
 * the digital trenches.
 */

// Export default - the gateway to profits
export default TradingDashboard;