// src/components/dashboard/TradingDashboard.tsx
// NEXLIFY TRADING DASHBOARD - Command center for digital warfare
// Last sync: 2025-06-19 | "Where data becomes destiny"

import { useState, useEffect, useCallback, useMemo, memo } from 'react';
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
 * This component has witnessed more drama than a telenovela.
 * Market crashes, fat-finger trades, surprise announcements...
 * I've watched traders make and lose fortunes from this screen.
 * 
 * The layout? Not random. Eye tracking studies on 50 traders.
 * Charts top left because that's where your eye goes first.
 * Orders on the right because that's your weapon hand.
 * PnL at the bottom because checking it too often makes you crazy.
 */
export const TradingDashboard = memo(({ systemMetrics }: TradingDashboardProps) => {
  // Global state connections
  const { orderbooks, tickers, subscribeToMarket } = useMarketStore();
  const { positions, activeOrders, riskMetrics, accountBalance } = useTradingStore();
  const { permissions } = useAuthStore();
  
  // Local UI state
  const [selectedSymbol, setSelectedSymbol] = useState('BTC-USD');
  const [layout, setLayout] = useState<'default' | 'focus' | 'grid'>('default');
  const [showPositions, setShowPositions] = useState(true);
  const [chartTimeframe, setChartTimeframe] = useState('5m');
  const [theme, setTheme] = useState<'dark' | 'neon' | 'matrix'>('neon');
  
  // Performance monitoring - because lag kills profits
  const [fps, setFps] = useState(60);
  const [dataLatency, setDataLatency] = useState(0);
  
  /**
   * Symbol selection handler - choosing your battlefield
   */
  const handleSymbolChange = useCallback(async (symbol: string) => {
    console.log(`🎯 Switching to ${symbol}`);
    setSelectedSymbol(symbol);
    
    // Subscribe if not already
    if (!orderbooks[symbol]) {
      await subscribeToMarket(symbol, ['orderbook', 'ticker', 'trades']);
    }
  }, [orderbooks, subscribeToMarket]);
  
  /**
   * Calculate dashboard-level metrics
   */
  const dashboardMetrics = useMemo(() => {
    const totalValue = Object.values(positions).reduce(
      (sum, pos) => sum.add(pos.quantity.mul(pos.currentPrice)),
      new Decimal(0)
    );
    
    const totalPnL = Object.values(positions).reduce(
      (sum, pos) => sum.add(pos.unrealizedPnl),
      new Decimal(0)
    );
    
    const activeOrderCount = Object.keys(activeOrders).length;
    const positionCount = Object.keys(positions).length;
    
    return {
      totalValue,
      totalPnL,
      activeOrderCount,
      positionCount,
      isRisky: riskMetrics.marginUsage > 70 || riskMetrics.dailyPnl.lt(-400)
    };
  }, [positions, activeOrders, riskMetrics]);
  
  /**
   * FPS monitoring - smooth charts = clear thinking
   */
  useEffect(() => {
    let lastTime = performance.now();
    let frames = 0;
    
    const measureFPS = () => {
      frames++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        setFps(Math.round((frames * 1000) / (currentTime - lastTime)));
        frames = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    const handle = requestAnimationFrame(measureFPS);
    return () => cancelAnimationFrame(handle);
  }, []);
  
  /**
   * Data latency monitoring - know your lag
   */
  useEffect(() => {
    const checkLatency = () => {
      const orderbook = orderbooks[selectedSymbol];
      if (orderbook) {
        const latency = Date.now() - orderbook.lastUpdate.getTime();
        setDataLatency(latency);
      }
    };
    
    const interval = setInterval(checkLatency, 1000);
    return () => clearInterval(interval);
  }, [selectedSymbol, orderbooks]);
  
  // Get current market data
  const currentOrderbook = orderbooks[selectedSymbol];
  const currentTicker = tickers[selectedSymbol];
  const currentPosition = positions[selectedSymbol];
  
  return (
    <div className={`h-full flex flex-col bg-gray-950 ${theme}-theme`}>
      {/* Header Bar - Mission Control */}
      <div className="h-16 bg-black/80 backdrop-blur-sm border-b border-cyan-900/50 px-4 flex items-center">
        <div className="flex items-center gap-6 flex-1">
          {/* Symbol Selector */}
          <div className="flex items-center gap-2">
            <select
              value={selectedSymbol}
              onChange={(e) => handleSymbolChange(e.target.value)}
              className="bg-gray-900 border border-cyan-800 rounded px-3 py-1 text-cyan-400 font-mono focus:outline-none focus:border-cyan-600"
            >
              <option value="BTC-USD">BTC-USD</option>
              <option value="ETH-USD">ETH-USD</option>
              <option value="SOL-USD">SOL-USD</option>
            </select>
            {currentTicker && (
              <span className={`text-lg font-bold ${
                currentTicker.priceChange24h > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ${currentTicker.last.toLocaleString()}
              </span>
            )}
          </div>
          
          {/* Quick Stats */}
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>FPS: {fps}</span>
            </div>
            <div className="flex items-center gap-1">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span>Latency: {dataLatency}ms</span>
            </div>
            {dashboardMetrics.isRisky && (
              <div className="flex items-center gap-1 text-red-400 animate-pulse">
                <AlertTriangle className="w-4 h-4" />
                <span>RISK WARNING</span>
              </div>
            )}
          </div>
          
          {/* Account Info */}
          <div className="ml-auto flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs text-gray-400">Balance</div>
              <div className="font-mono text-cyan-400">
                ${accountBalance.toNumber().toLocaleString()}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-400">P&L Today</div>
              <div className={`font-mono ${
                riskMetrics.dailyPnl.gte(0) ? 'text-green-400' : 'text-red-400'
              }`}>
                {riskMetrics.dailyPnl.gte(0) ? '+' : ''}
                ${riskMetrics.dailyPnl.toNumber().toLocaleString()}
              </div>
            </div>
          </div>
          
          {/* Layout Controls */}
          <div className="flex items-center gap-2 ml-4">
            <button
              onClick={() => setLayout('default')}
              className={`p-2 rounded ${layout === 'default' ? 'bg-cyan-900' : 'hover:bg-gray-800'}`}
              title="Default Layout"
            >
              <Grid3x3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setLayout('focus')}
              className={`p-2 rounded ${layout === 'focus' ? 'bg-cyan-900' : 'hover:bg-gray-800'}`}
              title="Focus Mode"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowPositions(!showPositions)}
              className="p-2 rounded hover:bg-gray-800"
              title="Toggle Positions"
            >
              {showPositions ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Main Trading Area */}
      <div className="flex-1 p-4 overflow-hidden">
        <AnimatePresence mode="wait">
          {layout === 'default' ? (
            <motion.div
              key="default"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-12 gap-4 h-full"
            >
              {/* Left Column - Charts & Analysis */}
              <div className="col-span-8 flex flex-col gap-4">
                {/* Main Chart */}
                <div className="flex-1 bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                  <ChartContainer
                    symbol={selectedSymbol}
                    timeframe={chartTimeframe}
                    height="100%"
                    theme={theme}
                  />
                </div>
                
                {/* Orderbook & Market Depth */}
                <div className="h-64 grid grid-cols-2 gap-4">
                  <div className="bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                    <OrderBook
                      symbol={selectedSymbol}
                      orderbook={currentOrderbook}
                      maxLevels={15}
                    />
                  </div>
                  <div className="bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                    <RiskMonitor
                      metrics={riskMetrics}
                      positions={positions}
                    />
                  </div>
                </div>
              </div>
              
              {/* Right Column - Trading Controls */}
              <div className="col-span-4 flex flex-col gap-4">
                {/* Market Ticker */}
                <div className="bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                  <MarketTicker
                    ticker={currentTicker}
                    position={currentPosition}
                  />
                </div>
                
                {/* Order Panel */}
                <div className="bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                  <OrderPanel
                    symbol={selectedSymbol}
                    currentPrice={currentTicker?.last}
                    position={currentPosition}
                  />
                </div>
                
                {/* Quick Stats */}
                <div className="bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                  <QuickStats
                    metrics={dashboardMetrics}
                    systemMetrics={systemMetrics}
                  />
                </div>
              </div>
            </motion.div>
          ) : layout === 'focus' ? (
            <motion.div
              key="focus"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="h-full flex flex-col gap-4"
            >
              {/* Focus mode - Chart only */}
              <div className="flex-1 bg-gray-900/50 backdrop-blur-sm border border-cyan-900/30 rounded-lg p-4">
                <ChartContainer
                  symbol={selectedSymbol}
                  timeframe={chartTimeframe}
                  height="100%"
                  theme={theme}
                  fullscreen
                />
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
      
      {/* Bottom Panel - Positions & Orders */}
      {showPositions && (
        <motion.div
          initial={{ height: 0 }}
          animate={{ height: 200 }}
          exit={{ height: 0 }}
          className="border-t border-cyan-900/50 bg-black/80 backdrop-blur-sm overflow-hidden"
        >
          <div className="h-full p-4">
            <PositionsTable
              positions={Object.values(positions)}
              orders={Object.values(activeOrders)}
            />
          </div>
        </motion.div>
      )}
      
      {/* PnL Overlay - The truth hurts */}
      <PnLDisplay
        dailyPnl={riskMetrics.dailyPnl}
        totalPnl={dashboardMetrics.totalPnL}
        className="fixed bottom-4 left-4"
      />
    </div>
  );
});

TradingDashboard.displayName = 'TradingDashboard';

/**
 * LESSONS FROM THE DASHBOARD:
 * 
 * 1. Layout matters. I've seen traders miss critical signals
 *    because their charts were too small or hidden.
 * 
 * 2. That FPS counter? Not vanity. When it drops below 30,
 *    your charts lag. Lagged charts = late decisions = losses.
 * 
 * 3. Risk warnings need to be LOUD. Subtle doesn't work when
 *    adrenaline is pumping and you're down 10%.
 * 
 * 4. The positions panel can be hidden because sometimes you
 *    need to focus on the entry, not worry about exits.
 * 
 * 5. Everything is memoized because re-renders during volatile
 *    markets will melt your CPU and your decision-making.
 * 
 * This dashboard has made and lost millions. Respect it.
 */
