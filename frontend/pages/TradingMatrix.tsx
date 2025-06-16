/**
 * Nexlify Trading Matrix - Where Chrome Meets the Street
 * The real-time trading interface where fortunes are made and lost
 */

import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  Zap, 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Crosshair,
  Shield,
  DollarSign,
  BarChart3,
  Activity
} from 'lucide-react';
import toast from 'react-hot-toast';

// Stores
import { useTradingStore } from '../stores/tradingStore';
import { useSettingsStore } from '../stores/settingsStore';

// Components
import { Card } from '../components/common/Card';
import { OrderBook } from '../components/trading/OrderBook';
import { TradingChart } from '../components/trading/TradingChart';
import { OrderForm } from '../components/trading/OrderForm';
import { OpenPositions } from '../components/trading/OpenPositions';
import { RecentTrades } from '../components/trading/RecentTrades';
import { SymbolSelector } from '../components/trading/SymbolSelector';
import { QuickPresets } from '../components/trading/QuickPresets';
import { MarketDepth } from '../components/trading/MarketDepth';

// Utils
import { formatCurrency, formatPercent, calculateRiskReward } from '../lib/utils';
import { playSound } from '../lib/sounds';

// Types
interface TradingViewConfig {
  layout: 'standard' | 'pro' | 'minimal';
  showOrderBook: boolean;
  showTradeHistory: boolean;
  showDepth: boolean;
  chartHeight: number;
}

export const TradingMatrix: React.FC = () => {
  const { 
    marketData,
    positions,
    placeOrder,
    cancelOrder,
    subscribeToSymbol,
    unsubscribeFromSymbol
  } = useTradingStore();
  
  const { 
    defaultExchange,
    favoriteSymbols,
    confirmOrders,
    defaultOrderType,
    riskLevel,
    soundEnabled
  } = useSettingsStore();

  // Local state - keeping it street-level clean
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [selectedExchange, setSelectedExchange] = useState(defaultExchange);
  const [viewConfig, setViewConfig] = useState<TradingViewConfig>({
    layout: 'standard',
    showOrderBook: true,
    showTradeHistory: true,
    showDepth: false,
    chartHeight: 500
  });
  
  // Order form state
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState(defaultOrderType);
  const [orderAmount, setOrderAmount] = useState('');
  const [orderPrice, setOrderPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');

  // Get current market data
  const currentMarket = marketData.get(selectedSymbol);
  const currentPrice = currentMarket?.price || 0;

  // Fetch order book data
  const { data: orderBookData, isLoading: orderBookLoading } = useQuery({
    queryKey: ['orderbook', selectedSymbol, selectedExchange],
    queryFn: async () => {
      const response = await apiClient.get(`/market/orderbook`, {
        params: { 
          symbol: selectedSymbol,
          exchange: selectedExchange,
          limit: 20
        }
      });
      return response.data;
    },
    refetchInterval: 2000, // Update every 2 seconds
    enabled: viewConfig.showOrderBook
  });

  // Place order mutation
  const placeOrderMutation = useMutation({
    mutationFn: async (orderData: any) => {
      if (confirmOrders) {
        const confirmed = await showOrderConfirmation(orderData);
        if (!confirmed) throw new Error('Order cancelled by user');
      }
      
      return placeOrder(orderData);
    },
    onSuccess: (order) => {
      if (order) {
        playSound('trade_execute');
        toast.success('Order placed successfully', {
          icon: '🚀',
          duration: 3000
        });
        
        // Reset form
        setOrderAmount('');
        setOrderPrice('');
        setStopLoss('');
        setTakeProfit('');
      }
    },
    onError: (error: any) => {
      playSound('error');
      toast.error(error.message || 'Order failed');
    }
  });

  // Handle symbol change
  const handleSymbolChange = useCallback((symbol: string) => {
    // Unsubscribe from old symbol
    if (selectedSymbol !== symbol) {
      unsubscribeFromSymbol(selectedSymbol);
    }
    
    // Subscribe to new symbol
    setSelectedSymbol(symbol);
    subscribeToSymbol(symbol);
    
    // Update price if available
    const marketData = marketData.get(symbol);
    if (marketData && orderType === 'limit') {
      setOrderPrice(marketData.price.toFixed(2));
    }
  }, [selectedSymbol, orderType, subscribeToSymbol, unsubscribeFromSymbol]);

  // Calculate position metrics
  const positionMetrics = useMemo(() => {
    if (!orderAmount || !currentPrice) return null;
    
    const amount = parseFloat(orderAmount);
    const price = orderType === 'limit' ? parseFloat(orderPrice) : currentPrice;
    const sl = stopLoss ? parseFloat(stopLoss) : null;
    const tp = takeProfit ? parseFloat(takeProfit) : null;
    
    const positionValue = amount * price;
    const riskAmount = sl ? Math.abs(price - sl) * amount : 0;
    const rewardAmount = tp ? Math.abs(tp - price) * amount : 0;
    const riskRewardRatio = riskAmount > 0 ? rewardAmount / riskAmount : 0;
    
    return {
      positionValue,
      riskAmount,
      rewardAmount,
      riskRewardRatio,
      riskPercent: riskAmount / positionValue * 100
    };
  }, [orderAmount, orderPrice, currentPrice, orderType, stopLoss, takeProfit]);

  // Submit order
  const handleSubmitOrder = () => {
    const orderData = {
      exchange: selectedExchange,
      symbol: selectedSymbol,
      side: orderSide,
      order_type: orderType,
      amount: parseFloat(orderAmount),
      price: orderType === 'limit' ? parseFloat(orderPrice) : undefined,
      stop_loss: stopLoss ? parseFloat(stopLoss) : undefined,
      take_profit: takeProfit ? parseFloat(takeProfit) : undefined
    };
    
    placeOrderMutation.mutate(orderData);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header Bar */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-4">
          <SymbolSelector
            value={selectedSymbol}
            onChange={handleSymbolChange}
            favorites={favoriteSymbols}
            showFavorites
          />
          
          {currentMarket && (
            <div className="flex items-center gap-6">
              <div>
                <p className="text-xs text-gray-400">Last Price</p>
                <p className="text-lg font-mono text-white">
                  {formatCurrency(currentMarket.price)}
                </p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400">24h Change</p>
                <p className={`text-lg font-mono flex items-center gap-1 ${
                  currentMarket.change24h >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {currentMarket.change24h >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  {formatPercent(currentMarket.change24h)}
                </p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400">24h Volume</p>
                <p className="text-lg font-mono text-white">
                  {formatCurrency(currentMarket.volume24h, 0)}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewConfig(prev => ({ ...prev, layout: 'minimal' }))}
            className={`px-3 py-1 text-sm rounded ${
              viewConfig.layout === 'minimal' 
                ? 'bg-cyan-500 text-black' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Minimal
          </button>
          <button
            onClick={() => setViewConfig(prev => ({ ...prev, layout: 'standard' }))}
            className={`px-3 py-1 text-sm rounded ${
              viewConfig.layout === 'standard' 
                ? 'bg-cyan-500 text-black' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Standard
          </button>
          <button
            onClick={() => setViewConfig(prev => ({ ...prev, layout: 'pro' }))}
            className={`px-3 py-1 text-sm rounded ${
              viewConfig.layout === 'pro' 
                ? 'bg-cyan-500 text-black' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Pro
          </button>
        </div>
      </div>

      {/* Main Trading Interface */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full grid grid-cols-12 gap-4 p-4">
          {/* Left Panel - Order Book & Market Depth */}
          {viewConfig.showOrderBook && (
            <div className="col-span-3 space-y-4">
              <Card title="Order Book" className="h-1/2">
                <OrderBook 
                  data={orderBookData}
                  loading={orderBookLoading}
                  currentPrice={currentPrice}
                  onPriceClick={(price) => setOrderPrice(price.toString())}
                />
              </Card>
              
              {viewConfig.showDepth && (
                <Card title="Market Depth" className="h-1/2">
                  <MarketDepth 
                    symbol={selectedSymbol}
                    exchange={selectedExchange}
                  />
                </Card>
              )}
            </div>
          )}

          {/* Center - Chart & Positions */}
          <div className={`${
            viewConfig.showOrderBook ? 'col-span-6' : 'col-span-9'
          } flex flex-col gap-4`}>
            <Card className="flex-1 min-h-0">
              <TradingChart 
                symbol={selectedSymbol}
                height={viewConfig.chartHeight}
                showIndicators
                showDrawingTools={viewConfig.layout === 'pro'}
              />
            </Card>
            
            <Card title="Open Positions" className="h-48 overflow-hidden">
              <OpenPositions 
                positions={positions.filter(p => p.symbol === selectedSymbol)}
                onClose={(positionId) => {
                  // Handle position close
                  console.log('Close position:', positionId);
                }}
              />
            </Card>
          </div>

          {/* Right Panel - Order Form & Quick Actions */}
          <div className="col-span-3 space-y-4">
            {/* Quick Presets */}
            <QuickPresets 
              onSelect={(preset) => {
                setOrderSide(preset.side);
                setOrderType(preset.orderType);
                if (preset.stopLoss) setStopLoss(preset.stopLoss.toString());
                if (preset.takeProfit) setTakeProfit(preset.takeProfit.toString());
              }}
            />

            {/* Order Form */}
            <Card title="Place Order" className="flex-1">
              <div className="space-y-4">
                {/* Buy/Sell Toggle */}
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setOrderSide('buy')}
                    className={`py-2 font-medium rounded transition-all ${
                      orderSide === 'buy'
                        ? 'bg-green-500 text-black'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    BUY
                  </button>
                  <button
                    onClick={() => setOrderSide('sell')}
                    className={`py-2 font-medium rounded transition-all ${
                      orderSide === 'sell'
                        ? 'bg-red-500 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    SELL
                  </button>
                </div>

                {/* Order Type */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setOrderType('market')}
                    className={`flex-1 py-1 text-sm rounded ${
                      orderType === 'market'
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                        : 'bg-gray-800 text-gray-400'
                    }`}
                  >
                    Market
                  </button>
                  <button
                    onClick={() => setOrderType('limit')}
                    className={`flex-1 py-1 text-sm rounded ${
                      orderType === 'limit'
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                        : 'bg-gray-800 text-gray-400'
                    }`}
                  >
                    Limit
                  </button>
                </div>

                {/* Amount Input */}
                <div>
                  <label className="text-xs text-gray-400">Amount</label>
                  <input
                    type="number"
                    value={orderAmount}
                    onChange={(e) => setOrderAmount(e.target.value)}
                    placeholder="0.00"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                {/* Price Input (for limit orders) */}
                {orderType === 'limit' && (
                  <div>
                    <label className="text-xs text-gray-400">Price</label>
                    <input
                      type="number"
                      value={orderPrice}
                      onChange={(e) => setOrderPrice(e.target.value)}
                      placeholder={currentPrice.toFixed(2)}
                      className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                )}

                {/* Risk Management */}
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <label className="text-xs text-gray-400">Stop Loss</label>
                      <input
                        type="number"
                        value={stopLoss}
                        onChange={(e) => setStopLoss(e.target.value)}
                        placeholder="Optional"
                        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:border-cyan-500 focus:outline-none"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-xs text-gray-400">Take Profit</label>
                      <input
                        type="number"
                        value={takeProfit}
                        onChange={(e) => setTakeProfit(e.target.value)}
                        placeholder="Optional"
                        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:border-cyan-500 focus:outline-none"
                      />
                    </div>
                  </div>
                </div>

                {/* Position Metrics */}
                {positionMetrics && (
                  <div className="p-3 bg-gray-900 rounded space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Position Value:</span>
                      <span className="font-mono">{formatCurrency(positionMetrics.positionValue)}</span>
                    </div>
                    {positionMetrics.riskAmount > 0 && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Risk:</span>
                          <span className="font-mono text-red-400">
                            {formatCurrency(positionMetrics.riskAmount)} ({formatPercent(positionMetrics.riskPercent)})
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">R:R Ratio:</span>
                          <span className={`font-mono ${
                            positionMetrics.riskRewardRatio >= 2 ? 'text-green-400' : 'text-yellow-400'
                          }`}>
                            1:{positionMetrics.riskRewardRatio.toFixed(2)}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmitOrder}
                  disabled={!orderAmount || (orderType === 'limit' && !orderPrice) || placeOrderMutation.isLoading}
                  className={`w-full py-3 font-medium rounded transition-all flex items-center justify-center gap-2 ${
                    orderSide === 'buy'
                      ? 'bg-green-500 hover:bg-green-600 text-black disabled:bg-green-900 disabled:text-green-700'
                      : 'bg-red-500 hover:bg-red-600 text-white disabled:bg-red-900 disabled:text-red-700'
                  }`}
                >
                  {placeOrderMutation.isLoading ? (
                    <>
                      <Activity className="w-4 h-4 animate-pulse" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Crosshair className="w-4 h-4" />
                      {orderSide === 'buy' ? 'LONG' : 'SHORT'} {selectedSymbol}
                    </>
                  )}
                </button>
              </div>
            </Card>

            {/* Recent Trades */}
            {viewConfig.showTradeHistory && (
              <Card title="Recent Trades" className="h-48">
                <RecentTrades 
                  symbol={selectedSymbol}
                  limit={10}
                />
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function for order confirmation
async function showOrderConfirmation(orderData: any): Promise<boolean> {
  // In a real app, this would show a modal
  // For now, just return true
  return true;
}
