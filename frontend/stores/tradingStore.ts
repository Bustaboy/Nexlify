/**
 * Nexlify Trading Store
 * The heart of the trading matrix - real-time market data, positions, and signals
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

// Types
import type { 
  MarketData, 
  Position, 
  Order, 
  TradingSignal, 
  Portfolio,
  PriceAlert,
  Exchange
} from '../types/trading';

// Utils
import { formatCurrency, calculatePnL } from '../lib/utils';
import { playSound } from '../lib/sounds';
import { apiClient } from '../lib/api';

interface TradingState {
  // Market Data
  marketData: Map<string, MarketData>;
  activeSymbols: string[];
  
  // Positions & Orders
  positions: Position[];
  openOrders: Order[];
  orderHistory: Order[];
  
  // Portfolio
  portfolio: Portfolio;
  balances: Map<string, number>;
  totalEquity: number;
  dailyPnL: number;
  
  // Trading Signals
  signals: TradingSignal[];
  activeStrategies: string[];
  
  // Alerts
  priceAlerts: PriceAlert[];
  
  // Connection Status
  isConnected: boolean;
  exchanges: Exchange[];
  socket: Socket | null;
  
  // Actions
  connectWebSocket: (token: string) => void;
  disconnectWebSocket: () => void;
  subscribeToSymbol: (symbol: string) => void;
  unsubscribeFromSymbol: (symbol: string) => void;
  placeOrder: (order: Partial<Order>) => Promise<Order | null>;
  cancelOrder: (orderId: string) => Promise<boolean>;
  closePosition: (positionId: string) => Promise<boolean>;
  updatePositions: () => Promise<void>;
  addPriceAlert: (alert: Partial<PriceAlert>) => void;
  removePriceAlert: (alertId: string) => void;
  setActiveStrategies: (strategies: string[]) => void;
}

// Constants
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export const useTradingStore = create<TradingState>()(
  subscribeWithSelector(
    devtools(
      immer((set, get) => ({
        // Initial state - empty streets, no chrome running
        marketData: new Map(),
        activeSymbols: ['BTC/USDT', 'ETH/USDT'], // Default pairs
        positions: [],
        openOrders: [],
        orderHistory: [],
        portfolio: {
          totalValue: 0,
          availableBalance: 0,
          marginUsed: 0,
          unrealizedPnL: 0,
          realizedPnL: 0,
          winRate: 0,
          sharpeRatio: 0
        },
        balances: new Map(),
        totalEquity: 0,
        dailyPnL: 0,
        signals: [],
        activeStrategies: [],
        priceAlerts: [],
        isConnected: false,
        exchanges: [],
        socket: null,

        // Connect to the neural net via WebSocket
        connectWebSocket: (token: string) => {
          const apiEndpoint = apiClient.defaults.baseURL || 'http://localhost:8000';
          const wsUrl = apiEndpoint.replace('http', 'ws');
          
          const socket = io(wsUrl, {
            auth: { token },
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: RECONNECT_DELAY,
            reconnectionAttempts: MAX_RECONNECT_ATTEMPTS
          });

          // Connection established - jack into the matrix
          socket.on('connect', () => {
            set((draft) => {
              draft.isConnected = true;
              draft.socket = socket as any;
            });
            
            toast.success('Neural link established', {
              icon: '🔗',
              duration: 2000
            });
            
            playSound('startup');
            
            // Subscribe to default symbols
            const { activeSymbols } = get();
            activeSymbols.forEach(symbol => {
              socket.emit('subscribe_market', { symbols: [symbol] });
            });
          });

          // Connection lost - flatlined
          socket.on('disconnect', (reason) => {
            set((draft) => {
              draft.isConnected = false;
            });
            
            toast.error(`Neural link severed: ${reason}`, {
              icon: '🔌',
            });
            
            playSound('error');
          });

          // Market data stream - the pulse of Night City's markets
          socket.on('market_update', (data: any) => {
            set((draft) => {
              const existing = draft.marketData.get(data.symbol) || {} as MarketData;
              
              const updated: MarketData = {
                ...existing,
                symbol: data.symbol,
                price: data.price,
                change24h: data.change,
                volume24h: data.volume,
                bid: data.bid || data.price,
                ask: data.ask || data.price,
                timestamp: new Date(data.timestamp)
              };
              
              draft.marketData.set(data.symbol, updated);
            });
            
            // Check price alerts
            get().checkPriceAlerts(data.symbol, data.price);
          });

          // Position updates - your stakes in the game
          socket.on('position_update', (position: Position) => {
            set((draft) => {
              const index = draft.positions.findIndex(p => p.id === position.id);
              
              if (index >= 0) {
                draft.positions[index] = position;
              } else {
                draft.positions.push(position);
              }
            });
            
            // Notify on significant P&L changes
            if (Math.abs(position.unrealizedPnL) > 100) {
              const isProfit = position.unrealizedPnL > 0;
              toast(
                `${position.symbol} ${isProfit ? 'profit' : 'loss'}: ${formatCurrency(position.unrealizedPnL)}`,
                {
                  icon: isProfit ? '💰' : '🔻',
                  style: {
                    borderColor: isProfit ? '#00ff00' : '#ff0000'
                  }
                }
              );
            }
          });

          // Order fills - when deals go through
          socket.on('order_filled', (order: Order) => {
            set((draft) => {
              // Remove from open orders
              draft.openOrders = draft.openOrders.filter(o => o.id !== order.id);
              // Add to history
              draft.orderHistory.unshift(order);
              // Keep history size manageable
              if (draft.orderHistory.length > 100) {
                draft.orderHistory = draft.orderHistory.slice(0, 100);
              }
            });
            
            playSound('trade_execute');
            
            toast.success(
              `Order filled: ${order.side.toUpperCase()} ${order.amount} ${order.symbol} @ ${formatCurrency(order.price)}`,
              { duration: 5000 }
            );
            
            // Update positions
            get().updatePositions();
          });

          // Trading signals - whispers from the AI
          socket.on('trading_signal', (signal: TradingSignal) => {
            set((draft) => {
              draft.signals.unshift(signal);
              // Keep last 50 signals
              if (draft.signals.length > 50) {
                draft.signals = draft.signals.slice(0, 50);
              }
            });
            
            // High confidence signals get special treatment
            if (signal.confidence > 0.8) {
              playSound('notification');
              
              toast(
                `🤖 AI Signal: ${signal.action.toUpperCase()} ${signal.symbol}`,
                {
                  duration: 6000,
                  style: {
                    background: '#1a1a1a',
                    border: '2px solid #ff00ff',
                    color: '#ff00ff'
                  }
                }
              );
              
              // Desktop notification for high-value signals
              if (signal.metadata?.expectedProfit && signal.metadata.expectedProfit > 500) {
                window.nexlify.notification.show({
                  title: 'High Value Trading Signal',
                  body: `${signal.strategy} suggests ${signal.action} ${signal.symbol}. Expected profit: ${formatCurrency(signal.metadata.expectedProfit)}`,
                  urgency: 'critical'
                });
              }
            }
          });

          // System status updates
          socket.on('system_status', (status: any) => {
            set((draft) => {
              draft.activeStrategies = status.active_strategies;
              draft.exchanges = status.connected_exchanges;
            });
          });
        },

        // Disconnect from the matrix
        disconnectWebSocket: () => {
          const { socket } = get();
          
          if (socket) {
            socket.disconnect();
            set((draft) => {
              draft.socket = null;
              draft.isConnected = false;
            });
          }
        },

        // Subscribe to market data feed
        subscribeToSymbol: (symbol: string) => {
          const { socket, activeSymbols } = get();
          
          if (!socket || activeSymbols.includes(symbol)) return;
          
          socket.emit('subscribe_market', { symbols: [symbol] });
          
          set((draft) => {
            draft.activeSymbols.push(symbol);
          });
          
          toast.success(`Monitoring ${symbol}`, {
            icon: '📊',
            duration: 2000
          });
        },

        // Unsubscribe from market data
        unsubscribeFromSymbol: (symbol: string) => {
          const { socket } = get();
          
          if (!socket) return;
          
          socket.emit('unsubscribe_market', { symbols: [symbol] });
          
          set((draft) => {
            draft.activeSymbols = draft.activeSymbols.filter(s => s !== symbol);
            draft.marketData.delete(symbol);
          });
        },

        // Place an order - commit to the trade
        placeOrder: async (orderData: Partial<Order>): Promise<Order | null> => {
          try {
            const response = await apiClient.post('/trading/order', orderData);
            const order = response.data;
            
            set((draft) => {
              draft.openOrders.push(order);
            });
            
            playSound('click');
            
            toast.success(`Order placed: ${order.side} ${order.symbol}`, {
              icon: '📝'
            });
            
            return order;
            
          } catch (error: any) {
            const message = error.response?.data?.detail || 'Order failed';
            
            toast.error(message, {
              duration: 5000
            });
            
            playSound('error');
            
            return null;
          }
        },

        // Cancel an order - back out of the deal
        cancelOrder: async (orderId: string): Promise<boolean> => {
          try {
            await apiClient.delete(`/trading/order/${orderId}`);
            
            set((draft) => {
              draft.openOrders = draft.openOrders.filter(o => o.id !== orderId);
            });
            
            toast.success('Order cancelled', {
              icon: '❌'
            });
            
            return true;
            
          } catch (error) {
            toast.error('Failed to cancel order');
            return false;
          }
        },

        // Close a position - cash out or cut losses
        closePosition: async (positionId: string): Promise<boolean> => {
          try {
            const position = get().positions.find(p => p.id === positionId);
            
            if (!position) {
              throw new Error('Position not found');
            }
            
            // Place market order to close
            const order = await get().placeOrder({
              exchange: position.exchange,
              symbol: position.symbol,
              side: position.side === 'long' ? 'sell' : 'buy',
              order_type: 'market',
              amount: position.amount
            });
            
            if (order) {
              toast.success(`Closing ${position.symbol} position`, {
                icon: '🏁'
              });
              
              return true;
            }
            
            return false;
            
          } catch (error) {
            toast.error('Failed to close position');
            return false;
          }
        },

        // Update all positions - refresh the portfolio
        updatePositions: async () => {
          try {
            const response = await apiClient.get('/trading/positions');
            const positions = response.data;
            
            set((draft) => {
              draft.positions = positions;
              
              // Calculate portfolio metrics
              let totalValue = 0;
              let unrealizedPnL = 0;
              
              positions.forEach((pos: Position) => {
                totalValue += pos.currentValue;
                unrealizedPnL += pos.unrealizedPnL;
              });
              
              draft.portfolio.totalValue = totalValue;
              draft.portfolio.unrealizedPnL = unrealizedPnL;
            });
            
          } catch (error) {
            console.error('Failed to update positions:', error);
          }
        },

        // Add price alert - set your tripwires
        addPriceAlert: (alert: Partial<PriceAlert>) => {
          const newAlert: PriceAlert = {
            id: `alert_${Date.now()}`,
            symbol: alert.symbol!,
            targetPrice: alert.targetPrice!,
            condition: alert.condition || 'above',
            message: alert.message || `Price alert for ${alert.symbol}`,
            active: true,
            createdAt: new Date()
          };
          
          set((draft) => {
            draft.priceAlerts.push(newAlert);
          });
          
          toast.success(`Price alert set for ${newAlert.symbol}`, {
            icon: '🔔'
          });
        },

        // Remove price alert
        removePriceAlert: (alertId: string) => {
          set((draft) => {
            draft.priceAlerts = draft.priceAlerts.filter(a => a.id !== alertId);
          });
        },

        // Set active trading strategies
        setActiveStrategies: (strategies: string[]) => {
          set((draft) => {
            draft.activeStrategies = strategies;
          });
          
          toast.success(`Active strategies: ${strategies.join(', ')}`, {
            icon: '🤖'
          });
        },

        // Internal: Check price alerts
        checkPriceAlerts: (symbol: string, price: number) => {
          const alerts = get().priceAlerts.filter(a => a.symbol === symbol && a.active);
          
          alerts.forEach(alert => {
            const triggered = alert.condition === 'above' 
              ? price >= alert.targetPrice
              : price <= alert.targetPrice;
              
            if (triggered) {
              // Deactivate alert
              set((draft) => {
                const index = draft.priceAlerts.findIndex(a => a.id === alert.id);
                if (index >= 0) {
                  draft.priceAlerts[index].active = false;
                }
              });
              
              // Notify user
              playSound('notification');
              
              toast(alert.message, {
                icon: '🚨',
                duration: 8000,
                style: {
                  background: '#1a1a1a',
                  border: '2px solid #ffff00',
                  color: '#ffff00'
                }
              });
              
              // Desktop notification
              window.nexlify.notification.show({
                title: 'Price Alert Triggered',
                body: `${alert.symbol} ${alert.condition} ${formatCurrency(alert.targetPrice)}`,
                urgency: 'critical'
              });
            }
          });
        }
      })),
      {
        name: 'NexlifyTrading'
      }
    )
  )
);

// Subscribe to important changes
useTradingStore.subscribe(
  (state) => state.portfolio.unrealizedPnL,
  (pnl) => {
    // Update tray icon with P&L status
    window.nexlify.trading.sendStatusUpdate({
      active: true,
      pnl: formatCurrency(pnl),
      positions: useTradingStore.getState().positions.length,
      alerts: useTradingStore.getState().priceAlerts.filter(a => a.active).length
    });
  }
);
