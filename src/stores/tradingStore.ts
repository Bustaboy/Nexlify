// src/stores/tradingStore.ts
// NEXLIFY TRADING STATE - Where decisions echo through eternity
// Last sync: 2025-06-21 | "Every order is a prayer, every fill an answered wish"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist, devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import Decimal from 'decimal.js';

export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit' | 'stop_loss' | 'take_profit';
Decimal.set({ precision: 18, rounding: Decimal.ROUND_DOWN });

// Export these interfaces so components can use them
export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: OrderType;
  status: 'pending' | 'open' | 'partially_filled' | 'filled' | 'cancelled' | 'rejected';
  quantity: Decimal;
  price?: Decimal;      // Limit price
  stopPrice?: Decimal;  // Stop/trigger price
  filledQuantity: Decimal;
  averageFillPrice?: Decimal;
  fees: Decimal;
  createdAt: Date;
  updatedAt: Date;
  positionId?: string;  // Links to position for stop_loss/take_profit
  metadata: Record<string, any>;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: Decimal;
  entryPrice: Decimal;
  currentPrice: Decimal;
  markPrice: Decimal;
  unrealizedPnL: Decimal;
  realizedPnL: Decimal;
  margin: Decimal;
  leverage: number;
  liquidationPrice?: Decimal;
  openTime: Date;
  openedAt: Date;
  lastUpdate: Date;
  stopLoss?: Decimal;
  takeProfit?: Decimal;
  pnlPercentage: Decimal;
}

export interface TradeHistory {
  id: string;
  orderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: Decimal;
  price: Decimal;
  fee: Decimal;
  timestamp: Date;
  closeTime: Date;
  date: Date;
  realizedPnL?: Decimal;
  openTime: Date;
}

export interface RiskMetrics {
  totalExposure: Decimal;
  marginUsage: number;
  dailyPnl: Decimal;
  dailyVolume: Decimal;
  winRate: number;
  averageWin: Decimal;
  averageLoss: Decimal;
  sharpeRatio: number;
  maxDrawdown: number;
  currentDrawdown: number;
}

export interface TradingPreferences {
  defaultOrderType: 'market' | 'limit';
  defaultTimeInForce: 'GTC' | 'GTT' | 'IOC' | 'FOK';
  confirmOrders: boolean;
  playSounds: boolean;
  autoStopLoss: boolean;
  stopLossPercent: number;
  takeProfitPercent: number;
}

export interface PnLMetrics {
  realized: number;
  unrealized: number;
  winRate: number;
  bestTrade: { symbol: string; pnl: number; date: Date };
  worstTrade: { symbol: string; pnl: number; date: Date };
}

// Also export parameter types
export interface PlaceOrderParams {
  symbol: string;
  side: 'buy' | 'sell';
  type: OrderType;
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: string;
  reduceOnly?: boolean;
  postOnly?: boolean;
  positionId?: string;  // For position-based orders
  metadata?: Record<string, any>;
}

export interface PositionSizeParams {
  balance: number;
  riskPercent: number;
  stopLossDistance: number;
  price: number;
}

interface TradingState {
  // State
  activeOrders: Record<string, Order>;
  orderHistory: Order[];
  positions: Record<string, Position>;
  closedPositions: Position[];
  trades: TradeHistory[];
  pnlHistory: TradeHistory[];
  riskMetrics: RiskMetrics;
  accountBalance: Decimal;
  buyingPower: Decimal;
  preferences: TradingPreferences;
  riskLimits: { riskLimitPerTrade: Decimal; maxDailyLoss: Decimal };
  dailyPnL: Decimal;
  winRate: number;
  profitFactor: number;
  watchlist: string[];
  isPlacingOrder: boolean;
  isLoadingPositions: boolean;
  lastError: string | null;
  
  // Actions
  placeOrder: (params: PlaceOrderParams) => Promise<Order>;
  cancelOrder: (orderId: string) => Promise<void>;
  cancelAllOrders: (symbol?: string) => Promise<void>;
  closePosition: (symbol: string, percentage?: number) => Promise<void>;
  closeAllPositions: () => Promise<void>;
  refreshPositions: () => Promise<void>;
  refreshOrders: () => Promise<void>;
  calculatePositionSize: (params: PositionSizeParams) => Decimal;
  updateRiskMetrics: () => void;
  setPreference: <K extends keyof TradingPreferences>(key: K, value: TradingPreferences[K]) => void;
  canPlaceOrder: (params: PlaceOrderParams) => { allowed: boolean; reason?: string };
  isWithinRiskLimits: () => boolean;
  getExposureBySymbol: (symbol: string) => Decimal;
  calculatePnLMetrics: () => PnLMetrics;
  setTradingLocked: (locked: boolean) => void;
  modifyPosition: (symbol: string, updates: Partial<Pick<Position, 'stopLoss' | 'takeProfit'>>) => Promise<void>;
}

export const useTradingStore = create<TradingState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        activeOrders: {},
        orderHistory: [],
        positions: {},
        closedPositions: [],
        trades: [],
        pnlHistory: [],
        riskMetrics: {
          totalExposure: new Decimal(0),
          marginUsage: 0,
          dailyPnl: new Decimal(0),
          dailyVolume: new Decimal(0),
          winRate: 0,
          averageWin: new Decimal(0),
          averageLoss: new Decimal(0),
          sharpeRatio: 0,
          maxDrawdown: 0,
          currentDrawdown: 0,
        },
        accountBalance: new Decimal(10000),
        buyingPower: new Decimal(10000),
        preferences: {
          defaultOrderType: 'limit',
          defaultTimeInForce: 'GTC',
          confirmOrders: true,
          playSounds: true,
          autoStopLoss: true,
          stopLossPercent: 2,
          takeProfitPercent: 6,
        },
        riskLimits: {
          riskLimitPerTrade: new Decimal(100),
          maxDailyLoss: new Decimal(500),
        },
        dailyPnL: new Decimal(0),
        winRate: 0,
        profitFactor: 0,
        watchlist: [],
        isPlacingOrder: false,
        isLoadingPositions: false,
        lastError: null,
        
        // Action implementations
        placeOrder: async (params: PlaceOrderParams) => {
			// Validate stop orders
			if ((params.type === 'stop' || params.type === 'stop_loss') && !params.stopPrice) {
				throw new Error('Stop price required for stop orders');
			}
  
			if (params.type === 'stop_limit' && (!params.stopPrice || !params.price)) {
				throw new Error('Stop limit orders require both stop and limit prices');
			}
  
			// Validate position-based orders
			if ((params.type === 'stop_loss' || params.type === 'take_profit') && !params.positionId) {
				throw new Error('Position ID required for position-based orders');
			}
          const { allowed, reason } = get().canPlaceOrder(params);
          if (!allowed) {
            throw new Error(reason || 'Order rejected by risk management');
          }
          
          set((draft) => {
            draft.isPlacingOrder = true;
            draft.lastError = null;
          });
          
          try {
            const quantity = new Decimal(params.quantity);
            const price = params.price ? new Decimal(params.price) : undefined;
            const stopPrice = params.stopPrice ? new Decimal(params.stopPrice) : undefined;
            
            const order: Order = {
              id: `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              symbol: params.symbol,
              side: params.side,
              type: params.type,
              status: 'pending',
              quantity,
              price,
              stopPrice,
              filledQuantity: new Decimal(0),
              fees: new Decimal(0),
              createdAt: new Date(),
              updatedAt: new Date(),
              metadata: params.metadata || {},
            };
            
            const response = await invoke<any>('place_order', {
              order: {
                symbol: params.symbol,
                side: params.side,
                type: params.type,
                quantity: params.quantity,
                price: params.price,
                stop_price: params.stopPrice,
                time_in_force: params.timeInForce,
                reduce_only: params.reduceOnly,
                post_only: params.postOnly,
                metadata: params.metadata,
              },
            });
            
            const placedOrder = {
              ...order,
              id: response.order_id,
              status: response.status.toLowerCase() as Order['status'],
            };
            
            set((draft) => {
              draft.activeOrders[placedOrder.id] = placedOrder;
              draft.isPlacingOrder = false;
            });
            
            console.log(`✅ Order placed: ${placedOrder.id}`);
            return placedOrder;
          } catch (error) {
            set((draft) => {
              draft.isPlacingOrder = false;
              draft.lastError = error instanceof Error ? error.message : 'Order failed';
            });
            throw error;
          }
        },
        
        cancelOrder: async (orderId: string) => {
          try {
            await invoke('cancel_order', { orderId });
            set((draft) => {
              const order = draft.activeOrders[orderId];
              if (order) {
                order.status = 'cancelled';
                order.updatedAt = new Date();
                draft.orderHistory.push(order);
                delete draft.activeOrders[orderId];
              }
            });
            console.log(`❌ Order cancelled: ${orderId}`);
          } catch (error) {
            set((draft) => {
              draft.lastError = error instanceof Error ? error.message : 'Cancel failed';
            });
            throw error;
          }
        },
        
        cancelAllOrders: async (symbol?: string) => {
          const orders = Object.values(get().activeOrders);
          const targetOrders = symbol 
            ? orders.filter(o => o.symbol === symbol)
            : orders;
            
          if (targetOrders.length === 0) {
            console.log('No orders to cancel');
            return;
          }
          
          console.log(`🚨 Cancelling ${targetOrders.length} orders...`);
          const results = await Promise.allSettled(
            targetOrders.map(order => get().cancelOrder(order.id))
          );
          
          const failed = results.filter(r => r.status === 'rejected').length;
          if (failed > 0) {
            throw new Error(`Failed to cancel ${failed} orders`);
          }
        },
        
        closePosition: async (symbol: string, percentage = 100) => {
          const position = get().positions[symbol];
          if (!position) {
            throw new Error(`No position found for ${symbol}`);
          }
          
          const closeQuantity = position.quantity.mul(percentage / 100);
          await get().placeOrder({
            symbol,
            side: position.side === 'long' ? 'sell' : 'buy',
            type: 'market',
            quantity: closeQuantity.toNumber(),
            metadata: { action: 'close_position', percentage },
          });
          
          console.log(`💰 Closing ${percentage}% of ${symbol} position`);
        },
        
        closeAllPositions: async () => {
          const positions = Object.values(get().positions);
          if (positions.length === 0) {
            console.log('No positions to close');
            return;
          }
          
          console.log(`🚨 CLOSING ALL ${positions.length} POSITIONS`);
          const results = await Promise.allSettled(
            positions.map(pos => get().closePosition(pos.symbol))
          );
          
          const failed = results.filter(r => r.status === 'rejected').length;
          if (failed > 0) {
            throw new Error(`Failed to close ${failed} positions`);
          }
        },
        
        refreshPositions: async () => {
          set((draft) => {
            draft.isLoadingPositions = true;
          });
          
          try {
            const response = await invoke<any>('get_positions', {});
            
            set((draft) => {
              draft.positions = {};
              response.positions.forEach((pos: any) => {
                const position: Position = {
                  id: pos.position.id || pos.position.symbol,
                  symbol: pos.position.symbol,
                  side: pos.position.side.toLowerCase() as 'long' | 'short',
                  quantity: new Decimal(pos.position.quantity),
                  entryPrice: new Decimal(pos.position.entry_price),
                  currentPrice: new Decimal(pos.position.current_price),
                  markPrice: new Decimal(pos.position.mark_price || pos.position.current_price),
                  unrealizedPnL: new Decimal(pos.position.unrealized_pnl),
                  realizedPnL: new Decimal(pos.position.realized_pnl),
                  margin: new Decimal(pos.position.margin_used),
                  leverage: pos.position.leverage || 1,
                  liquidationPrice: pos.risk_metrics.liquidation_price
                    ? new Decimal(pos.risk_metrics.liquidation_price)
                    : undefined,
                  openTime: new Date(pos.position.opened_at),
                  openedAt: new Date(pos.position.opened_at),
                  lastUpdate: new Date(pos.position.last_updated),
                  stopLoss: pos.position.stop_loss
                    ? new Decimal(pos.position.stop_loss)
                    : undefined,
                  takeProfit: pos.position.take_profit
                    ? new Decimal(pos.position.take_profit)
                    : undefined,
                  pnlPercentage: new Decimal(pos.position.pnl_percentage || 0),
                };
                draft.positions[position.symbol] = position;
              });
              draft.isLoadingPositions = false;
            });
            
            get().updateRiskMetrics();
          } catch (error) {
            set((draft) => {
              draft.isLoadingPositions = false;
              draft.lastError = error instanceof Error ? error.message : 'Failed to load positions';
            });
            throw error;
          }
        },
        
        refreshOrders: async () => {
          console.log('📋 Refreshing orders...');
          // Implementation would go here
        },
        
        calculatePositionSize: (params: PositionSizeParams) => {
          const riskAmount = new Decimal(params.balance).mul(params.riskPercent / 100);
          const shares = riskAmount.div(params.stopLossDistance);
          const positionValue = shares.mul(params.price);
          const maxPosition = new Decimal(params.balance).mul(0.5); // Max 50% of balance
          
          if (positionValue.gt(maxPosition)) {
            return maxPosition.div(params.price);
          }
          
          return shares;
        },
        
        updateRiskMetrics: () => {
          const positions = Object.values(get().positions);
          const trades = get().trades;
          
          // Calculate total exposure
          const totalExposure = positions.reduce(
            (sum, pos) => sum.add(pos.quantity.mul(pos.currentPrice)),
            new Decimal(0)
          );
          
          // Calculate daily P&L
          const today = new Date();
          today.setHours(0, 0, 0, 0);
          const todaysTrades = trades.filter(t => t.timestamp >= today);
          const dailyPnl = todaysTrades.reduce(
            (sum, trade) => sum.add(trade.realizedPnL || 0),
            new Decimal(0)
          );
          
          // Calculate win rate
          const completedTrades = trades.filter(t => t.realizedPnL !== undefined);
          const winningTrades = completedTrades.filter(t => t.realizedPnL!.gt(0));
          const winRate = completedTrades.length > 0
            ? (winningTrades.length / completedTrades.length) * 100
            : 0;
          
          // Calculate profit factor
          const totalWins = winningTrades.reduce(
            (sum, t) => sum.add(t.realizedPnL || 0),
            new Decimal(0)
          );
          const totalLosses = completedTrades
            .filter(t => t.realizedPnL!.lt(0))
            .reduce((sum, t) => sum.add(t.realizedPnL!.abs()), new Decimal(0));
          
          const profitFactor = totalLosses.eq(0)
            ? totalWins.gt(0) ? 999 : 0
            : totalWins.div(totalLosses).toNumber();
          
          set((draft) => {
            draft.riskMetrics.totalExposure = totalExposure;
            draft.riskMetrics.dailyPnl = dailyPnl;
            draft.riskMetrics.winRate = winRate;
            draft.dailyPnL = dailyPnl;
            draft.winRate = winRate;
            draft.profitFactor = profitFactor;
          });
        },
        
        setPreference: <K extends keyof TradingPreferences>(
          key: K,
          value: TradingPreferences[K]
        ) => {
          set((draft) => {
            draft.preferences[key] = value;
          });
        },
        
        canPlaceOrder: (params: PlaceOrderParams) => {
          const state = get();
          const balance = state.accountBalance;
          const orderValue = new Decimal(params.quantity).mul(params.price || 0);
          
          // Check if within risk limits
          if (orderValue.gt(state.riskLimits.riskLimitPerTrade)) {
            return {
              allowed: false,
              reason: `Order exceeds risk limit of $${state.riskLimits.riskLimitPerTrade}`,
            };
          }
          
          // Check daily loss limit
          if (state.dailyPnL.abs().gt(state.riskLimits.maxDailyLoss)) {
            return {
              allowed: false,
              reason: `Daily loss limit of $${state.riskLimits.maxDailyLoss} reached`,
            };
          }
          
          // Check buying power
          if (orderValue.gt(state.buyingPower)) {
            return {
              allowed: false,
              reason: 'Insufficient buying power',
            };
          }
          
          return { allowed: true };
        },
        
        isWithinRiskLimits: () => {
          const state = get();
          return state.dailyPnL.abs().lt(state.riskLimits.maxDailyLoss);
        },
        
        getExposureBySymbol: (symbol: string) => {
          const position = get().positions[symbol];
          if (!position) return new Decimal(0);
          return position.quantity.mul(position.currentPrice);
        },
        
        calculatePnLMetrics: () => {
          const positions = Object.values(get().positions);
          const trades = get().trades;
          
          const unrealized = positions.reduce(
            (sum, pos) => sum.add(pos.unrealizedPnL),
            new Decimal(0)
          );
          
          const realized = trades.reduce(
            (sum, trade) => sum.add(trade.realizedPnL || 0),
            new Decimal(0)
          );
          
          const completedTrades = trades.filter(t => t.realizedPnL !== undefined);
          const winningTrades = completedTrades.filter(t => t.realizedPnL!.gt(0));
          const winRate = completedTrades.length > 0
            ? (winningTrades.length / completedTrades.length) * 100
            : 0;
          
          const bestTrade = completedTrades.reduce(
            (best, trade) => {
              if (!trade.realizedPnL) return best;
              return trade.realizedPnL.gt(best.pnl)
                ? { symbol: trade.symbol, pnl: trade.realizedPnL.toNumber(), date: trade.date }
                : best;
            },
            { symbol: '', pnl: 0, date: new Date() }
          );
          
          const worstTrade = completedTrades.reduce(
            (worst, trade) => {
              if (!trade.realizedPnL) return worst;
              return trade.realizedPnL.lt(worst.pnl)
                ? { symbol: trade.symbol, pnl: trade.realizedPnL.toNumber(), date: trade.date }
                : worst;
            },
            { symbol: '', pnl: 0, date: new Date() }
          );
          
          return {
            realized: realized.toNumber(),
            unrealized: unrealized.toNumber(),
            winRate,
            bestTrade,
            worstTrade,
          };
        },
        
        setTradingLocked: (locked: boolean) => {
          console.log(locked ? '🔒 Trading LOCKED' : '🔓 Trading UNLOCKED');
          // Implementation would update a locked state
        },
        
        modifyPosition: async (symbol: string, updates: Partial<Pick<Position, 'stopLoss' | 'takeProfit'>>) => {
          const position = get().positions[symbol];
          if (!position) {
            throw new Error(`No position found for ${symbol}`);
          }
          
          try {
            await invoke('modify_position', {
              symbol,
              stopLoss: updates.stopLoss?.toNumber(),
              takeProfit: updates.takeProfit?.toNumber(),
            });
            
            set((draft) => {
              if (updates.stopLoss !== undefined) {
                draft.positions[symbol].stopLoss = updates.stopLoss;
              }
              if (updates.takeProfit !== undefined) {
                draft.positions[symbol].takeProfit = updates.takeProfit;
              }
              draft.positions[symbol].lastUpdate = new Date();
            });
            
            console.log(`✏️ Modified position for ${symbol}`, updates);
          } catch (error) {
            set((draft) => {
              draft.lastError = error instanceof Error ? error.message : 'Modification failed';
            });
            throw error;
          }
        },
      })),
      {
        name: 'nexlify-trading-store',
        partialize: (state) => ({
          preferences: state.preferences,
          watchlist: state.watchlist,
          riskLimits: state.riskLimits,
        }),
      }
    ),
    {
      name: 'NexlifyTradingStore',
    }
  )
);