// src/stores/tradingStore.ts
// NEXLIFY TRADING STATE - Where decisions echo through eternity
// Last sync: 2025-06-19 | "Every order is a prayer, every fill an answered wish"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist, devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import Decimal from 'decimal.js';

Decimal.set({ precision: 18, rounding: Decimal.ROUND_DOWN });

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop_loss' | 'take_profit';
  status: 'pending' | 'open' | 'partially_filled' | 'filled' | 'cancelled' | 'rejected';
  quantity: Decimal;
  price?: Decimal;
  stopPrice?: Decimal;
  filledQuantity: Decimal;
  averageFillPrice?: Decimal;
  fees: Decimal;
  createdAt: Date;
  updatedAt: Date;
  metadata: Record<string, any>;
}

interface Position {
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

interface TradeHistory {
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

interface RiskMetrics {
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

interface TradingPreferences {
  defaultOrderType: 'market' | 'limit';
  defaultTimeInForce: 'GTC' | 'GTT' | 'IOC' | 'FOK';
  confirmOrders: boolean;
  playSounds: boolean;
  autoStopLoss: boolean;
  stopLossPercent: number;
  takeProfitPercent: number;
}

interface PnLMetrics {
  realized: number;
  unrealized: number;
  winRate: number;
  bestTrade: { symbol: string; pnl: number; date: Date };
  worstTrade: { symbol: string; pnl: number; date: Date };
}

interface TradingState {
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

interface PlaceOrderParams {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop_loss' | 'take_profit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: string;
  reduceOnly?: boolean;
  postOnly?: boolean;
  metadata?: Record<string, any>;
}

interface PositionSizeParams {
  balance: number;
  riskPercent: number;
  stopLossDistance: number;
  price: number;
}

export const useTradingStore = create<TradingState>()(
  devtools(
    persist(
      immer((set, get) => ({
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
        
        placeOrder: async (params: PlaceOrderParams) => {
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
            let metadata = params.metadata || {};
            if (get().preferences.autoStopLoss && params.type === 'limit') {
              const stopLossPrice = params.side === 'buy'
                ? price!.mul(1 - get().preferences.stopLossPercent / 100)
                : price!.mul(1 + get().preferences.stopLossPercent / 100);
              metadata.autoStopLoss = stopLossPrice.toString();
              console.log(`🛡️ Auto stop-loss set at ${stopLossPrice.toString()}`);
            }
            const response = await invoke<any>('place_order', {
              symbol: params.symbol,
              side: params.side,
              orderType: params.type,
              quantity: quantity.toString(),
              price: price?.toString(),
              stopPrice: params.stopPrice?.toString(),
              timeInForce: params.timeInForce,
              metadata,
            });
            const order: Order = {
              id: response.order_id,
              symbol: params.symbol,
              side: params.side,
              type: params.type,
              status: 'pending',
              quantity,
              price,
              filledQuantity: new Decimal(0),
              fees: new Decimal(response.estimated_fees),
              createdAt: new Date(),
              updatedAt: new Date(),
              metadata,
            };
            set((draft) => {
              draft.activeOrders[order.id] = order;
              draft.isPlacingOrder = false;
            });
            if (get().preferences.playSounds) {
              playOrderSound('placed');
            }
            console.log(`📈 Order placed: ${order.side} ${order.quantity} ${order.symbol}`);
            get().updateRiskMetrics();
            return order;
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            set((draft) => {
              draft.isPlacingOrder = false;
              draft.lastError = errorMsg;
            });
            console.error('❌ Order failed:', errorMsg);
            if (get().preferences.playSounds) {
              playOrderSound('failed');
            }
            throw error;
          }
        },
        
        cancelOrder: async (orderId: string) => {
          try {
            await invoke('cancel_order', { orderId, reason: 'User requested' });
            set((draft) => {
              const order = draft.activeOrders[orderId];
              if (order) {
                order.status = 'cancelled';
                order.updatedAt = new Date();
                draft.orderHistory.unshift(order);
                delete draft.activeOrders[orderId];
              }
            });
            console.log(`🚫 Order ${orderId} cancelled`);
          } catch (error) {
            console.error('Failed to cancel order:', error);
            throw error;
          }
        },
        
        cancelAllOrders: async (symbol?: string) => {
          const orders = Object.values(get().activeOrders);
          const targetOrders = symbol ? orders.filter(o => o.symbol === symbol) : orders;
          console.log(`🚨 Cancelling ${targetOrders.length} orders...`);
          const results = await Promise.allSettled(
            targetOrders.map(order => get().cancelOrder(order.id))
          );
          const failed = results.filter(r => r.status === 'rejected').length;
          if (failed > 0) {
            console.warn(`⚠️ Failed to cancel ${failed} orders`);
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
        },
        
        calculatePositionSize: (params: PositionSizeParams) => {
          const riskAmount = new Decimal(params.balance).mul(params.riskPercent / 100);
          const shares = riskAmount.div(params.stopLossDistance);
          const positionValue = shares.mul(params.price);
          const maxPosition = new Decimal(params.balance).mul(0.5);
          if (positionValue.gt(maxPosition)) {
            return maxPosition.div(params.price);
          }
          return shares;
        },
        
        updateRiskMetrics: () => {
          const positions = Object.values(get().positions);
          const trades = get().trades;
          const totalExposure = positions.reduce(
            (sum, pos) => sum.add(pos.quantity.mul(pos.currentPrice)),
            new Decimal(0)
          );
          const today = new Date();
          today.setHours(0, 0, 0, 0);
          const todaysTrades = trades.filter(t => t.timestamp >= today);
          const dailyPnl = todaysTrades.reduce(
            (sum, trade) => sum.add(trade.realizedPnL || 0),
            new Decimal(0)
          );
          const completedTrades = trades.filter(t => t.realizedPnL !== undefined);
          const winningTrades = completedTrades.filter(t => t.realizedPnL!.gt(0));
          const winRate = completedTrades.length > 0
            ? (winningTrades.length / completedTrades.length) * 100
            : 0;
          const totalWins = winningTrades.reduce(
            (sum, t) => sum.add(t.realizedPnL || 0),
            new Decimal(0)
          );
          const totalLosses = completedTrades
            .filter(t => t.realizedPnL!.lt(0))
            .reduce((sum, t) => sum.add(t.realizedPnL!.abs()), new Decimal(0));
          const profitFactor = totalLosses.eq(0)
            ? totalWins.toNumber()
            : totalWins.div(totalLosses).toNumber();
          set((draft) => {
            draft.riskMetrics.totalExposure = totalExposure;
            draft.riskMetrics.dailyPnl = dailyPnl;
            draft.riskMetrics.winRate = winRate;
            draft.riskMetrics.marginUsage = totalExposure.div(draft.accountBalance).mul(100).toNumber();
            draft.dailyPnL = dailyPnl;
            draft.winRate = winRate;
            draft.profitFactor = profitFactor;
            draft.pnlHistory = draft.trades.map((t) => ({
              ...t,
              closeTime: t.timestamp,
              date: t.timestamp,
              realizedPnL: t.realizedPnL,
              openTime: t.timestamp,
            }));
          });
        },
        
        setPreference: <K extends keyof TradingPreferences>(key: K, value: TradingPreferences[K]) => {
          set((draft) => {
            draft.preferences[key] = value;
          });
        },
        
        canPlaceOrder: (params: PlaceOrderParams) => {
          const state = get();
          const orderValue = new Decimal(params.quantity).mul(params.price || 50000);
          if (state.riskMetrics.dailyPnl.lt(state.riskLimits.maxDailyLoss.neg())) {
            return { allowed: false, reason: 'Daily loss limit reached. Time to walk away.' };
          }
          if (orderValue.gt(state.riskLimits.riskLimitPerTrade)) {
            return { allowed: false, reason: `Order exceeds risk limit of $${state.riskLimits.riskLimitPerTrade}` };
          }
          if (orderValue.gt(state.buyingPower)) {
            return { allowed: false, reason: 'Insufficient buying power. Dreams need capital.' };
          }
          return { allowed: true };
        },
        
        isWithinRiskLimits: () => {
          const metrics = get().riskMetrics;
          return metrics.dailyPnl.gte(get().riskLimits.maxDailyLoss.neg()) && metrics.marginUsage < 80;
        },
        
        getExposureBySymbol: (symbol: string) => {
          const position = get().positions[symbol];
          if (!position) return new Decimal(0);
          return position.quantity.mul(position.currentPrice);
        },
        
        calculatePnLMetrics: () => {
          const trades = get().pnlHistory;
          const positions = Object.values(get().positions);
          const realized = trades.reduce(
            (sum, t) => sum + (t.realizedPnL?.toNumber() || 0),
            0
          );
          const unrealized = positions.reduce(
            (sum, p) => sum + p.unrealizedPnL.toNumber(),
            0
          );
          const winningTrades = trades.filter(t => t.realizedPnL && t.realizedPnL.gt(0));
          const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;
          const sortedTrades = [...trades].sort(
            (a, b) => (b.realizedPnL?.toNumber() || 0) - (a.realizedPnL?.toNumber() || 0)
          );
          const bestTrade = sortedTrades[0] || { symbol: '', pnl: 0, date: new Date() };
          const worstTrade = sortedTrades[sortedTrades.length - 1] || { symbol: '', pnl: 0, date: new Date() };
          return {
            realized,
            unrealized,
            winRate,
            bestTrade: {
              symbol: bestTrade.symbol,
              pnl: bestTrade.realizedPnL?.toNumber() || 0,
              date: bestTrade.date,
            },
            worstTrade: {
              symbol: worstTrade.symbol,
              pnl: worstTrade.realizedPnL?.toNumber() || 0,
              date: worstTrade.date,
            },
          };
        },
        
        setTradingLocked: (locked: boolean) => {
          set((draft) => {
            console.log(`Trading ${locked ? 'locked' : 'unlocked'}`);
          });
        },
        
        modifyPosition: async (symbol: string, updates: Partial<Pick<Position, 'stopLoss' | 'takeProfit'>>) => {
          const position = get().positions[symbol];
          if (!position) {
            throw new Error(`No position found for ${symbol}`);
          }
          try {
            await invoke('modify_position', {
              symbol,
              stopLoss: updates.stopLoss?.toString(),
              takeProfit: updates.takeProfit?.toString(),
            });
            set((draft) => {
              if (draft.positions[symbol]) {
                if (updates.stopLoss !== undefined) {
                  draft.positions[symbol].stopLoss = updates.stopLoss;
                }
                if (updates.takeProfit !== undefined) {
                  draft.positions[symbol].takeProfit = updates.takeProfit;
                }
                draft.positions[symbol].lastUpdate = new Date();
              }
            });
            console.log(`✏️ Modified position ${symbol}:`, updates);
          } catch (error) {
            console.error(`Failed to modify position ${symbol}:`, error);
            throw error;
          }
        },
      })),
      {
        name: 'nexlify-trading',
        partialize: (state) => ({
          preferences: state.preferences,
          orderHistory: state.orderHistory.slice(0, 100),
        }),
      }
    )
  )
);

function playOrderSound(type: 'placed' | 'filled' | 'failed') {
  if (typeof window === 'undefined') return;
  const sounds = {
    placed: '/sounds/order-placed.mp3',
    filled: '/sounds/order-filled.mp3',
    failed: '/sounds/order-failed.mp3',
  };
  const audio = new Audio(sounds[type]);
  audio.volume = 0.3;
  audio.play().catch(() => {});
}