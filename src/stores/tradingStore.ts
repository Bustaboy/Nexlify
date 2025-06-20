// src/stores/tradingStore.ts
// NEXLIFY TRADING STATE - Where decisions echo through eternity
// Last sync: 2025-06-19 | "Every order is a prayer, every fill an answered wish"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist, devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import Decimal from 'decimal.js';

/**
 * TRADING STORE - The chamber where bullets are loaded
 * 
 * This store has executed more orders than a mob boss. Every
 * function here was written in blood - usually mine. I've
 * fat-fingered market orders, forgotten stop losses, and
 * once accidentally went 100x leverage on a memecoin.
 * 
 * That last one? Actually made money. Sometimes the market
 * rewards stupidity. But don't count on it, hermano.
 */

// Configure Decimal.js - precision matters when dealing with money
Decimal.set({ precision: 18, rounding: Decimal.ROUND_DOWN });

// Types - the shape of our ammunition
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
  symbol: string;
  side: 'long' | 'short';
  quantity: Decimal;
  entryPrice: Decimal;
  currentPrice: Decimal;
  unrealizedPnl: Decimal;
  realizedPnl: Decimal;
  marginUsed: Decimal;
  liquidationPrice?: Decimal;
  openedAt: Date;
  lastUpdated: Date;
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
  pnl?: Decimal;
}

interface RiskMetrics {
  totalExposure: Decimal;
  marginUsage: number; // percentage
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
  riskLimitPerTrade: Decimal;
  maxDailyLoss: Decimal;
  autoStopLoss: boolean;
  stopLossPercent: number;
  takeProfitPercent: number;
}

interface TradingState {
  // Orders - our shots fired
  activeOrders: Record<string, Order>;
  orderHistory: Order[];
  
  // Positions - our soldiers in the field
  positions: Record<string, Position>;
  closedPositions: Position[];
  
  // Trade history - the record of battle
  trades: TradeHistory[];
  
  // Risk management - keeping us alive
  riskMetrics: RiskMetrics;
  accountBalance: Decimal;
  buyingPower: Decimal;
  
  // Preferences - how we like to fight
  preferences: TradingPreferences;
  
  // State flags
  isPlacingOrder: boolean;
  isLoadingPositions: boolean;
  lastError: string | null;
  
  // Actions - our arsenal
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
  
  // Risk checks - the safety catches
  canPlaceOrder: (params: PlaceOrderParams) => { allowed: boolean; reason?: string };
  isWithinRiskLimits: () => boolean;
  getExposureBySymbol: (symbol: string) => Decimal;
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
        // Initial state - locked and loaded
        activeOrders: {},
        orderHistory: [],
        positions: {},
        closedPositions: [],
        trades: [],
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
          currentDrawdown: 0
        },
        accountBalance: new Decimal(10000), // Starting balance - dreams begin here
        buyingPower: new Decimal(10000),
        preferences: {
          defaultOrderType: 'limit',
          defaultTimeInForce: 'GTC',
          confirmOrders: true,
          playSounds: true,
          riskLimitPerTrade: new Decimal(100), // $100 max risk per trade
          maxDailyLoss: new Decimal(500), // $500 daily stop
          autoStopLoss: true,
          stopLossPercent: 2, // 2% stop loss
          takeProfitPercent: 6 // 3:1 risk/reward
        },
        isPlacingOrder: false,
        isLoadingPositions: false,
        lastError: null,
        
        /**
         * Place an order - pulling the trigger
         * 
         * Every order I place, I think about my first big loss.
         * Market bought 10 BTC in 2017. No stop loss. "Diamond
         * hands," I said. Rode it from 19k to 3k. These days,
         * every order has a plan, every plan has an exit.
         */
        placeOrder: async (params: PlaceOrderParams) => {
          // Pre-flight checks
          const { allowed, reason } = get().canPlaceOrder(params);
          if (!allowed) {
            throw new Error(reason || 'Order rejected by risk management');
          }
          
          set((draft) => {
            draft.isPlacingOrder = true;
            draft.lastError = null;
          });
          
          try {
            // Convert to proper decimals for precision
            const quantity = new Decimal(params.quantity);
            const price = params.price ? new Decimal(params.price) : undefined;
            
            // Auto stop-loss if enabled - learned this the hard way
            let metadata = params.metadata || {};
            if (get().preferences.autoStopLoss && params.type === 'limit') {
              const stopLossPrice = params.side === 'buy' 
                ? price!.mul(1 - get().preferences.stopLossPercent / 100)
                : price!.mul(1 + get().preferences.stopLossPercent / 100);
              
              metadata.autoStopLoss = stopLossPrice.toString();
              console.log(`🛡️ Auto stop-loss set at ${stopLossPrice.toString()}`);
            }
            
            // Place via Tauri
            const response = await invoke<any>('place_order', {
              symbol: params.symbol,
              side: params.side,
              orderType: params.type,
              quantity: quantity.toString(),
              price: price?.toString(),
              stopPrice: params.stopPrice?.toString(),
              timeInForce: params.timeInForce,
              metadata
            });
            
            // Create order object
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
              metadata
            };
            
            // Update state
            set((draft) => {
              draft.activeOrders[order.id] = order;
              draft.isPlacingOrder = false;
            });
            
            // Play sound if enabled - that sweet confirmation chime
            if (get().preferences.playSounds) {
              playOrderSound('placed');
            }
            
            console.log(`📈 Order placed: ${order.side} ${order.quantity} ${order.symbol}`);
            
            // Update risk metrics
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
        
        /**
         * Cancel order - knowing when to fold
         */
        cancelOrder: async (orderId: string) => {
          try {
            await invoke('cancel_order', {
              orderId,
              reason: 'User requested'
            });
            
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
        
        /**
         * Cancel all orders - the panic button
         * 
         * Added this after the flash crash of '22. Sometimes
         * you just need to clear the board and reassess.
         * No shame in tactical retreat, hermano.
         */
        cancelAllOrders: async (symbol?: string) => {
          const orders = Object.values(get().activeOrders);
          const targetOrders = symbol 
            ? orders.filter(o => o.symbol === symbol)
            : orders;
          
          console.log(`🚨 Cancelling ${targetOrders.length} orders...`);
          
          const results = await Promise.allSettled(
            targetOrders.map(order => get().cancelOrder(order.id))
          );
          
          const failed = results.filter(r => r.status === 'rejected').length;
          if (failed > 0) {
            console.warn(`⚠️ Failed to cancel ${failed} orders`);
          }
        },
        
        /**
         * Close position - taking chips off the table
         */
        closePosition: async (symbol: string, percentage = 100) => {
          const position = get().positions[symbol];
          if (!position) {
            throw new Error(`No position found for ${symbol}`);
          }
          
          const closeQuantity = position.quantity.mul(percentage / 100);
          
          // Place market order to close
          await get().placeOrder({
            symbol,
            side: position.side === 'long' ? 'sell' : 'buy',
            type: 'market',
            quantity: closeQuantity.toNumber(),
            metadata: { 
              action: 'close_position',
              percentage 
            }
          });
          
          console.log(`💰 Closing ${percentage}% of ${symbol} position`);
        },
        
        /**
         * Close all positions - DEFCON 1
         */
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
        
        /**
         * Refresh positions - know where you stand
         */
        refreshPositions: async () => {
          set((draft) => {
            draft.isLoadingPositions = true;
          });
          
          try {
            const response = await invoke<any>('get_positions', {});
            
            set((draft) => {
              draft.positions = {};
              
              response.positions.forEach((pos: any) => {
                draft.positions[pos.position.symbol] = {
                  symbol: pos.position.symbol,
                  side: pos.position.side.toLowerCase() as 'long' | 'short',
                  quantity: new Decimal(pos.position.quantity),
                  entryPrice: new Decimal(pos.position.entry_price),
                  currentPrice: new Decimal(pos.position.current_price),
                  unrealizedPnl: new Decimal(pos.position.unrealized_pnl),
                  realizedPnl: new Decimal(pos.position.realized_pnl),
                  marginUsed: new Decimal(pos.position.margin_used),
                  liquidationPrice: pos.risk_metrics.liquidation_price 
                    ? new Decimal(pos.risk_metrics.liquidation_price)
                    : undefined,
                  openedAt: new Date(pos.position.opened_at),
                  lastUpdated: new Date(pos.position.last_updated)
                };
              });
              
              draft.isLoadingPositions = false;
            });
            
            // Update risk metrics with fresh data
            get().updateRiskMetrics();
            
          } catch (error) {
            set((draft) => {
              draft.isLoadingPositions = false;
              draft.lastError = error instanceof Error ? error.message : 'Failed to load positions';
            });
            throw error;
          }
        },
        
        /**
         * Refresh orders - tracking our bullets in flight
         */
        refreshOrders: async () => {
          // Similar to positions, fetch from backend
          console.log('📋 Refreshing orders...');
        },
        
        /**
         * Calculate position size - the Kelly Criterion's cousin
         * 
         * This formula? Written in the ashes of blown accounts.
         * Risk 1-2% per trade. Always. The market will humble
         * anyone who thinks they're special.
         */
        calculatePositionSize: (params: PositionSizeParams) => {
          const riskAmount = new Decimal(params.balance).mul(params.riskPercent / 100);
          const shares = riskAmount.div(params.stopLossDistance);
          const positionValue = shares.mul(params.price);
          
          // Never risk more than 50% of account on single position
          const maxPosition = new Decimal(params.balance).mul(0.5);
          
          if (positionValue.gt(maxPosition)) {
            return maxPosition.div(params.price);
          }
          
          return shares;
        },
        
        /**
         * Update risk metrics - the vital signs
         */
        updateRiskMetrics: () => {
          const positions = Object.values(get().positions);
          const trades = get().trades;
          
          // Calculate total exposure
          const totalExposure = positions.reduce(
            (sum, pos) => sum.add(pos.quantity.mul(pos.currentPrice)),
            new Decimal(0)
          );
          
          // Calculate daily PnL
          const today = new Date();
          today.setHours(0, 0, 0, 0);
          
          const todaysTrades = trades.filter(t => t.timestamp >= today);
          const dailyPnl = todaysTrades.reduce(
            (sum, trade) => sum.add(trade.pnl || 0),
            new Decimal(0)
          );
          
          // Win rate calculation
          const completedTrades = trades.filter(t => t.pnl !== undefined);
          const winningTrades = completedTrades.filter(t => t.pnl!.gt(0));
          const winRate = completedTrades.length > 0 
            ? (winningTrades.length / completedTrades.length) * 100 
            : 0;
          
          set((draft) => {
            draft.riskMetrics.totalExposure = totalExposure;
            draft.riskMetrics.dailyPnl = dailyPnl;
            draft.riskMetrics.winRate = winRate;
            draft.riskMetrics.marginUsage = totalExposure.div(draft.accountBalance).mul(100).toNumber();
          });
        },
        
        /**
         * Set preference - personalizing the weapon
         */
        setPreference: <K extends keyof TradingPreferences>(
          key: K, 
          value: TradingPreferences[K]
        ) => {
          set((draft) => {
            draft.preferences[key] = value;
          });
        },
        
        /**
         * Can place order - the pre-flight check
         */
        canPlaceOrder: (params: PlaceOrderParams) => {
          const state = get();
          const orderValue = new Decimal(params.quantity).mul(params.price || 50000);
          
          // Check daily loss limit
          if (state.riskMetrics.dailyPnl.lt(state.preferences.maxDailyLoss.neg())) {
            return { 
              allowed: false, 
              reason: 'Daily loss limit reached. Time to walk away.' 
            };
          }
          
          // Check position limits
          if (orderValue.gt(state.preferences.riskLimitPerTrade)) {
            return { 
              allowed: false, 
              reason: `Order exceeds risk limit of $${state.preferences.riskLimitPerTrade}` 
            };
          }
          
          // Check buying power
          if (orderValue.gt(state.buyingPower)) {
            return { 
              allowed: false, 
              reason: 'Insufficient buying power. Dreams need capital.' 
            };
          }
          
          return { allowed: true };
        },
        
        /**
         * Check if within risk limits
         */
        isWithinRiskLimits: () => {
          const metrics = get().riskMetrics;
          return metrics.dailyPnl.gte(get().preferences.maxDailyLoss.neg()) &&
                 metrics.marginUsage < 80; // 80% margin usage max
        },
        
        /**
         * Get exposure by symbol
         */
        getExposureBySymbol: (symbol: string) => {
          const position = get().positions[symbol];
          if (!position) return new Decimal(0);
          
          return position.quantity.mul(position.currentPrice);
        }
      })),
      {
        name: 'nexlify-trading',
        // Only persist preferences, not active data
        partialize: (state) => ({
          preferences: state.preferences,
          orderHistory: state.orderHistory.slice(0, 100) // Last 100 orders
        })
      }
    )
  )
);

/**
 * Order sounds - audio feedback for the neural cortex
 */
function playOrderSound(type: 'placed' | 'filled' | 'failed') {
  if (typeof window === 'undefined') return;
  
  const sounds = {
    placed: '/sounds/order-placed.mp3',
    filled: '/sounds/order-filled.mp3',
    failed: '/sounds/order-failed.mp3'
  };
  
  const audio = new Audio(sounds[type]);
  audio.volume = 0.3;
  audio.play().catch(() => {}); // Silent fail
}

/**
 * TRADING WISDOM (paid for in losses):
 * 
 * 1. That position size calculator? Use it. I once YOLOed
 *    50% of my account on a "sure thing." It wasn't.
 * 
 * 2. Daily loss limits save lives. Mine stopped me from
 *    revenge trading after a bad morning. Would've lost
 *    the house trying to "make it back."
 * 
 * 3. The close all positions button? That's your ejection
 *    seat. When the market goes crazy, sometimes the best
 *    position is cash.
 * 
 * 4. Order sounds might seem silly, but audio confirmation
 *    saved me from a double-order that would've blown my
 *    risk limits.
 * 
 * Remember: The market is a patient predator. Respect it,
 * or it will eat you alive.
 */
