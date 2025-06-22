// src/stores/marketStore.ts
// NEXLIFY MARKET STATE - The pulse of the digital bazaar
// Last sync: 2025-06-19 | "The market never sleeps, but it does have nightmares"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { subscribeWithSelector, devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { decode } from 'msgpackr';

// Types - structuring the chaos
interface OrderbookLevel {
  price: number;
  quantity: number;
  orderCount?: number;
}

interface MarketUpdate {
  symbol: string;
  price: number;
  volume: number;
}

interface Orderbook {
  symbol: string;
  bids: OrderbookLevel[];
  asks: OrderbookLevel[];
  lastUpdate: Date;
  spread: number;
  midPrice: number;
}

interface Ticker {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  volume24h: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  high24h: number;
  low24h: number;
}

interface Trade {
  id: string;
  symbol: string;
  price: number;
  quantity: number;
  side: 'buy' | 'sell';
  timestamp: Date;
}

interface Candle {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface MarketSubscription {
  symbol: string;
  types: string[];
  unlistener?: UnlistenFn;
  reconnectAttempts: number;
  lastHeartbeat: Date;
}

interface MarketState {
  orderbooks: Record<string, Orderbook>;
  tickers: Record<string, Ticker>;
  recentTrades: Record<string, Trade[]>;
  candles: Record<string, Candle[]>;
  updateMarketData: (updates: MarketUpdate[]) => void;
  marketData: Record<string, {
    price: number;           // Current price
    volume: number;          // Current volume
    previousPrice: number;   // Previous price for change calculations
    lastUpdate: Date;        // Timestamp of the last update
    changePercent24h: number; // Percentage change over 24 hours
    volume24h: number;       // Volume over the last 24 hours
    high24h: number;         // Highest price in the last 24 hours
    low24h: number;          // Lowest price in the last 24 hours
    avgVolume: number;       // Average volume over a period
  }>;
  volatilityIndex: number;
  totalVolume24h: number;
  topMovers: { symbol: string; changePercent: number }[],

  // Connection management - our neural links
  subscriptions: Record<string, MarketSubscription>;
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  marketHealth: 'operational' | 'degraded' | 'down';
  lastError: string | null;
  
  // Performance metrics - keeping score
  messagesPerSecond: number;
  totalMessagesProcessed: number;
  cacheHitRate: number;
  
  // Binary data handling - for when JSON is too slow
  binaryMode: boolean;
  compressionEnabled: boolean;
  
  // Actions - our moves in the market dance
  subscribeToMarket: (symbol: string, types?: string[]) => Promise<void>;
  unsubscribeFromMarket: (symbol: string) => Promise<void>;
  updateOrderbook: (symbol: string, data: any) => void;
  updateTicker: (symbol: string, data: any) => void;
  processTrade: (symbol: string, trade: Trade) => void;
  handleBinaryMessage: (data: Uint8Array) => void;
  handleMarketEvent: (event: any) => void;
  reconnectAll: () => Promise<void>;
  clearMarketData: (symbol?: string) => void;
  getSymbolData: (symbol: string) => { price: number; volume: number } | null;
  
  // Health monitoring - staying alive
  checkHealth: () => Promise<void>;
  getSpread: (symbol: string) => number | null;
  getMidPrice: (symbol: string) => number | null;
  getDepth: (symbol: string, side: 'bid' | 'ask', levels: number) => OrderbookLevel[];
}

// Constants - the rules of engagement
const MAX_TRADES_PER_SYMBOL = 100;
const MAX_ORDERBOOK_DEPTH = 50;
const HEARTBEAT_INTERVAL = 5000; // 5 seconds
const RECONNECT_MAX_ATTEMPTS = 5;
const RECONNECT_DELAY = 1000; // Start with 1 second, exponential backoff

export const useMarketStore = create<MarketState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // Initial state - empty battlefield
        orderbooks: {},
        tickers: {},
        recentTrades: {},
        candles: {},
        marketData: {},
        volatilityIndex: 0,
        totalVolume24h: 0,
        topMovers: [],
        subscriptions: {},
        connectionStatus: 'disconnected',
        marketHealth: 'operational',
        lastError: null,
        messagesPerSecond: 0,
        totalMessagesProcessed: 0,
        cacheHitRate: 0,
        binaryMode: true,
        compressionEnabled: true,
        
        subscribeToMarket: async (symbol: string, types: string[] = ['orderbook', 'ticker', 'trade', 'candles']) => {
          try {
            const existing = get().subscriptions[symbol];
            if (existing) {
              console.log(`📡 Already subscribed to ${symbol}`);
              return;
            }
            
            set((draft) => {
              draft.connectionStatus = 'connecting';
            });
            
            const response = await invoke<{
              subscription_id: string;
              symbol: string;
              data_types: string[];
              status: string;
              message: string;
            }>('subscribe_market_data', {
              symbol,
              dataTypes: types,
            });
            
            const unlistener = await listen<any>('market-update', (event) => {
              const state = get();
              if (event.payload.symbol === symbol) {
                state.handleMarketEvent(event.payload);
                set((draft) => {
                  draft.totalMessagesProcessed++;
                });
              }
            });
            
            set((draft) => {
              draft.subscriptions[symbol] = {
                symbol,
                types,
                unlistener,
                reconnectAttempts: 0,
                lastHeartbeat: new Date(),
              };
              draft.connectionStatus = 'connected';
            });
            
            console.log(`✅ Subscribed to ${symbol} - ${types.join(', ')}`);
            startHeartbeatMonitor(symbol);
          } catch (error) {
            console.error(`❌ Failed to subscribe to ${symbol}:`, error);
            set((draft) => {
              draft.lastError = error instanceof Error ? error.message : 'Unknown error';
              draft.connectionStatus = 'error';
            });
            throw error;
          }
        },
        
        unsubscribeFromMarket: async (symbol: string) => {
          const subscription = get().subscriptions[symbol];
          if (!subscription) return;
          
          try {
            await invoke('unsubscribe_market_data', {
              subscriptionId: symbol,
            });
            if (subscription.unlistener) {
              subscription.unlistener();
            }
            set((draft) => {
              delete draft.subscriptions[symbol];
            });
            console.log(`🔌 Unsubscribed from ${symbol}`);
          } catch (error) {
            console.error(`Failed to unsubscribe from ${symbol}:`, error);
          }
        },
        
        updateOrderbook: (symbol: string, data: any) => {
          set((draft) => {
            const bids = data.bids.slice(0, MAX_ORDERBOOK_DEPTH);
            const asks = data.asks.slice(0, MAX_ORDERBOOK_DEPTH);
            const spread = asks[0] && bids[0] ? asks[0].price - bids[0].price : 0;
            const midPrice = asks[0] && bids[0] ? (asks[0].price + bids[0].price) / 2 : 0;
            
            draft.orderbooks[symbol] = {
              symbol,
              bids,
              asks,
              lastUpdate: new Date(),
              spread,
              midPrice,
            };
          });
        },
        
        updateTicker: (symbol: string, data: any) => {
          set((draft) => {
            // Update ticker data
            draft.tickers[symbol] = {
              symbol,
              bid: data.bid,
              ask: data.ask,
              last: data.last,
              volume24h: data.volume24h,
              priceChange24h: data.priceChange24h,
              priceChangePercent24h: data.priceChangePercent24h,
              high24h: data.high24h,
              low24h: data.low24h,
            };
            
            // Define price and volume from data
            const price = data.last;
            const volume = data.volume24h;
            
            // Initialize marketData if it doesn't exist
            if (!draft.marketData[symbol]) {
              draft.marketData[symbol] = {
                price: 0,
                volume: 0,
                previousPrice: 0,
                lastUpdate: new Date(),
                changePercent24h: 0,
                volume24h: 0,
                high24h: 0,
                low24h: 0,
                avgVolume: 0
              };
            }
            
            // Update marketData
            draft.marketData[symbol] = {
              price,
              volume,
              previousPrice: draft.marketData[symbol]?.previousPrice || price,
              lastUpdate: new Date(),
              changePercent24h: data.priceChangePercent24h || draft.marketData[symbol]?.changePercent24h || 0,
              volume24h: volume,
              high24h: data.high24h || draft.marketData[symbol]?.high24h || price,
              low24h: data.low24h || draft.marketData[symbol]?.low24h || price,
              avgVolume: draft.marketData[symbol]?.avgVolume || volume
            };
            
            // Update totalVolume24h
            draft.totalVolume24h = Object.values(draft.tickers).reduce(
              (sum, ticker) => sum + ticker.volume24h,
              0
            );
            
            // Update topMovers
            draft.topMovers = Object.values(draft.tickers)
              .map(t => ({
                symbol: t.symbol,
                changePercent: t.priceChangePercent24h,
              }))
              .sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent))
              .slice(0, 5);
          });
        },
        
        processTrade: (symbol: string, trade: Trade) => {
          set((draft) => {
            if (!draft.recentTrades[symbol]) {
              draft.recentTrades[symbol] = [];
            }
            draft.recentTrades[symbol].unshift(trade);
            if (draft.recentTrades[symbol].length > MAX_TRADES_PER_SYMBOL) {
              draft.recentTrades[symbol].pop();
            }
            // Update volatilityIndex (example calculation)
            const prices = draft.recentTrades[symbol].map(t => t.price);
            const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
            const variance = prices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / prices.length;
            draft.volatilityIndex = Math.sqrt(variance) / avgPrice * 100;
          });
        },
        
        handleBinaryMessage: (data: Uint8Array) => {
          try {
            const decoded = decode(data);
            get().handleMarketEvent(decoded);
          } catch (error) {
            console.error('Failed to decode binary message:', error);
          }
        },
        
        handleMarketEvent: (event: any) => {
          const { updateOrderbook, updateTicker, processTrade } = get();
          switch (event.type) {
            case 'orderbook':
              updateOrderbook(event.symbol, event.data);
              break;
            case 'ticker':
              updateTicker(event.symbol, event.data);
              break;
            case 'trade':
              processTrade(event.symbol, event.data);
              break;
            case 'candles':
              set((draft) => {
                const key = `${event.symbol}:${event.timeframe || '1m'}`;
                draft.candles[key] = event.data.map((c: any) => ({
                  timestamp: new Date(c.timestamp),
                  open: c.open,
                  high: c.high,
                  low: c.low,
                  close: c.close,
                  volume: c.volume,
                }));
              });
              break;
            default:
              console.warn('Unknown market event type:', event.type);
          }
        },
        
        reconnectAll: async () => {
          const subscriptions = Object.values(get().subscriptions);
          console.log('🔄 Reconnecting all market feeds...');
          for (const sub of subscriptions) {
            if (sub.unlistener) {
              sub.unlistener();
            }
            const delay = RECONNECT_DELAY * Math.pow(2, sub.reconnectAttempts);
            await new Promise(resolve => setTimeout(resolve, delay));
            try {
              await get().subscribeToMarket(sub.symbol, sub.types);
              set((draft) => {
                draft.subscriptions[sub.symbol].reconnectAttempts = 0;
              });
            } catch (error) {
              set((draft) => {
                draft.subscriptions[sub.symbol].reconnectAttempts++;
              });
              if (sub.reconnectAttempts >= RECONNECT_MAX_ATTEMPTS) {
                console.error(`❌ Failed to reconnect ${sub.symbol} after ${RECONNECT_MAX_ATTEMPTS} attempts`);
              }
            }
          }
        },
        
        clearMarketData: (symbol?: string) => {
          set((draft) => {
            if (symbol) {
              delete draft.orderbooks[symbol];
              delete draft.tickers[symbol];
              delete draft.recentTrades[symbol];
              delete draft.candles[symbol];
              delete draft.marketData[symbol];
            } else {
              draft.orderbooks = {};
              draft.tickers = {};
              draft.recentTrades = {};
              draft.candles = {};
              draft.marketData = {};
            }
          });
        },
        
        updateMarketData: (updates: MarketUpdate[]) => {
          set((draft) => {
            updates.forEach(({ symbol, price, volume }) => {
              // Initialize if doesn't exist
              if (!draft.marketData[symbol]) {
                draft.marketData[symbol] = {
                  price: 0,
                  volume: 0,
                  previousPrice: 0,
                  lastUpdate: new Date(),
                  changePercent24h: 0,
                  volume24h: 0,
                  high24h: 0,
                  low24h: 0,
                  avgVolume: 0
                };
              }
              
              const existing = draft.marketData[symbol];
              
              // Update with all required properties
              draft.marketData[symbol] = {
                price,
                volume,
                previousPrice: existing.previousPrice || price,
                lastUpdate: new Date(),
                changePercent24h: existing.changePercent24h || 0,
                volume24h: existing.volume24h || volume,
                high24h: Math.max(existing.high24h || price, price),
                low24h: Math.min(existing.low24h || price, price),
                avgVolume: existing.avgVolume || volume
              };
            });
          });
        },
        
        checkHealth: async () => {
          const subs = Object.values(get().subscriptions);
          const now = new Date();
          let healthyCount = 0;
          for (const sub of subs) {
            const timeSinceHeartbeat = now.getTime() - sub.lastHeartbeat.getTime();
            if (timeSinceHeartbeat < HEARTBEAT_INTERVAL * 2) {
              healthyCount++;
            }
          }
          const healthPercentage = subs.length > 0 ? healthyCount / subs.length : 1;
          set((draft) => {
            if (healthPercentage === 1) {
              draft.marketHealth = 'operational';
            } else if (healthPercentage > 0.5) {
              draft.marketHealth = 'degraded';
            } else {
              draft.marketHealth = 'down';
            }
          });
        },
        
        getSpread: (symbol: string) => {
          const book = get().orderbooks[symbol];
          return book?.spread || null;
        },
        
        getMidPrice: (symbol: string) => {
          const book = get().orderbooks[symbol];
          return book?.midPrice || null;
        },
        
        getDepth: (symbol: string, side: 'bid' | 'ask', levels: number) => {
          const book = get().orderbooks[symbol];
          if (!book) return [];
          const orders = side === 'bid' ? book.bids : book.asks;
          return orders.slice(0, levels);
        },
        
        getSymbolData: (symbol: string) => {
          const data = get().marketData[symbol];
          return data || null;
        },
      }))
    ),
    {
      name: 'nexlify-market',
      partialize: (state) => ({
        connectionStatus: state.connectionStatus,
        marketHealth: state.marketHealth,
        messagesPerSecond: state.messagesPerSecond,
      }),
    }
  )
);

// Helper functions
const heartbeatIntervals: Record<string, number> = {};

function startHeartbeatMonitor(symbol: string) {
  if (heartbeatIntervals[symbol]) {
    clearInterval(heartbeatIntervals[symbol]);
  }
  heartbeatIntervals[symbol] = window.setInterval(() => {
    const store = useMarketStore.getState();
    const sub = store.subscriptions[symbol];
    if (sub) {
      const now = new Date();
      const timeSinceLastBeat = now.getTime() - sub.lastHeartbeat.getTime();
      if (timeSinceLastBeat > HEARTBEAT_INTERVAL * 3) {
        console.warn(`⚠️ No heartbeat from ${symbol} for ${timeSinceLastBeat}ms`);
      }
    }
  }, HEARTBEAT_INTERVAL);
}

// Performance monitoring
let messageCount = 0;
let lastRateCheck = Date.now();

setInterval(() => {
  const now = Date.now();
  const elapsed = (now - lastRateCheck) / 1000;
  const rate = messageCount / elapsed;
  useMarketStore.setState({ messagesPerSecond: rate });
  messageCount = 0;
  lastRateCheck = now;
}, 1000);

// Subscribe to message count changes
useMarketStore.subscribe(
  (state) => state.totalMessagesProcessed,
  () => messageCount++
);

/**
 * MARKET STORE WISDOM:
 * 
 * 1. Market data is a firehose. Every update costs CPU cycles.
 *    Use selectors wisely, subscribe only to what you need.
 * 
 * 2. Binary mode exists for a reason. JSON parsing during high
 *    volatility will melt your CPU. I've seen it happen.
 * 
 * 3. Heartbeat monitoring catches dead connections before they
 *    cost you money. The market doesn't pause for reconnects.
 * 
 * 4. Orderbook depth is capped at 50 levels. More than that is
 *    vanity, not trading. Your screen isn't that tall anyway.
 * 
 * 5. The volatility index is a simple standard deviation. Want
 *    something fancier? Calculate it in a worker thread.
 * 
 * "In the market matrix, data flows like blood through chrome veins.
 *  Miss a beat, and you're already behind."
 */