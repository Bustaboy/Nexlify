// src-tauri/src/state/market_cache.rs
// NEXLIFY MARKET CACHE - Real-time data flowing through the neural mesh
// Last sync: 2025-06-19 | "Data flows like blood through chrome veins"

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use bytes::Bytes;
use tokio::sync::broadcast;
use tracing::{debug, warn};

/// Market cache configuration - tune these for your chrome
const MAX_ORDERBOOK_DEPTH: usize = 100;        // How deep we peer into the market abyss
const MAX_TRADE_HISTORY: usize = 1000;         // Recent trades to keep in memory
const MAX_CANDLE_HISTORY: usize = 5000;        // OHLCV data points
const CACHE_CLEANUP_INTERVAL: u64 = 300;       // Seconds between neural garbage collection
const SNAPSHOT_INTERVAL: u64 = 60;             // Seconds between cache snapshots

/// The neural cache - where market data lives and breathes
#[derive(Debug)]
pub struct MarketCache {
    /// Orderbook data - the beating heart of the market
    orderbooks: DashMap<String, Arc<RwLock<OrderBook>>>,
    
    /// Recent trades - watching the blood flow
    recent_trades: DashMap<String, Arc<RwLock<VecDeque<Trade>>>>,
    
    /// Price tickers - the pulse of the sprawl
    tickers: DashMap<String, Arc<RwLock<Ticker>>>,
    
    /// Candle data for charting - seeing patterns in the chaos
    candles: DashMap<CandleKey, Arc<RwLock<VecDeque<Candle>>>>,
    
    /// WebSocket broadcast channels - spreading the signal
    broadcasters: DashMap<String, broadcast::Sender<MarketEvent>>,
    
    /// Cache statistics - keeping tabs on our neural efficiency
    stats: Arc<RwLock<CacheStats>>,
}

/// Orderbook structure - where buyers and sellers dance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub exchange: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub last_update: DateTime<Utc>,
    pub sequence_id: Option<u64>, // For detecting gaps in the feed
    pub checksum: Option<u32>,    // Integrity check
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: Option<u32>, // Some exchanges provide this
}

/// Individual trade data - each transaction in the digital bazaar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub timestamp: DateTime<Utc>,
    pub is_maker: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Price ticker - the rapid heartbeat of the market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub exchange: String,
    pub bid: f64,
    pub bid_size: f64,
    pub ask: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub last_size: f64,
    pub volume_24h: f64,
    pub vwap_24h: f64,
    pub price_change_24h: f64,
    pub price_change_percent_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub timestamp: DateTime<Utc>,
}

/// OHLCV candle data - patterns in the chaos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trades: u32,
}

/// Candle key for multi-timeframe support
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct CandleKey {
    pub symbol: String,
    pub exchange: String,
    pub timeframe: String, // "1m", "5m", "1h", etc.
}

/// Market events for broadcasting - spreading the signal through the mesh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    OrderbookUpdate {
        symbol: String,
        bids: Vec<PriceLevel>,
        asks: Vec<PriceLevel>,
    },
    TradeUpdate {
        symbol: String,
        trade: Trade,
    },
    TickerUpdate {
        symbol: String,
        ticker: Ticker,
    },
    CandleUpdate {
        symbol: String,
        timeframe: String,
        candle: Candle,
    },
}

/// Cache statistics - monitoring our neural efficiency
#[derive(Debug, Default)]
pub struct CacheStats {
    pub total_updates: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub bytes_processed: u64,
    pub last_cleanup: Option<DateTime<Utc>>,
    pub orderbook_count: usize,
    pub trade_count: usize,
    pub candle_count: usize,
}

impl MarketCache {
    /// Boot up the neural cache
    pub fn new() -> Arc<Self> {
        let cache = Arc::new(Self {
            orderbooks: DashMap::new(),
            recent_trades: DashMap::new(),
            tickers: DashMap::new(),
            candles: DashMap::new(),
            broadcasters: DashMap::new(),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        });
        
        // Spawn cleanup daemon - keeping our neural pathways clean
        let cache_clone = cache.clone();
        tokio::spawn(async move {
            cache_clone.cleanup_daemon().await;
        });
        
        cache
    }
    
    /// Update orderbook from WebSocket feed
    pub fn update_orderbook(&self, symbol: String, orderbook: OrderBook) {
        let mut stats = self.stats.write();
        stats.total_updates += 1;
        
        // Check sequence for gaps - data integrity is life
        if let Some(book) = self.orderbooks.get(&symbol) {
            let current = book.read();
            if let (Some(new_seq), Some(old_seq)) = (orderbook.sequence_id, current.sequence_id) {
                if new_seq != old_seq + 1 {
                    warn!(
                        "⚠️ Sequence gap detected for {}: {} -> {} - neural desync!",
                        symbol, old_seq, new_seq
                    );
                }
            }
        }
        
        // Update the cache
        self.orderbooks.insert(
            symbol.clone(),
            Arc::new(RwLock::new(orderbook.clone()))
        );
        
        // Broadcast update to subscribers
        if let Some(broadcaster) = self.broadcasters.get(&symbol) {
            let _ = broadcaster.send(MarketEvent::OrderbookUpdate {
                symbol: symbol.clone(),
                bids: orderbook.bids.clone(),
                asks: orderbook.asks.clone(),
            });
        }
        
        debug!("📊 Orderbook updated for {} - {} bids, {} asks", 
               symbol, orderbook.bids.len(), orderbook.asks.len());
    }
    
    /// Process trade from feed
    pub fn process_trade(&self, symbol: String, trade: Trade) {
        let trades = self.recent_trades
            .entry(symbol.clone())
            .or_insert_with(|| Arc::new(RwLock::new(VecDeque::with_capacity(MAX_TRADE_HISTORY))));
        
        let mut trades_guard = trades.write();
        
        // Maintain circular buffer - no memory bloat in our chrome
        if trades_guard.len() >= MAX_TRADE_HISTORY {
            trades_guard.pop_front();
        }
        
        trades_guard.push_back(trade.clone());
        
        // Update stats
        self.stats.write().total_updates += 1;
        
        // Broadcast the trade
        if let Some(broadcaster) = self.broadcasters.get(&symbol) {
            let _ = broadcaster.send(MarketEvent::TradeUpdate {
                symbol: symbol.clone(),
                trade,
            });
        }
    }
    
    /// Update ticker data
    pub fn update_ticker(&self, symbol: String, ticker: Ticker) {
        self.tickers.insert(symbol.clone(), Arc::new(RwLock::new(ticker.clone())));
        
        // Broadcast ticker update
        if let Some(broadcaster) = self.broadcasters.get(&symbol) {
            let _ = broadcaster.send(MarketEvent::TickerUpdate {
                symbol: symbol.clone(),
                ticker,
            });
        }
        
        self.stats.write().total_updates += 1;
    }
    
    /// Update from generic WebSocket message - the universal translator
    pub async fn update_from_websocket(&self, data: serde_json::Value) {
        // Parse based on message type - cada exchange has its own dialect
        if let Some(msg_type) = data.get("type").and_then(|v| v.as_str()) {
            match msg_type {
                "l2update" | "orderbook" => {
                    if let Ok(orderbook) = self.parse_orderbook_update(&data) {
                        self.update_orderbook(orderbook.symbol.clone(), orderbook);
                    }
                }
                "trade" | "match" => {
                    if let Ok(trade) = self.parse_trade(&data) {
                        self.process_trade(trade.symbol.clone(), trade);
                    }
                }
                "ticker" => {
                    if let Ok(ticker) = self.parse_ticker(&data) {
                        self.update_ticker(ticker.symbol.clone(), ticker);
                    }
                }
                _ => {
                    debug!("Unknown WebSocket message type: {}", msg_type);
                }
            }
        }
        
        // Track bytes processed
        let bytes = serde_json::to_vec(&data).unwrap_or_default().len();
        self.stats.write().bytes_processed += bytes as u64;
    }
    
    /// Get orderbook snapshot - peering into the market matrix
    pub fn get_orderbook(&self, symbol: &str) -> Option<OrderBook> {
        self.orderbooks.get(symbol).map(|book| {
            let mut stats = self.stats.write();
            stats.cache_hits += 1;
            book.read().clone()
        }).or_else(|| {
            self.stats.write().cache_misses += 1;
            None
        })
    }
    
    /// Get recent trades - watching the flow
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<Trade> {
        self.recent_trades.get(symbol).map(|trades| {
            let trades = trades.read();
            trades.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        }).unwrap_or_default()
    }
    
    /// Subscribe to market updates - jack into the feed
    pub fn subscribe(&self, symbol: String) -> broadcast::Receiver<MarketEvent> {
        let (tx, rx) = broadcast::channel(1000);
        self.broadcasters.insert(symbol, tx);
        rx
    }
    
    /// Cleanup daemon - neural garbage collection
    async fn cleanup_daemon(self: Arc<Self>) {
        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(CACHE_CLEANUP_INTERVAL)
        );
        
        loop {
            interval.tick().await;
            
            let now = Utc::now();
            let stale_threshold = now - Duration::minutes(30);
            
            // Clean stale orderbooks
            let stale_books: Vec<String> = self.orderbooks
                .iter()
                .filter(|entry| entry.value().read().last_update < stale_threshold)
                .map(|entry| entry.key().clone())
                .collect();
            
            for symbol in stale_books {
                self.orderbooks.remove(&symbol);
                debug!("🧹 Purged stale orderbook for {}", symbol);
            }
            
            // Update cleanup timestamp
            self.stats.write().last_cleanup = Some(now);
            
            debug!(
                "🧠 Neural cache cleanup complete - {} orderbooks, {} trades cached",
                self.orderbooks.len(),
                self.recent_trades.len()
            );
        }
    }
    
    // Parsing functions - translating exchange dialects
    
    fn parse_orderbook_update(&self, data: &serde_json::Value) -> Result<OrderBook, Box<dyn std::error::Error>> {
        // This would be exchange-specific parsing
        // Placeholder implementation
        Ok(OrderBook {
            symbol: data["product_id"].as_str().unwrap_or("UNKNOWN").to_string(),
            exchange: "coinbase".to_string(),
            bids: vec![],
            asks: vec![],
            last_update: Utc::now(),
            sequence_id: data["sequence"].as_u64(),
            checksum: None,
        })
    }
    
    fn parse_trade(&self, data: &serde_json::Value) -> Result<Trade, Box<dyn std::error::Error>> {
        Ok(Trade {
            id: data["trade_id"].as_str().unwrap_or("").to_string(),
            symbol: data["product_id"].as_str().unwrap_or("UNKNOWN").to_string(),
            price: data["price"].as_str().and_then(|p| p.parse().ok()).unwrap_or(0.0),
            quantity: data["size"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            side: if data["side"].as_str() == Some("buy") { TradeSide::Buy } else { TradeSide::Sell },
            timestamp: Utc::now(),
            is_maker: data["maker"].as_bool(),
        })
    }
    
    fn parse_ticker(&self, data: &serde_json::Value) -> Result<Ticker, Box<dyn std::error::Error>> {
        Ok(Ticker {
            symbol: data["product_id"].as_str().unwrap_or("UNKNOWN").to_string(),
            exchange: "coinbase".to_string(),
            bid: data["best_bid"].as_str().and_then(|p| p.parse().ok()).unwrap_or(0.0),
            bid_size: 0.0,
            ask: data["best_ask"].as_str().and_then(|p| p.parse().ok()).unwrap_or(0.0),
            ask_size: 0.0,
            last_price: data["price"].as_str().and_then(|p| p.parse().ok()).unwrap_or(0.0),
            last_size: 0.0,
            volume_24h: data["volume_24h"].as_str().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            vwap_24h: 0.0,
            price_change_24h: 0.0,
            price_change_percent_24h: 0.0,
            high_24h: data["high_24h"].as_str().and_then(|h| h.parse().ok()).unwrap_or(0.0),
            low_24h: data["low_24h"].as_str().and_then(|l| l.parse().ok()).unwrap_or(0.0),
            timestamp: Utc::now(),
        })
    }
    
    /// Get cache statistics - monitoring our neural efficiency
    pub fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        CacheStats {
            total_updates: stats.total_updates,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            bytes_processed: stats.bytes_processed,
            last_cleanup: stats.last_cleanup,
            orderbook_count: self.orderbooks.len(),
            trade_count: self.recent_trades.len(),
            candle_count: self.candles.len(),
        }
    }
}
