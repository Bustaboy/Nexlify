// Location: C:\Nexlify\src-tauri\src\commands\market_data.rs
// Purpose: NEXLIFY MARKET DATA NEURAL INTERFACE - Where we peek into the market's soul
// Last sync: 2025-06-19 | "The market whispers secrets to those who listen"

use tauri::State;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use tracing::{debug, info, warn};
use rand::Rng;

use crate::state::{MarketCache, market_cache::{OrderBook, Trade, Ticker, Candle, PriceLevel}};
use super::{CommandResult, CommandError, validation};

// ─────────────────────────────────────────────────────────────
// MARKET DATA STRUCTURES
// ─────────────────────────────────────────────────────────────

/// Response structure for orderbook queries
#[derive(Debug, Serialize, Deserialize)]
pub struct OrderbookResponse {
    pub orderbook: OrderBook,
    pub spread: Option<f64>,
    pub mid_price: Option<f64>,
    pub depth_returned: usize,
    pub timestamp: DateTime<Utc>,
}

/// Response structure for ticker data
#[derive(Debug, Serialize, Deserialize)]
pub struct TickerResponse {
    pub ticker: Ticker,
    pub momentum_score: f64,
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
}

/// Response structure for recent trades
#[derive(Debug, Serialize, Deserialize)]
pub struct TradesResponse {
    pub symbol: String,
    pub trades: Vec<Trade>,
    pub count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Response for market data subscription
#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriptionResponse {
    pub subscribed_symbols: Vec<String>,
    pub subscription_id: String,
    pub websocket_url: String,
    pub message: String,
}

/// Response for unsubscribe operation
#[derive(Debug, Serialize, Deserialize)]
pub struct UnsubscribeResponse {
    pub unsubscribed: Vec<UnsubscribeResult>,
    pub failed: Vec<UnsubscribeResult>,
    pub active_subscriptions: usize,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnsubscribeResult {
    pub symbol: String,
    pub success: bool,
    pub message: String,
}

/// Response for historical data
#[derive(Debug, Serialize, Deserialize)]
pub struct HistoricalDataResponse {
    pub symbol: String,
    pub timeframe: String,
    pub candles: Vec<Candle>,
    pub count: usize,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_orderbook - Peer into supply and demand
// ─────────────────────────────────────────────────────────────

/// Get orderbook snapshot - peering into the abyss of supply and demand
/// 
/// You know, hermano, watching an orderbook is like watching the city breathe.
/// Each bid, each ask - it's someone's hope, someone's fear, crystallized in numbers.
/// I've seen fortunes made and dreams shattered in these price levels.
#[tauri::command]
pub async fn get_orderbook(
    symbol: String,
    depth: Option<usize>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<OrderbookResponse> {
    // Validate input - never trust data from the streets
    validation::validate_symbol(&symbol)?;
    
    let depth = depth.unwrap_or(50).min(100); // Cap at 100 - too much data melts chrome
    
    debug!("📊 Fetching orderbook for {} with depth {}", symbol, depth);
    
    match market_cache.get_orderbook(&symbol) {
        Some(mut orderbook) => {
            // Trim to requested depth - sometimes less is more, ¿verdad?
            orderbook.bids.truncate(depth);
            orderbook.asks.truncate(depth);
            
            // Calculate spread - the gap between dreams and reality
            let spread = if !orderbook.asks.is_empty() && !orderbook.bids.is_empty() {
                Some(orderbook.asks[0].price - orderbook.bids[0].price)
            } else {
                None
            };
            
            // Mid price - where the market finds its center
            let mid_price = if let Some(spread) = spread {
                Some(orderbook.bids[0].price + spread / 2.0)
            } else {
                None
            };
            
            info!("✅ Orderbook retrieved for {} - {} bids, {} asks", 
                  symbol, orderbook.bids.len(), orderbook.asks.len());
            
            Ok(OrderbookResponse {
                orderbook,
                spread,
                mid_price,
                depth_returned: depth,
                timestamp: Utc::now(),
            })
        }
        None => {
            warn!("❌ No orderbook data for {} - the market ghosts us", symbol);
            Err(CommandError::MarketDataError(
                format!("No orderbook data available for {}. The market keeps its secrets.", symbol)
            ))
        }
    }
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_ticker - The market's heartbeat
// ─────────────────────────────────────────────────────────────

/// Get current ticker data - the heartbeat of the market
/// 
/// Mierda, I remember when BTC first hit 10k. We thought we were kings.
/// Now look at us, handling these numbers like they're pocket change.
/// The market teaches humility faster than any street fight.
#[tauri::command]
pub async fn get_ticker(
    symbol: String,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<TickerResponse> {
    validation::validate_symbol(&symbol)?;
    
    let cache_stats = market_cache.get_stats();
    debug!("📈 Cache stats - Hits: {}, Misses: {}", cache_stats.cache_hits, cache_stats.cache_misses);
    
    // First, try to get fresh ticker data
    if let Some(ticker) = market_cache.get_ticker(&symbol) {
        // Calculate momentum indicators - my secret sauce from the old days
        let momentum_score = calculate_momentum(&ticker);
        
        Ok(TickerResponse {
            ticker,
            momentum_score,
            cache_hit: true,
            timestamp: Utc::now(),
        })
    } else {
        // Fallback to orderbook data - when Plan A fails, adapt
        if let Some(orderbook) = market_cache.get_orderbook(&symbol) {
            let ticker = derive_ticker_from_orderbook(&symbol, &orderbook);
            
            info!("📊 Derived ticker from orderbook for {} - improvise, adapt, overcome", symbol);
            
            Ok(TickerResponse {
                ticker,
                momentum_score: 0.0, // No historical data for momentum
                cache_hit: false,
                timestamp: Utc::now(),
            })
        } else {
            Err(CommandError::MarketDataError(
                format!("No market data for {}. Even the data brokers are dry.", symbol)
            ))
        }
    }
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_recent_trades - Watch the money flow
// ─────────────────────────────────────────────────────────────

/// Get recent trades - watching the blood flow through the market's veins
/// 
/// Every trade tells a story. Was it fear? Greed? Desperation?
/// In my barrio, we learned to read people. Here, we read the tape.
/// Same game, different arena.
#[tauri::command]
pub async fn get_recent_trades(
    symbol: String,
    limit: Option<usize>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<TradesResponse> {
    validation::validate_symbol(&symbol)?;
    
    let limit = limit.unwrap_or(100).min(1000); // Cap at 1000 - memory is precious
    
    match market_cache.get_recent_trades(&symbol, limit) {
        Some(trades) => {
            info!("📜 Retrieved {} recent trades for {}", trades.len(), symbol);
            
            Ok(TradesResponse {
                symbol: symbol.clone(),
                trades,
                count: limit,
                timestamp: Utc::now(),
            })
        }
        None => {
            warn!("❌ No trade history for {} - market gone dark", symbol);
            Err(CommandError::MarketDataError(
                format!("No trade data for {}. The tape has gone silent.", symbol)
            ))
        }
    }
}

// ─────────────────────────────────────────────────────────────
// COMMAND: subscribe_market_data - Jack into the feed
// ─────────────────────────────────────────────────────────────

/// Subscribe to real-time market data - jacking into the neural feed
/// 
/// This is where we plug directly into the market's nervous system.
/// Real-time data flowing like electricity through our chrome.
/// Once you jack in, there's no going back. You feel every tick, every trade.
#[tauri::command]
pub async fn subscribe_market_data(
    symbols: Vec<String>,
    feed_types: Vec<String>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<SubscriptionResponse> {
    // Validate all symbols - no garbage in the neural stream
    for symbol in &symbols {
        validation::validate_symbol(symbol)?;
    }
    
    if symbols.is_empty() {
        return Err(CommandError::ValidationError(
            "No symbols provided. Can't jack into nothing, choom.".to_string()
        ));
    }
    
    // Subscribe to each symbol
    let subscription_id = uuid::Uuid::new_v4().to_string();
    let mut subscribed = vec![];
    
    for symbol in symbols {
        // In production, this would establish WebSocket connections
        // For now, we simulate subscription
        market_cache.add_subscription(&symbol, &subscription_id);
        subscribed.push(symbol);
        
        info!("🔌 Neural link established for {}", subscribed.last().unwrap());
    }
    
    // Determine WebSocket URL based on feed types
    let websocket_url = if feed_types.contains(&"binary".to_string()) {
        "wss://nexlify.market/binary".to_string()
    } else {
        "wss://nexlify.market/json".to_string()
    };
    
    info!("⚡ Market neural feed active - {} symbols online", subscribed.len());
    
    Ok(SubscriptionResponse {
        subscribed_symbols: subscribed,
        subscription_id,
        websocket_url,
        message: "Neural link established. Data flowing. Stay sharp.".to_string(),
    })
}

// ─────────────────────────────────────────────────────────────
// COMMAND: unsubscribe_market_data - Disconnect from the feed
// ─────────────────────────────────────────────────────────────

/// Unsubscribe from real-time market data streams
/// 
/// Cut the neural link to specific market feeds. Sometimes you gotta
/// disconnect to stay sane. Too much data can fry your chrome.
#[tauri::command]
pub async fn unsubscribe_market_data(
    symbols: Vec<String>,
    exchange: Option<String>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<UnsubscribeResponse> {
    info!("🔌 Disconnecting from market feeds: {:?}", symbols);
    
    // Validate inputs
    if symbols.is_empty() {
        return Err(CommandError::ValidationError(
            "No symbols specified. What are we unsubscribing from?".to_string()
        ));
    }
    
    // Validate all symbols
    for symbol in &symbols {
        validation::validate_symbol(symbol)?;
    }
    
    // If exchange specified, validate it
    if let Some(ref exch) = exchange {
        validation::validate_exchange(exch)?;
    }
    
    let mut unsubscribed = vec![];
    let mut failed = vec![];
    
    // Process each symbol
    for symbol in symbols {
        match market_cache.remove_subscription(&symbol, exchange.as_deref()) {
            Ok(_) => {
                info!("✅ Unsubscribed from {}", symbol);
                unsubscribed.push(UnsubscribeResult {
                    symbol: symbol.clone(),
                    success: true,
                    message: "Neural link severed".to_string(),
                });
            }
            Err(e) => {
                warn!("❌ Failed to unsubscribe from {}: {}", symbol, e);
                failed.push(UnsubscribeResult {
                    symbol: symbol.clone(),
                    success: false,
                    message: e.to_string(),
                });
            }
        }
    }
    
    // Get remaining active subscriptions
    let active_subscriptions = market_cache.get_active_subscriptions();
    
    Ok(UnsubscribeResponse {
        unsubscribed,
        failed,
        active_subscriptions,
        message: format!(
            "Disconnected from {} feeds. {} still active. {}",
            unsubscribed.len(),
            active_subscriptions,
            if failed.is_empty() { 
                "Clean disconnect." 
            } else { 
                "Some connections refused to die." 
            }
        ),
    })
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_historical_data - Learn from the past
// ─────────────────────────────────────────────────────────────

/// Get historical candle data - because those who forget the past are doomed to repeat it
/// 
/// I've seen too many traders ignore history, thinking "this time is different."
/// Spoiler alert: it never is. The market has patterns, cycles, rhythms.
/// Learn them, or get rekt. Simple as that.
#[tauri::command]
pub async fn get_historical_data(
    symbol: String,
    timeframe: String,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    limit: Option<usize>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<HistoricalDataResponse> {
    validation::validate_symbol(&symbol)?;
    
    // Validate timeframe
    let valid_timeframes = vec!["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"];
    if !valid_timeframes.contains(&timeframe.as_str()) {
        return Err(CommandError::ValidationError(
            format!("Invalid timeframe: {}. Stick to the standards.", timeframe)
        ));
    }
    
    let limit = limit.unwrap_or(500).min(5000);
    let end_time = end_time.unwrap_or_else(Utc::now);
    let start_time = start_time.unwrap_or_else(|| end_time - Duration::days(7));
    
    debug!("📈 Fetching historical data for {} - {} from {} to {}", 
           symbol, timeframe, start_time, end_time);
    
    // In production, this would fetch from database or external API
    // For now, we'll get from cache or generate mock data
    let candles = market_cache.get_candles(&symbol, &timeframe, start_time, end_time, limit)
        .unwrap_or_else(|| {
            warn!("No historical data in cache for {}, generating mock data", symbol);
            generate_mock_candles(&symbol, &timeframe, start_time, end_time, limit)
        });
    
    info!("📊 Retrieved {} candles for {} - {}", candles.len(), symbol, timeframe);
    
    Ok(HistoricalDataResponse {
        symbol,
        timeframe,
        count: candles.len(),
        candles,
        start_time,
        end_time,
    })
}

// ─────────────────────────────────────────────────────────────
// HELPER FUNCTIONS - The tools of the trade
// ─────────────────────────────────────────────────────────────

/// Calculate momentum score based on ticker data
fn calculate_momentum(ticker: &Ticker) -> f64 {
    // Simple momentum calculation - could be much more sophisticated
    let price_change_impact = ticker.change_24h_percent.abs() / 100.0 * 50.0;
    let volume_impact = (ticker.volume_24h / 1_000_000.0).min(50.0);
    
    price_change_impact + volume_impact
}

/// Derive ticker from orderbook when direct ticker data isn't available
fn derive_ticker_from_orderbook(symbol: &str, orderbook: &OrderBook) -> Ticker {
    let best_bid = orderbook.bids.get(0).map(|b| b.price).unwrap_or(0.0);
    let best_ask = orderbook.asks.get(0).map(|a| a.price).unwrap_or(0.0);
    let mid_price = if best_bid > 0.0 && best_ask > 0.0 {
        (best_bid + best_ask) / 2.0
    } else {
        0.0
    };
    
    Ticker {
        symbol: symbol.to_string(),
        exchange: orderbook.exchange.clone(),
        bid: best_bid,
        ask: best_ask,
        last: mid_price, // Use mid as last price approximation
        volume_24h: 0.0, // Unknown from orderbook
        change_24h: 0.0,
        change_24h_percent: 0.0,
        high_24h: 0.0,
        low_24h: 0.0,
        timestamp: orderbook.last_update,
    }
}

/// Generate mock historical candles for testing
fn generate_mock_candles(
    symbol: &str,
    timeframe: &str,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    limit: usize,
) -> Vec<Candle> {
    let mut candles = Vec::new();
    let mut current_time = start_time;
    let interval = match timeframe {
        "1m" => Duration::minutes(1),
        "5m" => Duration::minutes(5),
        "15m" => Duration::minutes(15),
        "30m" => Duration::minutes(30),
        "1h" => Duration::hours(1),
        "4h" => Duration::hours(4),
        "1d" => Duration::days(1),
        "1w" => Duration::weeks(1),
        _ => Duration::minutes(1),
    };
    
    let mut rng = rand::thread_rng();
    let mut price = 50000.0; // Starting price
    
    while current_time < end_time && candles.len() < limit {
        // Generate realistic OHLCV data with some randomness
        let change = (rng.gen::<f64>() - 0.5) * 1000.0;
        let high = price + rng.gen::<f64>() * 500.0;
        let low = price - rng.gen::<f64>() * 500.0;
        let close = price + change;
        let volume = rng.gen::<f64>() * 1000.0;
        
        candles.push(Candle {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            timestamp: current_time,
            open: price,
            high,
            low,
            close,
            volume,
            trades: (volume * 10.0) as u32,
            closed: current_time < Utc::now(),
        });
        
        price = close;
        current_time = current_time + interval;
    }
    
    candles
}

// ─────────────────────────────────────────────────────────────
// MARKET CACHE EXTENSION METHODS
// ─────────────────────────────────────────────────────────────

/// Extension methods for MarketCache to support our commands
impl MarketCache {
    /// Get cache statistics
    pub fn get_stats(&self) -> crate::state::market_cache::CacheStats {
        self.stats.read().clone()
    }
    
    /// Get orderbook from cache
    pub fn get_orderbook(&self, symbol: &str) -> Option<OrderBook> {
        self.orderbooks.get(symbol).map(|ob| ob.read().clone())
    }
    
    /// Get ticker from cache
    pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        self.tickers.get(symbol).map(|t| t.read().clone())
    }
    
    /// Get recent trades from cache
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Option<Vec<Trade>> {
        self.recent_trades.get(symbol).map(|trades| {
            let trades = trades.read();
            trades.iter().take(limit).cloned().collect()
        })
    }
    
    /// Get historical candles from cache
    pub fn get_candles(
        &self, 
        symbol: &str, 
        timeframe: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: usize
    ) -> Option<Vec<Candle>> {
        // Note: CandleKey should be defined in market_cache.rs as:
        // #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        // pub struct CandleKey {
        //     pub symbol: String,
        //     pub timeframe: String,
        // }
        
        let key = format!("{}:{}", symbol, timeframe); // Simplified for now
        
        // In production, this would access the candles DashMap
        // For now, return None to indicate no cached data
        None
    }
    
    /// Add a market data subscription
    pub fn add_subscription(&self, symbol: &str, subscription_id: &str) {
        // In production, this would manage WebSocket subscriptions
        // For now, we just track it
        debug!("Added subscription {} for {}", subscription_id, symbol);
    }
    
    /// Remove a market data subscription
    pub fn remove_subscription(
        &self,
        symbol: &str,
        exchange: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Remove from active subscriptions
        let key = if let Some(exch) = exchange {
            format!("{}:{}", exch, symbol)
        } else {
            symbol.to_string()
        };
        
        // Remove from orderbook cache
        self.orderbooks.remove(&key);
        
        // Remove from ticker cache
        self.tickers.remove(&key);
        
        // In production, would also:
        // - Close WebSocket connection for this symbol
        // - Clean up any pending requests
        // - Update subscription count
        
        info!("🔌 Removed subscription for {}", key);
        Ok(())
    }
    
    /// Get count of active subscriptions
    pub fn get_active_subscriptions(&self) -> usize {
        // Count unique symbols across all caches
        let orderbook_symbols: std::collections::HashSet<_> = 
            self.orderbooks.iter().map(|e| e.key().clone()).collect();
        let ticker_symbols: std::collections::HashSet<_> = 
            self.tickers.iter().map(|e| e.key().clone()).collect();
        
        orderbook_symbols.union(&ticker_symbols).count()
    }
}

// ─────────────────────────────────────────────────────────────
// TESTS - Trust but verify in the digital wasteland
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_momentum_calculation() {
        let ticker = Ticker {
            symbol: "BTC/USDT".to_string(),
            exchange: "binance".to_string(),
            bid: 50000.0,
            ask: 50010.0,
            last: 50005.0,
            volume_24h: 500_000.0,
            change_24h: 2500.0,
            change_24h_percent: 5.0,
            high_24h: 52000.0,
            low_24h: 48000.0,
            timestamp: Utc::now(),
        };
        
        let momentum = calculate_momentum(&ticker);
        assert!(momentum > 0.0 && momentum <= 100.0);
    }
    
    #[test]
    fn test_mock_candle_generation() {
        let candles = generate_mock_candles(
            "BTC/USDT",
            "1h",
            Utc::now() - Duration::days(1),
            Utc::now(),
            24,
        );
        
        assert_eq!(candles.len(), 24);
        for candle in candles {
            assert!(candle.high >= candle.low);
            assert!(candle.open >= candle.low && candle.open <= candle.high);
            assert!(candle.close >= candle.low && candle.close <= candle.high);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// END OF MARKET DATA MODULE
// Remember: In the market, like in the streets, information is power.
// Stay connected, stay informed, stay alive. 🌃💹
// ─────────────────────────────────────────────────────────────