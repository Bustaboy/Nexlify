// src-tauri/src/commands/market_data.rs
// NEXLIFY MARKET DATA NEURAL INTERFACE - Where we peek into the market's soul
// Last sync: 2025-06-19 | "The market whispers secrets to those who listen"

use tauri::State;
use tauri::Emitter;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use tracing::{debug, info, warn};

use crate::state::{MarketCache, market_cache::{OrderBook, Ticker, Candle}};
use super::{CommandResult, CommandError, validation};

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

/// Response structure for orderbook queries
#[derive(Debug, Serialize, Deserialize)]
pub struct OrderbookResponse {
    pub orderbook: OrderBook,
    pub spread: Option<f64>,
    pub mid_price: Option<f64>,
    pub depth_returned: usize,
    pub timestamp: DateTime<Utc>,
}

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
    if let Some(ticker) = market_cache.inner().get_ticker(&symbol) {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct TickerResponse {
    pub ticker: Ticker,
    pub momentum_score: f64, // -100 to 100, my proprietary indicator
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
}

/// Subscribe to real-time market data - jack into the feed
/// 
/// This is where we separate the tourists from the natives, choom.
/// Real-time data is like drinking from a fire hose. You either handle it
/// or it handles you. I've seen traders freeze up when the feed goes ballistic.
#[tauri::command]
pub async fn subscribe_market_data(
    symbol: String,
    data_types: Vec<String>,
    window: tauri::Window,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<SubscriptionInfo> {
    validation::validate_symbol(&symbol)?;
    
    // Validate data types - only the essentials
    let valid_types = vec!["orderbook", "trades", "ticker"];
    for data_type in &data_types {
        if !valid_types.contains(&data_type.as_str()) {
            return Err(CommandError::MarketDataError(
                format!("Invalid data type: {}. Stick to the basics, hermano.", data_type)
            ));
        }
    }
    
    let subscription_id = uuid::Uuid::new_v4().to_string();
    let mut rx = market_cache.subscribe(symbol.clone());
    
    info!("🔌 Neural link established for {} - types: {:?}", symbol, data_types);
    
    // Spawn listener task - this is where the magic happens
    let symbol_clone = symbol.clone();
    let data_types_clone = data_types.clone();
    
    tauri::async_runtime::spawn(async move {
        while let Ok(event) = rx.recv().await {
            // Filter based on requested data types
            let should_send = match &event {
                crate::state::market_cache::MarketEvent::OrderbookUpdate { .. } => {
                    data_types_clone.contains(&"orderbook".to_string())
                }
                crate::state::market_cache::MarketEvent::TradeUpdate { .. } => {
                    data_types_clone.contains(&"trades".to_string())
                }
                crate::state::market_cache::MarketEvent::TickerUpdate { .. } => {
                    data_types_clone.contains(&"ticker".to_string())
                }
                _ => false,
            };
            
            if should_send {
                // Emit to frontend - spreading the signal
                let _ = window.emit("market-update", &event);
            }
        }
        
        warn!("📡 Market feed disconnected for {} - the silence is deafening", symbol_clone);
    });
    
    Ok(SubscriptionInfo {
        subscription_id,
        symbol,
        data_types,
        status: "active".to_string(),
        message: "Neural link established. May the data flow through you.".to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    pub subscription_id: String,
    pub symbol: String,
    pub data_types: Vec<String>,
    pub status: String,
    pub message: String,
}

/// Unsubscribe from market data - sometimes you gotta unplug
/// 
/// Not every signal needs to reach your cortex, ¿me entiendes?
/// Sometimes the bravest thing is knowing when to disconnect.
#[tauri::command]
pub async fn unsubscribe_market_data(
    subscription_id: String,
) -> CommandResult<String> {
    // In a real implementation, we'd track subscriptions
    // For now, we acknowledge the request
    
    info!("🔌 Unplugging subscription: {}", subscription_id);
    
    Ok(format!("Disconnected {}. Sometimes silence is golden.", subscription_id))
}

/// Get historical candles - learning from the past to survive the future
/// 
/// They say those who don't learn from history are doomed to repeat it.
/// In trading, they're doomed to poverty. I've got scars from every major crash,
/// and each one taught me something. These candles? They're battle stories.
#[tauri::command]
pub async fn get_historical_candles(
    symbol: String,
    timeframe: String,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    limit: Option<usize>,
    market_cache: State<'_, Arc<MarketCache>>,
) -> CommandResult<CandleResponse> {
    validation::validate_symbol(&symbol)?;
    
    // Validate timeframe - gotta speak the same language
    let valid_timeframes = vec!["1m", "5m", "15m", "1h", "4h", "1d"];
    if !valid_timeframes.contains(&timeframe.as_str()) {
        return Err(CommandError::MarketDataError(
            format!("Invalid timeframe: {}. Time moves differently in the sprawl.", timeframe)
        ));
    }
    
    let limit = limit.unwrap_or(100).min(1000); // Cap at 1000 - memory is precious
    
    // For now, return mock data - in production, this would query our time-series DB
    let mut candles = Vec::new();
    let start = start_time.unwrap_or(Utc::now() - Duration::hours(24));
    let interval = parse_timeframe_to_duration(&timeframe)?;
    
    for i in 0..limit {
        let timestamp = start + (interval * i as i32);
        let base_price = 50000.0 + (i as f64 * 10.0); // Mock price movement
        
        candles.push(Candle {
            timestamp,
            open: base_price,
            high: base_price + 50.0,
            low: base_price - 30.0,
            close: base_price + 20.0,
            volume: 100.0 + (i as f64 * 5.0),
            trades: 100 + i as u32,
        });
    }
    
    info!("📊 Retrieved {} candles for {} {}", candles.len(), symbol, timeframe);
    
    Ok(CandleResponse {
        symbol,
        timeframe,
        candles: candles.clone(),
        start_time: start,
        end_time: candles.last().map(|c| c.timestamp).unwrap_or(start),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CandleResponse {
    pub symbol: String,
    pub timeframe: String,
    pub candles: Vec<Candle>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

// Helper functions - the tricks of the trade

/// Calculate momentum score from ticker data
/// This is my baby - developed it during the 2017 bull run
fn calculate_momentum(ticker: &Ticker) -> f64 {
    let price_change_weight = 0.4;
    let volume_weight = 0.3;
    let spread_weight = 0.3;
    
    // Normalize price change to -100 to 100
    let price_momentum = (ticker.price_change_percent_24h / 50.0 * 100.0)
        .max(-100.0)
        .min(100.0);
    
    // Volume momentum (comparing to assumed average)
    let avg_volume = 1000.0; // This would be calculated from historical data
    let volume_momentum = ((ticker.volume_24h / avg_volume - 1.0) * 50.0)
        .max(-100.0)
        .min(100.0);
    
    // Spread tightness (tighter = better liquidity = higher score)
    let spread = ticker.ask - ticker.bid;
    let spread_percentage = (spread / ticker.mid_price()) * 100.0;
    let spread_momentum = (1.0 - spread_percentage.min(1.0)) * 100.0;
    
    price_momentum * price_change_weight + 
    volume_momentum * volume_weight + 
    spread_momentum * spread_weight
}

/// Derive ticker from orderbook when direct ticker isn't available
/// Sometimes you gotta work with what you got, ¿no?
fn derive_ticker_from_orderbook(symbol: &str, orderbook: &OrderBook) -> Ticker {
    let bid = orderbook.bids.first().map(|b| b.price).unwrap_or(0.0);
    let ask = orderbook.asks.first().map(|a| a.price).unwrap_or(0.0);
    let mid = if bid > 0.0 && ask > 0.0 { (bid + ask) / 2.0 } else { 0.0 };
    
    Ticker {
        symbol: symbol.to_string(),
        exchange: orderbook.exchange.clone(),
        bid,
        bid_size: orderbook.bids.first().map(|b| b.quantity).unwrap_or(0.0),
        ask,
        ask_size: orderbook.asks.first().map(|a| a.quantity).unwrap_or(0.0),
        last_price: mid, // Best guess
        last_size: 0.0,
        volume_24h: 0.0, // Unknown from orderbook
        vwap_24h: 0.0,
        price_change_24h: 0.0,
        price_change_percent_24h: 0.0,
        high_24h: 0.0,
        low_24h: 0.0,
        timestamp: orderbook.last_update,
    }
}

/// Parse timeframe string to duration
fn parse_timeframe_to_duration(timeframe: &str) -> Result<Duration, CommandError> {
    match timeframe {
        "1m" => Ok(Duration::minutes(1)),
        "5m" => Ok(Duration::minutes(5)),
        "15m" => Ok(Duration::minutes(15)),
        "1h" => Ok(Duration::hours(1)),
        "4h" => Ok(Duration::hours(4)),
        "1d" => Ok(Duration::days(1)),
        _ => Err(CommandError::MarketDataError(
            format!("Unknown timeframe: {} - time is relative, but not that relative", timeframe)
        )),
    }
}

// Extension trait for ticker
impl Ticker {
    fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
}
