// src-tauri/src/commands/market_data.rs
// NEXLIFY MARKET DATA COMMANDS - The pulse of the digital bazaar
// Last sync: 2025-06-22 | "Information is ammunition in the war of profit"

use tauri::State;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, info, warn, error};

use crate::state::{
    AppState, MarketCache, TradingEngine,
    OrderBook as StateOrderBook,
    Ticker as StateTicker,
    Trade as StateTrade,
    Candle as StateCandle,
    OrderStatus
};
use super::{CommandResult, CommandError, validation};

/// Get orderbook data - see the battlefield before you charge
/// 
/// The orderbook is truth. It shows you where the bodies are buried
/// (limit orders) and where the next massacre might happen (big walls).
/// I've seen 10k BTC walls vanish in seconds. Never trust, always verify.
#[tauri::command]
pub async fn get_orderbook(
    symbol: String,
    depth: Option<u32>,
    market_cache: State<'_, Arc<RwLock<MarketCache>>>,
) -> CommandResult<OrderBookResponse> {
    validation::validate_symbol(&symbol)?;
    
    let depth = depth.unwrap_or(20).min(100); // Cap at 100 for performance
    
    let cache = market_cache.read();
    let orderbook = cache.get_orderbook(&symbol)
        .ok_or_else(|| CommandError::MarketDataError(
            format!("No orderbook data for {}. The market ghosts are silent.", symbol)
        ))?;
    
    // Calculate some chrome-plated metrics
    let total_bid_volume: f64 = orderbook.bids.iter().take(depth as usize).map(|o| o.quantity).sum();
    let total_ask_volume: f64 = orderbook.asks.iter().take(depth as usize).map(|o| o.quantity).sum();
    let mid_price = if !orderbook.bids.is_empty() && !orderbook.asks.is_empty() {
        (orderbook.bids[0].price + orderbook.asks[0].price) / 2.0
    } else {
        0.0
    };
    
    let imbalance = if total_bid_volume + total_ask_volume > 0.0 {
        (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    } else {
        0.0
    };
    
    debug!("📊 Orderbook for {} - Depth: {}, Imbalance: {:.2}%", 
           symbol, depth, imbalance * 100.0);
    
    Ok(OrderBookResponse {
        symbol: symbol.clone(),
        bids: orderbook.bids.iter().take(depth as usize).cloned().collect(),
        asks: orderbook.asks.iter().take(depth as usize).cloned().collect(),
        timestamp: orderbook.timestamp,
        mid_price,
        spread: orderbook.asks.first().map(|a| a.price).unwrap_or(0.0) - 
                orderbook.bids.first().map(|b| b.price).unwrap_or(0.0),
        imbalance,
        total_bid_volume,
        total_ask_volume,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderBookResponse {
    pub symbol: String,
    pub bids: Vec<crate::state::OrderBookLevel>,
    pub asks: Vec<crate::state::OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
    pub mid_price: f64,
    pub spread: f64,
    pub imbalance: f64, // -1 to 1, negative = sell pressure
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
}

/// Get ticker data - the heartbeat of the market
/// 
/// Tickers are like vital signs. When they flatline, something's wrong.
/// When they spike, something's happening. Learn to read the rhythm.
#[tauri::command]
pub async fn get_ticker(
    symbol: String,
    market_cache: State<'_, Arc<RwLock<MarketCache>>>,
) -> CommandResult<TickerResponse> {
    validation::validate_symbol(&symbol)?;
    
    let cache = market_cache.read();
    let ticker = cache.get_ticker(&symbol)
        .ok_or_else(|| CommandError::MarketDataError(
            format!("No ticker data for {}. Market's gone dark.", symbol)
        ))?;
    
    // Calculate momentum (simple but effective)
    let momentum = if ticker.last_price > 0.0 && ticker.open_24h > 0.0 {
        ((ticker.last_price - ticker.open_24h) / ticker.open_24h) * 100.0
    } else {
        0.0
    };
    
    // Volatility indicator (high-low spread)
    let volatility = if ticker.high_24h > 0.0 && ticker.low_24h > 0.0 {
        ((ticker.high_24h - ticker.low_24h) / ticker.low_24h) * 100.0
    } else {
        0.0
    };
    
    Ok(TickerResponse {
        symbol: ticker.symbol.clone(),
        bid: ticker.bid,
        ask: ticker.ask,
        last_price: ticker.last_price,
        volume_24h: ticker.volume_24h,
        change_24h: ticker.change_24h,
        change_percent_24h: ticker.change_percent_24h,
        high_24h: ticker.high_24h,
        low_24h: ticker.low_24h,
        open_24h: ticker.open_24h,
        timestamp: ticker.timestamp,
        momentum,
        volatility,
        health_status: match volatility {
            v if v > 20.0 => "EXTREME - Cyber-storm detected! 🌪️".to_string(),
            v if v > 10.0 => "HIGH - Market's running hot 🔥".to_string(),
            v if v > 5.0 => "MODERATE - Normal chaos levels 📊".to_string(),
            _ => "LOW - Eerily quiet... too quiet 🤫".to_string(),
        },
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TickerResponse {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub last_price: f64,
    pub volume_24h: f64,
    pub change_24h: f64,
    pub change_percent_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub open_24h: f64,
    pub timestamp: DateTime<Utc>,
    pub momentum: f64,
    pub volatility: f64,
    pub health_status: String,
}

/// Get recent trades - watch the money flow
/// 
/// Every trade tells a story. Big trades are whales moving. Small trades
/// are retail getting chopped. The tape never lies, but it sure loves to
/// mislead.
#[tauri::command]
pub async fn get_recent_trades(
    symbol: String,
    limit: Option<u32>,
    market_cache: State<'_, Arc<RwLock<MarketCache>>>,
) -> CommandResult<RecentTradesResponse> {
    validation::validate_symbol(&symbol)?;
    
    let limit = limit.unwrap_or(50).min(100);
    
    let cache = market_cache.read();
    let trades = cache.get_recent_trades(&symbol, limit as usize);
    
    if trades.is_empty() {
        return Err(CommandError::MarketDataError(
            format!("No trade data for {}. Market's frozen in time.", symbol)
        ));
    }
    
    // Analyze the flow
    let buy_volume: f64 = trades.iter()
        .filter(|t| t.is_buyer_maker)
        .map(|t| t.quantity)
        .sum();
    
    let sell_volume: f64 = trades.iter()
        .filter(|t| !t.is_buyer_maker)
        .map(|t| t.quantity)
        .sum();
    
    let avg_trade_size = trades.iter()
        .map(|t| t.quantity)
        .sum::<f64>() / trades.len() as f64;
    
    let large_trades = trades.iter()
        .filter(|t| t.quantity > avg_trade_size * 5.0)
        .count();
    
    Ok(RecentTradesResponse {
        symbol,
        trades: trades.into_iter().map(|t| TradeInfo {
            id: t.id,
            price: t.price,
            quantity: t.quantity,
            timestamp: t.timestamp,
            is_buyer_maker: t.is_buyer_maker,
            is_large: t.quantity > avg_trade_size * 5.0,
        }).collect(),
        buy_volume,
        sell_volume,
        volume_ratio: if sell_volume > 0.0 { buy_volume / sell_volume } else { 0.0 },
        avg_trade_size,
        large_trades,
        flow_analysis: match buy_volume / (buy_volume + sell_volume) {
            r if r > 0.7 => "BULLISH - Buyers in control 🟢".to_string(),
            r if r > 0.55 => "LEAN BULLISH - Slight buy pressure 📈".to_string(),
            r if r > 0.45 => "NEUTRAL - Balanced flow ⚖️".to_string(),
            r if r > 0.3 => "LEAN BEARISH - Sellers emerging 📉".to_string(),
            _ => "BEARISH - Sellers dominating 🔴".to_string(),
        },
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecentTradesResponse {
    pub symbol: String,
    pub trades: Vec<TradeInfo>,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub volume_ratio: f64,
    pub avg_trade_size: f64,
    pub large_trades: usize,
    pub flow_analysis: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TradeInfo {
    pub id: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: DateTime<Utc>,
    pub is_buyer_maker: bool,
    pub is_large: bool,
}

/// Subscribe to market data - jack into the matrix
/// 
/// Real-time data is the lifeblood of trading. Miss a tick, miss an opportunity.
/// But be careful - too much data can overwhelm. Filter the signal from the noise.
#[tauri::command]
pub async fn subscribe_market_data(
    symbol: String,
    data_types: Vec<String>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<SubscriptionResponse> {
    validation::validate_symbol(&symbol)?;
    
    if data_types.is_empty() {
        return Err(CommandError::MarketDataError(
            "No data types specified. Need to know what chrome you want.".to_string()
        ));
    }
    
    // Validate data types
    let valid_types = vec!["ticker", "orderbook", "trades", "candles"];
    for dt in &data_types {
        if !valid_types.contains(&dt.as_str()) {
            return Err(CommandError::MarketDataError(
                format!("Invalid data type: {}. Choose from: {:?}", dt, valid_types)
            ));
        }
    }
    
    info!("🔌 Subscribing to {} - Types: {:?}", symbol, data_types);
    
    // In production, this would establish WebSocket subscriptions
    // For now, we'll simulate the subscription
    let subscription_id = format!("SUB-{}-{}", symbol, uuid::Uuid::new_v4());
    
    // Update app state
    let mut state = app_state.write();
    state.active_subscriptions.insert(
        subscription_id.clone(),
        (symbol.clone(), data_types.clone())
    );
    
    Ok(SubscriptionResponse {
        subscription_id,
        symbol,
        data_types,
        status: "active".to_string(),
        message: "Jacked in. Data stream active. 🔌".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriptionResponse {
    pub subscription_id: String,
    pub symbol: String,
    pub data_types: Vec<String>,
    pub status: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Get market overview - bird's eye view of the battlefield
/// 
/// Sometimes you need to zoom out. See the whole market, not just your
/// little corner. The big picture often reveals what the details hide.
#[tauri::command]
pub async fn get_market_overview(
    market_cache: State<'_, Arc<RwLock<MarketCache>>>,
) -> CommandResult<MarketOverview> {
    let cache = market_cache.read();
    
    // Collect all tickers
    let mut market_caps: Vec<(String, f64)> = Vec::new();
    let mut gainers: Vec<(String, f64)> = Vec::new();
    let mut losers: Vec<(String, f64)> = Vec::new();
    let mut volume_leaders: Vec<(String, f64)> = Vec::new();
    
    // This is a simplified version - in production, we'd have proper market data
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"] {
        if let Some(ticker) = cache.get_ticker(symbol) {
            // Market cap (simplified - would need circulating supply)
            market_caps.push((symbol.to_string(), ticker.last_price * ticker.volume_24h));
            
            // Gainers/Losers
            if ticker.change_percent_24h > 0.0 {
                gainers.push((symbol.to_string(), ticker.change_percent_24h));
            } else {
                losers.push((symbol.to_string(), ticker.change_percent_24h));
            }
            
            // Volume leaders
            volume_leaders.push((symbol.to_string(), ticker.volume_24h));
        }
    }
    
    // Sort and limit
    gainers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    losers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    volume_leaders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let total_volume: f64 = volume_leaders.iter().map(|(_, v)| v).sum();
    let avg_change: f64 = (gainers.iter().map(|(_, c)| c).sum::<f64>() + 
                           losers.iter().map(|(_, c)| c).sum::<f64>()) / 
                          (gainers.len() + losers.len()) as f64;
    
    Ok(MarketOverview {
        total_market_cap: market_caps.iter().map(|(_, mc)| mc).sum(),
        total_24h_volume: total_volume,
        btc_dominance: 42.5, // Would calculate from real data
        market_sentiment: match avg_change {
            c if c > 5.0 => "EUPHORIC - Peak greed detected! 🚀".to_string(),
            c if c > 2.0 => "BULLISH - Green across the board 📈".to_string(),
            c if c > -2.0 => "NEUTRAL - Market catching its breath 😴".to_string(),
            c if c > -5.0 => "BEARISH - Red wedding in progress 📉".to_string(),
            _ => "PANIC - Blood in the streets! 🩸".to_string(),
        },
        top_gainers: gainers.into_iter().take(5).collect(),
        top_losers: losers.into_iter().take(5).collect(),
        volume_leaders: volume_leaders.into_iter().take(5).collect(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketOverview {
    pub total_market_cap: f64,
    pub total_24h_volume: f64,
    pub btc_dominance: f64,
    pub market_sentiment: String,
    pub top_gainers: Vec<(String, f64)>,
    pub top_losers: Vec<(String, f64)>,
    pub volume_leaders: Vec<(String, f64)>,
    pub timestamp: DateTime<Utc>,
}

/// Process market update and check for triggered stops
/// 
/// This is where the rubber meets the road. Every price update could
/// trigger a cascade of stop orders. Handle with care.
#[tauri::command]
pub async fn process_price_update(
    symbol: String,
    price: f64,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<Vec<String>> {
    validation::validate_symbol(&symbol)?;
    validation::validate_price(price)?;
    
    // Check for triggered stop orders
    let triggered_orders = {
        let engine = trading_engine.read();
        engine.check_stop_orders(&symbol, price)
    };
    
    if !triggered_orders.is_empty() {
        info!("⚡ {} stop orders triggered for {} at ${}", 
              triggered_orders.len(), symbol, price);
    }
    
    // Activate triggered orders
    let mut activated = Vec::new();
    for order_id in triggered_orders {
        let mut engine = trading_engine.write();
        match engine.activate_stop_order(&order_id) {
            Ok(()) => {
                activated.push(order_id.clone());
                // In production, submit to exchange here
                // For now, simulate execution
                let engine_clone = trading_engine.inner().clone();
                tokio::spawn(simulate_stop_execution(order_id, engine_clone));
            },
            Err(e) => {
                error!("Failed to activate stop order {}: {}", order_id, e);
            }
        }
    }
    
    Ok(activated)
}

/// Handle market data update from WebSocket
/// 
/// The firehose of data. Every update matters, but not every update
/// needs action. Filter, process, decide.
pub async fn handle_market_update(
    update_type: String,
    symbol: String,
    data: serde_json::Value,
    market_cache: State<'_, Arc<RwLock<MarketCache>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<()> {
    match update_type.as_str() {
        "ticker" => {
            if let Ok(ticker) = serde_json::from_value::<StateTicker>(data) {
                let mut cache = market_cache.write();
                let price = ticker.last_price;
                cache.update_ticker(symbol.clone(), ticker);
                
                // Check stop orders on price update
                drop(cache); // Release write lock
                process_price_update(symbol, price, trading_engine).await?;
            }
        },
        "orderbook" => {
            if let Ok(orderbook) = serde_json::from_value::<StateOrderBook>(data) {
                let mut cache = market_cache.write();
                cache.update_orderbook(symbol, orderbook);
            }
        },
        "trade" => {
            if let Ok(trade) = serde_json::from_value::<StateTrade>(data) {
                let mut cache = market_cache.write();
                let price = trade.price;
                cache.add_trade(symbol.clone(), trade);
                
                // Check stop orders on trade
                drop(cache); // Release write lock
                process_price_update(symbol, price, trading_engine).await?;
            }
        },
        _ => {
            warn!("Unknown market update type: {}", update_type);
        }
    }
    
    Ok(())
}

/// Simulate stop order execution
async fn simulate_stop_execution(
    order_id: String,
    trading_engine: Arc<RwLock<TradingEngine>>
) {
    // Simulate network delay to exchange
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Execute the order
    let mut engine = trading_engine.write();
    if let Some(mut order) = engine.active_orders.get_mut(&order_id) {
        let execution_price = match order.order_type {
            crate::state::OrderType::Stop | crate::state::OrderType::StopLoss => {
                // Market order - execute at current price (simplified)
                order.stop_price.unwrap_or(50000.0) * 1.001 // 0.1% slippage
            },
            crate::state::OrderType::StopLimit => {
                // Limit order - use the limit price
                order.price.unwrap_or(order.stop_price.unwrap_or(50000.0))
            },
            _ => order.price.unwrap_or(50000.0),
        };
        
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.average_fill_price = Some(execution_price);
        order.updated_at = Utc::now();
        
        info!("✅ Stop order {} executed @ ${:.2}", order_id, execution_price);
    }
}