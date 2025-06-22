// src-tauri/src/commands/trading.rs
// NEXLIFY TRADING ENGINE COMMANDS - Where decisions become destiny
// Last sync: 2025-06-22 | "Every trade is a roll of the dice, but we load them first"

use tauri::State;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::state::{
    AppState, TradingEngine, 
    Order, OrderSide, OrderType, OrderStatus,
    Position, PositionSide
};
use super::{CommandResult, CommandError, validation};

/// Place an order - pulling the trigger on fate
/// 
/// Listen up, choom. Every order you place is a bullet fired into the dark.
/// Sometimes it hits gold, sometimes it hits you in the foot. I've done both.
/// The trick isn't avoiding mistakes - it's surviving them.
#[tauri::command]
pub async fn place_order(
    symbol: String,
    side: String,
    order_type: String,
    quantity: f64,
    price: Option<f64>,
    stop_price: Option<f64>,
    time_in_force: Option<String>,
    position_id: Option<String>, // For position-based orders
    metadata: Option<std::collections::HashMap<String, String>>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PlaceOrderResponse> {
    // Validate everything - paranoia keeps you alive in the sprawl
    validation::validate_symbol(&symbol)?;
    validation::validate_quantity(quantity)?;
    
    if let Some(p) = price {
        validation::validate_price(p)?;
    }
    if let Some(sp) = stop_price {
        validation::validate_price(sp)?;
    }
    
    // Parse order side - buying hope or selling fear?
    let side = match side.to_lowercase().as_str() {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        _ => return Err(CommandError::TradingError(
            "Invalid side. In this game, you're either buying or selling. No fence-sitting.".to_string()
        )),
    };
    
    // Parse order type - now with ALL the chrome!
    let order_type = match order_type.to_lowercase().as_str() {
        "market" => {
            if price.is_some() {
                warn!("⚠️ Market order with price? Someone's confused...");
            }
            OrderType::Market
        },
        "limit" => {
            if price.is_none() {
                return Err(CommandError::TradingError(
                    "Limit order needs a price. Can't negotiate without knowing your worth.".to_string()
                ));
            }
            OrderType::Limit
        },
        "stop" => {
            // Standalone stop order - for breakouts and bailouts
            if stop_price.is_none() {
                return Err(CommandError::TradingError(
                    "Stop order needs a trigger price. Gotta know when to pull the ripcord.".to_string()
                ));
            }
            OrderType::Stop
        },
        "stop_limit" => {
            // Standalone stop limit - controlled chaos
            if stop_price.is_none() || price.is_none() {
                return Err(CommandError::TradingError(
                    "Stop limit needs both stop and limit prices. Set your boundaries, samurai.".to_string()
                ));
            }
            OrderType::StopLimit
        },
        "stop_loss" => {
            // Position protection - your safety net
            if stop_price.is_none() {
                return Err(CommandError::TradingError(
                    "Stop loss needs a price. No protection without preparation.".to_string()
                ));
            }
            if position_id.is_none() {
                return Err(CommandError::TradingError(
                    "Stop loss must be attached to a position. Can't protect what doesn't exist.".to_string()
                ));
            }
            OrderType::StopLoss
        },
        "take_profit" => {
            // Position target - greed management
            if price.is_none() {
                return Err(CommandError::TradingError(
                    "Take profit needs a target. Dreams without goals are just... dreams.".to_string()
                ));
            }
            if position_id.is_none() {
                return Err(CommandError::TradingError(
                    "Take profit must be attached to a position. No harvest without planting.".to_string()
                ));
            }
            OrderType::TakeProfit
        },
        _ => return Err(CommandError::TradingError(
            format!("Unknown order type: {}. We've got market, limit, stop, stop_limit, stop_loss, and take_profit. Pick your poison.", order_type)
        )),
    };
    
    // Validate position-based orders have valid position
    if matches!(order_type, OrderType::StopLoss | OrderType::TakeProfit) {
        let engine = trading_engine.read();
        if let Some(pos_id) = &position_id {
            if !engine.positions.contains_key(pos_id) {
                return Err(CommandError::TradingError(
                    format!("Position {} not found. Can't protect ghosts.", pos_id)
                ));
            }
        }
    }
    
    // Check circuit breakers - sometimes the system saves you from yourself
    {
        let state = app_state.read();
        state.check_circuit_breaker(&symbol, quantity * price.unwrap_or(50000.0))?;
    }
    
    // Risk check - are we about to do something stupid?
    let risk_check = perform_risk_check(&trading_engine, &symbol, side.clone(), quantity, price);
    if let Err(e) = risk_check {
        error!("🚨 Risk check failed: {}", e);
        return Err(e);
    }
    
    // Build the order - crafting our digital bullet
    let order_id = format!("ORD-{}-{}", symbol, Uuid::new_v4());
    let order = Order {
        id: order_id.clone(),
        exchange: "nexlify".to_string(), // Would be dynamic in production
        symbol: symbol.clone(),
        side,
        order_type: order_type.clone(),
        price,
        stop_price,
        quantity,
        status: match order_type {
            // Stop orders wait in the shadows
            OrderType::Stop | OrderType::StopLimit | OrderType::StopLoss => OrderStatus::Open,
            // Others execute immediately
            _ => OrderStatus::Pending,
        },
        created_at: Utc::now(),
        updated_at: Utc::now(),
        filled_quantity: 0.0,
        average_fill_price: None,
        position_id,
        metadata: metadata.unwrap_or_default(),
    };
    
    // Place the order - moment of truth
    let engine = trading_engine.write();
    engine.place_order(order.clone())?;
    
    info!("🎯 Order placed: {} - {:?} {} {} @ {:?}/{:?}", 
          order_id, order.order_type, order.side, quantity, price, stop_price);
    
    // Simulate order execution for market/limit orders
    // Stop orders wait for their trigger
    if matches!(order.order_type, OrderType::Market | OrderType::Limit | OrderType::TakeProfit) {
        tokio::spawn(simulate_order_execution(order_id.clone(), trading_engine.inner().clone()));
    }
    
    // Build response with appropriate message
    let message = match order.order_type {
        OrderType::Market => "Market order fired. May the spreads be ever in your favor.".to_string(),
        OrderType::Limit => "Limit order set. Patience is a virtue, FOMO is a vice.".to_string(),
        OrderType::Stop => "Stop order armed. Ready to strike when the moment comes.".to_string(),
        OrderType::StopLimit => "Stop limit locked and loaded. Precision over speed.".to_string(),
        OrderType::StopLoss => "Stop loss activated. Your safety net is in place.".to_string(),
        OrderType::TakeProfit => "Take profit set. Greed is good, but profits are better.".to_string(),
        _ => "Order placed. Let's see what the market gods decide.".to_string(),
    };
    
    Ok(PlaceOrderResponse {
        order_id,
        symbol,
        status: order.status.to_string(),
        message,
        estimated_fees: calculate_fees(quantity, price),
        risk_score: calculate_risk_score(&order),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PlaceOrderResponse {
    pub order_id: String,
    pub symbol: String,
    pub status: String,
    pub message: String,
    pub estimated_fees: f64,
    pub risk_score: f32, // 0-100, higher = riskier
    pub timestamp: DateTime<Utc>,
}

/// Cancel an order - knowing when to walk away
/// 
/// Mira, canceling an order isn't weakness. It's wisdom. I've seen too many
/// traders ride their pride straight to zero. The best traders? They cancel
/// more orders than they fill. Each cancel is a bullet dodged.
#[tauri::command]
pub async fn cancel_order(
    order_id: String,
    reason: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<CancelOrderResponse> {
    if order_id.is_empty() {
        return Err(CommandError::TradingError(
            "Order ID required. Can't cancel what we can't find.".to_string()
        ));
    }
    
    let reason = reason.unwrap_or_else(|| "User requested".to_string());
    
    info!("🚫 Canceling order {} - Reason: {}", order_id, reason);
    
    let engine = trading_engine.write();
    engine.cancel_order(&order_id)?;
    
    Ok(CancelOrderResponse {
        order_id: order_id.clone(),
        status: "cancelled".to_string(),
        message: format!("Order {} cancelled. Sometimes the best trade is the one you don't make.", order_id),
        reason,
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CancelOrderResponse {
    pub order_id: String,
    pub status: String,
    pub message: String,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Get current positions - know where you stand
/// 
/// Your positions are your soldiers in the field. You gotta know where each one is,
/// how they're doing, who's bleeding out. I check mine every hour - paranoid? Maybe.
/// Alive? Definitely.
#[tauri::command]
pub async fn get_positions(
    symbol: Option<String>,
    include_closed: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<Vec<PositionInfo>> {
    let engine = trading_engine.read();
    let include_closed = include_closed.unwrap_or(false);
    
    let positions: Vec<PositionInfo> = engine.positions
        .iter()
        .filter(|entry| {
            symbol.as_ref().map_or(true, |s| entry.key() == s)
        })
        .map(|entry| {
            let position = entry.value();
            let health = calculate_position_health(position);
            
            PositionInfo {
                id: entry.key().clone(),
                symbol: position.symbol.clone(),
                side: format!("{:?}", position.side),
                quantity: position.quantity,
                entry_price: position.entry_price,
                current_price: position.current_price,
                unrealized_pnl: position.unrealized_pnl,
                realized_pnl: position.realized_pnl,
                margin_used: position.margin_used,
                health_score: health,
                opened_at: position.opened_at,
                last_updated: position.last_updated,
            }
        })
        .collect();
    
    info!("📊 Retrieved {} positions", positions.len());
    Ok(positions)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionInfo {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub margin_used: f64,
    pub health_score: f32, // 0-100, position health
    pub opened_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// Close a position - sometimes you gotta cut and run
/// 
/// Closing a position is like ending a relationship. Sometimes it hurts,
/// sometimes it's relief, but it's always necessary. The market doesn't
/// care about your attachment to a trade.
#[tauri::command]
pub async fn close_position(
    position_id: String,
    percentage: Option<f64>,
    reason: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<ClosePositionResponse> {
    let percentage = percentage.unwrap_or(100.0);
    if percentage <= 0.0 || percentage > 100.0 {
        return Err(CommandError::TradingError(
            "Percentage must be between 0 and 100. Can't close what doesn't exist.".to_string()
        ));
    }
    
    let engine = trading_engine.read();
    let position = engine.positions.get(&position_id)
        .ok_or_else(|| CommandError::TradingError(
            format!("Position {} not found. Already gone, choom?", position_id)
        ))?;
    
    let close_quantity = position.quantity * (percentage / 100.0);
    let side = match position.side {
        PositionSide::Long => OrderSide::Sell,
        PositionSide::Short => OrderSide::Buy,
    };
    
    info!("💔 Closing {}% of position {} - {}", percentage, position_id, 
          reason.as_ref().unwrap_or(&"No reason given".to_string()));
    
    // Place market order to close
    drop(engine); // Release read lock
    
    // Create closing order
    let order_id = format!("CLOSE-{}-{}", position_id, Uuid::new_v4());
    let mut engine = trading_engine.write();
    
    let close_order = Order {
        id: order_id.clone(),
        exchange: position.exchange.clone(),
        symbol: position.symbol.clone(),
        side,
        order_type: OrderType::Market,
        price: None,
        stop_price: None,
        quantity: close_quantity,
        status: OrderStatus::Pending,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        filled_quantity: 0.0,
        average_fill_price: None,
        position_id: Some(position_id.clone()),
        metadata: {
            let mut meta = std::collections::HashMap::new();
            meta.insert("close_reason".to_string(), reason.unwrap_or_else(|| "manual".to_string()));
            meta.insert("close_percentage".to_string(), percentage.to_string());
            meta
        },
    };
    
    engine.place_order(close_order)?;
    
    Ok(ClosePositionResponse {
        position_id,
        order_id,
        quantity_closed: close_quantity,
        percentage_closed: percentage,
        message: if percentage >= 100.0 {
            "Position closed. On to the next battle.".to_string()
        } else {
            format!("Partial close executed. {}% still riding.", 100.0 - percentage)
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClosePositionResponse {
    pub position_id: String,
    pub order_id: String,
    pub quantity_closed: f64,
    pub percentage_closed: f64,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Get P&L report - the scoreboard of war
/// 
/// P&L is truth. Everything else is noise. Green days feel like chrome,
/// red days feel like rust. But remember - it's not about winning every
/// battle, it's about winning the war.
#[tauri::command]
pub async fn get_pnl_report(
    period: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PnLReport> {
    let engine = trading_engine.read();
    let pnl = engine.pnl_tracker.read();
    
    let period = period.unwrap_or_else(|| "daily".to_string());
    
    let (period_pnl, period_label) = match period.as_str() {
        "daily" => (pnl.daily_pnl, "Today"),
        "weekly" => (pnl.weekly_pnl, "This Week"),
        "monthly" => (pnl.monthly_pnl, "This Month"),
        "all" => (pnl.all_time_pnl, "All Time"),
        _ => (pnl.daily_pnl, "Today"),
    };
    
    Ok(PnLReport {
        period: period_label.to_string(),
        pnl: period_pnl,
        win_rate: pnl.win_rate,
        sharpe_ratio: pnl.sharpe_ratio,
        max_drawdown: pnl.max_drawdown,
        best_trade: pnl.best_trade.clone(),
        worst_trade: pnl.worst_trade.clone(),
        message: match period_pnl {
            p if p > 1000.0 => "Crushing it! The chrome shines bright today. 🚀".to_string(),
            p if p > 0.0 => "Green is good. Keep the discipline, avoid the greed. 📈".to_string(),
            p if p > -500.0 => "Red day, but still breathing. Tomorrow's another fight. 💪".to_string(),
            _ => "Rough waters. Remember: preservation over profit. 🛡️".to_string(),
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLReport {
    pub period: String,
    pub pnl: f64,
    pub win_rate: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
    pub best_trade: Option<(String, f64)>,
    pub worst_trade: Option<(String, f64)>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// === HELPER FUNCTIONS ===

/// Perform risk check before placing order
fn perform_risk_check(
    engine: &State<'_, Arc<RwLock<TradingEngine>>>,
    symbol: &str,
    side: OrderSide,
    quantity: f64,
    price: Option<f64>,
) -> CommandResult<()> {
    let engine = engine.read();
    let risk_params = engine.risk_params.read();
    
    // Check position size limits
    let position_value = quantity * price.unwrap_or(50000.0);
    if position_value > risk_params.max_position_size {
        return Err(CommandError::TradingError(
            format!("Position size ${:.2} exceeds limit ${:.2}. Even chrome has limits.", 
                    position_value, risk_params.max_position_size)
        ));
    }
    
    // Check daily loss limit
    let pnl = engine.pnl_tracker.read();
    if pnl.daily_pnl < -risk_params.max_daily_loss {
        return Err(CommandError::TradingError(
            "Daily loss limit reached. The market has spoken - fight another day.".to_string()
        ));
    }
    
    // Check symbol restrictions
    if risk_params.banned_symbols.contains(&symbol.to_string()) {
        return Err(CommandError::TradingError(
            format!("Symbol {} is restricted. Some battles aren't worth fighting.", symbol)
        ));
    }
    
    Ok(())
}

/// Calculate position health score
fn calculate_position_health(position: &Position) -> f32 {
    let pnl_percent = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100.0;
    let time_held = Utc::now().signed_duration_since(position.opened_at).num_hours() as f32;
    
    // Health based on P&L and time
    match pnl_percent {
        p if p > 5.0 => 100.0,
        p if p > 0.0 => 80.0 + (p * 4.0),
        p if p > -2.0 => 60.0 + (p * 10.0),
        p if p > -5.0 => 40.0 + (p * 4.0),
        _ => (20.0 + pnl_percent).max(0.0),
    }
}

/// Calculate estimated fees
fn calculate_fees(quantity: f64, price: Option<f64>) -> f64 {
    let value = quantity * price.unwrap_or(50000.0);
    value * 0.001 // 0.1% taker fee
}

/// Calculate risk score for an order
fn calculate_risk_score(order: &Order) -> f32 {
    let base_score = match order.order_type {
        OrderType::Market => 30.0,
        OrderType::Limit => 10.0,
        OrderType::Stop => 40.0,
        OrderType::StopLimit => 35.0,
        OrderType::StopLoss => 5.0,
        OrderType::TakeProfit => 5.0,
        _ => 50.0,
    };
    
    // Adjust for order size (simplified)
    let size_multiplier = (order.quantity * order.price.unwrap_or(50000.0) / 10000.0).min(2.0);
    
    (base_score * size_multiplier).min(100.0)
}

/// Simulate order execution (for development)
async fn simulate_order_execution(
    order_id: String,
    trading_engine: Arc<RwLock<TradingEngine>>,
) {
    // Simulate network delay
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let mut engine = trading_engine.write();
    if let Some(mut order) = engine.active_orders.get_mut(&order_id) {
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.average_fill_price = Some(order.price.unwrap_or(50000.0));
        order.updated_at = Utc::now();
        
        info!("✅ Order {} filled @ {}", order_id, order.average_fill_price.unwrap());
    }
}