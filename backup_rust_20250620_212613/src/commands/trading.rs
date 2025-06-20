// src-tauri/src/commands/trading.rs
// NEXLIFY TRADING ENGINE COMMANDS - Where decisions become destiny
// Last sync: 2025-06-19 | "Every trade is a roll of the dice, but we load them first"

use tauri::State;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, info, warn, error};

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
    
    // Parse order side - buying hope or selling fear?
    let side = match side.to_lowercase().as_str() {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        _ => return Err(CommandError::TradingError(
            "Invalid side. In this game, you're either buying or selling. No fence-sitting.".to_string()
        )),
    };
    
    // Parse order type - how brave are you feeling?
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
        "stop_loss" | "stop-loss" => {
            if stop_price.is_none() {
                return Err(CommandError::TradingError(
                    "Stop loss needs a trigger price. Gotta know when to cut and run.".to_string()
                ));
            }
            OrderType::StopLoss
        },
        "take_profit" | "take-profit" => {
            if price.is_none() {
                return Err(CommandError::TradingError(
                    "Take profit needs a target. Dreams without goals are just... dreams.".to_string()
                ));
            }
            OrderType::TakeProfit
        },
        _ => return Err(CommandError::TradingError(
            format!("Unknown order type: {}. Stick to the classics, ese.", order_type)
        )),
    };
    
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
    let order_id = format!("ORD-{}-{}", symbol, uuid::Uuid::new_v4());
    let order = Order {
        id: order_id.clone(),
        exchange: "coinbase".to_string(), // Would be dynamic in production
        symbol: symbol.clone(),
        side,
        order_type,
        price,
        quantity,
        status: OrderStatus::Pending,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        filled_quantity: 0.0,
        average_fill_price: None,
        metadata: metadata.unwrap_or_default(),
    };
    
    // Place the order - moment of truth
    let engine = trading_engine.write();
    engine.place_order(order.clone())?;
    
    info!("🎯 Order placed: {} - {} {} @ {:?}", 
          order_id, order.side as i32, quantity, price);
    
    // Simulate order execution for now - in production, this would hit the exchange
    tokio::spawn(simulate_order_execution(order_id.clone(), trading_engine.inner().clone()));
    
    Ok(PlaceOrderResponse {
        order_id,
        symbol,
        status: "pending".to_string(),
        message: match order.side {
            OrderSide::Buy => "Buy order fired into the matrix. May the markets be with you.".to_string(),
            OrderSide::Sell => "Sell order released. Sometimes letting go is winning.".to_string(),
        },
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
) -> CommandResult<PositionsResponse> {
    let engine = trading_engine.read();
    
    let positions: Vec<PositionInfo> = engine.positions
        .iter()
        .filter(|entry| {
            symbol.as_ref().map_or(true, |s| entry.key() == s)
        })
        .map(|entry| {
            let position = entry.value();
            let health = calculate_position_health(position);
            
            PositionInfo {
                position: position.clone(),
                health_score: health.score,
                health_status: health.status,
                time_held: format_duration(Utc::now() - position.opened_at),
                risk_metrics: calculate_position_risk(position),
            }
        })
        .collect();
    
    let total_value = positions.iter()
        .map(|p| p.position.quantity * p.position.current_price)
        .sum();
    
    let total_pnl = positions.iter()
        .map(|p| p.position.unrealized_pnl)
        .sum();
    
    info!("📊 Retrieved {} positions - Total value: ${:.2}", positions.len(), total_value);
    
    Ok(PositionsResponse {
        positions,
        total_value,
        total_pnl,
        position_count: positions.len(),
        margin_usage: calculate_margin_usage(&engine),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionsResponse {
    pub positions: Vec<PositionInfo>,
    pub total_value: f64,
    pub total_pnl: f64,
    pub position_count: usize,
    pub margin_usage: f32, // Percentage
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionInfo {
    pub position: Position,
    pub health_score: f32, // 0-100
    pub health_status: String, // "healthy", "warning", "danger", "critical"
    pub time_held: String,
    pub risk_metrics: RiskMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub var_95: f64, // Value at Risk 95%
    pub max_loss: f64,
    pub correlation_risk: f32, // 0-100
    pub liquidation_price: Option<f64>,
}

/// Get order history - learn from the past
/// 
/// Every order tells a story. The wins teach confidence, the losses teach wisdom.
/// I keep a journal of every trade - not just the numbers, but how I felt, what I saw.
/// That journal? It's worth more than any trading algorithm.
#[tauri::command]
pub async fn get_order_history(
    symbol: Option<String>,
    start_date: Option<DateTime<Utc>>,
    end_date: Option<DateTime<Utc>>,
    limit: Option<usize>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<OrderHistoryResponse> {
    let limit = limit.unwrap_or(100).min(1000);
    let start = start_date.unwrap_or(Utc::now() - chrono::Duration::days(7));
    let end = end_date.unwrap_or(Utc::now());
    
    let engine = trading_engine.read();
    let history = engine.order_history.read();
    
    let filtered_orders: Vec<OrderInfo> = history
        .iter()
        .filter(|order| {
            let in_time_range = order.created_at >= start && order.created_at <= end;
            let matches_symbol = symbol.as_ref().map_or(true, |s| &order.symbol == s);
            in_time_range && matches_symbol
        })
        .rev() // Most recent first
        .take(limit)
        .map(|order| {
            let execution_time = if matches!(order.status, OrderStatus::Filled) {
                Some(order.updated_at - order.created_at)
            } else {
                None
            };
            
            OrderInfo {
                order: order.clone(),
                execution_time_ms: execution_time.map(|d| d.num_milliseconds()),
                slippage: calculate_slippage(order),
                fees_paid: calculate_actual_fees(order),
            }
        })
        .collect();
    
    // Calculate statistics - the report card
    let stats = calculate_order_statistics(&filtered_orders);
    
    info!("📜 Retrieved {} orders from history", filtered_orders.len());
    
    Ok(OrderHistoryResponse {
        orders: filtered_orders,
        statistics: stats,
        period_start: start,
        period_end: end,
        total_count: history.len(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderHistoryResponse {
    pub orders: Vec<OrderInfo>,
    pub statistics: OrderStatistics,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderInfo {
    pub order: Order,
    pub execution_time_ms: Option<i64>,
    pub slippage: Option<f64>,
    pub fees_paid: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderStatistics {
    pub total_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub fill_rate: f32,
    pub average_execution_time_ms: f64,
    pub total_volume: f64,
    pub total_fees: f64,
}

/// Calculate PnL - the moment of truth
/// 
/// PnL is like looking in the mirror after a street fight. Sometimes you won,
/// sometimes you just survived. But those numbers? They don't lie. They're the
/// only truth in this game of shadows and chrome.
#[tauri::command]
pub async fn calculate_pnl(
    timeframe: String,
    include_fees: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PnLResponse> {
    let include_fees = include_fees.unwrap_or(true);
    
    let engine = trading_engine.read();
    let pnl_tracker = engine.pnl_tracker.read();
    
    let timeframe_pnl = match timeframe.to_lowercase().as_str() {
        "daily" | "day" | "1d" => pnl_tracker.daily_pnl,
        "weekly" | "week" | "1w" => pnl_tracker.weekly_pnl,
        "monthly" | "month" | "1m" => pnl_tracker.monthly_pnl,
        "all" | "lifetime" => pnl_tracker.all_time_pnl,
        _ => return Err(CommandError::TradingError(
            "Invalid timeframe. Time is money, but it still needs proper labels.".to_string()
        )),
    };
    
    // Calculate fees if requested
    let total_fees = if include_fees {
        calculate_period_fees(&engine, &timeframe)
    } else {
        0.0
    };
    
    let net_pnl = timeframe_pnl - total_fees;
    
    // Emotional impact assessment - because numbers affect more than wallets
    let emotional_impact = match net_pnl {
        x if x > 1000.0 => "Euphoric - don't let it go to your head".to_string(),
        x if x > 0.0 => "Positive - keep the discipline".to_string(),
        x if x > -100.0 => "Minor loss - part of the game".to_string(),
        x if x > -1000.0 => "Painful - time to review strategy".to_string(),
        _ => "Critical - consider stepping back".to_string(),
    };
    
    info!("💰 PnL calculated for {}: ${:.2} net", timeframe, net_pnl);
    
    Ok(PnLResponse {
        timeframe,
        gross_pnl: timeframe_pnl,
        fees: total_fees,
        net_pnl,
        best_trade: pnl_tracker.best_trade.clone(),
        worst_trade: pnl_tracker.worst_trade.clone(),
        win_rate: pnl_tracker.win_rate,
        sharpe_ratio: pnl_tracker.sharpe_ratio,
        max_drawdown: pnl_tracker.max_drawdown,
        emotional_impact,
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLResponse {
    pub timeframe: String,
    pub gross_pnl: f64,
    pub fees: f64,
    pub net_pnl: f64,
    pub best_trade: Option<(String, f64)>,
    pub worst_trade: Option<(String, f64)>,
    pub win_rate: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
    pub emotional_impact: String,
    pub timestamp: DateTime<Utc>,
}

// Helper functions - the tools of survival

/// Perform risk check before placing order
fn perform_risk_check(
    engine: &Arc<RwLock<TradingEngine>>,
    symbol: &str,
    side: OrderSide,
    quantity: f64,
    price: Option<f64>,
) -> Result<(), CommandError> {
    let engine = engine.read();
    let risk_params = engine.risk_params.read();
    
    // Check max position size
    let position_value = quantity * price.unwrap_or(50000.0); // Use a default for market orders
    if position_value > risk_params.max_position_size {
        return Err(CommandError::TradingError(
            format!("Position size ${:.2} exceeds max ${:.2}. Even cowboys have limits.", 
                    position_value, risk_params.max_position_size)
        ));
    }
    
    // Check daily loss limit
    let pnl = engine.pnl_tracker.read();
    if pnl.daily_pnl < -risk_params.max_daily_loss {
        return Err(CommandError::TradingError(
            format!("Daily loss limit reached: ${:.2}. Time to walk away, hermano.", 
                    risk_params.max_daily_loss)
        ));
    }
    
    // Check if symbol is allowed
    if !risk_params.allowed_symbols.contains(&symbol.to_string()) {
        return Err(CommandError::TradingError(
            format!("{} not in allowed symbols. Stick to the plan.", symbol)
        ));
    }
    
    Ok(())
}

/// Calculate position health - like a medical checkup for your trades
fn calculate_position_health(position: &Position) -> PositionHealth {
    let pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100.0;
    
    let (score, status) = match pnl_percent {
        x if x > 10.0 => (90.0, "healthy"),
        x if x > 0.0 => (70.0, "stable"),
        x if x > -5.0 => (50.0, "warning"),
        x if x > -10.0 => (30.0, "danger"),
        _ => (10.0, "critical"),
    };
    
    PositionHealth { score, status: status.to_string() }
}

struct PositionHealth {
    score: f32,
    status: String,
}

/// Calculate various fees
fn calculate_fees(quantity: f64, price: Option<f64>) -> f64 {
    let value = quantity * price.unwrap_or(50000.0);
    value * 0.006 // 0.6% taker fee
}

fn calculate_actual_fees(order: &Order) -> f64 {
    if order.filled_quantity > 0.0 {
        let value = order.filled_quantity * order.average_fill_price.unwrap_or(0.0);
        value * 0.006
    } else {
        0.0
    }
}

/// Calculate order slippage
fn calculate_slippage(order: &Order) -> Option<f64> {
    match (&order.price, &order.average_fill_price) {
        (Some(expected), Some(actual)) => Some((actual - expected).abs()),
        _ => None,
    }
}

/// Risk score calculation - how spicy is this trade?
fn calculate_risk_score(order: &Order) -> f32 {
    let mut score = 0.0;
    
    // Order type risk
    score += match order.order_type {
        OrderType::Market => 30.0, // Market orders are risky
        OrderType::Limit => 10.0,
        OrderType::StopLoss => 5.0, // Stop losses are protective
        OrderType::TakeProfit => 5.0,
        _ => 20.0,
    };
    
    // Size risk (assuming $50k position is "normal")
    let position_value = order.quantity * order.price.unwrap_or(50000.0);
    score += (position_value / 50000.0 * 20.0).min(40.0);
    
    // Volatility risk (would use real vol data in production)
    score += 20.0; // Placeholder
    
    score.min(100.0)
}

/// Format duration for human reading
fn format_duration(duration: chrono::Duration) -> String {
    if duration.num_days() > 0 {
        format!("{}d {}h", duration.num_days(), duration.num_hours() % 24)
    } else if duration.num_hours() > 0 {
        format!("{}h {}m", duration.num_hours(), duration.num_minutes() % 60)
    } else {
        format!("{}m", duration.num_minutes())
    }
}

/// Position risk calculations
fn calculate_position_risk(position: &Position) -> RiskMetrics {
    // These would use proper risk models in production
    RiskMetrics {
        var_95: position.quantity * position.current_price * 0.05, // 5% VaR
        max_loss: position.quantity * position.entry_price, // Total investment
        correlation_risk: 25.0, // Placeholder
        liquidation_price: if position.margin_used > 0.0 {
            Some(position.entry_price * 0.7) // 30% drop
        } else {
            None
        },
    }
}

/// Calculate margin usage across all positions
fn calculate_margin_usage(engine: &TradingEngine) -> f32 {
    let total_margin: f64 = engine.positions.iter()
        .map(|entry| entry.value().margin_used)
        .sum();
    
    // Assuming $10k account size - would be dynamic
    (total_margin / 10000.0 * 100.0) as f32
}

/// Calculate order statistics
fn calculate_order_statistics(orders: &[OrderInfo]) -> OrderStatistics {
    let filled = orders.iter().filter(|o| matches!(o.order.status, OrderStatus::Filled)).count();
    let cancelled = orders.iter().filter(|o| matches!(o.order.status, OrderStatus::Cancelled)).count();
    
    let avg_execution = orders.iter()
        .filter_map(|o| o.execution_time_ms)
        .collect::<Vec<_>>();
    
    let avg_execution_time = if !avg_execution.is_empty() {
        avg_execution.iter().sum::<i64>() as f64 / avg_execution.len() as f64
    } else {
        0.0
    };
    
    OrderStatistics {
        total_orders: orders.len(),
        filled_orders: filled,
        cancelled_orders: cancelled,
        fill_rate: if orders.is_empty() { 0.0 } else { (filled as f32 / orders.len() as f32) * 100.0 },
        average_execution_time_ms: avg_execution_time,
        total_volume: orders.iter().map(|o| o.order.quantity * o.order.price.unwrap_or(0.0)).sum(),
        total_fees: orders.iter().map(|o| o.fees_paid).sum(),
    }
}

/// Calculate fees for a period
fn calculate_period_fees(engine: &TradingEngine, timeframe: &str) -> f64 {
    // Simplified - would calculate based on actual timeframe
    let orders = engine.order_history.read();
    orders.iter()
        .filter(|o| matches!(o.status, OrderStatus::Filled))
        .map(|o| calculate_actual_fees(o))
        .sum()
}

/// Simulate order execution - in production this would be the exchange connector
async fn simulate_order_execution(order_id: String, engine: Arc<RwLock<TradingEngine>>) {
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    // Simulate fill
    if let Some((_, mut order)) = engine.write().active_orders.remove(&order_id) {
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.average_fill_price = order.price.or(Some(50000.0)); // Mock price
        order.updated_at = Utc::now();
        
        info!("✅ Order {} filled at {:?}", order_id, order.average_fill_price);
        
        // Update position
        // This would create/update actual positions in production
    }
}
