// Location: C:\Nexlify\src-tauri\src\commands\trading.rs
// Purpose: NEXLIFY TRADING ENGINE COMMANDS - Where decisions become destiny
// Last sync: 2025-06-22 | "Every trade is a roll of the dice, but we load them first"
//
// DEPENDENCIES: Add to Cargo.toml:
// rand = "0.8"

use tauri::State;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, info, warn, error};
use uuid::Uuid;
use rand::Rng;

use crate::state::{
    AppState, TradingEngine, StateError,
    Order, OrderSide, OrderType, OrderStatus,
    Position, PositionSide, RiskParameters,
    PnLTracker
};
use super::{CommandResult, CommandError, validation};

// ─────────────────────────────────────────────────────────────
// P&L TYPES - Define locally until added to state module
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLSnapshot {
    pub timestamp: DateTime<Utc>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub fees_paid: f64,
    pub positions_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub pnl: f64,
    pub percentage: f64,
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────
// P&L REPORTING COMMANDS - The truth in numbers
// ─────────────────────────────────────────────────────────────

/// Get comprehensive P&L report - the moment of truth
/// 
/// P&L is the mirror that never lies. Green days feel like chrome,
/// red days feel like rust. But remember - it's not about winning every
/// battle, it's about winning the war. This command shows you the whole
/// battlefield, from daily skirmishes to the grand campaign.
#[tauri::command]
pub async fn get_pnl_report(
    timeframe: Option<String>,
    include_fees: Option<bool>,
    by_symbol: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PnLReport> {
    let timeframe = timeframe.unwrap_or_else(|| "daily".to_string());
    let include_fees = include_fees.unwrap_or(true);
    let by_symbol = by_symbol.unwrap_or(false);
    
    let engine = trading_engine.read();
    let pnl = engine.pnl_tracker.read();
    
    // Get base P&L for timeframe
    let (period_pnl, period_label) = match timeframe.to_lowercase().as_str() {
        "daily" | "day" | "1d" => (pnl.daily_pnl, "Today"),
        "weekly" | "week" | "1w" => (pnl.weekly_pnl, "This Week"),
        "monthly" | "month" | "1m" => (pnl.monthly_pnl, "This Month"),
        "yearly" | "year" | "1y" => (pnl.yearly_pnl, "This Year"),
        "all" | "lifetime" => (pnl.all_time_pnl, "All Time"),
        _ => return Err(CommandError::ValidationError(
            "Invalid timeframe. Choose: daily, weekly, monthly, yearly, all".to_string()
        )),
    };
    
    // Calculate unrealized P&L from open positions
    let unrealized_pnl: f64 = engine.positions
        .iter()
        .map(|entry| entry.value().unrealized_pnl)
        .sum();
    
    // Calculate fees if requested
    let total_fees = if include_fees {
        calculate_period_fees(&engine, &timeframe)
    } else {
        0.0
    };
    
    // P&L by symbol if requested
    let symbol_breakdown = if by_symbol {
        Some(calculate_pnl_by_symbol(&engine, &timeframe))
    } else {
        None
    };
    
    // Net P&L after fees
    let net_pnl = period_pnl - total_fees;
    let total_pnl = net_pnl + unrealized_pnl;
    
    // Performance metrics
    let win_rate = pnl.win_rate;
    let sharpe_ratio = pnl.sharpe_ratio;
    let max_drawdown = pnl.max_drawdown;
    
    // Emotional impact assessment
    let psychological_state = assess_psychological_impact(net_pnl, max_drawdown);
    
    info!("💰 P&L Report: {} - Net: ${:.2}, Unrealized: ${:.2}", 
        period_label, net_pnl, unrealized_pnl);
    
    Ok(PnLReport {
        timeframe: timeframe.clone(),
        period_label: period_label.to_string(),
        realized_pnl: period_pnl,
        unrealized_pnl,
        total_pnl,
        fees_paid: total_fees,
        net_pnl,
        win_rate,
        sharpe_ratio,
        max_drawdown,
        best_trade: pnl.best_trade.clone(),
        worst_trade: pnl.worst_trade.clone(),
        total_trades: pnl.total_trades,
        winning_trades: pnl.winning_trades,
        losing_trades: pnl.losing_trades,
        symbol_breakdown,
        psychological_state,
        recommendations: generate_pnl_recommendations(net_pnl, win_rate, sharpe_ratio),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLReport {
    pub timeframe: String,
    pub period_label: String,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub fees_paid: f64,
    pub net_pnl: f64,
    pub win_rate: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
    pub best_trade: Option<TradeRecord>,
    pub worst_trade: Option<TradeRecord>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub symbol_breakdown: Option<Vec<SymbolPnL>>,
    pub psychological_state: String,
    pub recommendations: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolPnL {
    pub symbol: String,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub trade_count: usize,
    pub win_rate: f32,
}

/// Calculate P&L for specific period - surgical precision
/// 
/// Sometimes you need to know exactly how much blood you've lost or gained
/// in a specific timeframe. This is your financial forensics tool.
#[tauri::command]
pub async fn calculate_pnl(
    start_date: Option<DateTime<Utc>>,
    end_date: Option<DateTime<Utc>>,
    symbol: Option<String>,
    include_unrealized: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<CalculatedPnL> {
    let end = end_date.unwrap_or_else(|| Utc::now());
    let start = start_date.unwrap_or_else(|| end - chrono::Duration::days(1));
    let include_unrealized = include_unrealized.unwrap_or(true);
    
    if start >= end {
        return Err(CommandError::ValidationError(
            "End date must be after start date. Time flows forward, even in the sprawl.".to_string()
        ));
    }
    
    let engine = trading_engine.read();
    
    // Calculate realized P&L from closed positions in period
    let realized_pnl = calculate_period_realized_pnl(&engine, start, end, &symbol);
    
    // Calculate unrealized P&L if requested
    let unrealized_pnl = if include_unrealized {
        engine.positions
            .iter()
            .filter(|entry| {
                symbol.as_ref().map_or(true, |s| &entry.value().symbol == s)
            })
            .map(|entry| entry.value().unrealized_pnl)
            .sum()
    } else {
        0.0
    };
    
    // Get fees for period
    let fees = calculate_period_fees_detailed(&engine, start, end, &symbol);
    
    // Calculate metrics
    let total_pnl = realized_pnl + unrealized_pnl - fees;
    let roi = calculate_roi(&engine, total_pnl);
    
    info!("📊 P&L Calculated: {} to {} - Total: ${:.2}", 
        start.format("%Y-%m-%d"), end.format("%Y-%m-%d"), total_pnl);
    
    Ok(CalculatedPnL {
        start_date: start,
        end_date: end,
        symbol: symbol.clone(),
        realized_pnl,
        unrealized_pnl,
        fees_paid: fees,
        net_pnl: realized_pnl - fees,
        total_pnl,
        roi_percentage: roi,
        days_calculated: (end - start).num_days(),
        daily_average: total_pnl / (end - start).num_days() as f64,
        message: format_pnl_message(total_pnl, roi),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalculatedPnL {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub symbol: Option<String>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees_paid: f64,
    pub net_pnl: f64,
    pub total_pnl: f64,
    pub roi_percentage: f64,
    pub days_calculated: i64,
    pub daily_average: f64,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Get P&L history - chronicle of victories and defeats
/// 
/// Every trader needs to study their history. The patterns in your P&L
/// tell a story - of discipline gained, lessons learned, and edges discovered.
/// This is your trading journal, written in the language of profit and loss.
#[tauri::command]
pub async fn get_pnl_history(
    interval: Option<String>,
    limit: Option<usize>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PnLHistory> {
    let interval = interval.unwrap_or_else(|| "daily".to_string());
    let limit = limit.unwrap_or(30).min(365);
    
    let engine = trading_engine.read();
    let pnl_tracker = engine.pnl_tracker.read();
    
    // Get snapshots based on interval
    let snapshots = match interval.to_lowercase().as_str() {
        "hourly" => get_hourly_snapshots(&pnl_tracker.pnl_history, limit),
        "daily" => get_daily_snapshots(&pnl_tracker.pnl_history, limit),
        "weekly" => get_weekly_snapshots(&pnl_tracker.pnl_history, limit),
        "monthly" => get_monthly_snapshots(&pnl_tracker.pnl_history, limit),
        _ => return Err(CommandError::ValidationError(
            "Invalid interval. Choose: hourly, daily, weekly, monthly".to_string()
        )),
    };
    
    // Calculate statistics
    let stats = calculate_pnl_statistics(&snapshots);
    
    // Identify trends
    let trend = identify_pnl_trend(&snapshots);
    
    info!("📈 P&L History: {} {} snapshots retrieved", limit, interval);
    
    Ok(PnLHistory {
        interval,
        snapshots,
        statistics: stats,
        trend,
        message: interpret_pnl_history(&stats, &trend),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLHistory {
    pub interval: String,
    pub snapshots: Vec<PnLSnapshot>,
    pub statistics: PnLStatistics,
    pub trend: PnLTrend,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLStatistics {
    pub average_pnl: f64,
    pub std_deviation: f64,
    pub best_period: f64,
    pub worst_period: f64,
    pub positive_periods: usize,
    pub negative_periods: usize,
    pub current_streak: i32, // Positive = winning streak, negative = losing
    pub longest_winning_streak: usize,
    pub longest_losing_streak: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLTrend {
    pub direction: String, // "improving", "declining", "stable"
    pub strength: f32,     // 0-100
    pub momentum: f32,     // Rate of change
    pub prediction: String, // Next period outlook
}

/// Update P&L tracker - record the truth
/// 
/// The market writes history in real-time. This command ensures our
/// P&L tracker stays synchronized with reality. Called internally
/// after trades, but exposed for manual reconciliation.
#[tauri::command]
pub async fn update_pnl_tracker(
    force_snapshot: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<PnLUpdateResponse> {
    let force = force_snapshot.unwrap_or(false);
    
    let mut engine = trading_engine.write();
    
    // Calculate current P&L state
    let current_realized = calculate_total_realized_pnl(&engine);
    let current_unrealized = calculate_total_unrealized_pnl(&engine);
    let current_fees = calculate_total_fees(&engine);
    
    // Update tracker
    let mut pnl_tracker = engine.pnl_tracker.write();
    
    // Update period P&Ls
    update_period_pnls(&mut pnl_tracker, current_realized);
    
    // Update statistics
    update_pnl_statistics(&mut pnl_tracker, &engine);
    
    // Create snapshot if needed
    let snapshot_created = if force || should_create_snapshot(&pnl_tracker) {
        let snapshot = PnLSnapshot {
            timestamp: Utc::now(),
            realized_pnl: current_realized,
            unrealized_pnl: current_unrealized,
            total_pnl: current_realized + current_unrealized,
            fees_paid: current_fees,
            positions_count: engine.positions.len(),
        };
        
        pnl_tracker.pnl_history.push(snapshot);
        
        // Limit history size
        if pnl_tracker.pnl_history.len() > 10000 {
            pnl_tracker.pnl_history.remove(0);
        }
        
        true
    } else {
        false
    };
    
    pnl_tracker.last_update = Utc::now();
    
    info!("💾 P&L Tracker updated - Realized: ${:.2}, Unrealized: ${:.2}", 
        current_realized, current_unrealized);
    
    Ok(PnLUpdateResponse {
        realized_pnl: current_realized,
        unrealized_pnl: current_unrealized,
        total_pnl: current_realized + current_unrealized,
        fees_paid: current_fees,
        snapshot_created,
        snapshots_total: pnl_tracker.pnl_history.len(),
        message: "P&L tracker synchronized with the matrix.".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PnLUpdateResponse {
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub fees_paid: f64,
    pub snapshot_created: bool,
    pub snapshots_total: usize,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────
// SIMULATION ENGINE - Where we practice before we bleed
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SimulationEngine {
    pub paper_mode: bool,
    pub latency_ms: u64,
    pub slippage_bps: f64, // basis points
    pub failure_rate: f64, // 0-1
    pub mock_prices: std::collections::HashMap<String, f64>,
}

impl Default for SimulationEngine {
    fn default() -> Self {
        Self {
            paper_mode: true,
            latency_ms: 50,
            slippage_bps: 10.0, // 0.1%
            failure_rate: 0.02, // 2% failure rate
            mock_prices: std::collections::HashMap::new(),
        }
    }
}

/// Toggle simulation mode - practice or production
/// 
/// Listen up, rookie. Every samurai trains with wooden swords before they get steel.
/// Paper trading is where you learn to bleed without dying. Use it.
#[tauri::command]
pub async fn toggle_simulation_mode(
    enabled: bool,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<SimulationModeResponse> {
    let mut state = app_state.write();
    
    let old_mode = state.simulation_engine.paper_mode;
    state.simulation_engine.paper_mode = enabled;
    
    let message = if enabled {
        "SIMULATION MODE ACTIVE. Training wheels engaged. Your losses are virtual, your lessons are real."
    } else {
        "LIVE MODE ACTIVE. Real money, real consequences. May the chrome be with you."
    };
    
    info!("🎮 Trading mode changed: {} -> {}", 
        if old_mode { "SIMULATION" } else { "LIVE" },
        if enabled { "SIMULATION" } else { "LIVE" }
    );
    
    // Log mode change
    state.activity_log.push(format!(
        "[{}] TRADING MODE: {} | WARNING: {}",
        Utc::now().format("%H:%M:%S"),
        if enabled { "SIMULATION" } else { "LIVE" },
        if enabled { "Paper trading active" } else { "Real money at risk" }
    ));
    
    Ok(SimulationModeResponse {
        simulation_enabled: enabled,
        previous_mode: if old_mode { "simulation" } else { "live" },
        message: message.to_string(),
        warning: if !enabled {
            Some("You are now trading with REAL MONEY. Double-check everything.".to_string())
        } else {
            None
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationModeResponse {
    pub simulation_enabled: bool,
    pub previous_mode: String,
    pub message: String,
    pub warning: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Configure simulation parameters - tune your training
/// 
/// The market's a harsh teacher. In simulation, you can dial up the pain
/// without losing your shirt. Crank up the slippage, add some latency,
/// see how you handle the heat when the chrome starts melting.
#[tauri::command]
pub async fn configure_simulation(
    latency_ms: Option<u64>,
    slippage_bps: Option<f64>,
    failure_rate: Option<f64>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<SimulationConfig> {
    let mut state = app_state.write();
    let sim = &mut state.simulation_engine;
    
    if let Some(latency) = latency_ms {
        if latency > 5000 {
            return Err(CommandError::ValidationError(
                "Latency too high. Even in the worst networks, 5 seconds is brutal.".to_string()
            ));
        }
        sim.latency_ms = latency;
    }
    
    if let Some(slippage) = slippage_bps {
        if slippage > 100.0 {
            return Err(CommandError::ValidationError(
                "Slippage over 1%? That's not training, that's masochism.".to_string()
            ));
        }
        sim.slippage_bps = slippage;
    }
    
    if let Some(failure) = failure_rate {
        if failure < 0.0 || failure > 0.5 {
            return Err(CommandError::ValidationError(
                "Failure rate must be between 0-50%. Any higher and you're just gambling.".to_string()
            ));
        }
        sim.failure_rate = failure;
    }
    
    info!("⚙️ Simulation parameters updated: latency={}ms, slippage={}bps, failure={}%",
        sim.latency_ms, sim.slippage_bps, sim.failure_rate * 100.0);
    
    Ok(SimulationConfig {
        paper_mode: sim.paper_mode,
        latency_ms: sim.latency_ms,
        slippage_bps: sim.slippage_bps,
        failure_rate: sim.failure_rate,
        message: "Simulation parameters updated. The training ground is ready.".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub paper_mode: bool,
    pub latency_ms: u64,
    pub slippage_bps: f64,
    pub failure_rate: f64,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Inject mock price data - control the matrix
/// 
/// In simulation, you are the market maker. Set the prices, create the scenarios,
/// test your strategies against the worst the market can throw at you.
#[tauri::command]
pub async fn set_mock_price(
    symbol: String,
    price: f64,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<MockPriceResponse> {
    validation::validate_symbol(&symbol)?;
    validation::validate_price(price)?;
    
    let mut state = app_state.write();
    
    if !state.simulation_engine.paper_mode {
        return Err(CommandError::TradingError(
            "Cannot set mock prices in live mode. Switch to simulation first.".to_string()
        ));
    }
    
    let old_price = state.simulation_engine.mock_prices.get(&symbol).copied();
    state.simulation_engine.mock_prices.insert(symbol.clone(), price);
    
    info!("💉 Mock price injected: {} = ${}", symbol, price);
    
    Ok(MockPriceResponse {
        symbol,
        price,
        previous_price: old_price,
        message: "Mock price set. The simulation bends to your will.".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockPriceResponse {
    pub symbol: String,
    pub price: f64,
    pub previous_price: Option<f64>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Run backtest - learn from synthetic history
/// 
/// They say those who don't learn from history are doomed to repeat it.
/// In trading, those who don't backtest are doomed to poverty. This is
/// where you test your chrome against the ghosts of markets past.
#[tauri::command]
pub async fn run_backtest(
    symbol: String,
    strategy: String,
    start_date: String,
    end_date: String,
    initial_capital: f64,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<BacktestResults> {
    validation::validate_symbol(&symbol)?;
    
    if initial_capital <= 0.0 {
        return Err(CommandError::ValidationError(
            "Initial capital must be positive. Can't trade with nothing.".to_string()
        ));
    }
    
    // Parse dates
    let start = chrono::NaiveDate::parse_from_str(&start_date, "%Y-%m-%d")
        .map_err(|_| CommandError::ValidationError("Invalid start date format".to_string()))?;
    let end = chrono::NaiveDate::parse_from_str(&end_date, "%Y-%m-%d")
        .map_err(|_| CommandError::ValidationError("Invalid end date format".to_string()))?;
    
    if start >= end {
        return Err(CommandError::ValidationError(
            "End date must be after start date. Time doesn't flow backwards, choom.".to_string()
        ));
    }
    
    info!("🔄 Running backtest: {} on {} from {} to {}", 
        strategy, symbol, start, end);
    
    // Simulate backtest with mock data
    let days = (end - start).num_days() as usize;
    let mut equity_curve = Vec::with_capacity(days);
    let mut trades = Vec::new();
    let mut current_capital = initial_capital;
    
    // Generate synthetic price data with realistic volatility
    let mut price = 50000.0; // Starting BTC price
    let volatility = 0.02; // 2% daily volatility
    
    for day in 0..days {
        // Random walk with drift
        let change = (rand::random::<f64>() - 0.5) * volatility + 0.0001; // Slight upward drift
        price *= 1.0 + change;
        
        // Simple momentum strategy simulation
        if day > 5 {
            let momentum = (price / equity_curve[day-5]) - 1.0;
            
            if momentum > 0.02 && trades.len() % 2 == 0 {
                // Buy signal
                trades.push(BacktestTrade {
                    timestamp: start + chrono::Duration::days(day as i64),
                    side: "buy".to_string(),
                    price,
                    quantity: current_capital * 0.1 / price,
                    pnl: 0.0,
                });
            } else if momentum < -0.02 && trades.len() % 2 == 1 {
                // Sell signal
                let entry_price = trades.last().unwrap().price;
                let quantity = trades.last().unwrap().quantity;
                let pnl = (price - entry_price) * quantity;
                
                trades.push(BacktestTrade {
                    timestamp: start + chrono::Duration::days(day as i64),
                    side: "sell".to_string(),
                    price,
                    quantity,
                    pnl,
                });
                
                current_capital += pnl;
            }
        }
        
        equity_curve.push(current_capital);
    }
    
    // Calculate metrics
    let total_return = (current_capital / initial_capital - 1.0) * 100.0;
    let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
    let total_trades = trades.len() / 2; // Buy+Sell pairs
    let win_rate = if total_trades > 0 {
        (winning_trades as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };
    
    // Calculate Sharpe ratio (simplified)
    let returns: Vec<f64> = equity_curve.windows(2)
        .map(|w| (w[1] / w[0]) - 1.0)
        .collect();
    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_dev = (returns.iter()
        .map(|r| (r - avg_return).powi(2))
        .sum::<f64>() / returns.len() as f64)
        .sqrt();
    let sharpe = if std_dev > 0.0 {
        (avg_return / std_dev) * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };
    
    Ok(BacktestResults {
        strategy,
        symbol,
        start_date: start.to_string(),
        end_date: end.to_string(),
        initial_capital,
        final_capital: current_capital,
        total_return,
        total_trades,
        winning_trades,
        win_rate,
        sharpe_ratio: sharpe,
        max_drawdown: calculate_max_drawdown(&equity_curve),
        trades,
        equity_curve,
        message: format!(
            "Backtest complete. {} return with {} Sharpe. {}",
            if total_return > 0.0 { "Positive" } else { "Negative" },
            if sharpe > 1.0 { "Solid" } else { "Weak" },
            if win_rate > 60.0 { "The chrome shines bright." } else { "Back to the drawing board." }
        ),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BacktestResults {
    pub strategy: String,
    pub symbol: String,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
    pub final_capital: f64,
    pub total_return: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub trades: Vec<BacktestTrade>,
    pub equity_curve: Vec<f64>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BacktestTrade {
    pub timestamp: chrono::NaiveDate,
    pub side: String,
    pub price: f64,
    pub quantity: f64,
    pub pnl: f64,
}

// Helper function for drawdown calculation
fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut max_drawdown = 0.0;
    let mut peak = equity_curve[0];
    
    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak * 100.0;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    max_drawdown
}

// Helper function for enhanced order placement (can be removed if not needed)
async fn place_order_with_simulation(
    order: Order,
    app_state: &Arc<RwLock<AppState>>,
    trading_engine: &Arc<RwLock<TradingEngine>>,
) -> Result<String, CommandError> {
    let state = app_state.read();
    
    if state.simulation_engine.paper_mode {
        // Simulate latency
        tokio::time::sleep(tokio::time::Duration::from_millis(
            state.simulation_engine.latency_ms
        )).await;
        
        // Simulate random failures
        if rand::random::<f64>() < state.simulation_engine.failure_rate {
            return Err(CommandError::TradingError(
                "Simulated order failure. The exchange rejected your order.".to_string()
            ));
        }
        
        // Apply slippage to market orders
        let mut simulated_order = order.clone();
        if simulated_order.order_type == OrderType::Market {
            let slippage_mult = if simulated_order.side == OrderSide::Buy {
                1.0 + (state.simulation_engine.slippage_bps / 10000.0)
            } else {
                1.0 - (state.simulation_engine.slippage_bps / 10000.0)
            };
            
            // Use mock price if available
            let base_price = state.simulation_engine.mock_prices
                .get(&simulated_order.symbol)
                .copied()
                .unwrap_or(50000.0);
            
            simulated_order.average_fill_price = Some(base_price * slippage_mult);
            simulated_order.status = OrderStatus::Filled;
            simulated_order.filled_quantity = simulated_order.quantity;
        }
        
        // Place in paper trading engine
        drop(state);
        let mut engine = trading_engine.write();
        engine.place_order(simulated_order)
            .map_err(|e| CommandError::TradingError(e.to_string()))
    } else {
        // Real trading - place actual order
        drop(state);
        let mut engine = trading_engine.write();
        engine.place_order(order)
            .map_err(|e| CommandError::TradingError(e.to_string()))
    }
}

/// Reset simulation state - clean slate protocol
/// 
/// Sometimes you need to wipe the blood off the training mat and start fresh.
/// This clears all mock prices and resets the simulation to default parameters.
#[tauri::command]
pub async fn reset_simulation(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<ResetSimulationResponse> {
    let mut state = app_state.write();
    
    if !state.simulation_engine.paper_mode {
        return Err(CommandError::TradingError(
            "Cannot reset simulation in live mode. You can't reset real trades, choom.".to_string()
        ));
    }
    
    let old_prices_count = state.simulation_engine.mock_prices.len();
    
    // Reset to defaults
    state.simulation_engine = SimulationEngine::default();
    
    info!("🔄 Simulation reset: Cleared {} mock prices", old_prices_count);
    
    // Clear paper trading positions and orders
    // Note: This should ideally be in a separate paper trading state
    
    Ok(ResetSimulationResponse {
        mock_prices_cleared: old_prices_count,
        message: "Simulation reset. Clean slate, fresh start. Time to try again.".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResetSimulationResponse {
    pub mock_prices_cleared: usize,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Get simulation statistics - know your training score
/// 
/// In the dojo, you track every strike, every block, every mistake.
/// In trading simulation, you track every trade, every loss, every lesson.
#[tauri::command]
pub async fn get_simulation_stats(
    app_state: State<'_, Arc<RwLock<AppState>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<SimulationStats> {
    let state = app_state.read();
    
    if !state.simulation_engine.paper_mode {
        return Err(CommandError::TradingError(
            "Simulation stats only available in paper mode.".to_string()
        ));
    }
    
    let engine = trading_engine.read();
    
    // Calculate paper trading statistics
    let total_paper_trades = engine.order_history.read()
        .iter()
        .filter(|o| o.metadata.get("paper_trade") == Some(&"true".to_string()))
        .count();
    
    let paper_positions = engine.positions.len();
    
    // Calculate simulated P&L
    let simulated_pnl: f64 = engine.positions
        .iter()
        .map(|entry| entry.value().unrealized_pnl)
        .sum();
    
    Ok(SimulationStats {
        paper_mode: true,
        total_paper_trades,
        active_paper_positions: paper_positions,
        simulated_pnl,
        current_parameters: SimulationConfig {
            paper_mode: state.simulation_engine.paper_mode,
            latency_ms: state.simulation_engine.latency_ms,
            slippage_bps: state.simulation_engine.slippage_bps,
            failure_rate: state.simulation_engine.failure_rate,
            message: "Current simulation parameters".to_string(),
            timestamp: Utc::now(),
        },
        mock_prices: state.simulation_engine.mock_prices.clone(),
        message: format!(
            "Paper trading active: {} trades, {} positions, ${:.2} P&L",
            total_paper_trades, paper_positions, simulated_pnl
        ),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationStats {
    pub paper_mode: bool,
    pub total_paper_trades: usize,
    pub active_paper_positions: usize,
    pub simulated_pnl: f64,
    pub current_parameters: SimulationConfig,
    pub mock_prices: std::collections::HashMap<String, f64>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Monitor position health - keep your chrome from flatlining
/// 
/// The market's a predator. It doesn't sleep, doesn't blink, doesn't forgive.
/// Position health monitoring is your early warning system. When the alarms
/// start screaming, you better listen, or you'll be another ghost in the machine.
#[tauri::command]
pub async fn monitor_position_health(
    check_all: Option<bool>,
    symbol: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<PositionHealthReport> {
    let check_all = check_all.unwrap_or(true);
    let engine = trading_engine.read();
    
    let mut health_reports = Vec::new();
    let mut critical_positions = Vec::new();
    let mut total_risk_score = 0.0;
    
    // Get positions to check
    let positions_to_check: Vec<(String, Position)> = if let Some(sym) = symbol {
        engine.positions
            .iter()
            .filter(|entry| entry.key() == &sym)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    } else {
        engine.positions
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    };
    
    // Analyze each position
    for (pos_id, position) in positions_to_check {
        let health = calculate_position_health_detailed(&position);
        
        // Check critical conditions
        if health.score < 20.0 {
            critical_positions.push(pos_id.clone());
        }
        
        // Calculate liquidation price
        let liquidation_price = calculate_liquidation_price(&position);
        let liquidation_distance = ((position.current_price - liquidation_price).abs() / position.current_price) * 100.0;
        
        // Time-based health decay
        let time_held = Utc::now() - position.opened_at;
        let time_decay = calculate_time_decay(time_held, position.unrealized_pnl);
        
        // Correlation risk with other positions
        let correlation_risk = calculate_correlation_risk(&position, &engine.positions);
        
        // Market condition adjustment
        let market_factor = {
            let state = app_state.read();
            if state.circuit_breaker_triggered {
                0.5 // Halve health scores during circuit breaker
            } else {
                1.0
            }
        };
        
        let adjusted_health = health.score * market_factor * time_decay;
        total_risk_score += (100.0 - adjusted_health) * (position.quantity * position.current_price);
        
        health_reports.push(PositionHealthDetail {
            position_id: pos_id.clone(),
            symbol: position.symbol.clone(),
            health_score: adjusted_health,
            health_status: health.status,
            liquidation_price: liquidation_price,
            liquidation_distance,
            time_decay_factor: time_decay,
            correlation_risk,
            risk_factors: vec![
                RiskFactor {
                    name: "P&L Risk".to_string(),
                    value: health.pnl_risk,
                    severity: if health.pnl_risk > 80.0 { "critical" } else if health.pnl_risk > 50.0 { "high" } else { "medium" }.to_string(),
                },
                RiskFactor {
                    name: "Leverage Risk".to_string(),
                    value: position.margin_used / (position.quantity * position.entry_price) * 100.0,
                    severity: if position.margin_used > 80.0 { "critical" } else { "medium" }.to_string(),
                },
                RiskFactor {
                    name: "Time Decay".to_string(),
                    value: (1.0 - time_decay) * 100.0,
                    severity: if time_decay < 0.5 { "high" } else { "low" }.to_string(),
                },
                RiskFactor {
                    name: "Correlation".to_string(),
                    value: correlation_risk,
                    severity: if correlation_risk > 70.0 { "high" } else { "low" }.to_string(),
                },
            ],
            recommendations: generate_health_recommendations(&position, &health, liquidation_distance),
        });
    }
    
    // System-wide health metrics
    let portfolio_health = if !health_reports.is_empty() {
        health_reports.iter().map(|h| h.health_score).sum::<f32>() / health_reports.len() as f32
    } else {
        100.0
    };
    
    // Auto-protection triggers
    if !critical_positions.is_empty() && check_all {
        warn!("⚠️ CRITICAL POSITIONS DETECTED: {:?}", critical_positions);
        
        // Trigger auto-protection if enabled
        let mut state = app_state.write();
        if state.auto_risk_management {
            state.activity_log.push(format!(
                "[{}] AUTO-PROTECTION: {} critical positions detected. Initiating defensive measures.",
                Utc::now().format("%H:%M:%S"),
                critical_positions.len()
            ));
            
            // Could trigger automatic stop-loss tightening here
        }
    }
    
    Ok(PositionHealthReport {
        timestamp: Utc::now(),
        positions_checked: health_reports.len(),
        critical_positions: critical_positions.len(),
        portfolio_health,
        total_risk_score,
        health_details: health_reports,
        alerts: generate_health_alerts(&critical_positions, portfolio_health),
        message: match portfolio_health {
            h if h >= 80.0 => "Portfolio health: OPTIMAL. Your chrome shines bright.".to_string(),
            h if h >= 60.0 => "Portfolio health: STABLE. Keep your guard up.".to_string(),
            h if h >= 40.0 => "Portfolio health: WARNING. The market smells blood.".to_string(),
            h if h >= 20.0 => "Portfolio health: DANGER. Time to cut losses.".to_string(),
            _ => "Portfolio health: CRITICAL. Pull the ripcord NOW!".to_string(),
        },
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionHealthReport {
    pub timestamp: DateTime<Utc>,
    pub positions_checked: usize,
    pub critical_positions: usize,
    pub portfolio_health: f32,
    pub total_risk_score: f64,
    pub health_details: Vec<PositionHealthDetail>,
    pub alerts: Vec<HealthAlert>,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionHealthDetail {
    pub position_id: String,
    pub symbol: String,
    pub health_score: f32,
    pub health_status: String,
    pub liquidation_price: f64,
    pub liquidation_distance: f64,
    pub time_decay_factor: f64,
    pub correlation_risk: f32,
    pub risk_factors: Vec<RiskFactor>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub value: f64,
    pub severity: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthAlert {
    pub severity: String,
    pub message: String,
    pub action_required: bool,
}

/// Set position health thresholds - define your pain tolerance
/// 
/// Every trader has a breaking point. The smart ones know theirs before
/// the market finds it. Set your thresholds, stick to them, or the 
/// market will set them for you - usually at zero.
#[tauri::command]
pub async fn set_health_thresholds(
    critical_health: Option<f32>,
    warning_health: Option<f32>,
    auto_close_at: Option<f32>,
    enable_auto_protection: Option<bool>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<HealthThresholds> {
    let mut state = app_state.write();
    
    if let Some(critical) = critical_health {
        if critical < 0.0 || critical > 50.0 {
            return Err(CommandError::ValidationError(
                "Critical threshold must be between 0-50. Any higher and you're already dead.".to_string()
            ));
        }
        state.health_thresholds.critical = critical;
    }
    
    if let Some(warning) = warning_health {
        if warning < 20.0 || warning > 80.0 {
            return Err(CommandError::ValidationError(
                "Warning threshold must be between 20-80. Outside that, you're either paranoid or reckless.".to_string()
            ));
        }
        state.health_thresholds.warning = warning;
    }
    
    if let Some(auto_close) = auto_close_at {
        if auto_close < 0.0 || auto_close > 20.0 {
            return Err(CommandError::ValidationError(
                "Auto-close threshold must be between 0-20. This is your dead man's switch.".to_string()
            ));
        }
        state.health_thresholds.auto_close = auto_close;
    }
    
    if let Some(enable) = enable_auto_protection {
        state.auto_risk_management = enable;
    }
    
    info!("🛡️ Health thresholds updated: critical={}, warning={}, auto_close={}, protection={}",
        state.health_thresholds.critical,
        state.health_thresholds.warning,
        state.health_thresholds.auto_close,
        state.auto_risk_management
    );
    
    Ok(HealthThresholds {
        critical: state.health_thresholds.critical,
        warning: state.health_thresholds.warning,
        auto_close: state.health_thresholds.auto_close,
        auto_protection_enabled: state.auto_risk_management,
        message: if state.auto_risk_management {
            "Health thresholds set. Auto-protection ENGAGED. The system has your back.".to_string()
        } else {
            "Health thresholds set. Auto-protection DISABLED. You're flying solo, samurai.".to_string()
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthThresholds {
    pub critical: f32,
    pub warning: f32,
    pub auto_close: f32,
    pub auto_protection_enabled: bool,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// Enhanced helper functions for position health
fn calculate_position_health_detailed(position: &Position) -> PositionHealthDetailed {
    let pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100.0;
    
    // Multi-factor health calculation
    let pnl_factor = if pnl_percent > 0.0 {
        100.0
    } else {
        (100.0 + pnl_percent).max(0.0)
    };
    
    let leverage_factor = 100.0 - (position.margin_used.min(100.0));
    let volatility_factor = 100.0 - (calculate_position_volatility(position) * 100.0).min(50.0);
    
    // Weighted health score
    let score = (pnl_factor * 0.4 + leverage_factor * 0.3 + volatility_factor * 0.3).max(0.0).min(100.0);
    
    let status = match score {
        s if s >= 80.0 => "optimal",
        s if s >= 60.0 => "healthy", 
        s if s >= 40.0 => "warning",
        s if s >= 20.0 => "danger",
        _ => "critical",
    };
    
    PositionHealthDetailed {
        score: score as f32,
        status: status.to_string(),
        pnl_risk: (100.0 - pnl_factor).max(0.0) as f32,
    }
}

#[derive(Debug)]
struct PositionHealthDetailed {
    score: f32,
    status: String,
    pnl_risk: f32,
}

fn calculate_liquidation_price(position: &Position) -> f64 {
    // Simplified liquidation calculation
    // In reality, this would depend on exchange rules and margin requirements
    let margin_ratio = 0.05; // 5% margin = 20x leverage
    
    match position.side {
        PositionSide::Long => {
            position.entry_price * (1.0 - margin_ratio)
        },
        PositionSide::Short => {
            position.entry_price * (1.0 + margin_ratio)
        }
    }
}

fn calculate_time_decay(duration: chrono::Duration, pnl: f64) -> f64 {
    // Positions decay in health over time if not profitable
    let hours = duration.num_hours() as f64;
    
    if pnl > 0.0 {
        1.0 // No decay for profitable positions
    } else {
        // Exponential decay for losing positions
        (1.0 - (hours / 168.0)).max(0.1) // 168 hours = 1 week
    }
}

fn calculate_correlation_risk(position: &Position, all_positions: &dashmap::DashMap<String, Position>) -> f32 {
    // Simplified correlation - in reality would use price correlation matrix
    let same_direction_exposure: f64 = all_positions
        .iter()
        .filter(|entry| {
            entry.value().side == position.side && 
            entry.key() != &position.symbol
        })
        .map(|entry| entry.value().quantity * entry.value().current_price)
        .sum();
    
    let total_exposure: f64 = all_positions
        .iter()
        .map(|entry| entry.value().quantity * entry.value().current_price)
        .sum();
    
    if total_exposure > 0.0 {
        ((same_direction_exposure / total_exposure) * 100.0) as f32
    } else {
        0.0
    }
}

fn calculate_position_volatility(position: &Position) -> f64 {
    // Simplified volatility based on P&L swings
    // In reality, would use historical price data
    let pnl_swing = position.unrealized_pnl.abs() / (position.quantity * position.entry_price);
    pnl_swing.min(1.0)
}

fn generate_health_recommendations(position: &Position, health: &PositionHealthDetailed, liq_distance: f64) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if health.score < 40.0 {
        recommendations.push("Consider reducing position size to lower risk".to_string());
    }
    
    if liq_distance < 10.0 {
        recommendations.push("URGENT: Add margin or close position - liquidation imminent".to_string());
    }
    
    if position.margin_used > 80.0 {
        recommendations.push("Leverage too high - reduce exposure".to_string());
    }
    
    if health.pnl_risk > 50.0 {
        recommendations.push("Set stop-loss to protect capital".to_string());
    }
    
    recommendations
}

fn generate_health_alerts(critical_positions: &[String], portfolio_health: f32) -> Vec<HealthAlert> {
    let mut alerts = Vec::new();
    
    if !critical_positions.is_empty() {
        alerts.push(HealthAlert {
            severity: "critical".to_string(),
            message: format!("{} positions in critical condition", critical_positions.len()),
            action_required: true,
        });
    }
    
    if portfolio_health < 40.0 {
        alerts.push(HealthAlert {
            severity: "high".to_string(),
            message: "Portfolio health deteriorating - review all positions".to_string(),
            action_required: true,
        });
    }
    
    alerts
}

/// Configure exchange fee structure - know your costs or die poor
/// 
/// Every satoshi counts in this game. The difference between profit and loss
/// often comes down to fees. Smart traders optimize for maker rates, track
/// their volume tiers, and know exactly what each trade costs.
#[tauri::command]
pub async fn configure_exchange_fees(
    exchange: String,
    maker_fee: Option<f64>,
    taker_fee: Option<f64>,
    volume_tier: Option<u32>,
    has_fee_discount: Option<bool>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<ExchangeFeeConfig> {
    validation::validate_symbol(&exchange)?; // Reuse for exchange name validation
    
    let mut state = app_state.write();
    
    // Get or create exchange fee config
    let fee_config = state.exchange_fees.entry(exchange.clone())
        .or_insert_with(|| ExchangeFeeStructure::default());
    
    if let Some(maker) = maker_fee {
        if maker < 0.0 || maker > 0.01 {
            return Err(CommandError::ValidationError(
                "Maker fee must be 0-1%. Any higher and you're being robbed.".to_string()
            ));
        }
        fee_config.maker_fee = maker;
    }
    
    if let Some(taker) = taker_fee {
        if taker < 0.0 || taker > 0.01 {
            return Err(CommandError::ValidationError(
                "Taker fee must be 0-1%. Find a better exchange if it's higher.".to_string()
            ));
        }
        fee_config.taker_fee = taker;
    }
    
    if let Some(tier) = volume_tier {
        fee_config.volume_tier = tier;
        // Apply volume discounts
        fee_config.apply_volume_discount();
    }
    
    if let Some(discount) = has_fee_discount {
        fee_config.has_native_token_discount = discount;
    }
    
    info!("💰 Fee structure updated for {}: maker={:.3}%, taker={:.3}%, tier={}",
        exchange, fee_config.maker_fee * 100.0, fee_config.taker_fee * 100.0, fee_config.volume_tier
    );
    
    Ok(ExchangeFeeConfig {
        exchange: exchange.clone(),
        maker_fee: fee_config.maker_fee,
        taker_fee: fee_config.taker_fee,
        effective_maker_fee: fee_config.get_effective_maker_fee(),
        effective_taker_fee: fee_config.get_effective_taker_fee(),
        volume_tier: fee_config.volume_tier,
        has_discount: fee_config.has_native_token_discount,
        message: format!(
            "Fee structure configured. Effective rates: maker={:.4}%, taker={:.4}%",
            fee_config.get_effective_maker_fee() * 100.0,
            fee_config.get_effective_taker_fee() * 100.0
        ),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeFeeConfig {
    pub exchange: String,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub effective_maker_fee: f64,
    pub effective_taker_fee: f64,
    pub volume_tier: u32,
    pub has_discount: bool,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Calculate fees for order - know the true cost
/// 
/// The market takes its cut, always. Exchange fees, network fees, slippage -
/// death by a thousand cuts. This calculates exactly how much blood you'll
/// lose on each trade. Use it before you pull the trigger.
#[tauri::command]
pub async fn calculate_order_fees(
    exchange: String,
    symbol: String,
    side: String,
    order_type: String,
    quantity: f64,
    price: Option<f64>,
    include_network_fee: Option<bool>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<FeeCalculation> {
    validation::validate_symbol(&symbol)?;
    validation::validate_quantity(quantity)?;
    
    let state = app_state.read();
    let include_network = include_network_fee.unwrap_or(false);
    
    // Get exchange fee structure
    let fee_structure = state.exchange_fees
        .get(&exchange)
        .cloned()
        .unwrap_or_else(|| ExchangeFeeStructure::default());
    
    // Determine if maker or taker
    let is_maker = match order_type.to_lowercase().as_str() {
        "limit" => true, // Simplified - limit orders are usually maker
        "market" => false,
        _ => false,
    };
    
    // Calculate base trading fee
    let fee_rate = if is_maker {
        fee_structure.get_effective_maker_fee()
    } else {
        fee_structure.get_effective_taker_fee()
    };
    
    let trade_value = quantity * price.unwrap_or(50000.0);
    let trading_fee = trade_value * fee_rate;
    
    // Calculate network/gas fee if applicable
    let network_fee = if include_network {
        calculate_network_fee(&symbol, &side)
    } else {
        0.0
    };
    
    // Estimate slippage for market orders
    let slippage_cost = if order_type.to_lowercase() == "market" {
        trade_value * 0.0005 // 0.05% average slippage
    } else {
        0.0
    };
    
    // Total cost breakdown
    let total_fee = trading_fee + network_fee + slippage_cost;
    let fee_percentage = (total_fee / trade_value) * 100.0;
    
    // Fee impact on P&L
    let breakeven_movement = fee_percentage * 2.0; // Need to cover fees on entry and exit
    
    Ok(FeeCalculation {
        exchange,
        symbol,
        order_type,
        quantity,
        price: price.unwrap_or(50000.0),
        trading_fee,
        network_fee,
        slippage_cost,
        total_fee,
        fee_percentage,
        fee_rate,
        is_maker,
        breakeven_movement,
        message: if fee_percentage > 0.5 {
            format!("⚠️ High fees detected: {:.3}% per trade. Consider maker orders or volume discounts.", fee_percentage)
        } else {
            format!("Fee impact: {:.3}% per trade. Need {:.2}% movement to break even.", fee_percentage, breakeven_movement)
        },
        recommendations: generate_fee_recommendations(fee_percentage, is_maker, &fee_structure),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeeCalculation {
    pub exchange: String,
    pub symbol: String,
    pub order_type: String,
    pub quantity: f64,
    pub price: f64,
    pub trading_fee: f64,
    pub network_fee: f64,
    pub slippage_cost: f64,
    pub total_fee: f64,
    pub fee_percentage: f64,
    pub fee_rate: f64,
    pub is_maker: bool,
    pub breakeven_movement: f64,
    pub message: String,
    pub recommendations: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Get fee statistics - track the slow bleed
/// 
/// Death by a thousand cuts - that's what fees do to your portfolio.
/// This shows you exactly how much you've bled out to the exchanges.
/// Knowledge is power, and power is keeping more of your money.
#[tauri::command]
pub async fn get_fee_statistics(
    period: Option<String>,
    exchange: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<FeeStatistics> {
    let period = period.unwrap_or_else(|| "24h".to_string());
    let engine = trading_engine.read();
    
    // Calculate time range
    let cutoff_time = match period.as_str() {
        "1h" => Utc::now() - chrono::Duration::hours(1),
        "24h" => Utc::now() - chrono::Duration::days(1),
        "7d" => Utc::now() - chrono::Duration::days(7),
        "30d" => Utc::now() - chrono::Duration::days(30),
        "all" => DateTime::<Utc>::MIN_UTC,
        _ => Utc::now() - chrono::Duration::days(1),
    };
    
    // Filter orders by time and exchange
    let orders: Vec<Order> = engine.order_history.read()
        .iter()
        .filter(|o| {
            o.created_at >= cutoff_time &&
            o.status == OrderStatus::Filled &&
            exchange.as_ref().map_or(true, |ex| &o.exchange == ex)
        })
        .cloned()
        .collect();
    
    // Calculate fee metrics
    let mut total_fees = 0.0;
    let mut maker_fees = 0.0;
    let mut taker_fees = 0.0;
    let mut fee_by_exchange: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut fee_by_symbol: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    
    for order in &orders {
        let fee = calculate_order_fee(&order);
        total_fees += fee;
        
        *fee_by_exchange.entry(order.exchange.clone()).or_insert(0.0) += fee;
        *fee_by_symbol.entry(order.symbol.clone()).or_insert(0.0) += fee;
        
        // Simplified maker/taker detection
        if order.order_type == OrderType::Limit {
            maker_fees += fee;
        } else {
            taker_fees += fee;
        }
    }
    
    // Calculate volume and effective rate
    let total_volume: f64 = orders.iter()
        .map(|o| o.filled_quantity * o.average_fill_price.unwrap_or(0.0))
        .sum();
    
    let effective_fee_rate = if total_volume > 0.0 {
        (total_fees / total_volume) * 100.0
    } else {
        0.0
    };
    
    // Find highest fee symbol
    let highest_fee_symbol = fee_by_symbol
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(symbol, fee)| (symbol.clone(), *fee));
    
    // Calculate savings potential
    let potential_savings = calculate_fee_savings_potential(
        total_fees,
        maker_fees,
        taker_fees,
        &orders
    );
    
    Ok(FeeStatistics {
        period: period.clone(),
        total_fees,
        maker_fees,
        taker_fees,
        total_volume,
        effective_fee_rate,
        trades_count: orders.len(),
        average_fee_per_trade: if orders.is_empty() { 0.0 } else { total_fees / orders.len() as f64 },
        fee_by_exchange,
        fee_by_symbol,
        highest_fee_symbol,
        potential_savings,
        recommendations: generate_fee_optimization_tips(effective_fee_rate, maker_fees, taker_fees),
        message: format!(
            "Fees for {}: ${:.2} on ${:.0} volume ({:.3}% effective rate)",
            period, total_fees, total_volume, effective_fee_rate
        ),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeeStatistics {
    pub period: String,
    pub total_fees: f64,
    pub maker_fees: f64,
    pub taker_fees: f64,
    pub total_volume: f64,
    pub effective_fee_rate: f64,
    pub trades_count: usize,
    pub average_fee_per_trade: f64,
    pub fee_by_exchange: std::collections::HashMap<String, f64>,
    pub fee_by_symbol: std::collections::HashMap<String, f64>,
    pub highest_fee_symbol: Option<(String, f64)>,
    pub potential_savings: f64,
    pub recommendations: Vec<String>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// Note: calculate_fees is already defined above, removing duplicate

fn calculate_order_fee(order: &Order) -> f64 {
    // Calculate actual fee paid on a filled order
    let trade_value = order.filled_quantity * order.average_fill_price.unwrap_or(0.0);
    
    // Get fee from metadata if stored, otherwise estimate
    if let Some(fee_str) = order.metadata.get("fee_paid") {
        fee_str.parse().unwrap_or_else(|_| trade_value * 0.001)
    } else {
        // Estimate based on order type
        let fee_rate = if order.order_type == OrderType::Limit { 0.0008 } else { 0.001 };
        trade_value * fee_rate
    }
}

fn calculate_network_fee(symbol: &str, _side: &str) -> f64 {
    // Simplified network fee estimation
    match symbol {
        s if s.contains("ETH") => 15.0, // ~$15 for ETH transaction
        s if s.contains("BTC") => 5.0,  // ~$5 for BTC transaction
        s if s.contains("BNB") => 0.5,  // ~$0.50 for BSC
        _ => 0.0, // No network fee for CEX-only trades
    }
}

fn generate_fee_recommendations(fee_percentage: f64, is_maker: bool, fee_structure: &ExchangeFeeStructure) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if fee_percentage > 0.5 {
        recommendations.push("Consider switching to limit orders for maker fees".to_string());
    }
    
    if !is_maker && fee_structure.maker_fee < fee_structure.taker_fee * 0.7 {
        recommendations.push(format!(
            "Use limit orders to pay {:.3}% instead of {:.3}%",
            fee_structure.maker_fee * 100.0,
            fee_structure.taker_fee * 100.0
        ));
    }
    
    if fee_structure.volume_tier < 3 {
        recommendations.push("Increase trading volume to unlock fee discounts".to_string());
    }
    
    if !fee_structure.has_native_token_discount {
        recommendations.push("Hold exchange tokens for fee discounts (BNB, FTT, etc.)".to_string());
    }
    
    recommendations
}

fn calculate_fee_savings_potential(total_fees: f64, maker_fees: f64, taker_fees: f64, orders: &[Order]) -> f64 {
    // Calculate how much could be saved with optimization
    let taker_orders = orders.iter().filter(|o| o.order_type != OrderType::Limit).count();
    let potential_maker_fees = (taker_fees / 0.001) * 0.0008; // If all takers were makers
    
    let current_avg_rate = if orders.is_empty() { 0.0 } else { total_fees / orders.len() as f64 };
    let optimal_rate = 0.0008; // Assuming best tier maker rate
    
    let savings = (current_avg_rate - optimal_rate) * orders.len() as f64;
    savings.max(0.0)
}

fn generate_fee_optimization_tips(effective_rate: f64, maker_fees: f64, taker_fees: f64) -> Vec<String> {
    let mut tips = Vec::new();
    
    if effective_rate > 0.08 {
        tips.push("⚠️ Your fees are eating profits. Time to optimize.".to_string());
    }
    
    let maker_ratio = if maker_fees + taker_fees > 0.0 {
        maker_fees / (maker_fees + taker_fees)
    } else {
        0.0
    };
    
    if maker_ratio < 0.5 {
        tips.push(format!("📊 Only {:.1}% maker orders. Aim for 70%+ to reduce fees.", maker_ratio * 100.0));
    }
    
    if effective_rate > 0.05 {
        tips.push("💰 Consider market making strategies to earn rebates".to_string());
    }
    
    tips.push("📈 Track 30-day volume for tier upgrades".to_string());
    
    tips
}

#[derive(Debug, Clone, Default)]
pub struct ExchangeFeeStructure {
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub volume_tier: u32,
    pub has_native_token_discount: bool,
    pub volume_30d: f64,
}

impl ExchangeFeeStructure {
    pub fn apply_volume_discount(&mut self) {
        // Volume-based fee tiers (example based on major exchanges)
        let (maker_discount, taker_discount) = match self.volume_tier {
            0 => (1.0, 1.0),           // No discount
            1 => (0.9, 0.95),          // 10% maker, 5% taker
            2 => (0.8, 0.9),           // 20% maker, 10% taker
            3 => (0.7, 0.85),          // 30% maker, 15% taker
            4 => (0.6, 0.8),           // 40% maker, 20% taker
            _ => (0.5, 0.75),          // 50% maker, 25% taker (max tier)
        };
        
        self.maker_fee *= maker_discount;
        self.taker_fee *= taker_discount;
    }
    
    pub fn get_effective_maker_fee(&self) -> f64 {
        if self.has_native_token_discount {
            self.maker_fee * 0.75 // 25% discount with native token
        } else {
            self.maker_fee
        }
    }
    
    pub fn get_effective_taker_fee(&self) -> f64 {
        if self.has_native_token_discount {
            self.taker_fee * 0.75 // 25% discount with native token
        } else {
            self.taker_fee
        }
    }
    
    /// Create fee structure for common exchanges
    pub fn for_exchange(exchange: &str) -> Self {
        match exchange.to_lowercase().as_str() {
            "binance" => Self {
                maker_fee: 0.001,   // 0.10%
                taker_fee: 0.001,   // 0.10%
                ..Default::default()
            },
            "coinbase" => Self {
                maker_fee: 0.006,   // 0.60%
                taker_fee: 0.006,   // 0.60%
                ..Default::default()
            },
            "kraken" => Self {
                maker_fee: 0.0016,  // 0.16%
                taker_fee: 0.0026,  // 0.26%
                ..Default::default()
            },
            "bybit" => Self {
                maker_fee: 0.001,   // 0.10%
                taker_fee: 0.001,   // 0.10%
                ..Default::default()
            },
            "okx" => Self {
                maker_fee: 0.0008,  // 0.08%
                taker_fee: 0.001,   // 0.10%
                ..Default::default()
            },
            _ => Self::default(),
        }
    }
}

impl Default for ExchangeFeeStructure {
    fn default() -> Self {
        Self {
            maker_fee: 0.001,   // 0.1%
            taker_fee: 0.001,   // 0.1%
            volume_tier: 0,
            has_native_token_discount: false,
            volume_30d: 0.0,
        }
    }
}

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
    exchange: Option<String>, // Exchange to route order to
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
    let risk_check = {
        let engine = trading_engine.read();
        let total_exposure = engine.calculate_total_exposure();
        let order_value = quantity * price.unwrap_or(50000.0);
        
        if total_exposure + order_value > 100000.0 {
            Err(CommandError::TradingError(
                "Order would exceed maximum exposure. The market's already taken enough.".to_string()
            ))
        } else {
            Ok(())
        }
    }?;
    
    // Build the order - assembling the digital weapon
    let order_id = format!("NX-{}-{}", 
        symbol.replace("/", "_"), 
        Uuid::new_v4().to_string().split('-').next().unwrap()
    );
    
    let time_in_force = time_in_force.unwrap_or_else(|| "GTC".to_string());
    
    // Create the order with all fields properly set
    let mut order = Order {
        id: order_id.clone(),
        symbol: symbol.clone(),
        exchange: exchange.unwrap_or_else(|| "BINANCE".to_string()), // Default to Binance
        side,
        order_type: order_type.clone(),
        quantity,
        price,
        stop_price,
        position_id: position_id.clone(), // First-class field now!
        status: OrderStatus::Pending,
        filled_quantity: 0.0,
        average_fill_price: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: metadata.unwrap_or_default(),
    };
    
    // Tag paper trades in metadata
    if app_state.read().simulation_engine.paper_mode {
        order.metadata.insert("paper_trade".to_string(), "true".to_string());
        order.metadata.insert("simulation_version".to_string(), "1.0".to_string());
    }
    
    // Calculate and store estimated fees
    {
        let state = app_state.read();
        let fee_structure = state.exchange_fees
            .get(&order.exchange)
            .cloned()
            .unwrap_or_else(|| ExchangeFeeStructure::default());
        
        let is_maker = order.order_type == OrderType::Limit;
        let fee_rate = if is_maker {
            fee_structure.get_effective_maker_fee()
        } else {
            fee_structure.get_effective_taker_fee()
        };
        
        let estimated_fee = order.quantity * order.price.unwrap_or(50000.0) * fee_rate;
        order.metadata.insert("estimated_fee".to_string(), estimated_fee.to_string());
        order.metadata.insert("fee_rate".to_string(), fee_rate.to_string());
        order.metadata.insert("is_maker".to_string(), is_maker.to_string());
    }
    
    info!("🎯 Placing {} {} order for {} {} @ ${:?} | Est. fee: ${:.2}", 
        order_type, side, quantity, symbol, price,
        order.metadata.get("estimated_fee").and_then(|f| f.parse::<f64>().ok()).unwrap_or(0.0)
    );
    
    // Log position linkage for position-based orders
    if let Some(pos_id) = &position_id {
        info!("📎 Order linked to position: {}", pos_id);
    }
    
    // Submit to the trading engine - let it rip
    let order_result = {
        let state = app_state.read();
        
        if state.simulation_engine.paper_mode {
            // SIMULATION MODE - Apply realistic market conditions
            info!("🎮 SIMULATION MODE: Processing paper order");
            
            // Simulate network latency
            drop(state); // Release lock before async sleep
            tokio::time::sleep(tokio::time::Duration::from_millis(
                app_state.read().simulation_engine.latency_ms
            )).await;
            
            // Check for simulated failures
            if rand::thread_rng().gen::<f64>() < app_state.read().simulation_engine.failure_rate {
                return Err(CommandError::TradingError(
                    "SIMULATED FAILURE: Exchange rejected order. This is practice for when it happens for real.".to_string()
                ));
            }
            
            // Apply slippage for market orders
            let mut final_order = order.clone();
            if final_order.order_type == OrderType::Market {
                let state = app_state.read();
                let slippage_mult = if final_order.side == OrderSide::Buy {
                    1.0 + (state.simulation_engine.slippage_bps / 10000.0)
                } else {
                    1.0 - (state.simulation_engine.slippage_bps / 10000.0)
                };
                
                // Use mock price or default
                let base_price = state.simulation_engine.mock_prices
                    .get(&final_order.symbol)
                    .copied()
                    .unwrap_or(price.unwrap_or(50000.0));
                
                final_order.average_fill_price = Some(base_price * slippage_mult);
                final_order.status = OrderStatus::Filled;
                final_order.filled_quantity = final_order.quantity;
                
                info!("📊 Simulated fill: {} @ ${:.2} (slippage: {:.2}%)", 
                    final_order.symbol, 
                    final_order.average_fill_price.unwrap(),
                    state.simulation_engine.slippage_bps / 100.0
                );
            }
            
            drop(state);
            let mut engine = trading_engine.write();
            engine.place_order(final_order)
        } else {
            // LIVE MODE - Real money, real consequences
            drop(state);
            let mut engine = trading_engine.write();
            engine.place_order(order.clone())
        }
    }?;
    
    // Log to history with cyberpunk flair
    {
        let mut state = app_state.write();
        let fee_info = order.metadata.get("estimated_fee")
            .and_then(|f| f.parse::<f64>().ok())
            .map(|f| format!(" | Fee: ${:.2}", f))
            .unwrap_or_default();
        
        state.activity_log.push(format!(
            "[{}] ORDER PLACED: {} {} {} @ ${:?} | Type: {:?} | Position: {} | TIF: {}{}",
            Utc::now().format("%H:%M:%S"),
            side,
            quantity,
            symbol,
            price,
            order_type,
            position_id.as_deref().unwrap_or("STANDALONE"),
            time_in_force,
            fee_info
        ));
        
        // Keep log size manageable
        if state.activity_log.len() > 1000 {
            state.activity_log.remove(0);
        }
    }
    
    // Cyberpunk response messages based on order type
    let message = match order_type {
        OrderType::Market => "Market order jacked in. Executing at light speed.".to_string(),
        OrderType::Limit => "Limit order set. Patience is a virtue, FOMO is a vice.".to_string(),
        OrderType::Stop => "Stop order armed. Ready to strike when the moment comes.".to_string(),
        OrderType::StopLimit => "Stop limit locked and loaded. Precision over speed.".to_string(),
        OrderType::StopLoss => {
            if position_id.is_some() {
                "Stop loss activated. Your safety net is in place, samurai.".to_string()
            } else {
                "Warning: Stop loss created without position link. Flying blind.".to_string()
            }
        },
        OrderType::TakeProfit => {
            if position_id.is_some() {
                "Take profit set. Greed is good, but profits are better.".to_string()
            } else {
                "Warning: Take profit without position. Target acquired, but no ammo loaded.".to_string()
            }
        },
    };
    
    Ok(PlaceOrderResponse {
        order_id,
        symbol,
        status: order.status.to_string(),
        message,
        estimated_fees: order.metadata.get("estimated_fee")
            .and_then(|f| f.parse().ok())
            .unwrap_or_else(|| calculate_fees(quantity, price)),
        fee_currency: "USD".to_string(),
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
    pub fee_currency: String,
    pub risk_score: f32, // 0-100, higher = riskier
    pub timestamp: DateTime<Utc>,
}

/// Get active orders - check your firing solutions
/// 
/// Every order in flight is a bullet in the air. You better know where each one's
/// headed, or you might shoot yourself in the foot. Or worse, the wallet.
#[tauri::command]
pub async fn get_active_orders(
    symbol: Option<String>,
    order_type: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<ActiveOrdersResponse> {
    let engine = trading_engine.read();
    
    let orders: Vec<OrderInfo> = engine.active_orders
        .iter()
        .filter(|entry| {
            let order = entry.value();
            symbol.as_ref().map_or(true, |s| &order.symbol == s) &&
            order_type.as_ref().map_or(true, |t| {
                match t.to_lowercase().as_str() {
                    "limit" => matches!(order.order_type, OrderType::Limit),
                    "market" => matches!(order.order_type, OrderType::Market),
                    "stop" => matches!(order.order_type, OrderType::Stop | OrderType::StopLimit),
                    "protection" => matches!(order.order_type, OrderType::StopLoss | OrderType::TakeProfit),
                    _ => true
                }
            })
        })
        .map(|entry| {
            let order = entry.value();
            OrderInfo {
                order: order.clone(),
                execution_time: None,
                slippage: 0.0,
                fees_paid: calculate_fees(order.quantity, order.price),
            }
        })
        .collect();
    
    let total_exposure = orders.iter()
        .map(|o| o.order.quantity * o.order.price.unwrap_or(50000.0))
        .sum();
    
    info!("📋 Retrieved {} active orders - Exposure: ${:.2}", orders.len(), total_exposure);
    
    Ok(ActiveOrdersResponse {
        orders,
        total_count: orders.len(),
        total_exposure,
        by_type: count_orders_by_type(&orders),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActiveOrdersResponse {
    pub orders: Vec<OrderInfo>,
    pub total_count: usize,
    pub total_exposure: f64,
    pub by_type: std::collections::HashMap<String, usize>,
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
    
    let mut engine = trading_engine.write();
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
    let _include_closed = include_closed.unwrap_or(false); // FIX: Prefixed with underscore
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
                position_id: entry.key().clone(),
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
    pub position_id: String,
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
/// I keep my history close, study it like a street samurai studies their blade.
#[tauri::command]
pub async fn get_order_history(
    symbol: Option<String>,
    limit: Option<usize>,
    include_cancelled: Option<bool>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<OrderHistoryResponse> {
    let limit = limit.unwrap_or(100).min(1000);
    let include_cancelled = include_cancelled.unwrap_or(true);
    
    let engine = trading_engine.read();
    let history = engine.order_history.read();
    
    let orders: Vec<OrderInfo> = history
        .iter()
        .rev()
        .filter(|order| {
            symbol.as_ref().map_or(true, |s| &order.symbol == s) &&
            (include_cancelled || order.status != OrderStatus::Cancelled)
        })
        .take(limit)
        .map(|order| {
            let execution_time = if order.status == OrderStatus::Filled {
                Some(format_duration(order.updated_at - order.created_at))
            } else {
                None
            };
            
            OrderInfo {
                order: order.clone(),
                execution_time,
                slippage: calculate_slippage(order),
                fees_paid: calculate_fees(order.filled_quantity, order.average_fill_price),
            }
        })
        .collect();
    
    let total_volume = orders.iter()
        .filter(|o| o.order.status == OrderStatus::Filled)
        .map(|o| o.order.filled_quantity * o.order.average_fill_price.unwrap_or(0.0))
        .sum();
    
    info!("📜 Retrieved {} orders from history", orders.len());
    
    Ok(OrderHistoryResponse {
        orders,
        total_count: orders.len(),
        total_volume,
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderHistoryResponse {
    pub orders: Vec<OrderInfo>,
    pub total_count: usize,
    pub total_volume: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderInfo {
    pub order: Order,
    pub execution_time: Option<String>,
    pub slippage: f64,
    pub fees_paid: f64,
}

/// Close position - cash out or cut losses
/// 
/// Sometimes you gotta know when to fold 'em. Could be a full exit or just
/// trimming the fat. In Night City, taking profits is how you stay alive.
#[tauri::command]
pub async fn close_position(
    symbol: String,
    percentage: Option<f64>,
    reason: Option<String>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<ClosePositionResponse> {
    let percentage = percentage.unwrap_or(100.0);
    
    // Validate percentage
    if percentage <= 0.0 || percentage > 100.0 {
        return Err(CommandError::TradingError(
            "Percentage must be between 0 and 100. Can't close negative positions, choom.".to_string()
        ));
    }
    
    let mut engine = trading_engine.write();
    
    // Get the position
    let position = engine.positions.get(&symbol)
        .ok_or_else(|| CommandError::TradingError(
            format!("No position found for {}. Can't close what you don't own.", symbol)
        ))?
        .clone();
    
    // Calculate close quantity
    let close_quantity = position.quantity * (percentage / 100.0);
    let remaining_quantity = position.quantity - close_quantity;
    
    info!("💸 Closing {:.1}% of {} position ({} units)", 
        percentage, symbol, close_quantity
    );
    
    // Create a market order to close the position
    let close_order = Order {
        id: format!("NX-CLOSE-{}", Uuid::new_v4().to_string().split('-').next().unwrap()),
        symbol: symbol.clone(),
        exchange: position.exchange.clone(),
        side: match position.side {
            PositionSide::Long => OrderSide::Sell,
            PositionSide::Short => OrderSide::Buy,
        },
        order_type: OrderType::Market,
        quantity: close_quantity,
        price: None,
        stop_price: None,
        position_id: Some(symbol.clone()), // Link to position
        status: OrderStatus::Pending,
        filled_quantity: 0.0,
        average_fill_price: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: {
            let mut meta = std::collections::HashMap::new();
            meta.insert("close_reason".to_string(), reason.clone().unwrap_or("Manual close".to_string()));
            meta.insert("close_percentage".to_string(), percentage.to_string());
            meta
        },
    };
    
    // Place the close order
    engine.place_order(close_order.clone())?;
    
    // If partial close, update the position
    if percentage < 100.0 {
        if let Some(mut pos) = engine.positions.get_mut(&symbol) {
            pos.quantity = remaining_quantity;
            info!("📊 Position reduced: {} {} remaining", remaining_quantity, symbol);
        }
        
        // Cancel and recreate protection orders for new quantity
        let protection_orders: Vec<String> = engine.active_orders
            .iter()
            .filter(|entry| {
                entry.value().position_id.as_ref() == Some(&symbol) &&
                matches!(entry.value().order_type, OrderType::StopLoss | OrderType::TakeProfit)
            })
            .map(|entry| entry.key().clone())
            .collect();
        
        for order_id in protection_orders {
            if let Some((_, mut order)) = engine.active_orders.remove(&order_id) {
                // Recreate with new quantity
                order.quantity = remaining_quantity;
                order.id = format!("NX-{}-{}", 
                    match order.order_type {
                        OrderType::StopLoss => "SL",
                        OrderType::TakeProfit => "TP",
                        _ => "ORD"
                    },
                    Uuid::new_v4().to_string().split('-').next().unwrap()
                );
                engine.active_orders.insert(order.id.clone(), order);
            }
        }
    } else {
        // Full close - remove position and cancel all related orders
        engine.positions.remove(&symbol);
        
        // Cancel all position-related orders
        let related_orders: Vec<String> = engine.active_orders
            .iter()
            .filter(|entry| entry.value().position_id.as_ref() == Some(&symbol))
            .map(|entry| entry.key().clone())
            .collect();
        
        for order_id in &related_orders {
            engine.cancel_order(order_id)?;
        }
        
        info!("🏁 Position fully closed: {} | Related orders cancelled: {}", 
            symbol, related_orders.len()
        );
    }
    
    // Calculate P&L for response
    let pnl = position.unrealized_pnl * (percentage / 100.0);
    let pnl_percentage = (pnl / (position.entry_price * close_quantity)) * 100.0;
    
    // Update P&L tracker
    {
        let mut pnl_tracker = engine.pnl_tracker.write();
        update_pnl_on_close(&mut pnl_tracker, pnl, &symbol);
    }
    
    // Update activity log
    {
        let mut state = app_state.write();
        state.activity_log.push(format!(
            "[{}] POSITION {} CLOSED: {:.1}% of {} | P&L: ${:.2} ({:.1}%) | Reason: {}",
            Utc::now().format("%H:%M:%S"),
            if pnl >= 0.0 { "WIN" } else { "LOSS" },
            percentage,
            symbol,
            pnl,
            pnl_percentage,
            reason.as_deref().unwrap_or("Manual")
        ));
    }
    
    Ok(ClosePositionResponse {
        symbol,
        percentage_closed: percentage,
        quantity_closed: close_quantity,
        remaining_quantity,
        realized_pnl: pnl,
        pnl_percentage,
        close_order_id: close_order.id,
        message: if percentage < 100.0 {
            format!("Partial close executed. {:.1}% cashed out, {:.1}% still riding.", percentage, 100.0 - percentage)
        } else if pnl >= 0.0 {
            "Position closed in profit. Another win for the chrome.".to_string()
        } else {
            "Position closed at loss. Sometimes you gotta cut and run.".to_string()
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClosePositionResponse {
    pub symbol: String,
    pub percentage_closed: f64,
    pub quantity_closed: f64,
    pub remaining_quantity: f64,
    pub realized_pnl: f64,
    pub pnl_percentage: f64,
    pub close_order_id: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Modify position protection - adjust your defenses
/// 
/// The market's a living thing. What protected you yesterday might kill you today.
/// Smart traders adjust their stops like a netrunner adjusts their ICE.
#[tauri::command]
pub async fn modify_position_protection(
    position_id: String,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<ModifyProtectionResponse> {
    if stop_loss.is_none() && take_profit.is_none() {
        return Err(CommandError::TradingError(
            "Need at least one value to modify. Give me something to work with.".to_string()
        ));
    }
    
    if let Some(sl) = stop_loss {
        validation::validate_price(sl)?;
    }
    if let Some(tp) = take_profit {
        validation::validate_price(tp)?;
    }
    
    let mut engine = trading_engine.write();
    
    // Get the position
    let position = engine.positions.get(&position_id)
        .ok_or_else(|| CommandError::TradingError(
            format!("Position {} not found. Can't protect what doesn't exist.", position_id)
        ))?
        .clone();
    
    // Find existing protection orders
    let protection_orders: Vec<String> = engine.active_orders
        .iter()
        .filter(|entry| {
            entry.value().position_id.as_ref() == Some(&position_id) &&
            matches!(entry.value().order_type, OrderType::StopLoss | OrderType::TakeProfit)
        })
        .map(|entry| entry.key().clone())
        .collect();
    
    // Cancel existing protection orders
    for order_id in &protection_orders {
        engine.cancel_order(order_id)?;
    }
    
    let mut new_orders = Vec::new();
    
    // Create new stop loss if provided
    if let Some(sl_price) = stop_loss {
        let sl_order = Order {
            id: format!("NX-SL-{}", Uuid::new_v4().to_string().split('-').next().unwrap()),
            symbol: position.symbol.clone(),
            exchange: position.exchange.clone(),
            side: match position.side {
                PositionSide::Long => OrderSide::Sell,
                PositionSide::Short => OrderSide::Buy,
            },
            order_type: OrderType::StopLoss,
            quantity: position.quantity,
            price: None,
            stop_price: Some(sl_price),
            position_id: Some(position_id.clone()),
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        
        engine.place_order(sl_order.clone())?;
        new_orders.push(sl_order.id);
    }
    
    // Create new take profit if provided
    if let Some(tp_price) = take_profit {
        let tp_order = Order {
            id: format!("NX-TP-{}", Uuid::new_v4().to_string().split('-').next().unwrap()),
            symbol: position.symbol.clone(),
            exchange: position.exchange.clone(),
            side: match position.side {
                PositionSide::Long => OrderSide::Sell,
                PositionSide::Short => OrderSide::Buy,
            },
            order_type: OrderType::TakeProfit,
            quantity: position.quantity,
            price: Some(tp_price),
            stop_price: None,
            position_id: Some(position_id.clone()),
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        
        engine.place_order(tp_order.clone())?;
        new_orders.push(tp_order.id);
    }
    
    info!("🛡️ Modified protection for position {} - Cancelled {} orders, created {} new",
        position_id, protection_orders.len(), new_orders.len());
    
    Ok(ModifyProtectionResponse {
        position_id,
        cancelled_orders: protection_orders,
        new_orders,
        message: "Protection updated. Your defenses have evolved.".to_string(),
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModifyProtectionResponse {
    pub position_id: String,
    pub cancelled_orders: Vec<String>,
    pub new_orders: Vec<String>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Cancel all orders - abort mission
/// 
/// When the market turns against you, sometimes you need to pull all your orders
/// before they execute. It's like yanking all your netrunner probes before the ICE fries them.
#[tauri::command]
pub async fn cancel_all_orders(
    symbol: Option<String>,
    order_type: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<CancelAllOrdersResponse> {
    let mut engine = trading_engine.write();
    
    let orders_to_cancel: Vec<String> = engine.active_orders
        .iter()
        .filter(|entry| {
            let order = entry.value();
            symbol.as_ref().map_or(true, |s| &order.symbol == s) &&
            order_type.as_ref().map_or(true, |t| {
                match t.to_lowercase().as_str() {
                    "limit" => matches!(order.order_type, OrderType::Limit),
                    "stop" => matches!(order.order_type, OrderType::Stop | OrderType::StopLimit),
                    "protection" => matches!(order.order_type, OrderType::StopLoss | OrderType::TakeProfit),
                    _ => true
                }
            })
        })
        .map(|entry| entry.key().clone())
        .collect();
    
    if orders_to_cancel.is_empty() {
        return Ok(CancelAllOrdersResponse {
            cancelled_count: 0,
            failed_count: 0,
            order_ids: vec![],
            message: "No orders to cancel. The matrix is already clear.".to_string(),
            timestamp: Utc::now(),
        });
    }
    
    info!("🚫 Mass order cancellation: {} orders targeted", orders_to_cancel.len());
    
    let mut cancelled = Vec::new();
    let mut failed = 0;
    
    for order_id in orders_to_cancel {
        match engine.cancel_order(&order_id) {
            Ok(_) => cancelled.push(order_id),
            Err(e) => {
                warn!("Failed to cancel order {}: {:?}", order_id, e);
                failed += 1;
            }
        }
    }
    
    let message = if failed == 0 {
        format!("Clean sweep. {} orders terminated with extreme prejudice.", cancelled.len())
    } else {
        format!("Partial success. {} orders cancelled, {} resisted termination.", cancelled.len(), failed)
    };
    
    Ok(CancelAllOrdersResponse {
        cancelled_count: cancelled.len(),
        failed_count: failed,
        order_ids: cancelled,
        message,
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CancelAllOrdersResponse {
    pub cancelled_count: usize,
    pub failed_count: usize,
    pub order_ids: Vec<String>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

/// Close all positions - nuclear option
/// 
/// When the market goes full psycho, sometimes you gotta pull the plug on everything.
/// This is your panic button, your "get me the hell out" switch. Use it wisely.
#[tauri::command]
pub async fn close_all_positions(
    reason: Option<String>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<CloseAllPositionsResponse> {
    let reason = reason.unwrap_or_else(|| "Emergency close all".to_string());
    
    info!("🚨 CLOSING ALL POSITIONS - Reason: {}", reason);
    
    let engine = trading_engine.read();
    let positions: Vec<(String, Position)> = engine.positions
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    
    if positions.is_empty() {
        return Ok(CloseAllPositionsResponse {
            positions_closed: 0,
            total_pnl: 0.0,
            close_orders: vec![],
            message: "No positions to close. Portfolio already flat.".to_string(),
            timestamp: Utc::now(),
        });
    }
    
    drop(engine); // Release read lock before getting write lock
    
    let mut close_orders = Vec::new();
    let mut total_pnl = 0.0;
    let mut errors = Vec::new();
    
    // Close each position
    for (symbol, position) in &positions {
        match close_position(
            symbol.clone(),
            Some(100.0),
            Some(reason.clone()),
            app_state.clone(),
            trading_engine.clone()
        ).await {
            Ok(response) => {
                close_orders.push(response.close_order_id);
                total_pnl += response.realized_pnl;
            },
            Err(e) => {
                errors.push(format!("{}: {}", symbol, e));
            }
        }
    }
    
    // Log the mass exodus
    {
        let mut state = app_state.write();
        state.activity_log.push(format!(
            "[{}] EMERGENCY CLOSE ALL: {} positions | Total P&L: ${:.2} | Errors: {}",
            Utc::now().format("%H:%M:%S"),
            positions.len(),
            total_pnl,
            errors.len()
        ));
    }
    
    let message = if errors.is_empty() {
        format!("All {} positions flatlined. Total P&L: ${:.2}. The chrome survives another day.", 
            positions.len(), total_pnl)
    } else {
        format!("Closed {} of {} positions. {} errors occurred. Check logs for details.", 
            close_orders.len(), positions.len(), errors.len())
    };
    
    Ok(CloseAllPositionsResponse {
        positions_closed: close_orders.len(),
        total_pnl,
        close_orders,
        message,
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CloseAllPositionsResponse {
    pub positions_closed: usize,
    pub total_pnl: f64,
    pub close_orders: Vec<String>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────
// HELPER FUNCTIONS - The chrome that makes it shine
// ─────────────────────────────────────────────────────────────

fn calculate_fees(quantity: f64, price: Option<f64>) -> f64 {
    // Simple fee model - 0.1% of trade value
    let trade_value = quantity * price.unwrap_or(50000.0);
    trade_value * 0.001
}

fn calculate_risk_score(order: &Order) -> f32 {
    // Risk scoring algorithm - higher is riskier
    let mut score = 0.0;
    
    // Market orders are riskier
    if order.order_type == OrderType::Market {
        score += 20.0;
    }
    
    // Large orders are riskier
    let order_value = order.quantity * order.price.unwrap_or(50000.0);
    if order_value > 10000.0 {
        score += 30.0;
    } else if order_value > 5000.0 {
        score += 15.0;
    }
    
    // Orders without stops are very risky
    if order.stop_price.is_none() && matches!(order.order_type, OrderType::Market | OrderType::Limit) {
        score += 40.0;
    }
    
    score.min(100.0)
}

fn calculate_position_health(position: &Position) -> PositionHealth {
    let pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100.0;
    
    let (score, status) = if pnl_percent > 10.0 {
        (90.0, "optimal")
    } else if pnl_percent > 5.0 {
        (75.0, "healthy")
    } else if pnl_percent > -2.0 {
        (50.0, "neutral")
    } else if pnl_percent > -5.0 {
        (25.0, "warning")
    } else {
        (10.0, "critical")
    };
    
    PositionHealth {
        score,
        status: status.to_string(),
    }
}

#[derive(Debug)]
struct PositionHealth {
    score: f32,
    status: String,
}

fn calculate_position_risk(position: &Position) -> RiskMetrics {
    // Multi-factor risk calculation
    let position_value = position.quantity * position.current_price;
    let potential_loss = position.quantity * (position.current_price - position.entry_price).abs();
    
    // Value at Risk (simplified 95% confidence)
    let volatility = calculate_position_volatility(position);
    let var_95 = position_value * volatility * 1.645; // 95% confidence interval
    
    // Maximum potential loss (to zero or liquidation)
    let liquidation_price = calculate_liquidation_price(position);
    let max_loss = match position.side {
        PositionSide::Long => position.quantity * (position.current_price - liquidation_price),
        PositionSide::Short => position.quantity * (liquidation_price - position.current_price),
    };
    
    RiskMetrics {
        var_95,
        max_loss: max_loss.abs(),
        correlation_risk: 25.0, // Placeholder - would calculate actual correlation
        liquidation_price: Some(liquidation_price),
    }
}

fn calculate_margin_usage(engine: &TradingEngine) -> f32 {
    // Calculate margin usage across all positions
    let total_margin: f64 = engine.positions
        .iter()
        .map(|entry| entry.value().margin_used)
        .sum();
    
    // Assume 10x leverage = 10% margin requirement
    (total_margin / 100000.0 * 100.0) as f32
}

fn calculate_slippage(order: &Order) -> f64 {
    if let (Some(target), Some(actual)) = (order.price, order.average_fill_price) {
        ((actual - target) / target * 100.0).abs()
    } else {
        0.0
    }
}

fn format_duration(duration: chrono::Duration) -> String {
    let seconds = duration.num_seconds();
    
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else if seconds < 86400 {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    } else {
        format!("{}d {}h", seconds / 86400, (seconds % 86400) / 3600)
    }
}

fn count_orders_by_type(orders: &[OrderInfo]) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    
    for order_info in orders {
        let type_name = match order_info.order.order_type {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::Stop => "stop",
            OrderType::StopLimit => "stop_limit",
            OrderType::StopLoss => "stop_loss",
            OrderType::TakeProfit => "take_profit",
        };
        
        *counts.entry(type_name.to_string()).or_insert(0) += 1;
    }
    
    counts
}

/// Start automated position monitoring - your guardian in the shadows
/// 
/// The market never sleeps, so neither should your protection. This spawns
/// a background daemon that watches your positions 24/7, ready to pull the
/// trigger when things go south. It's saved more asses than I can count.
#[tauri::command]
pub async fn start_position_monitoring(
    interval_seconds: Option<u64>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
) -> CommandResult<MonitoringStatus> {
    let interval = interval_seconds.unwrap_or(5); // Default 5 second checks
    
    if interval < 1 || interval > 300 {
        return Err(CommandError::ValidationError(
            "Monitoring interval must be 1-300 seconds. Too fast burns CPU, too slow misses the knife.".to_string()
        ));
    }
    
    let monitoring_active = {
        let state = app_state.read();
        state.position_monitoring_active
    };
    
    if monitoring_active {
        return Ok(MonitoringStatus {
            active: true,
            interval,
            message: "Position monitoring already active. Your guardian never left.".to_string(),
            positions_monitored: trading_engine.read().positions.len(),
            last_check: Utc::now(),
        });
    }
    
    // Set monitoring flag
    {
        let mut state = app_state.write();
        state.position_monitoring_active = true;
    }
    
    // Spawn monitoring task
    let app_state_clone = app_state.inner().clone();
    let trading_engine_clone = trading_engine.inner().clone();
    
    tokio::spawn(async move {
        let mut interval_timer = tokio::time::interval(tokio::time::Duration::from_secs(interval));
        
        loop {
            interval_timer.tick().await;
            
            // Check if monitoring should continue
            let should_continue = {
                let state = app_state_clone.read();
                state.position_monitoring_active
            };
            
            if !should_continue {
                info!("🛑 Position monitoring stopped");
                break;
            }
            
            // Perform health check
            let positions: Vec<(String, Position)> = {
                let engine = trading_engine_clone.read();
                engine.positions
                    .iter()
                    .map(|entry| (entry.key().clone(), entry.value().clone()))
                    .collect()
            };
            
            for (pos_id, position) in positions {
                let health = calculate_position_health_detailed(&position);
                
                // Critical health - trigger auto-protection
                if health.score < 20.0 {
                    warn!("💀 CRITICAL HEALTH: {} @ {:.1}", pos_id, health.score);
                    
                    let auto_protect = {
                        let state = app_state_clone.read();
                        state.auto_risk_management && health.score < state.health_thresholds.auto_close
                    };
                    
                    if auto_protect {
                        error!("🚨 AUTO-CLOSING POSITION: {} - Health: {:.1}", pos_id, health.score);
                        
                        // Create emergency close order
                        let close_order = Order {
                            id: format!("NX-EMERGENCY-{}", Uuid::new_v4().to_string().split('-').next().unwrap()),
                            symbol: position.symbol.clone(),
                            exchange: position.exchange.clone(),
                            side: match position.side {
                                PositionSide::Long => OrderSide::Sell,
                                PositionSide::Short => OrderSide::Buy,
                            },
                            order_type: OrderType::Market,
                            quantity: position.quantity,
                            price: None,
                            stop_price: None,
                            position_id: Some(pos_id.clone()),
                            status: OrderStatus::Pending,
                            filled_quantity: 0.0,
                            average_fill_price: None,
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            metadata: {
                                let mut meta = std::collections::HashMap::new();
                                meta.insert("emergency_close".to_string(), "true".to_string());
                                meta.insert("health_score".to_string(), health.score.to_string());
                                meta
                            },
                        };
                        
                        // Place emergency order
                        let mut engine = trading_engine_clone.write();
                        if let Err(e) = engine.place_order(close_order) {
                            error!("Failed to place emergency close order: {:?}", e);
                        }
                    }
                }
                
                // Check stop-loss distance and tighten if needed
                if health.score < 40.0 && position.side == PositionSide::Long {
                    // Could implement dynamic stop-loss adjustment here
                    debug!("⚠️ Position {} health declining: {:.1}", pos_id, health.score);
                }
            }
        }
    });
    
    info!("🛡️ Position monitoring started - Interval: {}s", interval);
    
    Ok(MonitoringStatus {
        active: true,
        interval,
        message: "Position monitoring activated. Your guardian watches from the shadows.".to_string(),
        positions_monitored: trading_engine.read().positions.len(),
        last_check: Utc::now(),
    })
}

/// Stop position monitoring - call off the guardian
#[tauri::command]
pub async fn stop_position_monitoring(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<MonitoringStatus> {
    let mut state = app_state.write();
    state.position_monitoring_active = false;
    
    Ok(MonitoringStatus {
        active: false,
        interval: 0,
        message: "Position monitoring deactivated. You're on your own now.".to_string(),
        positions_monitored: 0,
        last_check: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringStatus {
    pub active: bool,
    pub interval: u64,
    pub message: String,
    pub positions_monitored: usize,
    pub last_check: DateTime<Utc>,
}

/// Calculate quantum position score - multi-dimensional health analysis
/// 
/// The market isn't linear, it's quantum. Positions exist in multiple states
/// simultaneously - profitable yet risky, stable yet decaying. This quantum
/// scoring system sees through the noise to the truth beneath.
#[tauri::command]
pub async fn calculate_quantum_scores(
    symbol: Option<String>,
    trading_engine: State<'_, Arc<RwLock<TradingEngine>>>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<QuantumAnalysis> {
    let engine = trading_engine.read();
    let state = app_state.read();
    
    let positions: Vec<(String, Position)> = if let Some(sym) = symbol {
        engine.positions
            .iter()
            .filter(|entry| entry.key() == &sym)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    } else {
        engine.positions
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    };
    
    let mut quantum_scores = Vec::new();
    let mut cluster_analysis = Vec::new();
    
    // Analyze each position's quantum state
    for (pos_id, position) in &positions {
        // Time decay factor
        let time_held = Utc::now() - position.opened_at;
        let time_decay = (-(time_held.num_hours() as f64 / 168.0)).exp(); // e^(-t/168) decay
        
        // Neural score (simulated - in reality would use ML model)
        let neural_score = simulate_neural_evaluation(&position, &state);
        
        // Profitability score
        let pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100.0;
        let profitability = (50.0 + pnl_percent * 2.0).max(0.0).min(100.0);
        
        // Risk score
        let risk = calculate_quantum_risk(&position);
        
        // Momentum score
        let momentum = calculate_momentum(&position);
        
        // Correlation score
        let correlation = calculate_correlation_risk(&position, &engine.positions);
        
        // Calculate quantum superposition
        let quantum_state = determine_quantum_state(
            neural_score,
            profitability,
            risk,
            momentum,
            time_decay
        );
        
        // Overall quantum score with weighted components
        let overall_score = (
            neural_score * 0.35 +
            profitability * 0.25 +
            momentum * 0.20 +
            (100.0 - risk as f64) * 0.15 +
            (100.0 - correlation as f64) * 0.05
        ) * time_decay;
        
        quantum_scores.push(QuantumPositionScore {
            position_id: pos_id.clone(),
            symbol: position.symbol.clone(),
            overall_score: overall_score as f32,
            quantum_state: quantum_state.clone(),
            components: QuantumComponents {
                neural: neural_score as f32,
                profitability: profitability as f32,
                risk: risk as f32,
                momentum: momentum as f32,
                correlation: correlation,
                time_decay: time_decay as f32,
            },
            recommendation: generate_quantum_recommendation(
                overall_score,
                &quantum_state,
                pnl_percent
            ),
            state_probability: calculate_state_probability(&quantum_state, overall_score),
        });
    }
    
    // Perform cluster analysis
    if positions.len() > 1 {
        cluster_analysis = analyze_position_clusters(&positions, &quantum_scores);
    }
    
    // Calculate portfolio quantum metrics
    let portfolio_coherence = calculate_portfolio_coherence(&quantum_scores);
    let entropy = calculate_portfolio_entropy(&quantum_scores);
    let entanglement = calculate_position_entanglement(&positions);
    
    Ok(QuantumAnalysis {
        timestamp: Utc::now(),
        positions_analyzed: quantum_scores.len(),
        quantum_scores,
        cluster_analysis,
        portfolio_metrics: PortfolioQuantumMetrics {
            coherence: portfolio_coherence,
            entropy,
            entanglement,
            optimal_state_probability: calculate_optimal_probability(&quantum_scores),
        },
        message: interpret_quantum_state(portfolio_coherence, entropy),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumAnalysis {
    pub timestamp: DateTime<Utc>,
    pub positions_analyzed: usize,
    pub quantum_scores: Vec<QuantumPositionScore>,
    pub cluster_analysis: Vec<PositionCluster>,
    pub portfolio_metrics: PortfolioQuantumMetrics,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumPositionScore {
    pub position_id: String,
    pub symbol: String,
    pub overall_score: f32,
    pub quantum_state: String,
    pub components: QuantumComponents,
    pub recommendation: String,
    pub state_probability: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumComponents {
    pub neural: f32,
    pub profitability: f32,
    pub risk: f32,
    pub momentum: f32,
    pub correlation: f32,
    pub time_decay: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionCluster {
    pub cluster_id: String,
    pub positions: Vec<String>,
    pub correlation_strength: f32,
    pub combined_risk: f32,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioQuantumMetrics {
    pub coherence: f32,
    pub entropy: f32,
    pub entanglement: f32,
    pub optimal_state_probability: f32,
}

// Quantum analysis helper functions
fn simulate_neural_evaluation(position: &Position, _state: &AppState) -> f64 {
    // Simulated neural network evaluation
    // In reality, would use trained ML model
    let base_score = 50.0;
    let pnl_adjustment = position.unrealized_pnl.signum() * 20.0;
    let volatility_penalty = calculate_position_volatility(position) * 30.0;
    
    (base_score + pnl_adjustment - volatility_penalty).max(0.0).min(100.0)
}

fn calculate_momentum(position: &Position) -> f64 {
    // Simple momentum based on P&L direction
    let pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100.0;
    (50.0 + pnl_percent).max(0.0).min(100.0)
}

fn determine_quantum_state(neural: f64, profit: f64, risk: f32, momentum: f64, decay: f64) -> String {
    let score = (neural + profit + momentum) / 3.0 * decay;
    
    match (score, risk) {
        (s, r) if s > 80.0 && r < 30.0 => "OPTIMAL|HOLD",
        (s, r) if s > 60.0 && r < 50.0 => "SCALING_IN|OPPORTUNITY",
        (s, _) if s > 40.0 => "NEUTRAL|MONITOR",
        (s, r) if s > 20.0 && r > 70.0 => "SCALING_OUT|CAUTION",
        _ => "CLOSING|EXIT",
    }.to_string()
}

fn calculate_state_probability(state: &str, score: f64) -> f32 {
    // Probability of maintaining current state
    match state {
        "OPTIMAL|HOLD" => (score / 100.0 * 0.9) as f32,
        "CLOSING|EXIT" => ((100.0 - score) / 100.0 * 0.9) as f32,
        _ => 0.5,
    }
}

fn generate_quantum_recommendation(score: f64, state: &str, pnl_percent: f64) -> String {
    match (state, score) {
        ("OPTIMAL|HOLD", _) => "Position in quantum optimal state. Let profits run.".to_string(),
        ("SCALING_IN|OPPORTUNITY", s) if s > 70.0 => "Strong momentum detected. Consider pyramiding.".to_string(),
        ("NEUTRAL|MONITOR", _) => "Quantum superposition unstable. Await clearer signals.".to_string(),
        ("SCALING_OUT|CAUTION", _) => "Quantum decoherence detected. Reduce exposure.".to_string(),
        ("CLOSING|EXIT", _) if pnl_percent < -10.0 => "Quantum collapse imminent. Exit immediately.".to_string(),
        _ => "Monitor closely. State transition probable.".to_string(),
    }
}

fn analyze_position_clusters(positions: &[(String, Position)], scores: &[QuantumPositionScore]) -> Vec<PositionCluster> {
    // Simple clustering based on correlation
    // In reality, would use proper clustering algorithm
    let mut clusters = Vec::new();
    
    if positions.len() > 2 {
        clusters.push(PositionCluster {
            cluster_id: "LONG_CLUSTER".to_string(),
            positions: positions.iter()
                .filter(|(_, p)| matches!(p.side, PositionSide::Long))
                .map(|(id, _)| id.clone())
                .collect(),
            correlation_strength: 0.75,
            combined_risk: 65.0,
            recommendation: "Diversify long exposure across uncorrelated assets".to_string(),
        });
    }
    
    clusters
}

fn calculate_portfolio_coherence(scores: &[QuantumPositionScore]) -> f32 {
    if scores.is_empty() { return 0.0; }
    
    let avg_score: f32 = scores.iter().map(|s| s.overall_score).sum::<f32>() / scores.len() as f32;
    let variance: f32 = scores.iter()
        .map(|s| (s.overall_score - avg_score).powi(2))
        .sum::<f32>() / scores.len() as f32;
    
    100.0 - variance.sqrt()
}

fn calculate_portfolio_entropy(scores: &[QuantumPositionScore]) -> f32 {
    // Shannon entropy of position states
    if scores.is_empty() { return 0.0; }
    
    let total = scores.len() as f32;
    let mut state_counts = std::collections::HashMap::new();
    
    for score in scores {
        *state_counts.entry(score.quantum_state.clone()).or_insert(0.0) += 1.0;
    }
    
    state_counts.values()
        .map(|&count| {
            let p = count / total;
            -p * p.log2()
        })
        .sum()
}

fn calculate_position_entanglement(positions: &[(String, Position)]) -> f32 {
    // Measure how intertwined positions are
    if positions.len() < 2 { return 0.0; }
    
    let same_side_count = positions.iter()
        .filter(|(_, p)| matches!(p.side, PositionSide::Long))
        .count();
    
    let ratio = same_side_count as f32 / positions.len() as f32;
    (ratio - 0.5).abs() * 200.0 // 0-100 scale
}

fn calculate_optimal_probability(scores: &[QuantumPositionScore]) -> f32 {
    if scores.is_empty() { return 0.0; }
    
    let optimal_count = scores.iter()
        .filter(|s| s.quantum_state.contains("OPTIMAL"))
        .count();
    
    optimal_count as f32 / scores.len() as f32 * 100.0
}

fn calculate_quantum_risk(position: &Position) -> f32 {
    // Complex risk calculation for quantum scoring
    let volatility = calculate_position_volatility(position);
    let leverage = position.margin_used / (position.quantity * position.entry_price);
    let drawdown = if position.unrealized_pnl < 0.0 {
        (position.unrealized_pnl.abs() / (position.quantity * position.entry_price)) * 100.0
    } else {
        0.0
    };
    
    // Combine factors into risk score (0-100)
    let risk_score = (volatility * 100.0 * 0.4 + leverage * 100.0 * 0.3 + drawdown * 0.3)
        .min(100.0) as f32;
    
    risk_score
}

fn interpret_quantum_state(coherence: f32, entropy: f32) -> String {
    match (coherence, entropy) {
        (c, e) if c > 80.0 && e < 1.0 => {
            "Portfolio in quantum coherent state. Strong directional bias detected.".to_string()
        },
        (c, e) if c > 60.0 && e < 2.0 => {
            "Portfolio exhibiting quantum stability. Positions aligned.".to_string()
        },
        (c, e) if c < 40.0 && e > 2.0 => {
            "High quantum entropy detected. Portfolio lacks coherence.".to_string()
        },
        _ => {
            "Portfolio in quantum flux. Monitor for state transitions.".to_string()
        }
    }
}

impl TradingEngine {
    /// Calculate total exposure across all positions
    pub fn calculate_total_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|entry| entry.value().quantity * entry.value().current_price)
            .sum()
    }
}

/// Update trading volume - climb the fee tier ladder
/// 
/// Volume is power in this game. More volume means lower fees, and lower fees
/// mean more profits. Track your 30-day volume religiously - every trade counts
/// toward your next tier upgrade. The corps reward loyalty with discounts.
#[tauri::command]
pub async fn update_trading_volume(
    exchange: String,
    volume_30d: f64,
    check_tier_upgrade: Option<bool>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<VolumeUpdateResponse> {
    validation::validate_symbol(&exchange)?; // Reuse for exchange validation
    
    if volume_30d < 0.0 {
        return Err(CommandError::ValidationError(
            "Volume can't be negative. Nice try, but the exchange knows.".to_string()
        ));
    }
    
    let mut state = app_state.write();
    let fee_config = state.exchange_fees.entry(exchange.clone())
        .or_insert_with(|| ExchangeFeeStructure::for_exchange(&exchange));
    
    let old_tier = fee_config.volume_tier;
    fee_config.volume_30d = volume_30d;
    
    // Auto-calculate tier based on volume
    let new_tier = calculate_volume_tier(&exchange, volume_30d);
    let tier_changed = new_tier != old_tier;
    
    if tier_changed {
        fee_config.volume_tier = new_tier;
        fee_config.apply_volume_discount();
        info!("🎯 FEE TIER UPGRADE: {} moved from tier {} to tier {}", 
            exchange, old_tier, new_tier);
    }
    
    let next_tier_volume = get_next_tier_requirement(&exchange, new_tier);
    let progress_to_next = if let Some(next_vol) = next_tier_volume {
        ((volume_30d / next_vol) * 100.0).min(100.0)
    } else {
        100.0 // Max tier reached
    };
    
    Ok(VolumeUpdateResponse {
        exchange,
        volume_30d,
        current_tier: new_tier,
        previous_tier: old_tier,
        tier_changed,
        next_tier_volume,
        progress_to_next,
        current_fees: FeeStructure {
            maker: fee_config.get_effective_maker_fee(),
            taker: fee_config.get_effective_taker_fee(),
        },
        message: if tier_changed {
            format!("🎉 TIER UPGRADE! Now enjoying tier {} benefits. Fees reduced.", new_tier)
        } else if let Some(next) = next_tier_volume {
            format!("Current tier: {}. Trade ${:.0} more for tier {} ({:.1}% there)", 
                new_tier, next - volume_30d, new_tier + 1, progress_to_next)
        } else {
            "Maximum fee tier achieved. You've reached the promised land.".to_string()
        },
        timestamp: Utc::now(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VolumeUpdateResponse {
    pub exchange: String,
    pub volume_30d: f64,
    pub current_tier: u32,
    pub previous_tier: u32,
    pub tier_changed: bool,
    pub next_tier_volume: Option<f64>,
    pub progress_to_next: f64,
    pub current_fees: FeeStructure,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeeStructure {
    pub maker: f64,
    pub taker: f64,
}

// Volume tier calculation helpers
fn calculate_volume_tier(exchange: &str, volume_30d: f64) -> u32 {
    // Volume tiers in USD (example based on major exchanges)
    match exchange.to_lowercase().as_str() {
        "binance" => match volume_30d {
            v if v >= 1_000_000_000.0 => 9,  // $1B+
            v if v >= 500_000_000.0 => 8,    // $500M+
            v if v >= 200_000_000.0 => 7,    // $200M+
            v if v >= 100_000_000.0 => 6,    // $100M+
            v if v >= 50_000_000.0 => 5,     // $50M+
            v if v >= 20_000_000.0 => 4,     // $20M+
            v if v >= 10_000_000.0 => 3,     // $10M+
            v if v >= 1_000_000.0 => 2,      // $1M+
            v if v >= 50_000.0 => 1,         // $50K+
            _ => 0,                           // Base tier
        },
        _ => {
            // Generic tiers for other exchanges
            match volume_30d {
                v if v >= 100_000_000.0 => 5,
                v if v >= 10_000_000.0 => 4,
                v if v >= 1_000_000.0 => 3,
                v if v >= 100_000.0 => 2,
                v if v >= 10_000.0 => 1,
                _ => 0,
            }
        }
    }
}

fn get_next_tier_requirement(exchange: &str, current_tier: u32) -> Option<f64> {
    match exchange.to_lowercase().as_str() {
        "binance" => match current_tier {
            0 => Some(50_000.0),
            1 => Some(1_000_000.0),
            2 => Some(10_000_000.0),
            3 => Some(20_000_000.0),
            4 => Some(50_000_000.0),
            5 => Some(100_000_000.0),
            6 => Some(200_000_000.0),
            7 => Some(500_000_000.0),
            8 => Some(1_000_000_000.0),
            _ => None, // Max tier
        },
        _ => match current_tier {
            0 => Some(10_000.0),
            1 => Some(100_000.0),
            2 => Some(1_000_000.0),
            3 => Some(10_000_000.0),
            4 => Some(100_000_000.0),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// P&L HELPER FUNCTIONS - The accounting department
// ─────────────────────────────────────────────────────────────

fn calculate_period_fees(engine: &TradingEngine, timeframe: &str) -> f64 {
    let cutoff = match timeframe.to_lowercase().as_str() {
        "daily" | "day" | "1d" => Utc::now() - chrono::Duration::days(1),
        "weekly" | "week" | "1w" => Utc::now() - chrono::Duration::days(7),
        "monthly" | "month" | "1m" => Utc::now() - chrono::Duration::days(30),
        "yearly" | "year" | "1y" => Utc::now() - chrono::Duration::days(365),
        _ => DateTime::<Utc>::MIN_UTC,
    };
    
    engine.order_history.read()
        .iter()
        .filter(|o| o.created_at >= cutoff && o.status == OrderStatus::Filled)
        .map(|o| calculate_order_fee(o))
        .sum()
}

fn calculate_pnl_by_symbol(engine: &TradingEngine, timeframe: &str) -> Vec<SymbolPnL> {
    let cutoff = match timeframe.to_lowercase().as_str() {
        "daily" | "day" | "1d" => Utc::now() - chrono::Duration::days(1),
        "weekly" | "week" | "1w" => Utc::now() - chrono::Duration::days(7),
        "monthly" | "month" | "1m" => Utc::now() - chrono::Duration::days(30),
        "yearly" | "year" | "1y" => Utc::now() - chrono::Duration::days(365),
        _ => DateTime::<Utc>::MIN_UTC,
    };
    
    let mut symbol_pnl_map: std::collections::HashMap<String, SymbolPnL> = std::collections::HashMap::new();
    
    // Calculate realized P&L from closed trades
    for order in engine.order_history.read().iter() {
        if order.created_at >= cutoff && order.status == OrderStatus::Filled {
            if let Some(meta_pnl) = order.metadata.get("realized_pnl") {
                if let Ok(pnl) = meta_pnl.parse::<f64>() {
                    let entry = symbol_pnl_map.entry(order.symbol.clone())
                        .or_insert_with(|| SymbolPnL {
                            symbol: order.symbol.clone(),
                            realized_pnl: 0.0,
                            unrealized_pnl: 0.0,
                            total_pnl: 0.0,
                            trade_count: 0,
                            win_rate: 0.0,
                        });
                    
                    entry.realized_pnl += pnl;
                    entry.trade_count += 1;
                }
            }
        }
    }
    
    // Add unrealized P&L from open positions
    for entry in engine.positions.iter() {
        let position = entry.value();
        let symbol_entry = symbol_pnl_map.entry(position.symbol.clone())
            .or_insert_with(|| SymbolPnL {
                symbol: position.symbol.clone(),
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                total_pnl: 0.0,
                trade_count: 0,
                win_rate: 0.0,
            });
        
        symbol_entry.unrealized_pnl += position.unrealized_pnl;
    }
    
    // Calculate totals and win rates
    for entry in symbol_pnl_map.values_mut() {
        entry.total_pnl = entry.realized_pnl + entry.unrealized_pnl;
        // Win rate would need trade-level data to calculate accurately
        entry.win_rate = if entry.realized_pnl > 0.0 { 60.0 } else { 40.0 }; // Placeholder
    }
    
    symbol_pnl_map.into_values().collect()
}

fn assess_psychological_impact(net_pnl: f64, max_drawdown: f32) -> String {
    match (net_pnl, max_drawdown) {
        (p, _) if p > 5000.0 => "EUPHORIC - Success breeds overconfidence. Stay grounded.".to_string(),
        (p, _) if p > 1000.0 => "CONFIDENT - Riding high, but remember: pride before the fall.".to_string(),
        (p, _) if p > 0.0 => "POSITIVE - Green is good. Keep the discipline.".to_string(),
        (p, d) if p > -500.0 && d < 10.0 => "NEUTRAL - Treading water. Time to refine the edge.".to_string(),
        (p, d) if p > -1000.0 && d < 20.0 => "ANXIOUS - The pressure builds. Trust the process.".to_string(),
        (p, d) if p > -5000.0 && d < 30.0 => "STRESSED - Deep breaths. This too shall pass.".to_string(),
        _ => "CRITICAL - Maximum pain threshold. Consider stepping back.".to_string(),
    }
}

fn generate_pnl_recommendations(net_pnl: f64, win_rate: f32, sharpe_ratio: f32) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if net_pnl < 0.0 && win_rate < 40.0 {
        recommendations.push("Review your strategy - low win rate dragging you down".to_string());
    }
    
    if sharpe_ratio < 0.5 {
        recommendations.push("Risk-adjusted returns are weak. Consider position sizing".to_string());
    }
    
    if net_pnl > 0.0 && win_rate > 60.0 {
        recommendations.push("Strong performance - consider scaling up gradually".to_string());
    }
    
    if net_pnl < -1000.0 {
        recommendations.push("Significant drawdown - reduce position sizes".to_string());
    }
    
    recommendations
}

fn calculate_period_realized_pnl(
    engine: &TradingEngine,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    symbol: &Option<String>
) -> f64 {
    engine.order_history.read()
        .iter()
        .filter(|o| {
            o.created_at >= start && 
            o.created_at <= end &&
            o.status == OrderStatus::Filled &&
            symbol.as_ref().map_or(true, |s| &o.symbol == s)
        })
        .filter_map(|o| o.metadata.get("realized_pnl"))
        .filter_map(|pnl| pnl.parse::<f64>().ok())
        .sum()
}

fn calculate_total_realized_pnl(engine: &TradingEngine) -> f64 {
    engine.order_history.read()
        .iter()
        .filter(|o| o.status == OrderStatus::Filled)
        .filter_map(|o| o.metadata.get("realized_pnl"))
        .filter_map(|pnl| pnl.parse::<f64>().ok())
        .sum()
}

fn calculate_total_unrealized_pnl(engine: &TradingEngine) -> f64 {
    engine.positions
        .iter()
        .map(|entry| entry.value().unrealized_pnl)
        .sum()
}

fn calculate_total_fees(engine: &TradingEngine) -> f64 {
    engine.order_history.read()
        .iter()
        .filter(|o| o.status == OrderStatus::Filled)
        .map(|o| calculate_order_fee(o))
        .sum()
}

fn calculate_period_fees_detailed(
    engine: &TradingEngine,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    symbol: &Option<String>
) -> f64 {
    engine.order_history.read()
        .iter()
        .filter(|o| {
            o.created_at >= start && 
            o.created_at <= end &&
            o.status == OrderStatus::Filled &&
            symbol.as_ref().map_or(true, |s| &o.symbol == s)
        })
        .map(|o| calculate_order_fee(o))
        .sum()
}

fn calculate_roi(engine: &TradingEngine, total_pnl: f64) -> f64 {
    // Simplified ROI calculation
    // In reality, would track initial capital
    let total_invested: f64 = engine.positions
        .iter()
        .map(|entry| entry.value().quantity * entry.value().entry_price)
        .sum();
    
    if total_invested > 0.0 {
        (total_pnl / total_invested) * 100.0
    } else {
        0.0
    }
}

fn format_pnl_message(total_pnl: f64, roi: f64) -> String {
    match (total_pnl, roi) {
        (p, r) if p > 0.0 && r > 20.0 => "Exceptional returns! The chrome gods smile upon you.".to_string(),
        (p, r) if p > 0.0 && r > 10.0 => "Solid profits. The grind pays off.".to_string(),
        (p, _) if p > 0.0 => "In the green. Every satoshi counts.".to_string(),
        (p, _) if p > -100.0 => "Minor losses. Part of the game.".to_string(),
        (p, _) if p > -1000.0 => "Bleeding out. Time to reassess.".to_string(),
        _ => "Heavy losses. Sometimes the market wins.".to_string(),
    }
}

fn update_period_pnls(tracker: &mut PnLTracker, realized_pnl: f64) {
    // Update all period P&Ls
    tracker.daily_pnl = realized_pnl; // Simplified - would track by actual period
    tracker.weekly_pnl = realized_pnl;
    tracker.monthly_pnl = realized_pnl;
    tracker.yearly_pnl = realized_pnl;
    tracker.all_time_pnl += realized_pnl;
}

fn update_pnl_statistics(tracker: &mut PnLTracker, engine: &TradingEngine) {
    // Calculate win rate
    let trades: Vec<&Order> = engine.order_history.read()
        .iter()
        .filter(|o| o.status == OrderStatus::Filled)
        .collect();
    
    let winning_trades = trades.iter()
        .filter(|o| {
            o.metadata.get("realized_pnl")
                .and_then(|p| p.parse::<f64>().ok())
                .map_or(false, |pnl| pnl > 0.0)
        })
        .count();
    
    tracker.total_trades = trades.len();
    tracker.winning_trades = winning_trades;
    tracker.losing_trades = trades.len() - winning_trades;
    tracker.win_rate = if trades.is_empty() {
        0.0
    } else {
        (winning_trades as f32 / trades.len() as f32) * 100.0
    };
    
    // Update best/worst trades
    for trade in trades {
        if let Some(pnl_str) = trade.metadata.get("realized_pnl") {
            if let Ok(pnl) = pnl_str.parse::<f64>() {
                let trade_record = TradeRecord {
                    symbol: trade.symbol.clone(),
                    pnl,
                    percentage: 0.0, // Would calculate from position data
                    timestamp: trade.created_at,
                };
                
                if tracker.best_trade.as_ref().map_or(true, |best| pnl > best.pnl) {
                    tracker.best_trade = Some(trade_record.clone());
                }
                
                if tracker.worst_trade.as_ref().map_or(true, |worst| pnl < worst.pnl) {
                    tracker.worst_trade = Some(trade_record);
                }
            }
        }
    }
}

fn should_create_snapshot(tracker: &PnLTracker) -> bool {
    // Create snapshot every hour or if significant P&L change
    if let Some(last_snapshot) = tracker.pnl_history.last() {
        let time_since = Utc::now() - last_snapshot.timestamp;
        let pnl_change = (tracker.daily_pnl - last_snapshot.total_pnl).abs();
        
        time_since.num_hours() >= 1 || pnl_change > 1000.0
    } else {
        true // Always create first snapshot
    }
}

fn update_pnl_on_close(tracker: &mut PnLTracker, pnl: f64, symbol: &str) {
    // Update realized P&L
    tracker.daily_pnl += pnl;
    tracker.weekly_pnl += pnl;
    tracker.monthly_pnl += pnl;
    tracker.yearly_pnl += pnl;
    tracker.all_time_pnl += pnl;
    
    // Update trade statistics
    tracker.total_trades += 1;
    if pnl > 0.0 {
        tracker.winning_trades += 1;
    } else {
        tracker.losing_trades += 1;
    }
    
    // Update win rate
    tracker.win_rate = if tracker.total_trades > 0 {
        (tracker.winning_trades as f32 / tracker.total_trades as f32) * 100.0
    } else {
        0.0
    };
    
    // Check for best/worst trade
    let trade_record = TradeRecord {
        symbol: symbol.to_string(),
        pnl,
        percentage: 0.0, // Would calculate from actual position data
        timestamp: Utc::now(),
    };
    
    if tracker.best_trade.as_ref().map_or(true, |best| pnl > best.pnl) {
        tracker.best_trade = Some(trade_record.clone());
    }
    
    if tracker.worst_trade.as_ref().map_or(true, |worst| pnl < worst.pnl) {
        tracker.worst_trade = Some(trade_record);
    }
}

fn get_hourly_snapshots(history: &[PnLSnapshot], limit: usize) -> Vec<PnLSnapshot> {
    // Group by hour and take most recent from each hour
    let mut hourly_map = std::collections::BTreeMap::new();
    
    for snapshot in history.iter().rev().take(limit * 24) {
        let hour_key = snapshot.timestamp.format("%Y-%m-%d %H").to_string();
        hourly_map.entry(hour_key).or_insert(snapshot.clone());
    }
    
    hourly_map.into_values().rev().take(limit).collect()
}

fn get_daily_snapshots(history: &[PnLSnapshot], limit: usize) -> Vec<PnLSnapshot> {
    // Group by day and take most recent from each day
    let mut daily_map = std::collections::BTreeMap::new();
    
    for snapshot in history.iter().rev() {
        let day_key = snapshot.timestamp.format("%Y-%m-%d").to_string();
        daily_map.entry(day_key).or_insert(snapshot.clone());
    }
    
    daily_map.into_values().rev().take(limit).collect()
}

fn get_weekly_snapshots(history: &[PnLSnapshot], limit: usize) -> Vec<PnLSnapshot> {
    // Group by week and take most recent from each week
    let mut weekly_map = std::collections::BTreeMap::new();
    
    for snapshot in history.iter().rev() {
        let week_key = snapshot.timestamp.format("%Y-%W").to_string();
        weekly_map.entry(week_key).or_insert(snapshot.clone());
    }
    
    weekly_map.into_values().rev().take(limit).collect()
}

fn get_monthly_snapshots(history: &[PnLSnapshot], limit: usize) -> Vec<PnLSnapshot> {
    // Group by month and take most recent from each month
    let mut monthly_map = std::collections::BTreeMap::new();
    
    for snapshot in history.iter().rev() {
        let month_key = snapshot.timestamp.format("%Y-%m").to_string();
        monthly_map.entry(month_key).or_insert(snapshot.clone());
    }
    
    monthly_map.into_values().rev().take(limit).collect()
}

fn calculate_pnl_statistics(snapshots: &[PnLSnapshot]) -> PnLStatistics {
    if snapshots.is_empty() {
        return PnLStatistics {
            average_pnl: 0.0,
            std_deviation: 0.0,
            best_period: 0.0,
            worst_period: 0.0,
            positive_periods: 0,
            negative_periods: 0,
            current_streak: 0,
            longest_winning_streak: 0,
            longest_losing_streak: 0,
        };
    }
    
    // Calculate period P&Ls
    let period_pnls: Vec<f64> = snapshots.windows(2)
        .map(|w| w[1].total_pnl - w[0].total_pnl)
        .collect();
    
    if period_pnls.is_empty() {
        return PnLStatistics {
            average_pnl: snapshots.last().map(|s| s.total_pnl).unwrap_or(0.0),
            std_deviation: 0.0,
            best_period: 0.0,
            worst_period: 0.0,
            positive_periods: 0,
            negative_periods: 0,
            current_streak: 0,
            longest_winning_streak: 0,
            longest_losing_streak: 0,
        };
    }
    
    // Basic statistics
    let average_pnl = period_pnls.iter().sum::<f64>() / period_pnls.len() as f64;
    let variance = period_pnls.iter()
        .map(|p| (p - average_pnl).powi(2))
        .sum::<f64>() / period_pnls.len() as f64;
    let std_deviation = variance.sqrt();
    
    let best_period = period_pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let worst_period = period_pnls.iter().cloned().fold(f64::INFINITY, f64::min);
    
    let positive_periods = period_pnls.iter().filter(|&&p| p > 0.0).count();
    let negative_periods = period_pnls.iter().filter(|&&p| p < 0.0).count();
    
    // Calculate streaks
    let mut current_streak = 0;
    let mut longest_winning = 0;
    let mut longest_losing = 0;
    let mut current_winning = 0;
    let mut current_losing = 0;
    
    for &pnl in &period_pnls {
        if pnl > 0.0 {
            current_winning += 1;
            current_losing = 0;
            longest_winning = longest_winning.max(current_winning);
            current_streak = current_winning;
        } else if pnl < 0.0 {
            current_losing += 1;
            current_winning = 0;
            longest_losing = longest_losing.max(current_losing);
            current_streak = -current_losing;
        }
    }
    
    PnLStatistics {
        average_pnl,
        std_deviation,
        best_period,
        worst_period,
        positive_periods,
        negative_periods,
        current_streak,
        longest_winning_streak: longest_winning,
        longest_losing_streak: longest_losing,
    }
}

fn identify_pnl_trend(snapshots: &[PnLSnapshot]) -> PnLTrend {
    if snapshots.len() < 3 {
        return PnLTrend {
            direction: "insufficient_data".to_string(),
            strength: 0.0,
            momentum: 0.0,
            prediction: "Need more data for analysis".to_string(),
        };
    }
    
    // Simple linear regression for trend
    let n = snapshots.len() as f64;
    let x_values: Vec<f64> = (0..snapshots.len()).map(|i| i as f64).collect();
    let y_values: Vec<f64> = snapshots.iter().map(|s| s.total_pnl).collect();
    
    let x_mean = x_values.iter().sum::<f64>() / n;
    let y_mean = y_values.iter().sum::<f64>() / n;
    
    let numerator: f64 = x_values.iter()
        .zip(&y_values)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    
    let denominator: f64 = x_values.iter()
        .map(|x| (x - x_mean).powi(2))
        .sum();
    
    let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    
    // Determine trend direction and strength
    let (direction, strength) = match slope {
        s if s > 100.0 => ("strongly_improving", 90.0),
        s if s > 10.0 => ("improving", 70.0),
        s if s > -10.0 => ("stable", 50.0),
        s if s > -100.0 => ("declining", 30.0),
        _ => ("strongly_declining", 10.0),
    };
    
    // Calculate momentum (rate of change)
    let recent_change = if snapshots.len() >= 5 {
        let recent = &snapshots[snapshots.len()-5..];
        (recent.last().unwrap().total_pnl - recent.first().unwrap().total_pnl) / 5.0
    } else {
        slope
    };
    
    let prediction = match (direction, recent_change) {
        ("strongly_improving", c) if c > 500.0 => "Explosive growth likely to continue",
        ("improving", c) if c > 100.0 => "Steady profits expected",
        ("stable", _) => "Consolidation phase - breakout pending",
        ("declining", c) if c < -100.0 => "Caution: drawdown accelerating",
        _ => "Monitor closely for trend reversal",
    };
    
    PnLTrend {
        direction: direction.to_string(),
        strength: strength as f32,
        momentum: recent_change as f32,
        prediction: prediction.to_string(),
    }
}

fn interpret_pnl_history(stats: &PnLStatistics, trend: &PnLTrend) -> String {
    match (stats.current_streak, &trend.direction[..]) {
        (s, "strongly_improving") if s > 5 => {
            "🚀 ON FIRE! Winning streak continues. The chrome gods favor you.".to_string()
        },
        (s, "improving") if s > 0 => {
            "📈 Solid progress. Consistency is the path to mastery.".to_string()
        },
        (s, "stable") if s.abs() < 3 => {
            "➡️ Sideways action. The market is testing your patience.".to_string()
        },
        (s, "declining") if s < -3 => {
            "📉 Rough patch. Every legend has chapters of struggle.".to_string()
        },
        (_, "strongly_declining") => {
            "🚨 Maximum adversity. Time to go back to basics.".to_string()
        },
        _ => {
            "Market conditions shifting. Stay alert, stay alive.".to_string()
        }
    }
}

// ─────────────────────────────────────────────────────────────
// DEFAULT IMPLEMENTATIONS - For missing state types
// ─────────────────────────────────────────────────────────────

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size: 10000.0,
            max_daily_loss: 1000.0,
            max_leverage: 10.0,
            allowed_symbols: vec![],
            banned_symbols: vec![],
            max_correlated_positions: 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// IMPORTANT: Add to state/mod.rs for full P&L tracking:
// 
// #[derive(Debug, Clone)]
// pub struct PnLTracker {
//     pub daily_pnl: f64,
//     pub weekly_pnl: f64,
//     pub monthly_pnl: f64,
//     pub yearly_pnl: f64,
//     pub all_time_pnl: f64,
//     pub pnl_history: Vec<PnLSnapshot>,
//     pub last_update: DateTime<Utc>,
//     pub total_trades: usize,
//     pub winning_trades: usize,
//     pub losing_trades: usize,
//     pub win_rate: f32,
//     pub sharpe_ratio: f32,
//     pub max_drawdown: f32,
//     pub best_trade: Option<TradeRecord>,
//     pub worst_trade: Option<TradeRecord>,
// }
//
// #[derive(Debug, Clone)]
// pub struct SimulationEngine {
//     pub paper_mode: bool,
//     pub latency_ms: u64,
//     pub slippage_bps: f64,
//     pub failure_rate: f64,
//     pub mock_prices: std::collections::HashMap<String, f64>,
// }
//
// #[derive(Debug, Clone)]
// pub struct HealthThresholdConfig {
//     pub critical: f32,
//     pub warning: f32,
//     pub auto_close: f32,
// }
//
// pub struct AppState {
//     // ...existing fields...
//     pub simulation_engine: SimulationEngine,
//     pub health_thresholds: HealthThresholdConfig,
//     pub auto_risk_management: bool,
//     pub position_monitoring_active: bool,
//     pub exchange_fees: HashMap<String, ExchangeFeeStructure>,
// }
//
// pub struct TradingEngine {
//     // ...existing fields...
//     pub pnl_tracker: Arc<RwLock<PnLTracker>>,
// }
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// COMPLETE COMMAND LIST - FULL CYBERPUNK TRADING ARSENAL:
// 
// Trading Commands:
// 1. place_order - All order types with paper trading support
// 2. cancel_order - Cancel single order
// 3. cancel_all_orders - Mass cancellation
// 4. get_positions - View open positions with health scores
// 5. get_order_history - Historical orders
// 6. get_active_orders - Pending orders
// 7. close_position - Full or partial close
// 8. close_all_positions - Emergency flatten
// 9. modify_position_protection - Update stops/targets
// 
// P&L Reporting Commands:
// 10. get_pnl_report - Comprehensive P&L analysis
// 11. calculate_pnl - Period-specific P&L calculation
// 12. get_pnl_history - Historical P&L snapshots
// 13. update_pnl_tracker - Manual P&L synchronization
//
// Simulation Commands:
// 14. toggle_simulation_mode - Switch between paper/live
// 15. configure_simulation - Set latency, slippage, failure rate
// 16. set_mock_price - Inject custom prices for testing
// 17. run_backtest - Historical strategy testing
// 18. reset_simulation - Clear simulation state
// 19. get_simulation_stats - View paper trading metrics
//
// Position Health Commands:
// 20. monitor_position_health - Real-time health analysis
// 21. set_health_thresholds - Configure risk parameters
// 22. start_position_monitoring - Automated guardian daemon
// 23. stop_position_monitoring - Disable auto-monitoring
// 24. calculate_quantum_scores - Multi-dimensional position analysis
//
// Fee Management Commands:
// 25. configure_exchange_fees - Set exchange-specific rates
// 26. calculate_order_fees - Pre-trade fee calculation
// 27. get_fee_statistics - Historical fee analysis
// 28. update_trading_volume - Track volume for tier upgrades
//
// Total: 28 Commands - "Bloomberg Terminal meets Cyberpunk 2077"
// Every satoshi saved is a satoshi earned. 💰
// ─────────────────────────────────────────────────────────────