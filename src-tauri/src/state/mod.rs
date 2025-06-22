// src-tauri/src/state/mod.rs
// NEXLIFY STATE MANAGEMENT - The collective consciousness
// Last sync: 2025-06-19 | "Data wants to be free, but the market wants to eat you alive"

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use arc_swap::ArcSwap;
use thiserror::Error;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

pub mod market_cache;
pub use market_cache::MarketCache;

/// Neural mesh errors - when the matrix glitches
#[derive(Error, Debug)]
pub enum StateError {
    #[error("Neural pathway blocked: {0}")]
    PathwayBlocked(String),
    
    #[error("Cache overflow - too much chrome: {0}")]
    CacheOverflow(String),
    
    #[error("Trading engine flatlined: {0}")]
    EngineFailure(String),
    
    #[error("Authentication breach detected: {0}")]
    AuthBreach(String),
}

/// Main application state - the beating heart of our trading terminal
#[derive(Debug)]
pub struct AppState {
    /// System boot time - when we jacked in
    boot_time: DateTime<Utc>,
    
    /// Active market connections - our neural links to the exchanges
    active_connections: DashMap<String, ConnectionStatus>,
    
    /// Performance metrics - keeping our chrome optimized
    metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// User preferences - how they like their matrix configured
    preferences: ArcSwap<UserPreferences>,
    
    /// Circuit breakers - for when the market tries to flatline us
    circuit_breakers: DashMap<String, CircuitBreaker>,
}

// ─────────────────────────────────────────────────────────────
// P&L TRACKING TYPES
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

#[derive(Debug, Clone)]
pub struct PnLTracker {
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
    pub yearly_pnl: f64,
    pub all_time_pnl: f64,
    pub pnl_history: Vec<PnLSnapshot>,
    pub last_update: DateTime<Utc>,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
    pub best_trade: Option<TradeRecord>,
    pub worst_trade: Option<TradeRecord>,
}

impl Default for PnLTracker {
    fn default() -> Self {
        Self {
            daily_pnl: 0.0,
            weekly_pnl: 0.0,
            monthly_pnl: 0.0,
            yearly_pnl: 0.0,
            all_time_pnl: 0.0,
            pnl_history: Vec::new(),
            last_update: Utc::now(),
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            best_trade: None,
            worst_trade: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// SIMULATION ENGINE
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SimulationEngine {
    pub paper_mode: bool,
    pub latency_ms: u64,
    pub slippage_bps: f64,
    pub failure_rate: f64,
    pub mock_prices: HashMap<String, f64>,
}

impl Default for SimulationEngine {
    fn default() -> Self {
        Self {
            paper_mode: true,
            latency_ms: 50,
            slippage_bps: 10.0, // 0.1%
            failure_rate: 0.02, // 2% failure rate
            mock_prices: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// HEALTH MONITORING
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HealthThresholdConfig {
    pub critical: f32,
    pub warning: f32,
    pub auto_close: f32,
}

impl Default for HealthThresholdConfig {
    fn default() -> Self {
        Self {
            critical: 20.0,
            warning: 40.0,
            auto_close: 10.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// FEE MANAGEMENT
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExchangeFeeStructure {
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub volume_tier: u32,
    pub has_native_token_discount: bool,
    pub volume_30d: f64,
}

// ─────────────────────────────────────────────────────────────
// UPDATE EXISTING STRUCTURES
// ─────────────────────────────────────────────────────────────

// Add these fields to your existing AppState struct:
pub struct AppState {
    // ...existing fields...
    
    // Add these new fields:
    pub simulation_engine: SimulationEngine,
    pub health_thresholds: HealthThresholdConfig,
    pub auto_risk_management: bool,
    pub position_monitoring_active: bool,
    pub exchange_fees: HashMap<String, ExchangeFeeStructure>,
}

// Add this field to your existing TradingEngine struct:
pub struct TradingEngine {
    // ...existing fields...
    
    // Add this new field:
    pub pnl_tracker: Arc<RwLock<PnLTracker>>,
}

// In your AppState implementation, update the new() or default() method:
impl AppState {
    pub fn new() -> Self {
        Self {
            // ...existing field initialization...
            
            // Add these:
            simulation_engine: SimulationEngine::default(),
            health_thresholds: HealthThresholdConfig::default(),
            auto_risk_management: false,
            position_monitoring_active: false,
            exchange_fees: HashMap::new(),
        }
    }
}

// In your TradingEngine implementation, update the new() or default() method:
impl TradingEngine {
    pub fn new() -> Self {
        Self {
            // ...existing field initialization...
            
            // Add this:
            pnl_tracker: Arc::new(RwLock::new(PnLTracker::default())),
        }
    }
}

/// Connection status to various market feeds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub exchange: String,
    pub connected: bool,
    pub latency_ms: f64,
    pub last_heartbeat: DateTime<Utc>,
    pub message_count: u64,
    pub error_count: u32,
}

/// Performance tracking - gotta stay fast in the sprawl
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub ws_messages_per_sec: f32,
    pub orders_per_sec: f32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub gc_cycles: u32,
    pub neural_load: f32, // Custom metric for our cyberpunk aesthetic
}

/// User preferences - their personal matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub theme: String, // "neon-blood", "chrome-noir", "matrix-green"
    pub chart_type: ChartType,
    pub notification_level: NotificationLevel,
    pub hotkeys: HashMap<String, String>,
    pub default_leverage: f32,
    pub risk_limit: f64,
    pub neural_assist: bool, // AI-powered trading hints
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    CandleStick,
    HeikinAshi,
    Renko,
    PointAndFigure,
    NeuralFlow, // Our custom cyberpunk chart
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationLevel {
    Silent,     // Ghost in the machine
    Critical,   // Only when shit hits the fan
    Normal,     // Standard alerts
    Verbose,    // Every little signal
}

/// Circuit breaker - prevents cascade failures when the market goes haywire
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub symbol: String,
    pub triggered: bool,
    pub trigger_count: u32,
    pub last_trigger: Option<DateTime<Utc>>,
    pub cooldown_until: Option<DateTime<Utc>>,
    pub threshold: CircuitThreshold,
}

#[derive(Debug, Clone)]
pub struct CircuitThreshold {
    pub max_loss_percent: f32,
    pub max_orders_per_minute: u32,
    pub max_position_size: f64,
}

impl AppState {
    /// Initialize the neural mesh
    pub fn new() -> Self {
        Self {
            boot_time: Utc::now(),
            active_connections: DashMap::new(),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            preferences: ArcSwap::from_pointee(UserPreferences::default()),
            circuit_breakers: DashMap::new(),
        }
    }
    
    /// How long we've been running in the sprawl
    pub fn get_uptime(&self) -> u64 {
        (Utc::now() - self.boot_time).num_seconds() as u64
    }
    
    /// Current memory footprint - RTX 2070 constraints, remember?
    pub fn get_memory_usage(&self) -> f32 {
        self.metrics.read().memory_usage
    }
    
    /// CPU burn rate - are we overclocking?
    pub fn get_cpu_usage(&self) -> f32 {
        self.metrics.read().cpu_usage
    }
    
    /// Active neural pathways to exchanges
    pub fn get_active_stream_count(&self) -> usize {
        self.active_connections
            .iter()
            .filter(|entry| entry.value().connected)
            .count()
    }
    
    /// Cache size in the neural matrix
    pub fn get_cache_size(&self) -> usize {
        // This would connect to MarketCache
        0 // Placeholder
    }
    
    /// Update connection status - keeping tabs on our links
    pub fn update_connection(&self, exchange: String, status: ConnectionStatus) {
        self.active_connections.insert(exchange, status);
    }
    
    /// Check if a circuit breaker should trip
    pub fn check_circuit_breaker(&self, symbol: &str, order_value: f64) -> Result<(), StateError> {
        if let Some(breaker) = self.circuit_breakers.get(symbol) {
            if breaker.triggered {
                if let Some(cooldown) = breaker.cooldown_until {
                    if Utc::now() < cooldown {
                        return Err(StateError::PathwayBlocked(
                            format!("Circuit breaker active for {} - cooldown until {}", symbol, cooldown)
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for UserPreferences {
    fn default() -> Self {
        let mut hotkeys = HashMap::new();
        hotkeys.insert("quick_buy".to_string(), "Ctrl+B".to_string());
        hotkeys.insert("quick_sell".to_string(), "Ctrl+S".to_string());
        hotkeys.insert("panic_close".to_string(), "Ctrl+Shift+X".to_string());
        
        Self {
            theme: "chrome-noir".to_string(),
            chart_type: ChartType::CandleStick,
            notification_level: NotificationLevel::Normal,
            hotkeys,
            default_leverage: 1.0,
            risk_limit: 1000.0,
            neural_assist: true,
			id: String::new(),
            symbol: String::new(),
            exchange: String::new(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 0.0,
            price: None,
            stop_price: None,
            position_id: None, // NEW
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// Trading engine state - where the real chrome lives
#[derive(Debug)]
pub struct TradingEngine {
    /// Active orders mapped by order ID
    pub active_orders: DashMap<String, Order>,
    
    /// Position tracking across all exchanges
    pub positions: DashMap<String, Position>,
    
    /// Order history for analysis
    pub order_history: Arc<RwLock<Vec<Order>>>,
    
    /// Risk management parameters
    pub risk_params: Arc<RwLock<RiskParameters>>,
    
    /// PnL tracking - are we winning, son?
    pub pnl_tracker: Arc<RwLock<PnLTracker>>,
}

/// Order representation - the bullets in our trading gun
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: Option<f64>,
	pub stop_price: Option<f64>,
    pub quantity: f64,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_quantity: f64,
    pub average_fill_price: Option<f64>,
    pub metadata: HashMap<String, String>, // For that extra chrome
	pub position_id: Option<String>, // NEW: Links position-based orders
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
// Basic order types
    Market,
    Limit,
    
    // Standalone stop orders (for entering/exiting positions)
    Stop,        // Triggers market order at stop price
    StopLimit,   // Triggers limit order at stop price
    
    // Position-based protection (attached to existing positions)
    StopLoss,    // Protective stop for existing position
    TakeProfit,  // Profit target for existing position
    
    // Advanced order types
    IcebergOrder, // Hidden size orders
    NeuralOrder,  // AI-suggested orders
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Position tracking - what we're holding in the digital vault
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub exchange: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub margin_used: f64,
    pub opened_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

/// Risk management - keeping us from flatlining
#[derive(Debug, Clone)]
pub struct RiskParameters {
    pub max_position_size: f64,
    pub max_daily_loss: f64,
    pub max_leverage: f32,
    pub allowed_symbols: Vec<String>,
    pub banned_symbols: Vec<String>,
    pub max_correlated_positions: u32,
}

/// PnL tracking - the scorecard in our digital warfare
#[derive(Debug, Default)]
pub struct PnLTracker {
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
    pub all_time_pnl: f64,
    pub best_trade: Option<(String, f64)>,
    pub worst_trade: Option<(String, f64)>,
    pub win_rate: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown: f32,
}

impl TradingEngine {
    pub fn new() -> Self {
        Self {
            active_orders: DashMap::new(),
            positions: DashMap::new(),
            order_history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            risk_params: Arc::new(RwLock::new(RiskParameters::default())),
            pnl_tracker: Arc::new(RwLock::new(PnLTracker::default())),
        }
    }
    
	
	
    /// Place a new order - firing into the market matrix
    pub fn place_order(&self, order: Order) -> Result<String, StateError> {
        // Risk checks first - don't be a gonk
        let risk_params = self.risk_params.read();
        
        if order.quantity * order.price.unwrap_or(0.0) > risk_params.max_position_size {
            return Err(StateError::EngineFailure(
                "Order exceeds maximum position size - neural safety engaged".to_string()
            ));
        }
        
        if risk_params.banned_symbols.contains(&order.symbol) {
            return Err(StateError::EngineFailure(
                format!("Symbol {} is on the banned list - too dangerous for our chrome", order.symbol)
            ));
        }
        
        let order_id = order.id.clone();
        self.active_orders.insert(order_id.clone(), order);
        
        Ok(order_id)
    }
    
    /// Cancel an order - pulling back from the edge
    pub fn cancel_order(&self, order_id: &str) -> Result<(), StateError> {
        if let Some((_, mut order)) = self.active_orders.remove(order_id) {
            order.status = OrderStatus::Cancelled;
            order.updated_at = Utc::now();
            
            // Add to history
            self.order_history.write().push(order);
            Ok(())
        } else {
            Err(StateError::EngineFailure(
                format!("Order {} not found in the neural mesh", order_id)
            ))
        }
    }
    
    /// Update PnL - counting our wins and losses in the sprawl
    pub fn update_pnl(&self, symbol: &str, pnl: f64) {
        let mut tracker = self.pnl_tracker.write();
        tracker.daily_pnl += pnl;
        tracker.all_time_pnl += pnl;
        
        // Track best/worst trades
        if pnl > 0.0 {
            if tracker.best_trade.as_ref().map_or(true, |(_, best)| pnl > *best) {
                tracker.best_trade = Some((symbol.to_string(), pnl));
            }
        } else {
            if tracker.worst_trade.as_ref().map_or(true, |(_, worst)| pnl < *worst) {
                tracker.worst_trade = Some((symbol.to_string(), pnl));
            }
        }
    }
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size: 10000.0,
            max_daily_loss: 1000.0,
            max_leverage: 3.0,
            allowed_symbols: vec![
                "BTC-USD".to_string(),
                "ETH-USD".to_string(),
                "SOL-USD".to_string(),
            ],
            banned_symbols: vec![], // No shitcoins in our neural mesh
            max_correlated_positions: 3,
        }
    }
}
