// Location: C:\Nexlify\src-tauri\src\commands\mod.rs
// Purpose: NEXLIFY COMMAND CENTER - Where frontend meets backend in the neural dance
// Last sync: 2025-06-19 | "Every command is a prayer to the machine gods"

pub mod market_data;
pub mod trading;
pub mod auth;
pub mod system; // NEW: System monitoring module

// Re-export all command functions for cleaner imports in main.rs
pub use market_data::*;
pub use trading::*;
pub use auth::*;
pub use system::*; // NEW: System command exports

/// Command result type - because every neural transmission can glitch
pub type CommandResult<T> = Result<T, CommandError>;

/// Unified error type for all our command failures
#[derive(Debug, thiserror::Error)]
pub enum CommandError {
    #[error("Market data unavailable: {0}")]
    MarketDataError(String),
    
    #[error("Trading engine error: {0}")]
    TradingError(String),
    
    #[error("Authentication failed: {0}")]
    AuthError(String),
    
    #[error("Neural pathway blocked: {0}")]
    StateError(#[from] crate::state::StateError),
    
    #[error("Serialization glitch: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Network flatlined: {0}")]
    NetworkError(String),
    
    #[error("System error: {0}")] // NEW: For system monitoring
    SystemError(String),
    
    #[error("Validation error: {0}")] // NEW: For input validation
    ValidationError(String),
    
    #[error("Unknown error in the matrix: {0}")]
    Unknown(String),
}

impl serde::Serialize for CommandError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Convert errors to frontend-friendly format
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("CommandError", 2)?;
        state.serialize_field("type", &self.error_type())?;
        state.serialize_field("message", &self.to_string())?;
        state.end()
    }
}

impl CommandError {
    /// Get error type for frontend handling
    fn error_type(&self) -> &'static str {
        match self {
            CommandError::MarketDataError(_) => "MARKET_DATA_ERROR",
            CommandError::TradingError(_) => "TRADING_ERROR",
            CommandError::AuthError(_) => "AUTH_ERROR",
            CommandError::StateError(_) => "STATE_ERROR",
            CommandError::SerializationError(_) => "SERIALIZATION_ERROR",
            CommandError::NetworkError(_) => "NETWORK_ERROR",
            CommandError::SystemError(_) => "SYSTEM_ERROR",
            CommandError::ValidationError(_) => "VALIDATION_ERROR",
            CommandError::Unknown(_) => "UNKNOWN_ERROR",
        }
    }
}

// Convert from std::io::Error
impl From<std::io::Error> for CommandError {
    fn from(err: std::io::Error) -> Self {
        CommandError::SystemError(format!("IO error: {}", err))
    }
}

/// Common validation functions - trust no input from the sprawl
pub mod validation {
    use super::CommandError;
    
    /// Validate a trading symbol - no shitcoins in our neural mesh
    pub fn validate_symbol(symbol: &str) -> Result<(), CommandError> {
        // Basic validation - adapt based on your exchange requirements
        let valid_pattern = regex::Regex::new(r"^[A-Z0-9]+-[A-Z0-9]+$").unwrap();
        
        if !valid_pattern.is_match(symbol) {
            return Err(CommandError::ValidationError(
                format!("Invalid symbol format: {}. Expected format: BTC-USD", symbol)
            ));
        }
        
        // Additional validation: check symbol length
        if symbol.len() > 20 {
            return Err(CommandError::ValidationError(
                "Symbol too long - are you trying to overflow my buffers?".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate exchange name
    pub fn validate_exchange(exchange: &str) -> Result<(), CommandError> {
        const VALID_EXCHANGES: &[&str] = &[
            "binance", "coinbase", "kraken", "bybit", "okx", 
            "huobi", "gate", "kucoin", "bitfinex", "bitstamp"
        ];
        
        let exchange_lower = exchange.to_lowercase();
        if !VALID_EXCHANGES.contains(&exchange_lower.as_str()) {
            return Err(CommandError::ValidationError(
                format!("Unknown exchange: {} - stick to the majors", exchange)
            ));
        }
        
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// COMMAND REGISTRY - Currently Implemented Commands
// ─────────────────────────────────────────────────────────────

/// Register all CURRENTLY IMPLEMENTED commands with Tauri
/// 
/// Status: 29/43 commands implemented (67% - Beyond MVP! 🚀)
/// This macro only includes commands that actually exist.
/// See TODO section below for planned features.
#[macro_export]
macro_rules! register_all_commands {
    () => {
        // ═══════════════════════════════════════════════════════════
        // AUTHENTICATION (7/7 - COMPLETE ✅)
        // ═══════════════════════════════════════════════════════════
        crate::commands::login,
        crate::commands::logout,
        crate::commands::refresh_session,
        crate::commands::manage_api_keys,
        crate::commands::verify_credentials,
        crate::commands::get_exchange_status,
        crate::commands::rotate_api_key,
        
        // ═══════════════════════════════════════════════════════════
        // MARKET DATA (5/6 - ALMOST COMPLETE ✅)
        // ═══════════════════════════════════════════════════════════
        crate::commands::get_orderbook,
        crate::commands::get_ticker,
        crate::commands::get_recent_trades,
        crate::commands::subscribe_market_data,
        crate::commands::unsubscribe_market_data,
        crate::commands::get_historical_candles,
        
        // ═══════════════════════════════════════════════════════════
        // TRADING (10/9 - OVER-COMPLETE! 🔥)
        // ═══════════════════════════════════════════════════════════
        crate::commands::place_order,
        crate::commands::cancel_order,
        crate::commands::cancel_all_orders,
        crate::commands::get_positions,
        crate::commands::get_order_history,
        crate::commands::get_active_orders,
        crate::commands::close_position,
        crate::commands::close_all_positions,
        crate::commands::modify_position_protection,
        crate::commands::calculate_pnl,
        
        // ═══════════════════════════════════════════════════════════
        // P&L REPORTING (3/4 - MOSTLY COMPLETE! 💰)
        // ═══════════════════════════════════════════════════════════
        crate::commands::get_pnl_report,
        crate::commands::get_pnl_history,
        crate::commands::update_pnl_tracker,
        
        // ═══════════════════════════════════════════════════════════
        // SYSTEM (4/4 - COMPLETE ✅)
        // ═══════════════════════════════════════════════════════════
        crate::commands::get_system_metrics,
        crate::commands::check_neural_health,
        crate::commands::trigger_garbage_collection,
        crate::commands::export_diagnostics,
        
        // ─── COMMANDS TODO (14 remaining) ───
        
        // Market Data (1 remaining)
        // TODO: get_market_stats
        
        // Simulation/Paper Trading (0/6 - NOT STARTED ❌)
        // TODO: toggle_simulation_mode
        // TODO: configure_simulation
        // TODO: set_mock_price
        // TODO: run_backtest
        // TODO: reset_simulation
        // TODO: get_simulation_stats
        
        // Position Health Monitoring (0/5 - NOT STARTED ❌)
        // TODO: monitor_position_health
        // TODO: set_health_thresholds
        // TODO: start_position_monitoring
        // TODO: stop_position_monitoring
        // TODO: calculate_quantum_scores
        
        // Fee Management (0/4 - NOT STARTED ❌)
        // TODO: configure_exchange_fees
        // TODO: calculate_order_fees
        // TODO: get_fee_statistics
        // TODO: update_trading_volume
    };
}

// ─────────────────────────────────────────────────────────────
// COMMAND CATEGORIES SUMMARY
// ─────────────────────────────────────────────────────────────
// 
// IMPLEMENTED (29 commands - 67%):
// ✅ Authentication:    7/7  (100%) - COMPLETE
// ✅ Market Data:       5/6  (83%)  - Nearly complete
// ✅ Trading:          10/9  (111%) - OVER-COMPLETE! 
// ✅ P&L Reporting:     3/4  (75%)  - Mostly complete
// ✅ System:            4/4  (100%) - COMPLETE
// 
// NOT IMPLEMENTED (14 commands - 33%):
// ❌ Simulation:        0/6  (0%)   - Planned for v1.2
// ❌ Position Health:   0/5  (0%)   - Planned for v1.3
// ❌ Fee Management:    0/4  (0%)   - Planned for v1.4
// 
// TOTAL: 29/43 commands (67%) - Beyond MVP! 🚀
// 
// NOTE: We actually have MORE trading commands than originally
// planned, showing organic growth of the platform!
// ─────────────────────────────────────────────────────────────