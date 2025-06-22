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
            return Err(CommandError::TradingError(
                format!("Invalid symbol format: {} - use format like BTC-USD", symbol)
            ));
        }
        
        // Length check - reasonable limits
        if symbol.len() < 5 || symbol.len() > 20 {
            return Err(CommandError::TradingError(
                "Symbol length out of bounds - are you trying to hack us?".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate order quantity - keeping it real
    pub fn validate_quantity(quantity: f64) -> Result<(), CommandError> {
        if quantity <= 0.0 {
            return Err(CommandError::TradingError(
                "Quantity must be positive - can't trade negative chrome".to_string()
            ));
        }
        
        if quantity.is_infinite() || quantity.is_nan() {
            return Err(CommandError::TradingError(
                "Quantity must be a real number - no infinity stones here".to_string()
            ));
        }
        
        // Max quantity check - prevent fat finger disasters
        if quantity > 1_000_000.0 {
            return Err(CommandError::TradingError(
                "Quantity exceeds maximum - even whales have limits".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate price - keeping the market honest
    pub fn validate_price(price: f64) -> Result<(), CommandError> {
        if price <= 0.0 {
            return Err(CommandError::TradingError(
                "Price must be positive - nothing's free in Night City".to_string()
            ));
        }
        
        if price.is_infinite() || price.is_nan() {
            return Err(CommandError::TradingError(
                "Price must be finite - we're not trading in parallel dimensions".to_string()
            ));
        }
        
        // Sanity check - prevent obvious mistakes
        if price > 10_000_000.0 {
            return Err(CommandError::TradingError(
                "Price exceeds reality check - double check your decimals".to_string()
            ));
        }
        
        Ok(())
    }
    
    // NEW: Additional validation functions for completeness
    
    /// Validate API key format
    pub fn validate_api_key(key: &str) -> Result<(), CommandError> {
        if key.len() < 32 {
            return Err(CommandError::ValidationError(
                "API key too short - real keys have substance".to_string()
            ));
        }
        
        if key.len() > 256 {
            return Err(CommandError::ValidationError(
                "API key too long - you trying to hack the matrix?".to_string()
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
// COMMAND REGISTRY EXTENSION - For the complete command list
// ─────────────────────────────────────────────────────────────

/// Helper to register all commands with Tauri
/// This macro includes all 43 commands from the original design
#[macro_export]
macro_rules! register_all_commands {
    () => {
        // Authentication (5)
        crate::commands::login,
        crate::commands::logout,
        crate::commands::refresh_session,
        crate::commands::manage_api_keys,
        crate::commands::rotate_api_key,
        
        // Market Data (6)
        crate::commands::get_orderbook,
        crate::commands::get_ticker,
        crate::commands::get_recent_trades,
        crate::commands::subscribe_market_data,
        crate::commands::unsubscribe_market_data,
        crate::commands::get_historical_data,
        
        // Core Trading (9)
        crate::commands::place_order,
        crate::commands::cancel_order,
        crate::commands::cancel_all_orders,
        crate::commands::get_positions,
        crate::commands::get_order_history,
        crate::commands::get_active_orders,
        crate::commands::close_position,
        crate::commands::close_all_positions,
        crate::commands::modify_position_protection,
        
        // P&L Reporting (4)
        crate::commands::get_pnl_report,
        crate::commands::calculate_pnl,
        crate::commands::get_pnl_history,
        crate::commands::update_pnl_tracker,
        
        // Simulation (6)
        crate::commands::toggle_simulation_mode,
        crate::commands::configure_simulation,
        crate::commands::set_mock_price,
        crate::commands::run_backtest,
        crate::commands::reset_simulation,
        crate::commands::get_simulation_stats,
        
        // Position Health (5)
        crate::commands::monitor_position_health,
        crate::commands::set_health_thresholds,
        crate::commands::start_position_monitoring,
        crate::commands::stop_position_monitoring,
        crate::commands::calculate_quantum_scores,
        
        // Fee Management (4)
        crate::commands::configure_exchange_fees,
        crate::commands::calculate_order_fees,
        crate::commands::get_fee_statistics,
        crate::commands::update_trading_volume,
        
        // System Commands (4)
        crate::commands::get_system_metrics,
        crate::commands::check_neural_health,
        crate::commands::trigger_garbage_collection,
        crate::commands::export_diagnostics
    };
}

// Note: The following are kept for compatibility with the enhanced version,
// but commented out to maintain exact original compatibility:
/*
/// Get all command names for debugging
pub fn get_all_commands() -> Vec<&'static str> { ... }

/// Command statistics
pub fn get_command_stats() -> CommandStats { ... }

/// Command aliases for backward compatibility
pub mod auth_aliases { ... }
pub mod market_aliases { ... }
*/
