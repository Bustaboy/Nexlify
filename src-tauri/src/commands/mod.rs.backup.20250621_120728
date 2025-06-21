// src-tauri/src/commands/mod.rs
// NEXLIFY COMMAND CENTER - Where frontend meets backend in the neural dance
// Last sync: 2025-06-19 | "Every command is a prayer to the machine gods"

pub mod market_data;
pub mod trading;
pub mod auth;

// Re-export all command functions for cleaner imports in main.rs

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
            CommandError::Unknown(_) => "UNKNOWN_ERROR",
        }
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
}
