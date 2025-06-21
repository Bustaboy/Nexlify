// src-tauri/src/websocket/mod.rs
// WebSocket module for real-time market data feeds

pub mod market_feed;

pub use market_feed::{MarketFeedManager, MarketSubscription, MarketUpdate};
