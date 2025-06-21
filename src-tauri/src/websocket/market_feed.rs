// src-tauri/src/websocket/market_feed.rs
// WebSocket market data feed handler with proper error handling and reconnection logic

use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{AppHandle, Emitter};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSubscription {
    pub symbol: String,
    pub data_types: Vec<String>,
    pub exchange: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MarketUpdate {
    pub symbol: String,
    pub data_type: String,
    pub data: serde_json::Value,
    pub timestamp: i64,
}

pub struct MarketFeedManager {
    app_handle: AppHandle,
    active_subscriptions: Arc<RwLock<Vec<MarketSubscription>>>,
    is_connected: Arc<RwLock<bool>>,
}

impl MarketFeedManager {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            active_subscriptions: Arc::new(RwLock::new(Vec::new())),
            is_connected: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn connect_and_run(&self) {
        let reconnect_delay = std::time::Duration::from_secs(5);
        
        loop {
            match self.connect_to_feed().await {
                Ok(_) => {
                    info!("[MARKET] Market feed connection closed normally");
                }
                Err(e) => {
                    error!("[ERROR] Market feed error: {}", e);
                }
            }
            
            // Update connection status
            *self.is_connected.write().await = false;
            self.emit_connection_status("disconnected").await;
            
            // Wait before reconnecting
            warn!("[RECONNECT] Reconnecting in {} seconds...", reconnect_delay.as_secs());
            tokio::time::sleep(reconnect_delay).await;
        }
    }

    async fn connect_to_feed(&self) -> Result<(), Box<dyn std::error::Error>> {
        // For development, we'll simulate a connection
        // In production, this would connect to real WebSocket endpoints
        
        info!("[CONNECT] Connecting to market data feed...");
        
        // Simulate connection delay
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        
        // Update connection status
        *self.is_connected.write().await = true;
        self.emit_connection_status("connected").await;
        
        info!("[SUCCESS] Market feed connected successfully");
        
        // Start simulated data feed
        self.run_simulated_feed().await?;
        
        Ok(())
    }

    async fn run_simulated_feed(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Simulate market data updates for development
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
        
        loop {
            interval.tick().await;
            
            // Check if still connected
            if !*self.is_connected.read().await {
                break;
            }
            
            // Get active subscriptions
            let subs = self.active_subscriptions.read().await.clone();
            
            for sub in subs {
                // Generate simulated data for each subscription
                if sub.data_types.contains(&"ticker".to_string()) {
                    self.emit_ticker_update(&sub.symbol).await?;
                }
                
                if sub.data_types.contains(&"orderbook".to_string()) {
                    self.emit_orderbook_update(&sub.symbol).await?;
                }
            }
        }
        
        Ok(())
    }

    async fn emit_ticker_update(&self, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        let price = 50000.0 + (rand::random::<f64>() * 1000.0 - 500.0);
        let volume = 1000.0 + rand::random::<f64>() * 100.0;
        
        let update = MarketUpdate {
            symbol: symbol.to_string(),
            data_type: "ticker".to_string(),
            data: serde_json::json!({
                "price": price,
                "volume_24h": volume,
                "change_24h": (rand::random::<f64>() * 10.0 - 5.0),
                "high_24h": price + 500.0,
                "low_24h": price - 500.0,
            }),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        self.app_handle.emit("market-update", update)?;
        Ok(())
    }

    async fn emit_orderbook_update(&self, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        let base_price = 50000.0;
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        // Generate simulated order book
        for i in 0..10 {
            let bid_price = base_price - (i as f64 * 10.0);
            let ask_price = base_price + (i as f64 * 10.0);
            let size = rand::random::<f64>() * 10.0;
            
            bids.push(vec![bid_price, size]);
            asks.push(vec![ask_price, size]);
        }
        
        let update = MarketUpdate {
            symbol: symbol.to_string(),
            data_type: "orderbook".to_string(),
            data: serde_json::json!({
                "bids": bids,
                "asks": asks,
                "timestamp": chrono::Utc::now().timestamp_millis(),
            }),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        self.app_handle.emit("market-update", update)?;
        Ok(())
    }

    pub async fn subscribe(&self, subscription: MarketSubscription) -> Result<(), String> {
        info!("[SUBSCRIBE] Subscribing to {} - types: {:?}", subscription.symbol, subscription.data_types);
        
        // Add to active subscriptions
        let mut subs = self.active_subscriptions.write().await;
        
        // Check if already subscribed
        if !subs.iter().any(|s| s.symbol == subscription.symbol) {
            subs.push(subscription.clone());
            
            // Emit subscription confirmation
            self.app_handle.emit("subscription-confirmed", &subscription)
                .map_err(|e| format!("Failed to emit confirmation: {}", e))?;
                
            info!("[CONFIRMED] Subscription confirmed for {}", subscription.symbol);
        } else {
            info!("[INFO] Already subscribed to {}", subscription.symbol);
        }
        
        Ok(())
    }

    pub async fn unsubscribe(&self, symbol: &str) -> Result<(), String> {
        info!("[UNSUBSCRIBE] Unsubscribing from {}", symbol);
        
        let mut subs = self.active_subscriptions.write().await;
        subs.retain(|s| s.symbol != symbol);
        
        // Emit unsubscription confirmation
        self.app_handle.emit("subscription-removed", symbol)
            .map_err(|e| format!("Failed to emit removal: {}", e))?;
            
        Ok(())
    }

    async fn emit_connection_status(&self, status: &str) {
        if let Err(e) = self.app_handle.emit("connection-status", serde_json::json!({
            "status": status,
            "timestamp": chrono::Utc::now().timestamp_millis()
        })) {
            error!("Failed to emit connection status: {}", e);
        }
    }
}

// Initialize the market feed in main.rs
pub async fn initialize_market_feed(app_handle: AppHandle) {
    let manager = MarketFeedManager::new(app_handle);
    
    // Store manager in app state if needed
    // app_handle.state::<Arc<MarketFeedManager>>().clone();
    
    // Run the feed in a background task
    tokio::spawn(async move {
        manager.connect_and_run().await;
    });
}
