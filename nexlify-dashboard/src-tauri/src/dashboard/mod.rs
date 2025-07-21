// Location: C:\Nexlify\nexlify-dashboard\src-tauri\src\dashboard\mod.rs
// Create this new file
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct DashboardState {
    pub trinity_state: Arc<RwLock<TrinityState>>,
    pub hardware_metrics: Arc<DashMap<String, HardwareMetric>>,
    pub cascade_predictions: Arc<DashMap<String, CascadePrediction>>,
    pub trading_metrics: Arc<RwLock<TradingMetrics>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TradingMetrics {
    pub daily_pnl: f64,
    pub total_pnl: f64,
    pub positions_open: i32,
    pub last_trade: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrinityState {
    pub market_oracle_confidence: f32,
    pub crowd_psyche_state: String,
    pub city_pulse_health: f32,
    pub fusion_alignment: f32,
    pub last_update: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CascadePrediction {
    pub id: String,
    pub event_type: String,
    pub confidence: f32,
    pub time_to_event_minutes: i64,
    pub affected_sectors: Vec<String>,
    pub profit_opportunity: f64,
    pub detected_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareMetric {
    pub gpu_usage: f32,
    pub gpu_memory_used_mb: u64,
    pub gpu_temp_celsius: f32,
    pub inference_latency_ms: f32,
    pub cpu_usage: f32,
    pub ram_usage_gb: f32,
    pub timestamp: DateTime<Utc>,
}

impl Default for TrinityState {
    fn default() -> Self {
        Self {
            market_oracle_confidence: 0.5,
            crowd_psyche_state: "DORMANT".to_string(),
            city_pulse_health: 0.85,
            fusion_alignment: 0.2,
            last_update: Utc::now(),
        }
    }
}