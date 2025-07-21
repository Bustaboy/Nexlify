// Location: C:\Nexlify\nexlify-dashboard\src-tauri\src\monitoring\mod.rs
// Create this new file
use crate::dashboard::{DashboardState, TrinityState, CascadePrediction, HardwareMetric};
use chrono::Utc;
use std::time::Duration;
use tokio::time::sleep;

pub async fn start_monitoring(state: DashboardState) {
    // Spawn trinity monitoring
    let trinity_state = state.clone();
    tokio::spawn(async move {
        loop {
            update_trinity_state(&trinity_state).await;
            sleep(Duration::from_millis(100)).await;
        }
    });

    // Spawn hardware monitoring
    let hardware_state = state.clone();
    tokio::spawn(async move {
        loop {
            update_hardware_metrics(&hardware_state).await;
            sleep(Duration::from_secs(1)).await;
        }
    });

    // Spawn cascade detection
    let cascade_state = state.clone();
    tokio::spawn(async move {
        loop {
            detect_cascades(&cascade_state).await;
            sleep(Duration::from_secs(5)).await;
        }
    });
}

async fn update_trinity_state(state: &DashboardState) {
    // Simulate Trinity data (replace with real Trinity integration)
    let mut trinity = state.trinity_state.write().await;
    
    // Simulate market dynamics
    trinity.market_oracle_confidence = 
        (trinity.market_oracle_confidence + rand::random::<f32>() * 0.1 - 0.05)
        .clamp(0.0, 1.0);
    
    // Update fusion alignment (this is where money is made)
    trinity.fusion_alignment = calculate_fusion_alignment(&trinity);
    
    trinity.last_update = Utc::now();
}

fn calculate_fusion_alignment(trinity: &TrinityState) -> f32 {
    // Simplified fusion calculation
    let oracle_weight = trinity.market_oracle_confidence * 0.4;
    let psyche_weight = match trinity.crowd_psyche_state.as_str() {
        "ERUPTING" => 0.9,
        "VOLATILE" => 0.7,
        "AGITATED" => 0.5,
        _ => 0.2,
    } * 0.3;
    let pulse_weight = (1.0 - trinity.city_pulse_health) * 0.3;
    
    oracle_weight + psyche_weight + pulse_weight
}

async fn detect_cascades(state: &DashboardState) {
    let trinity = state.trinity_state.read().await;
    
    if trinity.fusion_alignment > 0.8 {
        // CASCADE DETECTED! 
        let cascade = CascadePrediction {
            id: format!("CASCADE_{}", Utc::now().timestamp()),
            event_type: "MARKET_COLLAPSE".to_string(),
            confidence: trinity.fusion_alignment,
            time_to_event_minutes: 240, // 4 hours
            affected_sectors: vec!["TECH".to_string(), "FINANCE".to_string()],
            profit_opportunity: 500_000.0 * trinity.fusion_alignment as f64,
            detected_at: Utc::now(),
        };
        
        state.cascade_predictions.insert(cascade.id.clone(), cascade);
    }
}

async fn update_hardware_metrics(state: &DashboardState) {
    // Simulate hardware metrics (replace with real monitoring)
    let metric = HardwareMetric {
        gpu_usage: 45.0 + rand::random::<f32>() * 20.0,
        gpu_memory_used_mb: 4096,
        gpu_temp_celsius: 65.0 + rand::random::<f32>() * 10.0,
        inference_latency_ms: 8.5 + rand::random::<f32>() * 3.0,
        cpu_usage: 30.0 + rand::random::<f32>() * 20.0,
        ram_usage_gb: 12.5,
        timestamp: Utc::now(),
    };
    
    state.hardware_metrics.insert("current".to_string(), metric);
}

// Add rand crate function
use rand::Rng;
fn rand_random<T>() -> T 
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    rand::thread_rng().gen()
}