// Location: C:\Nexlify\nexlify-dashboard\src-tauri\src\commands.rs
// Create this new file
use crate::dashboard::{DashboardState, TrinityState, CascadePrediction};
use tauri::State;

#[tauri::command]
pub async fn get_trinity_state(
    state: State<'_, DashboardState>
) -> Result<TrinityState, String> {
    Ok(state.trinity_state.read().await.clone())
}

#[tauri::command]
pub async fn get_cascade_predictions(
    state: State<'_, DashboardState>
) -> Result<Vec<CascadePrediction>, String> {
    let predictions: Vec<CascadePrediction> = state
        .cascade_predictions
        .iter()
        .map(|entry| entry.value().clone())
        .collect();
    
    Ok(predictions)
}

#[tauri::command]
pub async fn get_hardware_metrics(
    state: State<'_, DashboardState>
) -> Result<HardwareMetric, String> {
    state
        .hardware_metrics
        .get("current")
        .map(|entry| entry.value().clone())
        .ok_or_else(|| "No hardware metrics available".to_string())
}

#[tauri::command]
pub async fn prepare_cascade_trading(
    cascade_id: String,
    state: State<'_, DashboardState>
) -> Result<String, String> {
    // This is where you'd integrate with Mission 85-I
    println!("🚨 PREPARING CASCADE TRADE: {}", cascade_id);
    
    // For now, just acknowledge
    Ok(format!("Ready to trade cascade: {}", cascade_id))
}