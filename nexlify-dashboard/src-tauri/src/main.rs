// Location: C:\Nexlify\nexlify-dashboard\src-tauri\src\main.rs
// Replace the entire contents of this file
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod dashboard;
mod monitoring;

use dashboard::{DashboardState, TradingMetrics};
use std::sync::Arc;
use tokio::sync::RwLock;

fn main() {
    // Initialize dashboard state
    let dashboard_state = DashboardState {
        trinity_state: Arc::new(RwLock::new(Default::default())),
        hardware_metrics: Arc::new(dashmap::DashMap::new()),
        cascade_predictions: Arc::new(dashmap::DashMap::new()),
        trading_metrics: Arc::new(RwLock::new(TradingMetrics::default())),
    };

    tauri::Builder::default()
        .manage(dashboard_state.clone())
        .setup(move |_app| {
            let state = dashboard_state.clone();
            
            // Start monitoring tasks in a proper runtime
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    monitoring::start_monitoring(state).await;
					//Keep runtime alive
					loop {
						tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
					}
                });
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_trinity_state,
            commands::get_cascade_predictions,
            commands::get_hardware_metrics,
            commands::prepare_cascade_trading,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}