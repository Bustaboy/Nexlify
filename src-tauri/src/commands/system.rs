// src-tauri/src/commands/system.rs
// System monitoring and status commands

use super::*;
use sysinfo::System;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatus {
    status: String,
    version: String,
    uptime: u64,
    connections: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    cpu_usage: f32,
    memory_usage: f64,
    total_memory: f64,
    ws_latency: u32,
    orders_per_sec: f32,
}

// Shared system info instance

#[tauri::command]
pub async fn get_system_status() -> Result<SystemStatus, CommandError> {
    Ok(SystemStatus {
        status: "operational".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        connections: 1, // TODO: Track actual connections
    })
}

#[tauri::command]
pub async fn get_system_metrics() -> Result<SystemMetrics, CommandError> {
    // For sysinfo 0.30, we need to import the trait
    use sysinfo::{ProcessorExt, SystemExt};
    
    // Update system info
    {
        let mut sys = SYSTEM.lock();
        sys.refresh_cpu();
        sys.refresh_memory();
    }
    
    // Get metrics
    let sys = SYSTEM.lock();
    
    // Get CPU usage - in sysinfo 0.30, we use get_processors()
    let cpu_usage = {
        let processors = sys.get_processors();
        if processors.is_empty() {
            0.0
        } else {
            processors.iter().map(|p| p.get_cpu_usage()).sum::<f32>() / processors.len() as f32
        }
    };
    
    // Memory is returned in KB, convert to bytes
    let memory_usage = sys.get_used_memory() as f64 * 1024.0;
    let total_memory = sys.get_total_memory() as f64 * 1024.0;
    
    Ok(SystemMetrics {
        cpu_usage,
        memory_usage,
        total_memory,
        ws_latency: rand::random::<u32>() % 50 + 10, // Simulated latency 10-60ms
        orders_per_sec: rand::random::<f32>() * 100.0, // Simulated order rate
    })
}
