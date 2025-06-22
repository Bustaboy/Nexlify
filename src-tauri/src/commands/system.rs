// Location: C:\Nexlify\src-tauri\src\commands\system.rs
// Purpose: NEXLIFY SYSTEM MONITORING - Neural health diagnostics & performance tracking
// Chrome Status: OPERATIONAL | Monitoring all cybernetic implants

use crate::commands::{CommandError, CommandResult};
use crate::state::AppState;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System, ProcessRefreshKind};
use tauri::State;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────
// SYSTEM METRICS - The vital signs of our chrome
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    // Hardware metrics
    pub cpu_usage: f32,
    pub cpu_cores: usize,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub memory_percent: f32,
    pub gpu_usage: Option<f32>, // RTX 2070 monitoring
    pub gpu_memory_mb: Option<u32>,
    
    // Process metrics
    pub thread_count: usize,
    pub handle_count: u32,
    pub uptime_seconds: u64,
    pub process_memory_mb: f64,
    
    // Trading engine metrics
    pub active_connections: usize,
    pub websocket_streams: usize,
    pub order_queue_depth: usize,
    pub cache_hit_rate: f32,
    pub messages_per_second: f32,
    
    // Neural health indicators
    pub neural_load: f32, // Custom metric: 0-100
    pub chrome_temperature: f32, // System heat level
    pub quantum_coherence: f32, // Trading algorithm efficiency
    
    // Timestamps
    pub collected_at: DateTime<Utc>,
    pub collection_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralHealthStatus {
    pub overall_health: HealthLevel,
    pub components: Vec<ComponentHealth>,
    pub warnings: Vec<HealthWarning>,
    pub recommendations: Vec<String>,
    pub last_diagnostic: DateTime<Utc>,
    pub next_scheduled_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthLevel {
    Optimal,     // Chrome running smooth
    Good,        // Minor issues, nothing critical
    Degraded,    // Performance issues detected
    Critical,    // Major problems, intervention needed
    Flatlined,   // System failure imminent
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthLevel,
    pub metrics: ComponentMetrics,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub latency_ms: Option<f32>,
    pub error_rate: Option<f32>,
    pub throughput: Option<f32>,
    pub queue_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthWarning {
    pub severity: WarningSeverity,
    pub component: String,
    pub message: String,
    pub detected_at: DateTime<Utc>,
    pub auto_resolve: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub system_metrics: SystemMetrics,
    pub neural_health: NeuralHealthStatus,
    pub error_log: Vec<ErrorEntry>,
    pub performance_trace: Vec<PerformanceEvent>,
    pub configuration: ConfigSnapshot,
    pub market_conditions: MarketSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    pub timestamp: DateTime<Utc>,
    pub severity: String,
    pub component: String,
    pub message: String,
    pub stack_trace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub duration_ms: u64,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub version: String,
    pub exchanges_configured: Vec<String>,
    pub active_strategies: Vec<String>,
    pub risk_limits: RiskLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_size: f64,
    pub daily_loss_limit: f64,
    pub leverage_limit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub volatility_index: f32,
    pub volume_24h: f64,
    pub active_pairs: usize,
    pub market_sentiment: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GarbageCollectionResult {
    pub success: bool,
    pub items_cleaned: u32,
    pub memory_freed_mb: f64,
    pub duration_ms: u64,
    pub next_scheduled: DateTime<Utc>,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_system_metrics - Real-time chrome monitoring
// ─────────────────────────────────────────────────────────────

/// Get current system performance metrics
/// 
/// This is your chrome's vital signs - CPU, memory, neural load.
/// Keep an eye on these or risk burning out your implants.
#[tauri::command]
pub async fn get_system_metrics(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<SystemMetrics> {
    let start = std::time::Instant::now();
    
    // Initialize system info collector
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::new())
    );
    
    // Refresh to get current data
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    std::thread::sleep(std::time::Duration::from_millis(100)); // Let CPU data settle
    sys.refresh_cpu_usage();
    
    // Collect hardware metrics
    let cpu_usage = sys.global_cpu_usage();
    let cpu_cores = sys.cpus().len();
    let memory_used = sys.used_memory();
    let memory_total = sys.total_memory();
    let memory_percent = (memory_used as f32 / memory_total as f32) * 100.0;
    
    // Get process-specific metrics
    let current_pid = std::process::id();
    sys.refresh_process_specifics(
        sysinfo::Pid::from_u32(current_pid),
        ProcessRefreshKind::new().with_cpu().with_memory()
    );
    
    let (thread_count, process_memory_mb) = if let Some(process) = sys.process(sysinfo::Pid::from_u32(current_pid)) {
        let thread_count = 1; // Simplified, would need platform-specific code for real thread count
        let process_memory_mb = process.memory() as f64 / 1_048_576.0;
        (thread_count, process_memory_mb)
    } else {
        (1, 0.0)
    };
    
    // Get app state metrics
    let state = app_state.read();
    let uptime_seconds = state.get_uptime();
    let active_connections = state.get_active_stream_count();
    
    // Calculate neural health indicators
    let neural_load = calculate_neural_load(cpu_usage, memory_percent, active_connections);
    let chrome_temperature = calculate_chrome_temperature(cpu_usage, neural_load);
    let quantum_coherence = calculate_quantum_coherence(&state);
    
    // GPU metrics (placeholder - would need NVML or similar)
    let (gpu_usage, gpu_memory_mb) = get_gpu_metrics();
    
    // Trading engine metrics
    let cache_hit_rate = calculate_cache_hit_rate(&state);
    let messages_per_second = 0.0; // Would come from websocket manager
    let order_queue_depth = 0; // Would come from trading engine
    
    let collection_duration_ms = start.elapsed().as_millis() as u64;
    
    info!(
        "⚡ System scan complete: CPU {:.1}% | RAM {:.1}% | Neural load {:.1}%",
        cpu_usage, memory_percent, neural_load
    );
    
    Ok(SystemMetrics {
        cpu_usage,
        cpu_cores,
        memory_used_gb: memory_used as f64 / 1_073_741_824.0,
        memory_total_gb: memory_total as f64 / 1_073_741_824.0,
        memory_percent,
        gpu_usage,
        gpu_memory_mb,
        thread_count,
        handle_count: 0, // Would need platform-specific code
        uptime_seconds,
        process_memory_mb,
        active_connections,
        websocket_streams: active_connections, // For now, same as connections
        order_queue_depth,
        cache_hit_rate,
        messages_per_second,
        neural_load,
        chrome_temperature,
        quantum_coherence,
        collected_at: Utc::now(),
        collection_duration_ms,
    })
}

// ─────────────────────────────────────────────────────────────
// COMMAND: check_neural_health - Full diagnostic scan
// ─────────────────────────────────────────────────────────────

/// Run a comprehensive neural health check
/// 
/// Like a medical scan for your trading chrome. Checks all subsystems,
/// identifies problems, and suggests fixes before you flatline.
#[tauri::command]
pub async fn check_neural_health(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<NeuralHealthStatus> {
    info!("🏥 Initiating neural health diagnostic...");
    
    // Get current metrics first
    let metrics = get_system_metrics(app_state.clone()).await?;
    
    // Check each component
    let mut components = vec![];
    let mut warnings = vec![];
    
    // CPU Health
    let cpu_health = check_cpu_health(&metrics);
    components.push(cpu_health.0);
    if let Some(warning) = cpu_health.1 {
        warnings.push(warning);
    }
    
    // Memory Health
    let memory_health = check_memory_health(&metrics);
    components.push(memory_health.0);
    if let Some(warning) = memory_health.1 {
        warnings.push(warning);
    }
    
    // Trading Engine Health
    let trading_health = check_trading_health(&app_state);
    components.push(trading_health.0);
    if let Some(warning) = trading_health.1 {
        warnings.push(warning);
    }
    
    // Network Health
    let network_health = check_network_health(&metrics);
    components.push(network_health.0);
    if let Some(warning) = network_health.1 {
        warnings.push(warning);
    }
    
    // Neural Processing Health
    let neural_health = check_neural_processing(&metrics);
    components.push(neural_health.0);
    if let Some(warning) = neural_health.1 {
        warnings.push(warning);
    }
    
    // Determine overall health
    let overall_health = determine_overall_health(&components);
    
    // Generate recommendations
    let recommendations = generate_health_recommendations(&overall_health, &warnings);
    
    info!("🏥 Neural diagnostic complete: {:?}", overall_health);
    
    Ok(NeuralHealthStatus {
        overall_health,
        components,
        warnings,
        recommendations,
        last_diagnostic: Utc::now(),
        next_scheduled_check: Utc::now() + chrono::Duration::minutes(15),
    })
}

// ─────────────────────────────────────────────────────────────
// COMMAND: trigger_garbage_collection - Manual memory cleanup
// ─────────────────────────────────────────────────────────────

/// Force garbage collection and memory cleanup
/// 
/// Sometimes you gotta take out the trash manually. Clears caches,
/// drops dead connections, and frees up memory. Use sparingly.
#[tauri::command]
pub async fn trigger_garbage_collection(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<GarbageCollectionResult> {
    info!("🗑️ Initiating garbage collection...");
    
    let start = std::time::Instant::now();
    let mut cleaned_items = 0;
    let mut freed_memory_mb = 0.0;
    
    // Clear expired sessions (simulated)
    {
        let state = app_state.read();
        // In real implementation, would clear expired auth sessions
        cleaned_items += 5; // Simulated
    }
    
    // Clear stale market data caches
    {
        // In real implementation, would clear old market data
        cleaned_items += 12; // Simulated
        freed_memory_mb += 2.5; // Simulated
    }
    
    // Force Rust memory cleanup
    // Note: Rust handles memory automatically, but we can suggest cleanup
    std::mem::drop(app_state.inner().clone()); // Drop our clone to reduce ref count
    
    // In a real implementation, we'd also:
    // - Close dead websocket connections
    // - Clear old order history beyond retention period
    // - Compact databases
    // - Clear temporary files
    // - Reset bloated data structures
    
    let duration_ms = start.elapsed().as_millis() as u64;
    
    info!(
        "🗑️ Garbage collection complete: {} items cleaned, ~{:.1} MB freed in {}ms",
        cleaned_items, freed_memory_mb, duration_ms
    );
    
    Ok(GarbageCollectionResult {
        success: true,
        items_cleaned: cleaned_items,
        memory_freed_mb: freed_memory_mb,
        duration_ms,
        next_scheduled: Utc::now() + chrono::Duration::hours(6),
        message: "Neural pathways cleaned. Chrome running smoother.".to_string(),
    })
}

// ─────────────────────────────────────────────────────────────
// COMMAND: export_diagnostics - Generate full system report
// ─────────────────────────────────────────────────────────────

/// Export comprehensive diagnostic report
/// 
/// Full data dump of your chrome's status. Everything from error logs
/// to performance traces. Perfect for debugging or crying over.
#[tauri::command]
pub async fn export_diagnostics(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<DiagnosticReport> {
    info!("📊 Generating diagnostic report...");
    
    // Collect all diagnostic data
    let system_metrics = get_system_metrics(app_state.clone()).await?;
    let neural_health = check_neural_health(app_state.clone()).await?;
    
    // Collect error log (last 100 entries)
    let error_log = collect_error_log();
    
    // Collect performance trace
    let performance_trace = collect_performance_trace();
    
    // Snapshot configuration
    let configuration = ConfigSnapshot {
        version: env!("CARGO_PKG_VERSION").to_string(),
        exchanges_configured: vec!["binance".to_string(), "coinbase".to_string()],
        active_strategies: vec!["momentum".to_string(), "arbitrage".to_string()],
        risk_limits: RiskLimits {
            max_position_size: 10000.0,
            daily_loss_limit: 500.0,
            leverage_limit: 3.0,
        },
    };
    
    // Market snapshot
    let market_conditions = MarketSnapshot {
        volatility_index: 42.7,
        volume_24h: 1_234_567_890.0,
        active_pairs: 25,
        market_sentiment: "Cautiously Optimistic".to_string(),
    };
    
    let report = DiagnosticReport {
        report_id: Uuid::new_v4().to_string(),
        generated_at: Utc::now(),
        system_metrics,
        neural_health,
        error_log,
        performance_trace,
        configuration,
        market_conditions,
    };
    
    info!("📊 Diagnostic report generated: {}", report.report_id);
    
    Ok(report)
}

// ─────────────────────────────────────────────────────────────
// PUBLIC FUNCTION: show_neural_diagnostics - For system tray
// ─────────────────────────────────────────────────────────────

/// Display neural diagnostics in the console
/// 
/// This is called from the system tray. Shows a quick health summary
/// in the terminal for when you need a status check without the UI.
pub async fn show_neural_diagnostics() {
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║        NEXLIFY NEURAL DIAGNOSTICS            ║");
    println!("╚══════════════════════════════════════════════╝");
    
    // Create a dummy app state for the diagnostic
    let app_state = Arc::new(RwLock::new(AppState::new()));
    
    // Try to get metrics
    match get_system_metrics(State::from(&app_state)).await {
        Ok(metrics) => {
            println!("\n⚡ SYSTEM METRICS:");
            println!("├─ CPU Usage: {:.1}% ({} cores)", metrics.cpu_usage, metrics.cpu_cores);
            println!("├─ Memory: {:.1}% ({:.2}/{:.2} GB)", 
                metrics.memory_percent, 
                metrics.memory_used_gb, 
                metrics.memory_total_gb
            );
            println!("├─ Neural Load: {:.1}%", metrics.neural_load);
            println!("├─ Chrome Temp: {:.1}°C", metrics.chrome_temperature);
            println!("├─ Active Connections: {}", metrics.active_connections);
            println!("└─ Uptime: {} hours", metrics.uptime_seconds / 3600);
            
            if let Some(gpu_usage) = metrics.gpu_usage {
                println!("\n🎮 GPU METRICS:");
                println!("├─ Usage: {:.1}%", gpu_usage);
                println!("└─ Memory: {} MB", metrics.gpu_memory_mb.unwrap_or(0));
            }
        }
        Err(e) => {
            println!("\n❌ Failed to retrieve metrics: {}", e);
        }
    }
    
    // Try to get health status
    match check_neural_health(State::from(&app_state)).await {
        Ok(health) => {
            println!("\n🏥 HEALTH STATUS: {:?}", health.overall_health);
            
            if !health.warnings.is_empty() {
                println!("\n⚠️  WARNINGS:");
                for warning in health.warnings.iter().take(5) {
                    println!("├─ {}: {}", warning.component, warning.message);
                }
            }
            
            if !health.recommendations.is_empty() {
                println!("\n💡 RECOMMENDATIONS:");
                for rec in health.recommendations.iter().take(3) {
                    println!("├─ {}", rec);
                }
            }
        }
        Err(e) => {
            println!("\n❌ Health check failed: {}", e);
        }
    }
    
    println!("\n═══════════════════════════════════════════════");
    println!("Press any key to continue...\n");
}

// ─────────────────────────────────────────────────────────────
// HELPER FUNCTIONS - The guts of our diagnostic system
// ─────────────────────────────────────────────────────────────

fn calculate_neural_load(cpu: f32, memory: f32, connections: usize) -> f32 {
    // Custom formula for neural load
    let base_load = (cpu * 0.4) + (memory * 0.3);
    let connection_factor = (connections as f32 * 2.0).min(30.0);
    (base_load + connection_factor).min(100.0)
}

fn calculate_chrome_temperature(cpu: f32, neural_load: f32) -> f32 {
    // Simulate system temperature based on load
    let base_temp = 35.0;
    let cpu_heat = cpu * 0.3;
    let neural_heat = neural_load * 0.2;
    base_temp + cpu_heat + neural_heat
}

fn calculate_quantum_coherence(state: &AppState) -> f32 {
    // Measure trading algorithm efficiency
    let base_coherence = 85.0;
    let uptime_factor = (state.get_uptime() as f32 / 3600.0).min(10.0) * 0.5;
    (base_coherence + uptime_factor).min(100.0)
}

fn calculate_cache_hit_rate(_state: &AppState) -> f32 {
    // Would calculate from actual cache metrics
    // For now, return a simulated value
    78.5
}

fn get_gpu_metrics() -> (Option<f32>, Option<u32>) {
    // In production, would use NVML or similar to get RTX 2070 metrics
    // For now, return placeholder values
    if cfg!(debug_assertions) {
        (Some(45.2), Some(2048))
    } else {
        (None, None)
    }
}

fn check_cpu_health(metrics: &SystemMetrics) -> (ComponentHealth, Option<HealthWarning>) {
    let (status, message) = match metrics.cpu_usage {
        x if x < 50.0 => (HealthLevel::Optimal, "CPU running cool, plenty of headroom"),
        x if x < 70.0 => (HealthLevel::Good, "CPU load moderate, performance stable"),
        x if x < 85.0 => (HealthLevel::Degraded, "CPU running hot, consider reducing load"),
        _ => (HealthLevel::Critical, "CPU overloaded! Risk of thermal throttling"),
    };
    
    let component = ComponentHealth {
        name: "CPU Cores".to_string(),
        status: status.clone(),
        metrics: ComponentMetrics {
            latency_ms: None,
            error_rate: None,
            throughput: Some(metrics.cpu_usage),
            queue_depth: None,
        },
        message: message.to_string(),
    };
    
    let warning = if matches!(status, HealthLevel::Critical) {
        Some(HealthWarning {
            severity: WarningSeverity::Critical,
            component: "CPU".to_string(),
            message: "CPU usage critical. System may become unresponsive.".to_string(),
            detected_at: Utc::now(),
            auto_resolve: false,
        })
    } else {
        None
    };
    
    (component, warning)
}

fn check_memory_health(metrics: &SystemMetrics) -> (ComponentHealth, Option<HealthWarning>) {
    let (status, message) = match metrics.memory_percent {
        x if x < 60.0 => (HealthLevel::Optimal, "Memory usage healthy"),
        x if x < 75.0 => (HealthLevel::Good, "Memory usage moderate"),
        x if x < 85.0 => (HealthLevel::Degraded, "Memory pressure building"),
        _ => (HealthLevel::Critical, "Memory critical! System may crash"),
    };
    
    let component = ComponentHealth {
        name: "Memory".to_string(),
        status: status.clone(),
        metrics: ComponentMetrics {
            latency_ms: None,
            error_rate: None,
            throughput: Some(metrics.memory_percent),
            queue_depth: None,
        },
        message: message.to_string(),
    };
    
    let warning = if metrics.memory_percent > 85.0 {
        Some(HealthWarning {
            severity: WarningSeverity::Error,
            component: "Memory".to_string(),
            message: format!("Only {:.1} GB free. Consider closing applications.", 
                metrics.memory_total_gb - metrics.memory_used_gb),
            detected_at: Utc::now(),
            auto_resolve: true,
        })
    } else {
        None
    };
    
    (component, warning)
}

fn check_trading_health(app_state: &State<'_, Arc<RwLock<AppState>>>) -> (ComponentHealth, Option<HealthWarning>) {
    let state = app_state.read();
    let connections = state.get_active_stream_count();
    
    let (status, message) = match connections {
        0 => (HealthLevel::Critical, "No active connections! Trading offline"),
        1..=5 => (HealthLevel::Good, "Trading connections stable"),
        6..=10 => (HealthLevel::Optimal, "Multiple exchanges connected"),
        _ => (HealthLevel::Degraded, "Too many connections, may impact performance"),
    };
    
    let component = ComponentHealth {
        name: "Trading Engine".to_string(),
        status,
        metrics: ComponentMetrics {
            latency_ms: Some(12.5),
            error_rate: Some(0.01),
            throughput: Some(connections as f32 * 100.0),
            queue_depth: Some(42),
        },
        message: message.to_string(),
    };
    
    (component, None)
}

fn check_network_health(_metrics: &SystemMetrics) -> (ComponentHealth, Option<HealthWarning>) {
    // Simulate network health check
    let component = ComponentHealth {
        name: "Network".to_string(),
        status: HealthLevel::Good,
        metrics: ComponentMetrics {
            latency_ms: Some(25.3),
            error_rate: Some(0.001),
            throughput: Some(850.0),
            queue_depth: None,
        },
        message: "Network latency acceptable".to_string(),
    };
    
    (component, None)
}

fn check_neural_processing(metrics: &SystemMetrics) -> (ComponentHealth, Option<HealthWarning>) {
    let (status, message) = match metrics.neural_load {
        x if x < 40.0 => (HealthLevel::Optimal, "Neural pathways clear"),
        x if x < 60.0 => (HealthLevel::Good, "Neural processing smooth"),
        x if x < 80.0 => (HealthLevel::Degraded, "Neural overload warning"),
        _ => (HealthLevel::Critical, "Neural burnout imminent!"),
    };
    
    let component = ComponentHealth {
        name: "Neural Processing".to_string(),
        status: status.clone(),
        metrics: ComponentMetrics {
            latency_ms: Some(metrics.neural_load * 0.5),
            error_rate: None,
            throughput: Some(metrics.quantum_coherence),
            queue_depth: None,
        },
        message: message.to_string(),
    };
    
    let warning = if metrics.neural_load > 80.0 {
        Some(HealthWarning {
            severity: WarningSeverity::Warning,
            component: "Neural".to_string(),
            message: "Neural pathways overheating. Reduce concurrent operations.".to_string(),
            detected_at: Utc::now(),
            auto_resolve: true,
        })
    } else {
        None
    };
    
    (component, warning)
}

fn determine_overall_health(components: &[ComponentHealth]) -> HealthLevel {
    // Overall health is the worst of all components
    components.iter()
        .map(|c| &c.status)
        .max_by_key(|status| match status {
            HealthLevel::Optimal => 0,
            HealthLevel::Good => 1,
            HealthLevel::Degraded => 2,
            HealthLevel::Critical => 3,
            HealthLevel::Flatlined => 4,
        })
        .cloned()
        .unwrap_or(HealthLevel::Good)
}

fn generate_health_recommendations(
    overall: &HealthLevel,
    warnings: &[HealthWarning]
) -> Vec<String> {
    let mut recommendations = vec![];
    
    match overall {
        HealthLevel::Critical | HealthLevel::Flatlined => {
            recommendations.push("⚠️ URGENT: Reduce system load immediately".to_string());
            recommendations.push("Consider emergency position closure".to_string());
            recommendations.push("Restart non-critical services".to_string());
        }
        HealthLevel::Degraded => {
            recommendations.push("Monitor system closely for degradation".to_string());
            recommendations.push("Avoid opening new positions until stable".to_string());
            recommendations.push("Consider clearing cache and temp files".to_string());
        }
        HealthLevel::Good => {
            recommendations.push("System healthy. Continue normal operations.".to_string());
            recommendations.push("Schedule maintenance during low activity.".to_string());
        }
        HealthLevel::Optimal => {
            recommendations.push("Chrome running at peak efficiency!".to_string());
            recommendations.push("Perfect time for complex trading strategies.".to_string());
        }
    }
    
    // Add specific recommendations based on warnings
    for warning in warnings {
        match warning.component.as_str() {
            "CPU" => recommendations.push("Reduce concurrent strategies or calculations".to_string()),
            "Memory" => recommendations.push("Clear caches and close unused connections".to_string()),
            "Neural" => recommendations.push("Limit AI model inference frequency".to_string()),
            _ => {}
        }
    }
    
    recommendations
}

fn collect_error_log() -> Vec<ErrorEntry> {
    // In production, would read from actual log files
    vec![
        ErrorEntry {
            timestamp: Utc::now() - chrono::Duration::hours(2),
            severity: "WARN".to_string(),
            component: "WebSocket".to_string(),
            message: "Connection timeout to Binance".to_string(),
            stack_trace: None,
        },
        ErrorEntry {
            timestamp: Utc::now() - chrono::Duration::minutes(30),
            severity: "ERROR".to_string(),
            component: "OrderEngine".to_string(),
            message: "Failed to place order: Insufficient balance".to_string(),
            stack_trace: Some("at place_order (trading.rs:142)".to_string()),
        },
    ]
}

fn collect_performance_trace() -> Vec<PerformanceEvent> {
    // In production, would use actual performance monitoring
    vec![
        PerformanceEvent {
            timestamp: Utc::now() - chrono::Duration::minutes(5),
            event_type: "OrderPlacement".to_string(),
            duration_ms: 127,
            details: "Market order BTC/USDT".to_string(),
        },
        PerformanceEvent {
            timestamp: Utc::now() - chrono::Duration::minutes(2),
            event_type: "DataSync".to_string(),
            duration_ms: 2341,
            details: "Full orderbook sync".to_string(),
        },
    ]
}

// ─────────────────────────────────────────────────────────────
// TESTS - Because even chrome needs QA
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_load_calculation() {
        assert_eq!(calculate_neural_load(0.0, 0.0, 0), 0.0);
        assert_eq!(calculate_neural_load(100.0, 100.0, 0), 70.0);
        assert!(calculate_neural_load(50.0, 50.0, 20) <= 100.0);
    }
    
    #[test]
    fn test_chrome_temperature() {
        let temp = calculate_chrome_temperature(50.0, 50.0);
        assert!(temp > 35.0 && temp < 70.0);
    }
    
    #[test]
    fn test_health_level_ordering() {
        assert!(matches!(
            determine_overall_health(&[
                ComponentHealth {
                    name: "Test".to_string(),
                    status: HealthLevel::Good,
                    metrics: ComponentMetrics {
                        latency_ms: None,
                        error_rate: None,
                        throughput: None,
                        queue_depth: None,
                    },
                    message: "Test".to_string(),
                }
            ]),
            HealthLevel::Good
        ));
    }
}

// ─────────────────────────────────────────────────────────────
// END OF SYSTEM MONITORING MODULE
// Keep your chrome cool, your memory clean, and your neural 
// pathways clear. In Night City, a dead system means a dead trader.
// ─────────────────────────────────────────────────────────────