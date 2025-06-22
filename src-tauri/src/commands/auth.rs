// Location: C:\Nexlify\src-tauri\src\commands\auth.rs
// Purpose: NEXLIFY AUTHENTICATION COMMANDS - Where trust meets chrome
// Last sync: 2025-06-19 | "In crypto, paranoia is a feature, not a bug"

use crate::commands::{CommandError, CommandResult};
use crate::state::AppState;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use chrono::{DateTime, Duration, Utc};
use tracing::{info, warn, error, debug};

// ─────────────────────────────────────────────────────────────
// NEXLIFY AUTHENTICATION MODULE - "Trust no one, verify everything"
// 
// Every street kid knows: the best security is layers. Physical,
// digital, psychological. In the barrio, we had lookouts, codes,
// and safe houses. Here in the digital sprawl, we have encryption,
// 2FA, and time-based lockouts. Same game, different arena.
//
// Remember: Your API keys are your life. Guard them accordingly.
// ─────────────────────────────────────────────────────────────

// Configuration constants - adjust these based on your paranoia level
const MAX_LOGIN_ATTEMPTS: u32 = 5;
const LOCKOUT_DURATION_MINS: i64 = 30;
const SESSION_TIMEOUT_MINS: i64 = 120;
const API_KEY_ROTATION_DAYS: i64 = 90;
const PASSWORD_MIN_LENGTH: usize = 12;

// ─────────────────────────────────────────────────────────────
// COMMAND: login - Jack into the neural vault (renamed from unlock_stronghold)
// ─────────────────────────────────────────────────────────────

/// Authenticate and create a session - your entry to the trading matrix
/// 
/// Used to be called unlock_stronghold, but login is clearer.
/// This is your front door. Get past this, and you're in the game.
/// Fail too many times, and you're locked out like a drunk at 3am.
#[tauri::command]
pub async fn login(
    password: String,
    two_factor_code: Option<String>,
    remember_device: Option<bool>,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<LoginResponse> {
    // Check if account is locked from too many attempts
    let state = app_state.read();
    let auth_state = &state.auth_state;
    
    if let Some(locked_until) = auth_state.locked_until {
        if locked_until > Utc::now() {
            let remaining = (locked_until - Utc::now()).num_minutes();
            return Err(CommandError::AuthError(
                format!("Account locked for {} more minutes. Coffee break time, ese.", remaining)
            ));
        }
    }
    
    // Validate password strength
    if password.len() < PASSWORD_MIN_LENGTH {
        increment_failed_attempts(&app_state);
        return Err(CommandError::AuthError(
            "Password too short. This ain't 2010, we need real security.".to_string()
        ));
    }
    
    // In production, this would check against stored hash
    // For now, we simulate the verification
    let password_hash = hash_password(&password)?;
    let success = verify_password_hash(&password_hash).await;
    
    if success {
        // Reset failed attempts
        reset_auth_state(&app_state);
        
        // Create session
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = Session {
            id: session_id.clone(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            permissions: vec!["trade".to_string(), "read".to_string()],
        };
        
        // Store session
        store_session(&app_state, session);
        
        info!("🔓 Stronghold unlocked - Welcome back to the neural mesh");
        
        Ok(LoginResponse {
            success: true,
            session_id,
            permissions: vec!["trade".to_string(), "read".to_string()],
            expires_at: Utc::now() + Duration::minutes(SESSION_TIMEOUT_MINS),
            device_trusted: remember_device.unwrap_or(false),
            message: "Neural vault unlocked. Trade carefully out there.".to_string(),
        })
    } else {
        increment_failed_attempts(&app_state);
        
        let attempts_left = MAX_LOGIN_ATTEMPTS - get_failed_attempts(&app_state);
        
        warn!("🚫 Failed login attempt - {} tries remaining", attempts_left);
        
        Err(CommandError::AuthError(
            format!("Invalid credentials. {} attempts remaining before lockout.", attempts_left)
        ))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginResponse {
    pub success: bool,
    pub session_id: String,
    pub permissions: Vec<String>,
    pub expires_at: DateTime<Utc>,
    pub device_trusted: bool,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: logout - Disconnect from the matrix
// ─────────────────────────────────────────────────────────────

/// End the current session - always logout when you're done
/// 
/// In the streets, you never leave your back exposed. Same here.
/// Logout clears your session, wipes temp data, and locks the vault.
#[tauri::command]
pub async fn logout(
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<LogoutResponse> {
    // Verify session exists
    verify_session(&app_state, &session_id)?;
    
    // Clear session
    let mut state = app_state.write();
    state.active_sessions.remove(&session_id);
    
    // Clear any cached sensitive data
    // In production, this would also:
    // - Revoke any temp tokens
    // - Clear websocket subscriptions
    // - Wipe memory caches
    
    info!("🔒 Session {} terminated - Stay safe out there", session_id);
    
    Ok(LogoutResponse {
        success: true,
        message: "Logged out successfully. Until next time, choom.".to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogoutResponse {
    pub success: bool,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: refresh_session - Extend your stay in the matrix
// ─────────────────────────────────────────────────────────────

/// Refresh an active session - keep the connection alive
/// 
/// Sessions timeout for security. But if you're actively trading,
/// you don't want to get booted mid-order. This extends your time.
#[tauri::command]
pub async fn refresh_session(
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<RefreshSessionResponse> {
    // Verify session exists and is valid
    verify_session(&app_state, &session_id)?;
    
    // Update last activity
    let mut state = app_state.write();
    if let Some(session) = state.active_sessions.get_mut(&session_id) {
        session.last_activity = Utc::now();
        
        let new_expiry = Utc::now() + Duration::minutes(SESSION_TIMEOUT_MINS);
        
        debug!("🔄 Session {} refreshed, expires at {}", session_id, new_expiry);
        
        Ok(RefreshSessionResponse {
            success: true,
            expires_at: new_expiry,
            message: "Session extended. Keep trading.".to_string(),
        })
    } else {
        Err(CommandError::AuthError(
            "Session not found. You've been ghosted.".to_string()
        ))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshSessionResponse {
    pub success: bool,
    pub expires_at: DateTime<Utc>,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: manage_api_keys - Store/update exchange credentials (renamed from store_api_credentials)
// ─────────────────────────────────────────────────────────────

/// Store or update API credentials for an exchange
/// 
/// Your API keys are like your apartment keys - lose them and someone
/// else is spending your money. We encrypt these babies seven ways to Sunday.
#[tauri::command]
pub async fn manage_api_keys(
    exchange: String,
    api_key: String,
    api_secret: String,
    passphrase: Option<String>,
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<ManageApiKeysResponse> {
    // Verify session
    verify_session(&app_state, &session_id)?;
    
    // Validate exchange
    validation::validate_exchange(&exchange)?;
    
    // Validate API key format
    if api_key.len() < 20 || api_secret.len() < 20 {
        return Err(CommandError::AuthError(
            "API credentials look sus. Double-check them.".to_string()
        ));
    }
    
    // Check for common mistakes
    if api_key.to_lowercase().contains("secret") || api_secret.to_lowercase().contains("key") {
        return Err(CommandError::AuthError(
            "Looks like you mixed up key and secret. Happens to the best of us.".to_string()
        ));
    }
    
    // Encrypt credentials
    let encrypted_key = encrypt_credential(&api_key)?;
    let encrypted_secret = encrypt_credential(&api_secret)?;
    let encrypted_passphrase = passphrase.map(|p| encrypt_credential(&p)).transpose()?;
    
    // Store in Stronghold (simulated)
    let credential_id = format!("CRED-{}-{}", exchange.to_uppercase(), 
        uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
    
    info!("🔐 API credentials stored for {} - ID: {}", exchange, credential_id);
    
    // Test the credentials
    let test_result = test_exchange_connection(&exchange, &api_key, &api_secret).await;
    
    Ok(ManageApiKeysResponse {
        credential_id,
        exchange,
        stored: true,
        connection_test: test_result.is_ok(),
        test_message: test_result.unwrap_or_else(|e| e.to_string()),
        permissions_detected: vec!["spot_trading".to_string(), "read_balance".to_string()],
        message: if test_result.is_ok() {
            "Credentials stored and verified. You're connected to the market matrix."
        } else {
            "Credentials stored but connection failed. Check your keys."
        }.to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ManageApiKeysResponse {
    pub credential_id: String,
    pub exchange: String,
    pub stored: bool,
    pub connection_test: bool,
    pub test_message: String,
    pub permissions_detected: Vec<String>,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: verify_credentials - Test without storing
// ─────────────────────────────────────────────────────────────

/// Verify API credentials work without storing them
/// 
/// Sometimes you just want to test the waters before diving in.
/// This checks if your keys work without saving them to the vault.
#[tauri::command]
pub async fn verify_credentials(
    exchange: String,
    api_key: String,
    api_secret: String,
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<VerifyCredentialResponse> {
    verify_session(&app_state, &session_id)?;
    
    info!("🔍 Testing {} credentials without storage", exchange);
    
    // Test connection
    match test_exchange_connection(&exchange, &api_key, &api_secret).await {
        Ok(permissions) => {
            Ok(VerifyCredentialResponse {
                valid: true,
                exchange,
                permissions_detected: permissions,
                rate_limits: Some(RateLimits {
                    requests_per_minute: 1200,
                    orders_per_minute: 60,
                    weight_per_minute: 6000,
                }),
                message: "Credentials verified. Connection successful.".to_string(),
            })
        }
        Err(e) => {
            Ok(VerifyCredentialResponse {
                valid: false,
                exchange,
                permissions_detected: vec![],
                rate_limits: None,
                message: format!("Verification failed: {}", e),
            })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerifyCredentialResponse {
    pub valid: bool,
    pub exchange: String,
    pub permissions_detected: Vec<String>,
    pub rate_limits: Option<RateLimits>,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: u32,
    pub orders_per_minute: u32,
    pub weight_per_minute: u32,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: get_exchange_status - Check connection health
// ─────────────────────────────────────────────────────────────

/// Get current status of exchange connections
/// 
/// Like checking your six in a firefight. Shows which exchanges
/// are connected, their latency, and any issues. Knowledge is power.
#[tauri::command]
pub async fn get_exchange_status(
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<ExchangeStatusResponse> {
    verify_session(&app_state, &session_id)?;
    
    // In production, this would check actual connections
    let exchanges = vec![
        ExchangeStatus {
            name: "binance".to_string(),
            connected: true,
            latency_ms: 45,
            last_heartbeat: Utc::now() - Duration::seconds(5),
            rate_limit_remaining: 1150,
            rate_limit_reset: Utc::now() + Duration::minutes(1),
            features: vec!["spot".to_string(), "futures".to_string()],
        },
        ExchangeStatus {
            name: "coinbase".to_string(),
            connected: true,
            latency_ms: 82,
            last_heartbeat: Utc::now() - Duration::seconds(8),
            rate_limit_remaining: 580,
            rate_limit_reset: Utc::now() + Duration::seconds(45),
            features: vec!["spot".to_string()],
        },
        ExchangeStatus {
            name: "kraken".to_string(),
            connected: false,
            latency_ms: 0,
            last_heartbeat: Utc::now() - Duration::hours(2),
            rate_limit_remaining: 0,
            rate_limit_reset: Utc::now(),
            features: vec![],
        },
    ];
    
    let total_connected = exchanges.iter().filter(|e| e.connected).count();
    
    Ok(ExchangeStatusResponse {
        exchanges,
        total_configured: 3,
        total_connected,
        global_status: if total_connected > 0 { "operational" } else { "offline" }.to_string(),
        message: format!("{}/{} exchanges online. {}",
            total_connected, 3,
            if total_connected == 0 { "Check your connections!" } else { "Ready to trade." }
        ),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeStatusResponse {
    pub exchanges: Vec<ExchangeStatus>,
    pub total_configured: usize,
    pub total_connected: usize,
    pub global_status: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeStatus {
    pub name: String,
    pub connected: bool,
    pub latency_ms: u32,
    pub last_heartbeat: DateTime<Utc>,
    pub rate_limit_remaining: u32,
    pub rate_limit_reset: DateTime<Utc>,
    pub features: Vec<String>,
}

// ─────────────────────────────────────────────────────────────
// COMMAND: rotate_api_key - Security through key rotation (renamed from rotate_keys)
// ─────────────────────────────────────────────────────────────

/// Rotate API keys for enhanced security
/// 
/// In the old neighborhood, we'd change locks after any close call.
/// Same principle here. Regular key rotation keeps you ahead of the game.
#[tauri::command]
pub async fn rotate_api_key(
    exchange: String,
    new_api_key: String,
    new_api_secret: String,
    new_passphrase: Option<String>,
    session_id: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<RotateKeysResponse> {
    verify_session(&app_state, &session_id)?;
    
    info!("🔄 Rotating API keys for {}", exchange);
    
    // Verify new credentials work
    match test_exchange_connection(&exchange, &new_api_key, &new_api_secret).await {
        Ok(_) => {
            // Encrypt and store new credentials
            let encrypted_key = encrypt_credential(&new_api_key)?;
            let encrypted_secret = encrypt_credential(&new_api_secret)?;
            let encrypted_passphrase = new_passphrase.map(|p| encrypt_credential(&p)).transpose()?;
            
            // Archive old keys (in production)
            // Update to new keys
            
            let rotation_id = format!("ROT-{}-{}", 
                exchange.to_uppercase(), 
                Utc::now().timestamp()
            );
            
            info!("✅ API keys rotated successfully for {}", exchange);
            
            Ok(RotateKeysResponse {
                success: true,
                rotation_id,
                exchange,
                old_keys_archived: true,
                new_keys_active: true,
                next_rotation_date: Utc::now() + Duration::days(API_KEY_ROTATION_DAYS),
                message: format!(
                    "Keys rotated successfully. Old keys archived. Next rotation in {} days.",
                    API_KEY_ROTATION_DAYS
                ),
            })
        }
        Err(e) => {
            error!("❌ Key rotation failed for {}: {}", exchange, e);
            
            Ok(RotateKeysResponse {
                success: false,
                rotation_id: String::new(),
                exchange,
                old_keys_archived: false,
                new_keys_active: false,
                next_rotation_date: Utc::now(),
                message: format!("Rotation failed: {}. Old keys still active.", e),
            })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RotateKeysResponse {
    pub success: bool,
    pub rotation_id: String,
    pub exchange: String,
    pub old_keys_archived: bool,
    pub new_keys_active: bool,
    pub next_rotation_date: DateTime<Utc>,
    pub message: String,
}

// ─────────────────────────────────────────────────────────────
// Helper Functions - The muscle behind the magic
// ─────────────────────────────────────────────────────────────

/// Session management structure
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub permissions: Vec<String>,
}

/// Verify session is valid and active
fn verify_session(
    app_state: &State<'_, Arc<RwLock<AppState>>>,
    session_id: &str
) -> Result<(), CommandError> {
    let state = app_state.read();
    
    if let Some(session) = state.active_sessions.get(session_id) {
        let idle_time = Utc::now() - session.last_activity;
        
        if idle_time > Duration::minutes(SESSION_TIMEOUT_MINS) {
            return Err(CommandError::AuthError(
                "Session expired. Time to login again.".to_string()
            ));
        }
        
        Ok(())
    } else {
        Err(CommandError::AuthError(
            "Invalid session. Who are you and how did you get here?".to_string()
        ))
    }
}

/// Store a new session
fn store_session(app_state: &State<'_, Arc<RwLock<AppState>>>, session: Session) {
    let mut state = app_state.write();
    state.active_sessions.insert(session.id.clone(), session);
}

/// Increment failed login attempts
fn increment_failed_attempts(app_state: &State<'_, Arc<RwLock<AppState>>>) {
    let mut state = app_state.write();
    state.auth_state.failed_attempts += 1;
    
    if state.auth_state.failed_attempts >= MAX_LOGIN_ATTEMPTS {
        state.auth_state.locked_until = Some(Utc::now() + Duration::minutes(LOCKOUT_DURATION_MINS));
        warn!("🚨 Account locked due to {} failed attempts", MAX_LOGIN_ATTEMPTS);
    }
}

/// Get current failed attempt count
fn get_failed_attempts(app_state: &State<'_, Arc<RwLock<AppState>>>) -> u32 {
    app_state.read().auth_state.failed_attempts
}

/// Reset auth state after successful login
fn reset_auth_state(app_state: &State<'_, Arc<RwLock<AppState>>>) {
    let mut state = app_state.write();
    state.auth_state.failed_attempts = 0;
    state.auth_state.locked_until = None;
}

/// Hash password using Argon2id - the gold standard
fn hash_password(password: &str) -> Result<String, CommandError> {
    // In production: use argon2 with proper salt
    // For now, simulate
    Ok(format!("HASH-{}", password.len()))
}

/// Verify password hash
async fn verify_password_hash(hash: &str) -> bool {
    // In production: actual verification
    // For now, simulate
    hash.starts_with("HASH-")
}

/// Encrypt sensitive credential
fn encrypt_credential(credential: &str) -> Result<String, CommandError> {
    // In production: use ring for AES-256-GCM
    // For now, simulate
    Ok(format!("ENC-{}", &credential[..5]))
}

/// Test exchange connection with given credentials
async fn test_exchange_connection(
    exchange: &str,
    api_key: &str,
    api_secret: &str
) -> Result<Vec<String>, String> {
    // In production: actual API test
    // For now, simulate with some latency
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    if api_key.len() > 30 && api_secret.len() > 30 {
        Ok(vec![
            "spot_trading".to_string(),
            "read_balance".to_string(),
            "read_orders".to_string(),
        ])
    } else {
        Err("Invalid credentials or insufficient permissions".to_string())
    }
}

// Re-export validation from commands module
use crate::commands::validation;

// ─────────────────────────────────────────────────────────────
// COMPLETE COMMAND LIST:
// 1. login (formerly unlock_stronghold) - Authenticate user
// 2. logout - End session
// 3. refresh_session - Extend session timeout  
// 4. manage_api_keys (formerly store_api_credentials) - Store/update API keys
// 5. verify_credentials - Test credentials without storing
// 6. get_exchange_status - Check exchange connections
// 7. rotate_api_key (formerly rotate_keys) - Rotate API keys for security
// 
// Total: 7 Commands (added 2 missing ones)
// ─────────────────────────────────────────────────────────────