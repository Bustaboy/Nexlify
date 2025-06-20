// src-tauri/src/error.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum CommandError {
    Auth(String),
    Network(String),
    Validation(String),
    NotFound(String),
    Crypto(String),
    Stronghold(String),
    Internal(String),
    NotImplemented(String),
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandError::Auth(msg) => write!(f, "Auth error: {}", msg),
            CommandError::Network(msg) => write!(f, "Network error: {}", msg),
            CommandError::Validation(msg) => write!(f, "Validation error: {}", msg),
            CommandError::NotFound(msg) => write!(f, "Not found: {}", msg),
            CommandError::Crypto(msg) => write!(f, "Crypto error: {}", msg),
            CommandError::Stronghold(msg) => write!(f, "Stronghold error: {}", msg),
            CommandError::Internal(msg) => write!(f, "Internal error: {}", msg),
            CommandError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl std::error::Error for CommandError {}

#[derive(Debug)]
pub enum StateError {
    InvalidState(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for StateError {}
