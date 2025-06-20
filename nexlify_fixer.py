#!/usr/bin/env python3
"""
Nexlify 3.0 Neural Repair Script - Surgical Chrome Fixes
Patches specific compilation errors without nuking your codebase
Netrunner: Bustaboy | Date: 2025-06-20
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

class NexlifyNeuralRepair:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_tauri = self.project_root / "src-tauri"
        self.backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixes_applied = []
        
    def create_backup(self):
        """Create backup before surgery - always have an exit strategy"""
        print("🔒 Creating neural backup...")
        critical_files = [
            "src-tauri/src/main.rs",
            "src-tauri/src/commands/auth.rs",
            "src-tauri/src/commands/mod.rs",
            "src-tauri/src/state/mod.rs",
            "src-tauri/src/state/market_cache.rs",
            "src-tauri/Cargo.toml",
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                backup_path = self.backup_dir / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, backup_path)
                print(f"  ✓ Backed up: {file_path}")
    
    def fix_main_rs(self):
        """Fix main.rs Tauri plugin initialization errors"""
        main_path = self.src_tauri / "src" / "main.rs"
        if not main_path.exists():
            print("❌ main.rs not found!")
            return
            
        print("\n🔧 Fixing main.rs neural pathways...")
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Missing build script OUT_DIR error
        if "tauri::generate_context!" in content and not (self.src_tauri / "build.rs").exists():
            print("  → Creating build.rs for Tauri context generation")
            build_rs = '''fn main() {
    tauri_build::build();
}
'''
            with open(self.src_tauri / "build.rs", 'w', encoding='utf-8') as f:
                f.write(build_rs)
            self.fixes_applied.append("Created build.rs")
        
        # Fix 2: Plugin Builder private struct errors
        plugin_fixes = [
            ('tauri_plugin_shell::Builder::new()', 'tauri_plugin_shell::init()'),
            ('tauri_plugin_process::Builder::new()', 'tauri_plugin_process::init()'),
            ('tauri_plugin_os::Builder::new()', 'tauri_plugin_os::init()'),
            ('tauri_plugin_http::Builder::new()', 'tauri_plugin_http::init()'),
            ('tauri_plugin_notification::Builder::new()', 'tauri_plugin_notification::init()'),
            ('tauri_plugin_dialog::Builder::new()', 'tauri_plugin_dialog::init()'),
            ('tauri_plugin_fs::Builder::new()', 'tauri_plugin_fs::init()'),
        ]
        
        for old_call, new_call in plugin_fixes:
            if old_call in content:
                content = re.sub(
                    rf'\.plugin\({re.escape(old_call)}\.build\(\)\)',
                    f'.plugin({new_call})',
                    content
                )
                self.fixes_applied.append(f"Fixed {old_call}")
        
        # Fix 3: Stronghold plugin initialization
        if 'tauri_plugin_stronghold::Builder::new()' in content:
            stronghold_fix = '''tauri_plugin_stronghold::Builder::new(|password| {
                // Use tauri's built-in password hashing
                use argon2::{Argon2, PasswordHasher};
                use argon2::password_hash::{rand_core::OsRng, SaltString};
                
                let salt = SaltString::generate(&mut OsRng);
                let argon2 = Argon2::default();
                
                argon2
                    .hash_password(password.as_bytes(), &salt)
                    .map(|hash| hash.to_string().into_bytes())
                    .unwrap_or_else(|_| password.as_bytes().to_vec())
            })'''
            
            content = re.sub(
                r'tauri_plugin_stronghold::Builder::new\(\)',
                stronghold_fix,
                content
            )
            self.fixes_applied.append("Fixed Stronghold initialization")
        
        # Fix 4: Remove unused imports
        if 'use tauri::generate_context;' in content:
            content = content.replace('use tauri::generate_context;', '')
            self.fixes_applied.append("Removed unused generate_context import")
        
        if content != original_content:
            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ main.rs neural pathways repaired")
        else:
            print("  ℹ main.rs already optimal")
    
    def fix_auth_rs(self):
        """Fix auth.rs rand::Rng import and mock implementations"""
        auth_path = self.src_tauri / "src" / "commands" / "auth.rs"
        if not auth_path.exists():
            print("❌ auth.rs not found!")
            return
            
        print("\n🔧 Fixing auth.rs security protocols...")
        
        with open(auth_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Missing rand::Rng trait import
        if 'use rand::Rng;' in content and 'use rand::{' not in content:
            # Replace the import with the correct one
            content = re.sub(
                r'use rand::Rng;',
                'use rand::{RngCore, Rng as RandRng};',
                content
            )
            # Update usage
            content = content.replace('rand::thread_rng()', 'rand::thread_rng()')
            content = content.replace('.gen::<f32>()', '.gen::<f32>()')
            self.fixes_applied.append("Fixed rand::Rng import")
        
        # Fix 2: Add generate_session_id if missing
        if 'fn generate_session_id()' not in content:
            session_id_impl = '''
fn generate_session_id() -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    let mut rng = rand::thread_rng();
    let session_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
    BASE64.encode(&session_bytes)
}
'''
            # Add before the last closing brace
            content = content.rstrip() + '\n' + session_id_impl + '\n'
            self.fixes_applied.append("Added generate_session_id function")
        
        if content != original_content:
            with open(auth_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ auth.rs security protocols updated")
        else:
            print("  ℹ auth.rs already secure")
    
    def fix_state_mod(self):
        """Fix state/mod.rs duplicate Clone derive"""
        state_path = self.src_tauri / "src" / "state" / "mod.rs"
        if not state_path.exists():
            print("❌ state/mod.rs not found!")
            return
            
        print("\n🔧 Fixing state module neural memory...")
        
        with open(state_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix duplicate Clone derive
        content = re.sub(
            r'#\[derive\((.*?)Clone(.*?)Clone(.*?)\)\]',
            r'#[derive(\1Clone\2\3)]',
            content
        )
        
        if content != original_content:
            with open(state_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ Removed duplicate Clone derive")
            self.fixes_applied.append("Fixed duplicate Clone derive")
        else:
            print("  ℹ state/mod.rs already clean")
    
    def fix_market_cache(self):
        """Fix market_cache.rs missing get_ticker method"""
        cache_path = self.src_tauri / "src" / "state" / "market_cache.rs"
        if not cache_path.exists():
            print("❌ market_cache.rs not found!")
            return
            
        print("\n🔧 Fixing market cache data matrix...")
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if get_ticker is missing
        if 'fn get_ticker' not in content and 'struct MarketCache' in content:
            # Find the impl block
            impl_match = re.search(r'impl\s+MarketCache\s*\{', content)
            if impl_match:
                # Add get_ticker method after the opening brace
                insert_pos = impl_match.end()
                ticker_method = '''
    pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        self.tickers.get(symbol).map(|entry| entry.clone())
    }
'''
                content = content[:insert_pos] + ticker_method + content[insert_pos:]
                self.fixes_applied.append("Added get_ticker method")
        
        if content != original_content:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ Market cache methods synchronized")
        else:
            print("  ℹ market_cache.rs already complete")
    
    def fix_cargo_toml(self):
        """Fix Cargo.toml missing dependencies"""
        cargo_path = self.src_tauri / "Cargo.toml"
        if not cargo_path.exists():
            print("❌ Cargo.toml not found!")
            return
            
        print("\n🔧 Updating Cargo.toml dependency matrix...")
        
        with open(cargo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Dependencies we need to ensure exist
        required_deps = {
            'rand': '"0.8"',
            'hmac': '"0.12"',
            'sha2': '"0.10"', 
            'hex': '"0.4"',
            'reqwest': '{ version = "0.12", features = ["json", "rustls-tls"] }',
            'argon2': '"0.5"',
        }
        
        # Build dependencies we need
        if '[build-dependencies]' not in content:
            content += '\n[build-dependencies]\ntauri-build = { version = "2.1.1", features = [] }\n'
            self.fixes_applied.append("Added build-dependencies section")
        elif 'tauri-build' not in content:
            # Add tauri-build to existing build-dependencies
            content = re.sub(
                r'\[build-dependencies\]',
                '[build-dependencies]\ntauri-build = { version = "2.1.1", features = [] }',
                content
            )
            self.fixes_applied.append("Added tauri-build dependency")
        
        # Add missing runtime dependencies
        for dep, version in required_deps.items():
            if f'{dep} =' not in content:
                # Find [dependencies] section
                dep_match = re.search(r'\[dependencies\]\n', content)
                if dep_match:
                    insert_pos = dep_match.end()
                    content = content[:insert_pos] + f'{dep} = {version}\n' + content[insert_pos:]
                    self.fixes_applied.append(f"Added {dep} dependency")
        
        if content != original_content:
            with open(cargo_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ Dependency matrix updated")
        else:
            print("  ℹ Cargo.toml already synchronized")
    
    def create_error_module(self):
        """Create error.rs if it doesn't exist"""
        error_path = self.src_tauri / "src" / "error.rs"
        if error_path.exists():
            print("\n  ℹ error.rs already exists")
            return
            
        print("\n🔧 Creating error handling module...")
        
        error_content = '''// src-tauri/src/error.rs
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
'''
        
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(error_content)
        
        # Also update main.rs to include the error module
        main_path = self.src_tauri / "src" / "main.rs"
        if main_path.exists():
            with open(main_path, 'r', encoding='utf-8') as f:
                main_content = f.read()
            
            if 'mod error;' not in main_content:
                # Add after other mod declarations
                main_content = re.sub(
                    r'(mod state;)',
                    r'\1\nmod error;',
                    main_content
                )
                with open(main_path, 'w', encoding='utf-8') as f:
                    f.write(main_content)
        
        print("  ✓ Error module created")
        self.fixes_applied.append("Created error.rs module")
    
    def run_repair(self):
        """Execute all neural repairs"""
        print("🌃 NEXLIFY NEURAL REPAIR PROTOCOL INITIATED")
        print("=" * 50)
        
        # Create backup first
        self.create_backup()
        
        # Run all fixes
        self.fix_cargo_toml()  # Fix dependencies first
        self.create_error_module()  # Create error module before main.rs
        self.fix_main_rs()
        self.fix_auth_rs()
        self.fix_state_mod()
        self.fix_market_cache()
        
        # Summary
        print("\n" + "=" * 50)
        print("🎯 REPAIR SUMMARY:")
        if self.fixes_applied:
            print(f"  ✓ {len(self.fixes_applied)} neural pathways repaired:")
            for fix in self.fixes_applied:
                print(f"    • {fix}")
        else:
            print("  ℹ All systems already optimal - no repairs needed")
        
        print(f"\n💾 Backup created at: {self.backup_dir}")
        print("\n🚀 Next steps:")
        print("  1. cd src-tauri && cargo clean")
        print("  2. cargo build --release")
        print("  3. cd .. && pnpm tauri:dev")
        
        return len(self.fixes_applied) > 0

if __name__ == "__main__":
    import sys
    
    # Check if we're in the right directory
    if not os.path.exists("src-tauri"):
        print("❌ Error: Must run from Nexlify project root (no src-tauri found)")
        print("   Usage: python nexlify_fixer.py")
        sys.exit(1)
    
    # Run the repair protocol
    repair = NexlifyNeuralRepair()
    success = repair.run_repair()
    
    sys.exit(0 if success else 1)
