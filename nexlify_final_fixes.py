#!/usr/bin/env python3
"""
Nexlify Final Three Fixes - The Last Stand
Targets the remaining 3 compilation errors with extreme prejudice
"""

import os
import re
from pathlib import Path
import shutil

class FinalFixer:
    def __init__(self, root_path="."):
        self.root = Path(root_path)
        self.src_tauri = self.root / "src-tauri"
        
    def execute_final_fixes(self):
        """Hit the last three targets"""
        print("🎯 NEXLIFY FINAL FIXER - LAST THREE TARGETS")
        print("=" * 50)
        
        # Fix 1: Create missing tray icon
        self.fix_tray_icon()
        
        # Fix 2: Fix rand properly in Cargo.toml and imports
        self.fix_rand_properly()
        
        # Fix 3: Fix get_ticker with correct DashMap usage
        self.fix_get_ticker_correctly()
        
        print("\n✅ All fixes applied. Try building again!")
    
    def fix_tray_icon(self):
        """Create the missing tray icon"""
        print("\n🎯 Target 1: Missing tray-icon.png...")
        
        icons_dir = self.src_tauri / "icons"
        tray_icon_path = icons_dir / "tray-icon.png"
        
        # First check if any icon exists we can copy
        if icons_dir.exists():
            # Look for any existing PNG we can use
            existing_pngs = list(icons_dir.glob("*.png"))
            if existing_pngs:
                # Copy the first PNG as tray icon
                shutil.copy(existing_pngs[0], tray_icon_path)
                print(f"  ✅ Created tray-icon.png from {existing_pngs[0].name}")
                return
        
        # If no icons exist at all, we need to handle this differently
        # Check if tauri.conf.json references tray icon
        tauri_conf_path = self.src_tauri / "tauri.conf.json"
        if tauri_conf_path.exists():
            content = tauri_conf_path.read_text(encoding='utf-8')
            
            # Remove or comment out tray icon config temporarily
            if '"trayIcon"' in content:
                print("  ⚠️  Disabling tray icon in config temporarily...")
                # Comment out the entire trayIcon section
                content = re.sub(
                    r'"trayIcon":\s*\{[^}]*\},?\s*',
                    '// "trayIcon": { /* disabled until icons are generated */ },\n',
                    content,
                    flags=re.DOTALL
                )
                tauri_conf_path.write_text(content, encoding='utf-8')
                print("  ✅ Disabled tray icon config - run icon generator later")
            else:
                print("  ℹ️  No tray icon configured")
    
    def fix_rand_properly(self):
        """Fix rand in both Cargo.toml and imports"""
        print("\n🎯 Target 2: Fixing rand imports properly...")
        
        # First, check Cargo.toml has rand with correct features
        cargo_path = self.src_tauri / "Cargo.toml"
        if cargo_path.exists():
            cargo_content = cargo_path.read_text(encoding='utf-8')
            
            # Check if rand is properly configured
            if 'rand = "0.8"' in cargo_content or 'rand = { version = "0.8"' in cargo_content:
                print("  ℹ️  rand dependency found in Cargo.toml")
            else:
                # Need to ensure rand has the right features
                if 'rand =' in cargo_content:
                    # Update existing rand entry
                    cargo_content = re.sub(
                        r'rand = "[^"]*"',
                        'rand = { version = "0.8", features = ["std", "std_rng"] }',
                        cargo_content
                    )
                else:
                    # Shouldn't happen based on our previous fix, but just in case
                    print("  ⚠️  rand not found in Cargo.toml - this is unexpected!")
                
                cargo_path.write_text(cargo_content, encoding='utf-8')
                print("  ✅ Updated rand with correct features in Cargo.toml")
        
        # Now fix the import in auth.rs
        auth_path = self.src_tauri / "src" / "commands" / "auth.rs"
        if auth_path.exists():
            auth_content = auth_path.read_text(encoding='utf-8')
            
            # Replace the import line completely
            auth_content = re.sub(
                r'use rand::\{[^}]*\};',
                'use rand::Rng;',
                auth_content
            )
            
            # Fix all thread_rng usage to include rand::
            auth_content = re.sub(
                r'(?<!rand::)thread_rng\(\)',
                'rand::thread_rng()',
                auth_content
            )
            
            # Special case for the line that might have been changed
            auth_content = re.sub(
                r'let mut rng = thread_rng\(\);',
                'let mut rng = rand::thread_rng();',
                auth_content
            )
            
            auth_path.write_text(auth_content, encoding='utf-8')
            print("  ✅ Fixed rand imports in auth.rs")
    
    def fix_get_ticker_correctly(self):
        """Fix get_ticker with correct DashMap handling"""
        print("\n🎯 Target 3: Fixing get_ticker type mismatch...")
        
        market_cache_path = self.src_tauri / "src" / "state" / "market_cache.rs"
        if not market_cache_path.exists():
            print("  ❌ market_cache.rs not found!")
            return
        
        content = market_cache_path.read_text(encoding='utf-8')
        
        # The issue is that entry.value() returns a reference to the guard
        # We need to dereference and clone
        # Look for the get_ticker function and replace its implementation
        
        # More flexible pattern matching
        pattern = r'(pub fn get_ticker\(&self, symbol: &str\) -> Option<Ticker>\s*\{)[^}]*(})'
        
        replacement = r'''\1
        self.tickers.get(symbol).map(|entry| {
            let ticker = entry.value();
            ticker.clone()
        })
    \2'''
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Also check if we need to import anything
        if 'use dashmap::DashMap;' not in content and 'DashMap' in content:
            # Add the import after other imports
            content = re.sub(
                r'(use std::.*?;)\n',
                r'\1\nuse dashmap::DashMap;\n',
                content,
                count=1
            )
        
        market_cache_path.write_text(content, encoding='utf-8')
        print("  ✅ Fixed get_ticker implementation")

if __name__ == "__main__":
    fixer = FinalFixer()
    fixer.execute_final_fixes()
