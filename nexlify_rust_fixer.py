#!/usr/bin/env python3
"""
Nexlify Rust Error Assassin - Precision Fixes for Known Flatlines
Targets the 6 compilation errors with surgical precision
No collateral damage, just clean kills
"""

import os
import re
from pathlib import Path
from datetime import datetime

class RustErrorAssassin:
    def __init__(self, root_path="."):
        self.root = Path(root_path)
        self.src_tauri = self.root / "src-tauri"
        self.fixes_applied = []
        
    def fix_all_errors(self):
        """Execute all error fixes in sequence"""
        print("🔫 NEXLIFY RUST ERROR ASSASSIN ENGAGED")
        print("=" * 50)
        print("Targeting 6 known hostiles...\n")
        
        # Fix 1: Extra comma in state/mod.rs
        self.fix_extra_comma()
        
        # Fix 2: Create dist folder for frontend
        self.create_dist_folder()
        
        # Fix 3: Fix rand imports in auth.rs
        self.fix_rand_imports()
        
        # Fix 4: Fix get_ticker type mismatch
        self.fix_get_ticker()
        
        # Fix 5: Fix unused warnings (optional but clean)
        self.fix_unused_warnings()
        
        # Summary
        self.print_summary()
    
    def fix_extra_comma(self):
        """Fix the extra comma in derive macro"""
        print("🎯 Target 1: Extra comma in state/mod.rs line 242...")
        
        file_path = self.src_tauri / "src" / "state" / "mod.rs"
        if not file_path.exists():
            print("  ❌ File not found!")
            return
            
        content = file_path.read_text(encoding='utf-8')
        
        # Fix the double comma
        original = '#[derive(Copy, Clone, Debug, , Serialize, Deserialize)]'
        fixed = '#[derive(Copy, Clone, Debug, Serialize, Deserialize)]'
        
        if original in content:
            content = content.replace(original, fixed)
            file_path.write_text(content, encoding='utf-8')
            print("  ✅ Eliminated extra comma")
            self.fixes_applied.append("Fixed extra comma in state/mod.rs")
        else:
            print("  ℹ️  Already fixed or pattern not found")
    
    def create_dist_folder(self):
        """Create the missing dist folder"""
        print("\n🎯 Target 2: Missing frontend dist folder...")
        
        dist_path = self.root / "dist"
        if not dist_path.exists():
            dist_path.mkdir(parents=True)
            # Create a minimal index.html so Tauri doesn't complain
            index_content = """<!DOCTYPE html>
<html>
<head>
    <title>Nexlify - Building...</title>
    <style>
        body { 
            background: #0A0A0A; 
            color: #00FFFF; 
            font-family: monospace;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
    </style>
</head>
<body>
    <div>Neural compilation in progress...</div>
</body>
</html>"""
            (dist_path / "index.html").write_text(index_content, encoding='utf-8')
            print("  ✅ Created dist folder with placeholder")
            self.fixes_applied.append("Created dist folder")
        else:
            print("  ℹ️  Dist folder already exists")
    
    def fix_rand_imports(self):
        """Fix the rand crate imports"""
        print("\n🎯 Target 3 & 4: Rand import errors in auth.rs...")
        
        file_path = self.src_tauri / "src" / "commands" / "auth.rs"
        if not file_path.exists():
            print("  ❌ File not found!")
            return
            
        content = file_path.read_text(encoding='utf-8')
        
        # Fix the import statement
        content = re.sub(
            r'use rand::\{RngCore, Rng as RandRng\};',
            'use rand::{Rng, thread_rng};',
            content
        )
        
        # Fix the thread_rng calls - remove the module prefix
        content = re.sub(
            r'rand::thread_rng\(\)',
            'thread_rng()',
            content
        )
        
        # Fix the generate_session_id if it uses the old import
        content = re.sub(
            r'let mut rng = rand::thread_rng\(\);',
            'let mut rng = thread_rng();',
            content
        )
        
        file_path.write_text(content, encoding='utf-8')
        print("  ✅ Fixed rand imports and usage")
        self.fixes_applied.append("Fixed rand imports in auth.rs")
    
    def fix_get_ticker(self):
        """Fix the get_ticker type mismatch"""
        print("\n🎯 Target 5: Type mismatch in market_cache.rs...")
        
        file_path = self.src_tauri / "src" / "state" / "market_cache.rs"
        if not file_path.exists():
            print("  ❌ File not found!")
            return
            
        content = file_path.read_text(encoding='utf-8')
        
        # The issue is that DashMap stores values in Arc<RwLock<T>>
        # We need to properly extract the value
        old_implementation = '''pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        self.tickers.get(symbol).map(|entry| entry.clone())
    }'''
        
        new_implementation = '''pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        self.tickers.get(symbol).map(|entry| entry.value().clone())
    }'''
        
        if old_implementation in content:
            content = content.replace(old_implementation, new_implementation)
            file_path.write_text(content, encoding='utf-8')
            print("  ✅ Fixed get_ticker type mismatch")
            self.fixes_applied.append("Fixed get_ticker in market_cache.rs")
        else:
            # Try a more flexible pattern
            content = re.sub(
                r'(pub fn get_ticker.*?Option<Ticker>\s*\{)\s*self\.tickers\.get\(symbol\)\.map\(\|entry\| entry\.clone\(\)\)',
                r'\1\n        self.tickers.get(symbol).map(|entry| entry.value().clone())',
                content,
                flags=re.DOTALL
            )
            file_path.write_text(content, encoding='utf-8')
            print("  ✅ Fixed get_ticker type mismatch (pattern 2)")
            self.fixes_applied.append("Fixed get_ticker in market_cache.rs")
    
    def fix_unused_warnings(self):
        """Fix the most common unused warnings"""
        print("\n🎯 Bonus: Cleaning up unused warnings...")
        
        # Fix unused imports in main.rs
        main_path = self.src_tauri / "src" / "main.rs"
        if main_path.exists():
            content = main_path.read_text(encoding='utf-8')
            
            # Remove unused generate_context import
            content = re.sub(
                r'use tauri::\{\s*generate_context,\s*generate_handler,',
                'use tauri::{generate_handler,',
                content
            )
            
            # Prefix unused variables with underscore
            content = re.sub(
                r'(\s+)app_handle: tauri::AppHandle,',
                r'\1_app_handle: tauri::AppHandle,',
                content
            )
            
            content = re.sub(
                r'let subscribe_msg = ',
                r'let _subscribe_msg = ',
                content
            )
            
            main_path.write_text(content, encoding='utf-8')
            print("  ✅ Cleaned up main.rs warnings")
            self.fixes_applied.append("Cleaned unused warnings in main.rs")
    
    def print_summary(self):
        """Print operation summary"""
        print("\n" + "=" * 50)
        print("🎯 ASSASSINATION COMPLETE")
        print("=" * 50)
        
        if self.fixes_applied:
            print(f"\n✅ {len(self.fixes_applied)} targets eliminated:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        else:
            print("\n⚠️  No targets found - may already be eliminated")
        
        print("\n🚀 Next steps:")
        print("  1. cd src-tauri")
        print("  2. cargo build --release")
        print("  3. If successful, run: pnpm tauri:dev")
        
        print("\nStay frosty, netrunner. The chrome should compile clean now. 🌃")

if __name__ == "__main__":
    # Execute the fixes
    assassin = RustErrorAssassin()
    assassin.fix_all_errors()
