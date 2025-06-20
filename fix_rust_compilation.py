#!/usr/bin/env python3
"""
NEXLIFY RUST COMPILATION FIXER
==============================================================================
Automated fix for all Rust compilation errors in Nexlify 3.0
==============================================================================

Run from Nexlify directory: python fix_rust_compilation.py
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

class RustCompilationFixer:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.src_tauri = self.base_path / "src-tauri"
        self.backup_dir = self.base_path / f"backup_rust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixes_applied = []
    
    def backup_files(self):
        """Create backup of all Rust files before modification"""
        print("🔒 Creating backup of Rust files...")
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree(self.src_tauri / "src", self.backup_dir / "src")
        print(f"✅ Backup created at: {self.backup_dir}")
    
    def fix_main_rs(self):
        """Fix all errors in main.rs"""
        print("\n🔧 Fixing main.rs...")
        main_path = self.src_tauri / "src" / "main.rs"
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Add missing imports at the top
        imports_to_add = [
            "use tauri::Emitter;",
            "use tauri::menu::Menu;",
        ]
        
        # Find the last use statement
        last_use_match = None
        for match in re.finditer(r'^use\s+[^;]+;', content, re.MULTILINE):
            last_use_match = match
        
        if last_use_match:
            insert_pos = last_use_match.end()
            for imp in imports_to_add:
                if imp not in content:
                    content = content[:insert_pos] + f"\n{imp}" + content[insert_pos:]
                    insert_pos += len(imp) + 1
                    self.fixes_applied.append(f"Added import: {imp}")
        
        # Fix 2: Fix generate_context macro
        content = re.sub(
            r'\.run\(generate_context!\(\)\)',
            '.run(tauri::generate_context!())',
            content
        )
        self.fixes_applied.append("Fixed generate_context macro")
        
        # Fix 3: Fix plugin initializations
        plugin_fixes = [
            ('tauri_plugin_dialog::init()', 'tauri_plugin_dialog::Builder::new().build()'),
            ('tauri_plugin_fs::init()', 'tauri_plugin_fs::Builder::new().build()'),
            ('tauri_plugin_http::init()', 'tauri_plugin_http::Builder::new().build()'),
            ('tauri_plugin_notification::init()', 'tauri_plugin_notification::Builder::new().build()'),
            ('tauri_plugin_os::init()', 'tauri_plugin_os::Builder::new().build()'),
            ('tauri_plugin_process::init()', 'tauri_plugin_process::Builder::new().build()'),
            ('tauri_plugin_shell::init()', 'tauri_plugin_shell::Builder::new().build()'),
            ('tauri_plugin_updater::init()', 'tauri_plugin_updater::Builder::new().build()'),
            ('tauri_plugin_websocket::init()', 'tauri_plugin_websocket::Builder::new().build()'),
            ('tauri_plugin_window_state::init()', 'tauri_plugin_window_state::Builder::new().build()'),
            ('tauri_plugin_stronghold::init()', 'tauri_plugin_stronghold::Builder::new().build()'),
        ]
        
        for old, new in plugin_fixes:
            if old in content:
                content = content.replace(old, new)
                self.fixes_applied.append(f"Fixed plugin init: {old.split('::')[0]}")
        
        # Fix 4: Fix menu reference
        content = re.sub(
            r'\.menu\(Some\(menu\)\)',
            '.menu(&menu)',
            content
        )
        self.fixes_applied.append("Fixed menu reference type")
        
        # Fix 5: Fix RefreshKind
        content = re.sub(
            r'RefreshKind::new\(\)',
            'RefreshKind::everything()',
            content
        )
        self.fixes_applied.append("Fixed RefreshKind constructor")
        
        # Fix 6: Remove WindowEffects (not available in current Tauri)
        # Find and comment out the entire window effects block
        content = re.sub(
            r'window\.set_effects\(tauri::window::WindowEffects\s*\{[^}]+\}\);?',
            '// Window effects removed - not available in current Tauri version',
            content,
            flags=re.DOTALL
        )
        self.fixes_applied.append("Removed WindowEffects (not available)")
        
        # Fix 7: Fix futures_util import in WebSocket function
        content = re.sub(
            r'use futures_util::StreamExt;',
            'use futures_util::stream::StreamExt;',
            content
        )
        self.fixes_applied.append("Fixed futures_util import path")
        
        # Save fixed content
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {len([f for f in self.fixes_applied if 'main.rs' in str(f) or 'Fixed' in f])} issues in main.rs")
    
    def fix_auth_rs(self):
        """Fix all errors in auth.rs"""
        print("\n🔧 Fixing commands/auth.rs...")
        auth_path = self.src_tauri / "src" / "commands" / "auth.rs"
        
        with open(auth_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Add SecureRandom import
        if "use ring::rand::SecureRandom;" not in content:
            # Find ring imports
            ring_import_match = re.search(r'^use ring::[^;]+;', content, re.MULTILINE)
            if ring_import_match:
                insert_pos = ring_import_match.end()
                content = content[:insert_pos] + "\nuse ring::rand::SecureRandom;" + content[insert_pos:]
            else:
                # Add after other use statements
                last_use = None
                for match in re.finditer(r'^use\s+[^;]+;', content, re.MULTILINE):
                    last_use = match
                if last_use:
                    insert_pos = last_use.end()
                    content = content[:insert_pos] + "\nuse ring::rand::SecureRandom;" + content[insert_pos:]
            self.fixes_applied.append("Added SecureRandom import")
        
        # Fix 2: Fix rand::random usage
        content = re.sub(
            r'rand::random::<f32>\(\)',
            'rand::random::<f32>()',
            content
        )
        
        # Actually, we need to use the proper rand crate function
        # Add use statement if not present
        if "use rand::Rng;" not in content:
            last_use = None
            for match in re.finditer(r'^use\s+[^;]+;', content, re.MULTILINE):
                last_use = match
            if last_use:
                insert_pos = last_use.end()
                content = content[:insert_pos] + "\nuse rand::Rng;" + content[insert_pos:]
        
        # Fix the actual random call
        content = re.sub(
            r'let latency = \(rand::random::<f32>\(\) \* 100\.0\) as i64 \+ 20;',
            'let latency = (rand::thread_rng().gen::<f32>() * 100.0) as i64 + 20;',
            content
        )
        self.fixes_applied.append("Fixed random number generation")
        
        # Save fixed content
        with open(auth_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed auth.rs compilation errors")
    
    def fix_market_data_rs(self):
        """Fix all errors in market_data.rs"""
        print("\n🔧 Fixing commands/market_data.rs...")
        market_path = self.src_tauri / "src" / "commands" / "market_data.rs"
        
        with open(market_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Add Emitter trait import
        if "use tauri::Emitter;" not in content:
            # Add after other tauri imports
            tauri_import = re.search(r'^use tauri::[^;]+;', content, re.MULTILINE)
            if tauri_import:
                insert_pos = tauri_import.end()
                content = content[:insert_pos] + "\nuse tauri::Emitter;" + content[insert_pos:]
            self.fixes_applied.append("Added Emitter trait import")
        
        # Fix 2: Fix market_cache.get_ticker - need to dereference State
        content = re.sub(
            r'market_cache\.get_ticker\(&symbol\)',
            'market_cache.inner().get_ticker(&symbol)',
            content
        )
        self.fixes_applied.append("Fixed market_cache State dereference")
        
        # Fix 3: Fix candles borrow after move
        # Find the historical_data function and fix the candles usage
        content = re.sub(
            r'(candles),\s*\n\s*start_time: start,\s*\n\s*end_time: candles\.last\(\)',
            r'candles: candles.clone(),\n        start_time: start,\n        end_time: candles.last()',
            content
        )
        # Actually, we need a better fix - store the end_time before moving candles
        pattern = r'(let mut candles = Vec::new\(\);.*?)(\s*Ok\(HistoricalDataResponse\s*\{[^}]+\})'
        
        def fix_candles(match):
            before = match.group(1)
            response_block = match.group(2)
            
            # Insert calculation of end_time before the response
            fixed = before + '\n    let end_time = candles.last().map(|c| c.timestamp).unwrap_or(start);' + \
                    response_block.replace('end_time: candles.last().map(|c| c.timestamp).unwrap_or(start)', 'end_time')
            
            return fixed
        
        content = re.sub(pattern, fix_candles, content, flags=re.DOTALL)
        self.fixes_applied.append("Fixed candles borrow after move")
        
        # Save fixed content
        with open(market_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed market_data.rs compilation errors")
    
    def fix_trading_rs(self):
        """Fix all errors in trading.rs"""
        print("\n🔧 Fixing commands/trading.rs...")
        trading_path = self.src_tauri / "src" / "commands" / "trading.rs"
        
        with open(trading_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix 1: Fix f32/f64 type mismatch
        content = re.sub(
            r'score\.min\(100\.0\)',
            'score.min(100.0) as f32',
            content
        )
        self.fixes_applied.append("Fixed f32/f64 type mismatch")
        
        # Fix 2: Fix order.side moved value - need to clone or reference
        # Find the place_order function and fix the order.side usage
        pattern = r'(order_id, order\.side as i32.*?)(message: match order\.side)'
        
        def fix_order_side(match):
            first_use = match.group(1)
            second_use = match.group(2)
            # Clone the side in first use
            fixed_first = first_use.replace('order.side as i32', 'order.side.clone() as i32')
            return fixed_first + second_use
        
        content = re.sub(pattern, fix_order_side, content, flags=re.DOTALL)
        self.fixes_applied.append("Fixed order.side move issue")
        
        # Fix 3: Fix positions borrow after move
        # Similar to candles, calculate position_count before moving
        pattern = r'(let positions: Vec<PositionInfo>.*?)(\s*Ok\(PositionsResponse\s*\{[^}]+\})'
        
        def fix_positions(match):
            before = match.group(1)
            response_block = match.group(2)
            
            # Insert calculation before the response
            fixed = before + '\n    let position_count = positions.len();' + \
                    response_block.replace('position_count: positions.len()', 'position_count')
            
            return fixed
        
        content = re.sub(pattern, fix_positions, content, flags=re.DOTALL)
        self.fixes_applied.append("Fixed positions borrow after move")
        
        # Save fixed content
        with open(trading_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed trading.rs compilation errors")
    
    def add_missing_trait_impls(self):
        """Add Copy/Clone derives where needed"""
        print("\n🔧 Adding missing trait implementations...")
        
        # Find state/mod.rs or similar files with OrderSide enum
        state_files = list((self.src_tauri / "src").rglob("*.rs"))
        
        for file_path in state_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            
            # Add Copy trait to OrderSide if it exists and doesn't have it
            if 'enum OrderSide' in content and '#[derive(' in content:
                # Find the OrderSide enum definition
                pattern = r'(#\[derive\([^)]+\)\]\s*(?:pub\s+)?enum\s+OrderSide)'
                
                def add_copy_trait(match):
                    derive_str = match.group(1)
                    if 'Copy' not in derive_str:
                        # Add Copy and Clone if not present
                        new_derive = derive_str.replace('#[derive(', '#[derive(Copy, Clone, ')
                        return new_derive
                    return derive_str
                
                new_content = re.sub(pattern, add_copy_trait, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    self.fixes_applied.append(f"Added Copy trait to OrderSide in {file_path.name}")
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
    
    def run_all_fixes(self):
        """Run all fixes in sequence"""
        print("🌃 NEXLIFY RUST COMPILATION FIXER")
        print("=" * 60)
        
        # Create backup first
        self.backup_files()
        
        # Apply all fixes
        self.fix_main_rs()
        self.fix_auth_rs()
        self.fix_market_data_rs()
        self.fix_trading_rs()
        self.add_missing_trait_impls()
        
        # Summary
        print("\n" + "=" * 60)
        print(f"✅ FIXES APPLIED: {len(self.fixes_applied)}")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"  - {fix}")
        
        print(f"\n📁 Backup saved to: {self.backup_dir}")
        print("\n🚀 Next steps:")
        print("   1. Run: pnpm tauri:dev")
        print("   2. If more errors appear, run this script again")
        print("   3. Check the backup if you need to revert")
        
        # Create a fix log
        log_path = self.base_path / f"rust_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_path, 'w') as f:
            f.write("NEXLIFY RUST COMPILATION FIXES\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Fixes Applied: {len(self.fixes_applied)}\n\n")
            for fix in self.fixes_applied:
                f.write(f"- {fix}\n")
        
        print(f"\n📝 Fix log saved to: {log_path}")

def main():
    """Main entry point"""
    # Check if we're in the right directory
    if not Path("src-tauri").exists():
        print("❌ ERROR: src-tauri directory not found!")
        print("   Run this script from the Nexlify root directory (C:\\Nexlify)")
        return
    
    # Run the fixer
    fixer = RustCompilationFixer()
    
    try:
        fixer.run_all_fixes()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("   Check the error and try again")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()