#!/usr/bin/env python3
"""
Nexlify Nuclear Fix - When subtle doesn't work, go direct
Applies EXACTLY what the compiler is asking for
"""

import os
from pathlib import Path

class NuclearFix:
    def __init__(self):
        self.root = Path(".")
        self.src_tauri = self.root / "src-tauri"
        
    def nuke_it_all(self):
        print("☢️ NUCLEAR OPTION ENGAGED")
        print("=" * 50)
        
        # Fix 1: Rand imports - use EXACTLY what compiler suggests
        self.fix_rand_nuclear()
        
        # Fix 2: get_ticker - completely rewrite it
        self.fix_get_ticker_nuclear()
        
        # Fix 3: Update Cargo.toml with proper rand features
        self.fix_cargo_toml()
        
        print("\n💀 Nuclear fixes deployed. Rebuilding...")
    
    def fix_rand_nuclear(self):
        """Use exactly what the compiler tells us"""
        print("\n🎯 Fixing rand imports with compiler suggestions...")
        
        auth_path = self.src_tauri / "src" / "commands" / "auth.rs"
        if not auth_path.exists():
            print("  ❌ auth.rs not found!")
            return
            
        content = auth_path.read_text(encoding='utf-8')
        
        # Replace the import with EXACTLY what compiler wants
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            if line.strip() == 'use rand::Rng;':
                # Replace with what compiler suggests
                new_lines.append('use ::rand::Rng;')
                new_lines.append('use ::rand::thread_rng;')
                print("  ✅ Fixed imports with :: prefix")
            elif 'rand::thread_rng()' in line:
                # Remove rand:: prefix since we're importing directly
                new_line = line.replace('rand::thread_rng()', 'thread_rng()')
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        auth_path.write_text(content, encoding='utf-8')
        print("  ✅ Fixed all thread_rng calls")
    
    def fix_get_ticker_nuclear(self):
        """Completely rewrite get_ticker to handle DashMap properly"""
        print("\n🎯 Nuclear rewrite of get_ticker...")
        
        market_cache_path = self.src_tauri / "src" / "state" / "market_cache.rs"
        if not market_cache_path.exists():
            print("  ❌ market_cache.rs not found!")
            return
            
        content = market_cache_path.read_text(encoding='utf-8')
        
        # Find and replace the entire get_ticker function
        lines = content.split('\n')
        new_lines = []
        in_get_ticker = False
        brace_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if 'pub fn get_ticker(&self, symbol: &str) -> Option<Ticker>' in line:
                # Start of function
                in_get_ticker = True
                new_lines.append(line)
                
                # Write new implementation
                new_lines.append('        // DashMap stores Arc<RwLock<T>>, we need to extract T')
                new_lines.append('        self.tickers.get(symbol)')
                new_lines.append('            .and_then(|entry| {')
                new_lines.append('                // entry is a RefMulti which derefs to Arc<RwLock<Ticker>>')
                new_lines.append('                // We need to clone the Arc, then try to read the lock')
                new_lines.append('                let arc_clone = entry.clone();')
                new_lines.append('                drop(entry); // Release the DashMap lock')
                new_lines.append('                ')
                new_lines.append('                // Now try to read the RwLock')
                new_lines.append('                arc_clone.try_read().ok().map(|guard| guard.clone())')
                new_lines.append('            })')
                
                # Skip the old implementation
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    if '{' in lines[i]:
                        brace_count += lines[i].count('{')
                    if '}' in lines[i]:
                        brace_count -= lines[i].count('}')
                    i += 1
                new_lines.append('    }')
                in_get_ticker = False
                continue
            else:
                new_lines.append(line)
            
            i += 1
        
        # Also ensure we have the right imports
        content = '\n'.join(new_lines)
        
        # Check if we need RwLock trait
        if 'use parking_lot::' not in content:
            # Add after other use statements
            use_lines = []
            other_lines = []
            for line in content.split('\n'):
                if line.startswith('use ') and not other_lines:
                    use_lines.append(line)
                else:
                    other_lines.append(line)
            
            # Add parking_lot import
            use_lines.append('use parking_lot::RwLock;')
            content = '\n'.join(use_lines + other_lines)
        
        market_cache_path.write_text(content, encoding='utf-8')
        print("  ✅ Rewrote get_ticker with proper DashMap handling")
    
    def fix_cargo_toml(self):
        """Ensure rand has the right features"""
        print("\n🎯 Fixing Cargo.toml rand features...")
        
        cargo_path = self.src_tauri / "Cargo.toml"
        if not cargo_path.exists():
            print("  ❌ Cargo.toml not found!")
            return
            
        content = cargo_path.read_text(encoding='utf-8')
        
        # Replace simple rand with featured version
        if 'rand = "0.8"' in content:
            content = content.replace(
                'rand = "0.8"',
                'rand = { version = "0.8", features = ["std", "std_rng"] }'
            )
            print("  ✅ Updated rand with required features")
        else:
            print("  ℹ️  Rand already has features or different version")
        
        cargo_path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    print("⚠️  NUCLEAR OPTION - LAST RESORT")
    print("This will forcefully fix the remaining errors")
    
    nuke = NuclearFix()
    nuke.nuke_it_all()
    
    print("\n🚀 Now run:")
    print("  cd src-tauri")
    print("  cargo clean")
    print("  cargo build --release")
