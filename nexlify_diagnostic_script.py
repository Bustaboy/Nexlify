#!/usr/bin/env python3
"""
Nexlify Diagnostic Script - What's Really Broken?
Scans your setup and tells you exactly what needs fixing
No bullshit, just facts
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class NexlifyDiagnostics:
    def __init__(self, root_path="."):
        self.root = Path(root_path)
        self.issues = []
        self.warnings = []
        self.info = []
        
    def run_diagnostics(self):
        """Run all diagnostic checks"""
        print("🔍 NEXLIFY DIAGNOSTIC SCAN")
        print("=" * 50)
        
        # Check Tauri structure
        self.check_tauri_structure()
        
        # Check frontend structure
        self.check_frontend_structure()
        
        # Check dependencies
        self.check_dependencies()
        
        # Check for compilation issues
        self.check_rust_compilation()
        
        # Generate report
        self.generate_report()
        
    def check_tauri_structure(self):
        """Check if Tauri backend has all required files"""
        print("\n📁 Checking Tauri structure...")
        
        tauri_path = self.root / "src-tauri"
        
        # Required files
        required_files = {
            "Cargo.toml": "Rust dependencies",
            "build.rs": "Build script",
            "tauri.conf.json": "Tauri configuration",
            "src/main.rs": "Main entry point",
            "src/error.rs": "Error handling",
        }
        
        for file, desc in required_files.items():
            full_path = tauri_path / file
            if full_path.exists():
                self.info.append(f"✅ {file} exists - {desc}")
            else:
                self.issues.append(f"❌ Missing {file} - {desc}")
        
        # Check for icons
        icons_path = tauri_path / "icons"
        if not icons_path.exists():
            self.issues.append("❌ Missing icons directory")
        else:
            icon_files = list(icons_path.glob("*"))
            if not icon_files:
                self.issues.append("❌ Icons directory exists but is empty")
            else:
                required_icons = ["icon.ico", "icon.png", "icon.icns"]
                for icon in required_icons:
                    if not (icons_path / icon).exists():
                        self.warnings.append(f"⚠️ Missing {icon}")
        
        # Check command implementations
        commands_path = tauri_path / "src" / "commands"
        if commands_path.exists():
            for cmd_file in ["mod.rs", "auth.rs", "market_data.rs", "trading.rs"]:
                cmd_path = commands_path / cmd_file
                if cmd_path.exists():
                    # Check if file has actual implementations (not just empty)
                    content = cmd_path.read_text()
                    if "todo!()" in content.lower() or "unimplemented!()" in content:
                        self.warnings.append(f"⚠️ {cmd_file} contains unimplemented functions")
                    elif len(content.strip()) < 100:
                        self.warnings.append(f"⚠️ {cmd_file} might be incomplete (very small)")
                else:
                    self.issues.append(f"❌ Missing command file: {cmd_file}")
    
    def check_frontend_structure(self):
        """Check if frontend has required files"""
        print("\n📁 Checking frontend structure...")
        
        # Check main frontend files
        frontend_files = {
            "package.json": "Node.js dependencies",
            "index.html": "HTML entry point",
            "tsconfig.json": "TypeScript configuration",
            "vite.config.ts": "Vite bundler config",
        }
        
        for file, desc in frontend_files.items():
            if (self.root / file).exists():
                self.info.append(f"✅ {file} exists - {desc}")
            else:
                self.issues.append(f"❌ Missing {file} - {desc}")
        
        # Check src directory
        src_path = self.root / "src"
        if src_path.exists():
            # Check for main app files
            if (src_path / "App.tsx").exists():
                self.info.append("✅ App.tsx exists")
            else:
                self.issues.append("❌ Missing src/App.tsx")
                
            if (src_path / "main.tsx").exists() or (src_path / "index.tsx").exists():
                self.info.append("✅ Entry point exists")
            else:
                self.warnings.append("⚠️ No main.tsx or index.tsx found")
    
    def check_dependencies(self):
        """Check if dependencies are installed"""
        print("\n📦 Checking dependencies...")
        
        # Check Node modules
        if (self.root / "node_modules").exists():
            self.info.append("✅ node_modules exists")
        else:
            self.issues.append("❌ node_modules missing - run 'pnpm install'")
        
        # Check Cargo target
        if (self.root / "src-tauri" / "target").exists():
            self.info.append("✅ Rust build directory exists")
        else:
            self.warnings.append("⚠️ No Rust build artifacts - first build?")
    
    def check_rust_compilation(self):
        """Try to check Rust compilation"""
        print("\n🔧 Checking Rust compilation readiness...")
        
        try:
            # Check if cargo is available
            result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.info.append(f"✅ Cargo installed: {result.stdout.strip()}")
            else:
                self.issues.append("❌ Cargo not found in PATH")
        except FileNotFoundError:
            self.issues.append("❌ Cargo not installed")
    
    def generate_report(self):
        """Generate diagnostic report"""
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 50)
        
        if self.issues:
            print("\n🚨 CRITICAL ISSUES (must fix):")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS (should check):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ All systems green! Ready to compile.")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        
        if any("icons" in issue.lower() for issue in self.issues):
            print("  1. Run the icon generator:")
            print("     python cyberpunk_icon_generator.py")
        
        if any("node_modules" in issue for issue in self.issues):
            print("  2. Install frontend dependencies:")
            print("     pnpm install")
        
        if any("index.html" in issue for issue in self.issues):
            print("  3. Create missing index.html")
        
        if self.warnings:
            print("  4. Review warnings for incomplete implementations")
        
        # Save detailed report
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("NEXLIFY DIAGNOSTIC REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ISSUES:\n")
            for issue in self.issues:
                f.write(f"  {issue}\n")
            
            f.write("\nWARNINGS:\n")
            for warning in self.warnings:
                f.write(f"  {warning}\n")
            
            f.write("\nINFO:\n")
            for info in self.info:
                f.write(f"  {info}\n")
        
        print(f"\n💾 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    diagnostics = NexlifyDiagnostics()
    diagnostics.run_diagnostics()
