#!/usr/bin/env python3
"""
🌃 NEXLIFY NEURAL DIAGNOSTIC v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run this first to scan your entire Nexlify setup!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess
import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime

def run_check(cmd, name):
    """Run diagnostic command and report"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        version = result.stdout.strip() or "NOT INSTALLED"
        status = "✅" if result.returncode == 0 else "❌"
        print(f"{status} {name}: {version}")
        return result.returncode == 0, version
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False, f"ERROR - {str(e)}"

def scan_for_sins(root_path="."):
    """Hunt for hardcoded values and corpo placeholder sins"""
    print("\n🔍 SCANNING FOR HARDCODED CHROME & PLACEHOLDERS...")
    
    sins_found = []
    patterns = {
        r'(localhost|127\.0\.0\.1):[0-9]+': "Hardcoded local endpoints",
        r'(TODO|FIXME|XXX|HACK)': "Dev comments that need attention",
        r'(test|dummy|fake|mock|placeholder)': "Test/placeholder code",
        r'["\']api[_-]?key["\']\s*[:=]\s*["\'][^"\']+["\']': "Possible hardcoded API keys",
        r'["\']secret["\']\s*[:=]\s*["\'][^"\']+["\']': "Hardcoded secrets",
        r'console\.(log|error|debug)': "Console statements (remove for prod)",
        r'[Cc]:\\\\|\/home\/|\/Users\/': "Hardcoded file paths",
    }
    
    extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml']
    exclude_dirs = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.next', 'archive'}
    
    file_count = 0
    for ext in extensions:
        for file_path in Path(root_path).rglob(f'*{ext}'):
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
                
            file_count += 1
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                for pattern, desc in patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        sins_found.append(f"{file_path}: {desc} ({len(matches)} found)")
            except Exception:
                pass
    
    print(f"  Scanned {file_count} files")
    
    if sins_found:
        print("\n⚠️  HARDCODED SINS DETECTED:")
        for sin in sins_found[:20]:  # Show first 20
            print(f"  - {sin}")
        if len(sins_found) > 20:
            print(f"  ... and {len(sins_found) - 20} more")
    else:
        print("✅ No obvious hardcoded values found (but check manually!)")
    
    return sins_found

print("🌃 NEXLIFY NEURAL DIAGNOSTIC v3.0")
print("=" * 70)
print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current Directory: {os.getcwd()}")
print("=" * 70)

# Store results for summary
results = {}

print("\n🔧 RUNTIME ENVIRONMENT CHECK:")
print("-" * 40)
checks = {
    "Python": "python --version",
    "Node.js": "node --version", 
    "pnpm": "pnpm --version",
    "npm": "npm --version",
    "Rust": "rustc --version",
    "Cargo": "cargo --version",
}

for name, cmd in checks.items():
    success, version = run_check(cmd, name)
    results[name] = {"success": success, "version": version}

# Check for Tauri CLI specifically
print("\n🎯 TAURI CHECKS:")
print("-" * 40)
tauri_checks = {
    "Tauri CLI (npm)": "pnpm tauri --version",
    "Tauri Info": "pnpm tauri info",
}

for name, cmd in tauri_checks.items():
    success, output = run_check(cmd, name)
    results[name] = {"success": success, "version": output}

print("\n📁 PROJECT STRUCTURE CHECK:")
print("-" * 40)
critical_paths = [
    ("src-tauri/Cargo.toml", "Tauri Rust config"),
    ("package.json", "Node.js config"),
    ("src-backend/core/arasaka_neural_net.py", "Main backend entry"),
    ("requirements-full.txt", "Python dependencies"),
    ("requirements.txt", "Base Python deps"),
    ("pnpm-lock.yaml", "pnpm lockfile"),
    (".env", "Environment config"),
    ("src-tauri/tauri.conf.json", "Tauri configuration"),
]

missing_files = []
for path, desc in critical_paths:
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {path} - {desc}")
    if not exists:
        missing_files.append(path)

# Check for multiple source directories
print("\n🗂️ SOURCE DIRECTORY ANALYSIS:")
print("-" * 40)
src_dirs = []
for pattern in ["src*", "source*"]:
    src_dirs.extend([str(p) for p in Path(".").glob(pattern) if p.is_dir()])

if src_dirs:
    print(f"Found {len(src_dirs)} source directories:")
    for src_dir in src_dirs:
        file_count = len(list(Path(src_dir).rglob("*.*")))
        print(f"  - {src_dir} ({file_count} files)")
else:
    print("⚠️ No source directories found!")

# Run the sin scanner
sins = scan_for_sins()

# Summary and recommendations
print("\n" + "=" * 70)
print("📊 DIAGNOSTIC SUMMARY:")
print("=" * 70)

critical_issues = []

# Check Python version
if results.get("Python", {}).get("success"):
    python_version = results["Python"]["version"]
    if "3.13" in python_version:
        critical_issues.append("Python 3.13 detected - DOWNGRADE TO 3.12.7 RECOMMENDED")
    elif "3.12" not in python_version:
        critical_issues.append(f"Python {python_version} - Upgrade to 3.12.7 recommended")

# Check for missing critical components
if not results.get("Rust", {}).get("success"):
    critical_issues.append("Rust not installed - CRITICAL for Tauri")
    
if not results.get("pnpm", {}).get("success"):
    critical_issues.append("pnpm not installed - Required for dependency management")

if missing_files:
    critical_issues.append(f"{len(missing_files)} critical files missing")

if len(src_dirs) > 3:
    critical_issues.append("Source code fragmentation detected - consolidation needed")

if sins:
    critical_issues.append(f"{len(sins)} hardcoded values/placeholders found")

if critical_issues:
    print("\n🚨 CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(critical_issues, 1):
        print(f"{i}. {issue}")
else:
    print("\n✅ No critical issues detected!")

print("\n🔧 RECOMMENDATIONS:")
if not results.get("Rust", {}).get("success"):
    print("1. Install Rust: https://rustup.rs/")
    print("   Then run: rustup default stable-x86_64-pc-windows-msvc")
    
if not results.get("pnpm", {}).get("success"):
    print("2. Install pnpm: npm install -g pnpm")
    
if "3.13" in results.get("Python", {}).get("version", ""):
    print("3. Downgrade Python: Install 3.12.7 from python.org")
    
if sins:
    print("4. Run security audit to clean hardcoded values")
    
print("\n💀 If all else fails: Run 'python recovery_protocol.py'")
print("\n🎮 Next step: Save diagnostic output and share with your netrunner!")
print("=" * 70)

# Save diagnostic report
report_path = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
print(f"\n💾 Saving full diagnostic report to: {report_path}")

with open(report_path, 'w') as f:
    f.write("NEXLIFY DIAGNOSTIC REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Directory: {os.getcwd()}\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("ENVIRONMENT:\n")
    for name, data in results.items():
        f.write(f"  {name}: {data['version']}\n")
    
    f.write("\nMISSING FILES:\n")
    for file in missing_files:
        f.write(f"  - {file}\n")
    
    if sins:
        f.write("\nHARDCODED VALUES FOUND:\n")
        for sin in sins[:50]:  # First 50
            f.write(f"  - {sin}\n")

print("\n🌃 Diagnostic complete. Welcome to Night City, choom.")
