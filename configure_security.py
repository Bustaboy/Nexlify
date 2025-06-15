#!/usr/bin/env python3
"""
Security Configuration Helper for Nexlify v2.0.8
Interactive setup for security features
"""

import json
import getpass
import sys
from pathlib import Path
from typing import Dict, Any
import ipaddress

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Please run the launcher first to create default configuration.")
        sys.exit(1)

def save_config(config_path: Path, config: Dict[str, Any]):
    """Save configuration file"""
    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix('.json.bak')
        config_path.rename(backup_path)
        print(f"📁 Backed up existing config to: {backup_path}")
    
    # Save new config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to: {config_path}")

def configure_master_password(config: Dict[str, Any]) -> bool:
    """Configure master password settings"""
    print("\n🔐 Master Password Configuration")
    print("-" * 40)
    
    current_enabled = config.get('security', {}).get('master_password_enabled', False)
    print(f"Currently {'enabled' if current_enabled else 'disabled'}")
    
    enable = input("\nEnable master password? (y/n) [n]: ").lower() == 'y'
    
    config['security']['master_password_enabled'] = enable
    
    if enable:
        print("\n⚠️  Important: The master password will be set when you first login via the GUI.")
        print("Make sure to remember it as it encrypts your sensitive data!")
        return True
    else:
        config['security']['master_password'] = ""
        print("✅ Master password disabled")
        return False

def configure_2fa(config: Dict[str, Any]) -> bool:
    """Configure 2FA settings"""
    print("\n📱 Two-Factor Authentication Configuration")
    print("-" * 40)
    
    current_enabled = config.get('security', {}).get('2fa_enabled', False)
    print(f"Currently {'enabled' if current_enabled else 'disabled'}")
    
    enable = input("\nEnable 2FA? (y/n) [n]: ").lower() == 'y'
    
    config['security']['2fa_enabled'] = enable
    
    if enable:
        print("\n✅ 2FA enabled")
        print("You'll need to set it up when you first login via the GUI.")
        print("A QR code will be displayed for your authenticator app.")
        return True
    else:
        config['security']['2fa_secret'] = ""
        print("✅ 2FA disabled")
        return False

def configure_ip_whitelist(config: Dict[str, Any]):
    """Configure IP whitelist"""
    print("\n🌐 IP Whitelist Configuration")
    print("-" * 40)
    
    current_enabled = config.get('security', {}).get('ip_whitelist_enabled', False)
    current_ips = config.get('security', {}).get('ip_whitelist', [])
    
    print(f"Currently {'enabled' if current_enabled else 'disabled'}")
    if current_ips:
        print("Current whitelist:")
        for ip in current_ips:
            print(f"  - {ip}")
    
    enable = input("\nEnable IP whitelist? (y/n) [n]: ").lower() == 'y'
    
    config['security']['ip_whitelist_enabled'] = enable
    
    if enable:
        print("\nEnter IP addresses or CIDR ranges (one per line, empty line to finish):")
        print("Examples: 192.168.1.100, 10.0.0.0/24")
        
        ips = []
        while True:
            ip = input("> ").strip()
            if not ip:
                break
                
            # Validate IP/CIDR
            try:
                if '/' in ip:
                    ipaddress.ip_network(ip, strict=False)
                else:
                    ipaddress.ip_address(ip)
                ips.append(ip)
                print(f"✅ Added: {ip}")
            except ValueError:
                print(f"❌ Invalid IP/CIDR: {ip}")
        
        # Always include localhost
        if '127.0.0.1' not in ips:
            ips.append('127.0.0.1')
            print("✅ Added localhost (127.0.0.1) automatically")
            
        config['security']['ip_whitelist'] = ips
        print(f"\n✅ IP whitelist enabled with {len(ips)} entries")
    else:
        print("✅ IP whitelist disabled")

def configure_session_timeout(config: Dict[str, Any]):
    """Configure session timeout"""
    print("\n⏱️  Session Timeout Configuration")
    print("-" * 40)
    
    current = config.get('security', {}).get('session_timeout_minutes', 60)
    print(f"Current timeout: {current} minutes")
    
    while True:
        timeout = input(f"\nEnter session timeout in minutes [{current}]: ").strip()
        if not timeout:
            timeout = current
            break
        
        try:
            timeout = int(timeout)
            if timeout < 5:
                print("❌ Minimum timeout is 5 minutes")
                continue
            if timeout > 1440:
                print("❌ Maximum timeout is 1440 minutes (24 hours)")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    config['security']['session_timeout_minutes'] = timeout
    print(f"✅ Session timeout set to {timeout} minutes")

def configure_api_rotation(config: Dict[str, Any]):
    """Configure API key rotation"""
    print("\n🔄 API Key Rotation Configuration")
    print("-" * 40)
    
    current = config.get('security', {}).get('api_key_rotation_days', 30)
    print(f"Current rotation period: {current} days")
    
    while True:
        days = input(f"\nEnter rotation period in days (0 to disable) [{current}]: ").strip()
        if not days:
            days = current
            break
        
        try:
            days = int(days)
            if days < 0:
                print("❌ Please enter 0 or a positive number")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    config['security']['api_key_rotation_days'] = days
    
    if days == 0:
        print("✅ API key rotation disabled")
    else:
        print(f"✅ API keys will rotate every {days} days")

def configure_security_limits(config: Dict[str, Any]):
    """Configure security limits"""
    print("\n🚨 Security Limits Configuration")
    print("-" * 40)
    
    # Max failed attempts
    current_attempts = config.get('security', {}).get('max_failed_attempts', 5)
    print(f"Current max failed attempts: {current_attempts}")
    
    while True:
        attempts = input(f"Max failed login attempts [{current_attempts}]: ").strip()
        if not attempts:
            attempts = current_attempts
            break
        
        try:
            attempts = int(attempts)
            if attempts < 3:
                print("❌ Minimum is 3 attempts")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    config['security']['max_failed_attempts'] = attempts
    
    # Lockout duration
    current_lockout = config.get('security', {}).get('lockout_duration_minutes', 30)
    print(f"\nCurrent lockout duration: {current_lockout} minutes")
    
    while True:
        lockout = input(f"Lockout duration in minutes [{current_lockout}]: ").strip()
        if not lockout:
            lockout = current_lockout
            break
        
        try:
            lockout = int(lockout)
            if lockout < 5:
                print("❌ Minimum is 5 minutes")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    config['security']['lockout_duration_minutes'] = lockout
    
    print(f"✅ Security limits configured")

def main():
    """Main configuration flow"""
    print("🔐 Nexlify Security Configuration Helper v2.0.8")
    print("=" * 50)
    
    # Load config
    config_path = Path("config/enhanced_config.json")
    config = load_config(config_path)
    
    # Ensure security section exists
    if 'security' not in config:
        config['security'] = {}
    
    # Configure each security feature
    try:
        # Master password
        master_pass_enabled = configure_master_password(config)
        
        # 2FA
        twofa_enabled = configure_2fa(config)
        
        # IP whitelist
        configure_ip_whitelist(config)
        
        # Session timeout
        configure_session_timeout(config)
        
        # API rotation
        configure_api_rotation(config)
        
        # Security limits
        configure_security_limits(config)
        
        # Save configuration
        print("\n" + "=" * 50)
        save = input("\nSave configuration? (y/n) [y]: ").lower()
        
        if save != 'n':
            save_config(config_path, config)
            
            # Print summary
            print("\n✨ Security Configuration Summary:")
            print(f"  - Master Password: {'Enabled' if master_pass_enabled else 'Disabled'}")
            print(f"  - 2FA: {'Enabled' if twofa_enabled else 'Disabled'}")
            print(f"  - IP Whitelist: {'Enabled' if config['security'].get('ip_whitelist_enabled') else 'Disabled'}")
            print(f"  - Session Timeout: {config['security'].get('session_timeout_minutes')} minutes")
            print(f"  - API Key Rotation: {config['security'].get('api_key_rotation_days')} days")
            
            if master_pass_enabled or twofa_enabled:
                print("\n📝 Next Steps:")
                if master_pass_enabled:
                    print("  - Set your master password when first logging into the GUI")
                if twofa_enabled:
                    print("  - Complete 2FA setup when first logging into the GUI")
        else:
            print("\n❌ Configuration not saved")
            
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
