# scripts/test_drawdown_protection.py
"""
Test script for Nexlify Drawdown Protection System
Demonstrates usage and integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.risk.nexlify_drawdown_protection import create_drawdown_protection
from src.risk.risk_manager import create_risk_manager

def simulate_trading_session():
    """Simulate a trading session with drawdown protection"""
    
    print("🌃 NEXLIFY DRAWDOWN PROTECTION TEST")
    print("=" * 50)
    
    # Configuration
    config = {
        'initial_balance': 10000,
        'drawdown_protection': {
            'yellow_threshold': 0.05,    # 5%
            'orange_threshold': 0.10,    # 10%
            'red_threshold': 0.15,       # 15%
            'black_threshold': 0.25,     # 25%
            'daily_loss_limit': 0.05,    # 5%
            'pause_on_drawdown': True,
            'reduce_size_on_drawdown': True,
            'equity_curve_trading': True
        },
        'risk_management': {
            'max_position_size': 0.1,
            'risk_per_trade': 0.01,
            'use_drawdown_protection': True
        }
    }
    
    # Create risk manager with drawdown protection
    risk_manager = create_risk_manager(config)
    
    # Simulate trading
    balance = config['initial_balance']
    balance_history = [balance]
    timestamps = [datetime.now()]
    
    print(f"\nStarting balance: ${balance:,.2f}")
    print("\nSimulating 100 trades...\n")
    
    # Simulate 100 trades
    for i in range(100):
        # Generate random trade result
        # 55% win rate with varying sizes
        is_win = random.random() < 0.55
        
        if is_win:
            # Win: 1-3% gain
            pnl_percent = random.uniform(0.01, 0.03)
        else:
            # Loss: 0.5-2% loss
            pnl_percent = -random.uniform(0.005, 0.02)
            
        # Apply to balance
        pnl = balance * pnl_percent
        balance += pnl
        
        # Update risk manager
        risk_manager.update_account_balance(balance)
        
        # Check if we can trade
        can_trade, reason = risk_manager.drawdown_protection.check_trade_allowed(
            'BTC/USDT', 
            balance * 0.1  # 10% position size
        )
        
        # Get current status
        dd_metrics = risk_manager.drawdown_protection.metrics
        current_level = risk_manager.drawdown_protection.current_level
        
        # Store history
        balance_history.append(balance)
        timestamps.append(datetime.now() + timedelta(minutes=i))
        
        # Print significant events
        if i % 10 == 0 or not can_trade or current_level.value != 'green_zone':
            print(f"Trade {i+1}:")
            print(f"  Balance: ${balance:,.2f}")
            print(f"  Drawdown: {dd_metrics.current_drawdown:.2%}")
            print(f"  Level: {current_level.value}")
            print(f"  Can Trade: {can_trade} ({reason})")
            print(f"  Position Multiplier: {risk_manager.drawdown_protection.position_multiplier:.0%}")
            print()
            
        # Simulate drawdown period
        if i == 50:
            print("📉 SIMULATING DRAWDOWN PERIOD...")
            # Force some losses
            for _ in range(5):
                loss = balance * random.uniform(0.02, 0.04)
                balance -= loss
                risk_manager.update_account_balance(balance)
                balance_history.append(balance)
                timestamps.append(timestamps[-1] + timedelta(minutes=1))
                
    # Final report
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Starting Balance: ${config['initial_balance']:,.2f}")
    print(f"Ending Balance: ${balance:,.2f}")
    print(f"Total Return: {(balance/config['initial_balance']-1)*100:.2f}%")
    
    # Get final metrics
    final_metrics = risk_manager.drawdown_protection.calculate_recovery_metrics()
    print("\nDRAWDOWN METRICS:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")
        
    # Generate report
    print("\n" + "=" * 50)
    print("GENERATING REPORT...")
    report = risk_manager.generate_risk_report()
    
    # Save report
    with open('drawdown_test_report.md', 'w') as f:
        f.write(report)
    print("Report saved to: drawdown_test_report.md")
    
    # Plot results
    plot_results(balance_history, timestamps, config['initial_balance'])
    
    return risk_manager

def plot_results(balance_history, timestamps, initial_balance):
    """Plot balance and drawdown history"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot balance
    ax1.plot(range(len(balance_history)), balance_history, 'cyan', linewidth=2)
    ax1.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Balance ($)', color='cyan')
    ax1.set_title('🌃 NEXLIFY DRAWDOWN PROTECTION TEST', fontsize=16, color='cyan')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and plot drawdown
    peak = initial_balance
    drawdowns = []
    for balance in balance_history:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak
        drawdowns.append(drawdown)
        
    ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color='red', alpha=0.5)
    ax2.plot(range(len(drawdowns)), drawdowns, 'red', linewidth=2)
    
    # Add threshold lines
    ax2.axhline(y=0.05, color='yellow', linestyle='--', alpha=0.7, label='Yellow Alert (5%)')
    ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Orange Alert (10%)')
    ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Red Zone (15%)')
    ax2.axhline(y=0.25, color='purple', linestyle='--', alpha=0.7, label='Flatline (25%)')
    
    ax2.set_ylabel('Drawdown (%)', color='red')
    ax2.set_xlabel('Trade Number')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig('drawdown_test_results.png', dpi=150, bbox_inches='tight')
    print("\nChart saved to: drawdown_test_results.png")
    plt.show()

async def test_async_features():
    """Test async trade checking"""
    print("\n🔄 TESTING ASYNC FEATURES...")
    
    config = {
        'initial_balance': 10000,
        'drawdown_protection': {},
        'risk_management': {}
    }
    
    risk_manager = create_risk_manager(config)
    
    # Test multiple trades simultaneously
    trades = [
        ('BTC/USDT', 'long', 0.1, 45000),
        ('ETH/USDT', 'long', 0.2, 3000),
        ('BNB/USDT', 'short', 0.15, 400)
    ]
    
    print("\nChecking multiple trades simultaneously...")
    tasks = []
    for symbol, side, size, price in trades:
        task = risk_manager.check_trade_allowed(symbol, side, size, price)
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    for (symbol, side, size, price), (allowed, reason, params) in zip(trades, results):
        print(f"\n{symbol} {side} {size} @ ${price}")
        print(f"  Allowed: {allowed}")
        print(f"  Reason: {reason}")
        print(f"  Adjusted Size: {params['size']}")

def main():
    """Main test function"""
    print("\n🚀 Starting Drawdown Protection Test Suite\n")
    
    # Run simulation
    risk_manager = simulate_trading_session()
    
    # Run async tests
    asyncio.run(test_async_features())
    
    print("\n✅ All tests completed!")
    print("\nCheck generated files:")
    print("  - drawdown_test_report.md")
    print("  - drawdown_test_results.png")

if __name__ == "__main__":
    main()
