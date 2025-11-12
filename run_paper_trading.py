#!/usr/bin/env python3
"""
Nexlify Paper Trading CLI

Command-line interface for running paper trading sessions with RL/ML agents.

Usage:
    # Train a single agent
    python run_paper_trading.py train --agent-type adaptive --episodes 100

    # Evaluate trained models
    python run_paper_trading.py evaluate --models models/model1.pt models/model2.pt

    # Run multi-agent paper trading session
    python run_paper_trading.py multi-agent --config config/paper_trading_config.json --duration 24

    # Quick start with default settings
    python run_paper_trading.py train

Examples:
    # Train adaptive RL agent for 50 episodes
    python run_paper_trading.py train --agent-type adaptive --episodes 50

    # Train ultra-optimized agent with GPU
    python run_paper_trading.py train --agent-type ultra --episodes 100

    # Compare multiple trained models
    python run_paper_trading.py evaluate --models models/*.pt --episodes 20

    # Run 1-hour multi-agent session
    python run_paper_trading.py multi-agent --duration 1
"""

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nexlify.backtesting.nexlify_paper_trading_runner import PaperTradingRunner


def create_default_config():
    """Create default configuration file"""
    config = {
        "paper_trading": {
            "initial_balance": 10000.0,
            "fee_rate": 0.001,
            "slippage": 0.0005,
            "update_interval": 60
        },
        "training": {
            "episodes": 100,
            "max_steps": 1000,
            "save_frequency": 10
        },
        "agents": [
            {
                "agent_id": "adaptive_rl_1",
                "agent_type": "rl_adaptive",
                "name": "Adaptive RL Agent",
                "enabled": True,
                "config": {}
            }
        ],
        "logging": {
            "level": "INFO",
            "file": "paper_trading/logs/session.log"
        }
    }

    config_path = Path("config/paper_trading_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created default config: {config_path}")
    return str(config_path)


def print_banner():
    """Print CLI banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           NEXLIFY PAPER TRADING SYSTEM                        ║
║           ML/RL Agent Training & Evaluation                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main CLI entry point"""
    print_banner()

    parser = argparse.ArgumentParser(
        description='Nexlify Paper Trading CLI - Train and evaluate ML/RL agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --agent-type adaptive --episodes 50
  %(prog)s evaluate --models models/*.pt
  %(prog)s multi-agent --duration 24
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a single RL agent')
    train_parser.add_argument('--agent-type', type=str, default='adaptive',
                             choices=['adaptive', 'ultra'],
                             help='Agent type (adaptive or ultra)')
    train_parser.add_argument('--episodes', type=int, default=100,
                             help='Number of training episodes (default: 100)')
    train_parser.add_argument('--config', type=str,
                             help='Path to configuration file')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--models', nargs='+', required=True,
                            help='Paths to trained model files')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes (default: 10)')
    eval_parser.add_argument('--config', type=str,
                            help='Path to configuration file')

    # Multi-agent command
    multi_parser = subparsers.add_parser('multi-agent',
                                        help='Run multi-agent paper trading session')
    multi_parser.add_argument('--config', type=str,
                             help='Path to configuration file (required for multi-agent)')
    multi_parser.add_argument('--duration', type=float,
                             help='Session duration in hours (omit for indefinite)')

    # Config command
    config_parser = subparsers.add_parser('create-config',
                                         help='Create default configuration file')

    args = parser.parse_args()

    # Handle no command
    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Create config command
    if args.mode == 'create-config':
        create_default_config()
        return

    # Get or create config
    config_path = getattr(args, 'config', None)
    if not config_path:
        print("⚠️  No config specified, using default settings")
        config_path = None
    elif not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("💡 Run 'python run_paper_trading.py create-config' to create one")
        sys.exit(1)

    # Create runner
    print(f"\n🚀 Starting Paper Trading System...")
    print(f"   Mode: {args.mode}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if config_path:
        print(f"   Config: {config_path}")

    runner = PaperTradingRunner(config_path=config_path)

    # Execute based on mode
    try:
        if args.mode == 'train':
            print(f"\n📚 Training {args.agent_type} agent for {args.episodes} episodes...")
            asyncio.run(runner.train_agent(
                agent_type=args.agent_type,
                episodes=args.episodes
            ))

        elif args.mode == 'evaluate':
            print(f"\n📊 Evaluating {len(args.models)} model(s) over {args.episodes} episodes...")
            import numpy as np
            asyncio.run(runner.evaluate_agents(
                model_paths=args.models,
                episodes=args.episodes
            ))

        elif args.mode == 'multi-agent':
            if not config_path:
                print("❌ Error: --config required for multi-agent mode")
                print("💡 Run 'python run_paper_trading.py create-config' first")
                sys.exit(1)

            duration_text = f"{args.duration} hours" if args.duration else "indefinite"
            print(f"\n🎯 Starting multi-agent session (duration: {duration_text})...")
            asyncio.run(runner.run_multi_agent_session(duration_hours=args.duration))

        print("\n✅ Paper trading session completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
