#!/usr/bin/env python3
"""
Multi-Dataset Sequential Training Script
Train RL agent on multiple datasets without losing knowledge between datasets

This script trains the agent on multiple cryptocurrencies or time periods,
preserving knowledge from previous training sessions.
"""

import sys
import subprocess
from pathlib import Path
import argparse
import json
from datetime import datetime

def run_training(symbol, data_days, resume_from=None, checkpoint_dir=None, balance=10000, agent_type='adaptive'):
    """
    Run a single training session

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        data_days: Days of historical data
        resume_from: Path to model to resume from
        checkpoint_dir: Directory for checkpoints
        balance: Initial balance
        agent_type: Type of agent

    Returns:
        Path to best model from this training
    """
    cmd = [
        'python', 'scripts/train_ml_rl_1000_rounds.py',
        '--symbol', symbol,
        '--data-days', str(data_days),
        '--balance', str(balance),
        '--agent-type', agent_type,
        '--checkpoint-dir', checkpoint_dir
    ]

    if resume_from:
        cmd.extend(['--resume', resume_from])

    print(f"\n{'='*80}")
    print(f"Training on: {symbol} ({data_days} days)")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"Saving to: {checkpoint_dir}")
    print(f"{'='*80}\n")

    # Run training
    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"\n⚠️  Training failed for {symbol}")
        return None

    # Return path to best model
    best_model_path = Path(checkpoint_dir) / 'best_model.pth'
    if best_model_path.exists():
        return str(best_model_path)
    else:
        # Fallback to final model
        final_model_path = Path(checkpoint_dir) / 'final_model_1000.pth'
        return str(final_model_path) if final_model_path.exists() else None


def main():
    """Main multi-dataset training function"""
    parser = argparse.ArgumentParser(
        description="Train RL agent on multiple datasets sequentially"
    )

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='List of symbols to train on (e.g., BTC/USDT ETH/USDT BNB/USDT)'
    )

    parser.add_argument(
        '--data-days',
        type=int,
        default=180,
        help='Days of historical data for each dataset (default: 180)'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=10000,
        help='Initial balance (default: 10000)'
    )

    parser.add_argument(
        '--agent-type',
        type=str,
        default='adaptive',
        choices=['adaptive', 'ultra', 'basic'],
        help='Agent type (default: adaptive)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='models/multi_dataset',
        help='Base directory for all checkpoints (default: models/multi_dataset)'
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['top5', 'top10', 'major', 'custom'],
        help='Use a preset dataset configuration'
    )

    args = parser.parse_args()

    # Define preset datasets
    presets = {
        'top5': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT'],
        'top10': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
                  'DOGE/USDT', 'SOL/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT'],
        'major': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    }

    # Get datasets to train on
    if args.preset:
        datasets = presets.get(args.preset, [])
        if not datasets:
            print(f"Unknown preset: {args.preset}")
            return 1
    elif args.datasets:
        datasets = args.datasets
    else:
        # Default: major cryptocurrencies
        datasets = presets['major']

    print("\n" + "="*80)
    print("  🎓 MULTI-DATASET SEQUENTIAL TRAINING")
    print("="*80)
    print(f"\nDatasets to train on: {len(datasets)}")
    for i, symbol in enumerate(datasets, 1):
        print(f"  {i}. {symbol}")
    print(f"\nData days per dataset: {args.data_days}")
    print(f"Initial balance: ${args.balance:,.2f}")
    print(f"Agent type: {args.agent_type}")
    print(f"Base directory: {args.base_dir}")
    print(f"\nTotal training episodes: {len(datasets) * 1000}")
    print("="*80 + "\n")

    # Confirm
    response = input("Start multi-dataset training? (Y/N): ")
    if response.upper() != 'Y':
        print("Training cancelled")
        return 0

    # Create base directory
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Training log
    training_log = {
        'start_time': datetime.now().isoformat(),
        'datasets': datasets,
        'data_days': args.data_days,
        'balance': args.balance,
        'agent_type': args.agent_type,
        'training_sessions': []
    }

    # Sequential training
    resume_from = None

    for i, symbol in enumerate(datasets, 1):
        print(f"\n{'#'*80}")
        print(f"# DATASET {i}/{len(datasets)}: {symbol}")
        print(f"{'#'*80}\n")

        # Create checkpoint directory for this dataset
        safe_symbol = symbol.replace('/', '_')
        checkpoint_dir = base_dir / f"step{i:02d}_{safe_symbol}"

        # Run training
        best_model = run_training(
            symbol=symbol,
            data_days=args.data_days,
            resume_from=resume_from,
            checkpoint_dir=str(checkpoint_dir),
            balance=args.balance,
            agent_type=args.agent_type
        )

        if not best_model:
            print(f"\n❌ Failed to complete training on {symbol}")
            print(f"Stopping multi-dataset training")
            break

        # Log this session
        training_log['training_sessions'].append({
            'step': i,
            'symbol': symbol,
            'checkpoint_dir': str(checkpoint_dir),
            'best_model': best_model,
            'resumed_from': resume_from
        })

        # Use this model for next dataset
        resume_from = best_model

        print(f"\n✅ Completed training on {symbol}")
        print(f"   Best model: {best_model}")
        print(f"   This knowledge will be preserved for next dataset")

    # Save training log
    training_log['end_time'] = datetime.now().isoformat()
    training_log['final_model'] = resume_from

    log_path = base_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("  ✅ MULTI-DATASET TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTrained on {len(training_log['training_sessions'])} datasets:")
    for session in training_log['training_sessions']:
        print(f"  ✓ {session['symbol']}")

    print(f"\nFinal model (trained on all datasets):")
    print(f"  {training_log['final_model']}")

    print(f"\nTraining log saved to:")
    print(f"  {log_path}")

    print(f"\nTo use this model in production:")
    print(f"  1. Copy: {training_log['final_model']}")
    print(f"  2. To: models/production/rl_agent.pth")

    print("\n" + "="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
