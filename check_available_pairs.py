#!/usr/bin/env python3
"""
Quick script to check which pairs are actually available on each exchange
"""
import ccxt
import sys

def check_pair_availability(symbol, exchanges=['coinbase', 'bitstamp', 'bitfinex', 'kraken']):
    """Check if a symbol is available on various exchanges"""
    print(f"\n{'='*80}")
    print(f"Checking availability of: {symbol}")
    print('='*80)

    available_on = []

    for exchange_name in exchanges:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': 10000})

            # Load markets
            markets = exchange.load_markets()

            if symbol in markets:
                print(f"✅ {exchange_name:12} - Available")
                available_on.append(exchange_name)
            else:
                print(f"❌ {exchange_name:12} - NOT available")

                # Try to find similar symbols
                similar = [s for s in markets.keys() if symbol.split('/')[0] in s]
                if similar:
                    print(f"   💡 Similar pairs: {', '.join(similar[:5])}")

        except Exception as e:
            print(f"⚠️  {exchange_name:12} - Error: {str(e)[:50]}")

    print()
    if available_on:
        print(f"✅ Available on: {', '.join(available_on)}")
    else:
        print(f"❌ Not available on any tested exchange")

    return available_on

if __name__ == "__main__":
    # Test the symbols you want to use
    pairs_to_test = [
        'BTC/USD',
        'ETH/USD',
        'MATIC/USD',
        'MATIC/USDT',
        'SOL/USD',
        'SOL/USDT',
        'ADA/USD',
        'ADA/USDT',
        'XRP/USD',
        'AVAX/USD',
        'AVAX/USDT',
        'DOT/USD',
        'DOT/USDT',
    ]

    if len(sys.argv) > 1:
        # Check specific pairs from command line
        pairs_to_test = sys.argv[1:]

    print("\n" + "="*80)
    print("EXCHANGE PAIR AVAILABILITY CHECKER")
    print("="*80)
    print("Testing pairs across: Coinbase, Bitstamp, Bitfinex, Kraken")

    summary = {}
    for pair in pairs_to_test:
        available = check_pair_availability(pair)
        summary[pair] = len(available)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Sort by availability
    sorted_pairs = sorted(summary.items(), key=lambda x: x[1], reverse=True)

    print("\n✅ Best pairs (available on 3+ exchanges):")
    for pair, count in sorted_pairs:
        if count >= 3:
            print(f"  {pair:15} - {count} exchanges")

    print("\n⚠️  Limited availability (1-2 exchanges):")
    for pair, count in sorted_pairs:
        if 1 <= count < 3:
            print(f"  {pair:15} - {count} exchanges")

    print("\n❌ Not available:")
    for pair, count in sorted_pairs:
        if count == 0:
            print(f"  {pair:15}")

    print("\n" + "="*80)
