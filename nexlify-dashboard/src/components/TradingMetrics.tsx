// Location: C:\Nexlify\nexlify-dashboard\src\components\TradingMetrics.tsx
// Status: NEW - Trading performance component
import { useEffect, useState } from 'react';
import { DollarSign, TrendingUp, Activity } from 'lucide-react';

export function TradingMetrics() {
  const [metrics, setMetrics] = useState({
    daily_pnl: 0,
    total_pnl: 0,
    positions_open: 0,
    last_trade: null
  });

  useEffect(() => {
    // Simulate trading metrics for demo
    const interval = setInterval(() => {
      setMetrics({
        daily_pnl: (Math.random() - 0.5) * 10000,
        total_pnl: 150000 + (Math.random() - 0.5) * 50000,
        positions_open: Math.floor(Math.random() * 10),
        last_trade: new Date().toISOString()
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getPnLColor = (value) => {
    return value >= 0 ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <h2 className="text-2xl font-bold text-nexlify-cyan mb-6 flex items-center gap-2">
        <DollarSign className="w-6 h-6" />
        TRADING PERFORMANCE
      </h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-gray-400 text-sm">Daily P&L</span>
          </div>
          <div className={`text-2xl font-bold ${getPnLColor(metrics.daily_pnl)}`}>
            ${metrics.daily_pnl.toLocaleString()}
          </div>
        </div>
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-purple-400" />
            <span className="text-gray-400 text-sm">Total P&L</span>
          </div>
          <div className={`text-2xl font-bold ${getPnLColor(metrics.total_pnl)}`}>
            ${metrics.total_pnl.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}
