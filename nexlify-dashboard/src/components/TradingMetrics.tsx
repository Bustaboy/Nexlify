// Location: C:\Nexlify\nexlify-dashboard\src\components\TradingMetrics.tsx
// Create this new file
import { useEffect, useState } from 'react';
import { DollarSign, TrendingUp, Activity } from 'lucide-react';

interface TradingMetrics {
  daily_pnl: number;
  total_pnl: number;
  positions_open: number;
  last_trade: string | null;
}

export function TradingMetrics() {
  const [metrics] = useState<TradingMetrics>({
    daily_pnl: 12450.50,
    total_pnl: 145200.00,
    positions_open: 3,
    last_trade: new Date().toISOString()
  });

  const getPnLColor = (value: number) => {
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

        <div className="bg-gray-900/50 p-4 rounded col-span-2">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-gray-400 text-sm">Open Positions</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.positions_open}
          </div>
        </div>
      </div>
    </div>
  );
}