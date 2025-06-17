// Location: /src/components/trading/ArbitrageMatrix/index.tsx
// Arbitrage Matrix - Cross-Exchange Neural Profit Scanner

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { GitMerge, Zap, TrendingUp, AlertCircle } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface ArbitrageOpportunity {
  id: string;
  pair: string;
  buyExchange: string;
  sellExchange: string;
  buyPrice: number;
  sellPrice: number;
  spread: number;
  spreadPercent: number;
  volume: number;
  profitEstimate: number;
  confidence: number;
  timeWindow: number;
  status: 'active' | 'executing' | 'expired';
}

export const ArbitrageMatrix: React.FC = () => {
  const { apiConfigs, tradingActive, metrics } = useDashboardStore();
  const theme = useThemeService();
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [scanning, setScanning] = useState(false);

  const activeExchanges = apiConfigs.filter(c => c.isActive && !c.isDex);
  const activeDexes = apiConfigs.filter(c => c.isActive && c.isDex);

  useEffect(() => {
    if (!tradingActive || activeExchanges.length < 2) return;

    const scanArbitrage = () => {
      setScanning(true);
      
      // Simulate arbitrage scanning
      setTimeout(() => {
        const pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'];
        const newOpps: ArbitrageOpportunity[] = [];
        
        // Generate CEX arbitrage
        activeExchanges.forEach((exchange1, i) => {
          activeExchanges.slice(i + 1).forEach(exchange2 => {
            pairs.forEach(pair => {
              if (Math.random() > 0.7) {
                const basePrice = pair === 'BTC/USDT' ? 45000 : pair === 'ETH/USDT' ? 3000 : 50;
                const spread = Math.random() * 100;
                const buyPrice = basePrice - spread / 2;
                const sellPrice = basePrice + spread / 2;
                
                newOpps.push({
                  id: `${Date.now()}-${Math.random()}`,
                  pair,
                  buyExchange: exchange1.exchange,
                  sellExchange: exchange2.exchange,
                  buyPrice,
                  sellPrice,
                  spread,
                  spreadPercent: (spread / basePrice) * 100,
                  volume: Math.random() * 10000,
                  profitEstimate: spread * Math.random() * 10,
                  confidence: 60 + Math.random() * 40,
                  timeWindow: Math.floor(Math.random() * 60) + 10,
                  status: 'active'
                });
              }
            });
          });
        });

        // Generate DEX-CEX arbitrage
        if (activeDexes.length > 0) {
          activeDexes.forEach(dex => {
            activeExchanges.forEach(cex => {
              if (Math.random() > 0.8) {
                const pair = pairs[Math.floor(Math.random() * pairs.length)];
                const basePrice = pair === 'BTC/USDT' ? 45000 : pair === 'ETH/USDT' ? 3000 : 50;
                const spread = Math.random() * 200;
                
                newOpps.push({
                  id: `${Date.now()}-${Math.random()}`,
                  pair,
                  buyExchange: Math.random() > 0.5 ? dex.exchange : cex.exchange,
                  sellExchange: Math.random() > 0.5 ? cex.exchange : dex.exchange,
                  buyPrice: basePrice - spread / 2,
                  sellPrice: basePrice + spread / 2,
                  spread,
                  spreadPercent: (spread / basePrice) * 100,
                  volume: Math.random() * 5000,
                  profitEstimate: spread * Math.random() * 5 - 50, // Account for gas
                  confidence: 50 + Math.random() * 40,
                  timeWindow: Math.floor(Math.random() * 30) + 5,
                  status: 'active'
                });
              }
            });
          });
        }

        setOpportunities(newOpps.sort((a, b) => b.profitEstimate - a.profitEstimate));
        setScanning(false);
      }, 2000);
    };

    scanArbitrage();
    const interval = setInterval(scanArbitrage, 15000);

    return () => clearInterval(interval);
  }, [tradingActive, activeExchanges.length, activeDexes.length]);

  const executeArbitrage = (oppId: string) => {
    setOpportunities(prev => prev.map(opp => 
      opp.id === oppId ? { ...opp, status: 'executing' } : opp
    ));

    setTimeout(() => {
      setOpportunities(prev => prev.filter(opp => opp.id !== oppId));
    }, 3000);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.info}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider flex items-center space-x-2" 
            style={{ color: theme.colors.info }}>
          <GitMerge className="w-5 h-5" />
          <GlitchText theme={theme.colors}>Cross-Exchange Arbitrage Scanner</GlitchText>
        </h3>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-400">
            Monitoring: <span className="font-bold" style={{ color: theme.colors.success }}>
              {activeExchanges.length} CEX + {activeDexes.length} DEX
            </span>
          </span>
          {scanning && (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Zap className="w-5 h-5" style={{ color: theme.colors.warning }} />
            </motion.div>
          )}
        </div>
      </div>

      {activeExchanges.length < 2 ? (
        <div className="text-center py-12 text-gray-500">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm mb-2">Minimum 2 active exchanges required for arbitrage</p>
          <p className="text-xs">Connect more exchanges in API configuration</p>
        </div>
      ) : opportunities.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <GitMerge className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm">Scanning for arbitrage opportunities...</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          <style jsx>{`
            .scrollbar-arb::-webkit-scrollbar {
              width: 6px;
            }
            .scrollbar-arb::-webkit-scrollbar-track {
              background: rgba(31, 41, 55, 0.5);
              border-radius: 3px;
            }
            .scrollbar-arb::-webkit-scrollbar-thumb {
              background: ${theme.colors.info}66;
              border-radius: 3px;
            }
            .scrollbar-arb::-webkit-scrollbar-thumb:hover {
              background: ${theme.colors.info}88;
            }
          `}</style>

          <div className="scrollbar-arb space-y-3">
            {opportunities.map((opp, idx) => (
              <motion.div
                key={opp.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className={`p-4 bg-gray-800/60 border rounded-lg transition-all ${
                  opp.status === 'executing' ? 'opacity-50' : 'hover:bg-gray-800/80'
                }`}
                style={{ 
                  borderColor: opp.profitEstimate > 100 ? `${theme.colors.success}66` : 
                              opp.profitEstimate > 0 ? `${theme.colors.warning}66` : 
                              `${theme.colors.danger}66`
                }}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="w-4 h-4" style={{ 
                      color: opp.profitEstimate > 0 ? theme.colors.success : theme.colors.danger 
                    }} />
                    <span className="font-mono font-bold">{opp.pair}</span>
                    <div className="flex items-center space-x-2 text-xs">
                      <span className="px-2 py-1 bg-gray-900/50 rounded" style={{ color: theme.colors.primary }}>
                        {opp.buyExchange}
                      </span>
                      <span className="text-gray-500">→</span>
                      <span className="px-2 py-1 bg-gray-900/50 rounded" style={{ color: theme.colors.primary }}>
                        {opp.sellExchange}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold" style={{ 
                      color: opp.profitEstimate > 0 ? theme.colors.success : theme.colors.danger 
                    }}>
                      {formatCredits(opp.profitEstimate)}
                    </div>
                    <div className="text-xs text-gray-500">Est. Profit</div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-3 text-xs mb-3">
                  <div>
                    <span className="text-gray-500">Buy</span>
                    <div className="font-mono">{opp.buyPrice.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Sell</span>
                    <div className="font-mono">{opp.sellPrice.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Spread</span>
                    <div className="font-bold" style={{ color: theme.colors.warning }}>
                      {formatPercent(opp.spreadPercent)}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">Confidence</span>
                    <div className="font-bold" style={{ 
                      color: opp.confidence > 80 ? theme.colors.success : theme.colors.warning 
                    }}>
                      {opp.confidence.toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 text-xs text-gray-400">
                    <span>Max volume: {formatCredits(opp.volume)}</span>
                    <span>•</span>
                    <span>Window: {opp.timeWindow}s</span>
                  </div>
                  <button
                    onClick={() => executeArbitrage(opp.id)}
                    disabled={opp.status === 'executing' || opp.profitEstimate < 0}
                    className="px-4 py-1.5 rounded text-xs font-mono font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      backgroundColor: opp.status === 'executing' ? `${theme.colors.warning}33` : `${theme.colors.success}33`,
                      color: opp.status === 'executing' ? theme.colors.warning : theme.colors.success,
                      border: `1px solid ${opp.status === 'executing' ? theme.colors.warning : theme.colors.success}`
                    }}
                  >
                    {opp.status === 'executing' ? 'EXECUTING...' : 'EXECUTE'}
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
};
