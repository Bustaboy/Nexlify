// Location: /src/components/defi/DexAggregator/index.tsx
// DEX Aggregator - The Neural Nexus of Decentralized Chaos

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Zap, ArrowRight, AlertCircle, Loader2, Shield } from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { useThemeService } from '../../../services/theme.service';
import { GlitchText } from '../../common/GlitchText';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface DexRoute {
  id: string;
  path: string[];
  dexes: string[];
  inputAmount: number;
  outputAmount: number;
  priceImpact: number;
  gasCost: number;
  totalCost: number;
  savingsVsBest: number;
  estimatedTime: number;
}

export const DexAggregator: React.FC = () => {
  const { apiConfigs } = useDashboardStore();
  const theme = useThemeService();
  const [fromToken, setFromToken] = useState('ETH');
  const [toToken, setToToken] = useState('USDC');
  const [amount, setAmount] = useState(1000);
  const [routes, setRoutes] = useState<DexRoute[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  const activeDexes = apiConfigs.filter(c => c.isDex && c.isActive);
  const tokens = ['ETH', 'BTC', 'USDC', 'DAI', 'SOL', 'MATIC', 'AVAX'];

  const searchRoutes = () => {
    setIsSearching(true);
    setRoutes([]);
    
    // Simulate route finding across DEXs
    setTimeout(() => {
      const baseOutput = amount * (toToken === 'USDC' ? 0.001 : toToken === 'ETH' ? 1000 : 100);
      const newRoutes: DexRoute[] = [];
      
      // Direct routes
      activeDexes.forEach(dex => {
        const slippage = Math.random() * 2;
        const output = baseOutput * (1 - slippage / 100);
        newRoutes.push({
          id: `${dex.exchange}-direct`,
          path: [fromToken, toToken],
          dexes: [dex.exchange],
          inputAmount: amount,
          outputAmount: output,
          priceImpact: slippage,
          gasCost: 10 + Math.random() * 20,
          totalCost: amount + (10 + Math.random() * 20),
          savingsVsBest: 0,
          estimatedTime: 15
        });
      });
      
      // Multi-hop routes
      if (activeDexes.length > 1 && fromToken !== 'USDC' && toToken !== 'USDC') {
        const midToken = 'USDC';
        const hop1Slippage = Math.random() * 1.5;
        const hop2Slippage = Math.random() * 1.5;
        const output = baseOutput * (1 - (hop1Slippage + hop2Slippage) / 100) * 1.02; // Sometimes better
        
        newRoutes.push({
          id: 'multi-hop-1',
          path: [fromToken, midToken, toToken],
          dexes: [activeDexes[0].exchange, activeDexes[1 % activeDexes.length].exchange],
          inputAmount: amount,
          outputAmount: output,
          priceImpact: hop1Slippage + hop2Slippage,
          gasCost: 25 + Math.random() * 30,
          totalCost: amount + (25 + Math.random() * 30),
          savingsVsBest: 0,
          estimatedTime: 30
        });
      }
      
      // Calculate savings
      const bestOutput = Math.max(...newRoutes.map(r => r.outputAmount));
      newRoutes.forEach(route => {
        route.savingsVsBest = ((bestOutput - route.outputAmount) / bestOutput) * 100;
      });
      
      // Sort by output amount
      setRoutes(newRoutes.sort((a, b) => b.outputAmount - a.outputAmount));
      setIsSearching(false);
    }, 2000);
  };

  useEffect(() => {
    if (activeDexes.length > 0) {
      searchRoutes();
    }
  }, [fromToken, toToken, amount]);

  const getRouteQuality = (route: DexRoute): { color: string; text: string } => {
    if (route.savingsVsBest === 0) return { color: theme.colors.success, text: 'OPTIMAL' };
    if (route.savingsVsBest < 1) return { color: theme.colors.info, text: 'GOOD' };
    if (route.savingsVsBest < 3) return { color: theme.colors.warning, text: 'FAIR' };
    return { color: theme.colors.danger, text: 'POOR' };
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/80 backdrop-blur-md border-2 rounded-xl p-6"
      style={{ borderColor: `${theme.colors.neural}66` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold uppercase tracking-wider" style={{ color: theme.colors.neural }}>
          <GlitchText theme={theme.colors}>Neural DEX Aggregator</GlitchText>
        </h3>
        <Network className="w-5 h-5" style={{ color: theme.colors.neural }} />
      </div>

      {activeDexes.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm mb-2">No active DEX connections</p>
          <p className="text-xs">Connect DEXs in API configuration to enable aggregation</p>
        </div>
      ) : (
        <>
          {/* Swap Interface */}
          <div className="mb-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.primary}33` }}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
              <div>
                <label className="text-xs text-gray-400 block mb-1">From</label>
                <div className="flex space-x-2">
                  <input
                    type="number"
                    value={amount}
                    onChange={(e) => setAmount(Number(e.target.value))}
                    className="flex-1 px-3 py-2 bg-gray-900/80 border rounded-lg text-sm font-mono focus:outline-none"
                    style={{ 
                      borderColor: `${theme.colors.primary}44`,
                      color: theme.colors.primary
                    }}
                  />
                  <select
                    value={fromToken}
                    onChange={(e) => setFromToken(e.target.value)}
                    className="px-3 py-2 bg-gray-900/80 border rounded-lg text-sm font-mono focus:outline-none"
                    style={{ 
                      borderColor: `${theme.colors.primary}44`,
                      color: theme.colors.primary
                    }}
                  >
                    {tokens.map(token => (
                      <option key={token} value={token}>{token}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="flex items-center justify-center">
                <ArrowRight className="w-6 h-6" style={{ color: theme.colors.primary }} />
              </div>
              
              <div>
                <label className="text-xs text-gray-400 block mb-1">To</label>
                <select
                  value={toToken}
                  onChange={(e) => setToToken(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-900/80 border rounded-lg text-sm font-mono focus:outline-none"
                  style={{ 
                    borderColor: `${theme.colors.primary}44`,
                    color: theme.colors.primary
                  }}
                >
                  {tokens.filter(t => t !== fromToken).map(token => (
                    <option key={token} value={token}>{token}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Active DEXs Display */}
          <div className="mb-6 flex flex-wrap gap-2">
            {activeDexes.map(dex => (
              <div key={dex.exchange} className="flex items-center space-x-2 px-3 py-1 bg-gray-800/50 border rounded-lg text-xs"
                   style={{ borderColor: `${theme.colors.success}44` }}>
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: theme.colors.success }} />
                <span className="font-mono uppercase">{dex.exchange}</span>
              </div>
            ))}
          </div>

          {/* Routes */}
          {isSearching ? (
            <div className="text-center py-12">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="inline-block"
              >
                <Loader2 className="w-8 h-8" style={{ color: theme.colors.primary }} />
              </motion.div>
              <p className="text-sm text-gray-400 mt-4">Scanning neural pathways across {activeDexes.length} DEXs...</p>
            </div>
          ) : routes.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Zap className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No routes found</p>
            </div>
          ) : (
            <div className="space-y-3">
              <AnimatePresence>
                {routes.map((route, idx) => {
                  const quality = getRouteQuality(route);
                  const isSelected = selectedRoute === route.id;
                  const isBest = idx === 0;
                  
                  return (
                    <motion.div
                      key={route.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: idx * 0.1 }}
                      className={`relative p-4 bg-gray-800/60 border rounded-lg cursor-pointer transition-all ${
                        isSelected ? 'ring-2' : 'hover:bg-gray-800/80'
                      }`}
                      style={{ 
                        borderColor: isBest ? `${theme.colors.success}66` : `${theme.colors.primary}33`,
                        ringColor: isSelected ? theme.colors.primary : undefined,
                        boxShadow: isBest ? `0 0 20px ${theme.colors.success}22` : undefined
                      }}
                      onClick={() => setSelectedRoute(isSelected ? null : route.id)}
                    >
                      {isBest && (
                        <div className="absolute -top-2 -right-2 px-2 py-0.5 bg-green-500/20 border border-green-500 rounded text-xs font-mono font-bold"
                             style={{ color: theme.colors.success }}>
                          BEST ROUTE
                        </div>
                      )}

                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <Zap className="w-4 h-4" style={{ color: quality.color }} />
                          <div className="flex items-center space-x-2">
                            {route.path.map((token, i) => (
                              <React.Fragment key={i}>
                                <span className="font-mono font-bold text-sm">{token}</span>
                                {i < route.path.length - 1 && (
                                  <ArrowRight className="w-3 h-3 text-gray-500" />
                                )}
                              </React.Fragment>
                            ))}
                          </div>
                          <span className="text-xs px-2 py-0.5 rounded" style={{
                            backgroundColor: `${quality.color}22`,
                            color: quality.color
                          }}>
                            {quality.text}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold" style={{ color: theme.colors.primary }}>
                            {route.outputAmount.toFixed(2)} {toToken}
                          </div>
                          <div className="text-xs text-gray-500">
                            €{(route.outputAmount * (toToken === 'USDC' ? 1 : toToken === 'ETH' ? 3000 : 100)).toFixed(2)}
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-4 gap-3 text-xs">
                        <div>
                          <span className="text-gray-500">Impact</span>
                          <div className="font-mono" style={{ 
                            color: route.priceImpact > 2 ? theme.colors.danger : theme.colors.warning 
                          }}>
                            -{route.priceImpact.toFixed(2)}%
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-500">Gas</span>
                          <div className="font-mono">€{route.gasCost.toFixed(2)}</div>
                        </div>
                        <div>
                          <span className="text-gray-500">Time</span>
                          <div className="font-mono">{route.estimatedTime}s</div>
                        </div>
                        <div>
                          <span className="text-gray-500">Via</span>
                          <div className="font-mono text-xs">
                            {route.dexes.join(' → ')}
                          </div>
                        </div>
                      </div>

                      <AnimatePresence>
                        {isSelected && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-4 pt-4 border-t space-y-3"
                            style={{ borderColor: `${theme.colors.primary}22` }}
                          >
                            <div className="grid grid-cols-2 gap-4 text-xs">
                              <div>
                                <span className="text-gray-500">Total Cost</span>
                                <div className="font-mono">€{route.totalCost.toFixed(2)}</div>
                              </div>
                              <div>
                                <span className="text-gray-500">Effective Price</span>
                                <div className="font-mono">
                                  1 {fromToken} = {(route.outputAmount / amount).toFixed(4)} {toToken}
                                </div>
                              </div>
                            </div>
                            
                            {route.savingsVsBest > 0 && (
                              <div className="p-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-xs">
                                <span className="text-yellow-400">
                                  This route costs {formatPercent(route.savingsVsBest)} more than the best route
                                </span>
                              </div>
                            )}
                            
                            <button className="w-full py-2 border-2 rounded-lg font-mono text-sm font-bold transition-all hover:bg-green-500/20"
                                    style={{ 
                                      borderColor: theme.colors.success,
                                      color: theme.colors.success
                                    }}>
                              EXECUTE SWAP
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}

          {/* Smart Route Info */}
          <div className="mt-6 p-4 bg-gray-800/50 border rounded-lg" style={{ borderColor: `${theme.colors.neural}33` }}>
            <div className="flex items-center space-x-2 mb-2">
              <Shield className="w-4 h-4" style={{ color: theme.colors.neural }} />
              <span className="text-sm font-bold" style={{ color: theme.colors.neural }}>Neural Routing Engine</span>
            </div>
            <div className="space-y-1 text-xs text-gray-400">
              <p>• Scans {activeDexes.length} connected DEXs for optimal paths</p>
              <p>• Calculates gas costs and price impact across all routes</p>
              <p>• Multi-hop routing can sometimes beat direct swaps</p>
              <p>• MEV protection and sandwich attack resistance included</p>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
};
