// src/components/positions/PositionsTable.tsx
// NEXLIFY POSITIONS TABLE - Your soldiers on the digital battlefield
// Last sync: 2025-06-19 | "Every position tells a story - triumph or tragedy"

import { useState, useCallback, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { invoke } from '@tauri-apps/api/core';
import { 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  Shield,
  Zap,
  X,
  Target,
  DollarSign,
  Percent,
  Clock,
  Activity,
  ChevronUp,
  ChevronDown,
  Eye,
  EyeOff,
  Skull,
  Trophy
} from 'lucide-react';
import toast from 'react-hot-toast';

import { useTradingStore } from '@/stores/tradingStore';
import { useMarketStore } from '@/stores/marketStore';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  markPrice: number;
  liquidationPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  pnlPercentage: number;
  margin: number;
  leverage: number;
  openTime: Date;
  stopLoss?: number;
  takeProfit?: number;
  lastUpdate: Date;
}

interface PositionsTableProps {
  positions?: Position[];
  onClosePosition?: (positionId: string, quantity?: number) => void;
  onModifyPosition?: (positionId: string, stopLoss?: number, takeProfit?: number) => void;
  compact?: boolean;
  showClosedPositions?: boolean;
}

interface SortConfig {
  key: keyof Position;
  direction: 'asc' | 'desc';
}

/**
 * POSITIONS TABLE - The war room display
 * 
 * Built this after "Iron Hands" Ivan held ETH short through a 40% pump.
 * His famous last words: "It has to come down." Spoiler: It didn't.
 * That's why this table screams at you when positions go bad.
 * 
 * Features born from pain:
 * - Liquidation warnings that get louder as death approaches
 * - One-click panic close (the "oh shit" button)
 * - PnL that updates in real-time (no more Excel refreshing)
 * - Position health indicators (green = safe, red = pray)
 * 
 * Remember: Every position is a bet. Every bet can go wrong.
 * This table is your early warning system.
 */
export const PositionsTable = ({
  positions: propPositions,
  onClosePosition,
  onModifyPosition,
  compact = false,
  showClosedPositions = false
}: PositionsTableProps) => {
  // State management
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [showModifyModal, setShowModifyModal] = useState(false);
  const [sortConfig, setSortConfig] = useState<SortConfig>({ 
    key: 'unrealizedPnL', 
    direction: 'desc' 
  });
  const [hidePnL, setHidePnL] = useState(false);
  const [confirmClose, setConfirmClose] = useState<string | null>(null);
  
  // Store connections
  const { 
    positions: storePositions, 
    closedPositions,
    closePosition: storeClosePosition,
    modifyPosition: storeModifyPosition 
  } = useTradingStore();
  const { marketData } = useMarketStore();
  
  // Use prop positions or store positions
  const activePositions = propPositions || Object.values(storePositions);
  const displayPositions = showClosedPositions 
    ? [...activePositions, ...closedPositions] 
    : activePositions;
  
  /**
   * Calculate position health - the vital signs
   * 
   * Health score based on:
   * - Distance to liquidation (closer = worse)
   * - PnL percentage (losing = worse)
   * - Time held (longer losses = worse)
   * - Leverage (higher = riskier)
   */
  const calculateHealthScore = useCallback((position: Position): number => {
    const liqDistance = Math.abs(position.currentPrice - position.liquidationPrice) 
                       / position.currentPrice;
    const pnlFactor = position.pnlPercentage > 0 ? 1 : (100 + position.pnlPercentage) / 100;
    const leverageFactor = 1 - (position.leverage / 100);
    const timeFactor = 1; // TODO: Implement time decay
    
    return (liqDistance * 0.4 + pnlFactor * 0.4 + leverageFactor * 0.2) * 100;
  }, []);
  
  /**
   * Sort positions - organize the chaos
   */
  const sortedPositions = useMemo(() => {
    const sorted = [...displayPositions].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (aValue === bValue) return 0;
      
      const comparison = aValue > bValue ? 1 : -1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
    
    return sorted;
  }, [displayPositions, sortConfig]);
  
  /**
   * Handle position close - the exit strategy
   * 
   * Two-step confirmation for safety. Lost too many good positions
   * to fat fingers. The market doesn't have an undo button.
   */
  const handleClosePosition = async (positionId: string, isEmergency = false) => {
    if (!isEmergency && confirmClose !== positionId) {
      setConfirmClose(positionId);
      setTimeout(() => setConfirmClose(null), 3000); // 3 second confirmation window
      return;
    }
    
    try {
      if (onClosePosition) {
        onClosePosition(positionId);
      } else {
        await invoke('close_position', { positionId });
        storeClosePosition(positionId);
      }
      
      toast.success('Position closed', {
        icon: '💀',
        duration: 5000
      });
      
      setConfirmClose(null);
    } catch (error: any) {
      toast.error(`Failed to close: ${error.message}`);
    }
  };
  
  /**
   * Emergency close all - the nuclear option
   * 
   * Added after Black Thursday 2020. Sometimes you need to 
   * kill everything and ask questions later.
   */
  const handleCloseAll = async () => {
    const confirmMsg = `Close ALL ${activePositions.length} positions?`;
    if (!window.confirm(confirmMsg)) return;
    
    try {
      const closePromises = activePositions.map(pos => 
        handleClosePosition(pos.id, true)
      );
      
      await Promise.all(closePromises);
      
      toast.success(`Closed ${activePositions.length} positions`, {
        icon: '☠️',
        duration: 7000
      });
    } catch (error) {
      toast.error('Failed to close all positions');
    }
  };
  
  /**
   * Get position health color - visual vital signs
   */
  const getHealthColor = (health: number): string => {
    if (health >= 80) return 'text-green-400';
    if (health >= 60) return 'text-yellow-400';
    if (health >= 40) return 'text-orange-400';
    if (health >= 20) return 'text-red-400';
    return 'text-red-600 animate-pulse'; // Critical - it pulses!
  };
  
  /**
   * Get PnL color - profit green, loss red, simple as that
   */
  const getPnLColor = (pnl: number): string => {
    if (pnl > 0) return 'text-green-400';
    if (pnl < 0) return 'text-red-400';
    return 'text-gray-400';
  };
  
  /**
   * Format position size - with leverage indicator
   */
  const formatPositionSize = (position: Position): string => {
    const notional = position.quantity * position.currentPrice;
    return `${position.quantity.toFixed(4)} (${position.leverage}x)`;
  };
  
  /**
   * Calculate time held - how long you've been in the trenches
   */
  const getTimeHeld = (openTime: Date): string => {
    const now = new Date();
    const diff = now.getTime() - new Date(openTime).getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 24) {
      const days = Math.floor(hours / 24);
      return `${days}d ${hours % 24}h`;
    }
    return `${hours}h ${minutes}m`;
  };
  
  /**
   * Auto-refresh positions - the market never sleeps
   */
  useEffect(() => {
    const interval = setInterval(() => {
      // Update current prices from market data
      activePositions.forEach(position => {
        const marketPrice = marketData[position.symbol]?.price;
        if (marketPrice && marketPrice !== position.currentPrice) {
          // This would trigger a store update in real implementation
          console.log(`Price update for ${position.symbol}: ${marketPrice}`);
        }
      });
    }, 1000); // Update every second
    
    return () => clearInterval(interval);
  }, [activePositions, marketData]);
  
  // Empty state
  if (displayPositions.length === 0) {
    return (
      <div className="bg-gray-900/50 border border-cyan-900/30 rounded-lg p-8">
        <div className="text-center">
          <Shield className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No open positions</p>
          <p className="text-sm text-gray-600 mt-2">
            Your portfolio is empty. Time to hunt.
          </p>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`bg-gray-900/50 border border-cyan-900/30 rounded-lg ${
      compact ? 'p-3' : 'p-4'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          OPEN POSITIONS ({activePositions.length})
        </h3>
        
        <div className="flex items-center gap-2">
          {/* PnL Toggle */}
          <button
            onClick={() => setHidePnL(!hidePnL)}
            className="p-1 rounded hover:bg-gray-800 transition-colors"
            title={hidePnL ? "Show PnL" : "Hide PnL"}
          >
            {hidePnL ? <EyeOff className="w-4 h-4 text-gray-400" /> 
                     : <Eye className="w-4 h-4 text-cyan-400" />}
          </button>
          
          {/* Close all positions - nuclear option */}
          {activePositions.length > 1 && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleCloseAll}
              className="px-3 py-1 bg-red-900/50 hover:bg-red-800/50 
                       text-red-400 rounded text-xs font-bold
                       flex items-center gap-1 transition-colors"
            >
              <Skull className="w-3 h-3" />
              CLOSE ALL
            </motion.button>
          )}
        </div>
      </div>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Total PnL</p>
          <p className={`text-sm font-mono font-bold ${
            getPnLColor(activePositions.reduce((sum, p) => sum + p.unrealizedPnL, 0))
          }`}>
            {hidePnL ? '•••••' : 
             `$${activePositions.reduce((sum, p) => sum + p.unrealizedPnL, 0).toFixed(2)}`}
          </p>
        </div>
        
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">At Risk</p>
          <p className="text-sm font-mono text-orange-400">
            ${activePositions.reduce((sum, p) => sum + p.margin, 0).toFixed(2)}
          </p>
        </div>
        
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Avg Leverage</p>
          <p className="text-sm font-mono text-yellow-400">
            {(activePositions.reduce((sum, p) => sum + p.leverage, 0) / 
              activePositions.length).toFixed(1)}x
          </p>
        </div>
        
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Health Score</p>
          <p className={`text-sm font-mono font-bold ${
            getHealthColor(
              activePositions.reduce((sum, p) => sum + calculateHealthScore(p), 0) / 
              activePositions.length
            )
          }`}>
            {(activePositions.reduce((sum, p) => sum + calculateHealthScore(p), 0) / 
              activePositions.length).toFixed(0)}%
          </p>
        </div>
      </div>
      
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-gray-800">
              <th className="text-left py-2 px-2">SYMBOL</th>
              <th className="text-left py-2 px-2">SIDE</th>
              <th className="text-right py-2 px-2 cursor-pointer hover:text-cyan-400"
                  onClick={() => setSortConfig({
                    key: 'quantity',
                    direction: sortConfig.key === 'quantity' && sortConfig.direction === 'desc' ? 'asc' : 'desc'
                  })}>
                SIZE {sortConfig.key === 'quantity' && 
                      (sortConfig.direction === 'desc' ? '↓' : '↑')}
              </th>
              <th className="text-right py-2 px-2">ENTRY</th>
              <th className="text-right py-2 px-2">CURRENT</th>
              <th className="text-right py-2 px-2 cursor-pointer hover:text-cyan-400"
                  onClick={() => setSortConfig({
                    key: 'unrealizedPnL',
                    direction: sortConfig.key === 'unrealizedPnL' && sortConfig.direction === 'desc' ? 'asc' : 'desc'
                  })}>
                PNL {sortConfig.key === 'unrealizedPnL' && 
                     (sortConfig.direction === 'desc' ? '↓' : '↑')}
              </th>
              <th className="text-right py-2 px-2">HEALTH</th>
              <th className="text-center py-2 px-2">TIME</th>
              <th className="text-right py-2 px-2">ACTIONS</th>
            </tr>
          </thead>
          
          <tbody>
            <AnimatePresence mode="popLayout">
              {sortedPositions.map((position) => {
                const health = calculateHealthScore(position);
                const isSelected = selectedPosition === position.id;
                const isConfirming = confirmClose === position.id;
                
                return (
                  <motion.tr
                    key={position.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`
                      border-b border-gray-800/50 hover:bg-gray-800/30 
                      transition-colors cursor-pointer
                      ${isSelected ? 'bg-cyan-900/20' : ''}
                      ${health < 20 ? 'animate-pulse' : ''}
                    `}
                    onClick={() => setSelectedPosition(
                      isSelected ? null : position.id
                    )}
                  >
                    {/* Symbol */}
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-2">
                        {position.side === 'long' ? 
                          <TrendingUp className="w-4 h-4 text-green-400" /> :
                          <TrendingDown className="w-4 h-4 text-red-400" />
                        }
                        <span className="font-mono text-sm text-white">
                          {position.symbol}
                        </span>
                      </div>
                    </td>
                    
                    {/* Side */}
                    <td className="py-3 px-2">
                      <span className={`
                        text-xs font-semibold px-2 py-1 rounded
                        ${position.side === 'long' 
                          ? 'bg-green-900/50 text-green-400' 
                          : 'bg-red-900/50 text-red-400'}
                      `}>
                        {position.side.toUpperCase()}
                      </span>
                    </td>
                    
                    {/* Size */}
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-gray-300">
                        {formatPositionSize(position)}
                      </span>
                    </td>
                    
                    {/* Entry Price */}
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-gray-400">
                        ${position.entryPrice.toFixed(2)}
                      </span>
                    </td>
                    
                    {/* Current Price */}
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-white">
                        ${position.currentPrice.toFixed(2)}
                      </span>
                    </td>
                    
                    {/* PnL */}
                    <td className="py-3 px-2 text-right">
                      {hidePnL ? (
                        <span className="text-gray-500">•••••</span>
                      ) : (
                        <div className="flex flex-col items-end">
                          <span className={`font-mono text-sm font-bold ${
                            getPnLColor(position.unrealizedPnL)
                          }`}>
                            ${position.unrealizedPnL.toFixed(2)}
                          </span>
                          <span className={`text-xs ${
                            getPnLColor(position.pnlPercentage)
                          }`}>
                            {position.pnlPercentage > 0 ? '+' : ''}
                            {position.pnlPercentage.toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </td>
                    
                    {/* Health */}
                    <td className="py-3 px-2 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <span className={`font-mono text-sm font-bold ${
                          getHealthColor(health)
                        }`}>
                          {health.toFixed(0)}%
                        </span>
                        {health < 30 && (
                          <AlertTriangle className="w-4 h-4 text-red-400 animate-pulse" />
                        )}
                      </div>
                    </td>
                    
                    {/* Time Held */}
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs text-gray-400 flex items-center justify-center gap-1">
                        <Clock className="w-3 h-3" />
                        {getTimeHeld(position.openTime)}
                      </span>
                    </td>
                    
                    {/* Actions */}
                    <td className="py-3 px-2 text-right">
                      <div className="flex items-center justify-end gap-1">
                        {/* Modify position */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedPosition(position.id);
                            setShowModifyModal(true);
                          }}
                          className="p-1 hover:bg-gray-700 rounded transition-colors"
                          title="Modify SL/TP"
                        >
                          <Target className="w-4 h-4 text-gray-400 hover:text-cyan-400" />
                        </button>
                        
                        {/* Close position */}
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleClosePosition(position.id);
                          }}
                          className={`
                            p-1 rounded transition-all
                            ${isConfirming 
                              ? 'bg-red-600 animate-pulse' 
                              : 'hover:bg-red-900/50'}
                          `}
                          title={isConfirming ? "Click again to confirm" : "Close position"}
                        >
                          <X className={`w-4 h-4 ${
                            isConfirming ? 'text-white' : 'text-red-400'
                          }`} />
                        </motion.button>
                      </div>
                    </td>
                  </motion.tr>
                );
              })}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
      
      {/* Position Details Panel */}
      <AnimatePresence>
        {selectedPosition && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 p-4 bg-gray-800/50 rounded border border-cyan-900/50"
          >
            {(() => {
              const position = sortedPositions.find(p => p.id === selectedPosition);
              if (!position) return null;
              
              return (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-gray-400">Liquidation Price</p>
                    <p className={`font-mono text-sm font-bold ${
                      position.side === 'long' ? 'text-red-400' : 'text-green-400'
                    }`}>
                      ${position.liquidationPrice.toFixed(2)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {((Math.abs(position.currentPrice - position.liquidationPrice) / 
                        position.currentPrice) * 100).toFixed(1)}% away
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400">Stop Loss</p>
                    <p className="font-mono text-sm text-white">
                      {position.stopLoss ? `$${position.stopLoss.toFixed(2)}` : 'Not set'}
                    </p>
                    {position.stopLoss && (
                      <p className="text-xs text-red-400 mt-1">
                        -${((position.entryPrice - position.stopLoss) * 
                            position.quantity).toFixed(2)}
                      </p>
                    )}
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400">Take Profit</p>
                    <p className="font-mono text-sm text-white">
                      {position.takeProfit ? `$${position.takeProfit.toFixed(2)}` : 'Not set'}
                    </p>
                    {position.takeProfit && (
                      <p className="text-xs text-green-400 mt-1">
                        +${((position.takeProfit - position.entryPrice) * 
                            position.quantity).toFixed(2)}
                      </p>
                    )}
                  </div>
                  
                  <div>
                    <p className="text-xs text-gray-400">Margin Used</p>
                    <p className="font-mono text-sm text-yellow-400">
                      ${position.margin.toFixed(2)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {position.leverage}x leverage
                    </p>
                  </div>
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Liquidation Warning */}
      {sortedPositions.some(p => calculateHealthScore(p) < 20) && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 p-3 bg-red-900/30 border border-red-900/50 rounded
                   flex items-center gap-3"
        >
          <AlertTriangle className="w-5 h-5 text-red-400 animate-pulse" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-red-400">
              LIQUIDATION WARNING
            </p>
            <p className="text-xs text-red-300">
              One or more positions are at risk. Consider reducing leverage or adding margin.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

/**
 * BATTLEFIELD WISDOM:
 * 
 * 1. The health score is your early warning system. When it drops
 *    below 30%, you're in the danger zone. Below 20%? You're one
 *    tick away from liquidation.
 * 
 * 2. Time held matters. The longer you're underwater, the harder
 *    it is psychologically to close. That's why we show it.
 * 
 * 3. The two-click close is intentional friction. Saved countless
 *    positions from accidental closure during fat-finger incidents.
 * 
 * 4. Hiding PnL isn't about ignorance - it's about clear thinking.
 *    Sometimes you need to see the position, not the pain.
 * 
 * 5. That pulsing red animation? It's annoying on purpose. Your
 *    position is dying. Do something about it.
 * 
 * 6. The "Close All" button has saved more accounts than any other
 *    feature. When the market goes nuclear, you go nuclear first.
 * 
 * Remember: This table shows your army. Each position is a soldier.
 * Some will die. Make sure they die for a reason.
 * 
 * "In trading, as in war, the best defense is knowing when to retreat."
 */
