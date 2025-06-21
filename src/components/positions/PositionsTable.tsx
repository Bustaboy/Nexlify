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
  Trophy,
} from 'lucide-react';
import toast from 'react-hot-toast';
import Decimal from 'decimal.js';

import { useTradingStore, Position } from '@/stores/tradingStore'; // Import Position
import { useMarketStore } from '@/stores/marketStore';

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

export const PositionsTable = ({
  positions: propPositions,
  onClosePosition,
  onModifyPosition,
  compact = false,
  showClosedPositions = false,
}: PositionsTableProps) => {
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [showModifyModal, setShowModifyModal] = useState(false);
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: 'unrealizedPnL',
    direction: 'desc',
  });
  const [hidePnL, setHidePnL] = useState(false);
  const [confirmClose, setConfirmClose] = useState<string | null>(null);

  const {
    positions: storePositions,
    closedPositions,
    closePosition: storeClosePosition,
    modifyPosition: storeModifyPosition,
  } = useTradingStore();
  const { marketData } = useMarketStore();

  const activePositions = propPositions || Object.values(storePositions);
  const displayPositions = showClosedPositions
    ? [...activePositions, ...closedPositions]
    : activePositions;

  const calculateHealthScore = useCallback((position: Position): number => {
    const liqDistance =
      Math.abs(
        position.currentPrice.toNumber() -
          (position.liquidationPrice?.toNumber() || position.currentPrice.toNumber())
      ) / position.currentPrice.toNumber();
    const pnlFactor =
      position.pnlPercentage.toNumber() > 0
        ? 1
        : (100 + position.pnlPercentage.toNumber()) / 100;
    const leverageFactor = 1 - position.leverage / 100;
    const timeFactor = 1; // TODO: Implement time decay

    return (liqDistance * 0.4 + pnlFactor * 0.4 + leverageFactor * 0.2) * 100;
  }, []);

  const sortedPositions = useMemo(() => {
    const sorted = [...displayPositions].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      // Handle Decimal comparisons
      const comparison =
        aValue instanceof Decimal && bValue instanceof Decimal
          ? aValue.cmp(bValue)
          : aValue > bValue
          ? 1
          : -1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
    return sorted;
  }, [displayPositions, sortConfig]);

  const handleClosePosition = async (positionId: string, isEmergency = false) => {
    if (!isEmergency && confirmClose !== positionId) {
      setConfirmClose(positionId);
      setTimeout(() => setConfirmClose(null), 3000);
      return;
    }

    try {
      const position = activePositions.find(p => p.id === positionId);
      if (!position) {
        throw new Error(`Position ${positionId} not found`);
      }

      if (onClosePosition) {
        onClosePosition(positionId);
      } else {
        await storeClosePosition(position.symbol); // Use symbol, not id
      }

      toast.success('Position closed', {
        icon: '💀',
        duration: 5000,
      });
      setConfirmClose(null);
    } catch (error: any) {
      toast.error(`Failed to close: ${error.message}`);
    }
  };

  const handleCloseAll = async () => {
    const confirmMsg = `Close ALL ${activePositions.length} positions?`;
    if (!window.confirm(confirmMsg)) return;

    try {
      const closePromises = activePositions.map(pos => handleClosePosition(pos.id, true));
      await Promise.all(closePromises);
      toast.success(`Closed ${activePositions.length} positions`, {
        icon: '☠️',
        duration: 7000,
      });
    } catch (error) {
      toast.error('Failed to close all positions');
    }
  };

  const getHealthColor = (health: number): string => {
    if (health >= 80) return 'text-green-400';
    if (health >= 60) return 'text-yellow-400';
    if (health >= 40) return 'text-orange-400';
    if (health >= 20) return 'text-red-400';
    return 'text-red-600 animate-pulse';
  };

  const getPnLColor = (pnl: Decimal | number): string => {
    const value = typeof pnl === 'number' ? pnl : pnl.toNumber();
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const formatPositionSize = (position: Position): string => {
    const notional = position.quantity.toNumber() * position.currentPrice.toNumber();
    return `${position.quantity.toFixed(4)} (${position.leverage}x)`;
  };

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

  useEffect(() => {
    const interval = setInterval(() => {
      activePositions.forEach(position => {
        const marketPrice = marketData[position.symbol]?.price;
        if (marketPrice && marketPrice !== position.currentPrice.toNumber()) {
          console.log(`Price update for ${position.symbol}: ${marketPrice}`);
        }
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [activePositions, marketData]);

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
    <div
      className={`bg-gray-900/50 border border-cyan-900/30 rounded-lg ${
        compact ? 'p-3' : 'p-4'
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          OPEN POSITIONS ({activePositions.length})
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setHidePnL(!hidePnL)}
            className="p-1 rounded hover:bg-gray-800 transition-colors"
            title={hidePnL ? 'Show PnL' : 'Hide PnL'}
          >
            {hidePnL ? (
              <EyeOff className="w-4 h-4 text-gray-400" />
            ) : (
              <Eye className="w-4 h-4 text-cyan-400" />
            )}
          </button>
          {activePositions.length > 1 && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleCloseAll}
              className="px-3 py-1 bg-red-900/50 hover:bg-red-800/50 text-red-400 rounded text-xs font-bold flex items-center gap-1 transition-colors"
            >
              <Skull className="w-3 h-3" />
              CLOSE ALL
            </motion.button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Total PnL</p>
          <p
            className={`text-sm font-mono font-bold ${getPnLColor(
              activePositions.reduce((sum, p) => sum.add(p.unrealizedPnL), new Decimal(0))
            )}`}
          >
            {hidePnL
              ? '•••••'
              : `$${activePositions
                  .reduce((sum, p) => sum.add(p.unrealizedPnL), new Decimal(0))
                  .toNumber()
                  .toFixed(2)}`}
          </p>
        </div>
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">At Risk</p>
          <p className="text-sm font-mono text-orange-400">
            ${activePositions
              .reduce((sum, p) => sum.add(p.margin), new Decimal(0))
              .toNumber()
              .toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Avg Leverage</p>
          <p className="text-sm font-mono text-yellow-400">
            {(
              activePositions.reduce((sum, p) => sum + p.leverage, 0) /
              activePositions.length
            ).toFixed(1)}
            x
          </p>
        </div>
        <div className="bg-gray-800/50 rounded p-2">
          <p className="text-xs text-gray-400">Health Score</p>
          <p
            className={`text-sm font-mono font-bold ${getHealthColor(
              activePositions.reduce((sum, p) => sum + calculateHealthScore(p), 0) /
                activePositions.length
            )}`}
          >
            {(
              activePositions.reduce((sum, p) => sum + calculateHealthScore(p), 0) /
              activePositions.length
            ).toFixed(0)}
            %
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-gray-800">
              <th className="text-left py-2 px-2">SYMBOL</th>
              <th className="text-left py-2 px-2">SIDE</th>
              <th
                className="text-right py-2 px-2 cursor-pointer hover:text-cyan-400"
                onClick={() =>
                  setSortConfig({
                    key: 'quantity',
                    direction:
                      sortConfig.key === 'quantity' && sortConfig.direction === 'desc'
                        ? 'asc'
                        : 'desc',
                  })
                }
              >
                SIZE {sortConfig.key === 'quantity' && (sortConfig.direction === 'desc' ? '↓' : '↑')}
              </th>
              <th className="text-right py-2 px-2">ENTRY</th>
              <th className="text-right py-2 px-2">CURRENT</th>
              <th
                className="text-right py-2 px-2 cursor-pointer hover:text-cyan-400"
                onClick={() =>
                  setSortConfig({
                    key: 'unrealizedPnL',
                    direction:
                      sortConfig.key === 'unrealizedPnL' && sortConfig.direction === 'desc'
                        ? 'asc'
                        : 'desc',
                  })
                }
              >
                PNL {sortConfig.key === 'unrealizedPnL' && (sortConfig.direction === 'desc' ? '↓' : '↑')}
              </th>
              <th className="text-right py-2 px-2">HEALTH</th>
              <th className="text-center py-2 px-2">TIME</th>
              <th className="text-right py-2 px-2">ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence mode="popLayout">
              {sortedPositions.map(position => {
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
                    onClick={() => setSelectedPosition(isSelected ? null : position.id)}
                  >
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-2">
                        {position.side === 'long' ? (
                          <TrendingUp className="w-4 h-4 text-green-400" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-400" />
                        )}
                        <span className="font-mono text-sm text-white">{position.symbol}</span>
                      </div>
                    </td>
                    <td className="py-3 px-2">
                      <span
                        className={`
                        text-xs font-semibold px-2 py-1 rounded
                        ${
                          position.side === 'long'
                            ? 'bg-green-900/50 text-green-400'
                            : 'bg-red-900/50 text-red-400'
                        }
                      `}
                      >
                        {position.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-gray-300">
                        {formatPositionSize(position)}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-gray-400">
                        ${position.entryPrice.toNumber().toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-right">
                      <span className="font-mono text-sm text-white">
                        ${position.currentPrice.toNumber().toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-right">
                      {hidePnL ? (
                        <span className="text-gray-500">•••••</span>
                      ) : (
                        <div className="flex flex-col items-end">
                          <span
                            className={`font-mono text-sm font-bold ${getPnLColor(
                              position.unrealizedPnL
                            )}`}
                          >
                            ${position.unrealizedPnL.toNumber().toFixed(2)}
                          </span>
                          <span
                            className={`text-xs ${getPnLColor(position.pnlPercentage)}`}
                          >
                            {position.pnlPercentage.toNumber() > 0 ? '+' : ''}
                            {position.pnlPercentage.toNumber().toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-2 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <span
                          className={`font-mono text-sm font-bold ${getHealthColor(health)}`}
                        >
                          {health.toFixed(0)}%
                        </span>
                        {health < 30 && (
                          <AlertTriangle className="w-4 h-4 text-red-400 animate-pulse" />
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs text-gray-400 flex items-center justify-center gap-1">
                        <Clock className="w-3 h-3" />
                        {getTimeHeld(position.openTime)}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={e => {
                            e.stopPropagation();
                            setSelectedPosition(position.id);
                            setShowModifyModal(true);
                          }}
                          className="p-1 hover:bg-gray-700 rounded transition-colors"
                          title="Modify SL/TP"
                        >
                          <Target className="w-4 h-4 text-gray-400 hover:text-cyan-400" />
                        </button>
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={e => {
                            e.stopPropagation();
                            handleClosePosition(position.id);
                          }}
                          className={`
                            p-1 rounded transition-all
                            ${isConfirming ? 'bg-red-600 animate-pulse' : 'hover:bg-red-900/50'}
                          `}
                          title={isConfirming ? 'Click again to confirm' : 'Close position'}
                        >
                          <X
                            className={`w-4 h-4 ${isConfirming ? 'text-white' : 'text-red-400'}`}
                          />
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
                    <p
                      className={`font-mono text-sm font-bold ${
                        position.side === 'long' ? 'text-red-400' : 'text-green-400'
                      }`}
                    >
                      ${(position.liquidationPrice?.toNumber() || 0).toFixed(2)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {(
                        (Math.abs(
                          position.currentPrice.toNumber() -
                            (position.liquidationPrice?.toNumber() || 0)
                        ) /
                          position.currentPrice.toNumber()) *
                        100
                      ).toFixed(1)}
                      % away
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Stop Loss</p>
                    <p className="font-mono text-sm text-white">
                      {position.stopLoss
                        ? `$${position.stopLoss.toNumber().toFixed(2)}`
                        : 'Not set'}
                    </p>
                    {position.stopLoss && (
                      <p className="text-xs text-red-400 mt-1">
                        -${(
                          (position.entryPrice.toNumber() - position.stopLoss.toNumber()) *
                          position.quantity.toNumber()
                        ).toFixed(2)}
                      </p>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Take Profit</p>
                    <p className="font-mono text-sm text-white">
                      {position.takeProfit
                        ? `$${position.takeProfit.toNumber().toFixed(2)}`
                        : 'Not set'}
                    </p>
                    {position.takeProfit && (
                      <p className="text-xs text-green-400 mt-1">
                        +${(
                          (position.takeProfit.toNumber() - position.entryPrice.toNumber()) *
                          position.quantity.toNumber()
                        ).toFixed(2)}
                      </p>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">Margin Used</p>
                    <p className="font-mono text-sm text-yellow-400">
                      ${position.margin.toNumber().toFixed(2)}
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

      {sortedPositions.some(p => calculateHealthScore(p) < 20) && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 p-3 bg-red-900/30 border border-red-900/50 rounded flex items-center gap-3"
        >
          <AlertTriangle className="w-5 h-5 text-red-400 animate-pulse" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-red-400">LIQUIDATION WARNING</p>
            <p className="text-xs text-red-300">
              One or more positions are at risk. Consider reducing leverage or adding margin.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};
