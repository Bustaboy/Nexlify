// src/components/trading/OrderPanel.tsx
// NEXLIFY NEURAL ORDER MATRIX - Where decisions become destiny
// Last sync: 2025-06-22 | "Every trade is a bet against the future"

import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { invoke } from '@tauri-apps/api/core';
import { 
  Zap, 
  DollarSign,
  Percent,
  Lock,
  Unlock,
  ShieldAlert, 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Shield,
  Target,
  Calculator,
  Brain,
  Gauge,
  Terminal,
  Cpu,
  Activity
} from 'lucide-react';
import toast from 'react-hot-toast';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore, Position } from '@/stores/tradingStore';
import type { OrderType } from '@/stores/tradingStore';
import { useAuthStore } from '@/stores/authStore';

// ═══════════════════════════════════════════════════════════════
// INTERFACE DEFINITIONS - Neural Protocol Types
// ═══════════════════════════════════════════════════════════════

interface OrderPanelProps {
  symbol: string;
  side?: 'buy' | 'sell';
  price?: number;
  onOrderPlaced?: (orderId: string) => void;
  compact?: boolean;
  position?: Position;
}

interface OrderFormData {
  symbol: string;
  side: 'buy' | 'sell';
  orderType: OrderType;
  quantity: string;
  price: string;
  stopPrice: string;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'GTX';
  postOnly: boolean;
  reduceOnly: boolean;
}

interface RiskMetrics {
  positionValue: number;
  accountRisk: number;
  potentialLoss: number;
  potentialProfit: number;
  riskRewardRatio: number;
  marginRequired: number;
  liquidationPrice: number;
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT - The Cyberpunk Trading Terminal
// ═══════════════════════════════════════════════════════════════

export const OrderPanel = ({
  symbol,
  side: initialSide,
  price: initialPrice,
  onOrderPlaced,
  compact = false,
  position
}: OrderPanelProps) => {
  
  // ─────────────────────────────────────────────────────────────
  // NEURAL STATE MANAGEMENT
  // ─────────────────────────────────────────────────────────────
  
  const availableOrderTypes = useMemo(() => {
    if (position) {
      return [
        { value: 'market', label: 'EMERGENCY FLATLINE', icon: Zap },
        { value: 'limit', label: 'PRECISION EXIT', icon: Target },
        { value: 'stop_loss', label: 'DEFENSE PROTOCOL', icon: Shield },
        { value: 'take_profit', label: 'HARVEST SEQUENCE', icon: DollarSign }
      ];
    } else {
      return [
        { value: 'market', label: 'INSTANT EXECUTION', icon: Zap },
        { value: 'limit', label: 'SNIPER MODE', icon: Target },
        { value: 'stop', label: 'TRAP CARD', icon: ShieldAlert },
        { value: 'stop_limit', label: 'ADVANCED TRAP', icon: Brain }
      ];
    }
  }, [position]);

  const [formData, setFormData] = useState<OrderFormData>({
    symbol,
    side: position ? (position.side === 'long' ? 'sell' : 'buy') : (initialSide || 'buy'),
    orderType: position ? 'stop_loss' : 'limit',
    quantity: position ? position.quantity.toString() : '',
    price: initialPrice?.toString() || '',
    stopPrice: '',
    timeInForce: 'GTC',
    postOnly: false,
    reduceOnly: !!position
  });

  const [showRiskCalculator, setShowRiskCalculator] = useState(false);
  const [riskPercentage, setRiskPercentage] = useState('1');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [neuralActivity, setNeuralActivity] = useState(0);

  // ─────────────────────────────────────────────────────────────
  // MARKET DATA INTEGRATION
  // ─────────────────────────────────────────────────────────────
  
  const { getSymbolData } = useMarketStore();
  const { 
    accountBalance, 
    positions, 
    riskLimits,
    dailyPnL,
    placeOrder,
    calculatePositionSize 
  } = useTradingStore();
  
  const symbolData = getSymbolData(symbol);
  const currentPrice = symbolData?.price || 0;
  const currentPosition = positions[symbol];

  // Neural activity simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setNeuralActivity(prev => (prev + 1) % 100);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // ─────────────────────────────────────────────────────────────
  // RISK CALCULATIONS - The Mathematics of Survival
  // ─────────────────────────────────────────────────────────────
  
  const riskMetrics = useMemo((): RiskMetrics => {
    const qty = parseFloat(formData.quantity) || 0;
    const price = formData.orderType === 'market' 
      ? currentPrice 
      : parseFloat(formData.price) || currentPrice;
    
    const positionValue = qty * price;
    const accountRisk = accountBalance.toNumber() > 0 
      ? (positionValue / accountBalance.toNumber()) * 100 
      : 0;
    
    const stopPrice = parseFloat(formData.stopPrice) || 0;
    const potentialLoss = formData.side === 'buy' && stopPrice > 0
      ? qty * (price - stopPrice)
      : formData.side === 'sell' && stopPrice > 0
        ? qty * (stopPrice - price)
        : 0;
    
    const potentialProfit = 0; // Would calculate based on TP
    const riskRewardRatio = potentialLoss > 0 && potentialProfit > 0
      ? potentialProfit / Math.abs(potentialLoss)
      : 0;
    
    const marginRequired = positionValue * 0.1; // 10x leverage
    const liquidationPrice = formData.side === 'buy'
      ? price * 0.9  // 10% down
      : price * 1.1; // 10% up
    
    return {
      positionValue,
      accountRisk,
      potentialLoss,
      potentialProfit,
      riskRewardRatio,
      marginRequired,
      liquidationPrice
    };
  }, [formData, currentPrice, accountBalance]);

  // ─────────────────────────────────────────────────────────────
  // VALIDATION PROTOCOLS
  // ─────────────────────────────────────────────────────────────
  
  const validateOrder = (): { valid: boolean; error?: string } => {
    const qty = parseFloat(formData.quantity);
    const price = parseFloat(formData.price);
    
    if (!qty || qty <= 0) {
      return { valid: false, error: 'INVALID QUANTITY DETECTED' };
    }
    
    if (formData.orderType !== 'market' && formData.orderType !== 'stop' && (!price || price <= 0)) {
      return { valid: false, error: 'INVALID PRICE MATRIX' };
    }
    
    if (riskMetrics.accountRisk > riskLimits.riskLimitPerTrade.toNumber()) {
      return { 
        valid: false, 
        error: `RISK LIMIT BREACH: ${riskLimits.riskLimitPerTrade.toNumber()}% MAX` 
      };
    }
    
    const availableBalance = accountBalance.toNumber() * 0.9;
    if (riskMetrics.marginRequired > availableBalance) {
      return { 
        valid: false, 
        error: 'INSUFFICIENT CREDITS' 
      };
    }
    
    const dailyLoss = dailyPnL.toNumber();
    if (Math.abs(dailyLoss) > riskLimits.maxDailyLoss.toNumber()) {
      return { 
        valid: false, 
        error: 'DAILY LOSS LIMIT ENGAGED' 
      };
    }
    
    if (riskMetrics.accountRisk > 10 && !showConfirmation) {
      setShowConfirmation(true);
      return { 
        valid: false, 
        error: 'LARGE ORDER DETECTED - CONFIRM TO PROCEED' 
      };
    }
    
    return { valid: true };
  };

  const validateStopPrice = useCallback((): { valid: boolean; error?: string } => {
    const stop = parseFloat(formData.stopPrice);
    if (!stop || stop <= 0) {
      return { valid: false, error: 'INVALID STOP PRICE' };
    }
    
    if (formData.orderType === 'stop_loss') {
      if (formData.side === 'buy' && stop >= currentPrice) {
        return { valid: false, error: 'STOP LOSS MUST BE BELOW CURRENT PRICE' };
      }
      if (formData.side === 'sell' && stop <= currentPrice) {
        return { valid: false, error: 'STOP LOSS MUST BE ABOVE CURRENT PRICE' };
      }
    }
    
    return { valid: true };
  }, [formData, currentPrice]);

  // ─────────────────────────────────────────────────────────────
  // ORDER EXECUTION PROTOCOL
  // ─────────────────────────────────────────────────────────────
  
  const handleSubmit = async () => {
    const validation = validateOrder();
    if (!validation.valid) {
      toast.error(validation.error!, {
        style: {
          background: '#1a1a2e',
          color: '#ff0040',
          border: '1px solid #ff0040'
        }
      });
      return;
    }

    if (['stop', 'stop_limit', 'stop_loss'].includes(formData.orderType)) {
      const stopValidation = validateStopPrice();
      if (!stopValidation.valid) {
        toast.error(stopValidation.error!, {
          style: {
            background: '#1a1a2e',
            color: '#ff0040',
            border: '1px solid #ff0040'
          }
        });
        return;
      }
    }

    setIsSubmitting(true);

    try {
      const orderId = await invoke<string>('place_order', {
        order: {
          symbol: formData.symbol,
          side: formData.side,
          type: formData.orderType,
          quantity: parseFloat(formData.quantity),
          price: formData.orderType === 'market' || formData.orderType === 'stop' 
            ? null 
            : parseFloat(formData.price),
          stop_price: ['stop', 'stop_limit', 'stop_loss'].includes(formData.orderType)
            ? parseFloat(formData.stopPrice)
            : null,
          time_in_force: formData.timeInForce,
          post_only: formData.postOnly,
          reduce_only: formData.reduceOnly,
          position_id: position?.id
        }
      });

      toast.success(`ORDER EXECUTED: ${orderId}`, {
        icon: '⚡',
        style: {
          background: '#1a1a2e',
          color: '#00ff88',
          border: '1px solid #00ff88'
        },
        duration: 5000
      });

      // Clear form
      setFormData(prev => ({
        ...prev,
        quantity: '',
        price: '',
        stopPrice: ''
      }));

      onOrderPlaced?.(orderId);
    } catch (error: any) {
      console.error('Order failed:', error);
      toast.error(error.message || 'EXECUTION FAILED', {
        style: {
          background: '#1a1a2e',
          color: '#ff0040',
          border: '1px solid #ff0040'
        }
      });
      
      if (error.message?.includes('insufficient')) {
        setShowRiskCalculator(true);
      }
    } finally {
      setIsSubmitting(false);
      setShowConfirmation(false);
    }
  };

  // ─────────────────────────────────────────────────────────────
  // RENDER PROTOCOL - The Visual Matrix
  // ─────────────────────────────────────────────────────────────
  
  return (
    <div className={`
      relative bg-black/80 border border-cyan-500/30 rounded-lg
      backdrop-blur-md overflow-hidden
      ${compact ? 'p-3' : 'p-4'}
    `}>
      {/* Neural Grid Background */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(cyan 1px, transparent 1px),
            linear-gradient(90deg, cyan 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }} />
      </div>

      {/* Scanning Line Animation */}
      <motion.div
        className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
        animate={{
          top: ['0%', '100%', '0%']
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "linear"
        }}
      />

      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
            <Terminal className="w-5 h-5" />
            {position ? 'POSITION_SHIELD.EXE' : 'ORDER_MATRIX.EXE'}
          </h3>
          
          {/* Neural Activity Indicator */}
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <div className="text-xs font-mono text-cyan-400">
              NEURAL: {neuralActivity}%
            </div>
          </div>
        </div>

        {/* Risk Status Bar */}
        <div className="mb-4 p-2 bg-gray-900/50 rounded border border-gray-800">
          <div className="flex items-center justify-between text-xs">
            <div className={`flex items-center gap-1 ${
              riskMetrics.accountRisk > 5 ? 'text-red-400' :
              riskMetrics.accountRisk > 2 ? 'text-yellow-400' : 'text-green-400'
            }`}>
              <Cpu className="w-3 h-3" />
              RISK: {riskMetrics.accountRisk.toFixed(1)}%
            </div>
            <div className="text-gray-400">
              MARGIN: ${riskMetrics.marginRequired.toFixed(2)}
            </div>
            <div className="text-gray-400">
              LIQ: ${riskMetrics.liquidationPrice.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Buy/Sell Toggle */}
        {!position && (
          <div className="grid grid-cols-2 gap-2 mb-4">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setFormData(prev => ({ ...prev, side: 'buy' }))}
              className={`
                py-3 rounded font-bold transition-all uppercase tracking-wider
                ${formData.side === 'buy'
                  ? 'bg-green-500/20 text-green-400 border border-green-400 shadow-lg shadow-green-500/20'
                  : 'bg-gray-900/50 text-gray-500 border border-gray-700 hover:border-gray-600'}
              `}
            >
              <TrendingUp className="w-4 h-4 inline mr-2" />
              LONG_PROTOCOL
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setFormData(prev => ({ ...prev, side: 'sell' }))}
              className={`
                py-3 rounded font-bold transition-all uppercase tracking-wider
                ${formData.side === 'sell'
                  ? 'bg-red-500/20 text-red-400 border border-red-400 shadow-lg shadow-red-500/20'
                  : 'bg-gray-900/50 text-gray-500 border border-gray-700 hover:border-gray-600'}
              `}
            >
              <TrendingDown className="w-4 h-4 inline mr-2" />
              SHORT_PROTOCOL
            </motion.button>
          </div>
        )}

        {/* Order Type Selection */}
        <div className="mb-4">
          <label className="block text-xs text-cyan-400 mb-1 uppercase tracking-wider font-mono">
            EXECUTION_MODE://
          </label>
          <select
            value={formData.orderType}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              orderType: e.target.value as OrderType
            }))}
            className="w-full bg-gray-900/80 border border-cyan-500/30 rounded px-3 py-2
                     text-cyan-300 font-mono text-sm
                     focus:outline-none focus:border-cyan-400 focus:shadow-lg focus:shadow-cyan-400/20
                     transition-all duration-200"
          >
            {availableOrderTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        {/* Quantity Input */}
        <div className="mb-4">
          <label className="block text-xs text-cyan-400 mb-1 uppercase tracking-wider font-mono">
            QUANTITY_UNITS://
          </label>
          <div className="relative">
            <input
              type="number"
              value={formData.quantity}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                quantity: e.target.value 
              }))}
              placeholder="0.00000000"
              className="w-full bg-gray-900/80 border border-cyan-500/30 rounded px-3 py-2 pr-20
                       text-cyan-300 font-mono placeholder-gray-600
                       focus:outline-none focus:border-cyan-400 focus:shadow-lg focus:shadow-cyan-400/20
                       transition-all duration-200"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1">
              {[25, 50, 75, 100].map(pct => (
                <button
                  key={pct}
                  onClick={() => {
                    const available = position 
                      ? position.quantity.toNumber()
                      : accountBalance.toNumber() * 0.9 / (currentPrice || 1);
                    const qty = available * (pct / 100);
                    setFormData(prev => ({ ...prev, quantity: qty.toFixed(8) }));
                  }}
                  className="px-2 py-1 text-xs bg-cyan-500/20 text-cyan-400 
                           rounded hover:bg-cyan-500/30 transition-colors font-mono"
                >
                  {pct}%
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Price Input (for limit orders) */}
        {formData.orderType !== 'market' && formData.orderType !== 'stop' && (
          <div className="mb-4">
            <label className="block text-xs text-cyan-400 mb-1 uppercase tracking-wider font-mono">
              {formData.orderType === 'take_profit' ? 'TARGET_PRICE://' : 'LIMIT_PRICE://'}
            </label>
            <input
              type="number"
              value={formData.price}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                price: e.target.value 
              }))}
              placeholder={currentPrice.toFixed(2)}
              className="w-full bg-gray-900/80 border border-cyan-500/30 rounded px-3 py-2
                       text-cyan-300 font-mono placeholder-gray-600
                       focus:outline-none focus:border-cyan-400 focus:shadow-lg focus:shadow-cyan-400/20
                       transition-all duration-200"
            />
          </div>
        )}

        {/* Stop Price (for stop orders) */}
        {['stop', 'stop_limit', 'stop_loss'].includes(formData.orderType) && (
          <div className="mb-4">
            <label className="block text-xs text-cyan-400 mb-1 uppercase tracking-wider font-mono">
              {formData.orderType === 'stop_loss' ? 'DEFENSE_TRIGGER://' : 'STOP_TRIGGER://'}
            </label>
            <input
              type="number"
              value={formData.stopPrice}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                stopPrice: e.target.value 
              }))}
              placeholder="0.00"
              className="w-full bg-gray-900/80 border border-cyan-500/30 rounded px-3 py-2
                       text-cyan-300 font-mono placeholder-gray-600
                       focus:outline-none focus:border-cyan-400 focus:shadow-lg focus:shadow-cyan-400/20
                       transition-all duration-200"
            />
          </div>
        )}

        {/* Advanced Options */}
        {!compact && (
          <div className="mb-4 space-y-2">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={formData.postOnly}
                onChange={(e) => setFormData(prev => ({ 
                  ...prev, 
                  postOnly: e.target.checked 
                }))}
                className="w-4 h-4 bg-gray-900 border-gray-600 rounded
                         text-cyan-400 focus:ring-cyan-400 focus:ring-offset-0"
              />
              <span className="font-mono">POST_ONLY_MODE</span>
            </label>
            
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={formData.reduceOnly}
                onChange={(e) => setFormData(prev => ({ 
                  ...prev, 
                  reduceOnly: e.target.checked 
                }))}
                className="w-4 h-4 bg-gray-900 border-gray-600 rounded
                         text-cyan-400 focus:ring-cyan-400 focus:ring-offset-0"
              />
              <span className="font-mono">REDUCE_ONLY</span>
            </label>
          </div>
        )}

        {/* Submit Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleSubmit}
          disabled={isSubmitting || (parseFloat(formData.quantity) || 0) <= 0}
          className={`
            w-full py-3 rounded font-bold transition-all
            flex items-center justify-center gap-2 uppercase tracking-wider
            ${isSubmitting || (parseFloat(formData.quantity) || 0) <= 0
              ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
              : formData.orderType.includes('stop') || formData.orderType === 'take_profit'
                ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-400 shadow-lg shadow-purple-500/20'
                : formData.side === 'buy'
                  ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400 border border-green-400 shadow-lg shadow-green-500/20'
                  : 'bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-400 shadow-lg shadow-red-500/20'
            }
          `}
        >
          {isSubmitting ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Gauge className="w-5 h-5" />
              </motion.div>
              PROCESSING...
            </>
          ) : showConfirmation ? (
            <>
              <ShieldAlert className="w-5 h-5" />
              CONFIRM_LARGE_ORDER
            </>
          ) : (
            <>
              {formData.orderType === 'stop_loss' && <Shield className="w-5 h-5" />}
              {formData.orderType === 'take_profit' && <Target className="w-5 h-5" />}
              {!['stop_loss', 'take_profit'].includes(formData.orderType) && (
                formData.side === 'buy' ? 
                  <TrendingUp className="w-5 h-5" /> : 
                  <TrendingDown className="w-5 h-5" />
              )}
              EXECUTE_{formData.orderType.replace('_', ' ').toUpperCase()}
            </>
          )}
        </motion.button>

        {/* Emergency Close Button */}
        {currentPosition && !compact && (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              setFormData(prev => ({
                ...prev,
                side: currentPosition.side === 'long' ? 'sell' : 'buy',
                orderType: 'market',
                quantity: currentPosition.quantity.abs().toString(),
                reduceOnly: true
              }));
              setTimeout(handleSubmit, 100);
            }}
            className="w-full mt-2 py-2 bg-orange-500/20 hover:bg-orange-500/30 rounded
                     text-sm font-bold text-orange-400 border border-orange-400
                     transition-all hover:shadow-lg hover:shadow-orange-500/20
                     flex items-center justify-center gap-2 uppercase tracking-wider"
          >
            <Lock className="w-4 h-4" />
            PANIC_EJECT.EXE
          </motion.button>
        )}

        {/* Risk Calculator Toggle */}
        {!compact && (
          <button
            onClick={() => setShowRiskCalculator(!showRiskCalculator)}
            className="w-full mt-2 py-1 text-xs text-cyan-400 hover:text-cyan-300
                     transition-colors flex items-center justify-center gap-1 font-mono"
          >
            <Calculator className="w-3 h-3" />
            RISK_CALCULATOR_v2.1
          </button>
        )}
      </div>
    </div>
  );
};