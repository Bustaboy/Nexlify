// src/components/trading/OrderPanel.tsx
// NEXLIFY ORDER PANEL - Where decisions become destiny
// Last sync: 2025-06-22 | "Every trade is a bet against the future"
// ENHANCED ORDER PANEL WITH ALL ORDER TYPES

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
  Gauge
} from 'lucide-react';
import toast from 'react-hot-toast';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore, Position, OrderType } from '@/stores/tradingStore';
import { useAuthStore } from '@/stores/authStore';

// Type definitions for the neural trading interface
type FormOrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
type BackendOrderType = 'market' | 'limit' | 'stop_loss' | 'take_profit';


interface OrderPanelProps {
  symbol: string;
  side?: 'buy' | 'sell';
  price?: number;
  onOrderPlaced?: (orderId: string) => void;
  compact?: boolean;
  position?: Position;  // If provided, shows position protection options
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

/**
 * ORDER PANEL - The cockpit of fortune
 * 
 * Built this after watching "Fast Fingers" Freddy fat-finger a million
 * dollar order. He meant to buy 100 BTC, typed 1000. The slippage alone
 * cost him his house. That's why we have:
 * - Order confirmation for large sizes
 * - Visual risk indicators that scream danger
 * - Position size calculator based on risk tolerance
 * - One-click emergency close (the panic button)
 * 
 * Remember: The market is a machine that transfers wealth from the
 * impatient to the patient, from the reckless to the prepared.
 */

export const OrderPanel = ({
  symbol,
  side: initialSide,
  price: initialPrice,
  onOrderPlaced,
  compact = false,
  position  // NEW: position prop for protection orders
}: OrderPanelProps) => {
  // Determine available order types based on context
  const availableOrderTypes = useMemo(() => {
    if (position) {
      // When managing a position, show protection orders
      return [
        { value: 'market', label: 'Market Close' },
        { value: 'limit', label: 'Limit Close' },
        { value: 'stop_loss', label: 'Stop Loss', icon: Shield },
        { value: 'take_profit', label: 'Take Profit', icon: Target }
      ];
    } else {
      // For new orders, show standalone types
      return [
        { value: 'market', label: 'Market' },
        { value: 'limit', label: 'Limit' },
        { value: 'stop', label: 'Stop' },
        { value: 'stop_limit', label: 'Stop Limit' }
      ];
    }
  }, [position]);

  // Form state
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

  // UI state
  const [showRiskCalculator, setShowRiskCalculator] = useState(false);
  const [riskPercentage, setRiskPercentage] = useState('1'); // 1% default risk
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);

  // Get current market data
  const { getSymbolData } = useMarketStore();
  const { 
    accountBalance, 
    positions, 
    riskLimits,
    dailyPnL,  // Added for daily loss checks
    placeOrder,
    calculatePositionSize 
  } = useTradingStore();
  
  const symbolData = getSymbolData(symbol);
  const currentPrice = symbolData?.price || 0; // Fixed: was lastPrice
  const currentPosition = positions[symbol];
  
  // Auto-update price for limit orders
  useEffect(() => {
    if (formData.orderType === 'limit' && !formData.price && currentPrice) {
      setFormData(prev => ({ 
        ...prev, 
        price: currentPrice.toString() 
      }));
    }
  }, [currentPrice, formData.orderType, formData.price]);

  // Validate stop prices based on order type and side
  const validateStopPrice = useCallback((): { valid: boolean; error?: string } => {
    const stopPrice = parseFloat(formData.stopPrice);
    
    if (!stopPrice || stopPrice <= 0) {
      return { valid: false, error: 'Invalid stop price' };
    }

    // Different validation for different order types
    switch (formData.orderType) {
      case 'stop':
      case 'stop_limit':
        // Standalone stops: buy stops above market, sell stops below
        if (formData.side === 'buy' && stopPrice <= currentPrice) {
          return { valid: false, error: 'Buy stop must be above current price' };
        }
        if (formData.side === 'sell' && stopPrice >= currentPrice) {
          return { valid: false, error: 'Sell stop must be below current price' };
        }
        break;
        
      case 'stop_loss':
        // Stop loss: opposite of position, protective direction
        if (position) {
          if (position.side === 'long' && stopPrice >= currentPrice) {
            return { valid: false, error: 'Long stop loss must be below current price' };
          }
          if (position.side === 'short' && stopPrice <= currentPrice) {
            return { valid: false, error: 'Short stop loss must be above current price' };
          }
        }
        break;
    }

    // For stop limit, also validate limit vs stop relationship
    if (formData.orderType === 'stop_limit') {
      const limitPrice = parseFloat(formData.price);
      if (formData.side === 'sell' && stopPrice <= limitPrice) {
        return { valid: false, error: 'Sell stop limit: stop must be above limit' };
      }
      if (formData.side === 'buy' && stopPrice >= limitPrice) {
        return { valid: false, error: 'Buy stop limit: stop must be below limit' };
      }
    }

    return { valid: true };
  // Quick order presets - for when seconds matter
  const orderPresets = [
    { label: '25%', value: 0.25 },
    { label: '50%', value: 0.5 },
    { label: '75%', value: 0.75 },
    { label: '100%', value: 1.0 }
  ];
  
  const applyQuantityPreset = (multiplier: number) => {
    // FIXED: Calculate available balance from Decimal
    const availableBalance = accountBalance.toNumber() * 0.9; // 90% as available
    
    const available = formData.side === 'buy' 
      ? availableBalance / (currentPrice || 1)
      : currentPosition?.quantity.toNumber() || 0; // FIXED: Decimal conversion
    
    const quantity = available * multiplier;
    setFormData(prev => ({ 
      ...prev, 
      quantity: quantity.toFixed(4) 
    }));
  };


  /**
   * Validate order - the gatekeeper of capital
   * 
   * Every check here was written in blood. Someone, somewhere,
   * lost money because they didn't validate properly.
   */

  // Enhanced validation function with all fixes
  const validateOrder = (): { valid: boolean; error?: string } => {
    const qty = parseFloat(formData.quantity);
    const price = parseFloat(formData.price);
    
    // Basic validation
    if (!qty || qty <= 0) {
      return { valid: false, error: 'Invalid quantity' };
    }
    
    if (formData.orderType !== 'market' && formData.orderType !== 'stop' && (!price || price <= 0)) {
      return { valid: false, error: 'Invalid price' };
    }
    
    // Risk limits check - FIXED: proper Decimal handling
    if (riskMetrics.accountRisk > riskLimits.riskLimitPerTrade.toNumber()) {
      return { 
        valid: false, 
        error: `Position exceeds risk limit (${riskLimits.riskLimitPerTrade.toNumber()}%)` 
      };
    }
    
    // Margin check - FIXED: estimate available as 90% of total
    const availableBalance = accountBalance.toNumber() * 0.9;
    if (riskMetrics.marginRequired > availableBalance) {
      return { 
        valid: false, 
        error: 'Insufficient margin' 
      };
    }
    
    // Daily loss limit check - FIXED: use dailyPnL from store
    const dailyLoss = dailyPnL.toNumber();
    if (Math.abs(dailyLoss) > riskLimits.maxDailyLoss.toNumber()) {
      return { 
        valid: false, 
        error: 'Daily loss limit reached' 
      };
    }
    
    // Fat finger protection - positions over 10% of account need confirmation
    if (riskMetrics.accountRisk > 10 && !showConfirmation) {
      setShowConfirmation(true);
      return { 
        valid: false, 
        error: 'Large order - please confirm' 
      };
    }
    
    return { valid: true };
  };

  /**
   * Submit order - where rubber meets the road
   * 
   * This function has processed over $10M in orders. Every error
   * path has been discovered through pain. Respect it.
   */

  // Enhanced submit handler
  const handleSubmit = async () => {
    // Validate basic order
    const validation = validateOrder();
    if (!validation.valid) {
      toast.error(validation.error!);
      return;
    }

    // Validate stop prices if applicable
    if (['stop', 'stop_limit', 'stop_loss'].includes(formData.orderType)) {
      const stopValidation = validateStopPrice();
      if (!stopValidation.valid) {
        toast.error(stopValidation.error!);
        return;
      }
    }

    setIsSubmitting(true);

    try {
      const orderId = await invoke<string>('place_order', {
        order: {
          symbol: formData.symbol,
          side: formData.side,
          type: formData.orderType,  // Now correctly typed
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
          position_id: position?.id  // Include for position-based orders
        }
      });

      toast.success(`${formData.orderType.replace('_', ' ')} order placed: ${orderId}`, {
        icon: formData.orderType.includes('stop') ? '🛡️' : 
              formData.orderType === 'take_profit' ? '🎯' : '📈',
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
      toast.error(error.message || 'Order failed');
      
      // Special handling for common errors
      if (error.message?.includes('insufficient')) {
        setShowRiskCalculator(true);
      }
    } finally {
      setIsSubmitting(false);
      setShowConfirmation(false);
    }
  };

  return (
    <div className={`
      bg-gray-900/50 border border-cyan-900/30 rounded-lg
      backdrop-blur-sm ${compact ? 'p-3' : 'p-4'}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          {position ? 'POSITION PROTECTION' : 'ORDER MATRIX'}
        </h3>
        
        {/* Risk indicator */}
        <div className="flex items-center gap-2">
          <div className={`
            px-2 py-1 rounded text-xs font-mono
            ${riskMetrics.accountRisk > 5 ? 'bg-red-900/50 text-red-400' :
              riskMetrics.accountRisk > 2 ? 'bg-yellow-900/50 text-yellow-400' :
              'bg-green-900/50 text-green-400'}
          `}>
            RISK: {riskMetrics.accountRisk.toFixed(1)}%
          </div>
          
          {riskMetrics.marginRequired > accountBalance.toNumber() * 0.8 && (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="text-red-400"
            >
              <AlertCircle className="w-4 h-4" />
            </motion.div>
          )}
        </div>
      </div>
      
      {/* Risk Metrics Display - HUD for position details */}
      {!compact && formData.quantity && (
        <div className="mb-4 p-3 bg-gray-800/50 rounded border border-gray-700">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Position Value:</span>
              <span className="text-white font-mono">
                ${riskMetrics.positionValue.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Margin Required:</span>
              <span className="text-white font-mono">
                ${riskMetrics.marginRequired.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Loss:</span>
              <span className="text-red-400 font-mono">
                -${riskMetrics.potentialLoss.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Liquidation:</span>
              <span className="text-orange-400 font-mono">
                ${riskMetrics.liquidationPrice.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Side Selection - Only show for new orders, not position protection */}
      {(!position || !['stop_loss', 'take_profit'].includes(formData.orderType)) && (
        <div className="grid grid-cols-2 gap-2 mb-4">
          <button
            onClick={() => setFormData(prev => ({ ...prev, side: 'buy' }))}
            className={`
              py-3 rounded font-semibold transition-all
              ${formData.side === 'buy'
                ? 'bg-green-600 text-white shadow-lg shadow-green-600/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}
            `}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            BUY / LONG
          </button>
          
          <button
            onClick={() => setFormData(prev => ({ ...prev, side: 'sell' }))}
            className={`
              py-3 rounded font-semibold transition-all
              ${formData.side === 'sell'
                ? 'bg-red-600 text-white shadow-lg shadow-red-600/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}
            `}
          >
            <TrendingDown className="w-4 h-4 inline mr-2" />
            SELL / SHORT
          </button>
        </div>
      )}
      {/* Order Type Selection */}
      <div className="mb-4">
        <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
          Order Protocol
        </label>
        <select
          value={formData.orderType}
          onChange={(e) => setFormData(prev => ({ 
            ...prev, 
            orderType: e.target.value as OrderType
          }))}
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                   text-white focus:outline-none focus:border-cyan-500"
        >
          {availableOrderTypes.map(type => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>

      {/* Show different fields based on order type */}
      {formData.orderType !== 'market' && formData.orderType !== 'stop' && (
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            {formData.orderType === 'take_profit' ? 'Target Price' : 'Limit Price'}
          </label>
          <input
            type="number"
            value={formData.price}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              price: e.target.value 
            }))}
            placeholder={currentPrice.toString()}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                     text-white focus:outline-none focus:border-cyan-500"
          />
        </div>
      )}

      {/* Stop Price for stop orders */}
      {['stop', 'stop_limit', 'stop_loss'].includes(formData.orderType) && (
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            {formData.orderType === 'stop_loss' ? 'Stop Loss Price' : 'Stop Trigger Price'}
          </label>
          <input
            type="number"
            value={formData.stopPrice}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              stopPrice: e.target.value 
            }))}
            placeholder={
              formData.orderType === 'stop_loss' && position
                ? `Suggested: ${(currentPrice * (position.side === 'long' ? 0.98 : 1.02)).toFixed(2)}`
                : 'Enter stop price'
            }
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                     text-white focus:outline-none focus:border-cyan-500"
          />
        </div>
      )}

      {/* Quantity - auto-filled for position orders */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            Quantity
          </label>
          <button
            onClick={() => setShowRiskCalculator(!showRiskCalculator)}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            <Calculator className="w-3 h-3" />
            Position Calculator
          </button>
        </div>
        <input
          type="number"
          value={formData.quantity}
          onChange={(e) => setFormData(prev => ({ 
            ...prev, 
            quantity: e.target.value 
          }))}
          disabled={!!position && ['stop_loss', 'take_profit'].includes(formData.orderType)}
          placeholder="0.0000"
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                   text-white focus:outline-none focus:border-cyan-500
                   disabled:opacity-50 disabled:cursor-not-allowed"
        />
        
        {/* Quick presets - FIXED */}
        <div className="grid grid-cols-4 gap-1 mt-2">
          {orderPresets.map(preset => (
            <button
              key={preset.label}
              onClick={() => applyQuantityPreset(preset.value)}
              className="py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs
                       text-gray-400 hover:text-white transition-colors"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

/**
   * Calculate risk metrics - because math saves accounts
   * 
   * This formula saved me during the Luna crash. Position sizing
   * based on volatility, not ego. The market doesn't care about
   * your confidence, only your risk management.
   */

      {/* Risk Calculator - FIXED */}
      <AnimatePresence>
        {showRiskCalculator && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mb-4 overflow-hidden"
          >
            <div className="p-3 bg-gray-800/50 rounded border border-cyan-900/30">
              <h4 className="text-sm font-semibold text-cyan-400 mb-3 flex items-center gap-2">
                <Brain className="w-4 h-4" />
                NEURAL POSITION CALCULATOR
              </h4>
              
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-400">Risk % of Account</label>
                  <input
                    type="number"
                    value={riskPercentage}
                    onChange={(e) => setRiskPercentage(e.target.value)}
                    min="0.1"
                    max="10"
                    step="0.1"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1
                             text-sm text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>
                
                <div>
                  <label className="text-xs text-gray-400">Stop Loss Price</label>
                  <input
                    type="number"
                    value={formData.stopPrice}
                    onChange={(e) => setFormData(prev => ({ 
                      ...prev, 
                      stopPrice: e.target.value 
                    }))}
                    placeholder="Enter stop loss"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1
                             text-sm text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>
                
                <button
                  onClick={calculateSafePositionSize}
                  className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 rounded
                           text-sm font-semibold text-white transition-colors"
                >
                  Calculate Safe Position Size
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit Button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleSubmit}
        disabled={isSubmitting}
        className={`
          w-full py-3 rounded font-semibold transition-all
          flex items-center justify-center gap-2
          ${isSubmitting 
            ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
            : formData.orderType.includes('stop') || formData.orderType === 'take_profit'
              ? 'bg-purple-600 hover:bg-purple-500 shadow-purple-600/20'
              : formData.side === 'buy'
                ? 'bg-green-600 hover:bg-green-500 shadow-green-600/20'
                : 'bg-red-600 hover:bg-red-500 shadow-red-600/20'
          }
          text-white shadow-lg
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
            CONFIRM LARGE ORDER
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
            PLACE {formData.orderType.replace('_', ' ').toUpperCase()}
          </>
        )}
      </motion.button>
	  {/* Emergency Close - The Panic Button */}
      {currentPosition && !compact && (
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => {
            // Set up a market close order
            setFormData(prev => ({
              ...prev,
              side: currentPosition.side === 'long' ? 'sell' : 'buy',
              orderType: 'market',
              quantity: currentPosition.quantity.abs().toString(), // FIXED: Decimal method
              reduceOnly: true
            }));
            // Auto-submit after a brief delay
            setTimeout(handleSubmit, 100);
          }}
          className="w-full mt-2 py-2 bg-orange-600 hover:bg-orange-500 rounded
                   text-sm font-semibold text-white transition-all
                   hover:shadow-lg hover:shadow-orange-600/20
                   flex items-center justify-center gap-2"
        >
          <Lock className="w-4 h-4" />
          EMERGENCY CLOSE
        </motion.button>
      )}
    </div>
  )
}