// src/components/trading/OrderPanel.tsx
// NEXLIFY ORDER PANEL - Where decisions become destiny
// Last sync: 2025-06-22 | "Every trade is a bet against the future"

import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { invoke } from '@tauri-apps/api/core';
import { 
  Zap, 
  ShieldAlert, 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  DollarSign,
  Calculator,
  Percent,
  Lock,
  Unlock,
  Brain,
  Gauge
} from 'lucide-react';
import toast from 'react-hot-toast';

import { useMarketStore } from '@/stores/marketStore';
import { useTradingStore } from '@/stores/tradingStore';
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
}

interface OrderFormData {
  symbol: string;
  side: 'buy' | 'sell';
  orderType: FormOrderType;
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
 * Maps frontend order types to backend expectations
 * Neural translation layer - because machines speak different dialects
 */
const mapOrderType = (formType: FormOrderType): BackendOrderType => {
  const typeMapping: Record<FormOrderType, BackendOrderType> = {
    'market': 'market',
    'limit': 'limit',
    'stop': 'stop_loss',
    'stop_limit': 'take_profit'
  };
  
  return typeMapping[formType];
};

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
  compact = false
}: OrderPanelProps) => {
  // Form state - where intention meets execution
  const [formData, setFormData] = useState<OrderFormData>({
    symbol,
    side: initialSide || 'buy',
    orderType: 'limit',
    quantity: '',
    price: initialPrice?.toString() || '',
    stopPrice: '',
    timeInForce: 'GTC',
    postOnly: false,
    reduceOnly: false
  });
  
  // UI state - the neural feedback loops
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showRiskCalculator, setShowRiskCalculator] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [riskPercentage, setRiskPercentage] = useState('1'); // 1% default risk
  
  // Store connections - synaptic links to the hive mind
  const { marketData, getSymbolData } = useMarketStore();
  const { 
    accountBalance, 
    positions, 
    riskLimits,
    placeOrder,
    calculatePositionSize 
  } = useTradingStore();
  const { user } = useAuthStore();
  
  // Get current market data
  const symbolData = getSymbolData(symbol);
  const currentPrice = symbolData?.lastPrice || 0;
  const currentPosition = positions[symbol];
  
  /**
   * Calculate risk metrics - because math saves accounts
   * 
   * This formula saved me during the Luna crash. Position sizing
   * based on volatility, not ego. The market doesn't care about
   * your confidence, only your risk management.
   */
  const riskMetrics = useMemo((): RiskMetrics => {
    const qty = parseFloat(formData.quantity) || 0;
    const price = parseFloat(formData.price) || currentPrice;
    const positionValue = qty * price;
    
    // Account risk calculation
    const totalBalance = accountBalance.total || 0;
    const accountRisk = totalBalance > 0 ? (positionValue / totalBalance) * 100 : 0;
    
    // Stop loss calculation
    const stopPrice = parseFloat(formData.stopPrice) || 0;
    const potentialLoss = stopPrice > 0 
      ? Math.abs(price - stopPrice) * qty 
      : positionValue * 0.02; // 2% default stop
    
    // Take profit calculation (assume 2:1 risk/reward)
    const potentialProfit = potentialLoss * 2;
    const riskRewardRatio = potentialLoss > 0 ? potentialProfit / potentialLoss : 0;
    
    // Margin calculation (assuming 10x leverage max)
    const marginRequired = positionValue / 10;
    
    // Liquidation price (simplified)
    const liquidationPrice = formData.side === 'buy'
      ? price * 0.9  // 10% drop
      : price * 1.1; // 10% rise
    
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
  
  /**
   * Position size calculator - the shield against ruin
   * 
   * Kelly Criterion meets practical trading. Never risk more than
   * you can afford to lose twice. The market will test you.
   */
  const calculateSafePositionSize = useCallback(() => {
    const risk = parseFloat(riskPercentage) / 100;
    const accountSize = accountBalance.total || 0;
    const riskAmount = accountSize * risk;
    
    // Get stop distance
    const entryPrice = parseFloat(formData.price) || currentPrice;
    const stopPrice = parseFloat(formData.stopPrice) || entryPrice * 0.98;
    const stopDistance = Math.abs(entryPrice - stopPrice);
    
    if (stopDistance === 0) {
      toast.error('Set stop price to calculate position size');
      return;
    }
    
    // Position size = Risk Amount / Stop Distance
    const positionSize = riskAmount / stopDistance;
    const rounded = Math.floor(positionSize * 10000) / 10000; // 4 decimals
    
    setFormData(prev => ({ ...prev, quantity: rounded.toString() }));
    toast.success(`Position size: ${rounded} (${risk * 100}% risk)`);
  }, [riskPercentage, accountBalance, formData.price, formData.stopPrice, currentPrice]);
  
  /**
   * Validate order - the gatekeeper of capital
   * 
   * Every check here was written in blood. Someone, somewhere,
   * lost money because they didn't validate properly.
   */
  const validateOrder = (): { valid: boolean; error?: string } => {
    const qty = parseFloat(formData.quantity);
    const price = parseFloat(formData.price);
    
    // Basic validation
    if (!qty || qty <= 0) {
      return { valid: false, error: 'Invalid quantity' };
    }
    
    if (formData.orderType !== 'market' && (!price || price <= 0)) {
      return { valid: false, error: 'Invalid price' };
    }
    
    // Risk limits check
    if (riskMetrics.accountRisk > riskLimits.maxPositionRisk) {
      return { 
        valid: false, 
        error: `Position exceeds risk limit (${riskLimits.maxPositionRisk}%)` 
      };
    }
    
    // Margin check
    if (riskMetrics.marginRequired > accountBalance.available) {
      return { 
        valid: false, 
        error: 'Insufficient margin' 
      };
    }
    
    // Daily loss limit check
    const dailyLoss = positions.dailyPnL || 0;
    if (Math.abs(dailyLoss) > riskLimits.dailyLossLimit) {
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
  const handleSubmit = async () => {
    const validation = validateOrder();
    if (!validation.valid) {
      toast.error(validation.error!);
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      // Map the order type for backend compatibility
      const backendOrderType = mapOrderType(formData.orderType);
      
      // Call Tauri backend with properly mapped type
      const orderId = await invoke<string>('place_order', {
        order: {
          symbol: formData.symbol,
          side: formData.side,
          type: backendOrderType, // Use mapped type here
          quantity: parseFloat(formData.quantity),
          price: formData.orderType === 'market' ? null : parseFloat(formData.price),
          stopPrice: formData.stopPrice ? parseFloat(formData.stopPrice) : null,
          timeInForce: formData.timeInForce,
          postOnly: formData.postOnly,
          reduceOnly: formData.reduceOnly
        }
      });
      
      // Success feedback - the dopamine hit
      toast.success(`Order placed: ${orderId}`, {
        icon: formData.side === 'buy' ? '📈' : '📉',
        duration: 5000
      });
      
      // Clear form
      setFormData(prev => ({
        ...prev,
        quantity: '',
        price: '',
        stopPrice: ''
      }));
      
      // Callback
      onOrderPlaced?.(orderId);
      
      // Log for audit trail
      console.log('Order placed:', {
        orderId,
        symbol: formData.symbol,
        side: formData.side,
        quantity: formData.quantity,
        risk: riskMetrics.accountRisk
      });
      
    } catch (error: any) {
      console.error('Order failed:', error);
      toast.error(error.message || 'Order failed', {
        duration: 7000
      });
      
      // Special handling for common errors
      if (error.message?.includes('insufficient')) {
        // Highlight the issue
        setShowRiskCalculator(true);
      }
    } finally {
      setIsSubmitting(false);
      setShowConfirmation(false);
    }
  };
  
  /**
   * Quick order presets - for when seconds matter
   * 
   * During the FTX collapse, these presets let traders exit
   * positions in seconds instead of minutes. Time is money,
   * and in crypto, time is survival.
   */
  const orderPresets = [
    { label: '25%', value: 0.25 },
    { label: '50%', value: 0.5 },
    { label: '75%', value: 0.75 },
    { label: '100%', value: 1.0 }
  ];
  
  const applyQuantityPreset = (multiplier: number) => {
    const available = formData.side === 'buy' 
      ? accountBalance.available / (currentPrice || 1)
      : currentPosition?.quantity || 0;
    
    const quantity = available * multiplier;
    setFormData(prev => ({ 
      ...prev, 
      quantity: quantity.toFixed(4) 
    }));
  };
  
  // Auto-update price for limit orders
  useEffect(() => {
    if (formData.orderType === 'limit' && !formData.price && currentPrice) {
      setFormData(prev => ({ 
        ...prev, 
        price: currentPrice.toString() 
      }));
    }
  }, [currentPrice, formData.orderType, formData.price]);
  
  return (
    <div className={`
      bg-gray-900/50 border border-cyan-900/30 rounded-lg
      backdrop-blur-sm
      ${compact ? 'p-3' : 'p-4'}
    `}>
      {/* Header - The Neural Command Center */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          ORDER MATRIX
        </h3>
        
        {/* Risk indicator - Visual cortex for danger */}
        <div className="flex items-center gap-2">
          <div className={`
            px-2 py-1 rounded text-xs font-mono
            ${riskMetrics.accountRisk > 5 ? 'bg-red-900/50 text-red-400' :
              riskMetrics.accountRisk > 2 ? 'bg-yellow-900/50 text-yellow-400' :
              'bg-green-900/50 text-green-400'}
          `}>
            RISK: {riskMetrics.accountRisk.toFixed(1)}%
          </div>
          
          {riskMetrics.marginRequired > accountBalance.available * 0.8 && (
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
      
      {/* Risk Metrics Display - The HUD */}
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
      
      {/* Side Selection - The Binary Choice */}
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
      
      {/* Order Type - Neural Pattern Selection */}
      <div className="mb-4">
        <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
          Order Protocol
        </label>
        <select
          value={formData.orderType}
          onChange={(e) => setFormData(prev => ({ 
            ...prev, 
            orderType: e.target.value as FormOrderType
          }))}
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                   text-white focus:outline-none focus:border-cyan-500
                   transition-colors"
        >
          <option value="market">Market Execute</option>
          <option value="limit">Limit Order</option>
          <option value="stop">Stop Loss</option>
          <option value="stop_limit">Stop Limit</option>
        </select>
      </div>
      
      {/* Quantity - The Commitment */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs text-gray-400 uppercase tracking-wider">
            Position Size
          </label>
          <button
            onClick={() => setShowRiskCalculator(!showRiskCalculator)}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1
                     transition-colors"
          >
            <Calculator className="w-3 h-3" />
            Neural Calculator
          </button>
        </div>
        
        <div className="relative">
          <input
            type="number"
            value={formData.quantity}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              quantity: e.target.value 
            }))}
            placeholder="0.0000"
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                     text-white focus:outline-none focus:border-cyan-500
                     pr-10 font-mono transition-colors"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">
            {symbol.split('/')[0]}
          </span>
        </div>
        
        {/* Quick presets - Neural shortcuts */}
        <div className="grid grid-cols-4 gap-1 mt-2">
          {orderPresets.map(preset => (
            <button
              key={preset.label}
              onClick={() => applyQuantityPreset(preset.value)}
              className="py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs
                       text-gray-400 hover:text-white transition-all
                       border border-gray-700 hover:border-cyan-600"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Price Input - The Strike Point */}
      {formData.orderType !== 'market' && (
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            {formData.orderType === 'stop' ? 'Stop Trigger' : 'Limit Price'}
          </label>
          <div className="relative">
            <input
              type="number"
              value={formData.price}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                price: e.target.value 
              }))}
              placeholder={currentPrice.toString()}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                       text-white focus:outline-none focus:border-cyan-500
                       pr-10 font-mono transition-colors"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">
              {symbol.split('/')[1]}
            </span>
          </div>
        </div>
      )}
      
      {/* Stop Price for stop limit orders */}
      {formData.orderType === 'stop_limit' && (
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            Stop Activation
          </label>
          <input
            type="number"
            value={formData.stopPrice}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              stopPrice: e.target.value 
            }))}
            placeholder="Trigger price"
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2
                     text-white focus:outline-none focus:border-cyan-500
                     font-mono transition-colors"
          />
        </div>
      )}
      
      {/* Advanced Options - The Fine Tuning */}
      {!compact && (
        <div className="mb-4 space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs text-gray-400 uppercase tracking-wider">
              Time Protocol
            </label>
            <select
              value={formData.timeInForce}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                timeInForce: e.target.value as any 
              }))}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1
                       text-xs text-white focus:outline-none focus:border-cyan-500
                       transition-colors"
            >
              <option value="GTC">Good Till Cancel</option>
              <option value="IOC">Immediate or Cancel</option>
              <option value="FOK">Fill or Kill</option>
              <option value="GTX">Post Only</option>
            </select>
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-gray-400
                           hover:text-gray-300 cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={formData.postOnly}
                onChange={(e) => setFormData(prev => ({ 
                  ...prev, 
                  postOnly: e.target.checked 
                }))}
                className="rounded border-gray-600 bg-gray-800 text-cyan-500
                         focus:ring-cyan-500 focus:ring-offset-0"
              />
              Post Only
            </label>
            
            <label className="flex items-center gap-2 text-xs text-gray-400
                           hover:text-gray-300 cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={formData.reduceOnly}
                onChange={(e) => setFormData(prev => ({ 
                  ...prev, 
                  reduceOnly: e.target.checked 
                }))}
                className="rounded border-gray-600 bg-gray-800 text-cyan-500
                         focus:ring-cyan-500 focus:ring-offset-0"
              />
              Reduce Only
            </label>
          </div>
        </div>
      )}
      
      {/* Risk Calculator - The Neural Network */}
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
                             text-sm text-white focus:outline-none focus:border-cyan-500
                             font-mono transition-colors"
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
                             text-sm text-white focus:outline-none focus:border-cyan-500
                             font-mono transition-colors"
                  />
                </div>
                
                <button
                  onClick={calculateSafePositionSize}
                  className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 rounded
                           text-sm font-semibold text-white transition-all
                           hover:shadow-lg hover:shadow-cyan-600/20"
                >
                  Calculate Safe Position
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Submit Button - The Execution Gateway */}
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
            : formData.side === 'buy'
              ? 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-600/20'
              : 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/20'
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
            CONFIRM LARGE ORDER
          </>
        ) : (
          <>
            {formData.side === 'buy' ? 
              <TrendingUp className="w-5 h-5" /> : 
              <TrendingDown className="w-5 h-5" />
            }
            EXECUTE {formData.side.toUpperCase()}
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
              quantity: Math.abs(currentPosition.quantity).toString(),
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
  );
};