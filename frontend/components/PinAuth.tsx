/**
 * Nexlify PIN Authentication
 * The gateway to the trading matrix - prove you belong
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Lock, AlertCircle, Shield, Zap } from 'lucide-react';
import clsx from 'clsx';

// Stores
import { useAuthStore } from '../../stores/authStore';
import { useSettingsStore } from '../../stores/settingsStore';

// Utils
import { playSound } from '../../lib/sounds';

export const PinAuth: React.FC = () => {
  const { authenticateWithPin, failedAttempts, isLocked, lockoutUntil } = useAuthStore();
  const { soundEnabled, matrixRainEffect } = useSettingsStore();
  
  const [pin, setPin] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [error, setError] = useState('');
  const [enable2FA, setEnable2FA] = useState(false);
  const [showPin, setShowPin] = useState(false);
  const [lockoutTimer, setLockoutTimer] = useState<string>('');
  
  const pinInputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const firstInputRef = useRef<HTMLInputElement>(null);
  
  // Focus first input on mount
  useEffect(() => {
    setTimeout(() => {
      firstInputRef.current?.focus();
    }, 500);
  }, []);
  
  // Handle lockout timer
  useEffect(() => {
    if (isLocked && lockoutUntil) {
      const interval = setInterval(() => {
        const now = new Date();
        const remaining = lockoutUntil.getTime() - now.getTime();
        
        if (remaining <= 0) {
          setLockoutTimer('');
          clearInterval(interval);
        } else {
          const minutes = Math.floor(remaining / 60000);
          const seconds = Math.floor((remaining % 60000) / 1000);
          setLockoutTimer(`${minutes}:${seconds.toString().padStart(2, '0')}`);
        }
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isLocked, lockoutUntil]);
  
  // Handle PIN input
  const handlePinChange = (index: number, value: string) => {
    if (value.length > 1) {
      value = value.slice(-1); // Take only last character
    }
    
    if (!/^\d*$/.test(value)) {
      return; // Only allow digits
    }
    
    const newPin = pin.split('');
    newPin[index] = value;
    const updatedPin = newPin.join('');
    
    setPin(updatedPin);
    
    // Move to next input
    if (value && index < 5) {
      pinInputRefs.current[index + 1]?.focus();
    }
    
    // Auto-submit when complete
    if (updatedPin.length === 6 && !updatedPin.includes('')) {
      handleAuthenticate(updatedPin);
    }
    
    // Play sound
    if (soundEnabled && value) {
      playSound('click');
    }
  };
  
  // Handle backspace
  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !pin[index] && index > 0) {
      pinInputRefs.current[index - 1]?.focus();
    }
  };
  
  // Handle paste
  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text').slice(0, 6);
    
    if (/^\d+$/.test(pastedData)) {
      setPin(pastedData.padEnd(6, ''));
      
      if (pastedData.length === 6) {
        handleAuthenticate(pastedData);
      }
    }
  };
  
  // Authenticate
  const handleAuthenticate = async (pinValue: string = pin) => {
    if (pinValue.length !== 6 || isAuthenticating || isLocked) {
      return;
    }
    
    setIsAuthenticating(true);
    setError('');
    
    try {
      const result = await authenticateWithPin(pinValue, enable2FA);
      
      if (!result.success) {
        setError(result.error || 'Authentication failed');
        setPin('');
        
        // Reset inputs
        pinInputRefs.current.forEach(input => {
          if (input) input.value = '';
        });
        firstInputRef.current?.focus();
        
        // Shake animation
        const container = document.getElementById('pin-container');
        container?.classList.add('shake');
        setTimeout(() => {
          container?.classList.remove('shake');
        }, 500);
      }
      
    } catch (error) {
      setError('Connection error. Please try again.');
    } finally {
      setIsAuthenticating(false);
    }
  };
  
  // Remaining attempts indicator
  const remainingAttempts = Math.max(0, 5 - failedAttempts);
  
  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background effects */}
      {matrixRainEffect && (
        <div className="absolute inset-0 opacity-10">
          <div className="matrix-rain" />
        </div>
      )}
      
      {/* Animated background grid */}
      <div className="absolute inset-0">
        <div className="cyber-grid" />
      </div>
      
      {/* Main content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 w-full max-w-md px-6"
      >
        {/* Logo and title */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="inline-flex items-center justify-center w-20 h-20 mb-4 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 backdrop-blur-sm border border-cyan-500/50"
          >
            <Shield className="w-10 h-10 text-cyan-400" />
          </motion.div>
          
          <h1 className="text-3xl font-bold text-white mb-2">
            NEXLIFY ACCESS CONTROL
          </h1>
          
          <p className="text-gray-400 text-sm">
            Enter your 6-digit PIN to jack into the trading matrix
          </p>
        </div>
        
        {/* PIN input */}
        <div className="space-y-6">
          <div 
            id="pin-container"
            className="flex justify-center space-x-3"
          >
            {[0, 1, 2, 3, 4, 5].map((index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 + index * 0.05 }}
              >
                <input
                  ref={(el) => {
                    pinInputRefs.current[index] = el;
                    if (index === 0) {
                      firstInputRef.current = el;
                    }
                  }}
                  type={showPin ? "text" : "password"}
                  inputMode="numeric"
                  maxLength={1}
                  disabled={isAuthenticating || isLocked}
                  onChange={(e) => handlePinChange(index, e.target.value)}
                  onKeyDown={(e) => handleKeyDown(index, e)}
                  onPaste={index === 0 ? handlePaste : undefined}
                  className={clsx(
                    "w-12 h-14 text-center text-xl font-mono",
                    "bg-gray-900/50 backdrop-blur-sm",
                    "border-2 rounded-lg transition-all duration-200",
                    "focus:outline-none focus:ring-2 focus:ring-cyan-500/50",
                    {
                      "border-gray-700": !pin[index],
                      "border-cyan-500": pin[index],
                      "border-red-500": error,
                      "opacity-50": isLocked
                    }
                  )}
                />
              </motion.div>
            ))}
          </div>
          
          {/* Error message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center justify-center space-x-2 text-red-500"
              >
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Lockout timer */}
          <AnimatePresence>
            {isLocked && lockoutTimer && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="text-center"
              >
                <div className="inline-flex items-center space-x-2 px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <Lock className="w-4 h-4 text-red-500" />
                  <span className="text-red-500 font-mono">
                    Locked: {lockoutTimer}
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Attempts remaining */}
          {failedAttempts > 0 && !isLocked && (
            <div className="text-center">
              <span className="text-xs text-gray-500">
                {remainingAttempts} attempts remaining
              </span>
              <div className="mt-2 h-1 bg-gray-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: "100%" }}
                  animate={{ width: `${(remainingAttempts / 5) * 100}%` }}
                  className={clsx(
                    "h-full transition-all duration-300",
                    {
                      "bg-green-500": remainingAttempts > 3,
                      "bg-yellow-500": remainingAttempts === 3 || remainingAttempts === 2,
                      "bg-red-500": remainingAttempts === 1
                    }
                  )}
                />
              </div>
            </div>
          )}
          
          {/* Options */}
          <div className="space-y-4">
            {/* 2FA toggle */}
            <label className="flex items-center justify-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={enable2FA}
                onChange={(e) => setEnable2FA(e.target.checked)}
                disabled={isAuthenticating || isLocked}
                className="w-4 h-4 rounded border-gray-600 bg-gray-900 text-cyan-500 focus:ring-cyan-500/50"
              />
              <span className="text-sm text-gray-400">
                Enable Two-Factor Authentication
              </span>
            </label>
            
            {/* Show PIN toggle */}
            <label className="flex items-center justify-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showPin}
                onChange={(e) => setShowPin(e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-900 text-cyan-500 focus:ring-cyan-500/50"
              />
              <span className="text-sm text-gray-400">
                Show PIN
              </span>
            </label>
          </div>
          
          {/* Loading state */}
          <AnimatePresence>
            {isAuthenticating && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center space-x-2"
              >
                <div className="flex space-x-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{
                        duration: 1,
                        delay: i * 0.2,
                        repeat: Infinity
                      }}
                      className="w-2 h-2 bg-cyan-500 rounded-full"
                    />
                  ))}
                </div>
                <span className="text-sm text-gray-400">
                  Authenticating...
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Footer */}
        <div className="mt-12 text-center">
          <p className="text-xs text-gray-500">
            Default PIN: 2077 • Change immediately after first login
          </p>
          
          <div className="mt-4 flex items-center justify-center space-x-4 text-xs text-gray-600">
            <span>Nexlify Trading Matrix v3.0</span>
            <span>•</span>
            <span className="flex items-center space-x-1">
              <Zap className="w-3 h-3" />
              <span>Neural Net Active</span>
            </span>
          </div>
        </div>
      </motion.div>
      
      {/* CSS for animations */}
      <style jsx>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-10px); }
          75% { transform: translateX(10px); }
        }
        
        .shake {
          animation: shake 0.5s ease-in-out;
        }
        
        .cyber-grid {
          background-image: 
            linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px);
          background-size: 50px 50px;
          animation: grid-move 20s linear infinite;
        }
        
        @keyframes grid-move {
          0% { transform: translate(0, 0); }
          100% { transform: translate(50px, 50px); }
        }
        
        .matrix-rain {
          background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Ctext x='0' y='15' font-family='monospace' font-size='15' fill='%2300ff00'%3E0%3C/text%3E%3C/svg%3E");
          animation: rain 5s linear infinite;
        }
        
        @keyframes rain {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>
    </div>
  );
};
