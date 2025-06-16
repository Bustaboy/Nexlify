// frontend/src/components/common/LoadingScreen.tsx

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Brain, Shield, Activity } from 'lucide-react';

// The loading screen - that moment between clicking and trading
// Like watching your consciousness upload to the matrix
// Every spinner rotation a heartbeat, every progress tick a synapse firing

const loadingMessages = [
  "Initializing neural networks...",
  "Connecting to trading matrix...",
  "Calibrating market sensors...",
  "Loading price algorithms...",
  "Establishing secure channels...",
  "Syncing with exchange APIs...",
  "Preparing risk protocols...",
  "Activating AI companion...",
  "Securing data streams...",
  "Jacking into Night City..."
];

const loadingTips = [
  "Pro tip: The best trades happen when others panic",
  "Remember: Risk management is your best friend",
  "Fact: 90% of traders lose money. Be the 10%",
  "Wisdom: The market can stay irrational longer than you can stay solvent",
  "Truth: Every loss is a lesson if you're paying attention",
  "Insight: Volume precedes price - watch the flow",
  "Reality: Hope is not a strategy, but discipline is",
  "Secret: The best traders know when NOT to trade",
  "Mantra: Plan your trade, trade your plan",
  "Law: Protect your capital above all else"
];

export const LoadingScreen: React.FC<{
  message?: string;
  progress?: number;
  showTips?: boolean;
}> = ({ 
  message, 
  progress,
  showTips = true 
}) => {
  const [currentMessage, setCurrentMessage] = useState(0);
  const [currentTip, setCurrentTip] = useState(0);
  const [dots, setDots] = useState('');

  // Cycle through loading messages
  useEffect(() => {
    if (!message) {
      const messageInterval = setInterval(() => {
        setCurrentMessage((prev) => (prev + 1) % loadingMessages.length);
      }, 2000);
      
      return () => clearInterval(messageInterval);
    }
  }, [message]);

  // Cycle through tips
  useEffect(() => {
    if (showTips) {
      // Show first tip immediately, then cycle
      const tipInterval = setInterval(() => {
        setCurrentTip((prev) => (prev + 1) % loadingTips.length);
      }, 5000);
      
      return () => clearInterval(tipInterval);
    }
  }, [showTips]);

  // Animate dots
  useEffect(() => {
    const dotsInterval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);
    
    return () => clearInterval(dotsInterval);
  }, []);

  return (
    <div className="fixed inset-0 bg-cyber-black flex items-center justify-center z-50">
      {/* Background matrix effect */}
      <div className="absolute inset-0 opacity-5">
        <div className="matrix-rain" />
      </div>
      
      {/* Animated grid */}
      <div className="absolute inset-0 cyber-grid opacity-10" />
      
      {/* Main content */}
      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        {/* Logo animation */}
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ 
            type: "spring", 
            stiffness: 100,
            damping: 20,
            duration: 1 
          }}
          className="mb-8"
        >
          <div className="relative inline-block">
            {/* Rotating outer ring */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ 
                duration: 20, 
                repeat: Infinity, 
                ease: "linear" 
              }}
              className="absolute inset-0 w-32 h-32 rounded-full border-2 border-neon-cyan/30"
              style={{
                boxShadow: '0 0 40px rgba(0, 255, 255, 0.5)',
              }}
            />
            
            {/* Pulsing center */}
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="relative w-32 h-32 rounded-full bg-cyber-black border-2 border-neon-cyan flex items-center justify-center"
            >
              <Zap className="w-16 h-16 text-neon-cyan" />
            </motion.div>
            
            {/* Orbiting icons */}
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ 
                duration: 30, 
                repeat: Infinity, 
                ease: "linear" 
              }}
              className="absolute inset-0 w-32 h-32"
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-4">
                <Brain className="w-4 h-4 text-neon-purple" />
              </div>
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-4">
                <Shield className="w-4 h-4 text-neon-green" />
              </div>
              <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4">
                <Activity className="w-4 h-4 text-neon-yellow" />
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-3xl font-bold mb-4 font-cyber"
        >
          <span className="text-neon-cyan glitch-text" data-text="NEXLIFY">
            NEXLIFY
          </span>
        </motion.h1>

        {/* Loading message */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="mb-8"
        >
          <p className="text-gray-400 text-sm mb-2">
            {message || loadingMessages[currentMessage]}{dots}
          </p>
          
          {/* Progress bar */}
          <div className="h-1 bg-cyber-dark rounded-full overflow-hidden mt-4">
            <motion.div
              className="h-full bg-gradient-to-r from-neon-cyan to-neon-purple"
              initial={{ width: 0 }}
              animate={{ width: progress ? `${progress}%` : '100%' }}
              transition={{
                duration: progress ? 0.3 : 20,
                ease: progress ? "easeOut" : "linear"
              }}
            />
          </div>
        </motion.div>

        {/* Tips section */}
        {showTips && (
          <AnimatePresence mode="wait">
            <motion.div
              key={currentTip}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.5 }}
              className="mt-12"
            >
              <p className="text-xs text-gray-600 italic">
                {loadingTips[currentTip]}
              </p>
            </motion.div>
          </AnimatePresence>
        )}

        {/* System status indicators */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-8 flex justify-center space-x-6 text-xs"
        >
          <StatusIndicator label="Neural Net" status="online" />
          <StatusIndicator label="Security" status="active" />
          <StatusIndicator label="Data Feed" status="syncing" />
        </motion.div>
      </div>

      {/* Corner decorations */}
      <div className="absolute top-4 left-4">
        <CornerDecoration />
      </div>
      <div className="absolute top-4 right-4 rotate-90">
        <CornerDecoration />
      </div>
      <div className="absolute bottom-4 left-4 -rotate-90">
        <CornerDecoration />
      </div>
      <div className="absolute bottom-4 right-4 rotate-180">
        <CornerDecoration />
      </div>
    </div>
  );
};

// Status indicator component
const StatusIndicator: React.FC<{
  label: string;
  status: 'online' | 'active' | 'syncing' | 'offline';
}> = ({ label, status }) => {
  const statusColors = {
    online: 'bg-green-500',
    active: 'bg-cyan-500',
    syncing: 'bg-yellow-500',
    offline: 'bg-red-500'
  };

  return (
    <div className="flex items-center space-x-2">
      <motion.div
        animate={status === 'syncing' ? {
          scale: [1, 1.2, 1],
          opacity: [1, 0.5, 1]
        } : {}}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className={`w-2 h-2 rounded-full ${statusColors[status]}`}
      />
      <span className="text-gray-500">{label}</span>
    </div>
  );
};

// Corner decoration component
const CornerDecoration: React.FC = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
    <path
      d="M0 0 L24 0 L24 8 L8 8 L8 24 L0 24 Z"
      stroke="currentColor"
      strokeWidth="1"
      className="text-neon-cyan/30"
      fill="none"
    />
  </svg>
);

// Simple loading spinner for inline use
export const LoadingSpinner: React.FC<{
  size?: 'sm' | 'md' | 'lg';
  color?: string;
}> = ({ size = 'md', color = 'text-neon-cyan' }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }}
      className={`${sizes[size]} ${color}`}
    >
      <svg
        className="w-full h-full"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    </motion.div>
  );
};

// Progress loading bar for operations
export const LoadingProgress: React.FC<{
  progress: number;
  label?: string;
  showPercent?: boolean;
}> = ({ progress, label, showPercent = true }) => (
  <div className="w-full">
    {(label || showPercent) && (
      <div className="flex justify-between text-xs text-gray-400 mb-2">
        <span>{label}</span>
        {showPercent && <span>{Math.round(progress)}%</span>}
      </div>
    )}
    <div className="h-2 bg-cyber-dark rounded-full overflow-hidden">
      <motion.div
        className="h-full bg-gradient-to-r from-neon-cyan to-neon-purple"
        initial={{ width: 0 }}
        animate={{ width: `${progress}%` }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      />
    </div>
  </div>
);
