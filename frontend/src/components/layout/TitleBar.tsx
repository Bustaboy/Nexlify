// frontend/src/components/layout/TitleBar.tsx

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Minus, 
  Square, 
  X, 
  Zap, 
  Wifi, 
  WifiOff,
  Shield,
  Activity
} from 'lucide-react';
import { cn } from '@lib/utils';
import { useConnectionStatus } from '@hooks/useWebSocket';
import { usePerformanceWarnings } from '@hooks/useSystemMonitor';
import { useSettingsStore } from '@stores/settingsStore';
import { isElectron } from '@lib/utils';

// The crown of our digital fortress - where control meets style
// Every button a choice, every indicator a story of battles fought in the data streams
// I've dragged this bar across three monitors more times than I can count

export const TitleBar: React.FC = () => {
  const { theme } = useSettingsStore();
  const { status: connectionStatus, icon: connectionIcon } = useConnectionStatus();
  const { performanceScore, warningLevel } = usePerformanceWarnings();
  
  const [isMaximized, setIsMaximized] = useState(false);
  const [time, setTime] = useState(new Date());

  // Keep time ticking - because in Night City, every second counts
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check if window is maximized
  useEffect(() => {
    if (!isElectron()) return;
    
    const checkMaximized = async () => {
      // This would check with Electron, placeholder for now
      setIsMaximized(false);
    };
    
    checkMaximized();
    
    // Listen for window state changes
    window.addEventListener('resize', checkMaximized);
    return () => window.removeEventListener('resize', checkMaximized);
  }, []);

  const handleMinimize = () => {
    if (isElectron()) {
      window.nexlify.window.minimize();
    }
  };

  const handleMaximize = () => {
    if (isElectron()) {
      window.nexlify.window.maximize();
      setIsMaximized(!isMaximized);
    }
  };

  const handleClose = () => {
    if (isElectron()) {
      window.nexlify.window.close();
    }
  };

  return (
    <div className={cn(
      "h-8 flex items-center justify-between",
      "bg-cyber-black border-b border-cyber-dark",
      "select-none cursor-default",
      // Make draggable except for buttons
      isElectron() && "app-drag"
    )}>
      {/* Left section - App identity */}
      <div className="flex items-center px-3 space-x-3">
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", duration: 0.5 }}
          className="flex items-center space-x-2"
        >
          <Zap className="w-4 h-4 text-neon-cyan" />
          <span className="text-xs font-cyber tracking-wider text-neon-cyan">
            NEXLIFY
          </span>
        </motion.div>
        
        {/* Version tag - wear it with pride */}
        <span className="text-xs text-gray-600">v3.0.0</span>
      </div>

      {/* Center section - Status indicators */}
      <div className="flex items-center space-x-4">
        {/* Connection status - your lifeline to the markets */}
        <div className="flex items-center space-x-1 group">
          <motion.div
            animate={{
              scale: connectionStatus === 'connected' ? [1, 1.2, 1] : 1,
            }}
            transition={{
              duration: 2,
              repeat: connectionStatus === 'connected' ? Infinity : 0,
              repeatType: 'loop'
            }}
          >
            {connectionStatus === 'connected' ? (
              <Wifi className="w-3 h-3 text-neon-green" />
            ) : connectionStatus === 'reconnecting' ? (
              <Activity className="w-3 h-3 text-neon-yellow animate-pulse" />
            ) : (
              <WifiOff className="w-3 h-3 text-neon-red" />
            )}
          </motion.div>
          <span className={cn(
            "text-xs font-mono transition-colors",
            connectionStatus === 'connected' ? 'text-green-400' : 
            connectionStatus === 'reconnecting' ? 'text-yellow-400' : 
            'text-red-400'
          )}>
            {connectionStatus.toUpperCase()}
          </span>
        </div>

        {/* System performance - know when your chrome's running hot */}
        <div className="flex items-center space-x-1">
          <div className={cn(
            "w-16 h-1 bg-cyber-dark rounded-full overflow-hidden",
            "relative group"
          )}>
            <motion.div
              className={cn(
                "h-full rounded-full",
                performanceScore > 80 ? "bg-neon-green" :
                performanceScore > 60 ? "bg-neon-yellow" :
                performanceScore > 40 ? "bg-neon-orange" :
                "bg-neon-red"
              )}
              initial={{ width: 0 }}
              animate={{ width: `${performanceScore}%` }}
              transition={{ duration: 0.5 }}
            />
            {/* Scanline effect - because details matter */}
            <div className="absolute inset-0 scan-lines opacity-30" />
          </div>
          <span className="text-xs text-gray-500 font-mono">
            {performanceScore}%
          </span>
        </div>

        {/* Security status - never let your guard down */}
        <motion.div
          whileHover={{ scale: 1.1 }}
          className="relative"
        >
          <Shield className={cn(
            "w-3 h-3 transition-colors",
            "text-neon-cyan opacity-80"
          )} />
          {/* Security pulse effect */}
          <motion.div
            className="absolute inset-0 rounded-full bg-neon-cyan/20"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.5, 0, 0.5]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeOut"
            }}
          />
        </motion.div>

        {/* Clock - time waits for no trader */}
        <div className="text-xs font-mono text-gray-500">
          {time.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
          })}
        </div>
      </div>

      {/* Right section - Window controls */}
      {isElectron() && (
        <div className="flex items-center no-drag">
          {/* Minimize - take a breather */}
          <motion.button
            whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
            whileTap={{ scale: 0.95 }}
            onClick={handleMinimize}
            className="h-8 w-12 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
          >
            <Minus className="w-3 h-3" />
          </motion.button>

          {/* Maximize/Restore - go big or go home */}
          <motion.button
            whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
            whileTap={{ scale: 0.95 }}
            onClick={handleMaximize}
            className="h-8 w-12 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
          >
            <Square className="w-3 h-3" />
          </motion.button>

          {/* Close - know when to walk away */}
          <motion.button
            whileHover={{ backgroundColor: 'rgba(255, 0, 0, 0.8)' }}
            whileTap={{ scale: 0.95 }}
            onClick={handleClose}
            className="h-8 w-12 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-3 h-3" />
          </motion.button>
        </div>
      )}

      {/* Glitch effect overlay - because perfection is an illusion */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-neon-cyan/5 to-transparent"
          animate={{
            x: ['-100%', '100%']
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      </div>

      {/* CSS for draggable regions */}
      <style jsx>{`
        .app-drag {
          -webkit-app-region: drag;
        }
        .no-drag {
          -webkit-app-region: no-drag;
        }
      `}</style>
    </div>
  );
};
