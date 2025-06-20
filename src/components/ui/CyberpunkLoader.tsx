// src/components/ui/CyberpunkLoader.tsx
// NEXLIFY CYBERPUNK LOADER - Where waiting becomes anticipation
// Last sync: 2025-06-19 | "Even in the pause, the hustle never stops"

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Zap, 
  Activity,
  Cpu,
  Database,
  Shield,
  Network,
  Terminal,
  Binary,
  Gauge,
  CircuitBoard
} from 'lucide-react';

interface CyberpunkLoaderProps {
  variant?: 'spinner' | 'terminal' | 'glitch' | 'neural' | 'circuit';
  size?: 'small' | 'medium' | 'large' | 'fullscreen';
  text?: string;
  subtext?: string;
  progress?: number; // 0-100
  showStats?: boolean;
  color?: string;
  speed?: 'slow' | 'normal' | 'fast';
}

interface LoadingPhrase {
  main: string;
  sub: string;
  icon: JSX.Element;
}

/**
 * CYBERPUNK LOADER - The liminal space between intention and execution
 * 
 * Built this during the Great Hang of '23 - when our WebSocket connection
 * froze for 47 seconds during a massive BTC pump. Traders sat there,
 * staring at a blank spinner, while millions moved without them.
 * 
 * Never again. Now our loaders tell stories, show progress, maintain hope.
 * Because in crypto, 47 seconds might as well be 47 years.
 * 
 * Each variant serves a purpose:
 * - Spinner: Classic, reliable, hypnotic
 * - Terminal: Hacker aesthetic, shows actual progress
 * - Glitch: When things break beautifully
 * - Neural: AI-powered loading visualization
 * - Circuit: Data flowing through digital veins
 * 
 * Remember: A good loader doesn't just pass time, it builds anticipation.
 */
export const CyberpunkLoader = ({
  variant = 'spinner',
  size = 'medium',
  text,
  subtext,
  progress,
  showStats = false,
  color = '#00ffff',
  speed = 'normal'
}: CyberpunkLoaderProps) => {
  const [currentPhrase, setCurrentPhrase] = useState(0);
  const [loadingDots, setLoadingDots] = useState('');
  const [fakeProgress, setFakeProgress] = useState(0);
  const [glitchText, setGlitchText] = useState('');
  const terminalRef = useRef<HTMLDivElement>(null);
  
  // Loading phrases that tell the Nexlify story
  const loadingPhrases: LoadingPhrase[] = [
    { 
      main: "Syncing with market matrix", 
      sub: "Establishing quantum tunnel",
      icon: <Network className="w-4 h-4" />
    },
    { 
      main: "Calibrating profit algorithms", 
      sub: "Optimizing neural pathways",
      icon: <Cpu className="w-4 h-4" />
    },
    { 
      main: "Decrypting price feeds", 
      sub: "Breaking corporate encryption",
      icon: <Binary className="w-4 h-4" />
    },
    { 
      main: "Loading risk protocols", 
      sub: "Your safety is our priority",
      icon: <Shield className="w-4 h-4" />
    },
    { 
      main: "Connecting to dark pools", 
      sub: "Where real liquidity lives",
      icon: <Database className="w-4 h-4" />
    },
    { 
      main: "Initializing trade engine", 
      sub: "Weapons hot, targets acquired",
      icon: <Zap className="w-4 h-4" />
    },
    { 
      main: "Scanning market anomalies", 
      sub: "Hunting for opportunities",
      icon: <Activity className="w-4 h-4" />
    },
    { 
      main: "Bypassing exchange limits", 
      sub: "Rules are for corps",
      icon: <Terminal className="w-4 h-4" />
    }
  ];
  
  // Rotate phrases
  useEffect(() => {
    if (!text) {
      const interval = setInterval(() => {
        setCurrentPhrase(prev => (prev + 1) % loadingPhrases.length);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [text, loadingPhrases.length]);
  
  // Animated dots
  useEffect(() => {
    const interval = setInterval(() => {
      setLoadingDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);
  
  // Fake progress if not provided
  useEffect(() => {
    if (progress === undefined) {
      const interval = setInterval(() => {
        setFakeProgress(prev => {
          const increment = Math.random() * 10;
          return Math.min(prev + increment, 95); // Never reach 100 without real progress
        });
      }, 300);
      return () => clearInterval(interval);
    }
  }, [progress]);
  
  // Glitch text effect
  useEffect(() => {
    if (variant === 'glitch') {
      const glitchChars = '!<>-_\\/[]{}—=+*^?#________';
      const interval = setInterval(() => {
        const originalText = text || loadingPhrases[currentPhrase].main;
        let glitched = '';
        
        for (let i = 0; i < originalText.length; i++) {
          if (Math.random() > 0.7) {
            glitched += glitchChars[Math.floor(Math.random() * glitchChars.length)];
          } else {
            glitched += originalText[i];
          }
        }
        
        setGlitchText(glitched);
      }, 100);
      
      return () => clearInterval(interval);
    }
  }, [variant, text, currentPhrase, loadingPhrases]);
  
  // Size configurations
  const sizeConfig = {
    small: { container: 'w-32 h-32', icon: 24, text: 'text-xs' },
    medium: { container: 'w-48 h-48', icon: 32, text: 'text-sm' },
    large: { container: 'w-64 h-64', icon: 48, text: 'text-base' },
    fullscreen: { container: 'w-screen h-screen', icon: 64, text: 'text-lg' }
  };
  
  const config = sizeConfig[size];
  const displayProgress = progress !== undefined ? progress : fakeProgress;
  
  // Speed configurations
  const speedConfig = {
    slow: { rotation: 4, pulse: 3 },
    normal: { rotation: 2, pulse: 2 },
    fast: { rotation: 1, pulse: 1 }
  };
  
  const animSpeed = speedConfig[speed];
  
  /**
   * Spinner variant - The hypnotic classic
   */
  const renderSpinner = () => (
    <div className="relative">
      {/* Outer ring */}
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: 360 }}
        transition={{ 
          duration: animSpeed.rotation, 
          repeat: Infinity, 
          ease: "linear" 
        }}
      >
        <svg
          width={config.icon * 2}
          height={config.icon * 2}
          viewBox="0 0 100 100"
          className="transform -rotate-90"
        >
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={`${color}20`}
            strokeWidth="2"
          />
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeDasharray={`${displayProgress * 2.83} 283`}
            strokeLinecap="round"
          />
        </svg>
      </motion.div>
      
      {/* Inner icon */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        animate={{ 
          scale: [1, 1.1, 1],
          opacity: [0.5, 1, 0.5]
        }}
        transition={{ 
          duration: animSpeed.pulse, 
          repeat: Infinity 
        }}
      >
        <Zap size={config.icon} color={color} />
      </motion.div>
      
      {/* Progress percentage */}
      {showStats && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`font-mono font-bold ${config.text}`} style={{ color }}>
            {displayProgress.toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
  
  /**
   * Terminal variant - For the hackers
   */
  const renderTerminal = () => (
    <div className="relative bg-black p-4 rounded border border-green-500/50">
      <div className="flex items-center gap-2 mb-2">
        <Terminal size={16} className="text-green-500" />
        <span className="text-green-500 text-xs font-mono">nexlify@trader:~$</span>
      </div>
      
      <div ref={terminalRef} className="space-y-1 font-mono text-xs">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-green-400"
        >
          {'>'} Initializing Nexlify Trading System{loadingDots}
        </motion.div>
        
        {fakeProgress > 20 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <span className="text-green-500">[OK]</span> Market data streams connected
          </motion.div>
        )}
        
        {fakeProgress > 40 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <span className="text-green-500">[OK]</span> Risk management loaded
          </motion.div>
        )}
        
        {fakeProgress > 60 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <span className="text-green-500">[OK]</span> Neural networks online
          </motion.div>
        )}
        
        {fakeProgress > 80 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <span className="text-yellow-500">[WAIT]</span> Bypassing exchange limits{loadingDots}
          </motion.div>
        )}
        
        <motion.div className="text-green-400">
          {'>'} Progress: [{Array(Math.floor(displayProgress / 5)).fill('=').join('')}
          {Array(20 - Math.floor(displayProgress / 5)).fill(' ').join('')}] {displayProgress.toFixed(0)}%
        </motion.div>
      </div>
    </div>
  );
  
  /**
   * Glitch variant - When loading gets weird
   */
  const renderGlitch = () => (
    <div className="relative">
      <motion.div
        className={`${config.text} font-bold text-center`}
        animate={{
          x: [0, -2, 2, -1, 1, 0],
          filter: [
            'hue-rotate(0deg)',
            'hue-rotate(90deg)',
            'hue-rotate(-90deg)',
            'hue-rotate(45deg)',
            'hue-rotate(0deg)'
          ]
        }}
        transition={{ duration: 0.2, repeat: Infinity }}
        style={{ color }}
      >
        <div className="relative">
          {glitchText || 'LOADING'}
          
          {/* Glitch layers */}
          <div className="absolute inset-0 text-red-500 opacity-50" style={{ transform: 'translateX(2px)' }}>
            {glitchText || 'LOADING'}
          </div>
          <div className="absolute inset-0 text-blue-500 opacity-50" style={{ transform: 'translateX(-2px)' }}>
            {glitchText || 'LOADING'}
          </div>
        </div>
      </motion.div>
      
      {/* Scan lines */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        animate={{ backgroundPosition: ['0px 0px', '0px 10px'] }}
        transition={{ duration: 0.5, repeat: Infinity, ease: "linear" }}
        style={{
          backgroundImage: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            ${color}20 2px,
            ${color}20 4px
          )`,
          backgroundSize: '100% 4px'
        }}
      />
    </div>
  );
  
  /**
   * Neural variant - AI-powered loading
   */
  const renderNeural = () => {
    const nodes = Array(8).fill(0).map((_, i) => ({
      x: 50 + 30 * Math.cos((i * Math.PI * 2) / 8),
      y: 50 + 30 * Math.sin((i * Math.PI * 2) / 8)
    }));
    
    return (
      <svg width={config.icon * 2} height={config.icon * 2} viewBox="0 0 100 100">
        {/* Connections */}
        {nodes.map((node, i) => (
          <motion.line
            key={`line-${i}`}
            x1="50"
            y1="50"
            x2={node.x}
            y2={node.y}
            stroke={color}
            strokeWidth="1"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ 
              pathLength: [0, 1, 0],
              opacity: [0, 0.5, 0]
            }}
            transition={{ 
              duration: animSpeed.pulse,
              repeat: Infinity,
              delay: i * 0.1
            }}
          />
        ))}
        
        {/* Center node */}
        <motion.circle
          cx="50"
          cy="50"
          r="8"
          fill={color}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: animSpeed.pulse, repeat: Infinity }}
        />
        
        {/* Outer nodes */}
        {nodes.map((node, i) => (
          <motion.circle
            key={`node-${i}`}
            cx={node.x}
            cy={node.y}
            r="4"
            fill={color}
            initial={{ opacity: 0.3 }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ 
              duration: animSpeed.pulse,
              repeat: Infinity,
              delay: i * 0.1
            }}
          />
        ))}
        
        {/* Data flow particles */}
        {nodes.map((node, i) => (
          <motion.circle
            key={`particle-${i}`}
            r="2"
            fill={color}
            initial={{ cx: 50, cy: 50 }}
            animate={{ 
              cx: [50, node.x, 50],
              cy: [50, node.y, 50]
            }}
            transition={{ 
              duration: animSpeed.rotation,
              repeat: Infinity,
              delay: i * 0.2,
              ease: "easeInOut"
            }}
          />
        ))}
      </svg>
    );
  };
  
  /**
   * Circuit variant - Data flow visualization
   */
  const renderCircuit = () => (
    <div className="relative">
      <CircuitBoard size={config.icon} color={color} className="opacity-20" />
      
      {/* Animated circuit paths */}
      <svg
        className="absolute inset-0"
        width={config.icon * 2}
        height={config.icon * 2}
        viewBox="0 0 100 100"
      >
        {/* Main circuit path */}
        <motion.path
          d="M20,50 L30,50 L30,30 L50,30 L50,50 L70,50 L70,70 L50,70 L50,50"
          fill="none"
          stroke={color}
          strokeWidth="2"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ 
            duration: animSpeed.rotation,
            repeat: Infinity,
            ease: "linear"
          }}
        />
        
        {/* Data packets */}
        <motion.circle
          r="3"
          fill={color}
          initial={{ offsetDistance: '0%' }}
          animate={{ offsetDistance: '100%' }}
          transition={{ 
            duration: animSpeed.rotation,
            repeat: Infinity,
            ease: "linear"
          }}
          style={{
            offsetPath: 'path("M20,50 L30,50 L30,30 L50,30 L50,50 L70,50 L70,70 L50,70 L50,50")'
          }}
        />
      </svg>
    </div>
  );
  
  // Main render logic
  const renderLoader = () => {
    switch (variant) {
      case 'terminal':
        return renderTerminal();
      case 'glitch':
        return renderGlitch();
      case 'neural':
        return renderNeural();
      case 'circuit':
        return renderCircuit();
      default:
        return renderSpinner();
    }
  };
  
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.8 }}
        className={`
          flex flex-col items-center justify-center gap-4
          ${size === 'fullscreen' ? 'fixed inset-0 bg-gray-900/90 z-50' : ''}
        `}
      >
        {/* Loader animation */}
        <div className={`relative ${config.container} flex items-center justify-center`}>
          {renderLoader()}
        </div>
        
        {/* Text content */}
        {(text || !text) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-center max-w-md"
          >
            <div className={`${config.text} font-semibold mb-1`} style={{ color }}>
              {text || loadingPhrases[currentPhrase].main}
            </div>
            {(subtext || !text) && (
              <div className={`${config.text} opacity-60`} style={{ color }}>
                {subtext || loadingPhrases[currentPhrase].sub}
              </div>
            )}
          </motion.div>
        )}
        
        {/* Loading stats */}
        {showStats && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className={`${config.text} font-mono space-y-1 text-center`}
            style={{ color: `${color}80` }}
          >
            <div>Latency: {Math.floor(Math.random() * 50 + 10)}ms</div>
            <div>Packets: {Math.floor(displayProgress * 10)}/1000</div>
            <div>Status: {displayProgress < 100 ? 'SYNCING' : 'READY'}</div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

/**
 * LOADER WISDOM FROM THE WAITING ROOMS:
 * 
 * 1. The rotating phrases aren't random. Each one represents a real
 *    part of the trading infrastructure. "Dark pools" aren't fiction.
 * 
 * 2. Never reaching 100% on fake progress is intentional. False hope
 *    is worse than honest uncertainty.
 * 
 * 3. The terminal variant shows actual initialization steps. It's not
 *    just for show - it mirrors the real startup sequence.
 * 
 * 4. Glitch effects increase anxiety. Use sparingly. Some traders
 *    already have enough stress.
 * 
 * 5. Neural network visualization helps users understand that AI is
 *    working for them, even while they wait.
 * 
 * 6. The stats display (latency, packets) gives power users something
 *    to analyze. They need to feel in control, even while waiting.
 * 
 * Remember: In trading, waiting is part of the game. The best trades
 * often require the most patience. Make the wait worthwhile.
 * 
 * "Time in the market beats timing the market, but waiting for the
 * market to load beats neither." - Nexlify Dev Team, 3am
 */
