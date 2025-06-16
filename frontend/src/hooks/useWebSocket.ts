// frontend/src/hooks/useWebSocket.ts
/**
 * High-Performance WebSocket Hook - The Neural Link to Market Data
 * Sub-second latency optimized for algorithmic trading
 * 
 * This isn't just a connection - it's your lifeline to the market.
 * Every millisecond counts when algos are hunting the same opportunities.
 * Been through enough market crashes to know: lag kills profit.
 */

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuthStore } from '@stores/authStore';
import { useTradingStore } from '@stores/tradingStore';
import { playSound } from '@lib/sounds';
import toast from 'react-hot-toast';

// Types - the data structures that define our reality
interface WebSocketState {
  isConnected: boolean;
  isReconnecting: boolean;
  latency: number;
  reconnectAttempts: number;
  lastHeartbeat: Date | null;
  messageQueue: number;
  bytesPerSecond: number;
}

interface PerformanceMetrics {
  avgLatency: number;
  maxLatency: number;
  minLatency: number;
  messagesPerSecond: number;
  reconnectionCount: number;
  uptime: number;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  heartbeatInterval?: number;
  reconnectOnFocus?: boolean;
  maxReconnectAttempts?: number;
  performanceMode?: boolean; // Ultra-low latency mode
  binaryProtocol?: boolean;  // For maximum speed
}

// Performance constants - tuned for speed demons
const PERF_CONSTANTS = {
  HEARTBEAT_INTERVAL: 15000,        // 15s - frequent enough to catch issues
  MAX_RECONNECT_ATTEMPTS: 10,       // Never give up, that's money on the line
  RECONNECT_BASE_DELAY: 100,        // Start fast
  RECONNECT_MAX_DELAY: 5000,        // But don't spam the server
  LATENCY_BUFFER_SIZE: 100,         // Track latency history
  PERF_SAMPLE_RATE: 1000,          // Sample every second
  HIGH_LATENCY_THRESHOLD: 50,       // Warn above 50ms
  CRITICAL_LATENCY_THRESHOLD: 100   // Panic above 100ms
} as const;

/**
 * The main hook - your gateway to real-time market data
 * Built for speed, designed for profit
 */
export function useWebSocket(
  enabled: boolean = true,
  options: UseWebSocketOptions = {}
) {
  const {
    autoConnect = true,
    heartbeatInterval = PERF_CONSTANTS.HEARTBEAT_INTERVAL,
    reconnectOnFocus = true,
    maxReconnectAttempts = PERF_CONSTANTS.MAX_RECONNECT_ATTEMPTS,
    performanceMode = true,
    binaryProtocol = false
  } = options;

  // Store connections
  const { token, userId } = useAuthStore();
  const { 
    updateMarketData,
    updatePositions,
    addSignal,
    setConnectionStatus 
  } = useTradingStore();

  // Refs for persistent state
  const socketRef = useRef<Socket | null>(null);
  const heartbeatTimerRef = useRef<NodeJS.Timeout>();
  const reconnectTimerRef = useRef<NodeJS.Timeout>();
  const latencyBufferRef = useRef<number[]>([]);
  const messageCountRef = useRef(0);
  const bytesCountRef = useRef(0);
  const startTimeRef = useRef<number>(Date.now());
  const lastMessageTimeRef = useRef<number>(Date.now());

  // State management
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isReconnecting: false,
    latency: 0,
    reconnectAttempts: 0,
    lastHeartbeat: null,
    messageQueue: 0,
    bytesPerSecond: 0
  });

  // Performance metrics calculation
  const performanceMetrics = useMemo<PerformanceMetrics>(() => {
    const latencies = latencyBufferRef.current;
    const now = Date.now();
    const uptime = now - startTimeRef.current;
    
    return {
      avgLatency: latencies.length > 0 
        ? latencies.reduce((a, b) => a + b, 0) / latencies.length 
        : 0,
      maxLatency: latencies.length > 0 ? Math.max(...latencies) : 0,
      minLatency: latencies.length > 0 ? Math.min(...latencies) : 0,
      messagesPerSecond: messageCountRef.current / (uptime / 1000),
      reconnectionCount: state.reconnectAttempts,
      uptime: uptime / 1000
    };
  }, [state.latency, state.reconnectAttempts]);

  // Latency tracking - every millisecond matters
  const trackLatency = useCallback((latency: number) => {
    const buffer = latencyBufferRef.current;
    buffer.push(latency);
    
    // Keep buffer size manageable
    if (buffer.length > PERF_CONSTANTS.LATENCY_BUFFER_SIZE) {
      buffer.shift();
    }

    // Alert on high latency - this is where profits die
    if (latency > PERF_CONSTANTS.CRITICAL_LATENCY_THRESHOLD) {
      console.error(`🚨 CRITICAL LATENCY: ${latency}ms - Market opportunities slipping away!`);
      toast.error(`High latency detected: ${latency}ms`, {
        id: 'high-latency',
        duration: 2000
      });
    } else if (latency > PERF_CONSTANTS.HIGH_LATENCY_THRESHOLD) {
      console.warn(`⚠️ High latency: ${latency}ms - Connection getting sluggish`);
    }
  }, []);

  // Heartbeat - the pulse that keeps us alive
  const sendHeartbeat = useCallback(() => {
    if (socketRef.current?.connected) {
      const startTime = performance.now();
      
      socketRef.current.emit('ping', { timestamp: startTime }, (response: any) => {
        const latency = performance.now() - startTime;
        
        trackLatency(latency);
        setState(prev => ({
          ...prev,
          latency,
          lastHeartbeat: new Date()
        }));
      });
    }
  }, [trackLatency]);

  // Connection logic - jack into the matrix
  const connect = useCallback(() => {
    if (!token || !enabled || socketRef.current?.connected) return;

    console.log('🔌 Jacking into the trading matrix... Latency is life, choom.');
    
    setState(prev => ({ ...prev, isReconnecting: true }));

    // Performance-optimized socket configuration
    const socketOptions = {
      auth: { token },
      timeout: 5000,
      forceNew: true,
      transports: performanceMode ? ['websocket'] : ['websocket', 'polling'],
      upgrade: false, // Stick to websocket for consistency
      rememberUpgrade: false,
      ...(binaryProtocol && { parser: require('socket.io-msgpack-parser') })
    };

    const socket = io(
      process.env.NODE_ENV === 'development' 
        ? 'http://localhost:8000'
        : window.location.origin,
      socketOptions
    );

    socketRef.current = socket;

    // Connection events - the moments that define success or failure
    socket.on('connect', () => {
      console.log('✅ Neural link established - We\'re in the matrix');
      
      setState(prev => ({
        ...prev,
        isConnected: true,
        isReconnecting: false,
        reconnectAttempts: 0
      }));

      setConnectionStatus(true);
      
      // Start the heartbeat - keep that pulse strong
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }
      heartbeatTimerRef.current = setInterval(sendHeartbeat, heartbeatInterval);

      playSound('connected');
      toast.success('Trading matrix online', { id: 'connection' });
    });

    socket.on('disconnect', (reason) => {
      console.warn(`🔌 Neural link severed: ${reason}`);
      
      setState(prev => ({
        ...prev,
        isConnected: false,
        isReconnecting: reason !== 'io client disconnect'
      }));

      setConnectionStatus(false);
      
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }

      if (reason !== 'io client disconnect') {
        scheduleReconnect();
      }
    });

    // Market data handlers - where the money flows
    socket.on('market_update', (data: any) => {
      messageCountRef.current++;
      bytesCountRef.current += JSON.stringify(data).length;
      lastMessageTimeRef.current = Date.now();
      
      updateMarketData(data.symbol, {
        price: data.price,
        change: data.change,
        volume: data.volume,
        timestamp: new Date(data.timestamp)
      });
    });

    // Position updates - track every eddy
    socket.on('position_update', (data: any) => {
      updatePositions(data);
      playSound('position_update');
    });

    // Trading signals - the AI's whispers
    socket.on('trading_signal', (signal: any) => {
      addSignal(signal);
      
      // High priority signals get special treatment
      if (signal.confidence > 0.8) {
        playSound('high_confidence_signal');
        toast.success(`High confidence signal: ${signal.action} ${signal.symbol}`, {
          duration: 5000
        });
      }
    });

    // Error handling - when the matrix glitches
    socket.on('connect_error', (error) => {
      console.error('🚨 Connection error:', error.message);
      setState(prev => ({
        ...prev,
        isConnected: false,
        isReconnecting: true
      }));
    });

    socket.on('error', (error) => {
      console.error('🚨 Socket error:', error);
      toast.error(`Connection error: ${error.message}`);
    });

  }, [token, enabled, performanceMode, binaryProtocol, sendHeartbeat, heartbeatInterval]);

  // Reconnection logic - never give up, that's money on the line
  const scheduleReconnect = useCallback(() => {
    if (state.reconnectAttempts >= maxReconnectAttempts) {
      console.error('🚨 Max reconnection attempts reached. Manual intervention required.');
      toast.error('Connection failed. Check your network and restart the application.');
      return;
    }

    const delay = Math.min(
      PERF_CONSTANTS.RECONNECT_BASE_DELAY * Math.pow(2, state.reconnectAttempts),
      PERF_CONSTANTS.RECONNECT_MAX_DELAY
    );

    console.log(`🔄 Reconnecting in ${delay}ms... Attempt ${state.reconnectAttempts + 1}/${maxReconnectAttempts}`);
    
    setState(prev => ({
      ...prev,
      reconnectAttempts: prev.reconnectAttempts + 1
    }));

    reconnectTimerRef.current = setTimeout(() => {
      if (!state.isConnected) {
        connect();
      }
    }, delay);

  }, [state.reconnectAttempts, state.isConnected, maxReconnectAttempts, connect]);

  // Disconnect - clean shutdown
  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting from trading matrix...');
    
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
    }
    
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      isReconnecting: false
    }));

    setConnectionStatus(false);
  }, []);

  // Subscription management - tell the matrix what we want to hear
  const subscribe = useCallback((symbols: string[]) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('subscribe_market', { symbols });
      console.log(`📡 Subscribed to market data: ${symbols.join(', ')}`);
    }
  }, []);

  const unsubscribe = useCallback((symbols: string[]) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('unsubscribe_market', { symbols });
      console.log(`📡 Unsubscribed from market data: ${symbols.join(', ')}`);
    }
  }, []);

  // Manual reconnect - when you need to force the connection
  const forceReconnect = useCallback(() => {
    console.log('🔄 Forcing reconnection...');
    disconnect();
    setTimeout(connect, 1000);
  }, [disconnect, connect]);

  // Effects
  useEffect(() => {
    if (enabled && autoConnect && token) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, autoConnect, token]);

  // Reconnect on window focus - catch missed opportunities
  useEffect(() => {
    if (!reconnectOnFocus) return;

    const handleFocus = () => {
      if (!state.isConnected && enabled) {
        console.log('🔄 Window focused - attempting reconnection');
        connect();
      }
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [reconnectOnFocus, state.isConnected, enabled, connect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
    };
  }, []);

  return {
    // Connection state
    ...state,
    
    // Performance metrics
    performanceMetrics,
    
    // Connection methods
    connect,
    disconnect,
    forceReconnect,
    
    // Subscription methods
    subscribe,
    unsubscribe,
    
    // Health check
    isHealthy: state.isConnected && state.latency < PERF_CONSTANTS.HIGH_LATENCY_THRESHOLD,
    
    // Direct socket access for advanced use cases
    socket: socketRef.current
  };
}

/**
 * Connection status hook - lightweight version for status displays
 */
export function useConnectionStatus() {
  const { isConnected, latency, performanceMetrics } = useWebSocket(true, { 
    performanceMode: true 
  });
  
  return {
    isConnected,
    latency,
    isHealthy: isConnected && latency < PERF_CONSTANTS.HIGH_LATENCY_THRESHOLD,
    performanceMetrics
  };
}
