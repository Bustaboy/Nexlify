// frontend/src/hooks/useWebSocket.ts

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuthStore } from '@stores/authStore';
import { useTradingStore } from '@stores/tradingStore';
import { playSound } from '@lib/sounds';
import { createWebSocketConnection } from '@lib/api';
import toast from 'react-hot-toast';

// The neural link to the backend - where data flows like memories through a braindance
// Sometimes it's smooth, sometimes it glitches, but it's always there, pulsing with life

interface WebSocketState {
  isConnected: boolean;
  isReconnecting: boolean;
  latency: number;
  reconnectAttempts: number;
  lastHeartbeat: Date | null;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  heartbeatInterval?: number;
  reconnectOnFocus?: boolean;
}

export function useWebSocket(
  enabled: boolean = true,
  options: UseWebSocketOptions = {}
) {
  const {
    autoConnect = true,
    heartbeatInterval = 30000, // 30 seconds - gotta keep that pulse going
    reconnectOnFocus = true
  } = options;

  const { token } = useAuthStore();
  const { 
    connectWebSocket, 
    disconnectWebSocket,
    subscribeToSymbol,
    activeSymbols 
  } = useTradingStore();

  const socketRef = useRef<Socket | null>(null);
  const heartbeatTimerRef = useRef<NodeJS.Timeout>();
  const reconnectTimerRef = useRef<NodeJS.Timeout>();

  // Connection state - the vital signs of our neural link
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isReconnecting: false,
    latency: 0,
    reconnectAttempts: 0,
    lastHeartbeat: null
  });

  // Heartbeat - checking if the connection's still breathing
  const sendHeartbeat = useCallback(() => {
    if (socketRef.current?.connected) {
      const startTime = Date.now();
      
      socketRef.current.emit('ping', {}, (response: any) => {
        const latency = Date.now() - startTime;
        setState(prev => ({
          ...prev,
          latency,
          lastHeartbeat: new Date()
        }));
        
        // If latency's getting bad, warn the user
        if (latency > 1000) {
          console.warn(`High latency detected: ${latency}ms - Night City's net is choppy tonight`);
        }
      });
    }
  }, []);

  // Connect to the matrix - jack in and ride the data streams
  const connect = useCallback(() => {
    if (!token || !enabled || socketRef.current?.connected) return;

    console.log('🔌 Jacking into the trading matrix...');
    
    const { url, options } = createWebSocketConnection(token);
    const socket = io(url, options);
    
    socketRef.current = socket;

    // Connection established - we're in
    socket.on('connect', () => {
      console.log('⚡ Neural link established - ID:', socket.id);
      
      setState(prev => ({
        ...prev,
        isConnected: true,
        isReconnecting: false,
        reconnectAttempts: 0
      }));
      
      // Clear any reconnect timers
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      
      // Start heartbeat monitoring
      heartbeatTimerRef.current = setInterval(sendHeartbeat, heartbeatInterval);
      
      // Resubscribe to active symbols - can't miss those price movements
      activeSymbols.forEach(symbol => {
        socket.emit('subscribe_market', { symbols: [symbol] });
      });
      
      // Let the trading store know we're connected
      connectWebSocket(token);
      
      // A little audio feedback never hurt
      playSound('startup', { volume: 0.3 });
      
      // Only show toast on reconnect, not initial connect
      if (state.reconnectAttempts > 0) {
        toast.success('Neural link restored', {
          icon: '🔗',
          duration: 2000
        });
      }
    });

    // Connection lost - we've flatlined
    socket.on('disconnect', (reason) => {
      console.warn('💔 Neural link severed:', reason);
      
      setState(prev => ({
        ...prev,
        isConnected: false,
        isReconnecting: true
      }));
      
      // Clear heartbeat
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }
      
      playSound('error', { volume: 0.2 });
      
      // Different messages for different disconnection reasons
      // Because context matters when you're drowning in the data stream
      switch (reason) {
        case 'io server disconnect':
          toast.error('Server terminated connection - check your creds', {
            icon: '🚫',
            duration: 5000
          });
          break;
        case 'ping timeout':
          toast.error('Connection timeout - Night City\'s net is glitching', {
            icon: '⏱️'
          });
          break;
        case 'transport close':
          toast.error('Transport failure - switching to backup routes', {
            icon: '🔌'
          });
          break;
        default:
          if (reason !== 'io client disconnect') {
            toast.error(`Connection lost: ${reason}`, {
              icon: '📡'
            });
          }
      }
      
      disconnectWebSocket();
    });

    // Reconnection attempts - never give up, never surrender
    socket.on('reconnect_attempt', (attemptNumber) => {
      setState(prev => ({
        ...prev,
        reconnectAttempts: attemptNumber
      }));
      
      // Give user feedback on reconnection attempts
      if (attemptNumber % 3 === 0) {
        console.log(`🔄 Reconnection attempt ${attemptNumber} - still fighting for that link...`);
      }
    });

    // Reconnection failed - time to check your chrome
    socket.on('reconnect_failed', () => {
      toast.error('Failed to restore neural link - check your connection', {
        icon: '❌',
        duration: 0 // Keep it visible until dismissed
      });
      
      setState(prev => ({
        ...prev,
        isReconnecting: false
      }));
    });

    // Handle errors with grace - or at least with style
    socket.on('error', (error) => {
      console.error('Socket error:', error);
      
      // Parse error for user-friendly message
      let message = 'Neural link error';
      
      if (error.type === 'TransportError') {
        message = 'Transport protocol failure - trying alternate routes';
      } else if (error.type === 'AuthError') {
        message = 'Authentication failed - your creds expired';
      }
      
      toast.error(message, {
        icon: '⚠️',
        duration: 4000
      });
    });

    // Custom events from our backend
    socket.on('rate_limited', (data) => {
      const waitTime = data.retry_after || 60;
      toast.error(`Rate limited - cooldown ${waitTime}s`, {
        icon: '🛑',
        duration: 5000
      });
    });

    socket.on('market_halted', (data) => {
      toast.error(`Market halted: ${data.symbol} - ${data.reason}`, {
        icon: '🚨',
        duration: 10000
      });
      playSound('alert_high');
    });

    socket.on('system_maintenance', (data) => {
      const maintenanceTime = new Date(data.scheduled_time);
      toast.error(
        `System maintenance scheduled: ${maintenanceTime.toLocaleTimeString()}`,
        {
          icon: '🔧',
          duration: 0 // Keep visible
        }
      );
    });

  }, [token, enabled, activeSymbols, state.reconnectAttempts, heartbeatInterval, sendHeartbeat, connectWebSocket, disconnectWebSocket]);

  // Disconnect - unplug from the matrix
  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting from trading matrix...');
    
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
    }
    
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    
    setState({
      isConnected: false,
      isReconnecting: false,
      latency: 0,
      reconnectAttempts: 0,
      lastHeartbeat: null
    });
    
    disconnectWebSocket();
  }, [disconnectWebSocket]);

  // Auto-connect when token changes or component mounts
  useEffect(() => {
    if (enabled && token && autoConnect) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [enabled, token, autoConnect]); // Intentionally not including connect/disconnect to avoid loops

  // Reconnect on window focus - because coming back from AFK should be seamless
  useEffect(() => {
    if (!reconnectOnFocus) return;
    
    const handleFocus = () => {
      if (enabled && token && !socketRef.current?.connected) {
        console.log('🔄 Window focused - attempting reconnection...');
        connect();
      }
    };
    
    window.addEventListener('focus', handleFocus);
    
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, [enabled, token, reconnectOnFocus, connect]);

  // Monitor connection health
  useEffect(() => {
    if (!state.isConnected || !state.lastHeartbeat) return;
    
    const checkHealthTimer = setInterval(() => {
      const now = Date.now();
      const lastBeat = state.lastHeartbeat?.getTime() || 0;
      const timeSinceLastBeat = now - lastBeat;
      
      // If we haven't heard a heartbeat in 2 intervals, something's wrong
      if (timeSinceLastBeat > heartbeatInterval * 2) {
        console.warn('❤️‍🩹 Heartbeat missed - connection might be dead');
        
        // Force reconnect
        if (socketRef.current) {
          socketRef.current.disconnect();
          setTimeout(() => connect(), 1000);
        }
      }
    }, heartbeatInterval);
    
    return () => clearInterval(checkHealthTimer);
  }, [state.isConnected, state.lastHeartbeat, heartbeatInterval, connect]);

  // Expose methods for manual control
  const manualReconnect = useCallback(() => {
    disconnect();
    setTimeout(() => connect(), 500);
  }, [connect, disconnect]);

  // Subscribe to new symbol
  const subscribe = useCallback((symbol: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('subscribe_market', { symbols: [symbol] });
      subscribeToSymbol(symbol);
    }
  }, [subscribeToSymbol]);

  // Send custom event
  const emit = useCallback((event: string, data: any, callback?: Function) => {
    if (socketRef.current?.connected) {
      if (callback) {
        socketRef.current.emit(event, data, callback);
      } else {
        socketRef.current.emit(event, data);
      }
    } else {
      console.warn(`Cannot emit ${event} - not connected`);
    }
  }, []);

  return {
    // State
    isConnected: state.isConnected,
    isReconnecting: state.isReconnecting,
    latency: state.latency,
    reconnectAttempts: state.reconnectAttempts,
    
    // Actions
    connect: manualReconnect,
    disconnect,
    subscribe,
    emit,
    
    // Raw socket for advanced usage
    socket: socketRef.current
  };
}

// Specialized hook for market data subscriptions
export function useMarketData(symbols: string[]) {
  const { subscribe } = useWebSocket();
  const { marketData } = useTradingStore();
  
  useEffect(() => {
    // Subscribe to all requested symbols
    symbols.forEach(symbol => subscribe(symbol));
  }, [symbols, subscribe]);
  
  // Return market data for requested symbols
  return symbols.reduce((acc, symbol) => {
    acc[symbol] = marketData.get(symbol) || null;
    return acc;
  }, {} as Record<string, any>);
}

// Hook for connection status in components
export function useConnectionStatus() {
  const { isConnected, latency, isReconnecting } = useWebSocket();
  
  const status = isConnected 
    ? 'connected' 
    : isReconnecting 
    ? 'reconnecting' 
    : 'disconnected';
    
  const color = isConnected
    ? 'text-neon-green'
    : isReconnecting
    ? 'text-neon-yellow'
    : 'text-neon-red';
    
  const icon = isConnected
    ? '🟢'
    : isReconnecting
    ? '🟡'
    : '🔴';
    
  return {
    status,
    color,
    icon,
    latency,
    isHealthy: isConnected && latency < 500
  };
}
