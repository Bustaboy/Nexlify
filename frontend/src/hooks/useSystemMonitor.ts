// frontend/src/hooks/useSystemMonitor.ts

import { useEffect, useState, useCallback, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@lib/api';
import { playSound } from '@lib/sounds';
import { isElectron } from '@lib/utils';
import toast from 'react-hot-toast';

// System monitoring - keeping tabs on your chrome's vital signs
// Because in Night City, ignoring warning signs gets you flatlined
// I've seen too many traders burn out their rigs chasing that perfect trade

interface SystemStats {
  cpu: {
    usage: number;
    temperature?: number;
    cores: number;
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  gpu?: {
    usage: number;
    memory: number;
    temperature?: number;
  };
  network: {
    latency: number;
    bandwidth: number;
    packetLoss: number;
  };
  trading: {
    openPositions: number;
    ordersPerMinute: number;
    apiCallsRemaining: number;
    wsMessagesPerSecond: number;
  };
}

interface HealthThresholds {
  cpu: { warning: number; critical: number };
  memory: { warning: number; critical: number };
  latency: { warning: number; critical: number };
  temperature: { warning: number; critical: number };
}

// Default thresholds - learned these the hard way
const DEFAULT_THRESHOLDS: HealthThresholds = {
  cpu: { warning: 80, critical: 95 },
  memory: { warning: 85, critical: 95 },
  latency: { warning: 500, critical: 1000 },
  temperature: { warning: 80, critical: 90 }
};

export function useSystemMonitor(options: {
  interval?: number;
  thresholds?: Partial<HealthThresholds>;
  enableAlerts?: boolean;
} = {}) {
  const {
    interval = 5000, // Check every 5 seconds - enough to catch problems, not enough to cause them
    thresholds = DEFAULT_THRESHOLDS,
    enableAlerts = true
  } = options;

  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'warning' | 'critical'>('healthy');
  const [alerts, setAlerts] = useState<string[]>([]);
  
  // Track alert cooldowns - nobody likes spam when their rig's melting
  const alertCooldowns = useRef<Map<string, number>>(new Map());
  const lastTempWarning = useRef<number>(0);

  // Fetch system stats from backend
  const { data: backendStats } = useQuery({
    queryKey: ['system-stats'],
    queryFn: async () => {
      const response = await apiClient.get('/system/stats');
      return response.data;
    },
    refetchInterval: interval,
    enabled: true
  });

  // Get Electron system stats if available
  const fetchElectronStats = useCallback(async () => {
    if (!isElectron()) return null;
    
    try {
      const stats = await window.nexlify.system.getStats();
      return stats;
    } catch (error) {
      console.error('Failed to get Electron stats:', error);
      return null;
    }
  }, []);

  // Check system health and fire alerts
  const checkSystemHealth = useCallback((stats: SystemStats) => {
    const newAlerts: string[] = [];
    let overallStatus: 'healthy' | 'warning' | 'critical' = 'healthy';
    
    // CPU check - when your neural processor's cooking
    if (stats.cpu.usage > thresholds.cpu.critical) {
      overallStatus = 'critical';
      newAlerts.push(`CPU critical: ${stats.cpu.usage.toFixed(1)}%`);
      
      if (shouldAlert('cpu-critical')) {
        playSound('alert_high', { volume: 0.7 });
        toast.error('CPU overload - trades might lag', {
          icon: '🔥',
          duration: 5000
        });
      }
    } else if (stats.cpu.usage > thresholds.cpu.warning) {
      if (overallStatus === 'healthy') overallStatus = 'warning';
      newAlerts.push(`CPU high: ${stats.cpu.usage.toFixed(1)}%`);
    }
    
    // Memory check - when your RAM's getting full of market data
    const memoryPercent = (stats.memory.used / stats.memory.total) * 100;
    if (memoryPercent > thresholds.memory.critical) {
      overallStatus = 'critical';
      newAlerts.push(`Memory critical: ${memoryPercent.toFixed(1)}%`);
      
      if (shouldAlert('memory-critical')) {
        playSound('alert_high', { volume: 0.7 });
        toast.error('Memory critical - system might crash', {
          icon: '💾',
          duration: 5000
        });
      }
    } else if (memoryPercent > thresholds.memory.warning) {
      if (overallStatus === 'healthy') overallStatus = 'warning';
      newAlerts.push(`Memory high: ${memoryPercent.toFixed(1)}%`);
    }
    
    // Temperature check - Night City's hot, but your rig shouldn't be
    if (stats.cpu.temperature) {
      if (stats.cpu.temperature > thresholds.temperature.critical) {
        overallStatus = 'critical';
        newAlerts.push(`Temperature critical: ${stats.cpu.temperature}°C`);
        
        // Temperature warnings are special - they mean hardware damage
        const now = Date.now();
        if (now - lastTempWarning.current > 60000) { // Once per minute max
          playSound('alert_high', { volume: 0.9 });
          toast.error('SYSTEM OVERHEATING - Throttling imminent', {
            icon: '🌡️',
            duration: 0, // Keep visible until dismissed
            style: {
              background: '#ff0000',
              color: '#ffffff'
            }
          });
          
          // If in Electron, offer to reduce performance
          if (isElectron()) {
            window.nexlify.notification.show({
              title: 'Critical Temperature Warning',
              body: `System temperature at ${stats.cpu.temperature}°C. Reduce trading activity to prevent damage.`,
              urgency: 'critical'
            });
          }
          
          lastTempWarning.current = now;
        }
      } else if (stats.cpu.temperature > thresholds.temperature.warning) {
        if (overallStatus === 'healthy') overallStatus = 'warning';
        newAlerts.push(`Temperature high: ${stats.cpu.temperature}°C`);
      }
    }
    
    // Network latency - when the net's running slow
    if (stats.network.latency > thresholds.latency.critical) {
      overallStatus = 'critical';
      newAlerts.push(`Latency critical: ${stats.network.latency}ms`);
      
      if (shouldAlert('latency-critical')) {
        toast.error('Network latency critical - trades may fail', {
          icon: '📡',
          duration: 4000
        });
      }
    } else if (stats.network.latency > thresholds.latency.warning) {
      if (overallStatus === 'healthy') overallStatus = 'warning';
      newAlerts.push(`Latency high: ${stats.network.latency}ms`);
    }
    
    // Trading-specific checks - the stuff that really matters
    if (stats.trading.apiCallsRemaining < 10) {
      newAlerts.push(`API limit approaching: ${stats.trading.apiCallsRemaining} calls left`);
      
      if (shouldAlert('api-limit') && stats.trading.apiCallsRemaining < 5) {
        playSound('alert_medium');
        toast.error('API rate limit approaching - slow down', {
          icon: '🚦',
          duration: 5000
        });
      }
    }
    
    setHealthStatus(overallStatus);
    setAlerts(newAlerts);
    
    // Personal touch - sometimes you need encouragement
    if (overallStatus === 'critical' && Math.random() < 0.1) {
      setTimeout(() => {
        toast('Hang in there, choom. Every legend has rough nights.', {
          icon: '💜',
          duration: 3000
        });
      }, 10000);
    }
  }, [thresholds, enableAlerts]);

  // Alert cooldown manager - prevents spam
  const shouldAlert = useCallback((alertType: string): boolean => {
    if (!enableAlerts) return false;
    
    const now = Date.now();
    const lastAlert = alertCooldowns.current.get(alertType) || 0;
    const cooldownPeriod = 300000; // 5 minutes
    
    if (now - lastAlert > cooldownPeriod) {
      alertCooldowns.current.set(alertType, now);
      return true;
    }
    
    return false;
  }, [enableAlerts]);

  // Combine backend and electron stats
  useEffect(() => {
    const updateStats = async () => {
      const electronStats = await fetchElectronStats();
      
      // Merge stats from different sources
      const combinedStats: SystemStats = {
        cpu: electronStats?.cpu || {
          usage: backendStats?.cpu_usage || 0,
          cores: backendStats?.cpu_cores || 1,
          temperature: electronStats?.cpu?.temperature
        },
        memory: electronStats?.memory || {
          used: backendStats?.memory_used || 0,
          total: backendStats?.memory_total || 1,
          percentage: backendStats?.memory_percent || 0
        },
        gpu: electronStats?.gpu,
        network: {
          latency: backendStats?.api_latency || 0,
          bandwidth: backendStats?.bandwidth || 0,
          packetLoss: backendStats?.packet_loss || 0
        },
        trading: {
          openPositions: backendStats?.open_positions || 0,
          ordersPerMinute: backendStats?.orders_per_minute || 0,
          apiCallsRemaining: backendStats?.api_calls_remaining || 1000,
          wsMessagesPerSecond: backendStats?.ws_messages_per_second || 0
        }
      };
      
      setSystemStats(combinedStats);
      checkSystemHealth(combinedStats);
    };
    
    if (backendStats) {
      updateStats();
    }
  }, [backendStats, fetchElectronStats, checkSystemHealth]);

  // Performance optimization suggestions
  const getOptimizationTips = useCallback((): string[] => {
    const tips: string[] = [];
    
    if (!systemStats) return tips;
    
    // Been there, done that - here's what actually helps
    if (systemStats.cpu.usage > 70) {
      tips.push('Close unused charts and indicators to reduce CPU load');
    }
    
    if (systemStats.memory.percentage > 80) {
      tips.push('Clear old trade history and logs to free up memory');
      tips.push('Reduce the number of active symbol subscriptions');
    }
    
    if (systemStats.network.latency > 300) {
      tips.push('Consider switching to a closer API endpoint');
      tips.push('Reduce WebSocket subscriptions for better latency');
    }
    
    if (systemStats.cpu.temperature && systemStats.cpu.temperature > 75) {
      tips.push('Improve case ventilation or reduce overclock');
      tips.push('Consider frame rate limiting in display settings');
    }
    
    if (systemStats.trading.wsMessagesPerSecond > 100) {
      tips.push('You might be subscribed to too many order books');
    }
    
    return tips;
  }, [systemStats]);

  // Force garbage collection if available (Electron only)
  const forceCleanup = useCallback(async () => {
    if (isElectron()) {
      try {
        await window.nexlify.system.forceGC();
        toast.success('Memory cleanup initiated', {
          icon: '🧹',
          duration: 2000
        });
      } catch (error) {
        console.error('Failed to force GC:', error);
      }
    }
  }, []);

  return {
    // Current stats
    systemStats,
    healthStatus,
    alerts,
    
    // Helpers
    getOptimizationTips,
    forceCleanup,
    
    // Individual checks for UI
    isHighCPU: systemStats ? systemStats.cpu.usage > thresholds.cpu.warning : false,
    isHighMemory: systemStats ? (systemStats.memory.used / systemStats.memory.total) > (thresholds.memory.warning / 100) : false,
    isHighLatency: systemStats ? systemStats.network.latency > thresholds.latency.warning : false,
    isOverheating: systemStats?.cpu.temperature ? systemStats.cpu.temperature > thresholds.temperature.warning : false,
    
    // Performance score (0-100) - gamify system health
    performanceScore: systemStats ? calculatePerformanceScore(systemStats, thresholds) : 100
  };
}

// Calculate a performance score - because everyone loves a number
function calculatePerformanceScore(
  stats: SystemStats, 
  thresholds: HealthThresholds
): number {
  let score = 100;
  
  // CPU impact (0-30 points)
  const cpuPenalty = Math.min(30, (stats.cpu.usage / thresholds.cpu.critical) * 30);
  score -= cpuPenalty;
  
  // Memory impact (0-25 points)
  const memoryPercent = (stats.memory.used / stats.memory.total) * 100;
  const memoryPenalty = Math.min(25, (memoryPercent / thresholds.memory.critical) * 25);
  score -= memoryPenalty;
  
  // Network impact (0-25 points)
  const latencyPenalty = Math.min(25, (stats.network.latency / thresholds.latency.critical) * 25);
  score -= latencyPenalty;
  
  // Temperature impact (0-20 points)
  if (stats.cpu.temperature) {
    const tempPenalty = Math.min(20, ((stats.cpu.temperature - 50) / 40) * 20);
    score -= Math.max(0, tempPenalty);
  }
  
  // Bonus points for low resource usage (street cred for efficiency)
  if (stats.cpu.usage < 30 && memoryPercent < 50) {
    score += 5; // Efficiency bonus
  }
  
  return Math.max(0, Math.min(100, Math.round(score)));
}

// Specialized hook for performance warnings in UI
export function usePerformanceWarnings() {
  const { alerts, healthStatus, performanceScore } = useSystemMonitor({
    enableAlerts: false // We'll handle alerts differently in UI
  });
  
  const warningLevel = healthStatus === 'critical' ? 'danger' 
    : healthStatus === 'warning' ? 'warning' 
    : performanceScore < 70 ? 'caution' 
    : 'good';
    
  const warningColor = warningLevel === 'danger' ? 'text-red-500'
    : warningLevel === 'warning' ? 'text-yellow-500'
    : warningLevel === 'caution' ? 'text-orange-500'
    : 'text-green-500';
    
  return {
    warningLevel,
    warningColor,
    alerts,
    performanceScore,
    shouldShowWarning: warningLevel !== 'good'
  };
}
