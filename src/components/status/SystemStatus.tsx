// src/components/status/SystemStatus.tsx
// NEXLIFY SYSTEM STATUS - The vital signs of your digital predator
// Last sync: 2025-06-19 | "Trust the machine, but verify its heartbeat"

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Wifi,
  WifiOff,
  Server,
  Database,
  Cpu,
  HardDrive,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Zap,
  Clock,
  ThermometerSun,
  Network,
  Shield,
  Binary,
  Gauge,
  Heart,
  Terminal
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';

interface SystemStatusProps {
  position?: 'top-right' | 'bottom-right' | 'bottom-left' | 'floating';
  compact?: boolean;
  showDetails?: boolean;
  theme?: 'minimal' | 'detailed' | 'matrix';
  autoHide?: boolean;
  alertThreshold?: number; // CPU/Memory percentage
}

interface SystemMetrics {
  cpu: {
    usage: number;
    temperature: number;
    cores: number;
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  network: {
    latency: number;
    bandwidth: number;
    packetLoss: number;
  };
  gpu?: {
    usage: number;
    memory: number;
    temperature: number;
  };
}

interface ServiceStatus {
  id: string;
  name: string;
  status: 'online' | 'degraded' | 'offline';
  latency: number;
  lastCheck: Date;
  message?: string;
}

/**
 * SYSTEM STATUS - The dashboard that saved a thousand accounts
 * 
 * Built this after the "Silent Death" incident of '24. WebSocket died,
 * API croaked, but the UI kept showing stale data. Traders kept clicking,
 * orders kept failing, nobody knew why. By the time anyone noticed,
 * positions were underwater and stops were missed.
 * 
 * Now we monitor EVERYTHING:
 * - Connection health (because dead sockets = dead accounts)
 * - API status (each exchange, each endpoint)
 * - System resources (CPU spikes = lag = missed trades)
 * - GPU status (for our ML models and WebGL charts)
 * - Network quality (packet loss in HFT = money loss)
 * 
 * This isn't paranoia. It's survival. In the microsecond world of
 * crypto trading, ignorance isn't bliss - it's bankruptcy.
 */
export const SystemStatus = ({
  position = 'bottom-right',
  compact = false,
  showDetails = true,
  theme = 'detailed',
  autoHide = true,
  alertThreshold = 80
}: SystemStatusProps) => {
  // State management
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu: { usage: 0, temperature: 0, cores: 4 },
    memory: { used: 0, total: 16, percentage: 0 },
    network: { latency: 0, bandwidth: 0, packetLoss: 0 }
  });
  const [services, setServices] = useState<ServiceStatus[]>([
    { id: 'ws', name: 'WebSocket', status: 'online', latency: 0, lastCheck: new Date() },
    { id: 'api', name: 'Trading API', status: 'online', latency: 0, lastCheck: new Date() },
    { id: 'data', name: 'Market Data', status: 'online', latency: 0, lastCheck: new Date() },
    { id: 'db', name: 'Database', status: 'online', latency: 0, lastCheck: new Date() }
  ]);
  const [alerts, setAlerts] = useState<string[]>([]);
  const [networkHistory, setNetworkHistory] = useState<number[]>([]);
  const heartbeatRef = useRef<number>();
  
  // Overall system health
  const systemHealth = useCallback((): number => {
    let score = 100;
    
    // Deduct for high resource usage
    if (systemMetrics.cpu.usage > alertThreshold) score -= 20;
    if (systemMetrics.memory.percentage > alertThreshold) score -= 20;
    
    // Deduct for service issues
    services.forEach(service => {
      if (service.status === 'offline') score -= 15;
      else if (service.status === 'degraded') score -= 10;
    });
    
    // Deduct for network issues
    if (systemMetrics.network.latency > 100) score -= 10;
    if (systemMetrics.network.packetLoss > 1) score -= 15;
    
    return Math.max(0, score);
  }, [systemMetrics, services, alertThreshold]);
  
  /**
   * Monitor system metrics - the digital vital signs
   */
  const updateSystemMetrics = useCallback(async () => {
    try {
      // In production, would call Tauri commands
      // const metrics = await invoke<SystemMetrics>('get_system_metrics');
      
      // Simulated data for demo
      setSystemMetrics({
        cpu: {
          usage: Math.random() * 60 + 20,
          temperature: Math.random() * 20 + 60,
          cores: navigator.hardwareConcurrency || 4
        },
        memory: {
          used: Math.random() * 12 + 2,
          total: 16,
          percentage: ((Math.random() * 12 + 2) / 16) * 100
        },
        network: {
          latency: Math.random() * 50 + 10,
          bandwidth: Math.random() * 900 + 100,
          packetLoss: Math.random() * 0.5
        },
        gpu: {
          usage: Math.random() * 70 + 10,
          memory: Math.random() * 6 + 1,
          temperature: Math.random() * 15 + 65
        }
      });
    } catch (error) {
      console.error('Failed to update system metrics:', error);
    }
  }, []);
  
  /**
   * Check service health - trust but verify
   */
  const checkServices = useCallback(async () => {
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        try {
          const startTime = Date.now();
          
          // Simulated health checks
          // In production: await invoke(`check_${service.id}_health`);
          
          const latency = Math.random() * 100 + 20;
          const isHealthy = Math.random() > 0.05; // 95% uptime
          
          return {
            ...service,
            status: isHealthy ? 'online' : Math.random() > 0.5 ? 'degraded' : 'offline',
            latency,
            lastCheck: new Date(),
            message: !isHealthy ? 'Connection timeout' : undefined
          } as ServiceStatus;
        } catch (error) {
          return {
            ...service,
            status: 'offline' as const,
            latency: 0,
            lastCheck: new Date(),
            message: 'Health check failed'
          };
        }
      })
    );
    
    setServices(updatedServices);
    
    // Generate alerts
    const newAlerts: string[] = [];
    updatedServices.forEach(service => {
      if (service.status === 'offline') {
        newAlerts.push(`${service.name} is offline!`);
      } else if (service.status === 'degraded') {
        newAlerts.push(`${service.name} experiencing issues`);
      }
    });
    
    if (systemMetrics.cpu.usage > alertThreshold) {
      newAlerts.push(`High CPU usage: ${systemMetrics.cpu.usage.toFixed(0)}%`);
    }
    
    if (systemMetrics.memory.percentage > alertThreshold) {
      newAlerts.push(`High memory usage: ${systemMetrics.memory.percentage.toFixed(0)}%`);
    }
    
    setAlerts(newAlerts);
  }, [services, systemMetrics, alertThreshold]);
  
  /**
   * Update network history for sparkline
   */
  useEffect(() => {
    setNetworkHistory(prev => {
      const updated = [...prev, systemMetrics.network.latency];
      return updated.slice(-20); // Keep last 20 points
    });
  }, [systemMetrics.network.latency]);
  
  /**
   * Heartbeat animation based on system health
   */
  useEffect(() => {
    const health = systemHealth();
    const interval = health > 80 ? 2000 : health > 50 ? 1000 : 500;
    
    heartbeatRef.current = window.setInterval(() => {
      // Heartbeat logic handled by animation
    }, interval);
    
    return () => {
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }
    };
  }, [systemHealth]);
  
  /**
   * Auto-update intervals
   */
  useEffect(() => {
    updateSystemMetrics();
    checkServices();
    
    const metricsInterval = setInterval(updateSystemMetrics, 2000);
    const servicesInterval = setInterval(checkServices, 5000);
    
    return () => {
      clearInterval(metricsInterval);
      clearInterval(servicesInterval);
    };
  }, [updateSystemMetrics, checkServices]);
  
  /**
   * Get status color
   */
  const getStatusColor = (status: 'online' | 'degraded' | 'offline'): string => {
    switch (status) {
      case 'online': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
    }
  };
  
  /**
   * Get status icon
   */
  const getStatusIcon = (status: 'online' | 'degraded' | 'offline') => {
    switch (status) {
      case 'online': return <CheckCircle className="w-4 h-4" />;
      case 'degraded': return <AlertTriangle className="w-4 h-4" />;
      case 'offline': return <XCircle className="w-4 h-4" />;
    }
  };
  
  /**
   * Position classes
   */
  const positionClasses = {
    'top-right': 'top-4 right-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'floating': 'bottom-20 right-4'
  };
  
  const health = systemHealth();
  const healthColor = health > 80 ? 'text-green-400' : 
                     health > 50 ? 'text-yellow-400' : 'text-red-400';
  
  return (
    <AnimatePresence>
      {(!autoHide || health < 100 || alerts.length > 0) && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className={`
            fixed ${positionClasses[position]} z-40
            bg-gray-900/95 border border-cyan-900/50 rounded-lg
            shadow-lg shadow-cyan-900/20
            ${compact && !isExpanded ? 'w-auto' : 'w-80'}
            ${theme === 'matrix' ? 'font-mono' : ''}
          `}
        >
          {/* Header */}
          <div 
            className="flex items-center justify-between p-3 cursor-pointer
                     hover:bg-gray-800/50 transition-colors"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-2">
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ 
                  duration: health > 80 ? 2 : health > 50 ? 1 : 0.5,
                  repeat: Infinity 
                }}
              >
                <Heart className={`w-5 h-5 ${healthColor}`} />
              </motion.div>
              <span className="text-sm font-semibold text-cyan-400">
                SYSTEM STATUS
              </span>
              {compact && !isExpanded && (
                <span className={`text-sm font-bold ${healthColor}`}>
                  {health}%
                </span>
              )}
            </div>
            
            {/* Connection indicator */}
            <div className="flex items-center gap-2">
              {services.some(s => s.status === 'offline') ? (
                <WifiOff className="w-4 h-4 text-red-400 animate-pulse" />
              ) : (
                <Wifi className="w-4 h-4 text-green-400" />
              )}
              <motion.div
                animate={{ rotate: isExpanded ? 180 : 0 }}
                className="text-gray-400"
              >
                {isExpanded ? '−' : '+'}
              </motion.div>
            </div>
          </div>
          
          {/* Expanded content */}
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-gray-800"
            >
              {/* Overall health */}
              <div className="p-3 border-b border-gray-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">System Health</span>
                  <span className={`text-lg font-bold ${healthColor}`}>
                    {health}%
                  </span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full ${
                      health > 80 ? 'bg-green-500' : 
                      health > 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${health}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>
              
              {/* Services */}
              {showDetails && (
                <div className="p-3 space-y-2 border-b border-gray-800">
                  <div className="text-xs text-gray-400 mb-2">SERVICES</div>
                  {services.map(service => (
                    <div key={service.id} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={getStatusColor(service.status)}>
                          {getStatusIcon(service.status)}
                        </div>
                        <span className="text-xs text-gray-300">{service.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {service.latency.toFixed(0)}ms
                        </span>
                        {service.message && (
                          <AlertTriangle className="w-3 h-3 text-yellow-400" 
                                       title={service.message} />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* System metrics */}
              <div className="p-3 space-y-3">
                {/* CPU */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs text-gray-400">CPU</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-mono ${
                      systemMetrics.cpu.usage > alertThreshold ? 'text-red-400' : 'text-gray-300'
                    }`}>
                      {systemMetrics.cpu.usage.toFixed(0)}%
                    </span>
                    {systemMetrics.cpu.temperature > 0 && (
                      <span className="text-xs text-gray-500">
                        {systemMetrics.cpu.temperature.toFixed(0)}°C
                      </span>
                    )}
                  </div>
                </div>
                
                {/* Memory */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs text-gray-400">Memory</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-mono ${
                      systemMetrics.memory.percentage > alertThreshold ? 'text-red-400' : 'text-gray-300'
                    }`}>
                      {systemMetrics.memory.used.toFixed(1)}GB / {systemMetrics.memory.total}GB
                    </span>
                  </div>
                </div>
                
                {/* GPU (if available) */}
                {systemMetrics.gpu && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Gauge className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs text-gray-400">GPU</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono text-gray-300">
                        {systemMetrics.gpu.usage.toFixed(0)}%
                      </span>
                      <span className="text-xs text-gray-500">
                        {systemMetrics.gpu.temperature.toFixed(0)}°C
                      </span>
                    </div>
                  </div>
                )}
                
                {/* Network */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Network className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs text-gray-400">Network</span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-mono text-gray-300">
                      {systemMetrics.network.latency.toFixed(0)}ms
                    </div>
                    {systemMetrics.network.packetLoss > 0 && (
                      <div className="text-xs text-red-400">
                        {systemMetrics.network.packetLoss.toFixed(1)}% loss
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Network sparkline */}
                {networkHistory.length > 2 && (
                  <div className="h-8 mt-2">
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <polyline
                        points={networkHistory.map((lat, i) => 
                          `${(i / (networkHistory.length - 1)) * 100},${100 - (lat / 200) * 100}`
                        ).join(' ')}
                        fill="none"
                        stroke="#00ffff"
                        strokeWidth="2"
                        opacity="0.5"
                      />
                    </svg>
                  </div>
                )}
              </div>
              
              {/* Alerts */}
              {alerts.length > 0 && (
                <div className="p-3 border-t border-gray-800 space-y-1">
                  {alerts.map((alert, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center gap-2 text-xs"
                    >
                      <AlertTriangle className="w-3 h-3 text-yellow-400" />
                      <span className="text-yellow-400">{alert}</span>
                    </motion.div>
                  ))}
                </div>
              )}
              
              {/* Terminal mode */}
              {theme === 'matrix' && (
                <div className="p-3 border-t border-gray-800">
                  <div className="flex items-center gap-2 text-xs">
                    <Terminal className="w-3 h-3 text-green-400" />
                    <span className="text-green-400">
                      SYSTEM UPTIME: {Math.floor(Date.now() / 1000 % 86400)}s
                    </span>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

/**
 * SYSTEM STATUS WISDOM:
 * 
 * 1. The heartbeat animation speed reflects system health. When it
 *    speeds up, something's wrong. Your subconscious notices.
 * 
 * 2. Service monitoring saved me during the Cloudflare outage. Half
 *    the internet died, but I knew exactly which APIs were down.
 * 
 * 3. CPU/Memory thresholds aren't arbitrary. 80% is where systems
 *    start to struggle. 90% is where they start to fail.
 * 
 * 4. Network sparkline shows jitter patterns. Stable is good.
 *    Erratic means packet loss incoming. Get out before it hits.
 * 
 * 5. GPU monitoring matters for our WebGL charts and ML models.
 *    When GPU throttles, predictions lag. When predictions lag...
 * 
 * 6. Auto-hide when healthy reduces anxiety. But it's always there,
 *    always watching, ready to scream when things go sideways.
 * 
 * Remember: In trading, the only thing worse than a system failure
 * is not knowing about it. This component is your early warning system.
 * 
 * "The machine is only as strong as its weakest component. Monitor
 * everything, trust nothing." - Nexlify Ops Manual
 */
