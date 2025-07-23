// Location: C:\Nexlify\nexlify-dashboard\src\components\TrinityDashboard.tsx
// Mission: 80-I Trinity Dashboard with Adaptive Visual Integration
// Dependencies: Adaptive Visual System, Tauri, Recharts, Lucide
// Context: Production dashboard with full cyberpunk chrome
// Status: PRESERVES ALL FUNCTIONALITY + ADDS ADAPTIVE VISUALS

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { 
  Brain, Eye, Zap, AlertTriangle, Activity, Cpu, HardDrive, 
  Thermometer, DollarSign, TrendingUp, TrendingDown 
} from 'lucide-react';

// Import Adaptive Visual System
import { useAdaptiveVisuals } from '../systems/adaptive/hooks/useAdaptiveVisuals';
import { AdaptiveNeonText } from '../systems/adaptive/components/AdaptiveNeonText';
import { AdaptiveChromeButton } from '../systems/adaptive/components/AdaptiveChromeButton';
import { AdaptiveGlitchText } from '../systems/adaptive/components/AdaptiveGlitchText';

// Original interfaces preserved
interface TrinityState {
  market_oracle_confidence: number;
  crowd_psyche_state: string;
  city_pulse_health: number;
  fusion_alignment: number;
  last_update: string;
}

interface CascadePrediction {
  id: string;
  event_type: string;
  confidence: number;
  time_to_event_minutes: number;
  affected_sectors: string[];
  predicted_impact: string;
}

interface HardwareMetrics {
  gpu_usage: number;
  gpu_temp: number;
  vram_used: number;
  vram_total: number;
  inference_time_ms: number;
}

interface TradingMetrics {
  daily_pnl: number;
  total_pnl: number;
  positions_open: number;
  cascade_trades_executed: number;
  success_rate: number;
}

// Trinity Consciousness Component with Adaptive Visuals
export function TrinityConsciousness() {
  const [trinity, setTrinity] = useState<TrinityState | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const { features, audioEngine } = useAdaptiveVisuals();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const state = await invoke<TrinityState>('get_trinity_state');
        setTrinity(state);
        
        // Add to history for chart
        setHistory(prev => [...prev.slice(-50), {
          time: new Date().toLocaleTimeString(),
          fusion: state.fusion_alignment,
          oracle: state.market_oracle_confidence,
          pulse: state.city_pulse_health
        }]);

        // Audio feedback for critical states
        if (state.fusion_alignment > 0.8 && features?.audioVisualization.enabled) {
          audioEngine?.playCascadeWarning('high');
        }
      } catch (error) {
        console.error('Failed to fetch Trinity state:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [features, audioEngine]);

  const getFusionColor = (alignment: number) => {
    if (alignment > 0.8) return 'red';
    if (alignment > 0.6) return 'yellow';
    return 'green';
  };

  const getPsycheGlow = (state: string) => {
    switch(state) {
      case 'ERUPTING': return 'red';
      case 'VOLATILE': return 'orange';
      case 'AGITATED': return 'yellow';
      default: return 'cyan';
    }
  };

  if (!trinity) return <div className="text-gray-400">Loading Trinity...</div>;

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <AdaptiveNeonText importance="critical" color="cyan" pulse>
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6" />
          TRINITY CONSCIOUSNESS
        </h2>
      </AdaptiveNeonText>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* Market Oracle Module */}
        <div className="bg-purple-950/30 p-4 rounded border border-purple-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Eye className="w-4 h-4 text-purple-400" />
            <span className="text-purple-400 text-sm">Market Oracle</span>
          </div>
          <AdaptiveNeonText importance="high" color="purple">
            <div className="text-2xl font-bold">
              {(trinity.market_oracle_confidence * 100).toFixed(1)}%
            </div>
          </AdaptiveNeonText>
        </div>

        {/* Crowd Psyche Module */}
        <div className="bg-orange-950/30 p-4 rounded border border-orange-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-orange-400" />
            <span className="text-orange-400 text-sm">Crowd Psyche</span>
          </div>
          <AdaptiveGlitchText 
            text={trinity.crowd_psyche_state}
            intensity={trinity.crowd_psyche_state === 'ERUPTING' ? 0.8 : 0.3}
          />
        </div>

        {/* City Pulse Module */}
        <div className="bg-blue-950/30 p-4 rounded border border-blue-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-blue-400 text-sm">City Pulse</span>
          </div>
          <AdaptiveNeonText importance="medium" color="blue">
            <div className="text-2xl font-bold">
              {(trinity.city_pulse_health * 100).toFixed(1)}%
            </div>
          </AdaptiveNeonText>
        </div>
      </div>

      {/* Fusion Alignment - Critical Metric */}
      <div className="mb-6 p-4 bg-gray-900/50 rounded border border-gray-700">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400">Consciousness Fusion</span>
          <AdaptiveNeonText 
            importance="critical" 
            color={getFusionColor(trinity.fusion_alignment) as any}
            pulse={trinity.fusion_alignment > 0.7}
          >
            <span className="text-2xl font-bold">
              {(trinity.fusion_alignment * 100).toFixed(1)}%
            </span>
          </AdaptiveNeonText>
        </div>
        <div className="w-full bg-gray-800 rounded h-2">
          <div 
            className={`h-2 rounded transition-all duration-300`}
            style={{
              width: `${trinity.fusion_alignment * 100}%`,
              backgroundColor: trinity.fusion_alignment > 0.8 ? '#ef4444' : 
                              trinity.fusion_alignment > 0.6 ? '#eab308' : '#10b981',
              boxShadow: features?.neonGlow.enabled ? 
                `0 0 10px ${trinity.fusion_alignment > 0.8 ? '#ef4444' : '#10b981'}` : 'none'
            }}
          />
        </div>
      </div>

      {/* Fusion History Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1a1a1a', 
                border: '1px solid #333',
                borderRadius: '4px'
              }} 
            />
            <Line 
              type="monotone" 
              dataKey="fusion" 
              stroke="#ef4444" 
              strokeWidth={2}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="oracle" 
              stroke="#a855f7" 
              strokeWidth={1}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="pulse" 
              stroke="#3b82f6" 
              strokeWidth={1}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// Cascade Alert Component with Adaptive Visuals
export function CascadeAlert() {
  const [cascades, setCascades] = useState<CascadePrediction[]>([]);
  const [executingTrade, setExecutingTrade] = useState<string | null>(null);
  const { setTradingActive, setCascadeDetected, audioEngine } = useAdaptiveVisuals();

  useEffect(() => {
    const fetchCascades = async () => {
      try {
        const predictions = await invoke<CascadePrediction[]>('get_cascade_predictions');
        
        // Check for new high-confidence cascades
        const highConfidence = predictions.filter(p => p.confidence > 0.8);
        if (highConfidence.length > 0 && cascades.length === 0) {
          setCascadeDetected(true);
          audioEngine?.playEffect('alert');
        } else if (highConfidence.length === 0) {
          setCascadeDetected(false);
        }
        
        setCascades(predictions);
      } catch (error) {
        console.error('Failed to fetch cascade predictions:', error);
      }
    };

    fetchCascades();
    const interval = setInterval(fetchCascades, 5000);
    return () => clearInterval(interval);
  }, [cascades.length, setCascadeDetected, audioEngine]);

  const executeCascadeTrade = async (cascadeId: string) => {
    setExecutingTrade(cascadeId);
    setTradingActive(true);
    
    try {
      await invoke('execute_cascade_trade', { cascadeId });
      audioEngine?.playUIFeedback('success');
    } catch (error) {
      console.error('Trade execution failed:', error);
      audioEngine?.playUIFeedback('error');
    } finally {
      setExecutingTrade(null);
      setTradingActive(false);
    }
  };

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <AdaptiveNeonText importance="high" color="red" pulse={cascades.length > 0}>
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <AlertTriangle className="w-6 h-6" />
          CASCADE DETECTION
        </h2>
      </AdaptiveNeonText>

      {cascades.length === 0 ? (
        <div className="text-gray-400 text-center py-8">
          No cascades detected
        </div>
      ) : (
        <div className="space-y-4">
          {cascades.map(cascade => (
            <div 
              key={cascade.id}
              className={`border rounded p-4 ${
                cascade.confidence > 0.8 
                  ? 'border-red-500 bg-red-950/20' 
                  : 'border-yellow-500 bg-yellow-950/20'
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <AdaptiveGlitchText 
                  text={cascade.event_type}
                  intensity={cascade.confidence}
                />
                <AdaptiveNeonText 
                  importance="critical" 
                  color={cascade.confidence > 0.8 ? 'red' : 'yellow'}
                >
                  <span className="text-sm font-bold">
                    {(cascade.confidence * 100).toFixed(0)}%
                  </span>
                </AdaptiveNeonText>
              </div>
              
              <div className="text-sm text-gray-400 mb-2">
                ETA: {cascade.time_to_event_minutes} minutes
              </div>
              
              <div className="text-sm text-gray-300 mb-3">
                Impact: {cascade.predicted_impact}
              </div>
              
              <AdaptiveChromeButton
                priority="critical"
                variant={cascade.confidence > 0.8 ? 'danger' : 'warning'}
                onClick={() => executeCascadeTrade(cascade.id)}
                loading={executingTrade === cascade.id}
                disabled={executingTrade !== null}
              >
                {executingTrade === cascade.id ? 'EXECUTING...' : 'EXECUTE TRADE'}
              </AdaptiveChromeButton>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Hardware Monitor with Adaptive Visuals
export function HardwareMonitor() {
  const [metrics, setMetrics] = useState<HardwareMetrics | null>(null);
  const { capabilities } = useAdaptiveVisuals();

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await invoke<HardwareMetrics>('get_hardware_metrics');
        setMetrics(data);
      } catch (error) {
        console.error('Failed to fetch hardware metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const getTempColor = (temp: number) => {
    if (temp > 80) return 'red';
    if (temp > 70) return 'orange';
    return 'green';
  };

  if (!metrics) return <div className="text-gray-400">Loading hardware...</div>;

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <AdaptiveNeonText importance="medium" color="green">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Cpu className="w-6 h-6" />
          HARDWARE STATUS
        </h2>
      </AdaptiveNeonText>

      <div className="grid grid-cols-2 gap-4">
        {/* GPU Usage */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-green-400" />
            <span className="text-gray-400 text-sm">GPU Usage</span>
          </div>
          <AdaptiveNeonText importance="medium" color="green">
            <div className="text-2xl font-bold">
              {metrics.gpu_usage.toFixed(1)}%
            </div>
          </AdaptiveNeonText>
        </div>

        {/* GPU Temperature */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Thermometer className="w-4 h-4 text-orange-400" />
            <span className="text-gray-400 text-sm">Temperature</span>
          </div>
          <AdaptiveNeonText 
            importance="medium" 
            color={getTempColor(metrics.gpu_temp) as any}
          >
            <div className="text-2xl font-bold">
              {metrics.gpu_temp}°C
            </div>
          </AdaptiveNeonText>
        </div>

        {/* VRAM Usage */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-4 h-4 text-blue-400" />
            <span className="text-gray-400 text-sm">VRAM</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {(metrics.vram_used / 1024).toFixed(1)}GB
          </div>
          <div className="text-xs text-gray-500">
            of {(metrics.vram_total / 1024).toFixed(1)}GB
          </div>
        </div>

        {/* Inference Time */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-gray-400 text-sm">Inference</span>
          </div>
          <AdaptiveNeonText 
            importance="high" 
            color={metrics.inference_time_ms < 10 ? 'green' : 'yellow'}
          >
            <div className="text-2xl font-bold">
              {metrics.inference_time_ms.toFixed(1)}ms
            </div>
          </AdaptiveNeonText>
        </div>
      </div>

      {/* Hardware Info */}
      {capabilities && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          Running on: {capabilities.gpu.renderer}
        </div>
      )}
    </div>
  );
}

// Trading Metrics with Adaptive Visuals
export function TradingMetrics() {
  const [metrics, setMetrics] = useState<TradingMetrics | null>(null);
  const { audioEngine } = useAdaptiveVisuals();

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await invoke<TradingMetrics>('get_trading_metrics');
        
        // Celebrate profits
        if (metrics && data.daily_pnl > metrics.daily_pnl && data.daily_pnl > 0) {
          audioEngine?.playUIFeedback('success');
        }
        
        setMetrics(data);
      } catch (error) {
        console.error('Failed to fetch trading metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [metrics, audioEngine]);

  if (!metrics) return <div className="text-gray-400">Loading metrics...</div>;

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'green' : 'red';
  };

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <AdaptiveNeonText importance="high" color="cyan">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <DollarSign className="w-6 h-6" />
          TRADING PERFORMANCE
        </h2>
      </AdaptiveNeonText>

      <div className="grid grid-cols-2 gap-4">
        {/* Daily P&L */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-gray-400 text-sm">Daily P&L</span>
          </div>
          <AdaptiveNeonText 
            importance="critical" 
            color={getPnLColor(metrics.daily_pnl) as any}
            pulse={Math.abs(metrics.daily_pnl) > 10000}
          >
            <div className="text-2xl font-bold">
              ${metrics.daily_pnl.toLocaleString()}
            </div>
          </AdaptiveNeonText>
        </div>

        {/* Total P&L */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-purple-400" />
            <span className="text-gray-400 text-sm">Total P&L</span>
          </div>
          <AdaptiveNeonText 
            importance="high" 
            color={getPnLColor(metrics.total_pnl) as any}
          >
            <div className="text-2xl font-bold">
              ${metrics.total_pnl.toLocaleString()}
            </div>
          </AdaptiveNeonText>
        </div>

        {/* Open Positions */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-gray-400 text-sm">Open Positions</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.positions_open}
          </div>
        </div>

        {/* Success Rate */}
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-yellow-400" />
            <span className="text-gray-400 text-sm">Success Rate</span>
          </div>
          <AdaptiveNeonText 
            importance="medium" 
            color={metrics.success_rate > 60 ? 'green' : 'yellow'}
          >
            <div className="text-2xl font-bold">
              {metrics.success_rate.toFixed(1)}%
            </div>
          </AdaptiveNeonText>
        </div>
      </div>

      {/* Cascade Trading Stats */}
      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="text-sm text-gray-400">
          Cascade Trades Executed: {metrics.cascade_trades_executed}
        </div>
      </div>
    </div>
  );
}

// Main Dashboard Component (preserves original structure)
export function TrinityDashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const { performanceMode } = useAdaptiveVisuals();

  useEffect(() => {
    const checkConnection = async () => {
      try {
        await invoke('ping');
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-nexlify-black font-mono overflow-hidden">
      <div className="relative z-10 p-6">
        <header className="flex justify-between items-center mb-8">
          <AdaptiveNeonText importance="critical" color="cyan" pulse>
            <h1 className="text-4xl font-bold">
              NEXLIFY CONTROL CENTER
            </h1>
          </AdaptiveNeonText>
          
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            } animate-pulse`} />
            <span className="text-gray-400">
              {isConnected ? 'Trinity Online' : 'Connecting...'}
            </span>
            {performanceMode === 'trading' && (
              <span className="ml-4 text-yellow-400 text-sm">
                [TRADING MODE ACTIVE]
              </span>
            )}
          </div>
        </header>

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-8">
            <TrinityConsciousness />
          </div>
          
          <div className="col-span-4">
            <CascadeAlert />
          </div>
          
          <div className="col-span-6">
            <HardwareMonitor />
          </div>
          
          <div className="col-span-6">
            <TradingMetrics />
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrinityDashboard;