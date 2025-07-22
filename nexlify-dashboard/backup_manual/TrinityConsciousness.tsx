// Location: C:\Nexlify\nexlify-dashboard\src\components\TrinityConsciousness.tsx
// Status: REVISED - Fixed incomplete file
import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, Eye, Zap } from 'lucide-react';

interface TrinityState {
  market_oracle_confidence: number;
  crowd_psyche_state: string;
  city_pulse_health: number;
  fusion_alignment: number;
  last_update: string;
}

export function TrinityConsciousness() {
  const [trinity, setTrinity] = useState<TrinityState | null>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const state = await invoke<TrinityState>('get_trinity_state');
        setTrinity(state);
ECHO is off.
        // Add to history for chart
        setHistory(prev => [...prev.slice(-50), {
          time: new Date().toLocaleTimeString(),
          fusion: state.fusion_alignment,
          oracle: state.market_oracle_confidence,
          pulse: state.city_pulse_health
        }]);
      } catch (error) {
        console.error('Failed to fetch Trinity state:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const getFusionColor = (alignment: number) => {
    if (alignment > 0.8) return 'text-red-500';
    if (alignment > 0.6) return 'text-yellow-500';
    return 'text-green-500';
  };

  if (!trinity) return <div className="text-gray-400">Loading Trinity...</div>;

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <h2 className="text-2xl font-bold text-nexlify-cyan mb-6 flex items-center gap-2">
        <Brain className="w-6 h-6" />
        TRINITY CONSCIOUSNESS
      </h2>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-purple-950/30 p-4 rounded border border-purple-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Eye className="w-4 h-4 text-purple-400" />
            <span className="text-purple-400 text-sm">Market Oracle</span>
          </div>
          <div className="text-2xl font-bold text-purple-300">
            {(trinity.market_oracle_confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-orange-950/30 p-4 rounded border border-orange-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-orange-400" />
            <span className="text-orange-400 text-sm">Crowd Psyche</span>
          </div>
          <div className="text-xl font-bold text-orange-300">
            {trinity.crowd_psyche_state}
          </div>
        </div>

        <div className="bg-green-950/30 p-4 rounded border border-green-500/50">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-green-400" />
            <span className="text-green-400 text-sm">City Pulse</span>
          </div>
          <div className="text-2xl font-bold text-green-300">
            {(trinity.city_pulse_health * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400">Fusion Alignment</span>
          <span className={`text-2xl font-bold ${getFusionColor(trinity.fusion_alignment)}`}>
            {(trinity.fusion_alignment * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-800 rounded-full h-4">
          <div 
            className={`h-full rounded-full transition-all duration-300 ${
              trinity.fusion_alignment > 0.8 ? 'bg-red-500 animate-pulse' : 'bg-cyan-500'
            }`}
            style={{ width: `${trinity.fusion_alignment * 100}%` }}
          />
        </div>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#000', border: '1px solid #0ff' }}
            />
            <Line 
              type="monotone" 
              dataKey="fusion" 
              stroke="#ff0040" 
              strokeWidth={2}
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="oracle" 
              stroke="#b300ff" 
              strokeWidth={1}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
