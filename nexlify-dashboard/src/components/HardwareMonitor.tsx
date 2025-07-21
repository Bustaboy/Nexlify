// Location: C:\Nexlify\nexlify-dashboard\src\components\HardwareMonitor.tsx
// Create this new file
import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Cpu, HardDrive, Zap, Thermometer } from 'lucide-react';

interface HardwareMetric {
  gpu_usage: number;
  gpu_memory_used_mb: number;
  gpu_temp_celsius: number;
  inference_latency_ms: number;
  cpu_usage: number;
  ram_usage_gb: number;
  timestamp: string;
}

export function HardwareMonitor() {
  const [metrics, setMetrics] = useState<HardwareMetric | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await invoke<HardwareMetric>('get_hardware_metrics');
        setMetrics(data);
      } catch (error) {
        console.error('Failed to fetch hardware metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 1000);
    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <div>Loading hardware metrics...</div>;

  const getLatencyColor = (latency: number) => {
    if (latency < 10) return 'text-green-400';
    if (latency < 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <h2 className="text-2xl font-bold text-nexlify-cyan mb-6 flex items-center gap-2">
        <Cpu className="w-6 h-6" />
        HARDWARE PERFORMANCE
      </h2>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-gray-400 text-sm">GPU Usage</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.gpu_usage.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-500">
            {metrics.gpu_memory_used_mb} MB VRAM
          </div>
        </div>

        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Thermometer className="w-4 h-4 text-orange-400" />
            <span className="text-gray-400 text-sm">Temperature</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.gpu_temp_celsius}°C
          </div>
        </div>

        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-4 h-4 text-blue-400" />
            <span className="text-gray-400 text-sm">System RAM</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.ram_usage_gb.toFixed(1)} GB
          </div>
        </div>

        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-purple-400" />
            <span className="text-gray-400 text-sm">Inference</span>
          </div>
          <div className={`text-2xl font-bold ${getLatencyColor(metrics.inference_latency_ms)}`}>
            {metrics.inference_latency_ms.toFixed(1)}ms
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-gray-900/30 rounded">
        <p className="text-sm text-gray-400">
          Trinity Performance: {metrics.inference_latency_ms < 10 ? '✅ OPTIMAL' : '⚠️ SUBOPTIMAL'}
        </p>
      </div>
    </div>
  );
}