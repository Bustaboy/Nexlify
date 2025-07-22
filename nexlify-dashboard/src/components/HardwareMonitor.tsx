// Location: C:\Nexlify\nexlify-dashboard\src\components\HardwareMonitor.tsx
// Status: NEW - Hardware monitoring component
import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Cpu, HardDrive, Zap, Thermometer } from 'lucide-react';

export function HardwareMonitor() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await invoke('get_hardware_metrics');
        setMetrics(data);
      } catch (error) {
        console.error('Failed to fetch hardware metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 1000);
    return () => clearInterval(interval);
  }, []);

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
          <div className="text-2xl font-bold text-white">--%</div>
        </div>
        <div className="bg-gray-900/50 p-4 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            <span className="text-gray-400 text-sm">CPU Usage</span>
          </div>
          <div className="text-2xl font-bold text-white">--%</div>
        </div>
      </div>
    </div>
  );
}
