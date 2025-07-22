// Location: C:\Nexlify\nexlify-dashboard\src\components\CascadeAlert.tsx
// Status: NEW - Cascade prediction component
import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, DollarSign } from 'lucide-react';

export function CascadeAlert() {
  const [primaryCascade, setPrimaryCascade] = useState(null);

  useEffect(() => {
    const fetchCascades = async () => {
      try {
        const predictions = await invoke('get_cascade_predictions');
        console.log('Cascade predictions:', predictions);
      } catch (error) {
        console.error('Failed to fetch cascades:', error);
      }
    };

    fetchCascades();
    const interval = setInterval(fetchCascades, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="cyber-border bg-black/90 p-6 rounded-lg">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-6 h-6 text-yellow-400" />
        <h2 className="text-2xl font-bold text-nexlify-cyan">CASCADE DETECTION</h2>
      </div>
      <p className="text-gray-400">Monitoring for market cascades...</p>
    </div>
  );
}
