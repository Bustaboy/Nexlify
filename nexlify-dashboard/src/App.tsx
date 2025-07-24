// Location: C:\Nexlify\nexlify-dashboard\src\App.tsx
// Mission: 80-I Main App Entry with Adaptive Visual System
// Dependencies: AdaptiveVisualProvider, Original Components
// Context: Minimal wrapper that preserves your existing App structure

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './App.css';

// Import adaptive visual system
import { AdaptiveVisualProvider } from './systems/adaptive/core/AdaptiveVisualProvider';
import { AdaptiveVisualLayer } from './systems/adaptive/components/AdaptiveVisualLayer';

// Import your existing components (unchanged)
import { TrinityConsciousness } from './components/TrinityConsciousness';
import { CascadeAlert } from './components/CascadeAlert';
import { HardwareMonitor } from './components/HardwareMonitor';
import { TradingMetrics } from './components/TradingMetrics';

// Import adaptive styles
import './systems/adaptive/styles/adaptive.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);

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
    <AdaptiveVisualProvider
      config={{
        // Auto-detect hardware capabilities
        autoDetect: true,
        
        // Conservative defaults for trading safety
        defaultMode: 'balanced',
        maxGPUPercent: 25,
        maxMemoryMB: 400,
        targetFPS: 60,
        
        // Enable cyberpunk audio
        enableAudio: true,
        
        // Debug in development
        debugMode: import.meta.env?.DEV || false
      }}
    >
      {/* Adaptive visual effects layer */}
      <AdaptiveVisualLayer />
      
      {/* YOUR ORIGINAL APP STRUCTURE - UNCHANGED */}
      <div className="min-h-screen bg-nexlify-black font-mono overflow-hidden">
        {/* Cyberpunk grid background */}
        <div className="fixed inset-0 opacity-20 pointer-events-none">
          <div className="absolute inset-0" 
               style={{
                 backgroundImage: `
                   linear-gradient(cyan 1px, transparent 1px),
                   linear-gradient(90deg, cyan 1px, transparent 1px)
                 `,
                 backgroundSize: '50px 50px'
               }} 
          />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto">
          <header className="mb-8">
            <h1 className="text-5xl font-bold text-nexlify-cyan mb-2
                           animate-glow tracking-wider">
              NEXLIFY CONTROL
            </h1>
            <div className="flex items-center gap-4">
              <div className={`w-3 h-3 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              } `} />
              <span className="text-gray-400">
                {isConnected ? 'Trinity Online' : 'Connecting...'}
              </span>
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
    </AdaptiveVisualProvider>
  );
}

export default App;