// Location: C:\Nexlify\nexlify-dashboard\src\App.tsx
// Replace the entire contents of this file
import { useEffect, useState } from 'react';
import { TrinityConsciousness } from './components/TrinityConsciousness';
import { CascadeAlert } from './components/CascadeAlert';
import { HardwareMonitor } from './components/HardwareMonitor';
import { TradingMetrics } from './components/TradingMetrics';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize connection to Rust backend
    setTimeout(() => setIsConnected(true), 1000);
  }, []);

  return (
    <div className="min-h-screen bg-black p-4">
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
            } animate-pulse`} />
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
  );
}

export default App;