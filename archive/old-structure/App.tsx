// Location: /src/App.tsx
// Main Neural Interface - Where All Chrome Converges

import React, { useEffect } from 'react';
import { Dashboard } from './pages/Dashboard';
import { useDashboardStore } from './stores/dashboardStore';
import { mockDataService } from './services/mockData.service';
import './styles/globals.css';

function App() {
  // Initialize the neural grid
  useEffect(() => {
    // Set up mock data updates for demo
    const { startMockDataStream, stopMockDataStream } = mockDataService;
    
    // Initialize theme from localStorage
    const savedTheme = localStorage.getItem('nexlify_theme_settings');
    if (savedTheme) {
      try {
        const themeSettings = JSON.parse(savedTheme);
        useDashboardStore.setState({ themeSettings });
      } catch (e) {
        console.error('Failed to load theme settings:', e);
      }
    }

    // Start the data stream
    startMockDataStream();

    // Set up keyboard shortcuts
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ctrl/Cmd + L = Lock screen
      if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        useDashboardStore.setState({ isLocked: true });
      }
      
      // Ctrl/Cmd + E = Emergency protocol
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        const state = useDashboardStore.getState();
        if (!state.emergencyProtocol.isActive) {
          // Would normally trigger the modal, but for demo we'll just log
          console.log('Emergency Protocol shortcut triggered');
        }
      }
      
      // Ctrl/Cmd + F = Fullscreen
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        useDashboardStore.getState().toggleFullscreen();
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    // Clean up on unmount
    return () => {
      stopMockDataStream();
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  // Add global styles for the cyberpunk theme
  useEffect(() => {
    // Add cyber grid background
    document.body.style.background = `
      linear-gradient(to bottom, #0a0f1b, #000000),
      radial-gradient(ellipse at center, #1a1a2e 0%, #000000 100%)
    `;
    document.body.style.minHeight = '100vh';
    document.body.style.margin = '0';
    document.body.style.fontFamily = 'monospace';
    
    // Add global CSS for cyber grid effect
    const style = document.createElement('style');
    style.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
      
      * {
        box-sizing: border-box;
      }
      
      body {
        overflow-x: hidden;
        color: #ffffff;
      }
      
      /* Cyber grid background */
      .bg-cyber-grid {
        background-image: 
          linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        background-position: -1px -1px;
        animation: grid-move 10s linear infinite;
      }
      
      @keyframes grid-move {
        0% { background-position: -1px -1px; }
        100% { background-position: 49px 49px; }
      }
      
      /* Scrollbar styling */
      ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      
      ::-webkit-scrollbar-track {
        background: rgba(31, 41, 55, 0.5);
        border-radius: 5px;
      }
      
      ::-webkit-scrollbar-thumb {
        background: #00FFFF66;
        border-radius: 5px;
        border: 1px solid #00FFFF33;
      }
      
      ::-webkit-scrollbar-thumb:hover {
        background: #00FFFF88;
      }
      
      /* Selection color */
      ::selection {
        background: #00FFFF44;
        color: #FFFFFF;
      }
      
      /* Input focus glow */
      input:focus, select:focus, textarea:focus {
        outline: none;
        box-shadow: 0 0 0 2px #00FFFF44, 0 0 20px #00FFFF22;
      }
      
      /* Button hover effects */
      button {
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
      }
      
      button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
      }
      
      button:hover::before {
        width: 300px;
        height: 300px;
      }
      
      /* Glitch effect for text */
      @keyframes glitch {
        0% {
          text-shadow: 0.05em 0 0 #00ffff, -0.05em -0.025em 0 #ff00ff,
            0.025em 0.05em 0 #ffff00;
        }
        15% {
          text-shadow: 0.05em 0 0 #00ffff, -0.05em -0.025em 0 #ff00ff,
            0.025em 0.05em 0 #ffff00;
        }
        16% {
          text-shadow: -0.05em -0.025em 0 #00ffff, 0.025em 0.025em 0 #ff00ff,
            -0.05em -0.05em 0 #ffff00;
        }
        49% {
          text-shadow: -0.05em -0.025em 0 #00ffff, 0.025em 0.025em 0 #ff00ff,
            -0.05em -0.05em 0 #ffff00;
        }
        50% {
          text-shadow: 0.025em 0.05em 0 #00ffff, 0.05em 0 0 #ff00ff,
            0 -0.05em 0 #ffff00;
        }
        99% {
          text-shadow: 0.025em 0.05em 0 #00ffff, 0.05em 0 0 #ff00ff,
            0 -0.05em 0 #ffff00;
        }
        100% {
          text-shadow: -0.025em 0 0 #00ffff, -0.025em -0.025em 0 #ff00ff,
            -0.025em -0.05em 0 #ffff00;
        }
      }
      
      /* Loading animation */
      @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px #00FFFF; }
        50% { box-shadow: 0 0 20px #00FFFF, 0 0 30px #00FFFF; }
        100% { box-shadow: 0 0 5px #00FFFF; }
      }
      
      /* Tooltip styles */
      .tooltip {
        position: relative;
      }
      
      .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: #00FFFF;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #00FFFF44;
        font-size: 0.75rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s;
      }
      
      .tooltip:hover::after {
        opacity: 1;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return <Dashboard />;
}

export default App;
