import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { ArrowUpRight, ArrowDownRight, Activity, Brain, Shield, TrendingUp, AlertTriangle, Settings, User, LogOut, DollarSign, BarChart3, Zap, Lock, Cpu, Database, Globe } from 'lucide-react';

// Cyberpunk color palette
const colors = {
  neonPink: '#FF006E',
  neonBlue: '#00F5FF',
  neonPurple: '#BD00FF',
  neonGreen: '#00FF88',
  darkBg: '#0A0A0F',
  darkBg2: '#151521',
  darkBg3: '#1F1F2E',
  textPrimary: '#FFFFFF',
  textSecondary: '#A0A0B0',
  success: '#00FF88',
  danger: '#FF006E',
  warning: '#FFB800',
  gridLine: '#2A2A3E'
};

// Mock data generators
const generateCandleData = () => {
  const data = [];
  let basePrice = 50000;
  
  for (let i = 0; i < 24; i++) {
    const open = basePrice;
    const change = (Math.random() - 0.5) * 1000;
    const close = basePrice + change;
    const high = Math.max(open, close) + Math.random() * 500;
    const low = Math.min(open, close) - Math.random() * 500;
    
    data.push({
      time: `${i}:00`,
      open,
      high,
      low,
      close,
      volume: Math.random() * 1000000
    });
    
    basePrice = close;
  }
  
  return data;
};

const generatePortfolioData = () => {
  const data = [];
  let value = 10000;
  
  for (let i = 0; i < 30; i++) {
    value = value * (1 + (Math.random() - 0.45) * 0.05);
    data.push({
      day: i + 1,
      value: value.toFixed(2),
      profit: ((value - 10000) / 10000 * 100).toFixed(2)
    });
  }
  
  return data;
};

const NexlifyApp = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [marketData, setMarketData] = useState(generateCandleData());
  const [portfolioData, setPortfolioData] = useState(generatePortfolioData());
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [aiSignal, setAiSignal] = useState({ action: 'BUY', confidence: 0.87 });
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 45,
    memory: 62,
    latency: 12,
    uptime: 99.9
  });
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update last candle
      setMarketData(prev => {
        const newData = [...prev];
        const lastCandle = newData[newData.length - 1];
        const change = (Math.random() - 0.5) * 100;
        lastCandle.close = Math.max(0, lastCandle.close + change);
        lastCandle.high = Math.max(lastCandle.high, lastCandle.close);
        lastCandle.low = Math.min(lastCandle.low, lastCandle.close);
        lastCandle.volume = lastCandle.volume + Math.random() * 10000;
        return newData;
      });
      
      // Update system metrics
      setSystemMetrics(prev => ({
        cpu: Math.max(0, Math.min(100, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, prev.memory + (Math.random() - 0.5) * 5)),
        latency: Math.max(1, prev.latency + (Math.random() - 0.5) * 2),
        uptime: 99.9
      }));
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  const currentPrice = useMemo(() => {
    return marketData[marketData.length - 1]?.close || 0;
  }, [marketData]);
  
  const priceChange = useMemo(() => {
    if (marketData.length < 2) return 0;
    const current = marketData[marketData.length - 1].close;
    const previous = marketData[0].open;
    return ((current - previous) / previous * 100).toFixed(2);
  }, [marketData]);
  
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-900 p-3 rounded-lg border border-gray-700 shadow-2xl">
          <p className="text-gray-400 text-sm">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  
  const renderDashboard = () => (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Chart */}
      <div className="lg:col-span-2 bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-2xl">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <BarChart3 className="text-blue-400" />
              {selectedSymbol}
            </h2>
            <div className="flex items-center gap-4 mt-2">
              <span className="text-3xl font-mono text-white">
                ${currentPrice.toFixed(2)}
              </span>
              <span className={`flex items-center text-lg font-semibold ${parseFloat(priceChange) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {parseFloat(priceChange) >= 0 ? <ArrowUpRight /> : <ArrowDownRight />}
                {Math.abs(priceChange)}%
              </span>
            </div>
          </div>
          <div className="flex gap-2">
            {['1H', '4H', '1D', '1W'].map(tf => (
              <button
                key={tf}
                className="px-4 py-2 bg-gray-800 text-gray-400 rounded-lg hover:bg-gray-700 hover:text-white transition-all duration-200"
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
        
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={marketData}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors.neonBlue} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={colors.neonBlue} stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.gridLine} />
            <XAxis dataKey="time" stroke={colors.textSecondary} />
            <YAxis stroke={colors.textSecondary} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="close"
              stroke={colors.neonBlue}
              fillOpacity={1}
              fill="url(#colorPrice)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      {/* AI Signal Panel */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-2xl">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Brain className="text-purple-400" />
          Neural Analysis
        </h3>
        
        <div className={`p-6 rounded-lg mb-4 ${aiSignal.action === 'BUY' ? 'bg-green-900/20 border border-green-500/30' : 'bg-red-900/20 border border-red-500/30'}`}>
          <div className="text-center mb-4">
            <div className={`text-4xl font-bold ${aiSignal.action === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
              {aiSignal.action}
            </div>
            <div className="text-gray-400 mt-2">Signal Strength</div>
          </div>
          
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-xs font-semibold inline-block uppercase text-purple-400">
                  Confidence
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold inline-block text-purple-400">
                  {(aiSignal.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded-full bg-gray-800">
              <div
                style={{ width: `${aiSignal.confidence * 100}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-purple-500 to-pink-500"
              />
            </div>
          </div>
        </div>
        
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Entry Price</span>
            <span className="text-white font-mono">${(currentPrice * 0.998).toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Stop Loss</span>
            <span className="text-red-400 font-mono">${(currentPrice * 0.98).toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Take Profit</span>
            <span className="text-green-400 font-mono">${(currentPrice * 1.03).toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Risk/Reward</span>
            <span className="text-yellow-400 font-mono">1:2.5</span>
          </div>
        </div>
        
        <button className="w-full mt-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-200 transform hover:scale-105">
          Execute Trade
        </button>
      </div>
      
      {/* Portfolio Performance */}
      <div className="lg:col-span-2 bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-2xl">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="text-green-400" />
          Portfolio Performance
        </h3>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={portfolioData}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.gridLine} />
            <XAxis dataKey="day" stroke={colors.textSecondary} />
            <YAxis stroke={colors.textSecondary} />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="value"
              stroke={colors.neonGreen}
              strokeWidth={3}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
        
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">+12.5%</div>
            <div className="text-sm text-gray-400">Total Return</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">1.85</div>
            <div className="text-sm text-gray-400">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">67%</div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">-8.2%</div>
            <div className="text-sm text-gray-400">Max Drawdown</div>
          </div>
        </div>
      </div>
      
      {/* System Status */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-2xl">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Activity className="text-yellow-400" />
          System Status
        </h3>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400 flex items-center gap-1">
                <Cpu size={16} /> CPU Usage
              </span>
              <span className="text-white">{systemMetrics.cpu.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{
                  width: `${systemMetrics.cpu}%`,
                  backgroundColor: systemMetrics.cpu > 80 ? colors.danger : systemMetrics.cpu > 60 ? colors.warning : colors.success
                }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400 flex items-center gap-1">
                <Database size={16} /> Memory
              </span>
              <span className="text-white">{systemMetrics.memory.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{
                  width: `${systemMetrics.memory}%`,
                  backgroundColor: systemMetrics.memory > 85 ? colors.danger : systemMetrics.memory > 70 ? colors.warning : colors.success
                }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400 flex items-center gap-1">
                <Zap size={16} /> API Latency
              </span>
              <span className="text-white">{systemMetrics.latency.toFixed(0)}ms</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, systemMetrics.latency / 50 * 100)}%`,
                  backgroundColor: systemMetrics.latency > 30 ? colors.danger : systemMetrics.latency > 20 ? colors.warning : colors.success
                }}
              />
            </div>
          </div>
          
          <div className="pt-2 border-t border-gray-800">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 flex items-center gap-1">
                <Globe size={16} /> Uptime
              </span>
              <span className="text-green-400 font-bold">{systemMetrics.uptime}%</span>
            </div>
          </div>
        </div>
        
        <div className="mt-6 p-3 bg-gray-800 rounded-lg">
          <div className="flex items-center gap-2 text-sm">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-gray-400">All systems operational</span>
          </div>
        </div>
      </div>
    </div>
  );
  
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Animated background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute inset-0">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-pulse"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${3 + Math.random() * 4}s`
              }}
            >
              <div className="w-1 h-1 bg-blue-400 rounded-full opacity-50" />
            </div>
          ))}
        </div>
      </div>
      
      {/* Header */}
      <header className="relative z-10 bg-gray-900/80 backdrop-blur-lg border-b border-gray-800">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                NEXLIFY
              </h1>
              <nav className="hidden md:flex gap-6">
                {['Dashboard', 'Trading', 'Portfolio', 'AI Lab', 'Settings'].map((item) => (
                  <button
                    key={item}
                    onClick={() => setActiveTab(item.toLowerCase())}
                    className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                      activeTab === item.toLowerCase()
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </nav>
            </div>
            
            <div className="flex items-center gap-4">
              <button className="p-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 transition-all duration-200">
                <Shield size={20} />
              </button>
              <button className="p-2 rounded-lg bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 transition-all duration-200">
                <User size={20} />
              </button>
              <button className="p-2 rounded-lg bg-gray-800 text-red-400 hover:text-red-300 hover:bg-gray-700 transition-all duration-200">
                <LogOut size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-6 py-8">
        {activeTab === 'dashboard' && renderDashboard()}
        
        {activeTab === 'trading' && (
          <div className="text-center py-20">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Trading Interface
            </h2>
            <p className="text-gray-400">Advanced trading terminal coming soon...</p>
          </div>
        )}
        
        {activeTab === 'portfolio' && (
          <div className="text-center py-20">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
              Portfolio Manager
            </h2>
            <p className="text-gray-400">Manage your digital assets...</p>
          </div>
        )}
        
        {activeTab === 'ai lab' && (
          <div className="text-center py-20">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              AI Laboratory
            </h2>
            <p className="text-gray-400">Train and deploy neural trading models...</p>
          </div>
        )}
        
        {activeTab === 'settings' && (
          <div className="text-center py-20">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-yellow-400 to-red-400 bg-clip-text text-transparent">
              System Settings
            </h2>
            <p className="text-gray-400">Configure your trading environment...</p>
          </div>
        )}
      </main>
      
      {/* Footer Status Bar */}
      <footer className="fixed bottom-0 left-0 right-0 z-10 bg-gray-900/80 backdrop-blur-lg border-t border-gray-800">
        <div className="container mx-auto px-6 py-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-6">
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-gray-400">Connected to Nexus</span>
              </span>
              <span className="text-gray-400">|</span>
              <span className="text-gray-400">Block: 783,492</span>
              <span className="text-gray-400">|</span>
              <span className="text-gray-400">Gas: 25 gwei</span>
            </div>
            <div className="flex items-center gap-4 text-gray-400">
              <span>v2.0.0</span>
              <span>© 2024 Nexlify</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default NexlifyApp;
