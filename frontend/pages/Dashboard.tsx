/**
 * Nexlify Dashboard - The Neural Command Center
 * Where all the streams converge, where eddies are counted
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign,
  BarChart3,
  Zap,
  AlertTriangle,
  Clock
} from 'lucide-react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Stores
import { useTradingStore } from '../stores/tradingStore';
import { useSettingsStore } from '../stores/settingsStore';

// Components
import { Card } from '../components/common/Card';
import { MetricCard } from '../components/dashboard/MetricCard';
import { PositionsList } from '../components/dashboard/PositionsList';
import { SignalsFeed } from '../components/dashboard/SignalsFeed';
import { MarketOverview } from '../components/dashboard/MarketOverview';
import { PerformanceChart } from '../components/dashboard/PerformanceChart';
import { NeuralNetworkVisualization } from '../components/dashboard/NeuralNetworkVisualization';

// API
import { apiClient } from '../lib/api';
import { formatCurrency, formatPercent } from '../lib/utils';

// Types
interface DashboardStats {
  totalEquity: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  weeklyPnL: number;
  monthlyPnL: number;
  openPositions: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  sharpeRatio: number;
  maxDrawdown: number;
  tradesToday: number;
  activeStrategies: number;
}

export const Dashboard: React.FC = () => {
  const { 
    portfolio, 
    positions, 
    signals, 
    marketData,
    isConnected 
  } = useTradingStore();
  
  const { theme, hideBalances } = useSettingsStore();
  
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | 'YTD'>('1D');
  const [selectedMetric, setSelectedMetric] = useState<'pnl' | 'positions' | 'volume'>('pnl');

  // Fetch dashboard stats
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await apiClient.get<DashboardStats>('/analytics/dashboard-stats');
      return response.data;
    },
    refetchInterval: 30000 // Update every 30 seconds
  });

  // Fetch performance history
  const { data: performanceData } = useQuery({
    queryKey: ['performance-history', timeframe],
    queryFn: async () => {
      const response = await apiClient.get('/analytics/performance', {
        params: { timeframe }
      });
      return response.data;
    }
  });

  // Calculate real-time metrics
  const realtimeMetrics = {
    totalValue: hideBalances ? '•••••' : formatCurrency(portfolio.totalValue),
    unrealizedPnL: portfolio.unrealizedPnL,
    unrealizedPnLPercent: portfolio.totalValue > 0 
      ? (portfolio.unrealizedPnL / portfolio.totalValue) * 100 
      : 0,
    activePositions: positions.length,
    pendingSignals: signals.filter(s => s.timestamp > new Date(Date.now() - 5 * 60000)).length
  };

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Activity className="w-6 h-6 text-cyan-400" />
              Neural Command Center
            </h1>
            <p className="text-sm text-gray-400 mt-1">
              {isConnected ? (
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  Neural link active • All systems operational
                </span>
              ) : (
                <span className="flex items-center gap-1 text-red-400">
                  <span className="w-2 h-2 bg-red-400 rounded-full" />
                  Neural link severed • Reconnecting...
                </span>
              )}
            </p>
          </div>

          {/* Timeframe selector */}
          <div className="flex items-center gap-2">
            {(['1D', '1W', '1M', '3M', 'YTD'] as const).map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-sm rounded transition-all ${
                  timeframe === tf
                    ? 'bg-cyan-500 text-black font-medium'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Total Equity"
            value={realtimeMetrics.totalValue}
            change={stats?.dailyPnLPercent || 0}
            icon={DollarSign}
            trend={stats?.dailyPnL || 0}
            loading={statsLoading}
          />

          <MetricCard
            title="Unrealized P&L"
            value={hideBalances ? '•••••' : formatCurrency(realtimeMetrics.unrealizedPnL)}
            change={realtimeMetrics.unrealizedPnLPercent}
            icon={realtimeMetrics.unrealizedPnL >= 0 ? TrendingUp : TrendingDown}
            trend={realtimeMetrics.unrealizedPnL}
            accentColor={realtimeMetrics.unrealizedPnL >= 0 ? 'green' : 'red'}
          />

          <MetricCard
            title="Win Rate"
            value={formatPercent(stats?.winRate || 0)}
            subtitle={`${stats?.tradesToday || 0} trades today`}
            icon={BarChart3}
            loading={statsLoading}
          />

          <MetricCard
            title="Active Positions"
            value={realtimeMetrics.activePositions.toString()}
            subtitle={`${realtimeMetrics.pendingSignals} pending signals`}
            icon={Zap}
            accentColor="purple"
          />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Performance Chart */}
          <div className="lg:col-span-2 space-y-6">
            <Card title="Performance Overview" className="h-96">
              <div className="flex items-center justify-between mb-4">
                <div className="flex gap-2">
                  {(['pnl', 'positions', 'volume'] as const).map((metric) => (
                    <button
                      key={metric}
                      onClick={() => setSelectedMetric(metric)}
                      className={`px-3 py-1 text-xs rounded transition-all ${
                        selectedMetric === metric
                          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                    >
                      {metric.toUpperCase()}
                    </button>
                  ))}
                </div>

                {stats && (
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-gray-400">Sharpe:</span>
                      <span className="text-white font-mono">
                        {stats.sharpeRatio.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-gray-400">Max DD:</span>
                      <span className="text-red-400 font-mono">
                        {formatPercent(stats.maxDrawdown)}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <PerformanceChart 
                data={performanceData}
                metric={selectedMetric}
                height={280}
              />
            </Card>

            {/* Active Positions */}
            <Card title="Active Positions" className="max-h-96 overflow-hidden">
              <PositionsList 
                positions={positions}
                marketData={marketData}
                hideBalances={hideBalances}
              />
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Neural Network Status */}
            <Card title="Neural Network Activity" className="h-64">
              <NeuralNetworkVisualization 
                activeStrategies={stats?.activeStrategies || 0}
                signalsPerMinute={signals.length}
                isActive={isConnected}
              />
            </Card>

            {/* Market Overview */}
            <Card title="Market Overview">
              <MarketOverview 
                marketData={marketData}
                favoriteSymbols={['BTC/USDT', 'ETH/USDT']}
              />
            </Card>

            {/* Recent Signals */}
            <Card 
              title="AI Trading Signals" 
              className="max-h-96"
              action={
                <span className="text-xs text-gray-400">
                  {signals.length} signals • Last 5 min
                </span>
              }
            >
              <SignalsFeed 
                signals={signals.slice(0, 10)}
                onSignalClick={(signal) => {
                  // Navigate to trading page with signal
                  console.log('Signal clicked:', signal);
                }}
              />
            </Card>
          </div>
        </div>

        {/* Bottom Section - Risk Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Avg Win/Loss</p>
                <p className="text-lg font-mono text-white">
                  {formatCurrency(stats?.avgWin || 0)} / 
                  <span className="text-red-400"> {formatCurrency(Math.abs(stats?.avgLoss || 0))}</span>
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-cyan-400 opacity-50" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">24h Volume</p>
                <p className="text-lg font-mono text-white">
                  {hideBalances ? '•••••' : formatCurrency(24567.89)}
                </p>
              </div>
              <Activity className="w-8 h-8 text-purple-400 opacity-50" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Risk Level</p>
                <p className="text-lg font-semibold text-yellow-400">
                  MODERATE
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-yellow-400 opacity-50" />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
