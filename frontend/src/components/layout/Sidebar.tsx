// frontend/src/components/layout/Sidebar.tsx

import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  TrendingUp,
  Shield,
  BarChart3,
  Bot,
  Trophy,
  Lock,
  ScrollText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  AlertTriangle,
  Activity,
  Brain
} from 'lucide-react';
import { cn } from '@lib/utils';
import { playSound } from '@lib/sounds';
import { useTradingStore } from '@stores/tradingStore';
import { useSettingsStore } from '@stores/settingsStore';

// The navigation spine of our operation - each icon a portal to power
// Been clicking these same buttons for years, worn smooth like river stones
// Each one holds memories - profits made, losses learned, nights survived

interface NavItem {
  path: string;
  label: string;
  icon: React.ElementType;
  badge?: number | string;
  badgeColor?: string;
  description?: string;
}

const navigationItems: NavItem[] = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    description: 'Neural command center'
  },
  {
    path: '/trading',
    label: 'Trading Matrix',
    icon: TrendingUp,
    description: 'Where chrome meets street'
  },
  {
    path: '/risk',
    label: 'Risk Matrix',
    icon: AlertTriangle,
    description: 'Keep your chrome intact'
  },
  {
    path: '/analytics',
    label: 'Analytics',
    icon: BarChart3,
    description: 'Data tells no lies'
  },
  {
    path: '/ai',
    label: 'AI Companion',
    icon: Brain,
    description: 'Your digital conscience'
  },
  {
    path: '/achievements',
    label: 'Achievements',
    icon: Trophy,
    description: 'Street cred earned'
  },
  {
    path: '/security',
    label: 'Security',
    icon: Lock,
    description: 'Guard your secrets'
  },
  {
    path: '/audit',
    label: 'Audit Trail',
    icon: ScrollText,
    description: 'Every move recorded'
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: Settings,
    description: 'Tune your chrome'
  }
];

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const { positions, signals } = useTradingStore();
  const { soundEnabled } = useSettingsStore();
  
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  // Calculate dynamic badges - real-time intel
  const getBadgeForPath = (path: string): { badge?: number | string; color?: string } => {
    switch (path) {
      case '/trading':
        const activePositions = positions.filter(p => p.status === 'open').length;
        return activePositions > 0 
          ? { badge: activePositions, color: 'bg-neon-green' }
          : {};
          
      case '/ai':
        const recentSignals = signals.filter(
          s => Date.now() - new Date(s.timestamp).getTime() < 300000 // 5 min
        ).length;
        return recentSignals > 0
          ? { badge: recentSignals, color: 'bg-neon-purple' }
          : {};
          
      case '/risk':
        const riskyPositions = positions.filter(
          p => p.unrealizedPnLPercent < -5 // 5% loss
        ).length;
        return riskyPositions > 0
          ? { badge: riskyPositions, color: 'bg-neon-red animate-pulse' }
          : {};
          
      default:
        return {};
    }
  };

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
    if (soundEnabled) playSound('click');
  };

  return (
    <motion.aside
      initial={{ x: -280 }}
      animate={{ 
        x: 0,
        width: isCollapsed ? 64 : 280 
      }}
      transition={{ type: "spring", damping: 20 }}
      className={cn(
        "relative flex flex-col h-full",
        "bg-cyber-black border-r border-cyber-dark",
        "overflow-hidden"
      )}
    >
      {/* Header with logo */}
      <div className="p-4 border-b border-cyber-dark">
        <motion.div
          animate={{ justifyContent: isCollapsed ? 'center' : 'space-between' }}
          className="flex items-center"
        >
          {!isCollapsed ? (
            <div className="flex items-center space-x-3">
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="relative"
              >
                <Zap className="w-6 h-6 text-neon-cyan" />
                <div className="absolute inset-0 blur-sm">
                  <Zap className="w-6 h-6 text-neon-cyan" />
                </div>
              </motion.div>
              <div>
                <h1 className="text-sm font-cyber text-white">NEXLIFY</h1>
                <p className="text-xs text-gray-500">TRADING MATRIX</p>
              </div>
            </div>
          ) : (
            <Zap className="w-6 h-6 text-neon-cyan mx-auto" />
          )}
        </motion.div>
      </div>

      {/* Navigation items */}
      <nav className="flex-1 py-4 overflow-y-auto custom-scrollbar">
        <ul className="space-y-1 px-2">
          {navigationItems.map((item) => {
            const { badge, color } = getBadgeForPath(item.path);
            const isActive = location.pathname === item.path;
            const Icon = item.icon;

            return (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  onMouseEnter={() => {
                    setHoveredItem(item.path);
                    if (soundEnabled) playSound('hover', { volume: 0.1 });
                  }}
                  onMouseLeave={() => setHoveredItem(null)}
                  onClick={() => {
                    if (soundEnabled) playSound('click');
                  }}
                  className={cn(
                    "relative flex items-center px-3 py-2 rounded-lg",
                    "transition-all duration-200 group",
                    isActive ? [
                      "bg-neon-cyan/20 text-neon-cyan",
                      "shadow-[inset_0_0_20px_rgba(0,255,255,0.2)]",
                      "border border-neon-cyan/30"
                    ] : [
                      "text-gray-400 hover:text-white",
                      "hover:bg-cyber-dark"
                    ]
                  )}
                >
                  {/* Active indicator - that sweet dopamine hit */}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 w-1 h-full bg-neon-cyan rounded-r"
                      initial={false}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    />
                  )}

                  {/* Icon with glow effect */}
                  <div className="relative">
                    <Icon className={cn(
                      "w-5 h-5 transition-all",
                      isActive && "filter drop-shadow-[0_0_8px_rgba(0,255,255,0.8)]"
                    )} />
                    
                    {/* Pulse effect on hover - subtle but satisfying */}
                    {hoveredItem === item.path && !isActive && (
                      <motion.div
                        className="absolute inset-0 rounded-full bg-white/20"
                        initial={{ scale: 1, opacity: 0 }}
                        animate={{ scale: 2, opacity: 0 }}
                        transition={{ duration: 0.6 }}
                      />
                    )}
                  </div>

                  {/* Label and description */}
                  <AnimatePresence>
                    {!isCollapsed && (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        transition={{ duration: 0.2 }}
                        className="ml-3 flex-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-sm">
                            {item.label}
                          </span>
                          
                          {/* Badge - the numbers that matter */}
                          {badge !== undefined && (
                            <motion.span
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className={cn(
                                "px-2 py-0.5 text-xs font-bold rounded-full",
                                color || "bg-cyber-gray text-white"
                              )}
                            >
                              {badge}
                            </motion.span>
                          )}
                        </div>
                        
                        {/* Description - street wisdom in subtitle form */}
                        {item.description && (
                          <p className="text-xs text-gray-500 mt-0.5">
                            {item.description}
                          </p>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Collapsed state badge */}
                  {isCollapsed && badge !== undefined && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className={cn(
                        "absolute -top-1 -right-1 w-2 h-2 rounded-full",
                        color || "bg-cyber-gray"
                      )}
                    />
                  )}
                </NavLink>

                {/* Tooltip for collapsed state */}
                {isCollapsed && hoveredItem === item.path && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-50"
                  >
                    <div className="bg-cyber-dark border border-cyber-gray px-3 py-2 rounded-lg shadow-xl">
                      <p className="text-sm font-medium text-white whitespace-nowrap">
                        {item.label}
                      </p>
                      {item.description && (
                        <p className="text-xs text-gray-400 mt-0.5">
                          {item.description}
                        </p>
                      )}
                    </div>
                  </motion.div>
                )}
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Collapse toggle */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={toggleSidebar}
        className={cn(
          "absolute top-1/2 -right-3 transform -translate-y-1/2",
          "w-6 h-6 rounded-full",
          "bg-cyber-dark border border-cyber-gray",
          "flex items-center justify-center",
          "text-gray-400 hover:text-white hover:border-neon-cyan",
          "transition-all duration-200 z-10"
        )}
      >
        {isCollapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </motion.button>

      {/* Bottom section - your street cred */}
      {!isCollapsed && (
        <div className="p-4 border-t border-cyber-dark">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Street Level</span>
            <span className="text-neon-cyan font-bold">42</span>
          </div>
          <div className="mt-2 h-1 bg-cyber-dark rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-neon-cyan to-neon-purple"
              initial={{ width: 0 }}
              animate={{ width: '67%' }}
              transition={{ duration: 1, delay: 0.5 }}
            />
          </div>
          <p className="text-xs text-gray-600 mt-1">
            2,847 XP to next level
          </p>
        </div>
      )}

      {/* Ambient glow effect */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-32 h-32 bg-neon-cyan/10 blur-3xl" />
        <div className="absolute bottom-0 right-0 w-32 h-32 bg-neon-purple/10 blur-3xl" />
      </div>
    </motion.aside>
  );
};
