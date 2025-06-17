// Location: /src/components/layout/TabNavigation/index.tsx
// Tab navigation bar for switching between dashboard views

import React from 'react';
import styled from 'styled-components';
import { Monitor, TrendingUp, ShieldAlert, Sliders } from 'lucide-react';
import { DashboardTab } from '../../../types/dashboard.types';
import { TabButton } from '../../common/TabButton';
import { useDashboardStore } from '../../../stores/dashboardStore';

const NavContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0 1.25rem 1rem;
  overflow-x: auto;
  
  /* Custom scrollbar for overflow */
  &::-webkit-scrollbar {
    height: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(31, 41, 55, 0.5);
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: var(--color-primary-60);
    border-radius: 3px;
    transition: background var(--transition-fast);
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: var(--color-primary-80);
  }
`;

interface TabConfig {
  id: DashboardTab;
  label: string;
  icon: React.ElementType;
  getNotificationCount?: () => number;
}

const TABS: TabConfig[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: Monitor
  },
  {
    id: 'trading',
    label: 'Trading Ops',
    icon: TrendingUp
  },
  {
    id: 'risk',
    label: 'Risk Analysis',
    icon: ShieldAlert,
    getNotificationCount: () => {
      const { alerts } = useDashboardStore.getState();
      return alerts.filter(a => a.severity === 'critical').length;
    }
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Sliders
  }
];

export const TabNavigation: React.FC = () => {
  const { activeTab, setActiveTab, emergencyProtocol } = useDashboardStore();
  
  return (
    <NavContainer>
      {TABS.map(tab => (
        <TabButton
          key={tab.id}
          active={activeTab === tab.id}
          onClick={() => setActiveTab(tab.id)}
          icon={tab.icon}
          label={tab.label}
          notification={tab.getNotificationCount?.()}
          disabled={emergencyProtocol.isActive && tab.id === 'trading'}
        />
      ))}
    </NavContainer>
  );
};

export default TabNavigation;
