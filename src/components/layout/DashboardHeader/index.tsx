// Location: /src/components/layout/DashboardHeader/index.tsx
// Main dashboard header with navigation and controls

import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Layers, 
  Cpu, 
  Server, 
  Palette, 
  RefreshCw, 
  Maximize2, 
  Minimize2,
  AlertOctagon,
  Lock
} from 'lucide-react';
import { useDashboardStore } from '../../../stores/dashboardStore';
import { THEMES } from '../../../constants/dashboard.constants';
import { GlitchText } from '../../common/GlitchText';
import { Button } from '../../../styles/styled';
import { ThemeService } from '../../../services/theme.service';

interface DashboardHeaderProps {
  onAPIConfigClick: () => void;
  onEmergencyClick: () => void;
}

const HeaderContainer = styled(motion.div)`
  position: sticky;
  top: 0;
  z-index: 40;
  background: rgba(3, 7, 18, 0.9);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--color-primary-40);
`;

const HeaderContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.25rem;
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1.5rem;
`;

const LogoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const LogoWrapper = styled.div`
  position: relative;
  width: 3rem;
  height: 3rem;
`;

const SpinningCpu = styled(motion.div)`
  position: absolute;
  inset: 0;
  opacity: 0.5;
`;

const MainLogo = styled(Layers)`
  position: relative;
  z-index: 10;
  width: 3rem;
  height: 3rem;
  color: var(--color-primary);
  filter: drop-shadow(0 0 20px var(--color-primary));
  stroke-width: 2.5;
`;

const BrandInfo = styled.div``;

const BrandName = styled.h1`
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: -0.025em;
  background: linear-gradient(
    to right, 
    var(--color-primary), 
    var(--color-accent), 
    var(--color-neural)
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 0 30px var(--color-primary));
  -webkit-text-stroke: 1px var(--color-primary-30);
`;

const BrandTagline = styled.p`
  font-size: 0.875rem;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
`;

const ConnectionStatus = styled.div<{ $status: 'online' | 'degraded' | 'offline' }>`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 1rem;
  border-radius: 9999px;
  border: 2px solid;
  backdrop-filter: blur(8px);
  
  ${({ $status }) => {
    switch ($status) {
      case 'online':
        return `
          border-color: rgba(0, 255, 65, 0.5);
          background: rgba(0, 255, 65, 0.1);
        `;
      case 'degraded':
        return `
          border-color: rgba(255, 170, 0, 0.5);
          background: rgba(255, 170, 0, 0.1);
        `;
      default:
        return `
          border-color: rgba(255, 23, 68, 0.5);
          background: rgba(255, 23, 68, 0.1);
        `;
    }
  }}
`;

const StatusIcon = styled(Server)<{ $status: 'online' | 'degraded' | 'offline' }>`
  width: 1rem;
  height: 1rem;
  color: ${({ $status }) => {
    switch ($status) {
      case 'online': return 'var(--color-success)';
      case 'degraded': return 'var(--color-warning)';
      default: return 'var(--color-danger)';
    }
  }};
`;

const ExchangeList = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
`;

const ExchangeName = styled.span`
  font-weight: bold;
  color: var(--color-success);
`;

const StatusDot = styled.div<{ $status: 'online' | 'degraded' | 'offline' }>`
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  
  background: ${({ $status }) => {
    switch ($status) {
      case 'online': return 'var(--color-success)';
      case 'degraded': return 'var(--color-warning)';
      default: return 'var(--color-danger)';
    }
  }};
`;

const RightSection = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const ControlButton = styled.button<{ $active?: boolean }>`
  position: relative;
  padding: 0.75rem;
  background: rgba(31, 41, 55, 0.5);
  border: 1px solid #374151;
  border-radius: var(--radius-lg);
  color: #9CA3AF;
  cursor: pointer;
  transition: all var(--transition-fast);
  
  &:hover:not(:disabled) {
    border-color: var(--color-primary-50);
    color: var(--color-primary);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  ${({ $active }) => $active && `
    background: var(--color-success-30);
    border-color: var(--color-success);
    color: var(--color-success);
    box-shadow: 0 0 20px var(--color-success-60);
    
    svg {
      animation: spin 2s linear infinite;
    }
  `}
`;

const EmergencyButton = styled(Button)<{ $isActive: boolean }>`
  ${({ $isActive }) => $isActive && `
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  `}
`;

const ThemeDropdown = styled.div`
  position: absolute;
  right: 0;
  top: calc(100% + 0.5rem);
  background: rgba(17, 24, 39, 0.95);
  border: 2px solid var(--color-primary-40);
  border-radius: var(--radius-lg);
  padding: 0.75rem;
  z-index: 50;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  min-width: 200px;
`;

const ThemeOption = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  padding: 0.5rem 1rem;
  background: ${({ $active }) => $active ? 'rgba(31, 41, 55, 0.8)' : 'transparent'};
  border: none;
  border-radius: var(--radius-md);
  color: ${({ $active }) => $active ? 'white' : '#9CA3AF'};
  cursor: pointer;
  transition: all var(--transition-fast);
  text-align: left;
  
  &:hover {
    background: rgba(31, 41, 55, 0.8);
    color: white;
  }
`;

const ThemeColorDot = styled.div<{ $color: string }>`
  width: 1rem;
  height: 1rem;
  border-radius: var(--radius-sm);
  background: ${({ $color }) => $color};
  box-shadow: 0 0 10px ${({ $color }) => $color};
`;

const ThemeName = styled.span`
  font-family: monospace;
  font-size: 0.875rem;
`;

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  onAPIConfigClick,
  onEmergencyClick
}) => {
  const {
    metrics,
    apiConfigs,
    emergencyProtocol,
    themeSettings,
    autoRefresh,
    isFullscreen,
    setAutoRefresh,
    setFullscreen,
    setThemeSettings
  } = useDashboardStore();
  
  const [showThemeSelector, setShowThemeSelector] = useState(false);
  
  const activeExchanges = apiConfigs.filter(c => c.isActive);
  const themeService = ThemeService.getInstance();
  
  const handleThemeChange = (themeName: keyof typeof THEMES) => {
    const newSettings = { ...themeSettings, currentTheme: themeName };
    setThemeSettings(newSettings);
    themeService.saveThemeSettings(newSettings);
    setShowThemeSelector(false);
  };
  
  return (
    <HeaderContainer
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      <HeaderContent>
        <LeftSection>
          <LogoContainer>
            <LogoWrapper>
              <SpinningCpu
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              >
                <Cpu size={48} color="var(--color-neural)" />
              </SpinningCpu>
              <MainLogo />
            </LogoWrapper>
            <BrandInfo>
              <GlitchText as="h1">
                <BrandName>NEXLIFY</BrandName>
              </GlitchText>
              <BrandTagline>
                Neural Chrome Trading v7.1 • {THEMES[themeSettings.currentTheme].name}
              </BrandTagline>
            </BrandInfo>
          </LogoContainer>
          
          <ConnectionStatus $status={metrics.connectionStatus}>
            <StatusIcon $status={metrics.connectionStatus} />
            <ExchangeList>
              {activeExchanges.length > 0 ? (
                activeExchanges.map((config, idx) => (
                  <React.Fragment key={config.exchange}>
                    {idx > 0 && <span style={{ color: '#4B5563' }}>•</span>}
                    <ExchangeName>{config.exchange}</ExchangeName>
                  </React.Fragment>
                ))
              ) : (
                <span style={{ color: '#9CA3AF' }}>No active exchanges</span>
              )}
            </ExchangeList>
            <StatusDot $status={metrics.connectionStatus} />
          </ConnectionStatus>
        </LeftSection>
        
        <RightSection>
          {emergencyProtocol.isActive ? (
            <EmergencyButton
              $variant="danger"
              $size="sm"
              $isActive={true}
              onClick={onEmergencyClick}
            >
              <Lock size={16} />
              <span>Protocol Active</span>
            </EmergencyButton>
          ) : (
            <Button
              $variant="secondary"
              $size="sm"
              onClick={onEmergencyClick}
              style={{ gap: '0.5rem' }}
            >
              <AlertOctagon size={16} />
              <span>Emergency</span>
            </Button>
          )}
          
          <div style={{ position: 'relative' }}>
            <ControlButton
              onClick={() => setShowThemeSelector(!showThemeSelector)}
              title="Theme"
            >
              <Palette size={20} />
            </ControlButton>
            
            {showThemeSelector && (
              <ThemeDropdown>
                {Object.entries(THEMES).map(([key, theme]) => (
                  <ThemeOption
                    key={key}
                    $active={themeSettings.currentTheme === key}
                    onClick={() => handleThemeChange(key as keyof typeof THEMES)}
                  >
                    <ThemeColorDot $color={theme.colors.primary} />
                    <ThemeName>{theme.name}</ThemeName>
                  </ThemeOption>
                ))}
              </ThemeDropdown>
            )}
          </div>
          
          <ControlButton onClick={onAPIConfigClick} title="Exchange Configuration">
            <Server size={20} />
          </ControlButton>
          
          <ControlButton
            $active={autoRefresh && !emergencyProtocol.isActive}
            onClick={() => setAutoRefresh(!autoRefresh)}
            disabled={emergencyProtocol.isActive}
            title="Auto Refresh"
          >
            <RefreshCw size={20} />
          </ControlButton>
          
          <ControlButton
            onClick={() => setFullscreen(!isFullscreen)}
            title="Toggle Fullscreen"
          >
            {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
          </ControlButton>
        </RightSection>
      </HeaderContent>
    </HeaderContainer>
  );
};

export default DashboardHeader;
