// Location: /src/components/dashboard/TradingControls/index.tsx
// Trading control panel with leverage and position management

import React from 'react';
import styled, { css } from 'styled-components';
import { BaseCard, Label, Input } from '../../../styles/styled';
import { useDashboardStore } from '../../../stores/dashboardStore';

const ControlsCard = styled(BaseCard)`
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  background: rgba(17, 24, 39, 0.8);
  backdrop-filter: blur(12px);
  border: 1px solid var(--color-primary-40);
`;

const ControlsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
`;

const ControlGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const SliderContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const Slider = styled.input<{ $disabled?: boolean }>`
  flex: 1;
  height: 0.5rem;
  background: #374151;
  border-radius: 9999px;
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  opacity: ${({ $disabled }) => $disabled ? 0.5 : 1};
  
  /* Create dynamic fill based on value */
  background: ${({ value, min = 0, max = 100 }) => {
    const percentage = ((Number(value) - Number(min)) / (Number(max) - Number(min))) * 100;
    return `linear-gradient(to right, var(--color-primary) 0%, var(--color-primary) ${percentage}%, #374151 ${percentage}%, #374151 100%)`;
  }};
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 1.25rem;
    height: 1.25rem;
    background: var(--color-primary);
    border-radius: 50%;
    cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'grab'};
    box-shadow: 0 0 15px var(--color-primary-60);
    transition: all var(--transition-fast);
    
    &:hover {
      transform: scale(1.1);
      box-shadow: 0 0 20px var(--color-primary-80);
    }
    
    &:active {
      cursor: grabbing;
      transform: scale(0.95);
    }
  }
  
  &::-moz-range-thumb {
    width: 1.25rem;
    height: 1.25rem;
    background: var(--color-primary);
    border: none;
    border-radius: 50%;
    cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'grab'};
    box-shadow: 0 0 15px var(--color-primary-60);
    transition: all var(--transition-fast);
    
    &:hover {
      transform: scale(1.1);
      box-shadow: 0 0 20px var(--color-primary-80);
    }
    
    &:active {
      cursor: grabbing;
      transform: scale(0.95);
    }
  }
`;

const SliderValue = styled.span`
  min-width: 3rem;
  text-align: right;
  font-family: monospace;
  font-weight: bold;
  color: var(--color-primary);
  text-shadow: 0 0 10px var(--color-primary-60);
`;

const InfoDisplay = styled.div<{ $variant?: 'default' | 'success' | 'warning' | 'danger' }>`
  padding: 0.75rem 1rem;
  background: rgba(31, 41, 55, 0.8);
  border: 2px solid;
  border-radius: var(--radius-lg);
  font-family: monospace;
  font-weight: bold;
  text-align: center;
  
  ${({ $variant = 'default' }) => {
    switch ($variant) {
      case 'success':
        return css`
          border-color: var(--color-success-50);
          color: var(--color-success);
        `;
      case 'warning':
        return css`
          border-color: var(--color-warning-50);
          color: var(--color-warning);
        `;
      case 'danger':
        return css`
          border-color: var(--color-danger-50);
          color: var(--color-danger);
        `;
      default:
        return css`
          border-color: var(--color-primary-40);
          color: var(--color-primary);
        `;
    }
  }}
`;

const PositionInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
`;

const InfoLabel = styled.span`
  font-size: 0.75rem;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

export const TradingControls: React.FC = () => {
  const { 
    metrics, 
    emergencyProtocol,
    updateMetrics 
  } = useDashboardStore();
  
  const getMarginLevelVariant = () => {
    if (metrics.marginLevel > 150) return 'success';
    if (metrics.marginLevel > 100) return 'warning';
    return 'danger';
  };
  
  return (
    <ControlsCard
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <ControlsGrid>
        <ControlGroup>
          <Label>Leverage</Label>
          <SliderContainer>
            <Slider
              type="range"
              min="1"
              max="20"
              value={metrics.leverage}
              onChange={(e) => updateMetrics({ leverage: Number(e.target.value) })}
              $disabled={emergencyProtocol.isActive}
              disabled={emergencyProtocol.isActive}
            />
            <SliderValue>{metrics.leverage}x</SliderValue>
          </SliderContainer>
        </ControlGroup>
        
        <ControlGroup>
          <Label>Max Position Size</Label>
          <Input
            type="number"
            value={metrics.maxPositionSize}
            onChange={(e) => updateMetrics({ maxPositionSize: Number(e.target.value) })}
            disabled={emergencyProtocol.isActive}
            style={{ fontFamily: 'monospace', color: 'var(--color-primary)' }}
          />
        </ControlGroup>
        
        <ControlGroup>
          <Label>Open Positions</Label>
          <InfoDisplay>
            <PositionInfo>
              <InfoLabel>Active Positions</InfoLabel>
              <span>{metrics.openPositions} / 20</span>
            </PositionInfo>
          </InfoDisplay>
        </ControlGroup>
        
        <ControlGroup>
          <Label>Margin Level</Label>
          <InfoDisplay $variant={getMarginLevelVariant()}>
            <PositionInfo>
              <InfoLabel>Current Level</InfoLabel>
              <span>{metrics.marginLevel.toFixed(0)}%</span>
            </PositionInfo>
          </InfoDisplay>
        </ControlGroup>
      </ControlsGrid>
    </ControlsCard>
  );
};

export default TradingControls;
