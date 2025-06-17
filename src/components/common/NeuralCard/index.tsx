// Location: /src/components/common/NeuralCard/index.tsx
// Enhanced info card component with cyberpunk styling

import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, HelpCircle, LucideIcon } from 'lucide-react';
import { BaseCard } from '../../../styles/styled';
import { formatCredits, formatPercent } from '../../../utils/dashboard.utils';

interface NeuralCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: LucideIcon;
  color?: string;
  subtitle?: string;
  description?: string;
  className?: string;
}

const CardContainer = styled(BaseCard)`
  position: relative;
  background: rgba(17, 24, 39, 0.9);
  backdrop-filter: blur(12px);
  border: 1px solid var(--color-primary-60);
  padding: 1.25rem;
  
  &:hover {
    transform: scale(1.02);
    box-shadow: 0 0 30px var(--color-primary-60);
  }
`;

const GridBackground = styled.div`
  position: absolute;
  inset: 0;
  opacity: 0.2;
  background-image: 
    linear-gradient(var(--color-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--color-grid) 1px, transparent 1px);
  background-size: 20px 20px;
  pointer-events: none;
`;

const AccentGradient = styled.div<{ $color?: string }>`
  position: absolute;
  top: 0;
  right: 0;
  width: 6rem;
  height: 6rem;
  border-bottom-left-radius: 100%;
  opacity: 0.3;
  background: linear-gradient(
    to bottom right, 
    ${({ $color }) => $color || 'var(--color-primary)'}66, 
    transparent
  );
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
`;

const TitleGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const IconContainer = styled.div<{ $color?: string }>`
  padding: 0.625rem;
  border-radius: var(--radius-lg);
  background: rgba(31, 41, 55, 0.8);
  color: ${({ $color }) => $color || 'var(--color-primary)'};
  box-shadow: 0 0 15px ${({ $color }) => $color || 'var(--color-primary)'}44;
  transition: all var(--transition-normal);
  
  .group:hover & {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
`;

const Title = styled.span`
  font-size: 0.875rem;
  color: #D1D5DB;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
`;

const ChangeIndicator = styled.div<{ $positive: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.25rem;
  color: ${({ $positive }) => $positive ? 'var(--color-success)' : 'var(--color-danger)'};
  font-size: 0.75rem;
  font-weight: bold;
  font-family: monospace;
`;

const ValueContainer = styled.div`
  position: relative;
  z-index: 10;
`;

const Value = styled.div<{ $color?: string }>`
  font-size: 1.5rem;
  font-weight: bold;
  font-family: monospace;
  letter-spacing: -0.025em;
  color: ${({ $color }) => $color || 'var(--color-primary)'};
  text-shadow: 0 0 20px ${({ $color }) => $color || 'var(--color-primary)'}66;
`;

const Subtitle = styled.div`
  font-size: 0.75rem;
  color: #9CA3AF;
  font-family: monospace;
  margin-top: 0.25rem;
`;

const Tooltip = styled(motion.div)`
  position: absolute;
  left: 0;
  bottom: 100%;
  margin-bottom: 0.5rem;
  width: 16rem;
  padding: 0.75rem;
  background: rgba(17, 24, 39, 0.95);
  border: 1px solid var(--color-primary-60);
  border-radius: var(--radius-lg);
  font-size: 0.75rem;
  color: #D1D5DB;
  z-index: 50;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(12px);
`;

const HelpIcon = styled(HelpCircle)`
  width: 1rem;
  height: 1rem;
  color: #6B7280;
  cursor: help;
  transition: color var(--transition-fast);
  
  &:hover {
    color: #9CA3AF;
  }
`;

export const NeuralCard: React.FC<NeuralCardProps> = ({
  title,
  value,
  change,
  icon: Icon,
  color,
  subtitle,
  description,
  className = ''
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  
  const formattedValue = typeof value === 'number' 
    ? formatCredits(value) 
    : value;
  
  return (
    <CardContainer
      className={`group ${className}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
    >
      <GridBackground />
      <AccentGradient $color={color} />
      
      <Header>
        <TitleGroup>
          <IconContainer $color={color}>
            <Icon size={20} />
          </IconContainer>
          <Title>{title}</Title>
          {description && (
            <div style={{ position: 'relative' }}>
              <HelpIcon
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
              />
              <AnimatePresence>
                {showTooltip && (
                  <Tooltip
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.2 }}
                  >
                    {description}
                  </Tooltip>
                )}
              </AnimatePresence>
            </div>
          )}
        </TitleGroup>
        {change !== undefined && (
          <ChangeIndicator $positive={change >= 0}>
            {change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
            <span>{formatPercent(change)}</span>
          </ChangeIndicator>
        )}
      </Header>
      
      <ValueContainer>
        <Value $color={color}>{formattedValue}</Value>
        {subtitle && <Subtitle>{subtitle}</Subtitle>}
      </ValueContainer>
    </CardContainer>
  );
};

export default NeuralCard;
