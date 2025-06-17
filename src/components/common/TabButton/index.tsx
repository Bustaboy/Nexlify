// Location: /src/components/common/TabButton/index.tsx
// Cyberpunk-styled tab navigation button

import React from 'react';
import styled, { css } from 'styled-components';
import { LucideIcon } from 'lucide-react';

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: LucideIcon;
  label: string;
  notification?: number;
  disabled?: boolean;
}

const StyledButton = styled.button<{ $active: boolean; $hasNotification: boolean }>`
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-lg);
  font-family: monospace;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  cursor: pointer;
  transition: all var(--transition-normal);
  
  ${({ $active }) => $active ? css`
    background: var(--color-primary-30);
    border: 2px solid var(--color-primary);
    color: white;
    box-shadow: 0 0 30px var(--color-primary-80);
  ` : css`
    background: rgba(31, 41, 55, 0.5);
    border: 1px solid #374151;
    color: #9CA3AF;
    
    &:hover:not(:disabled) {
      color: #D1D5DB;
      border-color: #4B5563;
      background: rgba(31, 41, 55, 0.8);
    }
  `}
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
  }
`;

const IconContainer = styled.span`
  display: flex;
  align-items: center;
`;

const NotificationBadge = styled.div`
  position: absolute;
  top: -0.5rem;
  right: -0.5rem;
  min-width: 1.25rem;
  height: 1.25rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-danger);
  color: white;
  font-size: 0.75rem;
  font-weight: bold;
  border-radius: var(--radius-full);
  padding: 0 0.25rem;
  box-shadow: 0 0 10px var(--color-danger);
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
`;

export const TabButton: React.FC<TabButtonProps> = ({
  active,
  onClick,
  icon: Icon,
  label,
  notification,
  disabled = false
}) => {
  return (
    <StyledButton
      $active={active}
      $hasNotification={!!notification}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
    >
      <IconContainer>
        <Icon size={16} />
      </IconContainer>
      <span>{label}</span>
      {notification !== undefined && notification > 0 && (
        <NotificationBadge>
          {notification > 99 ? '99+' : notification}
        </NotificationBadge>
      )}
    </StyledButton>
  );
};

export default TabButton;
