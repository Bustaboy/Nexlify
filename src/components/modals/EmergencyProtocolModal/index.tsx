// Location: /src/components/modals/EmergencyProtocolModal/index.tsx
// Emergency stop protocol modal - the panic button when trades go sideways

import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertOctagon, 
  XCircle, 
  Lock, 
  AlertTriangle, 
  Eye, 
  EyeOff 
} from 'lucide-react';
import { EmergencyProtocol } from '../../../types/dashboard.types';
import { formatCredits } from '../../../utils/dashboard.utils';
import { Button, Input, Label } from '../../../styles/styled';

interface EmergencyProtocolModalProps {
  isOpen: boolean;
  onClose: () => void;
  onActivate: (password: string) => void;
  protocol: EmergencyProtocol;
  positions: Array<{ symbol: string; pnl: number }>;
}

const Overlay = styled(motion.div)`
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(8px);
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
`;

const ModalContainer = styled(motion.div)`
  background: rgba(17, 24, 39, 0.95);
  border: 2px solid var(--color-danger);
  border-radius: var(--radius-2xl);
  padding: 1.5rem;
  max-width: 32rem;
  width: 100%;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.75), 
              0 0 40px var(--color-danger-60);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--color-danger);
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const AlertBox = styled.div<{ $variant?: 'danger' | 'warning' }>`
  background: ${({ $variant = 'danger' }) => 
    $variant === 'danger' ? 'rgba(255, 23, 68, 0.1)' : 'rgba(255, 170, 0, 0.1)'
  };
  border: 1px solid ${({ $variant = 'danger' }) => 
    $variant === 'danger' ? 'rgba(255, 23, 68, 0.3)' : 'rgba(255, 170, 0, 0.3)'
  };
  border-radius: var(--radius-lg);
  padding: 1rem;
  margin-bottom: 1.5rem;
`;

const ProtocolList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const ProtocolItem = styled.li`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #D1D5DB;
  
  svg {
    color: var(--color-danger);
    flex-shrink: 0;
  }
`;

const PositionsSection = styled.div`
  margin-bottom: 1.5rem;
`;

const SectionTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: bold;
  color: #9CA3AF;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const PositionsList = styled.div`
  max-height: 12rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(31, 41, 55, 0.5);
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: var(--color-danger-60);
    border-radius: 3px;
  }
`;

const PositionItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: rgba(31, 41, 55, 0.5);
  border-radius: var(--radius-md);
`;

const Symbol = styled.span`
  font-size: 0.875rem;
  font-family: monospace;
  color: #E5E7EB;
`;

const PnL = styled.span<{ $positive: boolean }>`
  font-size: 0.875rem;
  font-weight: bold;
  font-family: monospace;
  color: ${({ $positive }) => $positive ? 'var(--color-success)' : 'var(--color-danger)'};
`;

const PasswordSection = styled.div`
  margin-bottom: 1.5rem;
`;

const PasswordInputWrapper = styled.div`
  position: relative;
`;

const PasswordInput = styled(Input)`
  padding-right: 2.5rem;
`;

const TogglePasswordButton = styled.button`
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: #9CA3AF;
  cursor: pointer;
  padding: 0.5rem;
  transition: color var(--transition-fast);
  
  &:hover {
    color: #D1D5DB;
  }
  
  &:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
    border-radius: var(--radius-sm);
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
`;

const ProtocolStatus = styled.div`
  background: rgba(255, 23, 68, 0.1);
  border: 1px solid rgba(255, 23, 68, 0.3);
  border-radius: var(--radius-lg);
  padding: 1rem;
  margin-bottom: 1.5rem;
`;

const StatusHeader = styled.p`
  font-size: 1.125rem;
  font-weight: bold;
  color: var(--color-danger);
  margin-bottom: 0.5rem;
`;

const StatusInfo = styled.p`
  font-size: 0.875rem;
  color: #9CA3AF;
  margin: 0.25rem 0;
`;

export const EmergencyProtocolModal: React.FC<EmergencyProtocolModalProps> = ({
  isOpen,
  onClose,
  onActivate,
  protocol,
  positions
}) => {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  
  if (!isOpen) return null;
  
  const sortedPositions = [...positions].sort((a, b) => a.pnl - b.pnl);
  const totalLoss = sortedPositions
    .filter(p => p.pnl < 0)
    .reduce((sum, p) => sum + p.pnl, 0);
  
  const handleActivate = () => {
    if (password) {
      onActivate(password);
      setPassword('');
    }
  };
  
  return (
    <AnimatePresence>
      <Overlay
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <ModalContainer
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
        >
          <Header>
            <AlertOctagon size={32} color="var(--color-danger)" />
            <Title>EMERGENCY PROTOCOL</Title>
          </Header>
          
          {!protocol.isActive ? (
            <>
              <AlertBox>
                <p style={{ fontSize: '0.875rem', color: '#E5E7EB', fontWeight: 600, marginBottom: '1rem' }}>
                  This will immediately:
                </p>
                <ProtocolList>
                  <ProtocolItem>
                    <XCircle size={16} />
                    <span>Close all positions (losing positions first)</span>
                  </ProtocolItem>
                  <ProtocolItem>
                    <Lock size={16} />
                    <span>Lock all trading operations</span>
                  </ProtocolItem>
                  <ProtocolItem>
                    <AlertTriangle size={16} />
                    <span>Require password to reset</span>
                  </ProtocolItem>
                </ProtocolList>
              </AlertBox>
              
              <PositionsSection>
                <SectionTitle>Positions to close (loss-making first):</SectionTitle>
                <PositionsList>
                  {sortedPositions.map((pos, idx) => (
                    <PositionItem key={idx}>
                      <Symbol>{pos.symbol}</Symbol>
                      <PnL $positive={pos.pnl >= 0}>
                        {formatCredits(pos.pnl)}
                      </PnL>
                    </PositionItem>
                  ))}
                </PositionsList>
                {totalLoss < 0 && (
                  <div style={{ marginTop: '0.5rem', textAlign: 'right' }}>
                    <span style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>
                      Total Loss: 
                    </span>
                    <span style={{ 
                      fontSize: '0.875rem', 
                      fontWeight: 'bold', 
                      color: 'var(--color-danger)',
                      marginLeft: '0.5rem'
                    }}>
                      {formatCredits(totalLoss)}
                    </span>
                  </div>
                )}
              </PositionsSection>
              
              <PasswordSection>
                <Label>Set Emergency Password</Label>
                <PasswordInputWrapper>
                  <PasswordInput
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="Enter secure password"
                    $hasError={false}
                  />
                  <TogglePasswordButton
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </TogglePasswordButton>
                </PasswordInputWrapper>
              </PasswordSection>
              
              <ButtonGroup>
                <Button $variant="secondary" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  $variant="danger"
                  onClick={handleActivate}
                  disabled={!password}
                >
                  ACTIVATE EMERGENCY STOP
                </Button>
              </ButtonGroup>
            </>
          ) : (
            <>
              <ProtocolStatus>
                <StatusHeader>EMERGENCY STOP ACTIVE</StatusHeader>
                <StatusInfo>
                  Triggered: {new Date(protocol.triggeredAt!).toLocaleString()}
                </StatusInfo>
                <StatusInfo>
                  Reason: {protocol.reason}
                </StatusInfo>
              </ProtocolStatus>
              
              <PositionsSection>
                <SectionTitle>Closed Positions:</SectionTitle>
                <PositionsList>
                  {protocol.closedPositions.map((pos, idx) => (
                    <PositionItem key={idx}>
                      <Symbol>{pos.symbol}</Symbol>
                      <PnL $positive={false}>
                        {formatCredits(pos.loss)}
                      </PnL>
                    </PositionItem>
                  ))}
                </PositionsList>
                <div style={{ marginTop: '0.5rem', textAlign: 'right' }}>
                  <span style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>
                    Total Closed: 
                  </span>
                  <span style={{ 
                    fontSize: '0.875rem', 
                    fontWeight: 'bold', 
                    color: 'var(--color-danger)',
                    marginLeft: '0.5rem'
                  }}>
                    {formatCredits(protocol.closedPositions.reduce((sum, p) => sum + p.loss, 0))}
                  </span>
                </div>
              </PositionsSection>
              
              <PasswordSection>
                <Label>Enter Password to Reset</Label>
                <PasswordInputWrapper>
                  <PasswordInput
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="Enter emergency password"
                    $hasError={false}
                  />
                </PasswordInputWrapper>
              </PasswordSection>
              
              <ButtonGroup>
                <Button $variant="secondary" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  $variant="success"
                  onClick={handleActivate}
                  disabled={!password}
                >
                  RESET PROTOCOL
                </Button>
              </ButtonGroup>
            </>
          )}
        </ModalContainer>
      </Overlay>
    </AnimatePresence>
  );
};

export default EmergencyProtocolModal;
