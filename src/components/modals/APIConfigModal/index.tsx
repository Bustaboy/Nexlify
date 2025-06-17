// Location: /src/components/modals/APIConfigModal/index.tsx
// Multi-exchange API configuration modal

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Server, XCircle, GitMerge } from 'lucide-react';
import { APIConfig } from '../../../types/dashboard.types';
import { DEFAULT_ENDPOINTS } from '../../../constants/dashboard.constants';
import { Button, Input, Label } from '../../../styles/styled';

interface APIConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (configs: APIConfig[]) => void;
  currentConfigs: APIConfig[];
}

const Overlay = styled(motion.div)`
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(8px);
  z-index: 200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
`;

const ModalContainer = styled(motion.div)`
  background: rgba(17, 24, 39, 0.95);
  border: 2px solid var(--color-primary-60);
  border-radius: var(--radius-2xl);
  max-width: 64rem;
  width: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.75);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem;
  border-bottom: 1px solid var(--color-primary-30);
`;

const Title = styled.h2`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--color-primary);
  
  svg {
    width: 1.75rem;
    height: 1.75rem;
  }
`;

const ModeToggle = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(31, 41, 55, 0.8);
  border: 1px solid var(--color-primary-40);
  border-radius: var(--radius-lg);
`;

const ModeButton = styled.button<{ $active: boolean }>`
  padding: 0.5rem 0.75rem;
  border-radius: var(--radius-md);
  font-size: 0.75rem;
  font-weight: bold;
  font-family: monospace;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  transition: all var(--transition-fast);
  border: 1px solid;
  cursor: pointer;
  
  background: ${({ $active }) => 
    $active ? 'var(--color-warning-30)' : 'var(--color-success-30)'
  };
  color: ${({ $active }) => 
    $active ? 'var(--color-warning)' : 'var(--color-success)'
  };
  border-color: ${({ $active }) => 
    $active ? 'var(--color-warning)' : 'var(--color-success)'
  };
  box-shadow: ${({ $active }) => 
    $active ? '0 0 10px var(--color-warning-40)' : '0 0 10px var(--color-success-40)'
  };
`;

const Content = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(31, 41, 55, 0.5);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: var(--color-primary-60);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: var(--color-primary-80);
  }
`;

const ExchangeCard = styled.div<{ $isActive: boolean }>`
  background: rgba(31, 41, 55, 0.6);
  border: 2px solid;
  border-color: ${({ $isActive }) => 
    $isActive ? 'var(--color-success-60)' : 'var(--color-primary-30)'
  };
  border-radius: var(--radius-xl);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  backdrop-filter: blur(8px);
  transition: all var(--transition-normal);
  
  ${({ $isActive }) => $isActive && `
    box-shadow: 0 0 20px var(--color-success-30);
  `}
`;

const ExchangeHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const ExchangeName = styled.h3`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.125rem;
  font-weight: bold;
  text-transform: capitalize;
  color: var(--color-accent);
`;

const StatusIndicator = styled.div<{ $status: 'active' | 'configured' | 'offline' }>`
  width: 0.75rem;
  height: 0.75rem;
  border-radius: 50%;
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  
  background: ${({ $status }) => {
    switch ($status) {
      case 'active': return 'var(--color-success)';
      case 'configured': return 'var(--color-warning)';
      default: return 'var(--color-danger)';
    }
  }};
  
  box-shadow: ${({ $status }) => {
    switch ($status) {
      case 'active': return '0 0 15px var(--color-success)';
      case 'configured': return '0 0 15px var(--color-warning)';
      default: return '0 0 15px var(--color-danger)';
    }
  }};
`;

const StatusBadge = styled.span<{ $status: 'active' | 'configured' | 'offline' }>`
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-md);
  font-size: 0.75rem;
  font-family: monospace;
  font-weight: bold;
  margin-left: 0.5rem;
  
  background: ${({ $status }) => {
    switch ($status) {
      case 'active': return 'var(--color-success-20)';
      case 'configured': return 'var(--color-warning-20)';
      default: return 'var(--color-danger-20)';
    }
  }};
  
  color: ${({ $status }) => {
    switch ($status) {
      case 'active': return 'var(--color-success)';
      case 'configured': return 'var(--color-warning)';
      default: return 'var(--color-danger)';
    }
  }};
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
`;

const FormField = styled.div`
  display: flex;
  flex-direction: column;
`;

const Footer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-top: 1px solid var(--color-primary-30);
`;

const StatusSummary = styled.div`
  font-size: 0.875rem;
  color: #9CA3AF;
  
  span {
    font-weight: bold;
    margin: 0 0.25rem;
  }
`;

const InfoBox = styled.div`
  background: rgba(31, 41, 55, 0.4);
  border: 2px solid var(--color-neural-40);
  border-radius: var(--radius-xl);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
`;

const InfoTitle = styled.h4`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  font-weight: bold;
  color: var(--color-neural);
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const ExchangeStats = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin-top: 1rem;
`;

const StatCard = styled.div`
  text-align: center;
  padding: 0.5rem;
  background: rgba(17, 24, 39, 0.5);
  border-radius: var(--radius-lg);
`;

const StatValue = styled.div<{ $color: string }>`
  font-size: 1.5rem;
  font-weight: bold;
  color: ${({ $color }) => $color};
`;

const StatLabel = styled.div`
  font-size: 0.75rem;
  color: #6B7280;
`;

export const APIConfigModal: React.FC<APIConfigModalProps> = ({
  isOpen,
  onClose,
  onSave,
  currentConfigs
}) => {
  const [configs, setConfigs] = useState<APIConfig[]>(currentConfigs);
  const [testingAPI, setTestingAPI] = useState<string | null>(null);
  const [sandboxMode, setSandboxMode] = useState(true);
  
  useEffect(() => {
    if (currentConfigs.length === 0) {
      // Initialize with default configs
      const defaultConfigs = ['binance', 'kraken', 'coinbase'].map(exchange => ({
        exchange,
        endpoint: DEFAULT_ENDPOINTS[exchange as keyof typeof DEFAULT_ENDPOINTS][sandboxMode ? 'testnet' : 'mainnet'],
        apiKey: '',
        apiSecret: '',
        testnet: sandboxMode,
        rateLimit: 1200,
        isActive: false
      }));
      setConfigs(defaultConfigs);
    } else {
      setConfigs(currentConfigs);
    }
  }, [currentConfigs, sandboxMode]);
  
  const handleTest = async (exchange: string) => {
    setTestingAPI(exchange);
    // MOCK_DATA: Simulate API test
    await new Promise(resolve => setTimeout(resolve, 2000));
    setTestingAPI(null);
  };
  
  const updateConfig = (exchange: string, field: keyof APIConfig, value: any) => {
    setConfigs(prev => {
      const existing = prev.find(c => c.exchange === exchange);
      if (existing) {
        return prev.map(c => 
          c.exchange === exchange ? { ...c, [field]: value } : c
        );
      } else {
        return [...prev, {
          exchange,
          endpoint: DEFAULT_ENDPOINTS[exchange as keyof typeof DEFAULT_ENDPOINTS][sandboxMode ? 'testnet' : 'mainnet'],
          apiKey: '',
          apiSecret: '',
          testnet: sandboxMode,
          rateLimit: 1200,
          isActive: false,
          [field]: value
        }];
      }
    });
  };
  
  const getExchangeStatus = (config: APIConfig): 'active' | 'configured' | 'offline' => {
    if (config.isActive && config.apiKey && config.apiSecret) return 'active';
    if (config.apiKey && config.apiSecret) return 'configured';
    return 'offline';
  };
  
  if (!isOpen) return null;
  
  const activeCount = configs.filter(c => c.isActive).length;
  const configuredCount = configs.filter(c => c.apiKey && c.apiSecret && !c.isActive).length;
  const totalCount = configs.filter(c => c.apiKey && c.apiSecret).length;
  
  return (
    <AnimatePresence>
      <Overlay
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <ModalContainer
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={e => e.stopPropagation()}
        >
          <Header>
            <Title>
              <Server />
              Neural Exchange Matrix
            </Title>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <ModeToggle>
                <span style={{ fontSize: '0.875rem', color: '#9CA3AF' }}>Mode:</span>
                <ModeButton
                  $active={sandboxMode}
                  onClick={() => setSandboxMode(!sandboxMode)}
                >
                  {sandboxMode ? 'SANDBOX' : 'LIVE'}
                </ModeButton>
              </ModeToggle>
              <XCircle 
                size={24}
                color="#9CA3AF"
                cursor="pointer"
                onClick={onClose}
                style={{ transition: 'color 0.2s' }}
                onMouseEnter={e => e.currentTarget.style.color = '#FFF'}
                onMouseLeave={e => e.currentTarget.style.color = '#9CA3AF'}
              />
            </div>
          </Header>
          
          <Content>
            {['binance', 'kraken', 'coinbase'].map(exchange => {
              const config = configs.find(c => c.exchange === exchange) || {
                exchange,
                endpoint: DEFAULT_ENDPOINTS[exchange as keyof typeof DEFAULT_ENDPOINTS][sandboxMode ? 'testnet' : 'mainnet'],
                apiKey: '',
                apiSecret: '',
                testnet: sandboxMode,
                rateLimit: 1200,
                isActive: false
              };
              
              const status = getExchangeStatus(config);
              const isConnected = config.apiKey && config.apiSecret;
              
              return (
                <ExchangeCard key={exchange} $isActive={config.isActive}>
                  <ExchangeHeader>
                    <ExchangeName>
                      <StatusIndicator $status={status} />
                      <span>{exchange}</span>
                      {status !== 'offline' && (
                        <StatusBadge $status={status}>
                          [{status.toUpperCase()}]
                        </StatusBadge>
                      )}
                    </ExchangeName>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {isConnected && (
                        <Button
                          $variant={config.isActive ? 'danger' : 'success'}
                          $size="sm"
                          onClick={() => updateConfig(exchange, 'isActive', !config.isActive)}
                        >
                          {config.isActive ? 'DEACTIVATE' : 'ACTIVATE'}
                        </Button>
                      )}
                      <Button
                        $variant="primary"
                        $size="sm"
                        disabled={testingAPI === exchange || !isConnected}
                        onClick={() => handleTest(exchange)}
                      >
                        {testingAPI === exchange ? 'Testing...' : 'Test Connection'}
                      </Button>
                    </div>
                  </ExchangeHeader>
                  
                  <FormGrid>
                    <FormField>
                      <Label>API Endpoint</Label>
                      <Input
                        type="text"
                        value={config.endpoint}
                        onChange={e => updateConfig(exchange, 'endpoint', e.target.value)}
                      />
                    </FormField>
                    
                    <FormField>
                      <Label>Rate Limit (req/min)</Label>
                      <Input
                        type="number"
                        value={config.rateLimit}
                        onChange={e => updateConfig(exchange, 'rateLimit', Number(e.target.value))}
                      />
                    </FormField>
                    
                    <FormField>
                      <Label>API Key</Label>
                      <Input
                        type="text"
                        value={config.apiKey}
                        onChange={e => updateConfig(exchange, 'apiKey', e.target.value)}
                        placeholder="Enter API key"
                      />
                    </FormField>
                    
                    <FormField>
                      <Label>API Secret</Label>
                      <Input
                        type="password"
                        value={config.apiSecret}
                        onChange={e => updateConfig(exchange, 'apiSecret', e.target.value)}
                        placeholder="Enter API secret"
                      />
                    </FormField>
                  </FormGrid>
                  
                  <div style={{ marginTop: '1rem', fontSize: '0.75rem', color: '#6B7280' }}>
                    {sandboxMode ? 
                      "🧪 Sandbox mode - Safe for testing with fake funds" : 
                      "⚠️ LIVE MODE - Real funds at risk! Double-check everything!"
                    }
                  </div>
                </ExchangeCard>
              );
            })}
            
            <InfoBox>
              <InfoTitle>
                <GitMerge size={20} />
                MULTI-EXCHANGE ARBITRAGE SYSTEM
              </InfoTitle>
              <p style={{ fontSize: '0.75rem', color: '#9CA3AF', lineHeight: 1.6 }}>
                Connect multiple exchanges to enable cross-exchange arbitrage, better liquidity, and risk distribution. 
                Each exchange operates independently with its own API limits and balance. The Neural AI will automatically 
                detect arbitrage opportunities between active exchanges and execute trades within milliseconds.
              </p>
              <ExchangeStats>
                <StatCard>
                  <StatValue $color="var(--color-success)">{activeCount}</StatValue>
                  <StatLabel>Active</StatLabel>
                </StatCard>
                <StatCard>
                  <StatValue $color="var(--color-warning)">{configuredCount}</StatValue>
                  <StatLabel>Configured</StatLabel>
                </StatCard>
                <StatCard>
                  <StatValue $color="var(--color-primary)">{totalCount}</StatValue>
                  <StatLabel>Total</StatLabel>
                </StatCard>
              </ExchangeStats>
            </InfoBox>
          </Content>
          
          <Footer>
            <StatusSummary>
              <span style={{ color: 'var(--color-success)' }}>{activeCount}</span> active •
              <span style={{ color: 'var(--color-warning)' }}>{configuredCount}</span> configured •
              <span style={{ color: 'var(--color-primary)' }}>{totalCount}</span> total
            </StatusSummary>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <Button $variant="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button
                $variant="primary"
                onClick={() => {
                  onSave(configs.map(c => ({ ...c, testnet: sandboxMode })));
                  onClose();
                }}
              >
                Save Configuration
              </Button>
            </div>
          </Footer>
        </ModalContainer>
      </Overlay>
    </AnimatePresence>
  );
};

export default APIConfigModal;
