// Location: /src/components/security/NeuralLock.tsx
// Nexlify Neural Lock - Advanced security with 2FA, hardware key, and remote wipe framework

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Lock,
  Unlock,
  Shield,
  Fingerprint,
  Key,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Smartphone,
  Cpu,
  Eye,
  EyeOff,
  RefreshCw,
  AlertOctagon,
  Zap,
  Network,
  ShieldAlert,
  Terminal,
  Trash2,
  HardDrive,
  Wifi,
  WifiOff,
  Clock,
  Activity
} from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import { GlitchText } from '@/components/common/GlitchText';
import { authenticateUser, verify2FA, checkHardwareKey } from '@/services/auth.service';
import { THEMES } from '@/constants/themes';

interface NeuralLockProps {
  onUnlock: (sessionToken: string) => void;
  onEmergencyWipe?: () => void;
  failedAttempts?: number;
  lastActivity?: number;
  isLocked: boolean;
}

interface SecurityEvent {
  id: string;
  type: 'login_attempt' | 'intrusion_detected' | 'anomaly' | 'hardware_key' | '2fa';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: number;
  location?: string;
}

interface RemoteWipeConfig {
  enabled: boolean;
  triggers: {
    failedAttempts: number;
    geoFence: boolean;
    panicPhrase: string;
    deadManSwitch: boolean;
    deadManTimeout: number; // hours
  };
  wipeLevel: 'soft' | 'hard' | 'quantum';
  confirmationRequired: boolean;
}

export const NeuralLock: React.FC<NeuralLockProps> = ({
  onUnlock,
  onEmergencyWipe,
  failedAttempts = 0,
  lastActivity,
  isLocked
}) => {
  const { theme } = useTheme();
  const [authMethod, setAuthMethod] = useState<'password' | 'biometric' | 'hardware'>('password');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  
  // 2FA State
  const [requires2FA, setRequires2FA] = useState(false);
  const [twoFactorCode, setTwoFactorCode] = useState('');
  const [twoFactorMethod, setTwoFactorMethod] = useState<'app' | 'sms' | 'email'>('app');
  
  // Hardware Key State
  const [hardwareKeyDetected, setHardwareKeyDetected] = useState(false);
  const [hardwareKeyStatus, setHardwareKeyStatus] = useState<'waiting' | 'verifying' | 'success' | 'failed'>('waiting');
  
  // Security Events
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [intrusionDetected, setIntrusionDetected] = useState(false);
  
  // Remote Wipe Framework
  const [remoteWipeConfig] = useState<RemoteWipeConfig>({
    enabled: true,
    triggers: {
      failedAttempts: 5,
      geoFence: true,
      panicPhrase: 'DELETE_EVERYTHING_NOW',
      deadManSwitch: true,
      deadManTimeout: 24
    },
    wipeLevel: 'hard',
    confirmationRequired: true
  });
  
  // Pattern Lock State
  const [showPatternLock, setShowPatternLock] = useState(false);
  const [patternPoints, setPatternPoints] = useState<number[]>([]);
  const patternRef = useRef<HTMLDivElement>(null);
  
  // Biometric simulation
  const [biometricScanning, setBiometricScanning] = useState(false);
  
  // Session timer
  const [sessionTime, setSessionTime] = useState(0);
  
  useEffect(() => {
    if (!isLocked) return;
    
    const timer = setInterval(() => {
      setSessionTime(prev => prev + 1);
    }, 1000);
    
    return () => clearInterval(timer);
  }, [isLocked]);
  
  // Monitor failed attempts for remote wipe
  useEffect(() => {
    if (failedAttempts >= remoteWipeConfig.triggers.failedAttempts && remoteWipeConfig.enabled) {
      triggerEmergencyProtocol('Too many failed attempts');
    }
  }, [failedAttempts]);
  
  // Hardware key detection
  useEffect(() => {
    if (authMethod !== 'hardware') return;
    
    // Simulate hardware key detection
    const detectKey = async () => {
      try {
        // In production, this would use WebAuthn API
        const devices = await navigator.usb?.getDevices?.() || [];
        setHardwareKeyDetected(devices.length > 0);
      } catch (error) {
        console.error('Hardware key detection failed:', error);
      }
    };
    
    detectKey();
    const interval = setInterval(detectKey, 2000);
    
    return () => clearInterval(interval);
  }, [authMethod]);
  
  const addSecurityEvent = (event: Omit<SecurityEvent, 'id' | 'timestamp'>) => {
    setSecurityEvents(prev => [{
      ...event,
      id: `event_${Date.now()}`,
      timestamp: Date.now()
    }, ...prev].slice(0, 10));
  };
  
  const handleAuthentication = async () => {
    setIsAuthenticating(true);
    setAuthError(null);
    
    try {
      addSecurityEvent({
        type: 'login_attempt',
        severity: 'info',
        message: `Authentication attempt via ${authMethod}`,
        location: 'Rotterdam, NL'
      });
      
      let authenticated = false;
      
      switch (authMethod) {
        case 'password':
          authenticated = await authenticateUser(password);
          break;
          
        case 'biometric':
          setBiometricScanning(true);
          await new Promise(resolve => setTimeout(resolve, 2000));
          authenticated = Math.random() > 0.1; // Simulate 90% success rate
          setBiometricScanning(false);
          break;
          
        case 'hardware':
          if (hardwareKeyDetected) {
            setHardwareKeyStatus('verifying');
            authenticated = await checkHardwareKey();
            setHardwareKeyStatus(authenticated ? 'success' : 'failed');
          }
          break;
      }
      
      if (authenticated) {
        // Check if 2FA is required
        if (Math.random() > 0.5) { // Simulate 2FA requirement
          setRequires2FA(true);
          addSecurityEvent({
            type: '2fa',
            severity: 'info',
            message: '2FA verification required'
          });
        } else {
          completeAuthentication();
        }
      } else {
        throw new Error('Authentication failed');
      }
    } catch (error) {
      setAuthError('Neural handshake failed. Access denied.');
      addSecurityEvent({
        type: 'login_attempt',
        severity: 'warning',
        message: 'Failed authentication attempt',
        location: 'Rotterdam, NL'
      });
      
      // Check for intrusion
      if (failedAttempts > 2) {
        setIntrusionDetected(true);
        addSecurityEvent({
          type: 'intrusion_detected',
          severity: 'critical',
          message: 'Potential intrusion detected - multiple failed attempts'
        });
      }
    } finally {
      setIsAuthenticating(false);
    }
  };
  
  const handle2FAVerification = async () => {
    setIsAuthenticating(true);
    
    try {
      const verified = await verify2FA(twoFactorCode, twoFactorMethod);
      
      if (verified) {
        addSecurityEvent({
          type: '2fa',
          severity: 'info',
          message: '2FA verification successful'
        });
        completeAuthentication();
      } else {
        throw new Error('Invalid 2FA code');
      }
    } catch (error) {
      setAuthError('2FA verification failed');
      addSecurityEvent({
        type: '2fa',
        severity: 'warning',
        message: '2FA verification failed'
      });
    } finally {
      setIsAuthenticating(false);
    }
  };
  
  const completeAuthentication = () => {
    const sessionToken = `SESSION_${Date.now().toString(16).toUpperCase()}`;
    onUnlock(sessionToken);
    
    addSecurityEvent({
      type: 'login_attempt',
      severity: 'info',
      message: 'Authentication successful',
      location: 'Rotterdam, NL'
    });
  };
  
  const triggerEmergencyProtocol = (reason: string) => {
    addSecurityEvent({
      type: 'anomaly',
      severity: 'critical',
      message: `EMERGENCY PROTOCOL TRIGGERED: ${reason}`
    });
    
    if (onEmergencyWipe && remoteWipeConfig.confirmationRequired) {
      // In production, this would show a confirmation dialog
      onEmergencyWipe();
    }
  };
  
  const formatSessionTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  if (!isLocked) return null;
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black z-[400] overflow-hidden"
    >
      {/* Background Matrix Effect */}
      <div className="absolute inset-0 opacity-20">
        <MatrixRain theme={theme} />
      </div>
      
      {/* Scan Lines Effect */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          background: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            ${theme.primary}22 2px,
            ${theme.primary}22 4px
          )`
        }}
      />
      
      {/* Main Content */}
      <div className="relative z-10 h-full flex items-center justify-center p-6">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="w-full max-w-md"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              animate={{ rotate: intrusionDetected ? [0, -5, 5, -5, 5, 0] : 0 }}
              transition={{ duration: 0.5, repeat: intrusionDetected ? Infinity : 0 }}
              className="inline-block mb-4"
            >
              <div className="relative">
                <Shield 
                  className="w-24 h-24 mx-auto" 
                  style={{ 
                    color: intrusionDetected ? theme.danger : theme.primary,
                    filter: `drop-shadow(0 0 40px ${intrusionDetected ? theme.danger : theme.primary})`
                  }}
                />
                {intrusionDetected && (
                  <AlertOctagon 
                    className="absolute -top-2 -right-2 w-8 h-8 animate-pulse" 
                    style={{ color: theme.danger }}
                  />
                )}
              </div>
            </motion.div>
            
            <h1 className="text-3xl font-black mb-2">
              <GlitchText theme={theme}>
                {intrusionDetected ? 'SECURITY BREACH DETECTED' : 'NEURAL LOCK ENGAGED'}
              </GlitchText>
            </h1>
            
            <p className="text-sm text-gray-400 font-mono">
              {lastActivity ? (
                <>Last activity: {new Date(lastActivity).toLocaleTimeString()}</>
              ) : (
                <>Session time: {formatSessionTime(sessionTime)}</>
              )}
            </p>
            
            {failedAttempts > 0 && (
              <p className="text-sm mt-2 font-mono" style={{ color: theme.warning }}>
                Failed attempts: {failedAttempts} / {remoteWipeConfig.triggers.failedAttempts}
              </p>
            )}
          </div>
          
          {/* Authentication Methods */}
          {!requires2FA ? (
            <div className="space-y-6">
              {/* Method Selector */}
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => setAuthMethod('password')}
                  className={`p-3 border-2 rounded-lg transition-all ${
                    authMethod === 'password' ? 'border-opacity-100' : 'border-opacity-30'
                  }`}
                  style={{
                    borderColor: authMethod === 'password' ? theme.primary : `${theme.primary}44`,
                    backgroundColor: authMethod === 'password' ? `${theme.primary}11` : 'transparent'
                  }}
                >
                  <Key className="w-6 h-6 mx-auto" style={{ color: theme.primary }} />
                </button>
                
                <button
                  onClick={() => setAuthMethod('biometric')}
                  className={`p-3 border-2 rounded-lg transition-all ${
                    authMethod === 'biometric' ? 'border-opacity-100' : 'border-opacity-30'
                  }`}
                  style={{
                    borderColor: authMethod === 'biometric' ? theme.primary : `${theme.primary}44`,
                    backgroundColor: authMethod === 'biometric' ? `${theme.primary}11` : 'transparent'
                  }}
                >
                  <Fingerprint className="w-6 h-6 mx-auto" style={{ color: theme.primary }} />
                </button>
                
                <button
                  onClick={() => setAuthMethod('hardware')}
                  className={`p-3 border-2 rounded-lg transition-all ${
                    authMethod === 'hardware' ? 'border-opacity-100' : 'border-opacity-30'
                  }`}
                  style={{
                    borderColor: authMethod === 'hardware' ? theme.primary : `${theme.primary}44`,
                    backgroundColor: authMethod === 'hardware' ? `${theme.primary}11` : 'transparent'
                  }}
                >
                  <Cpu className="w-6 h-6 mx-auto" style={{ color: theme.primary }} />
                </button>
              </div>
              
              {/* Authentication Input */}
              <AnimatePresence mode="wait">
                {authMethod === 'password' && (
                  <motion.div
                    key="password"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-4"
                  >
                    <div className="relative">
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleAuthentication()}
                        placeholder="Enter neural passphrase"
                        className="w-full px-4 py-3 bg-gray-900/80 border-2 rounded-lg pr-12 font-mono focus:outline-none transition-colors"
                        style={{
                          borderColor: authError ? theme.danger : `${theme.primary}66`,
                          color: theme.accent
                        }}
                      />
                      <button
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                      >
                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                    
                    <button
                      onClick={() => setShowPatternLock(!showPatternLock)}
                      className="text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      Use pattern lock instead
                    </button>
                  </motion.div>
                )}
                
                {authMethod === 'biometric' && (
                  <motion.div
                    key="biometric"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="text-center py-8"
                  >
                    <div className="relative inline-block">
                      <Fingerprint 
                        className={`w-32 h-32 ${biometricScanning ? 'animate-pulse' : ''}`}
                        style={{ 
                          color: theme.primary,
                          filter: `drop-shadow(0 0 30px ${theme.primary})`
                        }}
                      />
                      {biometricScanning && (
                        <motion.div
                          className="absolute inset-0 rounded-full"
                          animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                          style={{ backgroundColor: theme.primary }}
                        />
                      )}
                    </div>
                    <p className="mt-4 text-sm text-gray-400">
                      {biometricScanning ? 'Scanning biometric signature...' : 'Touch sensor to authenticate'}
                    </p>
                  </motion.div>
                )}
                
                {authMethod === 'hardware' && (
                  <motion.div
                    key="hardware"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="text-center py-8 space-y-4"
                  >
                    <div className="flex items-center justify-center space-x-4">
                      <Cpu 
                        className="w-16 h-16"
                        style={{ 
                          color: hardwareKeyDetected ? theme.success : theme.warning,
                          filter: `drop-shadow(0 0 20px ${hardwareKeyDetected ? theme.success : theme.warning})`
                        }}
                      />
                      <div className="text-left">
                        <p className="font-bold">
                          {hardwareKeyDetected ? 'Hardware Key Detected' : 'Insert Hardware Key'}
                        </p>
                        <p className="text-sm text-gray-400">
                          {hardwareKeyStatus === 'verifying' ? 'Verifying...' : 'YubiKey or compatible device'}
                        </p>
                      </div>
                    </div>
                    
                    {hardwareKeyStatus === 'failed' && (
                      <p className="text-sm" style={{ color: theme.danger }}>
                        Hardware key verification failed
                      </p>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
              
              {/* Error Message */}
              {authError && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg"
                >
                  <p className="text-sm text-red-400 font-mono">{authError}</p>
                </motion.div>
              )}
              
              {/* Authenticate Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAuthentication}
                disabled={isAuthenticating || (authMethod === 'password' && !password)}
                className="w-full py-4 rounded-lg font-bold uppercase tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  backgroundColor: intrusionDetected ? `${theme.danger}33` : `${theme.primary}33`,
                  border: `2px solid ${intrusionDetected ? theme.danger : theme.primary}`,
                  color: intrusionDetected ? theme.danger : theme.primary,
                  boxShadow: `0 0 30px ${intrusionDetected ? theme.danger : theme.primary}44`
                }}
              >
                {isAuthenticating ? (
                  <div className="flex items-center justify-center space-x-2">
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Authenticating...</span>
                  </div>
                ) : (
                  'Authenticate'
                )}
              </motion.button>
            </div>
          ) : (
            // 2FA Verification
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-6"
            >
              <div className="text-center">
                <Smartphone 
                  className="w-16 h-16 mx-auto mb-4" 
                  style={{ 
                    color: theme.primary,
                    filter: `drop-shadow(0 0 20px ${theme.primary})`
                  }}
                />
                <h3 className="text-xl font-bold mb-2">Two-Factor Authentication</h3>
                <p className="text-sm text-gray-400">
                  Enter the code from your authenticator app
                </p>
              </div>
              
              <div className="flex justify-center space-x-2">
                {[...Array(6)].map((_, idx) => (
                  <input
                    key={idx}
                    type="text"
                    maxLength={1}
                    value={twoFactorCode[idx] || ''}
                    onChange={(e) => {
                      const newCode = twoFactorCode.split('');
                      newCode[idx] = e.target.value;
                      setTwoFactorCode(newCode.join(''));
                      
                      // Auto-focus next input
                      if (e.target.value && idx < 5) {
                        const nextInput = e.target.nextSibling as HTMLInputElement;
                        nextInput?.focus();
                      }
                    }}
                    className="w-12 h-12 text-center bg-gray-900/80 border-2 rounded-lg font-mono text-lg focus:outline-none"
                    style={{
                      borderColor: `${theme.primary}66`,
                      color: theme.primary
                    }}
                  />
                ))}
              </div>
              
              <div className="flex justify-center space-x-4 text-sm">
                {(['app', 'sms', 'email'] as const).map(method => (
                  <button
                    key={method}
                    onClick={() => setTwoFactorMethod(method)}
                    className={`px-3 py-1 rounded transition-colors ${
                      twoFactorMethod === method 
                        ? 'text-white' 
                        : 'text-gray-400 hover:text-gray-300'
                    }`}
                    style={{
                      backgroundColor: twoFactorMethod === method ? `${theme.primary}33` : 'transparent'
                    }}
                  >
                    {method.toUpperCase()}
                  </button>
                ))}
              </div>
              
              <button
                onClick={handle2FAVerification}
                disabled={twoFactorCode.length !== 6 || isAuthenticating}
                className="w-full py-4 rounded-lg font-bold uppercase tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  backgroundColor: `${theme.primary}33`,
                  border: `2px solid ${theme.primary}`,
                  color: theme.primary,
                  boxShadow: `0 0 30px ${theme.primary}44`
                }}
              >
                Verify
              </button>
            </motion.div>
          )}
          
          {/* Security Events Log */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-8 p-4 bg-gray-900/50 border rounded-lg max-h-40 overflow-y-auto"
            style={{ borderColor: `${theme.primary}33` }}
          >
            <h4 className="text-xs font-mono uppercase mb-2 text-gray-400">Security Log</h4>
            <div className="space-y-1">
              {securityEvents.map(event => (
                <div 
                  key={event.id} 
                  className="text-xs font-mono flex items-center space-x-2"
                  style={{
                    color: event.severity === 'critical' ? theme.danger :
                           event.severity === 'warning' ? theme.warning :
                           theme.success
                  }}
                >
                  <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                  <span>-</span>
                  <span>{event.message}</span>
                </div>
              ))}
            </div>
          </motion.div>
          
          {/* Remote Wipe Indicator */}
          {remoteWipeConfig.enabled && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="mt-4 flex items-center justify-center space-x-2 text-xs text-gray-500"
            >
              <Trash2 className="w-3 h-3" />
              <span>Remote wipe {remoteWipeConfig.wipeLevel} mode active</span>
            </motion.div>
          )}
        </motion.div>
      </div>
      
      {/* Pattern Lock Overlay */}
      <AnimatePresence>
        {showPatternLock && (
          <PatternLockOverlay
            theme={theme}
            onComplete={(pattern) => {
              setPassword(pattern.join(''));
              setShowPatternLock(false);
              handleAuthentication();
            }}
            onCancel={() => setShowPatternLock(false)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Pattern Lock Component
const PatternLockOverlay: React.FC<{
  theme: typeof THEMES.nexlify.colors;
  onComplete: (pattern: number[]) => void;
  onCancel: () => void;
}> = ({ theme, onComplete, onCancel }) => {
  const [pattern, setPattern] = useState<number[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  
  const handlePointSelect = (point: number) => {
    if (!pattern.includes(point)) {
      setPattern([...pattern, point]);
    }
  };
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center"
      onClick={onCancel}
    >
      <motion.div
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.8 }}
        className="bg-gray-900 border-2 rounded-xl p-8"
        style={{ borderColor: theme.primary }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-xl font-bold mb-6 text-center" style={{ color: theme.primary }}>
          Draw Pattern
        </h3>
        
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[...Array(9)].map((_, idx) => (
            <button
              key={idx}
              onMouseDown={() => {
                setIsDrawing(true);
                handlePointSelect(idx);
              }}
              onMouseEnter={() => {
                if (isDrawing) handlePointSelect(idx);
              }}
              onMouseUp={() => {
                setIsDrawing(false);
                if (pattern.length >= 4) {
                  onComplete(pattern);
                }
              }}
              className={`w-16 h-16 rounded-full border-2 transition-all ${
                pattern.includes(idx) 
                  ? 'scale-90' 
                  : 'hover:scale-110'
              }`}
              style={{
                borderColor: pattern.includes(idx) ? theme.primary : `${theme.primary}44`,
                backgroundColor: pattern.includes(idx) ? `${theme.primary}33` : 'transparent',
                boxShadow: pattern.includes(idx) ? `0 0 20px ${theme.primary}` : undefined
              }}
            >
              <div 
                className="w-2 h-2 rounded-full mx-auto"
                style={{ backgroundColor: pattern.includes(idx) ? theme.primary : `${theme.primary}66` }}
              />
            </button>
          ))}
        </div>
        
        <p className="text-center text-sm text-gray-400">
          Connect at least 4 points
        </p>
      </motion.div>
    </motion.div>
  );
};

// Matrix Rain Effect Component
const MatrixRain: React.FC<{ theme: typeof THEMES.nexlify.colors }> = ({ theme }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const columns = Math.floor(canvas.width / 20);
    const drops: number[] = new Array(columns).fill(1);
    
    const matrix = 'NEXLIFY01サイバーパンク2077ネオン街CHROME';
    const matrixArray = matrix.split('');
    
    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.fillStyle = theme.primary;
      ctx.font = '15px monospace';
      
      for (let i = 0; i < drops.length; i++) {
        const text = matrixArray[Math.floor(Math.random() * matrixArray.length)];
        ctx.fillText(text, i * 20, drops[i] * 20);
        
        if (drops[i] * 20 > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        
        drops[i]++;
      }
    };
    
    const interval = setInterval(draw, 35);
    
    return () => clearInterval(interval);
  }, [theme]);
  
  return <canvas ref={canvasRef} className="absolute inset-0" />;
};
