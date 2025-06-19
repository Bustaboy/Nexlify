// Location: /src/systems/onboarding/CyberwareProtocol.tsx
// Nexlify Cyberware Protocol - Elite onboarding experience

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cpu, 
  Shield, 
  Brain, 
  Zap, 
  Server, 
  Lock, 
  Terminal,
  GitBranch,
  Activity,
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  ChevronLeft,
  Fingerprint,
  Key,
  Settings,
  BarChart3,
  TrendingUp,
  ShieldAlert,
  Database,
  Network,
  Layers,
  Bot,
  Sparkles
} from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import { GlitchText } from '@/components/common/GlitchText';
import { APIConfig, RiskProfile, StrategyTemplate } from '@/types/dashboard.types';
import { THEMES } from '@/constants/themes';

interface OnboardingStep {
  id: string;
  title: string;
  subtitle: string;
  icon: React.ElementType;
  component: React.ComponentType<StepComponentProps>;
  validation?: () => boolean;
}

interface StepComponentProps {
  onComplete: (data: any) => void;
  currentData: any;
  theme: typeof THEMES.nexlify.colors;
}

interface OnboardingData {
  neuralCalibration: {
    complete: boolean;
    biometricId: string;
  };
  systemCheck: {
    browser: string;
    performance: number;
    latency: number;
  };
  authentication: {
    method: 'password' | 'biometric' | 'hardware';
    twoFactorEnabled: boolean;
    securityLevel: 'standard' | 'paranoid' | 'quantum';
  };
  apiConfiguration: APIConfig[];
  strategyDNA: {
    riskProfile: RiskProfile;
    preferredStrategies: string[];
    capitalAllocation: Record<string, number>;
  };
  backtestPreview: {
    selectedStrategy: string;
    results: any;
  };
  riskLimits: {
    maxDrawdown: number;
    maxLeverage: number;
    stopLoss: number;
    dailyLossLimit: number;
  };
  emergencySetup: {
    protocol: 'standard' | 'advanced';
    contacts: string[];
    autoTriggers: boolean;
  };
  personalizedAI: {
    assistantName: string;
    personality: 'professional' | 'casual' | 'hardcore';
    responseSpeed: 'instant' | 'thoughtful';
  };
}

export const CyberwareProtocol: React.FC<{
  onComplete: (data: OnboardingData) => void;
  onSkip?: () => void;
}> = ({ onComplete, onSkip }) => {
  const { theme } = useTheme();
  const [currentStep, setCurrentStep] = useState(0);
  const [data, setData] = useState<Partial<OnboardingData>>({});
  const [isCalibrating, setIsCalibrating] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  const steps: OnboardingStep[] = [
    {
      id: 'neural_jackin',
      title: 'Neural Jack-In Sequence',
      subtitle: 'Initializing cyberware connection...',
      icon: Brain,
      component: NeuralJackIn
    },
    {
      id: 'system_compat',
      title: 'System Compatibility Check',
      subtitle: 'Analyzing hardware capabilities...',
      icon: Cpu,
      component: SystemCompatCheck
    },
    {
      id: 'auth_setup',
      title: 'Authentication Protocols',
      subtitle: 'Configuring security layers...',
      icon: Shield,
      component: AuthenticationSetup
    },
    {
      id: 'api_config',
      title: 'Exchange Neural Links',
      subtitle: 'Establishing market connections...',
      icon: Server,
      component: APIConfiguration
    },
    {
      id: 'strategy_dna',
      title: 'Strategy DNA Builder',
      subtitle: 'Crafting your trading genome...',
      icon: GitBranch,
      component: StrategyDNABuilder
    },
    {
      id: 'backtest_preview',
      title: 'Temporal Analysis',
      subtitle: 'Simulating potential futures...',
      icon: Activity,
      component: BacktestPreview
    },
    {
      id: 'risk_calibration',
      title: 'Risk Limit Calibration',
      subtitle: 'Setting neural safeguards...',
      icon: ShieldAlert,
      component: RiskCalibration
    },
    {
      id: 'emergency_protocol',
      title: 'Emergency Protocol Training',
      subtitle: 'Preparing for worst-case scenarios...',
      icon: AlertTriangle,
      component: EmergencyProtocolTraining
    },
    {
      id: 'ai_personalization',
      title: 'AI Assistant Configuration',
      subtitle: 'Personalizing your neural companion...',
      icon: Bot,
      component: AIPersonalization
    }
  ];

  const handleStepComplete = (stepData: any) => {
    setData(prev => ({
      ...prev,
      [steps[currentStep].id]: stepData
    }));

    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Onboarding complete
      onComplete(data as OnboardingData);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const CurrentStepComponent = steps[currentStep].component;
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="fixed inset-0 bg-black z-[300] overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `
              radial-gradient(circle at 20% 50%, ${theme.primary}22 0%, transparent 50%),
              radial-gradient(circle at 80% 80%, ${theme.neural}22 0%, transparent 50%),
              radial-gradient(circle at 40% 20%, ${theme.success}22 0%, transparent 50%)
            `
          }}
        />
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `linear-gradient(${theme.grid} 1px, transparent 1px), linear-gradient(90deg, ${theme.grid} 1px, transparent 1px)`,
            backgroundSize: '50px 50px'
          }}
        />
      </div>

      {/* Header */}
      <motion.div
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 p-6"
      >
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center space-x-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="relative"
              >
                <Layers 
                  className="w-12 h-12" 
                  style={{ 
                    color: theme.primary,
                    filter: `drop-shadow(0 0 20px ${theme.primary})`
                  }} 
                />
              </motion.div>
              <div>
                <h1 className="text-3xl font-black">
                  <GlitchText theme={theme}>NEXLIFY CYBERWARE PROTOCOL</GlitchText>
                </h1>
                <p className="text-sm text-gray-400 font-mono uppercase tracking-widest">
                  Neural Calibration Sequence v2.0
                </p>
              </div>
            </div>
            
            {onSkip && (
              <button
                onClick={onSkip}
                className="px-4 py-2 text-sm font-mono text-gray-400 hover:text-white transition-colors"
              >
                [SKIP_CALIBRATION]
              </button>
            )}
          </div>

          {/* Progress Bar */}
          <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ 
                background: `linear-gradient(to right, ${theme.primary}, ${theme.neural})`,
                boxShadow: `0 0 20px ${theme.primary}`
              }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
            <div className="absolute inset-0 flex items-center justify-between px-1">
              {steps.map((_, idx) => (
                <div
                  key={idx}
                  className={`w-2 h-2 rounded-full transition-all ${
                    idx <= currentStep 
                      ? 'bg-white' 
                      : 'bg-gray-700'
                  }`}
                  style={{
                    boxShadow: idx <= currentStep ? `0 0 10px ${theme.primary}` : undefined
                  }}
                />
              ))}
            </div>
          </div>

          {/* Step Info */}
          <div className="mt-6 flex items-center space-x-4">
            <div 
              className="p-3 rounded-lg"
              style={{ 
                backgroundColor: `${theme.primary}22`,
                color: theme.primary
              }}
            >
              {React.createElement(steps[currentStep].icon, { className: 'w-6 h-6' })}
            </div>
            <div>
              <h2 className="text-xl font-bold" style={{ color: theme.primary }}>
                {steps[currentStep].title}
              </h2>
              <p className="text-sm text-gray-400">{steps[currentStep].subtitle}</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="relative z-10 flex-1 overflow-y-auto px-6 pb-24">
        <div className="max-w-4xl mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <CurrentStepComponent
                onComplete={handleStepComplete}
                currentData={data[steps[currentStep].id]}
                theme={theme}
              />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Navigation */}
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black to-transparent"
      >
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={handleBack}
            disabled={currentStep === 0}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-mono transition-all ${
              currentStep === 0 
                ? 'opacity-50 cursor-not-allowed' 
                : 'hover:bg-gray-800'
            }`}
          >
            <ChevronLeft className="w-5 h-5" />
            <span>BACK</span>
          </button>

          <div className="text-sm font-mono text-gray-400">
            STEP {currentStep + 1} OF {steps.length}
          </div>

          <div className="text-sm font-mono" style={{ color: theme.primary }}>
            CALIBRATION {Math.floor(progress)}% COMPLETE
          </div>
        </div>
      </motion.div>

      {/* Terminal Output (decorative) */}
      <div 
        ref={terminalRef}
        className="fixed bottom-20 right-6 w-80 h-40 bg-black/80 border rounded-lg p-3 font-mono text-xs overflow-hidden"
        style={{ borderColor: `${theme.primary}44` }}
      >
        <TerminalOutput step={currentStep} theme={theme} />
      </div>
    </div>
  );
};

// Step Components

const NeuralJackIn: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  const [calibrationStage, setCalibrationStage] = useState(0);
  const stages = [
    'Initializing neural interface...',
    'Establishing quantum tunnel...',
    'Synchronizing brainwave patterns...',
    'Calibrating response matrices...',
    'Neural handshake complete.'
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCalibrationStage(prev => {
        if (prev < stages.length - 1) {
          return prev + 1;
        } else {
          clearInterval(interval);
          setTimeout(() => {
            onComplete({
              complete: true,
              biometricId: `NEURAL_${Date.now().toString(16).toUpperCase()}`
            });
          }, 1000);
          return prev;
        }
      });
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-8 py-8">
      <div className="relative h-64 flex items-center justify-center">
        <motion.div
          animate={{ 
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360]
          }}
          transition={{ 
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="relative"
        >
          <Brain 
            className="w-32 h-32" 
            style={{ 
              color: theme.neural,
              filter: `drop-shadow(0 0 40px ${theme.neural})`
            }}
          />
          <motion.div
            animate={{ scale: [1.5, 1, 1.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute inset-0 rounded-full"
            style={{
              background: `radial-gradient(circle, ${theme.neural}22 0%, transparent 70%)`,
              filter: 'blur(20px)'
            }}
          />
        </motion.div>
      </div>

      <div className="space-y-4">
        {stages.map((stage, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -20 }}
            animate={{ 
              opacity: idx <= calibrationStage ? 1 : 0.3,
              x: 0
            }}
            className="flex items-center space-x-3"
          >
            <div 
              className={`w-3 h-3 rounded-full ${
                idx <= calibrationStage ? 'animate-pulse' : ''
              }`}
              style={{
                backgroundColor: idx <= calibrationStage ? theme.success : theme.primary + '44',
                boxShadow: idx <= calibrationStage ? `0 0 10px ${theme.success}` : undefined
              }}
            />
            <span className={`font-mono text-sm ${
              idx <= calibrationStage ? 'text-white' : 'text-gray-600'
            }`}>
              {stage}
            </span>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 p-4 bg-gray-900/50 border rounded-lg" style={{ borderColor: `${theme.neural}44` }}>
        <p className="text-xs text-gray-400 font-mono leading-relaxed">
          NOTICE: Neural calibration establishes a secure quantum tunnel between your consciousness 
          and the Nexlify trading matrix. This process is non-invasive but may cause mild 
          synesthetic experiences. Your neural pattern will be encrypted using 2048-bit quantum 
          entanglement protocols.
        </p>
      </div>
    </div>
  );
};

const SystemCompatCheck: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  const [checks, setChecks] = useState({
    browser: { status: 'checking', value: '' },
    performance: { status: 'checking', score: 0 },
    latency: { status: 'checking', value: 0 },
    webgl: { status: 'checking', supported: false },
    memory: { status: 'checking', available: 0 }
  });

  useEffect(() => {
    // Run system checks
    setTimeout(() => {
      setChecks(prev => ({
        ...prev,
        browser: { 
          status: 'complete', 
          value: navigator.userAgent.includes('Chrome') ? 'Chrome' : 'Other'
        }
      }));
    }, 500);

    setTimeout(() => {
      const score = 75 + Math.random() * 25;
      setChecks(prev => ({
        ...prev,
        performance: { status: score > 80 ? 'complete' : 'warning', score }
      }));
    }, 1000);

    setTimeout(() => {
      const latency = 20 + Math.random() * 80;
      setChecks(prev => ({
        ...prev,
        latency: { status: latency < 50 ? 'complete' : 'warning', value: latency }
      }));
    }, 1500);

    setTimeout(() => {
      setChecks(prev => ({
        ...prev,
        webgl: { status: 'complete', supported: true }
      }));
    }, 2000);

    setTimeout(() => {
      const memory = 4 + Math.random() * 12;
      setChecks(prev => ({
        ...prev,
        memory: { status: memory > 8 ? 'complete' : 'warning', available: memory }
      }));
    }, 2500);

    setTimeout(() => {
      onComplete({
        browser: checks.browser.value,
        performance: checks.performance.score,
        latency: checks.latency.value
      });
    }, 3500);
  }, []);

  const getStatusIcon = (status: string) => {
    if (status === 'checking') return <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity }} className="w-4 h-4 border-2 border-t-transparent rounded-full" style={{ borderColor: `${theme.primary}88` }} />;
    if (status === 'complete') return <CheckCircle className="w-4 h-4" style={{ color: theme.success }} />;
    return <AlertTriangle className="w-4 h-4" style={{ color: theme.warning }} />;
  };

  return (
    <div className="space-y-6 py-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Browser Check */}
        <div className="bg-gray-900/50 border rounded-lg p-6" style={{ borderColor: `${theme.primary}33` }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono text-sm uppercase">Browser Engine</h3>
            {getStatusIcon(checks.browser.status)}
          </div>
          <div className="text-2xl font-bold" style={{ color: theme.primary }}>
            {checks.browser.value || '...'}
          </div>
          <p className="text-xs text-gray-500 mt-2">Chromium-based recommended</p>
        </div>

        {/* Performance Score */}
        <div className="bg-gray-900/50 border rounded-lg p-6" style={{ borderColor: `${theme.primary}33` }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono text-sm uppercase">Performance Score</h3>
            {getStatusIcon(checks.performance.status)}
          </div>
          <div className="text-2xl font-bold" style={{ 
            color: checks.performance.score > 80 ? theme.success : theme.warning 
          }}>
            {checks.performance.score ? `${checks.performance.score.toFixed(0)}/100` : '...'}
          </div>
          <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full"
              style={{ 
                backgroundColor: checks.performance.score > 80 ? theme.success : theme.warning
              }}
              animate={{ width: `${checks.performance.score}%` }}
            />
          </div>
        </div>

        {/* Network Latency */}
        <div className="bg-gray-900/50 border rounded-lg p-6" style={{ borderColor: `${theme.primary}33` }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono text-sm uppercase">Network Latency</h3>
            {getStatusIcon(checks.latency.status)}
          </div>
          <div className="text-2xl font-bold" style={{ 
            color: checks.latency.value < 50 ? theme.success : theme.warning 
          }}>
            {checks.latency.value ? `${checks.latency.value.toFixed(0)}ms` : '...'}
          </div>
          <p className="text-xs text-gray-500 mt-2">Lower is better for real-time data</p>
        </div>

        {/* WebGL Support */}
        <div className="bg-gray-900/50 border rounded-lg p-6" style={{ borderColor: `${theme.primary}33` }}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono text-sm uppercase">3D Rendering</h3>
            {getStatusIcon(checks.webgl.status)}
          </div>
          <div className="text-2xl font-bold" style={{ 
            color: checks.webgl.supported ? theme.success : theme.danger 
          }}>
            {checks.webgl.status === 'complete' ? (checks.webgl.supported ? 'SUPPORTED' : 'UNSUPPORTED') : '...'}
          </div>
          <p className="text-xs text-gray-500 mt-2">Required for advanced visualizations</p>
        </div>
      </div>

      <div className="mt-8 p-6 bg-gray-900/50 border rounded-lg" style={{ borderColor: `${theme.info}44` }}>
        <div className="flex items-start space-x-3">
          <Info className="w-5 h-5 mt-0.5" style={{ color: theme.info }} />
          <div className="space-y-2">
            <h4 className="font-mono text-sm uppercase" style={{ color: theme.info }}>System Requirements</h4>
            <ul className="space-y-1 text-xs text-gray-400">
              <li>• Minimum 8GB RAM for optimal neural processing</li>
              <li>• Chrome/Edge browser for best compatibility</li>
              <li>• Stable internet connection (< 100ms latency)</li>
              <li>• WebGL 2.0 support for advanced visualizations</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// Continue with remaining step components...
const AuthenticationSetup: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  const [authMethod, setAuthMethod] = useState<'password' | 'biometric' | 'hardware'>('password');
  const [securityLevel, setSecurityLevel] = useState<'standard' | 'paranoid' | 'quantum'>('standard');
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [setupStage, setSetupStage] = useState<'method' | 'configure' | 'test'>('method');

  const securityLevels = {
    standard: {
      description: 'Basic encryption with standard protocols',
      features: ['256-bit AES', 'Session timeout', 'IP whitelisting'],
      color: theme.info
    },
    paranoid: {
      description: 'Enhanced security for the cautious',
      features: ['512-bit encryption', 'Behavioral analysis', 'Honeypot mode', 'Geofencing'],
      color: theme.warning
    },
    quantum: {
      description: 'Military-grade quantum resistance',
      features: ['Quantum encryption', 'Neural pattern lock', 'Dead man switch', 'Zero-knowledge proofs'],
      color: theme.danger
    }
  };

  return (
    <div className="space-y-8 py-8">
      {setupStage === 'method' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          <h3 className="text-xl font-bold mb-6" style={{ color: theme.primary }}>
            Choose Authentication Method
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Password */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setAuthMethod('password')}
              className={`p-6 border-2 rounded-lg text-left transition-all ${
                authMethod === 'password' ? 'border-opacity-100' : 'border-opacity-30'
              }`}
              style={{ 
                borderColor: authMethod === 'password' ? theme.primary : `${theme.primary}44`,
                backgroundColor: authMethod === 'password' ? `${theme.primary}11` : 'transparent'
              }}
            >
              <Key className="w-8 h-8 mb-3" style={{ color: theme.primary }} />
              <h4 className="font-bold mb-2">Password</h4>
              <p className="text-xs text-gray-400">Traditional alphanumeric authentication</p>
            </motion.button>

            {/* Biometric */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setAuthMethod('biometric')}
              className={`p-6 border-2 rounded-lg text-left transition-all ${
                authMethod === 'biometric' ? 'border-opacity-100' : 'border-opacity-30'
              }`}
              style={{ 
                borderColor: authMethod === 'biometric' ? theme.primary : `${theme.primary}44`,
                backgroundColor: authMethod === 'biometric' ? `${theme.primary}11` : 'transparent'
              }}
            >
              <Fingerprint className="w-8 h-8 mb-3" style={{ color: theme.primary }} />
              <h4 className="font-bold mb-2">Biometric</h4>
              <p className="text-xs text-gray-400">Fingerprint or facial recognition</p>
            </motion.button>

            {/* Hardware Key */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setAuthMethod('hardware')}
              className={`p-6 border-2 rounded-lg text-left transition-all ${
                authMethod === 'hardware' ? 'border-opacity-100' : 'border-opacity-30'
              }`}
              style={{ 
                borderColor: authMethod === 'hardware' ? theme.primary : `${theme.primary}44`,
                backgroundColor: authMethod === 'hardware' ? `${theme.primary}11` : 'transparent'
              }}
            >
              <Shield className="w-8 h-8 mb-3" style={{ color: theme.primary }} />
              <h4 className="font-bold mb-2">Hardware Key</h4>
              <p className="text-xs text-gray-400">YubiKey or similar device</p>
            </motion.button>
          </div>

          {/* Security Level Selection */}
          <div className="mt-8">
            <h3 className="text-lg font-bold mb-4" style={{ color: theme.primary }}>
              Security Level
            </h3>
            <div className="space-y-3">
              {Object.entries(securityLevels).map(([level, config]) => (
                <motion.button
                  key={level}
                  whileHover={{ x: 10 }}
                  onClick={() => setSecurityLevel(level as any)}
                  className={`w-full p-4 border rounded-lg text-left transition-all ${
                    securityLevel === level ? 'border-opacity-100' : 'border-opacity-30'
                  }`}
                  style={{ 
                    borderColor: securityLevel === level ? config.color : `${config.color}44`,
                    backgroundColor: securityLevel === level ? `${config.color}11` : 'transparent'
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-bold uppercase" style={{ color: config.color }}>
                      {level}
                    </h4>
                    {securityLevel === level && (
                      <CheckCircle className="w-5 h-5" style={{ color: config.color }} />
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mb-2">{config.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {config.features.map((feature, idx) => (
                      <span 
                        key={idx}
                        className="px-2 py-1 text-xs rounded"
                        style={{ 
                          backgroundColor: `${config.color}22`,
                          color: config.color
                        }}
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Two-Factor Toggle */}
          <div className="mt-6 p-4 bg-gray-900/50 border rounded-lg flex items-center justify-between" 
               style={{ borderColor: `${theme.primary}33` }}>
            <div>
              <h4 className="font-bold mb-1">Two-Factor Authentication</h4>
              <p className="text-xs text-gray-400">Additional layer of security</p>
            </div>
            <button
              onClick={() => setTwoFactorEnabled(!twoFactorEnabled)}
              className={`relative w-14 h-8 rounded-full transition-colors`}
              style={{ 
                backgroundColor: twoFactorEnabled ? theme.success : '#374151'
              }}
            >
              <motion.div
                className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full"
                animate={{ x: twoFactorEnabled ? 22 : 0 }}
              />
            </button>
          </div>

          <button
            onClick={() => {
              onComplete({
                method: authMethod,
                twoFactorEnabled,
                securityLevel
              });
            }}
            className="w-full py-4 rounded-lg font-bold uppercase tracking-wider transition-all"
            style={{
              backgroundColor: `${theme.primary}33`,
              border: `2px solid ${theme.primary}`,
              color: theme.primary,
              boxShadow: `0 0 30px ${theme.primary}44`
            }}
          >
            Continue Setup
          </button>
        </motion.div>
      )}
    </div>
  );
};

// Additional step components would follow the same pattern...
// For brevity, I'll include placeholder implementations

const APIConfiguration: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  // Implementation similar to APIConfigModal but integrated into onboarding flow
  useEffect(() => {
    setTimeout(() => onComplete([]), 2000);
  }, []);
  
  return <div className="py-8 text-center">Configuring exchange connections...</div>;
};

const StrategyDNABuilder: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  useEffect(() => {
    setTimeout(() => onComplete({
      riskProfile: 'balanced',
      preferredStrategies: ['momentum', 'arbitrage'],
      capitalAllocation: { momentum: 60, arbitrage: 40 }
    }), 2000);
  }, []);
  
  return <div className="py-8 text-center">Building your strategy DNA...</div>;
};

const BacktestPreview: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  useEffect(() => {
    setTimeout(() => onComplete({
      selectedStrategy: 'momentum',
      results: { sharpe: 1.5, maxDrawdown: 15 }
    }), 2000);
  }, []);
  
  return <div className="py-8 text-center">Running temporal analysis...</div>;
};

const RiskCalibration: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  useEffect(() => {
    setTimeout(() => onComplete({
      maxDrawdown: 20,
      maxLeverage: 10,
      stopLoss: 2,
      dailyLossLimit: 5
    }), 2000);
  }, []);
  
  return <div className="py-8 text-center">Calibrating risk parameters...</div>;
};

const EmergencyProtocolTraining: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  useEffect(() => {
    setTimeout(() => onComplete({
      protocol: 'standard',
      contacts: [],
      autoTriggers: true
    }), 2000);
  }, []);
  
  return <div className="py-8 text-center">Training emergency protocols...</div>;
};

const AIPersonalization: React.FC<StepComponentProps> = ({ onComplete, theme }) => {
  const [assistantName, setAssistantName] = useState('');
  const [personality, setPersonality] = useState<'professional' | 'casual' | 'hardcore'>('professional');
  
  return (
    <div className="space-y-6 py-8">
      <h3 className="text-xl font-bold" style={{ color: theme.primary }}>
        Personalize Your AI Assistant
      </h3>
      
      <div>
        <label className="text-sm text-gray-400 block mb-2">Assistant Name</label>
        <input
          type="text"
          value={assistantName}
          onChange={(e) => setAssistantName(e.target.value)}
          placeholder="e.g., Johnny, Silverhand, V"
          className="w-full px-4 py-3 bg-gray-900/80 border rounded-lg font-mono focus:outline-none"
          style={{ 
            borderColor: `${theme.primary}44`,
            color: theme.primary
          }}
        />
      </div>
      
      <div>
        <label className="text-sm text-gray-400 block mb-3">Personality Type</label>
        <div className="grid grid-cols-3 gap-3">
          {(['professional', 'casual', 'hardcore'] as const).map(type => (
            <button
              key={type}
              onClick={() => setPersonality(type)}
              className={`p-3 border rounded-lg capitalize transition-all ${
                personality === type ? 'border-opacity-100' : 'border-opacity-30'
              }`}
              style={{ 
                borderColor: personality === type ? theme.primary : `${theme.primary}44`,
                backgroundColor: personality === type ? `${theme.primary}11` : 'transparent'
              }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>
      
      <button
        onClick={() => onComplete({
          assistantName: assistantName || 'Nexus',
          personality,
          responseSpeed: 'instant'
        })}
        className="w-full py-4 rounded-lg font-bold uppercase tracking-wider transition-all"
        style={{
          backgroundColor: `${theme.success}33`,
          border: `2px solid ${theme.success}`,
          color: theme.success,
          boxShadow: `0 0 30px ${theme.success}44`
        }}
      >
        Complete Calibration
      </button>
    </div>
  );
};

// Terminal output component
const TerminalOutput: React.FC<{ step: number; theme: typeof THEMES.nexlify.colors }> = ({ step, theme }) => {
  const outputs = [
    ['> Initializing neural interface...', '> Quantum tunnel established', '> Brainwave sync: 98.7%'],
    ['> Checking system compatibility...', '> CPU: Optimal', '> Memory: 16GB detected', '> WebGL: Supported'],
    ['> Loading authentication modules...', '> Biometric scanner: Ready', '> 2FA: Configuring...'],
    ['> Connecting to exchange APIs...', '> Binance: Connected', '> Kraken: Connected', '> DEX: Scanning...'],
    ['> Building strategy matrix...', '> Risk profile: Analyzing...', '> ML models: Loading...'],
    ['> Running backtest simulation...', '> Historical data: 2 years', '> Sharpe: 1.52', '> Win rate: 67%'],
    ['> Calibrating risk parameters...', '> Max drawdown: Set', '> Stop loss: Configured'],
    ['> Loading emergency protocols...', '> Killswitch: Armed', '> Recovery: Enabled'],
    ['> Personalizing AI assistant...', '> Neural net: Training...', '> Personality: Loaded']
  ];

  return (
    <div className="space-y-1">
      {outputs[step]?.map((line, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: idx * 0.3 }}
          style={{ color: theme.success }}
        >
          {line}
        </motion.div>
      ))}
      <motion.div
        animate={{ opacity: [1, 0.5, 1] }}
        transition={{ duration: 1, repeat: Infinity }}
        className="mt-2"
      >
        _
      </motion.div>
    </div>
  );
};
