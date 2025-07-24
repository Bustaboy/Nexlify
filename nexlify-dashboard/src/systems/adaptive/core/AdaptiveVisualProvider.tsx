// Location: nexlify-dashboard/src/systems/adaptive/core/AdaptiveVisualProvider.tsx
// Mission: 80-I.1 Core Visual System Provider
// Dependencies: All previous components
// Context: The orchestrator that manages the entire adaptive visual system

import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';
import { HardwareCapabilities, VisualFeatureSet } from '../types';
import { hardwareProfiler } from './HardwareProfiler';
import { FeatureCalculator, PerformanceConstraints } from './FeatureCalculator';
import { performanceMonitor, PerformanceMetrics } from './PerformanceMonitor';
import { ShaderManager } from '../renderers/ShaderManager';
import { AudioEngine } from '../renderers/AudioEngine';
import { openDB, IDBPDatabase } from 'idb';

export interface AdaptiveVisualContextValue {
  // State
  capabilities: HardwareCapabilities | null;
  features: VisualFeatureSet | null;
  performanceMode: 'trading' | 'balanced' | 'visual' | 'emergency';
  isInitialized: boolean;
  error: Error | null;
  
  // Metrics
  metrics: PerformanceMetrics | null;
  gpuScore: number;
  
  // Controls
  setPerformanceMode: (mode: 'trading' | 'balanced' | 'visual' | 'emergency') => void;
  setTradingActive: (active: boolean) => void;
  setCascadeDetected: (detected: boolean) => void;
  refreshHardware: () => Promise<void>;
  
  // Feature overrides
  overrideFeature: (featureName: string, enabled: boolean) => void;
  resetOverrides: () => void;
  
  // Managers
  shaderManager: ShaderManager | null;
  audioEngine: AudioEngine | null;
}

const AdaptiveVisualContext = createContext<AdaptiveVisualContextValue>({
  capabilities: null,
  features: null,
  performanceMode: 'balanced',
  isInitialized: false,
  error: null,
  metrics: null,
  gpuScore: 0,
  setPerformanceMode: () => {},
  setTradingActive: () => {},
  setCascadeDetected: () => {},
  refreshHardware: async () => {},
  overrideFeature: () => {},
  resetOverrides: () => {},
  shaderManager: null,
  audioEngine: null
});

export const useAdaptiveVisuals = () => {
  const context = useContext(AdaptiveVisualContext);
  if (!context) {
    throw new Error('useAdaptiveVisuals must be used within AdaptiveVisualProvider');
  }
  return context;
};

interface AdaptiveVisualProviderProps {
  children: React.ReactNode;
  config?: {
    autoDetect?: boolean;
    defaultMode?: 'trading' | 'balanced' | 'visual';
    maxGPUPercent?: number;
    maxMemoryMB?: number;
    maxCPUPercent?: number;
    targetFPS?: number;
    enableAudio?: boolean;
    debugMode?: boolean;
  };
}

export const AdaptiveVisualProvider: React.FC<AdaptiveVisualProviderProps> = ({
  children,
  config = {}
}) => {
  // Configuration
  const {
    autoDetect = true,
    defaultMode = 'balanced',
    maxGPUPercent = 30,
    maxMemoryMB = 500,
    maxCPUPercent = 20,
    targetFPS = 60,
    enableAudio = true,
    debugMode = false
  } = config;
  
  // State
  const [capabilities, setCapabilities] = useState<HardwareCapabilities | null>(null);
  const [features, setFeatures] = useState<VisualFeatureSet | null>(null);
  const [performanceMode, setPerformanceMode] = useState<typeof defaultMode>(defaultMode);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [gpuScore, setGpuScore] = useState(0);
  
  // Runtime state
  const [tradingActive, setTradingActive] = useState(false);
  const [cascadeDetected, setCascadeDetected] = useState(false);
  const [featureOverrides, setFeatureOverrides] = useState<Record<string, boolean>>({});
  
  // Managers
  const [shaderManager, setShaderManager] = useState<ShaderManager | null>(null);
  const [audioEngine, setAudioEngine] = useState<AudioEngine | null>(null);
  
  // Refs
  const db = useRef<IDBPDatabase | null>(null);
  const adaptationTimer = useRef<number | null>(null);
  const lastAdaptation = useRef<number>(0);
  
  // Initialize system
  useEffect(() => {
    if (!autoDetect) return;
    
    const initialize = async () => {
      try {
        console.log('[ADAPTIVE] Initializing visual system...');
        
        // Initialize database
        db.current = await openDB('nexlify-adaptive', 1, {
          upgrade(database) {
            database.createObjectStore('preferences', { keyPath: 'id' });
            database.createObjectStore('metrics', { keyPath: 'timestamp' });
          }
        });
        
        // Load preferences
        const savedPrefs = await loadPreferences();
        if (savedPrefs) {
          setPerformanceMode(savedPrefs.performanceMode || defaultMode);
          setFeatureOverrides(savedPrefs.featureOverrides || {});
        }
        
        // Initialize hardware profiler
        await hardwareProfiler.initialize();
        const caps = hardwareProfiler.getCapabilities();
        
        if (!caps) {
          throw new Error('Failed to detect hardware capabilities');
        }
        
        setCapabilities(caps);
        setGpuScore(hardwareProfiler.getGPUScore());
        
        // Initialize managers
        const shader = ShaderManager.getInstance(caps);
        await shader.initialize();
        setShaderManager(shader);
        
        if (enableAudio) {
          const audio = new AudioEngine();
          await audio.initialize();
          setAudioEngine(audio);
        }
        
        // Start performance monitoring
        performanceMonitor.start(shader.getGLContext());
        const unsubscribe = performanceMonitor.subscribe(handleMetricsUpdate);
        
        // Calculate initial features
        updateFeatures(caps, performanceMode, false, false);
        
        setIsInitialized(true);
        console.log('[ADAPTIVE] Visual system initialized successfully');
        
        // Cleanup on unmount
        return () => {
          unsubscribe();
          performanceMonitor.stop();
          shader.dispose();
          audio?.dispose();
        };
        
      } catch (err) {
        console.error('[ADAPTIVE] Initialization failed:', err);
        setError(err as Error);
        
        // Fallback to minimal features
        setFeatures(getMinimalFeatures());
        setIsInitialized(true);
      }
    };
    
    initialize();
  }, [autoDetect]);
  
  // Update features when constraints change
  useEffect(() => {
    if (!capabilities || !isInitialized) return;
    
    updateFeatures(capabilities, performanceMode, tradingActive, cascadeDetected);
  }, [performanceMode, tradingActive, cascadeDetected, capabilities, isInitialized]);
  
  // Handle performance metrics updates
  const handleMetricsUpdate = useCallback((newMetrics: PerformanceMetrics) => {
    setMetrics(newMetrics);
    
    // Check for performance issues
    const issues = performanceMonitor.checkPerformanceIssues();
    
    if (issues.severity === 'high' && performanceMode !== 'trading') {
      console.warn('[ADAPTIVE] Performance issues detected:', issues.issues);
      
      // Throttle adaptations (max once per 5 seconds)
      const now = Date.now();
      if (now - lastAdaptation.current > 5000) {
        lastAdaptation.current = now;
        adaptToPerformance(newMetrics);
      }
    }
    
    // Debug logging
    if (debugMode && newMetrics.fps > 0) {
      performanceMonitor.logMetrics();
    }
  }, [performanceMode, debugMode]);
  
  // Update features based on current state
  const updateFeatures = useCallback((
    caps: HardwareCapabilities,
    mode: typeof performanceMode,
    trading: boolean,
    cascade: boolean
  ) => {
    const constraints: PerformanceConstraints = {
      mode: mode,
      maxGPUPercent: maxGPUPercent,
      maxMemoryMB: maxMemoryMB,
      maxCPUPercent: maxCPUPercent,
      targetFPS: targetFPS,
      currentLoad: {
        gpu: metrics?.gpu.utilization || 0,
        cpu: metrics?.cpu.utilization || 0,
        memory: metrics?.memory.used || 0
      },
      tradingActive: trading,
      cascadeDetected: cascade,
      thermalThrottle: caps.thermal.throttleState !== 'none'
    };
    
    let calculatedFeatures = FeatureCalculator.calculateFeatures(caps, constraints);
    
    // Apply overrides
    for (const [feature, enabled] of Object.entries(featureOverrides)) {
		if (feature in calculatedFeatures && calculatedFeatures[feature]) {
			(calculatedFeatures as any)[feature].enabled = enabled;
		}
	}
    
    setFeatures(calculatedFeatures);
    
    // Update shader manager
    if (shaderManager) {
      shaderManager.updateFeatures(calculatedFeatures);
    }
    
    // Update audio engine
    if (audioEngine && calculatedFeatures.audioVisualization) {
      audioEngine.setEnabled(calculatedFeatures.audioVisualization.enabled);
    }
  }, [metrics, maxGPUPercent, maxMemoryMB, maxCPUPercent, targetFPS, featureOverrides, shaderManager, audioEngine]);
  
  // Adapt to performance issues
  const adaptToPerformance = useCallback((currentMetrics: PerformanceMetrics) => {
    if (!features) return;
    
    console.log('[ADAPTIVE] Adapting to performance issues...');
    
    // Thermal adaptation
    if (capabilities && currentMetrics.gpu.temperature > 75) {
      const adjusted = FeatureCalculator.adjustForThermal(
        features,
        currentMetrics.gpu.temperature,
        capabilities.thermal.maxTemp
      );
      setFeatures(adjusted);
      return;
    }
    
    // Memory pressure adaptation
    if (currentMetrics.memory.pressure === 'high') {
      const adjusted = FeatureCalculator.adjustForMemoryPressure(
        features,
        currentMetrics.memory.available
      );
      setFeatures(adjusted);
      return;
    }
    
    // FPS adaptation
    if (currentMetrics.fps < 30 && performanceMode !== 'trading') {
      console.warn('[ADAPTIVE] Emergency performance mode activated');
      setPerformanceMode('emergency');
    }
  }, [features, capabilities, performanceMode]);
  
  // Save preferences
  const savePreferences = useCallback(async () => {
    if (!db.current) return;
    
    try {
      await db.current.put('preferences', {
        id: 'current',
        performanceMode: performanceMode,
        featureOverrides: featureOverrides,
        timestamp: Date.now()
      });
    } catch (err) {
      console.error('[ADAPTIVE] Failed to save preferences:', err);
    }
  }, [performanceMode, featureOverrides]);
  
  // Load preferences
  const loadPreferences = async () => {
    if (!db.current) return null;
    
    try {
      return await db.current.get('preferences', 'current');
    } catch (err) {
      console.error('[ADAPTIVE] Failed to load preferences:', err);
      return null;
    }
  };
  
  // Save preferences when they change
  useEffect(() => {
    savePreferences();
  }, [savePreferences]);
  
  // Context methods
  const overrideFeature = useCallback((featureName: string, enabled: boolean) => {
    setFeatureOverrides(prev => ({ ...prev, [featureName]: enabled }));
  }, []);
  
  const resetOverrides = useCallback(() => {
    setFeatureOverrides({});
  }, []);
  
  const refreshHardware = useCallback(async () => {
    try {
      const newCaps = await hardwareProfiler.refreshCapabilities();
      setCapabilities(newCaps);
      setGpuScore(hardwareProfiler.getGPUScore());
    } catch (err) {
      console.error('[ADAPTIVE] Hardware refresh failed:', err);
    }
  }, []);
  
  // Get minimal features for fallback
  const getMinimalFeatures = (): VisualFeatureSet => {
    return FeatureCalculator['getMinimalFeatures']();
  };
  
  const contextValue: AdaptiveVisualContextValue = {
    capabilities,
    features,
    performanceMode,
    isInitialized,
    error,
    metrics,
    gpuScore,
    setPerformanceMode,
    setTradingActive,
    setCascadeDetected,
    refreshHardware,
    overrideFeature,
    resetOverrides,
    shaderManager,
    audioEngine
  };
  
  return (
    <AdaptiveVisualContext.Provider value={contextValue}>
      {children}
    </AdaptiveVisualContext.Provider>
  );
};

// Export context for external use
export { AdaptiveVisualContext };