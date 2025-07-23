// Location: nexlify-dashboard/src/systems/adaptive/types/hardware.types.ts
// Mission: 80-I.1 Adaptive Visual System Type Definitions
// Dependencies: None
// Context: Strong typing for all hardware capabilities

export interface GPUInfo {
  vendor: string;
  renderer: string;
  tier: 'unknown' | 'integrated' | 'discrete' | 'workstation';
  architecture?: 'nvidia' | 'amd' | 'intel' | 'apple' | 'other';
  generation?: string; // RTX 40, RTX 30, etc.
}

export interface HardwareCapabilities {
  // GPU Information
  gpu: GPUInfo;
  vram: number; // MB
  vramAvailable: number; // MB currently available
  computeUnits: number; // CUDA cores, stream processors, etc.
  
  // System Resources
  systemMemory: number; // GB
  cpuCores: number;
  cpuThreads: number;
  cpuSpeed: number; // GHz
  
  // WebGL Capabilities
  webglVersion: 1 | 2;
  maxTextureSize: number;
  maxTextureUnits: number;
  maxVertexAttributes: number;
  maxVaryingVectors: number;
  maxFragmentUniforms: number;
  maxVertexUniforms: number;
  supportedExtensions: string[];
  
  // Shader Capabilities
  shaderPrecision: {
    vertexHighp: boolean;
    fragmentHighp: boolean;
    precisionBits: number;
  };
  
  // Performance Benchmarks
  benchmarks: {
    trianglesPerSecond: number;
    pixelFillRate: number;
    shaderOperationsPerSecond: number;
    textureUploadSpeed: number; // MB/s
    drawCallsPerFrame: number;
  };
  
  // Thermal State
  thermal: {
    currentTemp: number; // Celsius
    maxTemp: number; // Thermal limit
    throttleState: 'none' | 'light' | 'moderate' | 'severe';
    fanSpeed: number; // Percentage
  };
  
  // Power State
  power: {
    mode: 'battery' | 'ac';
    batteryLevel?: number; // Percentage
    powerLimit: number; // Watts
    currentDraw: number; // Watts
  };
  
  // Display Information
  display: {
    width: number;
    height: number;
    refreshRate: number;
    hdr: boolean;
    colorGamut: 'srgb' | 'p3' | 'rec2020';
    pixelDensity: number;
  };
  
  // Platform
  platform: {
    os: 'windows' | 'macos' | 'linux' | 'android' | 'ios';
    browser: string;
    version: string;
    mobile: boolean;
  };
  
  // Timestamp
  detectedAt: number;
  profileVersion: string;
}

// Feature definitions
export interface VisualFeature {
  name: string;
  enabled: boolean;
  quality: number; // 0-1
  priority: 'critical' | 'high' | 'medium' | 'low';
  gpuCost: number; // Estimated GPU %
  memoryCost: number; // MB
  cpuCost: number; // Estimated CPU %
}

export interface VisualFeatureSet {
  // Core Effects
  matrixRain: VisualFeature & {
    density: number; // 0-1
    speed: number; // 0-1
    complexity: 'simple' | 'standard' | 'complex';
    colorScheme: 'classic' | 'custom';
  };
  
  scanlines: VisualFeature & {
    intensity: number; // 0-1
    thickness: number; // pixels
    speed: number; // scan speed
    interference: boolean;
  };
  
  glitchEffects: VisualFeature & {
    probability: number; // 0-0.01
    intensity: number; // 0-1
    types: ('displacement' | 'color' | 'noise' | 'datamosh')[];
    duration: { min: number; max: number };
  };
  
  neonGlow: VisualFeature & {
    intensity: number; // 0-2
    layers: number; // 1-5
    pulseFrequency: number; // Hz
    colors: string[]; // Hex colors
  };
  
  // Advanced Effects
  particleSystem: VisualFeature & {
    maxParticles: number;
    emissionRate: number; // particles/second
    physics: boolean;
    collisions: boolean;
  };
  
  postProcessing: VisualFeature & {
    bloom: boolean;
    bloomIntensity: number;
    chromaticAberration: boolean;
    vignette: boolean;
    filmGrain: boolean;
    motionBlur: boolean;
  };
  
  shaderEffects: VisualFeature & {
    complexity: 'basic' | 'standard' | 'advanced';
    customShaders: string[]; // Shader names
    computeShaders: boolean;
  };
  
  audioVisualization: VisualFeature & {
    reactive: boolean;
    frequencyBands: number;
    smoothing: number;
    visualization: 'bars' | 'waveform' | 'spectrum' | 'neural';
  };
  
  // Performance Features
  asyncRendering: boolean;
  frameRateTarget: 30 | 60 | 90 | 120 | 144 | 165 | 240;
  dynamicResolution: boolean;
  temporalUpsampling: boolean;
}