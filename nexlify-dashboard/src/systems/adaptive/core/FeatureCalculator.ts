// Location: nexlify-dashboard/src/systems/adaptive/core/FeatureCalculator.ts
// Mission: 80-I.1 Feature Calculation Based on Hardware
// Dependencies: hardware.types.ts, features.types.ts
// Context: Determines which visual features to enable based on real capabilities

import { HardwareCapabilities, VisualFeatureSet, VisualFeature } from '../types';

export interface PerformanceConstraints {
  mode: 'trading' | 'balanced' | 'visual' | 'emergency';
  maxGPUPercent: number;
  maxMemoryMB: number;
  maxCPUPercent: number;
  targetFPS: number;
  currentLoad: {
    gpu: number;
    cpu: number;
    memory: number;
  };
  tradingActive: boolean;
  cascadeDetected: boolean;
  thermalThrottle: boolean;
}

export class FeatureCalculator {
  private static readonly FEATURE_COSTS = {
    matrixRain: {
      gpu: { base: 15, perQuality: 10 },
      memory: { base: 50, perQuality: 30 },
      cpu: { base: 5, perQuality: 2 }
    },
    scanlines: {
      gpu: { base: 3, perQuality: 2 },
      memory: { base: 10, perQuality: 5 },
      cpu: { base: 1, perQuality: 0.5 }
    },
    glitchEffects: {
      gpu: { base: 8, perQuality: 5 },
      memory: { base: 30, perQuality: 20 },
      cpu: { base: 10, perQuality: 5 }
    },
    neonGlow: {
      gpu: { base: 5, perQuality: 3 },
      memory: { base: 20, perQuality: 10 },
      cpu: { base: 2, perQuality: 1 }
    },
    particleSystem: {
      gpu: { base: 20, perQuality: 15 },
      memory: { base: 100, perQuality: 50 },
      cpu: { base: 15, perQuality: 8 }
    },
    postProcessing: {
      gpu: { base: 25, perQuality: 15 },
      memory: { base: 200, perQuality: 100 },
      cpu: { base: 10, perQuality: 5 }
    },
    shaderEffects: {
      gpu: { base: 10, perQuality: 8 },
      memory: { base: 50, perQuality: 30 },
      cpu: { base: 5, perQuality: 3 }
    },
    audioVisualization: {
      gpu: { base: 5, perQuality: 3 },
      memory: { base: 30, perQuality: 15 },
      cpu: { base: 20, perQuality: 10 }
    }
  };

  private static readonly MODE_MULTIPLIERS = {
    trading: { gpu: 0.1, memory: 0.2, cpu: 0.1 },
    balanced: { gpu: 0.5, memory: 0.6, cpu: 0.5 },
    visual: { gpu: 1.0, memory: 1.0, cpu: 1.0 },
    emergency: { gpu: 0.0, memory: 0.0, cpu: 0.0 }
  };

  static calculateFeatures(
    capabilities: HardwareCapabilities,
    constraints: PerformanceConstraints
  ): VisualFeatureSet {
    console.log('[FEATURES] Calculating optimal feature set...');
    
    // Get base GPU score
    const gpuScore = this.calculateGPUScore(capabilities);
    
    // Apply mode multipliers
    const modeMultiplier = this.MODE_MULTIPLIERS[constraints.mode];
    
    // Calculate available resources
    const availableResources = {
      gpu: constraints.maxGPUPercent * modeMultiplier.gpu - constraints.currentLoad.gpu,
      memory: Math.min(
        constraints.maxMemoryMB * modeMultiplier.memory,
        capabilities.vramAvailable * 0.8 // Leave 20% headroom
      ),
      cpu: constraints.maxCPUPercent * modeMultiplier.cpu - constraints.currentLoad.cpu
    };
    
    // Emergency overrides
    if (constraints.thermalThrottle) {
      availableResources.gpu *= 0.5;
      availableResources.cpu *= 0.5;
    }
    
    if (constraints.cascadeDetected) {
      availableResources.gpu *= 0.3;
      availableResources.cpu *= 0.3;
    }
    
    if (constraints.tradingActive) {
      // Trading gets absolute priority
      return this.getMinimalFeatures();
    }
    
    // Calculate features based on priority
    const features = this.allocateFeatures(
      capabilities,
      availableResources,
      gpuScore,
      constraints.targetFPS
    );
    
    console.log('[FEATURES] Calculated features:', features);
    return features;
  }

  private static allocateFeatures(
    capabilities: HardwareCapabilities,
    available: { gpu: number; memory: number; cpu: number },
    gpuScore: number,
    targetFPS: number
  ): VisualFeatureSet {
    const allocated = { gpu: 0, memory: 0, cpu: 0 };
    const features: Partial<VisualFeatureSet> = {};
    
    // Priority order based on visual impact vs cost
    const priorityOrder = [
      { name: 'scanlines', minScore: 5 },
      { name: 'neonGlow', minScore: 15 },
      { name: 'matrixRain', minScore: 25 },
      { name: 'glitchEffects', minScore: 30 },
      { name: 'shaderEffects', minScore: 35 },
      { name: 'audioVisualization', minScore: 40 },
      { name: 'particleSystem', minScore: 50 },
      { name: 'postProcessing', minScore: 60 }
    ];
    
    // Base features
    features.asyncRendering = capabilities.benchmarks.drawCallsPerFrame > 1000;
    features.frameRateTarget = this.calculateTargetFPS(capabilities, targetFPS);
    features.dynamicResolution = gpuScore > 40;
    features.temporalUpsampling = gpuScore > 60 && capabilities.webglVersion === 2;
    
    // Allocate features by priority
    for (const { name, minScore } of priorityOrder) {
      if (gpuScore < minScore) continue;
      
      const feature = this.tryAllocateFeature(
        name,
        capabilities,
        available,
        allocated,
        gpuScore
      );
      
      if (feature) {
        (features as any)[name] = feature;
      }
    }
    
    return features as VisualFeatureSet;
  }

  private static tryAllocateFeature(
    featureName: string,
    capabilities: HardwareCapabilities,
    available: { gpu: number; memory: number; cpu: number },
    allocated: { gpu: number; memory: number; cpu: number },
    gpuScore: number
  ): any {
    const costs = this.FEATURE_COSTS[featureName as keyof typeof this.FEATURE_COSTS];
    if (!costs) return null;
    
    // Start with minimal quality
    let quality = 0.1;
    let lastValidFeature = null;
    
    while (quality <= 1.0) {
      const gpuCost = costs.gpu.base + costs.gpu.perQuality * quality;
      const memoryCost = costs.memory.base + costs.memory.perQuality * quality;
      const cpuCost = costs.cpu.base + costs.cpu.perQuality * quality;
      
      // Check if we can afford this quality level
      if (allocated.gpu + gpuCost <= available.gpu &&
          allocated.memory + memoryCost <= available.memory &&
          allocated.cpu + cpuCost <= available.cpu) {
        
        // Create feature config at this quality
        const feature = this.createFeatureConfig(
          featureName,
          quality,
          capabilities,
          gpuScore
        );
        
        if (feature) {
          lastValidFeature = feature;
          allocated.gpu += gpuCost;
          allocated.memory += memoryCost;
          allocated.cpu += cpuCost;
        }
      } else {
        // Can't afford higher quality
        break;
      }
      
      quality += 0.1;
    }
    
    return lastValidFeature;
  }

  private static createFeatureConfig(
    featureName: string,
    quality: number,
    capabilities: HardwareCapabilities,
    gpuScore: number
  ): any {
    const baseFeature: VisualFeature = {
      name: featureName,
      enabled: true,
      quality: quality,
      priority: 'medium',
      gpuCost: 0,
      memoryCost: 0,
      cpuCost: 0
    };
    
    switch (featureName) {
      case 'matrixRain':
        return {
          ...baseFeature,
          density: quality,
          speed: 0.5 + quality * 0.5,
          complexity: quality > 0.7 ? 'complex' : quality > 0.4 ? 'standard' : 'simple',
          colorScheme: 'classic'
        };
        
      case 'scanlines':
        return {
          ...baseFeature,
          intensity: quality * 0.5, // Subtle effect
          thickness: Math.max(1, Math.round(quality * 3)),
          speed: 0.5 + quality * 0.5,
          interference: quality > 0.5
        };
        
      case 'glitchEffects':
        const types = ['displacement'];
        if (quality > 0.3) types.push('color');
        if (quality > 0.6) types.push('noise');
        if (quality > 0.8) types.push('datamosh');
        
        return {
          ...baseFeature,
          probability: quality * 0.005, // Max 0.005
          intensity: quality,
          types: types,
          duration: { min: 50, max: 50 + quality * 150 }
        };
        
      case 'neonGlow':
        return {
          ...baseFeature,
          intensity: quality * 2, // 0-2 range
          layers: Math.max(1, Math.round(quality * 5)),
          pulseFrequency: 0.5,
          colors: ['#00ffff', '#ff00ff', '#ffff00']
        };
        
      case 'particleSystem':
        return {
          ...baseFeature,
          maxParticles: Math.round(quality * 10000),
          emissionRate: Math.round(quality * 100),
          physics: quality > 0.5,
          collisions: quality > 0.8 && gpuScore > 70
        };
        
      case 'postProcessing':
        return {
          ...baseFeature,
          bloom: true,
          bloomIntensity: quality,
          chromaticAberration: quality > 0.5,
          vignette: true,
          filmGrain: quality > 0.7,
          motionBlur: quality > 0.8 && capabilities.benchmarks.pixelFillRate > 5000000000
        };
        
      case 'shaderEffects':
        return {
          ...baseFeature,
          complexity: quality > 0.7 ? 'advanced' : quality > 0.4 ? 'standard' : 'basic',
          customShaders: quality > 0.5 ? ['cascade', 'neural'] : [],
          computeShaders: quality > 0.8 && capabilities.webglVersion === 2
        };
        
      case 'audioVisualization':
        return {
          ...baseFeature,
          reactive: true,
          frequencyBands: Math.round(16 + quality * 48), // 16-64 bands
          smoothing: 0.8,
          visualization: quality > 0.7 ? 'neural' : quality > 0.5 ? 'spectrum' : 'bars'
        };
        
      default:
        return null;
    }
  }

  private static calculateTargetFPS(
    capabilities: HardwareCapabilities,
    requestedFPS: number
  ): 30 | 60 | 90 | 120 | 144 | 165 | 240 {
    const refreshRate = capabilities.display.refreshRate;
    const possibleFPS = [30, 60, 90, 120, 144, 165, 240] as const;
    
    // Find the highest FPS that divides evenly into refresh rate
    // and doesn't exceed requested FPS
    for (let i = possibleFPS.length - 1; i >= 0; i--) {
      const fps = possibleFPS[i];
      if (fps <= requestedFPS && fps <= refreshRate) {
        // Check if we can actually achieve this FPS
        const canAchieve = this.canAchieveFPS(capabilities, fps);
        if (canAchieve) return fps;
      }
    }
    
    return 30; // Fallback
  }

  private static canAchieveFPS(capabilities: HardwareCapabilities, targetFPS: number): boolean {
    const frameTime = 1000 / targetFPS; // ms per frame
    
    // Rough estimation based on benchmarks
    const drawCallsNeeded = 500; // Typical for our visuals
    const drawCallTime = drawCallsNeeded / capabilities.benchmarks.drawCallsPerFrame * 16.67;
    
    return drawCallTime < frameTime * 0.8; // Leave 20% headroom
  }

  private static calculateGPUScore(capabilities: HardwareCapabilities): number {
    // Multi-factor scoring
    const factors = {
      computeUnits: {
        value: capabilities.computeUnits,
        max: 16384, // RTX 4090
        weight: 0.25
      },
      vram: {
        value: capabilities.vram,
        max: 24576, // 24GB
        weight: 0.15
      },
      trianglesPerSecond: {
        value: capabilities.benchmarks.trianglesPerSecond,
        max: 1000000000, // 1 billion
        weight: 0.15
      },
      pixelFillRate: {
        value: capabilities.benchmarks.pixelFillRate,
        max: 50000000000, // 50 gigapixels
        weight: 0.15
      },
      shaderOps: {
        value: capabilities.benchmarks.shaderOperationsPerSecond,
        max: 10000000000, // 10 billion
        weight: 0.15
      },
      webglVersion: {
        value: capabilities.webglVersion,
        max: 2,
        weight: 0.05
      },
      drawCalls: {
        value: capabilities.benchmarks.drawCallsPerFrame,
        max: 10000,
        weight: 0.1
      }
    };
    
    let score = 0;
    for (const factor of Object.values(factors)) {
      const normalized = Math.min(1, factor.value / factor.max);
      score += normalized * factor.weight * 100;
    }
    
    // Apply platform penalties
    if (capabilities.platform.mobile) score *= 0.7;
    if (capabilities.power.mode === 'battery') score *= 0.8;
    if (capabilities.thermal.throttleState !== 'none') score *= 0.7;
    
    return Math.round(score);
  }

  private static getMinimalFeatures(): VisualFeatureSet {
    // Absolute minimum for trading mode
    return {
      matrixRain: {
        name: 'matrixRain',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        density: 0,
        speed: 0,
        complexity: 'simple',
        colorScheme: 'classic'
      },
      scanlines: {
        name: 'scanlines',
        enabled: true,
        quality: 0.2,
        priority: 'low',
        gpuCost: 2,
        memoryCost: 5,
        cpuCost: 1,
        intensity: 0.1,
        thickness: 1,
        speed: 0.5,
        interference: false
      },
      glitchEffects: {
        name: 'glitchEffects',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        probability: 0,
        intensity: 0,
        types: [],
        duration: { min: 0, max: 0 }
      },
      neonGlow: {
        name: 'neonGlow',
        enabled: true,
        quality: 0.1,
        priority: 'low',
        gpuCost: 2,
        memoryCost: 10,
        cpuCost: 1,
        intensity: 0.2,
        layers: 1,
        pulseFrequency: 0,
        colors: ['#00ffff']
      },
      particleSystem: {
        name: 'particleSystem',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        maxParticles: 0,
        emissionRate: 0,
        physics: false,
        collisions: false
      },
      postProcessing: {
        name: 'postProcessing',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        bloom: false,
        bloomIntensity: 0,
        chromaticAberration: false,
        vignette: false,
        filmGrain: false,
        motionBlur: false
      },
      shaderEffects: {
        name: 'shaderEffects',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        complexity: 'basic',
        customShaders: [],
        computeShaders: false
      },
      audioVisualization: {
        name: 'audioVisualization',
        enabled: false,
        quality: 0,
        priority: 'low',
        gpuCost: 0,
        memoryCost: 0,
        cpuCost: 0,
        reactive: false,
        frequencyBands: 0,
        smoothing: 0,
        visualization: 'bars'
      },
      asyncRendering: true,
      frameRateTarget: 60,
      dynamicResolution: false,
      temporalUpsampling: false
    };
  }

  // Utility methods for runtime adjustment
  static adjustForThermal(
    features: VisualFeatureSet,
    temperature: number,
    maxTemp: number
  ): VisualFeatureSet {
    const thermalRatio = temperature / maxTemp;
    
    if (thermalRatio > 0.9) {
      // Emergency thermal throttle
      console.warn('[THERMAL] Emergency throttle at', temperature, '°C');
      return this.getMinimalFeatures();
    }
    
    if (thermalRatio > 0.8) {
      // Heavy throttle
      const adjusted = { ...features };
      adjusted.particleSystem.enabled = false;
      adjusted.postProcessing.enabled = false;
      adjusted.matrixRain.quality *= 0.5;
      adjusted.neonGlow.intensity *= 0.5;
      return adjusted;
    }
    
    if (thermalRatio > 0.7) {
      // Light throttle
      const adjusted = { ...features };
      adjusted.particleSystem.maxParticles *= 0.5;
      adjusted.postProcessing.motionBlur = false;
      adjusted.postProcessing.filmGrain = false;
      return adjusted;
    }
    
    return features;
  }

  static adjustForMemoryPressure(
    features: VisualFeatureSet,
    availableMemory: number
  ): VisualFeatureSet {
    if (availableMemory < 100) {
      // Critical memory pressure
      console.warn('[MEMORY] Critical pressure, disabling heavy features');
      const adjusted = { ...features };
      adjusted.particleSystem.enabled = false;
      adjusted.postProcessing.enabled = false;
      adjusted.shaderEffects.customShaders = [];
      return adjusted;
    }
    
    if (availableMemory < 200) {
      // Moderate pressure
      const adjusted = { ...features };
      adjusted.particleSystem.maxParticles = Math.min(1000, adjusted.particleSystem.maxParticles);
      adjusted.postProcessing.bloomIntensity *= 0.5;
      return adjusted;
    }
    
    return features;
  }

  static calculateResourceUsage(features: VisualFeatureSet): {
    gpu: number;
    memory: number;
    cpu: number;
  } {
    let gpu = 0;
    let memory = 0;
    let cpu = 0;
    
    // Calculate actual usage based on enabled features
    for (const [name, config] of Object.entries(features)) {
      if (typeof config === 'object' && 'enabled' in config && config.enabled) {
        const costs = this.FEATURE_COSTS[name as keyof typeof this.FEATURE_COSTS];
        if (costs) {
          gpu += costs.gpu.base + costs.gpu.perQuality * config.quality;
          memory += costs.memory.base + costs.memory.perQuality * config.quality;
          cpu += costs.cpu.base + costs.cpu.perQuality * config.quality;
        }
      }
    }
    
    return { gpu, memory, cpu };
  }
}