// Location: nexlify-dashboard/src/systems/adaptive/core/PerformanceMonitor.ts
// Mission: 80-I.1 Real-time Performance Monitoring
// Dependencies: None
// Context: Tracks actual performance metrics for dynamic adaptation

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  frameTimeVariance: number;
  droppedFrames: number;
  
  gpu: {
    utilization: number;
    memory: number;
    temperature: number;
  };
  
  cpu: {
    utilization: number;
    temperature: number;
  };
  
  memory: {
    used: number;
    available: number;
    pressure: 'none' | 'low' | 'medium' | 'high';
  };
  
  rendering: {
    drawCalls: number;
    triangles: number;
    shaderSwitches: number;
    textureBinds: number;
  };
  
  timestamp: number;
}

export type PerformanceCallback = (metrics: PerformanceMetrics) => void;

export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  
  private isMonitoring = false;
  private callbacks: Set<PerformanceCallback> = new Set();
  private metrics: PerformanceMetrics;
  
  // Frame timing
  private frameCount = 0;
  private lastFrameTime = 0;
  private frameTimes: number[] = [];
  private droppedFrames = 0;
  
  // WebGL instrumentation
  private gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  private glExtensions: {
    disjointTimer?: any;
  } = {};
  
  // Performance observers
  private observers: {
    memory?: PerformanceObserver;
    longtask?: PerformanceObserver;
  } = {};
  
  private constructor() {
    this.metrics = this.getDefaultMetrics();
  }
  
  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }
  
  start(gl?: WebGLRenderingContext | WebGL2RenderingContext): void {
    if (this.isMonitoring) return;
    
    console.log('[MONITOR] Starting performance monitoring');
    this.isMonitoring = true;
    
    // Store GL context for instrumentation
    if (gl) {
      this.gl = gl;
      this.setupGLInstrumentation(gl);
    }
    
    // Setup performance observers
    this.setupObservers();
    
    // Start monitoring loop
    this.monitorLoop();
  }
  
  stop(): void {
    if (!this.isMonitoring) return;
    
    console.log('[MONITOR] Stopping performance monitoring');
    this.isMonitoring = false;
    
    // Cleanup observers
    Object.values(this.observers).forEach(observer => observer?.disconnect());
    this.observers = {};
  }
  
  subscribe(callback: PerformanceCallback): () => void {
    this.callbacks.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.callbacks.delete(callback);
    };
  }
  
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }
  
  recordFrame(): void {
    const now = performance.now();
    
    if (this.lastFrameTime > 0) {
      const frameTime = now - this.lastFrameTime;
      this.frameTimes.push(frameTime);
      
      // Keep last 120 frames for variance calculation
      if (this.frameTimes.length > 120) {
        this.frameTimes.shift();
      }
      
      // Check for dropped frames (>33ms for 30fps minimum)
      if (frameTime > 33) {
        this.droppedFrames++;
      }
    }
    
    this.lastFrameTime = now;
    this.frameCount++;
  }
  
  private monitorLoop(): void {
    if (!this.isMonitoring) return;
    
    const now = performance.now();
    
    // Calculate FPS
    if (this.frameTimes.length > 0) {
      const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      this.metrics.fps = Math.round(1000 / avgFrameTime);
      this.metrics.frameTime = avgFrameTime;
      this.metrics.frameTimeVariance = this.calculateVariance(this.frameTimes);
      this.metrics.droppedFrames = this.droppedFrames;
    }
    
    // Update other metrics
    this.updateMemoryMetrics();
    this.updateCPUMetrics();
    this.updateGPUMetrics();
    
    this.metrics.timestamp = now;
    
    // Notify subscribers
    this.notifySubscribers();
    
    // Reset frame counter periodically
    if (this.frameCount > 1000) {
      this.frameCount = 0;
      this.droppedFrames = 0;
    }
    
    // Schedule next update (every second)
    setTimeout(() => this.monitorLoop(), 1000);
  }
  
  private setupGLInstrumentation(gl: WebGLRenderingContext | WebGL2RenderingContext): void {
    // Get timer extension for GPU timing
    this.glExtensions.disjointTimer = 
      gl.getExtension('EXT_disjoint_timer_query_webgl2') ||
      gl.getExtension('EXT_disjoint_timer_query');
    
    // Instrument draw calls
    const originalDrawArrays = gl.drawArrays.bind(gl);
    const originalDrawElements = gl.drawElements.bind(gl);
    
    let drawCalls = 0;
    let triangles = 0;
    
    gl.drawArrays = (mode: number, first: number, count: number) => {
      drawCalls++;
      if (mode === gl.TRIANGLES) triangles += count / 3;
      originalDrawArrays(mode, first, count);
    };
    
    gl.drawElements = (mode: number, count: number, type: number, offset: number) => {
      drawCalls++;
      if (mode === gl.TRIANGLES) triangles += count / 3;
      originalDrawElements(mode, count, type, offset);
    };
    
    // Reset counters periodically
    setInterval(() => {
      this.metrics.rendering.drawCalls = drawCalls;
      this.metrics.rendering.triangles = Math.round(triangles);
      drawCalls = 0;
      triangles = 0;
    }, 1000);
  }
  
  private setupObservers(): void {
    // Memory pressure observer (if available)
    if ('memory' in performance) {
      try {
        this.observers.memory = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            console.log('[MONITOR] Memory pressure event:', entry);
          }
        });
        this.observers.memory.observe({ entryTypes: ['memory'] });
      } catch (e) {
        // Not supported
      }
    }
    
    // Long task observer
    try {
      this.observers.longtask = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) {
            console.warn('[MONITOR] Long task detected:', entry.duration, 'ms');
          }
        }
      });
      this.observers.longtask.observe({ entryTypes: ['longtask'] });
    } catch (e) {
      // Not supported
    }
  }
  
  private updateMemoryMetrics(): void {
    if (performance.memory) {
      const memory = (performance as any).memory;
      this.metrics.memory.used = Math.round(memory.usedJSHeapSize / 1024 / 1024);
      this.metrics.memory.available = Math.round(
        (memory.jsHeapSizeLimit - memory.usedJSHeapSize) / 1024 / 1024
      );
      
      // Calculate pressure
      const usageRatio = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
      if (usageRatio > 0.9) {
        this.metrics.memory.pressure = 'high';
      } else if (usageRatio > 0.7) {
        this.metrics.memory.pressure = 'medium';
      } else if (usageRatio > 0.5) {
        this.metrics.memory.pressure = 'low';
      } else {
        this.metrics.memory.pressure = 'none';
      }
    }
  }
  
  private updateCPUMetrics(): void {
    // Estimate CPU usage based on frame times and long tasks
    // This is approximate since we can't directly access CPU usage from browser
    const targetFrameTime = 16.67; // 60 FPS
    const cpuEstimate = Math.min(100, (this.metrics.frameTime / targetFrameTime) * 50);
    
    this.metrics.cpu.utilization = Math.round(cpuEstimate);
    this.metrics.cpu.temperature = 50; // Would need native API
  }
  
  private updateGPUMetrics(): void {
    // GPU metrics would need native API access
    // For now, estimate based on rendering complexity
    const complexity = this.metrics.rendering.drawCalls * this.metrics.rendering.triangles;
    const gpuEstimate = Math.min(100, complexity / 10000);
    
    this.metrics.gpu.utilization = Math.round(gpuEstimate);
    this.metrics.gpu.memory = 1000; // Placeholder
    this.metrics.gpu.temperature = 60; // Placeholder
  }
  
  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    
    return Math.sqrt(variance);
  }
  
  private notifySubscribers(): void {
    const metrics = this.getMetrics();
    this.callbacks.forEach(callback => {
      try {
        callback(metrics);
      } catch (error) {
        console.error('[MONITOR] Subscriber error:', error);
      }
    });
  }
  
  private getDefaultMetrics(): PerformanceMetrics {
    return {
      fps: 0,
      frameTime: 0,
      frameTimeVariance: 0,
      droppedFrames: 0,
      gpu: {
        utilization: 0,
        memory: 0,
        temperature: 0
      },
      cpu: {
        utilization: 0,
        temperature: 0
      },
      memory: {
        used: 0,
        available: 0,
        pressure: 'none'
      },
      rendering: {
        drawCalls: 0,
        triangles: 0,
        shaderSwitches: 0,
        textureBinds: 0
      },
      timestamp: 0
    };
  }
  
  // Debug utilities
  logMetrics(): void {
    console.table({
      FPS: this.metrics.fps,
      'Frame Time': `${this.metrics.frameTime.toFixed(2)}ms`,
      'Dropped Frames': this.metrics.droppedFrames,
      'GPU Usage': `${this.metrics.gpu.utilization}%`,
      'CPU Usage': `${this.metrics.cpu.utilization}%`,
      'Memory Used': `${this.metrics.memory.used}MB`,
      'Draw Calls': this.metrics.rendering.drawCalls,
      'Triangles': this.metrics.rendering.triangles
    });
  }
  
  // Alerts for performance issues
  checkPerformanceIssues(): {
    severity: 'none' | 'low' | 'medium' | 'high';
    issues: string[];
  } {
    const issues: string[] = [];
    let severity: 'none' | 'low' | 'medium' | 'high' = 'none';
    
    // FPS issues
    if (this.metrics.fps < 30) {
      issues.push(`Low FPS: ${this.metrics.fps}`);
      severity = 'high';
    } else if (this.metrics.fps < 50) {
      issues.push(`Below target FPS: ${this.metrics.fps}`);
      severity = severity === 'none' ? 'medium' : severity;
    }
    
    // Frame time variance (stuttering)
    if (this.metrics.frameTimeVariance > 10) {
      issues.push(`High frame time variance: ${this.metrics.frameTimeVariance.toFixed(2)}ms`);
      severity = severity === 'none' ? 'medium' : severity;
    }
    
    // Memory pressure
    if (this.metrics.memory.pressure === 'high') {
      issues.push('High memory pressure');
      severity = 'high';
    } else if (this.metrics.memory.pressure === 'medium') {
      issues.push('Medium memory pressure');
      severity = severity === 'none' ? 'low' : severity;
    }
    
    // Dropped frames
    if (this.metrics.droppedFrames > 10) {
      issues.push(`Dropped frames: ${this.metrics.droppedFrames}`);
      severity = severity === 'none' ? 'medium' : severity;
    }
    
    return { severity, issues };
  }
}

// Export singleton
export const performanceMonitor = PerformanceMonitor.getInstance();