// Location: nexlify-dashboard/src/systems/adaptive/core/HardwareProfiler.ts
// Mission: 80-I.1 Hardware Detection with Zero Assumptions
// Dependencies: hardware.types.ts
// Context: Detect EVERYTHING, assume NOTHING

import { HardwareCapabilities, GPUInfo } from '../types/hardware.types';
import { openDB, IDBPDatabase } from 'idb';

export class HardwareProfiler {
  private static instance: HardwareProfiler;
  private db: IDBPDatabase | null = null;
  private capabilities: HardwareCapabilities | null = null;
  private benchmarkCanvas: HTMLCanvasElement | null = null;
  private benchmarkGl: WebGL2RenderingContext | WebGLRenderingContext | null = null;
  
  private constructor() {
    // Singleton pattern
  }
  
  static getInstance(): HardwareProfiler {
    if (!HardwareProfiler.instance) {
      HardwareProfiler.instance = new HardwareProfiler();
    }
    return HardwareProfiler.instance;
  }
  
  async initialize(): Promise<void> {
    // Initialize IndexedDB for caching
    try {
      this.db = await openDB('nexlify-hardware', 1, {
        upgrade(db) {
          db.createObjectStore('profiles', { keyPath: 'id' });
          db.createObjectStore('benchmarks', { keyPath: 'timestamp' });
        }
      });
      
      // Check for cached profile
      const cached = await this.loadCachedProfile();
      if (cached && this.isProfileValid(cached)) {
        this.capabilities = cached;
        console.log('[HARDWARE] Using cached profile:', cached);
        return;
      }
    } catch (error) {
      console.error('[HARDWARE] IndexedDB init failed:', error);
      // Continue without cache
    }
    
    // Run full detection
    await this.detectCapabilities();
  }
  
  async detectCapabilities(): Promise<HardwareCapabilities> {
    console.log('[HARDWARE] Starting full detection...');
    
    try {
      // Create benchmark canvas
      this.benchmarkCanvas = document.createElement('canvas');
      this.benchmarkCanvas.width = 1920;
      this.benchmarkCanvas.height = 1080;
      
      // Try WebGL 2 first, fall back to WebGL 1
      this.benchmarkGl = this.benchmarkCanvas.getContext('webgl2', {
        powerPreference: 'high-performance',
        failIfMajorPerformanceCaveat: false,
        antialias: false,
        depth: false,
        stencil: false,
        alpha: false
      }) as WebGL2RenderingContext;
      
      const isWebGL2 = !!this.benchmarkGl;
      
      if (!this.benchmarkGl) {
        this.benchmarkGl = this.benchmarkCanvas.getContext('webgl', {
          powerPreference: 'high-performance',
          failIfMajorPerformanceCaveat: false
        }) as WebGLRenderingContext;
      }
      
      if (!this.benchmarkGl) {
        throw new Error('WebGL not supported - potato detected');
      }
      
      const gl = this.benchmarkGl;
      
      // Detect GPU info
      const gpuInfo = this.detectGPU(gl);
      
      // Get WebGL capabilities
      const webglCaps = this.detectWebGLCapabilities(gl, isWebGL2);
      
      // Run performance benchmarks
      const benchmarks = await this.runBenchmarks(gl);
      
      // Get system info
      const systemInfo = this.detectSystemInfo();
      
      // Get display info
      const displayInfo = this.detectDisplayInfo();
      
      // Detect thermal state (if available)
      const thermalInfo = await this.detectThermalState();
      
      // Detect power state
      const powerInfo = await this.detectPowerState();
      
      // Calculate compute units based on GPU
      const computeUnits = this.estimateComputeUnits(gpuInfo);
      
      // Calculate VRAM
      const vramInfo = this.estimateVRAM(gpuInfo, webglCaps);
      
      // Build capabilities object
      this.capabilities = {
        gpu: gpuInfo,
        vram: vramInfo.total,
        vramAvailable: vramInfo.available,
        computeUnits: computeUnits,
        systemMemory: systemInfo.memory,
        cpuCores: systemInfo.cores,
        cpuThreads: systemInfo.threads,
        cpuSpeed: systemInfo.speed,
        webglVersion: isWebGL2 ? 2 : 1,
        maxTextureSize: webglCaps.maxTextureSize,
        maxTextureUnits: webglCaps.maxTextureUnits,
        maxVertexAttributes: webglCaps.maxVertexAttributes,
        maxVaryingVectors: webglCaps.maxVaryingVectors,
        maxFragmentUniforms: webglCaps.maxFragmentUniforms,
        maxVertexUniforms: webglCaps.maxVertexUniforms,
        supportedExtensions: webglCaps.extensions,
        shaderPrecision: webglCaps.shaderPrecision,
        benchmarks: benchmarks,
        thermal: thermalInfo,
        power: powerInfo,
        display: displayInfo,
        platform: this.detectPlatform(),
        detectedAt: Date.now(),
        profileVersion: '1.0.0'
      };
      
      // Cache the profile
      await this.cacheProfile(this.capabilities);
      
      // Cleanup
      this.cleanup();
      
      console.log('[HARDWARE] Detection complete:', this.capabilities);
      return this.capabilities;
      
    } catch (error) {
      console.error('[HARDWARE] Detection failed:', error);
      
      // Return minimal fallback capabilities
      const fallback = this.getMinimalCapabilities();
      this.capabilities = fallback;
      return fallback;
    }
  }
  
  private detectGPU(gl: WebGLRenderingContext | WebGL2RenderingContext): GPUInfo {
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    
    if (debugInfo) {
      const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || 'Unknown';
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || 'Unknown';
      
      return {
        vendor: vendor,
        renderer: renderer,
        tier: this.classifyGPUTier(renderer),
        architecture: this.detectArchitecture(vendor, renderer),
        generation: this.detectGeneration(renderer)
      };
    }
    
    // Fallback detection
    const vendor = gl.getParameter(gl.VENDOR) || 'Unknown';
    const renderer = gl.getParameter(gl.RENDERER) || 'Unknown';
    
    return {
      vendor: vendor,
      renderer: renderer,
      tier: 'unknown',
      architecture: 'other',
      generation: undefined
    };
  }
  
  private classifyGPUTier(renderer: string): 'integrated' | 'discrete' | 'workstation' | 'unknown' {
    const integrated = ['Intel', 'UHD', 'Iris', 'Mali', 'Adreno', 'Apple GPU'];
    const workstation = ['Quadro', 'FirePro', 'Radeon Pro', 'A100', 'A6000'];
    
    if (integrated.some(term => renderer.includes(term))) return 'integrated';
    if (workstation.some(term => renderer.includes(term))) return 'workstation';
    if (renderer.includes('NVIDIA') || renderer.includes('AMD') || renderer.includes('Radeon')) return 'discrete';
    
    return 'unknown';
  }
  
  private detectArchitecture(vendor: string, renderer: string): GPUInfo['architecture'] {
    if (vendor.includes('NVIDIA') || renderer.includes('NVIDIA')) return 'nvidia';
    if (vendor.includes('AMD') || vendor.includes('ATI') || renderer.includes('Radeon')) return 'amd';
    if (vendor.includes('Intel') || renderer.includes('Intel')) return 'intel';
    if (vendor.includes('Apple') || renderer.includes('Apple')) return 'apple';
    return 'other';
  }
  
  private detectGeneration(renderer: string): string | undefined {
    // NVIDIA
    if (renderer.includes('RTX 40')) return 'RTX 40';
    if (renderer.includes('RTX 30')) return 'RTX 30';
    if (renderer.includes('RTX 20')) return 'RTX 20';
    if (renderer.includes('GTX 16')) return 'GTX 16';
    if (renderer.includes('GTX 10')) return 'GTX 10';
    
    // AMD
    if (renderer.includes('RX 7')) return 'RDNA 3';
    if (renderer.includes('RX 6')) return 'RDNA 2';
    if (renderer.includes('RX 5')) return 'RDNA 1';
    
    // Intel
    if (renderer.includes('Arc A')) return 'Arc Alchemist';
    if (renderer.includes('Xe')) return 'Xe';
    
    // Apple
    if (renderer.includes('M3')) return 'Apple M3';
    if (renderer.includes('M2')) return 'Apple M2';
    if (renderer.includes('M1')) return 'Apple M1';
    
    return undefined;
  }
  
  private detectWebGLCapabilities(gl: WebGLRenderingContext | WebGL2RenderingContext, isWebGL2: boolean) {
    const highpVertex = gl.getShaderPrecisionFormat(gl.VERTEX_SHADER, gl.HIGH_FLOAT);
    const highpFragment = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    
    return {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxTextureUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
      maxVertexAttributes: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
      maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
      maxFragmentUniforms: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
      maxVertexUniforms: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
      extensions: gl.getSupportedExtensions() || [],
      shaderPrecision: {
        vertexHighp: highpVertex ? highpVertex.precision > 0 : false,
        fragmentHighp: highpFragment ? highpFragment.precision > 0 : false,
        precisionBits: highpFragment ? highpFragment.precision : 0
      }
    };
  }
  
  private async runBenchmarks(gl: WebGLRenderingContext | WebGL2RenderingContext) {
    console.log('[BENCHMARK] Starting performance tests...');
    
    const results = {
      trianglesPerSecond: 0,
      pixelFillRate: 0,
      shaderOperationsPerSecond: 0,
      textureUploadSpeed: 0,
      drawCallsPerFrame: 0
    };
    
    try {
      // Triangle throughput test
      results.trianglesPerSecond = await this.benchmarkTriangles(gl);
      
      // Pixel fill rate test
      results.pixelFillRate = await this.benchmarkFillRate(gl);
      
      // Shader operations test
      results.shaderOperationsPerSecond = await this.benchmarkShaderOps(gl);
      
      // Texture upload speed
      results.textureUploadSpeed = await this.benchmarkTextureUpload(gl);
      
      // Draw calls test
      results.drawCallsPerFrame = await this.benchmarkDrawCalls(gl);
      
    } catch (error) {
      console.error('[BENCHMARK] Test failed:', error);
    }
    
    console.log('[BENCHMARK] Results:', results);
    return results;
  }
  
  private async benchmarkTriangles(gl: WebGLRenderingContext | WebGL2RenderingContext): Promise<number> {
    // Create shader program
    const vertexShader = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;
    
    const fragmentShader = `
      precision mediump float;
      void main() {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
      }
    `;
    
    const program = this.createProgram(gl, vertexShader, fragmentShader);
    if (!program) return 0;
    
    // Create massive triangle buffer
    const triangleCount = 100000;
    const vertices = new Float32Array(triangleCount * 6); // 3 vertices * 2 coords
    
    // Fill with random triangles
    for (let i = 0; i < vertices.length; i++) {
      vertices[i] = Math.random() * 2 - 1;
    }
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    gl.useProgram(program);
    
    // Benchmark
    const frames = 60;
    const start = performance.now();
    
    for (let i = 0; i < frames; i++) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, triangleCount * 3);
      gl.finish(); // Force GPU sync
    }
    
    const elapsed = performance.now() - start;
    const trianglesPerSecond = (triangleCount * frames) / (elapsed / 1000);
    
    // Cleanup
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    
    return Math.round(trianglesPerSecond);
  }
  
  private async benchmarkFillRate(gl: WebGLRenderingContext | WebGL2RenderingContext): Promise<number> {
    // Full screen quad shader
    const vertexShader = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;
    
    const fragmentShader = `
      precision mediump float;
      uniform float time;
      void main() {
        // Complex shader to stress fill rate
        vec2 uv = gl_FragCoord.xy / vec2(${this.benchmarkCanvas!.width}.0, ${this.benchmarkCanvas!.height}.0);
        float r = sin(uv.x * 10.0 + time) * 0.5 + 0.5;
        float g = sin(uv.y * 10.0 + time * 1.1) * 0.5 + 0.5;
        float b = sin((uv.x + uv.y) * 10.0 + time * 1.2) * 0.5 + 0.5;
        gl_FragColor = vec4(r, g, b, 1.0);
      }
    `;
    
    const program = this.createProgram(gl, vertexShader, fragmentShader);
    if (!program) return 0;
    
    // Full screen quad
    const vertices = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      1, -1, 1, 1, -1, 1
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    const timeLoc = gl.getUniformLocation(program, 'time');
    gl.useProgram(program);
    
    // Benchmark
    const frames = 60;
    const start = performance.now();
    
    for (let i = 0; i < frames; i++) {
      gl.uniform1f(timeLoc, i * 0.01);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.finish();
    }
    
    const elapsed = performance.now() - start;
    const pixelsPerFrame = this.benchmarkCanvas!.width * this.benchmarkCanvas!.height;
    const pixelsPerSecond = (pixelsPerFrame * frames) / (elapsed / 1000);
    
    // Cleanup
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    
    return Math.round(pixelsPerSecond);
  }
  
  private async benchmarkShaderOps(gl: WebGLRenderingContext | WebGL2RenderingContext): Promise<number> {
    // Complex compute-like shader
    const vertexShader = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;
    
    const fragmentShader = `
      precision highp float;
      uniform float time;
      
      float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
      }
      
      void main() {
        vec2 uv = gl_FragCoord.xy;
        float result = 0.0;
        
        // Intensive calculations
        for (int i = 0; i < 100; i++) {
          float fi = float(i);
          result += noise(uv + vec2(fi * 0.1, time));
          result += sin(result * 3.14159 + fi);
          result += cos(result * 2.71828 - fi);
          result = fract(result * 1.618);
        }
        
        gl_FragColor = vec4(result, result, result, 1.0);
      }
    `;
    
    const program = this.createProgram(gl, vertexShader, fragmentShader);
    if (!program) return 0;
    
    // Small quad to focus on shader ops, not fill rate
    const size = 256;
    this.benchmarkCanvas!.width = size;
    this.benchmarkCanvas!.height = size;
    gl.viewport(0, 0, size, size);
    
    const vertices = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      1, -1, 1, 1, -1, 1
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    const timeLoc = gl.getUniformLocation(program, 'time');
    gl.useProgram(program);
    
    // Benchmark
    const frames = 30;
    const start = performance.now();
    
    for (let i = 0; i < frames; i++) {
      gl.uniform1f(timeLoc, i * 0.01);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.finish();
    }
    
    const elapsed = performance.now() - start;
    const opsPerPixel = 100 * 5; // 100 iterations * 5 operations
    const totalOps = size * size * opsPerPixel * frames;
    const opsPerSecond = totalOps / (elapsed / 1000);
    
    // Restore canvas size
    this.benchmarkCanvas!.width = 1920;
    this.benchmarkCanvas!.height = 1080;
    
    // Cleanup
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    
    return Math.round(opsPerSecond);
  }
  
  private async benchmarkTextureUpload(gl: WebGLRenderingContext | WebGL2RenderingContext): Promise<number> {
    const textureSize = 2048;
    const textures: WebGLTexture[] = [];
    const data = new Uint8Array(textureSize * textureSize * 4);
    
    // Fill with random data
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.floor(Math.random() * 256);
    }
    
    // Create textures
    const textureCount = 10;
    for (let i = 0; i < textureCount; i++) {
      const texture = gl.createTexture();
      if (texture) textures.push(texture);
    }
    
    // Benchmark upload
    const start = performance.now();
    
    for (const texture of textures) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        textureSize, textureSize, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, data
      );
      gl.finish(); // Force sync
    }
    
    const elapsed = performance.now() - start;
    const bytesUploaded = textureSize * textureSize * 4 * textureCount;
    const mbPerSecond = (bytesUploaded / 1024 / 1024) / (elapsed / 1000);
    
    // Cleanup
    textures.forEach(texture => gl.deleteTexture(texture));
    
    return Math.round(mbPerSecond);
  }
  
  private async benchmarkDrawCalls(gl: WebGLRenderingContext | WebGL2RenderingContext): Promise<number> {
    // Simple shader
    const vertexShader = `
      attribute vec2 position;
      uniform vec2 offset;
      void main() {
        gl_Position = vec4(position * 0.1 + offset, 0.0, 1.0);
      }
    `;
    
    const fragmentShader = `
      precision mediump float;
      uniform vec3 color;
      void main() {
        gl_FragColor = vec4(color, 1.0);
      }
    `;
    
    const program = this.createProgram(gl, vertexShader, fragmentShader);
    if (!program) return 0;
    
    // Small triangle
    const vertices = new Float32Array([
      0, 0.1, -0.05, -0.05, 0.05, -0.05
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
    
    const offsetLoc = gl.getUniformLocation(program, 'offset');
    const colorLoc = gl.getUniformLocation(program, 'color');
    gl.useProgram(program);
    
    // Find maximum draw calls per frame at 60 FPS
    let drawCalls = 100;
    let maxDrawCalls = 0;
    
    while (drawCalls < 10000) {
      const start = performance.now();
      
      gl.clear(gl.COLOR_BUFFER_BIT);
      
      for (let i = 0; i < drawCalls; i++) {
        const x = (i % 10) * 0.2 - 0.9;
        const y = Math.floor(i / 10) * 0.2 - 0.9;
        gl.uniform2f(offsetLoc, x, y);
        gl.uniform3f(colorLoc, i / drawCalls, 1 - i / drawCalls, 0.5);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      
      gl.finish();
      
      const elapsed = performance.now() - start;
      
      if (elapsed < 16.67) { // 60 FPS target
        maxDrawCalls = drawCalls;
        drawCalls += 100;
      } else {
        break;
      }
    }
    
    // Cleanup
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    
    return maxDrawCalls;
  }
  
  private createProgram(gl: WebGLRenderingContext | WebGL2RenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram | null {
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    
    if (!vertexShader || !fragmentShader) return null;
    
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);
    
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
      console.error('[SHADER] Vertex compilation failed:', gl.getShaderInfoLog(vertexShader));
      return null;
    }
    
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);
    
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      console.error('[SHADER] Fragment compilation failed:', gl.getShaderInfoLog(fragmentShader));
      return null;
    }
    
    const program = gl.createProgram();
    if (!program) return null;
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('[SHADER] Program linking failed:', gl.getProgramInfoLog(program));
      return null;
    }
    
    // Clean up shaders after linking
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    
    return program;
  }
  
  private detectSystemInfo() {
    return {
      memory: (navigator as any).deviceMemory || this.estimateMemory(),
      cores: navigator.hardwareConcurrency || 4,
      threads: navigator.hardwareConcurrency || 4, // Assume SMT/HT
      speed: this.estimateCPUSpeed()
    };
  }
  
  private estimateMemory(): number {
    // Estimate based on platform
    if (this.isMobile()) return 4;
    if (performance.memory) {
      const limit = (performance.memory as any).jsHeapSizeLimit;
      if (limit > 4000000000) return 16;
      if (limit > 2000000000) return 8;
    }
    return 8; // Default assumption
  }
  
  private estimateCPUSpeed(): number {
    // Very rough estimation based on simple benchmark
    const iterations = 1000000;
    const start = performance.now();
    let sum = 0;
    
    for (let i = 0; i < iterations; i++) {
      sum += Math.sqrt(i) * Math.sin(i);
    }
    
    const elapsed = performance.now() - start;
    
    // Map elapsed time to approximate GHz
    if (elapsed < 10) return 4.5;
    if (elapsed < 20) return 3.5;
    if (elapsed < 40) return 2.5;
    return 1.5;
  }
  
  private estimateComputeUnits(gpu: GPUInfo): number {
    const renderer = gpu.renderer.toLowerCase();
    
    // NVIDIA mappings
    const nvidiaMap: Record<string, number> = {
      '4090': 16384, '4080': 9728, '4070 ti': 7680, '4070': 5888,
      '3090': 10496, '3080': 8704, '3070': 5888, '3060': 3584,
      '2080': 2944, '2070': 2304, '2060': 1920,
      '1080': 2560, '1070': 1920, '1060': 1280, '1050': 640
    };
    
    for (const [model, cores] of Object.entries(nvidiaMap)) {
      if (renderer.includes(model)) return cores;
    }
    
    // AMD mappings
    if (renderer.includes('rx 7900')) return 6144;
    if (renderer.includes('rx 6900')) return 5120;
    if (renderer.includes('rx 6800')) return 3840;
    if (renderer.includes('rx 6700')) return 2560;
    if (renderer.includes('rx 5700')) return 2560;
    
    // Intel
    if (renderer.includes('arc a770')) return 4096;
    if (renderer.includes('arc a750')) return 3584;
    if (renderer.includes('arc a380')) return 1024;
    if (renderer.includes('uhd')) return 96;
    if (renderer.includes('iris')) return 96;
    
    // Apple
    if (renderer.includes('m3 max')) return 40;
    if (renderer.includes('m3 pro')) return 18;
    if (renderer.includes('m3')) return 10;
    if (renderer.includes('m2')) return 10;
    if (renderer.includes('m1')) return 8;
    
    // Mobile
    if (renderer.includes('adreno')) return 512;
    if (renderer.includes('mali')) return 384;
    
    // Default based on tier
    if (gpu.tier === 'workstation') return 8192;
    if (gpu.tier === 'discrete') return 2048;
    if (gpu.tier === 'integrated') return 96;
    
    return 512; // Unknown default
  }
  
  private estimateVRAM(gpu: GPUInfo, webglCaps: any): { total: number; available: number } {
    const renderer = gpu.renderer.toLowerCase();
    
    // NVIDIA mappings
    const nvidiaVRAM: Record<string, number> = {
      '4090': 24576, '4080': 16384, '4070 ti': 12288, '4070': 12288,
      '3090': 24576, '3080': 10240, '3070': 8192, '3060': 12288,
      '2080': 8192, '2070': 8192, '2060': 6144,
      '1080': 8192, '1070': 8192, '1060': 6144, '1050': 4096
    };
    
    for (const [model, vram] of Object.entries(nvidiaVRAM)) {
      if (renderer.includes(model)) {
        return { total: vram, available: vram * 0.8 }; // Assume 80% available
      }
    }
    
    // AMD
    if (renderer.includes('rx 7900')) return { total: 24576, available: 19660 };
    if (renderer.includes('rx 6900')) return { total: 16384, available: 13107 };
    if (renderer.includes('rx 6800')) return { total: 16384, available: 13107 };
    
    // Try to estimate from max texture size
    const maxTexture = webglCaps.maxTextureSize;
    if (maxTexture >= 16384) return { total: 8192, available: 6553 };
    if (maxTexture >= 8192) return { total: 4096, available: 3276 };
    if (maxTexture >= 4096) return { total: 2048, available: 1638 };
    
    // Default fallback
    return { total: 2048, available: 1638 };
  }
  
  private detectDisplayInfo() {
    const width = window.screen.width * window.devicePixelRatio;
    const height = window.screen.height * window.devicePixelRatio;
    
    return {
      width: width,
      height: height,
      refreshRate: this.detectRefreshRate(),
      hdr: this.detectHDR(),
      colorGamut: this.detectColorGamut(),
      pixelDensity: window.devicePixelRatio
    };
  }
  
  private detectRefreshRate(): number {
    // Check if we have the Screen API refresh rate
    if ('refreshRate' in window.screen) {
      return (window.screen as any).refreshRate || 60;
    }
    
    // Try to detect via requestAnimationFrame timing
    // This would need multiple samples for accuracy
    return 60; // Default for now
  }
  
  private detectHDR(): boolean {
    if (window.matchMedia('(dynamic-range: high)').matches) return true;
    if (window.matchMedia('(color-gamut: p3)').matches) return true;
    return false;
  }
  
  private detectColorGamut(): 'srgb' | 'p3' | 'rec2020' {
    if (window.matchMedia('(color-gamut: rec2020)').matches) return 'rec2020';
    if (window.matchMedia('(color-gamut: p3)').matches) return 'p3';
    return 'srgb';
  }
  
  private async detectThermalState() {
    // Would need native API access for real thermal data
    // For now, return estimated values
    return {
      currentTemp: 50,
      maxTemp: 85,
      throttleState: 'none' as const,
      fanSpeed: 30
    };
  }
  
  private async detectPowerState() {
    let mode: 'battery' | 'ac' = 'ac';
    let batteryLevel: number | undefined;
    
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery();
        mode = battery.charging ? 'ac' : 'battery';
        batteryLevel = Math.round(battery.level * 100);
      } catch (e) {
        // Battery API not available
      }
    }
    
    return {
      mode: mode,
      batteryLevel: batteryLevel,
      powerLimit: mode === 'battery' ? 30 : 150, // Watts
      currentDraw: 50 // Estimated
    };
  }
  
  private detectPlatform() {
    const userAgent = navigator.userAgent.toLowerCase();
    const platform = navigator.platform.toLowerCase();
    
    let os: HardwareCapabilities['platform']['os'] = 'windows';
    if (platform.includes('mac')) os = 'macos';
    else if (platform.includes('linux')) os = 'linux';
    else if (userAgent.includes('android')) os = 'android';
    else if (userAgent.includes('iphone') || userAgent.includes('ipad')) os = 'ios';
    
    return {
      os: os,
      browser: this.detectBrowser(),
      version: navigator.appVersion,
      mobile: this.isMobile()
    };
  }
  
  private detectBrowser(): string {
    const userAgent = navigator.userAgent.toLowerCase();
    if (userAgent.includes('firefox')) return 'firefox';
    if (userAgent.includes('safari') && !userAgent.includes('chrome')) return 'safari';
    if (userAgent.includes('edge')) return 'edge';
    if (userAgent.includes('chrome')) return 'chrome';
    return 'unknown';
  }
  
  private isMobile(): boolean {
    return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(navigator.userAgent);
  }
  
  private async loadCachedProfile(): Promise<HardwareCapabilities | null> {
    if (!this.db) return null;
    
    try {
      const profile = await this.db.get('profiles', 'current');
      return profile || null;
    } catch (error) {
      console.error('[HARDWARE] Failed to load cached profile:', error);
      return null;
    }
  }
  
  private async cacheProfile(capabilities: HardwareCapabilities): Promise<void> {
    if (!this.db) return;
    
    try {
      await this.db.put('profiles', { ...capabilities, id: 'current' });
      console.log('[HARDWARE] Profile cached successfully');
    } catch (error) {
      console.error('[HARDWARE] Failed to cache profile:', error);
    }
  }
  
  private isProfileValid(profile: HardwareCapabilities): boolean {
    // Check if profile is less than 24 hours old
    const age = Date.now() - profile.detectedAt;
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    if (age > maxAge) {
      console.log('[HARDWARE] Cached profile too old, re-detecting');
      return false;
    }
    
    // Verify profile has required fields
    const requiredFields = ['gpu', 'vram', 'benchmarks', 'webglVersion'];
    for (const field of requiredFields) {
      if (!(field in profile)) {
        console.log('[HARDWARE] Cached profile missing field:', field);
        return false;
      }
    }
    
    return true;
  }
  
  private getMinimalCapabilities(): HardwareCapabilities {
    // Absolute minimum fallback for when everything fails
    return {
      gpu: {
        vendor: 'Unknown',
        renderer: 'Unknown',
        tier: 'unknown',
        architecture: 'other'
      },
      vram: 2048,
      vramAvailable: 1024,
      computeUnits: 96,
      systemMemory: 4,
      cpuCores: 2,
      cpuThreads: 2,
      cpuSpeed: 1.5,
      webglVersion: 1,
      maxTextureSize: 2048,
      maxTextureUnits: 8,
      maxVertexAttributes: 8,
      maxVaryingVectors: 8,
      maxFragmentUniforms: 16,
      maxVertexUniforms: 128,
      supportedExtensions: [],
      shaderPrecision: {
        vertexHighp: false,
        fragmentHighp: false,
        precisionBits: 0
      },
      benchmarks: {
        trianglesPerSecond: 1000000,
        pixelFillRate: 100000000,
        shaderOperationsPerSecond: 50000000,
        textureUploadSpeed: 50,
        drawCallsPerFrame: 100
      },
      thermal: {
        currentTemp: 50,
        maxTemp: 85,
        throttleState: 'none',
        fanSpeed: 30
      },
      power: {
        mode: 'ac',
        powerLimit: 30,
        currentDraw: 15
      },
      display: {
        width: 1920,
        height: 1080,
        refreshRate: 60,
        hdr: false,
        colorGamut: 'srgb',
        pixelDensity: 1
      },
      platform: {
        os: 'windows',
        browser: 'chrome',
        version: navigator.appVersion,
        mobile: false
      },
      detectedAt: Date.now(),
      profileVersion: '1.0.0'
    };
  }
  
  private cleanup(): void {
    if (this.benchmarkGl) {
      // Get extension to lose context
      const loseContext = this.benchmarkGl.getExtension('WEBGL_lose_context');
      if (loseContext) {
        loseContext.loseContext();
      }
    }
    
    if (this.benchmarkCanvas) {
      this.benchmarkCanvas.width = 1;
      this.benchmarkCanvas.height = 1;
      this.benchmarkCanvas = null;
    }
    
    this.benchmarkGl = null;
  }
  
  // Public getters
  getCapabilities(): HardwareCapabilities | null {
    return this.capabilities;
  }
  
  async refreshCapabilities(): Promise<HardwareCapabilities> {
    return await this.detectCapabilities();
  }
  
  // Utility methods
  getGPUScore(): number {
    if (!this.capabilities) return 0;
    
    // Weighted scoring based on benchmarks and specs
    const weights = {
      computeUnits: 0.25,
      vram: 0.15,
      triangles: 0.15,
      fillRate: 0.15,
      shaderOps: 0.15,
      drawCalls: 0.1,
      webglVersion: 0.05
    };
    
    // Normalize scores (0-100)
    const scores = {
      computeUnits: Math.min(100, (this.capabilities.computeUnits / 16384) * 100),
      vram: Math.min(100, (this.capabilities.vram / 24576) * 100),
      triangles: Math.min(100, (this.capabilities.benchmarks.trianglesPerSecond / 100000000) * 100),
      fillRate: Math.min(100, (this.capabilities.benchmarks.pixelFillRate / 10000000000) * 100),
      shaderOps: Math.min(100, (this.capabilities.benchmarks.shaderOperationsPerSecond / 1000000000) * 100),
      drawCalls: Math.min(100, (this.capabilities.benchmarks.drawCallsPerFrame / 5000) * 100),
      webglVersion: this.capabilities.webglVersion === 2 ? 100 : 50
    };
    
    // Calculate weighted score
    let totalScore = 0;
    for (const [key, weight] of Object.entries(weights)) {
      totalScore += scores[key as keyof typeof scores] * weight;
    }
    
    return Math.round(totalScore);
  }
  
  canRunFeature(feature: string, quality: number = 1): boolean {
    if (!this.capabilities) return false;
    
    const score = this.getGPUScore();
    
    // Feature requirements (score needed at quality 1.0)
    const requirements: Record<string, number> = {
      matrixRain: 20,
      scanlines: 5,
      glitchEffects: 25,
      neonGlow: 15,
      particleSystem: 40,
      postProcessing: 50,
      audioVisualization: 30,
      shaderEffects: 35
    };
    
    const required = requirements[feature] || 50;
    const adjustedRequired = required * quality;
    
    return score >= adjustedRequired;
  }
}

// Export singleton instance
export const hardwareProfiler = HardwareProfiler.getInstance();