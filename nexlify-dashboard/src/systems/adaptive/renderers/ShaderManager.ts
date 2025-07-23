// Location: nexlify-dashboard/src/systems/adaptive/renderers/ShaderManager.ts
// Mission: 80-I.1 WebGL Shader Management System
// Dependencies: hardware.types.ts, features.types.ts
// Context: Manages all WebGL shaders and programs for visual effects

import { HardwareCapabilities, VisualFeatureSet } from '../types';

interface ShaderProgram {
  program: WebGLProgram;
  uniforms: Map<string, WebGLUniformLocation>;
  attributes: Map<string, number>;
  lastUsed: number;
}

export class ShaderManager {
  private gl: WebGL2RenderingContext | WebGLRenderingContext | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private programs: Map<string, ShaderProgram> = new Map();
  private currentProgram: string | null = null;
  private isWebGL2: boolean = false;
  
  // Shader sources
  private shaderSources: Map<string, { vertex: string; fragment: string }> = new Map();
  
  constructor(private capabilities: HardwareCapabilities) {
    this.initializeShaderSources();
  }
  
  async initialize(): Promise<void> {
    console.log('[SHADER] Initializing shader manager...');
    
    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.capabilities.display.width;
    this.canvas.height = this.capabilities.display.height;
    this.canvas.style.position = 'fixed';
    this.canvas.style.top = '0';
    this.canvas.style.left = '0';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.pointerEvents = 'none';
    this.canvas.style.zIndex = '1';
    
    // Try WebGL 2 first
    this.gl = this.canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
      failIfMajorPerformanceCaveat: false
    }) as WebGL2RenderingContext;
    
    if (this.gl) {
      this.isWebGL2 = true;
    } else {
      // Fall back to WebGL 1
      this.gl = this.canvas.getContext('webgl', {
        alpha: true,
        antialias: false,
        depth: false,
        powerPreference: 'high-performance'
      }) as WebGLRenderingContext;
    }
    
    if (!this.gl) {
      throw new Error('WebGL not supported');
    }
    
    console.log('[SHADER] Using WebGL', this.isWebGL2 ? '2' : '1');
    
    // Setup GL state
    this.setupGLState();
    
    // Compile base shaders
    await this.compileBaseShaders();
  }
  
  private initializeShaderSources(): void {
    // Matrix rain shader
    this.shaderSources.set('matrixRain', {
      vertex: `
        ${this.isWebGL2 ? '#version 300 es' : ''}
        ${this.isWebGL2 ? 'in' : 'attribute'} vec2 a_position;
        void main() {
          gl_Position = vec4(a_position, 0.0, 1.0);
        }
      `,
      fragment: `
        ${this.isWebGL2 ? '#version 300 es' : ''}
        precision highp float;
        
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_density;
        uniform float u_speed;
        uniform float u_complexity;
        ${this.isWebGL2 ? 'out vec4 fragColor;' : ''}
        
        float random(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        float noise(vec2 st) {
          vec2 i = floor(st);
          vec2 f = fract(st);
          
          float a = random(i);
          float b = random(i + vec2(1.0, 0.0));
          float c = random(i + vec2(0.0, 1.0));
          float d = random(i + vec2(1.0, 1.0));
          
          vec2 u = f * f * (3.0 - 2.0 * f);
          
          return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }
        
        void main() {
          vec2 uv = ${this.isWebGL2 ? 'gl_FragCoord.xy' : 'gl_FragCoord.xy'} / u_resolution;
          vec2 pos = uv * vec2(50.0 * u_density, 30.0);
          vec2 ipos = floor(pos);
          vec2 fpos = fract(pos);
          
          float columnSpeed = random(vec2(ipos.x, 0.0)) * 0.5 + 0.5;
          float offset = random(vec2(ipos.x, 1.0)) * 10.0;
          float rain = fract((u_time * u_speed * columnSpeed + offset) * 0.5);
          
          // Character brightness
          float charBrightness = 0.0;
          if (u_complexity > 0.5) {
            // Complex: animated characters
            float charTime = u_time * 10.0 + random(ipos) * 100.0;
            charBrightness = noise(vec2(ipos.x * 0.1, ipos.y * 0.1 + charTime));
          } else {
            // Simple: static characters
            charBrightness = random(ipos);
          }
          
          // Fade trail
          float fade = 1.0 - smoothstep(0.0, 1.0, rain);
          float glow = exp(-5.0 * rain);
          
          // Color with green tint
          vec3 color = vec3(0.0, 1.0, 0.2);
          color *= charBrightness * fade * (0.5 + 0.5 * glow);
          
          // Leading edge highlight
          if (abs(rain - fpos.y) < 0.1) {
            color += vec3(0.2, 0.8, 0.4) * glow;
          }
          
          ${this.isWebGL2 ? 'fragColor' : 'gl_FragColor'} = vec4(color, fade * 0.5);
        }
      `
    });
    
    // Scanline shader
    this.shaderSources.set('scanlines', {
      vertex: this.shaderSources.get('matrixRain')!.vertex,
      fragment: `
        ${this.isWebGL2 ? '#version 300 es' : ''}
        precision mediump float;
        
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_intensity;
        uniform float u_thickness;
        uniform float u_speed;
        ${this.isWebGL2 ? 'out vec4 fragColor;' : ''}
        
        void main() {
          vec2 uv = ${this.isWebGL2 ? 'gl_FragCoord.xy' : 'gl_FragCoord.xy'} / u_resolution;
          
          // Moving scanlines
          float scanline = sin((uv.y + u_time * u_speed * 0.1) * u_resolution.y / u_thickness);
          scanline = smoothstep(0.0, 1.0, scanline * 0.5 + 0.5);
          
          // Interference pattern
          float interference = sin(uv.y * 800.0 + u_time * 2.0) * 0.02;
          
          // Combine effects
          float alpha = (1.0 - scanline) * u_intensity + interference;
          
          ${this.isWebGL2 ? 'fragColor' : 'gl_FragColor'} = vec4(0.0, 1.0, 0.8, alpha * 0.5);
        }
      `
    });
    
    // Glitch shader
    this.shaderSources.set('glitch', {
      vertex: this.shaderSources.get('matrixRain')!.vertex,
      fragment: `
        ${this.isWebGL2 ? '#version 300 es' : ''}
        precision highp float;
        
        uniform float u_time;
        uniform vec2 u_resolution;
        uniform float u_intensity;
        uniform float u_glitchTime;
        uniform sampler2D u_scene;
        ${this.isWebGL2 ? 'out vec4 fragColor;' : ''}
        
        float random(vec2 st) {
          return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        void main() {
          vec2 uv = ${this.isWebGL2 ? 'gl_FragCoord.xy' : 'gl_FragCoord.xy'} / u_resolution;
          vec2 distortedUV = uv;
          
          // Displacement glitch
          float glitchStrength = step(0.99, random(vec2(u_glitchTime)));
          if (glitchStrength > 0.0) {
            float displacement = random(vec2(u_glitchTime * 10.0, uv.y)) * 0.1;
            distortedUV.x += displacement * u_intensity;
          }
          
          // Color channel split
          vec4 color;
          color.r = texture2D(u_scene, distortedUV + vec2(0.01 * u_intensity, 0.0)).r;
          color.g = texture2D(u_scene, distortedUV).g;
          color.b = texture2D(u_scene, distortedUV - vec2(0.01 * u_intensity, 0.0)).b;
          color.a = 1.0;
          
          // Noise blocks
          float blockNoise = step(0.98, random(floor(uv * 10.0) + u_glitchTime));
          if (blockNoise > 0.0) {
            color = vec4(random(uv + u_glitchTime), random(uv + u_glitchTime + 1.0), random(uv + u_glitchTime + 2.0), 1.0);
          }
          
          ${this.isWebGL2 ? 'fragColor' : 'gl_FragColor'} = color;
        }
      `
    });
    
    // Neon glow shader (post-process)
    this.shaderSources.set('neonGlow', {
      vertex: this.shaderSources.get('matrixRain')!.vertex,
      fragment: `
        ${this.isWebGL2 ? '#version 300 es' : ''}
        precision mediump float;
        
        uniform sampler2D u_scene;
        uniform vec2 u_resolution;
        uniform float u_intensity;
        uniform float u_layers;
        ${this.isWebGL2 ? 'out vec4 fragColor;' : ''}
        
        void main() {
          vec2 uv = ${this.isWebGL2 ? 'gl_FragCoord.xy' : 'gl_FragCoord.xy'} / u_resolution;
          vec4 color = texture2D(u_scene, uv);
          
          // Multi-layer glow
          vec4 glow = vec4(0.0);
          float totalWeight = 0.0;
          
          for (float i = 1.0; i <= 5.0; i++) {
            if (i > u_layers) break;
            
            float radius = i * 2.0;
            float weight = 1.0 / (i * i);
            
            // Sample in a cross pattern for performance
            glow += texture2D(u_scene, uv + vec2(radius, 0.0) / u_resolution) * weight;
            glow += texture2D(u_scene, uv - vec2(radius, 0.0) / u_resolution) * weight;
            glow += texture2D(u_scene, uv + vec2(0.0, radius) / u_resolution) * weight;
            glow += texture2D(u_scene, uv - vec2(0.0, radius) / u_resolution) * weight;
            
            totalWeight += weight * 4.0;
          }
          
          glow /= totalWeight;
          
          // Combine with original
          ${this.isWebGL2 ? 'fragColor' : 'gl_FragColor'} = color + glow * u_intensity;
        }
      `
    });
    
    // Add more shaders as needed...
  }
  
  private setupGLState(): void {
    if (!this.gl) return;
    
    const gl = this.gl;
    
    // Enable blending for transparency
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Clear color
    gl.clearColor(0, 0, 0, 0);
    
    // Viewport
    gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
    
    // Disable depth test for 2D effects
    gl.disable(gl.DEPTH_TEST);
  }
  
  private async compileBaseShaders(): Promise<void> {
    console.log('[SHADER] Compiling base shaders...');
    
    for (const [name, sources] of this.shaderSources.entries()) {
      try {
        await this.compileProgram(name, sources.vertex, sources.fragment);
      } catch (err) {
        console.error(`[SHADER] Failed to compile ${name}:`, err);
      }
    }
  }
  
  async compileProgram(name: string, vertexSource: string, fragmentSource: string): Promise<void> {
    if (!this.gl) throw new Error('GL context not initialized');
    
    const gl = this.gl;
    
    // Create shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    // Create program
    const program = gl.createProgram();
    if (!program) throw new Error('Failed to create program');
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program link failed: ${error}`);
    }
    
    // Get uniforms and attributes
    const uniforms = new Map<string, WebGLUniformLocation>();
    const attributes = new Map<string, number>();
    
    const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
      const info = gl.getActiveUniform(program, i);
      if (info) {
        const location = gl.getUniformLocation(program, info.name);
        if (location) uniforms.set(info.name, location);
      }
    }
    
    const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < attributeCount; i++) {
      const info = gl.getActiveAttrib(program, i);
      if (info) {
        const location = gl.getAttribLocation(program, info.name);
        if (location >= 0) attributes.set(info.name, location);
      }
    }
    
    // Clean up shaders
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    
    // Store program
    this.programs.set(name, {
      program,
      uniforms,
      attributes,
      lastUsed: 0
    });
    
    console.log(`[SHADER] Compiled ${name} with ${uniforms.size} uniforms, ${attributes.size} attributes`);
  }
  
  private compileShader(type: number, source: string): WebGLShader {
    if (!this.gl) throw new Error('GL context not initialized');
    
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error('Failed to create shader');
    
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${error}\nSource:\n${source}`);
    }
    
    return shader;
  }
  
  useProgram(name: string): ShaderProgram | null {
    const program = this.programs.get(name);
    if (!program || !this.gl) return null;
    
    if (this.currentProgram !== name) {
      this.gl.useProgram(program.program);
      this.currentProgram = name;
      program.lastUsed = Date.now();
    }
    
    return program;
  }
  
  setUniform(programName: string, uniformName: string, value: any): void {
    const program = this.programs.get(programName);
    if (!program || !this.gl) return;
    
    const location = program.uniforms.get(uniformName);
    if (!location) return;
    
    const gl = this.gl;
    
    // Use current program
    if (this.currentProgram !== programName) {
      this.useProgram(programName);
    }
    
    // Set uniform based on type
    if (typeof value === 'number') {
      gl.uniform1f(location, value);
    } else if (Array.isArray(value)) {
      switch (value.length) {
        case 2:
          gl.uniform2fv(location, value);
          break;
        case 3:
          gl.uniform3fv(location, value);
          break;
        case 4:
          gl.uniform4fv(location, value);
          break;
        case 16:
          gl.uniformMatrix4fv(location, false, value);
          break;
      }
    } else if (value instanceof WebGLTexture) {
      // Texture binding handled separately
    }
  }
  
  updateFeatures(features: VisualFeatureSet): void {
    // Update shader compilation based on features
    // This could compile/remove shaders based on enabled features
    console.log('[SHADER] Updating features:', features);
  }
  
  getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }
  
  getGLContext(): WebGLRenderingContext | WebGL2RenderingContext | null {
    return this.gl;
  }
  
  dispose(): void {
    console.log('[SHADER] Disposing shader manager...');
    
    if (this.gl) {
      // Delete all programs
      for (const [name, program] of this.programs.entries()) {
        this.gl.deleteProgram(program.program);
      }
      
      // Lose context
      const loseContext = this.gl.getExtension('WEBGL_lose_context');
      if (loseContext) {
        loseContext.loseContext();
      }
    }
    
    // Remove canvas
    if (this.canvas && this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas);
    }
    
    // Clear references
    this.programs.clear();
    this.gl = null;
    this.canvas = null;
  }
}