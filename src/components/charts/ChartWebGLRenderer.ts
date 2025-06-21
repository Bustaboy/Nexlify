// src/components/charts/ChartWebGLRenderer.ts
// NEXLIFY WEBGL RENDERER - Where GPUs dream of electric sheep
// Last sync: 2025-06-19 | "60 FPS or die trying"

import * as THREE from 'three';

interface RenderConfig {
  width: number;
  height: number;
  backgroundColor: number;
  gridColor: number;
  candleUpColor: number;
  candleDownColor: number;
  volumeOpacity: number;
  antiAlias: boolean;
  pixelRatio: number;
}

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface RenderStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  points: number;
  memory: number;
  gpuTime: number;
}

export class ChartWebGLRenderer {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private renderStats: RenderStats;
  
  private candleGeometry: THREE.InstancedBufferGeometry;
  private wickGeometry: THREE.InstancedBufferGeometry;
  private volumeGeometry: THREE.InstancedBufferGeometry;
  private gridGeometry: THREE.BufferGeometry;
  
  private candleUpMaterial: THREE.ShaderMaterial;
  private candleDownMaterial: THREE.ShaderMaterial;
  private wickMaterial: THREE.LineBasicMaterial;
  private volumeMaterial: THREE.ShaderMaterial;
  private gridMaterial: THREE.LineBasicMaterial;
  
  private candleMesh: THREE.InstancedMesh;
  private wickMesh: THREE.InstancedMesh;
  private volumeMesh: THREE.InstancedMesh;
  private gridMesh: THREE.LineSegments;
  
  private candles: CandleData[] = [];
  private priceRange = { min: 0, max: 0 };
  private timeRange = { start: 0, end: 0 };
  private maxVolume = 0;
  
  private frameCount = 0;
  private lastFrameTime = performance.now();
  
  private mouse = new THREE.Vector2();
  private raycaster = new THREE.Raycaster();
  private hoveredCandle: number | null = null;
  
  private config: RenderConfig;
  private pulseEffect: THREE.Points | null = null;
  
  constructor(canvas: HTMLCanvasElement, theme: string) {
    this.config = {
      width: 1920,
      height: 1080,
      backgroundColor: theme === 'neon' ? 0x0a0a0f : 0x0a0a0a,
      gridColor: 0x1a1a1a,
      candleUpColor: 0x00ff88,
      candleDownColor: 0xff0044,
      volumeOpacity: 0.3,
      antiAlias: true,
      pixelRatio: window.devicePixelRatio || 1,
    };
    this.renderStats = {
      fps: 0,
      drawCalls: 0,
      triangles: 0,
      points: 0,
      memory: 0,
      gpuTime: 0,
    };
    
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.backgroundColor);
    
    const aspect = this.config.width / this.config.height;
    this.camera = new THREE.OrthographicCamera(
      -aspect * 100,
      aspect * 100,
      100,
      -100,
      0.1,
      1000
    );
    this.camera.position.z = 100;
    
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: this.config.antiAlias,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
      stencil: false,
      depth: false,
    });
    
    this.renderer.setPixelRatio(this.config.pixelRatio);
    this.renderer.setSize(this.config.width, this.config.height);
    
    this.initializeGeometries();
    this.initializeMaterials();
    this.initializeMeshes();
    this.initializeGrid();
    
    this.setupInteraction(canvas);
    this.animate();
  }
  
  private initializeGeometries() {
    this.candleGeometry = new THREE.InstancedBufferGeometry();
    this.wickGeometry = new THREE.InstancedBufferGeometry();
    this.volumeGeometry = new THREE.InstancedBufferGeometry();
    
    const vertices = new Float32Array([
      -0.5, -0.5, 0,
      0.5, -0.5, 0,
      0.5, 0.5, 0,
      -0.5, 0.5, 0,
    ]);
    const indices = [0, 1, 2, 0, 2, 3];
    
    this.candleGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    this.candleGeometry.setIndex(indices);
    this.wickGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    this.wickGeometry.setIndex(indices);
    this.volumeGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    this.volumeGeometry.setIndex(indices);
    
    const maxInstances = 100000;
    const positions = new Float32Array(maxInstances * 3);
    const scales = new Float32Array(maxInstances * 3);
    const colors = new Float32Array(maxInstances * 3);
    
    this.candleGeometry.setAttribute(
      'instancePosition',
      new THREE.InstancedBufferAttribute(positions, 3)
    );
    this.candleGeometry.setAttribute(
      'instanceScale',
      new THREE.InstancedBufferAttribute(scales, 3)
    );
    this.candleGeometry.setAttribute(
      'instanceColor',
      new THREE.InstancedBufferAttribute(colors, 3)
    );
    
    this.wickGeometry.setAttribute(
      'instancePosition',
      new THREE.InstancedBufferAttribute(positions.slice(), 3)
    );
    this.wickGeometry.setAttribute(
      'instanceScale',
      new THREE.InstancedBufferAttribute(scales.slice(), 3)
    );
    
    this.volumeGeometry.setAttribute(
      'instancePosition',
      new THREE.InstancedBufferAttribute(positions.slice(), 3)
    );
    this.volumeGeometry.setAttribute(
      'instanceScale',
      new THREE.InstancedBufferAttribute(scales.slice(), 3)
    );
    this.volumeGeometry.setAttribute(
      'instanceColor',
      new THREE.InstancedBufferAttribute(colors.slice(), 3)
    );
  }
  
  private initializeMaterials() {
    const vertexShader = `
      attribute vec3 instancePosition;
      attribute vec3 instanceScale;
      attribute vec3 instanceColor;
      varying vec3 vColor;
      void main() {
        vColor = instanceColor;
        vec3 transformed = position * instanceScale + instancePosition;
        vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, 1.0);
      }
    `;
    
    this.candleUpMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      side: THREE.DoubleSide,
    }) as any;
    
    this.candleDownMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      side: THREE.DoubleSide,
    }) as any;
    
    this.wickMaterial = new THREE.LineBasicMaterial({
      color: this.config.gridColor,
      linewidth: 1,
    });
    
    const volumeFragmentShader = `
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, ${this.config.volumeOpacity});
      }
    `;
    
    this.volumeMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: volumeFragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
    }) as any;
    
    this.gridMaterial = new THREE.LineBasicMaterial({
      color: this.config.gridColor,
      transparent: true,
      opacity: 0.3,
    });
  }
  
  private initializeMeshes() {
    this.candleMesh = new THREE.InstancedMesh(
      this.candleGeometry,
      this.candleUpMaterial,
      100000
    );
    this.candleMesh.frustumCulled = false;
    this.scene.add(this.candleMesh);
    
    this.wickMesh = new THREE.InstancedMesh(
      this.wickGeometry,
      this.wickMaterial,
      100000
    );
    this.wickMesh.frustumCulled = false;
    this.scene.add(this.wickMesh);
    
    this.volumeMesh = new THREE.InstancedMesh(
      this.volumeGeometry,
      this.volumeMaterial,
      100000
    );
    this.volumeMesh.frustumCulled = false;
    this.volumeMesh.position.z = -1;
    this.scene.add(this.volumeMesh);
  }
  
  private initializeGrid() {
    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const priceLines = 10;
    for (let i = 0; i <= priceLines; i++) {
      const y = (i / priceLines) * 200 - 100;
      positions.push(-200, y, 0, 200, y, 0);
    }
    const timeLines = 20;
    for (let i = 0; i <= timeLines; i++) {
      const x = (i / timeLines) * 400 - 200;
      positions.push(x, -100, 0, x, 100, 0);
    }
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.gridMesh = new THREE.LineSegments(geometry, this.gridMaterial);
    this.gridMesh.position.z = -2;
    this.scene.add(this.gridMesh);
  }
  
  updateData(candles: CandleData[]) {
    this.candles = candles;
    if (candles.length === 0) return;
    
    this.priceRange.min = Infinity;
    this.priceRange.max = -Infinity;
    this.maxVolume = 0;
    
    candles.forEach((candle) => {
      this.priceRange.min = Math.min(this.priceRange.min, candle.low);
      this.priceRange.max = Math.max(this.priceRange.max, candle.high);
      this.maxVolume = Math.max(this.maxVolume, candle.volume);
    });
    
    const priceMargin = (this.priceRange.max - this.priceRange.min) * 0.1;
    this.priceRange.min -= priceMargin;
    this.priceRange.max += priceMargin;
    
    this.timeRange.start = candles[0].time;
    this.timeRange.end = candles[candles.length - 1].time;
    
    this.updateInstances();
  }
  
  private updateInstances() {
    const candlePositions = this.candleGeometry.getAttribute('instancePosition') as THREE.InstancedBufferAttribute;
    const candleScales = this.candleGeometry.getAttribute('instanceScale') as THREE.InstancedBufferAttribute;
    const candleColors = this.candleGeometry.getAttribute('instanceColor') as THREE.InstancedBufferAttribute;
    
    const wickPositions = this.wickGeometry.getAttribute('instancePosition') as THREE.InstancedBufferAttribute;
    const wickScales = this.wickGeometry.getAttribute('instanceScale') as THREE.InstancedBufferAttribute;
    
    const volumePositions = this.volumeGeometry.getAttribute('instancePosition') as THREE.InstancedBufferAttribute;
    const volumeScales = this.volumeGeometry.getAttribute('instanceScale') as THREE.InstancedBufferAttribute;
    const volumeColors = this.volumeGeometry.getAttribute('instanceColor') as THREE.InstancedBufferAttribute;
    
    const priceScale = 200 / (this.priceRange.max - this.priceRange.min);
    const timeScale = 400 / (this.timeRange.end - this.timeRange.start);
    const candleWidth = (400 / this.candles.length) * 0.8;
    
    const upColor = new THREE.Color(this.config.candleUpColor);
    const downColor = new THREE.Color(this.config.candleDownColor);
    
    this.candles.forEach((candle, i) => {
      const isUp = candle.close >= candle.open;
      const x = ((candle.time - this.timeRange.start) * timeScale) - 200;
      const bodyBottom = Math.min(candle.open, candle.close);
      const bodyTop = Math.max(candle.open, candle.close);
      const bodyY = ((bodyBottom + bodyTop) / 2 - this.priceRange.min) * priceScale - 100;
      const bodyHeight = Math.max(0.5, (bodyTop - bodyBottom) * priceScale);
      
      candlePositions.setXYZ(i, x, bodyY, 0);
      candleScales.setXYZ(i, candleWidth, bodyHeight, 1);
      
      const color = isUp ? upColor : downColor;
      candleColors.setXYZ(i, color.r, color.g, color.b);
      
      const wickY = ((candle.low + candle.high) / 2 - this.priceRange.min) * priceScale - 100;
      const wickHeight = (candle.high - candle.low) * priceScale;
      
      wickPositions.setXYZ(i, x, wickY, 0);
      wickScales.setXYZ(i, candleWidth * 0.1, wickHeight, 1);
      
      const volumeHeight = (candle.volume / this.maxVolume) * 50;
      const volumeY = -100 + volumeHeight / 2;
      
      volumePositions.setXYZ(i, x, volumeY, 0);
      volumeScales.setXYZ(i, candleWidth, volumeHeight, 1);
      
      const volumeIntensity = candle.volume / this.maxVolume;
      volumeColors.setXYZ(i, volumeIntensity, volumeIntensity * 0.5, volumeIntensity * 0.2);
    });
    
    this.candleMesh.count = this.candles.length;
    this.wickMesh.count = this.candles.length;
    this.volumeMesh.count = this.candles.length;
    
    candlePositions.needsUpdate = true;
    candleScales.needsUpdate = true;
    candleColors.needsUpdate = true;
    
    wickPositions.needsUpdate = true;
    wickScales.needsUpdate = true;
    
    volumePositions.needsUpdate = true;
    volumeScales.needsUpdate = true;
    volumeColors.needsUpdate = true;
  }
  
  private setupInteraction(canvas: HTMLCanvasElement) {
    canvas.addEventListener('mousemove', (event) => {
      const rect = canvas.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    });
  }
  
  private animate() {
    const currentTime = performance.now();
    this.frameCount++;
    
    if (currentTime >= this.lastFrameTime + 1000) {
      this.renderStats.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFrameTime));
      this.frameCount = 0;
      this.lastFrameTime = currentTime;
    }
    
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }
  
  pulseAtPrice(price: number) {
    if (this.pulseEffect) {
      this.scene.remove(this.pulseEffect);
    }
    
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([0, price, 0]);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({
      color: 0x00ff88,
      size: 5,
      sizeAttenuation: false,
    });
    
    this.pulseEffect = new THREE.Points(geometry, material);
    this.scene.add(this.pulseEffect);
    
    setTimeout(() => {
      if (this.pulseEffect) {
        this.scene.remove(this.pulseEffect);
        this.pulseEffect = null;
      }
    }, 500);
  }
  
  destroy() {
    this.scene.clear();
    this.renderer.dispose();
    this.candleGeometry.dispose();
    this.wickGeometry.dispose();
    this.volumeGeometry.dispose();
    this.gridGeometry.dispose();
    this.candleUpMaterial.dispose();
    this.candleDownMaterial.dispose();
    this.wickMaterial.dispose();
    this.volumeMaterial.dispose();
    this.gridMaterial.dispose();
  }
}