// src/components/effects/NeuralBackground.tsx
// NEXLIFY NEURAL BACKGROUND - The digital rain that drowns corpo dreams
// Last sync: 2025-06-21 | "We are the ghost in their machine"

import { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';

interface NeuralBackgroundProps {
  variant?: 'matrix' | 'circuit' | 'glitch' | 'waves' | 'grid';
  intensity?: 'low' | 'medium' | 'high' | 'extreme';
  color?: string;
  interactive?: boolean;
  particles?: boolean;
  fps?: number;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  life: number;
  maxLife: number;
  color: string;
  connections: number[];
}

/**
 * NEURAL BACKGROUND - Where cyberspace bleeds into reality
 * 
 * Fixed the TypeScript matrix glitch. The drops now flow through
 * proper memory channels, not hacked onto the canvas context.
 * 
 * Each effect tells a story:
 * - Matrix: Classic digital rain, because we're all living in it
 * - Circuit: Neural pathways of the market's collective consciousness  
 * - Glitch: When reality breaks, profit leaks through the cracks
 * - Waves: The ebb and flow of greed and fear
 * - Grid: The underlying structure they don't want you to see
 * 
 * This isn't decoration. It's a reminder that we operate in the spaces
 * between their rules, in the glitches they can't patch.
 */
export const NeuralBackground = ({
  variant = 'matrix',
  intensity = 'medium',
  color = '#00ffff',
  interactive = true,
  particles = true,
  fps = 30
}: NeuralBackgroundProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const mouseRef = useRef({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  
  // Matrix rain drops - stored properly now
  const dropsRef = useRef<number[]>([]);
  
  // Performance optimization - frame limiting
  const frameInterval = 1000 / fps;
  let then = Date.now();
  
  /**
   * Matrix rain effect - The classic
   * 
   * Every character represents a trade, a decision, a dream.
   * Watch them fall. Most hit bottom. Some find their mark.
   */
  const drawMatrixRain = useCallback((
    ctx: CanvasRenderingContext2D, 
    width: number, 
    height: number,
    frame: number
  ) => {
    // Character set - mix of code and market symbols
    const chars = '01アイウエオカキクケコサシスセソ$€¥£₿%+-*/=<>[]{}()';
    const charSize = intensity === 'extreme' ? 10 : 
                    intensity === 'high' ? 14 : 
                    intensity === 'medium' ? 18 : 22;
    const columns = Math.floor(width / charSize);
    
    // Initialize drops if needed
    if (dropsRef.current.length !== columns) {
      dropsRef.current = Array(columns).fill(0);
    }
    
    // Fade effect - ghosting previous frames
    ctx.fillStyle = `rgba(0, 0, 0, ${intensity === 'extreme' ? 0.02 : 
                                     intensity === 'high' ? 0.05 : 
                                     intensity === 'medium' ? 0.08 : 0.1})`;
    ctx.fillRect(0, 0, width, height);
    
    // Set text style
    ctx.fillStyle = color;
    ctx.font = `${charSize}px monospace`;
    ctx.textAlign = 'center';
    
    // Draw characters
    for (let i = 0; i < dropsRef.current.length; i++) {
      // Random character
      const char = chars[Math.floor(Math.random() * chars.length)];
      const x = i * charSize + charSize / 2;
      const y = dropsRef.current[i] * charSize;
      
      // Brightness based on position
      const brightness = 1 - (y / height) * 0.5;
      ctx.globalAlpha = brightness;
      
      // Interactive effect - brighter near mouse
      if (interactive && mouseRef.current.x && mouseRef.current.y) {
        const dist = Math.sqrt(
          Math.pow(x - mouseRef.current.x, 2) + 
          Math.pow(y - mouseRef.current.y, 2)
        );
        if (dist < 100) {
          ctx.globalAlpha = 1;
          ctx.shadowBlur = 20;
          ctx.shadowColor = color;
        }
      }
      
      ctx.fillText(char, x, y);
      
      // Reset shadow
      ctx.shadowBlur = 0;
      
      // Move drop
      if (y > height && Math.random() > 0.975) {
        dropsRef.current[i] = 0;
      }
      
      // Update position based on intensity
      dropsRef.current[i] += intensity === 'extreme' ? 1.5 :
                             intensity === 'high' ? 1.2 :
                             intensity === 'medium' ? 1 : 0.8;
    }
    
    ctx.globalAlpha = 1;
  }, [color, intensity, interactive]);
  
  /**
   * Circuit effect - Neural pathways
   * 
   * Each connection is a trade route, each node a decision point.
   * The market is a circuit. Current flows where resistance is lowest.
   */
  const drawCircuit = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    // Clear with fade
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, width, height);
    
    // Circuit parameters
    const nodeCount = intensity === 'extreme' ? 30 :
                     intensity === 'high' ? 20 :
                     intensity === 'medium' ? 15 : 10;
    const connectionDistance = 150;
    
    // Generate nodes
    const nodes = [];
    for (let i = 0; i < nodeCount; i++) {
      const angle = (i / nodeCount) * Math.PI * 2 + frame * 0.001;
      const radius = 100 + Math.sin(angle * 3) * 50;
      nodes.push({
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        pulse: Math.sin(frame * 0.05 + i) * 0.5 + 0.5
      });
    }
    
    // Draw connections
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 1;
    
    nodes.forEach((node, i) => {
      nodes.forEach((other, j) => {
        if (i >= j) return;
        
        const dist = Math.sqrt(
          Math.pow(node.x - other.x, 2) + 
          Math.pow(node.y - other.y, 2)
        );
        
        if (dist < connectionDistance) {
          const opacity = 1 - (dist / connectionDistance);
          ctx.globalAlpha = opacity * 0.5;
          
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(other.x, other.y);
          ctx.stroke();
          
          // Data packets
          if (Math.random() < 0.01) {
            const progress = (frame % 100) / 100;
            const px = node.x + (other.x - node.x) * progress;
            const py = node.y + (other.y - node.y) * progress;
            
            ctx.fillStyle = color;
            ctx.globalAlpha = 1;
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      });
    });
    
    // Draw nodes
    nodes.forEach(node => {
      ctx.fillStyle = color;
      ctx.globalAlpha = node.pulse;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 4 + node.pulse * 2, 0, Math.PI * 2);
      ctx.fill();
      
      // Node glow
      ctx.strokeStyle = color;
      ctx.globalAlpha = node.pulse * 0.5;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 8 + node.pulse * 4, 0, Math.PI * 2);
      ctx.stroke();
    });
    
    ctx.globalAlpha = 1;
  }, [color, intensity]);
  
  /**
   * Glitch effect - Reality breaking
   * 
   * Sometimes the system shows its true face. Fragmented. Broken.
   * In those moments, fortunes are made by those who see clearly.
   */
  const drawGlitch = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    // Random glitch intensity
    if (Math.random() > 0.95) {
      // Save current canvas
      const imageData = ctx.getImageData(0, 0, width, height);
      
      // Glitch parameters
      const glitchHeight = Math.random() * 100 + 20;
      const glitchY = Math.random() * (height - glitchHeight);
      const offset = (Math.random() - 0.5) * 50;
      
      // Shift section
      ctx.putImageData(imageData, offset, 0);
      
      // Color channel separation
      if (Math.random() > 0.5) {
        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
        ctx.fillRect(0, glitchY, width, glitchHeight);
        ctx.fillStyle = 'rgba(0, 255, 255, 0.1)';
        ctx.fillRect(offset * 2, glitchY, width, glitchHeight);
      }
      
      ctx.globalCompositeOperation = 'source-over';
    }
    
    // Scanlines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    
    for (let y = 0; y < height; y += 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Noise
    if (frame % 2 === 0) {
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const size = Math.random() * 2;
        
        ctx.fillStyle = `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.5)`;
        ctx.fillRect(x, y, size, size);
      }
    }
  }, []);
  
  /**
   * Wave effect - Market oscillations
   * 
   * Bull runs, bear markets, sideways chop.
   * It all moves in waves. Ride them or drown.
   */
  const drawWaves = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, width, height);
    
    const waveCount = intensity === 'extreme' ? 8 :
                     intensity === 'high' ? 6 :
                     intensity === 'medium' ? 4 : 2;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    for (let w = 0; w < waveCount; w++) {
      ctx.beginPath();
      ctx.globalAlpha = 0.5 - (w * 0.05);
      
      for (let x = 0; x <= width; x += 5) {
        const baseY = height / 2 + (w * 30);
        const amplitude = 50 - (w * 5);
        const frequency = 0.01 + (w * 0.002);
        const phase = frame * 0.02 + w;
        
        const y = baseY + Math.sin(x * frequency + phase) * amplitude;
        
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
    }
    
    ctx.globalAlpha = 1;
  }, [color, intensity]);
  
  /**
   * Grid effect - The underlying structure
   * 
   * They want you to see chaos. But look closer.
   * There's always a pattern. Always a system. Find it, exploit it.
   */
  const drawGrid = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    ctx.clearRect(0, 0, width, height);
    
    const gridSize = intensity === 'extreme' ? 20 :
                    intensity === 'high' ? 30 :
                    intensity === 'medium' ? 40 : 50;
    
    ctx.strokeStyle = color + '20';
    ctx.lineWidth = 1;
    
    // Perspective transform
    const perspectiveY = height * 0.3;
    const vanishingPoint = { x: width / 2, y: perspectiveY };
    
    // Horizontal lines with perspective
    for (let y = perspectiveY; y < height; y += gridSize) {
      const progress = (y - perspectiveY) / (height - perspectiveY);
      const lineWidth = progress * width;
      const startX = vanishingPoint.x - lineWidth / 2;
      
      ctx.globalAlpha = progress * 0.5;
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(startX + lineWidth, y);
      ctx.stroke();
    }
    
    // Vertical lines with perspective
    const verticalLines = 20;
    for (let i = 0; i <= verticalLines; i++) {
      const angle = (i / verticalLines - 0.5) * Math.PI;
      const startX = vanishingPoint.x + Math.sin(angle) * width;
      
      ctx.beginPath();
      ctx.moveTo(vanishingPoint.x, vanishingPoint.y);
      ctx.lineTo(startX, height);
      ctx.stroke();
    }
    
    // Animated highlight
    const highlightY = perspectiveY + (frame % 100) * ((height - perspectiveY) / 100);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 1 - ((frame % 100) / 100);
    ctx.lineWidth = 2;
    
    const highlightProgress = (highlightY - perspectiveY) / (height - perspectiveY);
    const highlightWidth = highlightProgress * width;
    const highlightStartX = vanishingPoint.x - highlightWidth / 2;
    
    ctx.beginPath();
    ctx.moveTo(highlightStartX, highlightY);
    ctx.lineTo(highlightStartX + highlightWidth, highlightY);
    ctx.stroke();
    
    ctx.globalAlpha = 1;
  }, [color, intensity]);
  
  /**
   * Particle system update - Digital life
   * 
   * Each particle is a trader, a transaction, a dream.
   * Watch them connect, disconnect, live, die. Just like traders.
   */
  const updateParticles = useCallback((
    particles: Particle[],
    width: number,
    height: number
  ): Particle[] => {
    return particles.map(p => {
      // Update position
      p.x += p.vx;
      p.y += p.vy;
      
      // Wrap around edges
      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;
      
      // Update life
      p.life--;
      
      // Respawn if dead
      if (p.life <= 0) {
        p.x = Math.random() * width;
        p.y = Math.random() * height;
        p.vx = (Math.random() - 0.5) * 2;
        p.vy = (Math.random() - 0.5) * 2;
        p.life = p.maxLife;
      }
      
      // Find connections
      p.connections = [];
      particles.forEach((other, i) => {
        if (other === p) return;
        
        const dist = Math.sqrt(
          Math.pow(p.x - other.x, 2) + 
          Math.pow(p.y - other.y, 2)
        );
        
        if (dist < 100) {
          p.connections.push(i);
        }
      });
      
      return p;
    });
  }, []);
  
  /**
   * Main animation loop - Where the magic happens
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Initialize particles if enabled
    let particleSystem: Particle[] = [];
    if (particles) {
      const particleCount = intensity === 'extreme' ? 100 :
                          intensity === 'high' ? 75 :
                          intensity === 'medium' ? 50 : 25;
      
      for (let i = 0; i < particleCount; i++) {
        particleSystem.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          size: Math.random() * 3 + 1,
          life: Math.random() * 100 + 100,
          maxLife: 200,
          color: color,
          connections: []
        });
      }
    }
    
    let frame = 0;
    
    const animate = () => {
      const now = Date.now();
      const elapsed = now - then;
      
      if (elapsed > frameInterval) {
        then = now - (elapsed % frameInterval);
        
        // Draw based on variant
        switch (variant) {
          case 'matrix':
            drawMatrixRain(ctx, canvas.width, canvas.height, frame);
            break;
          case 'circuit':
            drawCircuit(ctx, canvas.width, canvas.height, frame);
            break;
          case 'glitch':
            drawGlitch(ctx, canvas.width, canvas.height, frame);
            break;
          case 'waves':
            drawWaves(ctx, canvas.width, canvas.height, frame);
            break;
          case 'grid':
            drawGrid(ctx, canvas.width, canvas.height, frame);
            break;
        }
        
        // Draw particles on top
        if (particles && particleSystem.length > 0) {
          particleSystem = updateParticles(particleSystem, canvas.width, canvas.height);
          
          // Draw connections
          ctx.strokeStyle = color + '20';
          ctx.lineWidth = 1;
          
          particleSystem.forEach((p, i) => {
            p.connections.forEach(ci => {
              const other = particleSystem[ci];
              ctx.beginPath();
              ctx.moveTo(p.x, p.y);
              ctx.lineTo(other.x, other.y);
              ctx.stroke();
            });
          });
          
          // Draw particles
          particleSystem.forEach(p => {
            const alpha = p.life / p.maxLife;
            ctx.fillStyle = p.color.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
          });
        }
        
        frame++;
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [variant, intensity, color, interactive, particles, fps, frameInterval,
      drawMatrixRain, drawCircuit, drawGlitch, drawWaves, drawGrid, updateParticles]);
  
  /**
   * Mouse tracking for interactive effects
   */
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!interactive) return;
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
    }
  }, [interactive]);
  
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{ 
          opacity: intensity === 'extreme' ? 0.8 :
                  intensity === 'high' ? 0.6 :
                  intensity === 'medium' ? 0.4 : 0.3,
          mixBlendMode: 'screen'
        }}
      />
      
      {/* Overlay gradient for depth */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 50% 50%, transparent 0%, rgba(0,0,0,0.5) 100%)`
        }}
      />
    </div>
  );
};

/**
 * NEURAL BACKGROUND WISDOM - CYBERPUNK EDITION:
 * 
 * 1. Canvas context is read-only. Don't hack properties onto it.
 *    Use refs, closures, or external state. The matrix has rules.
 * 
 * 2. Performance matters. 60 FPS keeps traders happy. 30 FPS makes
 *    them nervous. Below that, they think your app is broken.
 * 
 * 3. Every effect has meaning. Matrix rain = data flow. Circuits =
 *    connections. Glitches = opportunities. Choose wisely.
 * 
 * 4. Interactive effects create engagement. When the background
 *    responds to mouse movement, users feel connected to the data.
 * 
 * 5. Intensity levels let users choose their own adventure. Some
 *    want subtle ambiance. Others want full cyberpunk overload.
 * 
 * Remember: The background sets the mood. Make it memorable,
 * make it smooth, make it cyberpunk as fuck.
 */