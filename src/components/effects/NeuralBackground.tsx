// src/components/effects/NeuralBackground.tsx
// NEXLIFY NEURAL BACKGROUND - The digital rain that drowns corpo dreams
// Last sync: 2025-06-19 | "We are the ghost in their machine"

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
 * Built this after a 48-hour coding marathon fueled by synth-coffee
 * and pure spite. The corps wanted their sterile Bloomberg terminals.
 * We gave them this - a living, breathing representation of the data
 * streams that flow through the digital underground.
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
    
    // Initialize drops
    if (!ctx.drops) {
      ctx.drops = Array(columns).fill(0);
    }
    
    // Fade effect - ghosting previous frames
    ctx.fillStyle = `rgba(0, 0, 0, ${intensity === 'extreme' ? 0.02 : 0.05})`;
    ctx.fillRect(0, 0, width, height);
    
    // Set text properties
    ctx.font = `${charSize}px "Courier New", monospace`;
    ctx.textAlign = 'center';
    
    // Draw characters
    for (let i = 0; i < ctx.drops.length; i++) {
      const char = chars[Math.floor(Math.random() * chars.length)];
      const x = i * charSize + charSize / 2;
      const y = ctx.drops[i] * charSize;
      
      // Color variation based on position and interaction
      const distToMouse = interactive ? 
        Math.sqrt(Math.pow(x - mouseRef.current.x, 2) + 
                  Math.pow(y - mouseRef.current.y, 2)) : Infinity;
      
      const brightness = distToMouse < 100 ? 1 : 
                        distToMouse < 200 ? 0.8 : 0.6;
      
      ctx.fillStyle = color.replace(')', `, ${brightness})`).replace('rgb', 'rgba');
      ctx.fillText(char, x, y);
      
      // Reset drop when it reaches bottom or randomly
      if (y > height || Math.random() > 0.98) {
        ctx.drops[i] = 0;
      }
      
      // Variable speed based on intensity
      ctx.drops[i] += intensity === 'extreme' ? 1.5 :
                     intensity === 'high' ? 1 :
                     intensity === 'medium' ? 0.7 : 0.5;
    }
  }, [color, intensity, interactive]);
  
  /**
   * Circuit board effect - Neural pathways of profit
   * 
   * Each connection is a correlation, each node a decision point.
   * Watch the signals propagate. Information is currency here.
   */
  const drawCircuit = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    ctx.clearRect(0, 0, width, height);
    
    // Grid of nodes
    const nodeSize = intensity === 'extreme' ? 30 : 50;
    const nodes: Array<{x: number, y: number, active: boolean}> = [];
    
    for (let x = nodeSize; x < width; x += nodeSize * 2) {
      for (let y = nodeSize; y < height; y += nodeSize * 2) {
        nodes.push({
          x: x + (Math.random() - 0.5) * nodeSize * 0.3,
          y: y + (Math.random() - 0.5) * nodeSize * 0.3,
          active: Math.random() > 0.7
        });
      }
    }
    
    // Draw connections
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 1;
    
    nodes.forEach((node, i) => {
      if (!node.active) return;
      
      // Connect to nearby nodes
      nodes.forEach((other, j) => {
        if (i === j || !other.active) return;
        
        const dist = Math.sqrt(
          Math.pow(node.x - other.x, 2) + 
          Math.pow(node.y - other.y, 2)
        );
        
        if (dist < nodeSize * 3) {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          
          // Circuit-style routing (90-degree angles)
          if (Math.random() > 0.5) {
            ctx.lineTo(node.x, other.y);
            ctx.lineTo(other.x, other.y);
          } else {
            ctx.lineTo(other.x, node.y);
            ctx.lineTo(other.x, other.y);
          }
          
          ctx.stroke();
        }
      });
    });
    
    // Draw nodes
    nodes.forEach(node => {
      // Glow effect
      if (node.active) {
        const gradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, 10
        );
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fillRect(node.x - 10, node.y - 10, 20, 20);
      }
      
      // Node core
      ctx.fillStyle = node.active ? color : color + '40';
      ctx.fillRect(node.x - 3, node.y - 3, 6, 6);
    });
    
    // Traveling pulses
    const pulsePhase = (frame * 0.02) % 1;
    ctx.shadowBlur = 20;
    ctx.shadowColor = color;
    
    nodes.forEach((node, i) => {
      if (node.active && Math.random() > 0.98) {
        const targetNode = nodes[Math.floor(Math.random() * nodes.length)];
        if (targetNode.active) {
          const x = node.x + (targetNode.x - node.x) * pulsePhase;
          const y = node.y + (targetNode.y - node.y) * pulsePhase;
          
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
    
    ctx.shadowBlur = 0;
  }, [color, intensity]);
  
  /**
   * Glitch effect - Reality.exe has stopped responding
   * 
   * When the system breaks, we feast on the chaos.
   * Each glitch is an opportunity, each error a doorway.
   */
  const drawGlitch = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    frame: number
  ) => {
    // Save current image data
    const imageData = ctx.getImageData(0, 0, width, height);
    
    // Clear with slight fade
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, width, height);
    
    // Glitch blocks
    const glitchCount = intensity === 'extreme' ? 20 :
                       intensity === 'high' ? 15 :
                       intensity === 'medium' ? 10 : 5;
    
    for (let i = 0; i < glitchCount; i++) {
      if (Math.random() > 0.3) continue;
      
      const blockWidth = Math.random() * width * 0.3;
      const blockHeight = Math.random() * 50 + 5;
      const x = Math.random() * width;
      const y = Math.random() * height;
      
      // Color channel shift
      const shift = Math.random() * 20 - 10;
      
      ctx.fillStyle = `rgba(${
        Math.random() > 0.5 ? '255, 0, 0' : '0, 255, 255'
      }, ${Math.random() * 0.5})`;
      
      ctx.fillRect(x + shift, y, blockWidth, blockHeight);
    }
    
    // Scan lines
    ctx.strokeStyle = color + '20';
    ctx.lineWidth = 1;
    
    for (let y = 0; y < height; y += 4) {
      if (Math.random() > 0.8) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }
    
    // Digital noise
    if (intensity === 'high' || intensity === 'extreme') {
      for (let i = 0; i < 1000; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const brightness = Math.random();
        
        ctx.fillStyle = `rgba(0, 255, 255, ${brightness})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
    
    // Corrupted text
    if (Math.random() > 0.95) {
      ctx.font = '20px monospace';
      ctx.fillStyle = color;
      const glitchText = ['ERROR', '0xDEADBEEF', 'SEGFAULT', 'PROFIT.EXE', '///'];
      const text = glitchText[Math.floor(Math.random() * glitchText.length)];
      
      ctx.fillText(text, Math.random() * width, Math.random() * height);
    }
  }, [color, intensity]);
  
  /**
   * Wave effect - The tides of the market
   * 
   * Bulls and bears, fear and greed, pump and dump.
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
    
    // Perspective transform
    const perspective = 400;
    const centerX = width / 2;
    const centerY = height / 2;
    
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 1;
    
    // Horizontal lines
    for (let y = -height; y < height * 2; y += gridSize) {
      ctx.beginPath();
      
      for (let x = -width; x < width * 2; x += gridSize) {
        const z = (y - centerY + frame * 2) % (height * 2);
        const scale = perspective / (perspective + z);
        
        const projX = centerX + (x - centerX) * scale;
        const projY = centerY + (y - centerY - frame * 2) * scale;
        
        if (x === -width) {
          ctx.moveTo(projX, projY);
        } else {
          ctx.lineTo(projX, projY);
        }
      }
      
      ctx.stroke();
    }
    
    // Vertical lines
    for (let x = -width; x < width * 2; x += gridSize) {
      ctx.beginPath();
      
      for (let y = -height; y < height * 2; y += gridSize) {
        const z = (y - centerY + frame * 2) % (height * 2);
        const scale = perspective / (perspective + z);
        
        const projX = centerX + (x - centerX) * scale;
        const projY = centerY + (y - centerY - frame * 2) * scale;
        
        if (y === -height) {
          ctx.moveTo(projX, projY);
        } else {
          ctx.lineTo(projX, projY);
        }
      }
      
      ctx.stroke();
    }
    
    // Glow at intersection with mouse
    if (interactive && isHovered) {
      const glowRadius = 100;
      const gradient = ctx.createRadialGradient(
        mouseRef.current.x, mouseRef.current.y, 0,
        mouseRef.current.x, mouseRef.current.y, glowRadius
      );
      gradient.addColorStop(0, color + '80');
      gradient.addColorStop(1, 'transparent');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(
        mouseRef.current.x - glowRadius,
        mouseRef.current.y - glowRadius,
        glowRadius * 2,
        glowRadius * 2
      );
    }
  }, [color, intensity, interactive, isHovered]);
  
  /**
   * Particle system - The atoms of profit and loss
   * 
   * Each particle is a trade, a decision, a possibility.
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
                  intensity === 'medium' ? 0.4 : 0.3
        }}
      />
      
      {/* Overlay gradient for depth */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-900/50 to-gray-900" />
      
      {/* Scanline effect */}
      {(variant === 'glitch' || variant === 'circuit') && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          animate={{
            backgroundPosition: ['0% 0%', '0% 100%']
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'linear'
          }}
          style={{
            backgroundImage: `repeating-linear-gradient(
              0deg,
              transparent,
              transparent 2px,
              ${color}10 2px,
              ${color}10 4px
            )`,
            backgroundSize: '100% 4px'
          }}
        />
      )}
    </div>
  );
};

/**
 * NEURAL BACKGROUND WISDOM:
 * 
 * Every effect here serves a purpose beyond aesthetics:
 * 
 * 1. Matrix rain reminds us we're all just data in someone else's
 *    algorithm. But we can learn to read the patterns.
 * 
 * 2. Circuit boards show the connections. In trading, everything
 *    is connected. Miss one signal, lose the whole game.
 * 
 * 3. Glitch effects represent opportunity. When systems break,
 *    smart traders profit from the chaos.
 * 
 * 4. Waves visualize market cycles. What goes up must come down.
 *    What crashes must bounce. Eventually.
 * 
 * 5. The grid is the underlying structure of the market. Hidden
 *    from retail, visible to those who know where to look.
 * 
 * 6. Interactive particles respond to your presence. Just like
 *    the market responds to your trades. You are never just an
 *    observer.
 * 
 * This isn't just eye candy, choom. It's a meditation on the nature
 * of digital finance. Stare long enough, and you'll see the patterns
 * that make millionaires.
 * 
 * "The street always wins. But sometimes, just sometimes, we can
 * ride along for a few blocks." - Night City Proverb
 */
