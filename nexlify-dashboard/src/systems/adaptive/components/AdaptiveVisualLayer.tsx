// Location: nexlify-dashboard/src/systems/adaptive/components/AdaptiveVisualLayer.tsx
// Mission: 80-I.1 Main Visual Rendering Layer - FIXED
// Dependencies: All previous components
// Context: The actual visual layer that renders based on features

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';
import { performanceMonitor } from '../core/PerformanceMonitor';

export const AdaptiveVisualLayer: React.FC = () => {
  const {
    features,
    capabilities,
    shaderManager,
    audioEngine,
    performanceMode
  } = useAdaptiveVisuals();
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number>();
  const lastFrameTime = useRef<number>(0);
  const [isRendering, setIsRendering] = useState(false);
  
  // Initialize rendering
  useEffect(() => {
    if (!features || !capabilities || !shaderManager) return;
    
    const canvas = shaderManager.getCanvas();
    if (!canvas) return;
    
    // Only add canvas to DOM if not already added
    if (canvas.parentElement !== document.body) {
      // Remove any orphaned canvases first
      document.querySelectorAll('canvas[data-adaptive="true"]').forEach(c => {
        if (c !== canvas && c.parentElement) {
          c.parentElement.removeChild(c);
        }
      });
      
      document.body.appendChild(canvas);
    }
    
    canvasRef.current = canvas;
    setIsRendering(true);
    
    // Start render loop
    const render = (timestamp: number) => {
      if (!canvasRef.current) return;
      
      // Track frame timing
      performanceMonitor.recordFrame();
      
      const deltaTime = timestamp - lastFrameTime.current;
      lastFrameTime.current = timestamp;
      
      // Render effects based on features
      renderFrame(timestamp, deltaTime);
      
      animationRef.current = requestAnimationFrame(render);
    };
    
    animationRef.current = requestAnimationFrame(render);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      setIsRendering(false);
    };
  }, [features, capabilities, shaderManager]);
  
  const renderFrame = useCallback((timestamp: number, deltaTime: number) => {
    if (!shaderManager || !features) return;
    
    const gl = shaderManager.getGLContext();
    if (!gl) return;
    
    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Render effects in order
    if (features?.matrixRain?.enabled) {
      renderMatrixRain(timestamp);
    }
    
	// DISABLE SCANLINES TO REMOVE PULSING
	/*
    if (features?.scanlines?.enabled) {
      renderScanlines(timestamp);
    }
    */
	
    if (features?.glitchEffects?.enabled && shouldGlitch(timestamp)) {
      renderGlitch(timestamp);
    }
    
    // Post-processing effects would go here
    if (features?.postProcessing?.enabled) {
      // Render to framebuffer first, then apply post-processing
    }
  }, [shaderManager, features]);
  
  const renderMatrixRain = useCallback((timestamp: number) => {
    if (!shaderManager || !features) return;
    
    const program = shaderManager.useProgram('matrixRain');
    if (!program) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Set uniforms
    shaderManager.setUniform('matrixRain', 'u_time', timestamp / 1000);
    shaderManager.setUniform('matrixRain', 'u_resolution', [canvas.width, canvas.height]);
    shaderManager.setUniform('matrixRain', 'u_density', Math.max(0.1, features?.matrixRain?.density || 0.5));
	shaderManager.setUniform('matrixRain', 'u_speed', Math.max(0.1, features?.matrixRain?.speed || 0.5));
    shaderManager.setUniform('matrixRain', 'u_complexity', 
      features?.matrixRain?.complexity === 'complex' ? 1.0 :
      features?.matrixRain?.complexity === 'standard' ? 0.5 : 0.0
    );
    
    // Draw full-screen quad
    drawFullScreenQuad();
  }, [shaderManager, features]);
  
  const renderScanlines = useCallback((timestamp: number) => {
    if (!shaderManager || !features) return;
    
    const program = shaderManager.useProgram('scanlines');
    if (!program) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Set uniforms
    shaderManager.setUniform('scanlines', 'u_time', timestamp / 1000);
    shaderManager.setUniform('scanlines', 'u_resolution', [canvas.width, canvas.height]);
    shaderManager.setUniform('scanlines', 'u_intensity', features?.scanlines?.intensity || 0);
    shaderManager.setUniform('scanlines', 'u_thickness', features?.scanlines?.thickness || 1);
    shaderManager.setUniform('scanlines', 'u_speed', features?.scanlines?.speed || 0);
    
    // Draw full-screen quad
    drawFullScreenQuad();
  }, [shaderManager, features]);
  
  const shouldGlitch = useCallback((timestamp: number): boolean => {
    if (!features) return false;
    
    // Random glitch based on probability
    return Math.random() < (features?.glitchEffects?.probability || 0);
  }, [features]);
  
  const renderGlitch = useCallback((timestamp: number) => {
    if (!shaderManager || !features || !audioEngine) return;
    
    // Play glitch sound
    if (features?.audioVisualization?.enabled) {
      audioEngine.playEffect('glitch', { volume: 0.3 });
    }
    
    // Glitch visual effect would be implemented here
    // This would involve copying the current framebuffer and distorting it
  }, [shaderManager, features, audioEngine]);
  
  const drawFullScreenQuad = useCallback(() => {
    const gl = shaderManager?.getGLContext();
    if (!gl) return;
    
    // Use cached quad buffer or create one
    // For now, using immediate mode for simplicity
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const positionAttrib = 0; // Assuming location 0
    gl.enableVertexAttribArray(positionAttrib);
    gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);
    
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    gl.deleteBuffer(buffer);
  }, [shaderManager]);
  
  // Performance overlay (debug mode)
  if (process.env.NODE_ENV === 'development' && performanceMode !== 'trading') {
    return (
      <>
        <div className="fixed top-4 right-4 bg-black/80 p-2 rounded text-xs font-mono text-green-400 z-50">
          <div>Rendering: {isRendering ? 'Active' : 'Inactive'}</div>
          <div>Features: {Object.values(features || {}).filter((f: any) => f.enabled).length}</div>
          <div>Canvas Count: {document.querySelectorAll('canvas').length}</div>
        </div>
      </>
    );
  }
  
  return null; // Canvas is added directly to body
};