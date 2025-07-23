// Location: nexlify-dashboard/src/systems/adaptive/components/AdaptiveChromeButton.tsx
// Mission: 80-I.1 Adaptive Chrome Button
// Dependencies: useAdaptiveVisuals hook
// Context: Button that adapts visual complexity to hardware

import React, { useCallback, useState } from 'react';
import { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';

interface AdaptiveChromeButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  priority: 'critical' | 'high' | 'normal' | 'low';
  variant?: 'primary' | 'danger' | 'ghost';
  disabled?: boolean;
  className?: string;
  loading?: boolean;
}

export const AdaptiveChromeButton: React.FC<AdaptiveChromeButtonProps> = ({
  children,
  onClick,
  priority,
  variant = 'primary',
  disabled = false,
  className = '',
  loading = false
}) => {
  const { features, audioEngine } = useAdaptiveVisuals();
  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);
  
  const canAnimate = features?.neonGlow.enabled && !disabled && !loading;
  const hasAudio = features?.audioVisualization.enabled && audioEngine;
  
  const handleClick = useCallback(() => {
    if (disabled || loading) return;
    
    if (hasAudio) {
      audioEngine?.playUIFeedback('click');
    }
    
    onClick?.();
  }, [disabled, loading, hasAudio, audioEngine, onClick]);
  
  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
    
    if (hasAudio && canAnimate) {
      audioEngine?.playUIFeedback('hover');
    }
  }, [hasAudio, canAnimate, audioEngine]);
  
  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    setIsPressed(false);
  }, []);
  
  const handleMouseDown = useCallback(() => {
    setIsPressed(true);
  }, []);
  
  const handleMouseUp = useCallback(() => {
    setIsPressed(false);
  }, []);
  
  // Base styles
  const baseStyles = "relative px-6 py-3 font-mono text-sm uppercase tracking-wider transition-all duration-200 overflow-hidden";
  
  // Variant styles
  const variantStyles = {
    primary: "bg-cyan-900/50 text-cyan-400 border border-cyan-500/50",
    danger: "bg-red-900/50 text-red-400 border border-red-500/50",
    ghost: "bg-transparent text-gray-400 border border-gray-700"
  };
  
  // State styles
  const stateStyles = disabled || loading
    ? "opacity-50 cursor-not-allowed"
    : "cursor-pointer";
  
  // Adaptive hover/active styles
  const interactionStyles = canAnimate && !disabled ? {
    ...(isHovered && {
      borderColor: variant === 'primary' ? '#00ffff' : variant === 'danger' ? '#ff0040' : '#888888',
      boxShadow: features?.neonGlow.intensity > 0.5 
        ? `0 0 ${10 * features.neonGlow.intensity}px ${variant === 'primary' ? '#00ffff' : '#ff0040'}40`
        : undefined,
      transform: features?.shaderEffects.enabled ? 'translateY(-1px)' : undefined
    }),
    ...(isPressed && {
      transform: 'translateY(1px)',
      boxShadow: 'none'
    })
  } : {};
  
  // Corner cut style (cyberpunk aesthetic)
  const clipPath = features?.shaderEffects.enabled
    ? 'polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px))'
    : undefined;
  
  return (
    <button
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      disabled={disabled || loading}
      className={`${baseStyles} ${variantStyles[variant]} ${stateStyles} ${className}`}
      style={{
        clipPath,
        ...interactionStyles
      }}
    >
      {/* Background effects for high-end hardware */}
      {canAnimate && features?.particleSystem.enabled && isHovered && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
        </div>
      )}
      
      {/* Loading spinner */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      
      {/* Button content */}
      <span className="relative z-10">
        {children}
      </span>
      
      {/* Glitch effect on hover for high priority buttons */}
      {canAnimate && priority === 'critical' && isHovered && features?.glitchEffects.enabled && (
        <span 
          className="absolute inset-0 flex items-center justify-center opacity-50"
          style={{
            transform: 'translate(2px, -1px)',
            color: '#ff00ff',
            clipPath
          }}
          aria-hidden="true"
        >
          {children}
        </span>
      )}
    </button>
  );
};