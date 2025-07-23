// Location: nexlify-dashboard/src/systems/adaptive/components/AdaptiveNeonText.tsx
// Mission: 80-I.1 Adaptive Neon Text Component
// Dependencies: useAdaptiveVisuals hook
// Context: Text that glows based on hardware capabilities

import React, { useMemo } from 'react';
import { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';

interface AdaptiveNeonTextProps {
  children: React.ReactNode;
  importance: 'critical' | 'high' | 'medium' | 'low';
  color?: 'cyan' | 'magenta' | 'yellow' | 'green' | 'red';
  className?: string;
  pulse?: boolean;
}

export const AdaptiveNeonText: React.FC<AdaptiveNeonTextProps> = ({
  children,
  importance,
  color = 'cyan',
  className = '',
  pulse = false
}) => {
  const { features } = useAdaptiveVisuals();
  
  const shouldGlow = useMemo(() => {
    if (!features?.neonGlow?.enabled) return false;
    
    const intensityThreshold = {
      critical: 0,
      high: 0.5,
      medium: 1.0,
      low: 1.5
    };
    
    return features.neonGlow.intensity >= intensityThreshold[importance];
  }, [features, importance]);
  
  const glowStyle = useMemo(() => {
    if (!shouldGlow || !features) return {};
    
    const colorMap = {
      cyan: '#00ffff',
      magenta: '#ff00ff',
      yellow: '#ffff00',
      green: '#00ff80',
      red: '#ff0040'
    };
    
    const baseColor = colorMap[color];
    const shadows: string[] = [];
    
    // Generate glow layers based on feature settings
    for (let i = 0; i < features.neonGlow.layers; i++) {
      const spread = (i + 1) * 5 * features.neonGlow.intensity;
      const opacity = 0.8 - (i * 0.15);
      shadows.push(`0 0 ${spread}px ${baseColor}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`);
    }
    
    return {
      color: baseColor,
      textShadow: shadows.join(', '),
      transition: 'all 0.3s ease',
      ...(pulse && features.neonGlow.pulseFrequency > 0 ? {
        animation: `neon-pulse ${1 / features.neonGlow.pulseFrequency}s ease-in-out infinite`
      } : {})
    };
  }, [shouldGlow, features, color, pulse]);
  
  return (
    <span 
      className={`relative ${className}`}
      style={glowStyle}
    >
      {children}
      
      {/* Reflection effect for high-end hardware */}
      {shouldGlow && features?.neonGlow.intensity > 1.5 && (
        <span 
          className="absolute top-0 left-0 opacity-30 transform scale-y-[-1] translate-y-full"
          style={{
            ...glowStyle,
            filter: 'blur(2px)',
            maskImage: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.3))'
          }}
          aria-hidden="true"
        >
          {children}
        </span>
      )}
    </span>
  );
};