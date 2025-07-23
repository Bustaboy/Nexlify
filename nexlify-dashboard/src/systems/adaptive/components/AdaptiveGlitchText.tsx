// Location: nexlify-dashboard/src/systems/adaptive/components/AdaptiveGlitchText.tsx
// Mission: 80-I.1 Adaptive Glitch Text Effect - FIXED
// Dependencies: useAdaptiveVisuals hook
// Context: Text that glitches based on hardware capabilities

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';

interface AdaptiveGlitchTextProps {
  text: string;
  className?: string;
  intensity?: number; // 0-1, overrides feature intensity
  static?: boolean; // Disable animation
}

const GLITCH_CHARS = '!<>-_\\/[]{}—=+*^?#________アイウエオカキクケコサシスセソタチツテト';

export const AdaptiveGlitchText: React.FC<AdaptiveGlitchTextProps> = ({
  text,
  className = '',
  intensity,
  static: isStatic = false
}) => {
  const { features } = useAdaptiveVisuals();
  const [displayText, setDisplayText] = useState(text);
  const [isGlitching, setIsGlitching] = useState(false);
  
  const effectiveIntensity = useMemo(() => {
    if (!features?.glitchEffects?.enabled) return 0;
    return intensity ?? features?.glitchEffects?.intensity ?? 0;
  }, [features, intensity]);
  
  const glitchProbability = useMemo(() => {
    if (!features?.glitchEffects?.enabled || isStatic) return 0;
    return features?.glitchEffects?.probability ?? 0;
  }, [features, isStatic]);
  
  const glitchTypes = useMemo(() => {
    if (!features?.glitchEffects?.enabled) return [];
    return features?.glitchEffects?.types ?? [];
  }, [features]);
  
  const applyGlitch = useCallback((originalText: string): string => {
    if (!effectiveIntensity) return originalText;
    
    const chars = originalText.split('');
    const glitchCount = Math.ceil(chars.length * effectiveIntensity * 0.3);
    
    for (let i = 0; i < glitchCount; i++) {
      const index = Math.floor(Math.random() * chars.length);
      
      if (glitchTypes.includes('displacement')) {
        // Character replacement
        chars[index] = GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)];
      }
      
      if (glitchTypes.includes('color') && Math.random() > 0.5) {
        // Would apply color in rendering, not in text
      }
    }
    
    return chars.join('');
  }, [effectiveIntensity, glitchTypes]);
  
  useEffect(() => {
    if (!glitchProbability || isStatic) {
      setDisplayText(text);
      return;
    }
    
    const interval = setInterval(() => {
      if (Math.random() < glitchProbability) {
        setIsGlitching(true);
        setDisplayText(applyGlitch(text));
        
        // Glitch duration
        const duration = features?.glitchEffects?.duration ?? { min: 50, max: 150 };
        const glitchTime = duration.min + Math.random() * (duration.max - duration.min);
        
        setTimeout(() => {
          setDisplayText(text);
          setIsGlitching(false);
        }, glitchTime);
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, [text, glitchProbability, applyGlitch, isStatic, features]);
  
  const glitchStyle = useMemo(() => {
    if (!isGlitching || !glitchTypes.length) return {};
    
    const styles: React.CSSProperties = {};
    
    if (glitchTypes.includes('displacement')) {
      styles.transform = `translate(${Math.random() * 4 - 2}px, ${Math.random() * 2 - 1}px)`;
    }
    
    if (glitchTypes.includes('noise')) {
      styles.filter = `contrast(${1 + Math.random() * 0.5})`;
    }
    
    return styles;
  }, [isGlitching, glitchTypes]);
  
  // Render with color channel split effect
  if (isGlitching && glitchTypes.includes('color')) {
    return (
      <span className={`relative inline-block ${className}`} style={glitchStyle}>
        <span className="relative z-10">{displayText}</span>
        <span 
          className="absolute top-0 left-0 text-cyan-400 opacity-70"
          style={{ transform: 'translate(1px, 0)' }}
          aria-hidden="true"
        >
          {displayText}
        </span>
        <span 
          className="absolute top-0 left-0 text-red-400 opacity-70"
          style={{ transform: 'translate(-1px, 0)' }}
          aria-hidden="true"
        >
          {displayText}
        </span>
      </span>
    );
  }
  
  return (
    <span className={className} style={glitchStyle}>
      {displayText}
    </span>
  );
};