// Location: nexlify-dashboard/src/systems/adaptive/hooks/useAdaptiveVisuals.ts
// Mission: 80-I.1 Convenience Hooks
// Dependencies: AdaptiveVisualProvider
// Context: Easy access to specific visual features

export { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';

import { useAdaptiveVisuals } from '../core/AdaptiveVisualProvider';
import { useMemo } from 'react';

// Check if a specific feature is enabled
export const useVisualFeature = (featureName: string) => {
  const { features } = useAdaptiveVisuals();
  
  return useMemo(() => {
    if (!features) return null;
    return (features as any)[featureName];
  }, [features, featureName]);
};

// Get hardware score
export const useHardwareScore = () => {
  const { gpuScore } = useAdaptiveVisuals();
  return gpuScore;
};

// Check if in high performance mode
export const useIsHighPerformance = () => {
  const { performanceMode, gpuScore } = useAdaptiveVisuals();
  
  return useMemo(() => {
    return performanceMode === 'visual' && gpuScore > 60;
  }, [performanceMode, gpuScore]);
};

// Get current performance metrics
export const usePerformanceMetrics = () => {
  const { metrics } = useAdaptiveVisuals();
  return metrics;
};

// Check if trading is active
export const useTradingMode = () => {
  const { performanceMode } = useAdaptiveVisuals();
  return performanceMode === 'trading';
};