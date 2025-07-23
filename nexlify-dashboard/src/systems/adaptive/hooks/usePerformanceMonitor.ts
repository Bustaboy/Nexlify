// Location: nexlify-dashboard/src/systems/adaptive/hooks/usePerformanceMonitor.ts
// Mission: 80-I.1 Performance Monitoring Hook
// Dependencies: PerformanceMonitor
// Context: React hook for performance monitoring

import { useEffect, useState } from 'react';
import { performanceMonitor, PerformanceMetrics } from '../core/PerformanceMonitor';

export const usePerformanceMonitor = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [issues, setIssues] = useState<{ severity: string; issues: string[] }>({
    severity: 'none',
    issues: []
  });
  
  useEffect(() => {
    const unsubscribe = performanceMonitor.subscribe((newMetrics) => {
      setMetrics(newMetrics);
      
      const performanceIssues = performanceMonitor.checkPerformanceIssues();
      setIssues(performanceIssues);
    });
    
    return unsubscribe;
  }, []);
  
  return {
    metrics,
    issues,
    isHealthy: issues.severity === 'none' || issues.severity === 'low'
  };
};