// Location: nexlify-dashboard/src/systems/adaptive/__tests__/integration.test.tsx
// Mission: 80-I.1 Integration Tests
// Dependencies: All components
// Context: Ensure adaptive system works correctly

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AdaptiveVisualProvider } from '../core/AdaptiveVisualProvider';
import { AdaptiveNeonText } from '../components/AdaptiveNeonText';
import { hardwareProfiler } from '../core/HardwareProfiler';

// Mock hardware profiler
jest.mock('../core/HardwareProfiler');

describe('Adaptive Visual System', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });
  
  test('renders with minimal features on low-end hardware', async () => {
    // Mock potato hardware
    (hardwareProfiler.getCapabilities as jest.Mock).mockReturnValue({
      gpu: { renderer: 'Intel UHD Graphics 630' },
      vram: 2048,
      computeUnits: 96,
      benchmarks: {
        trianglesPerSecond: 1000000,
        pixelFillRate: 100000000
      }
    });
    
    (hardwareProfiler.getGPUScore as jest.Mock).mockReturnValue(15);
    
    const { container } = render(
      <AdaptiveVisualProvider>
        <AdaptiveNeonText importance="high">Test</AdaptiveNeonText>
      </AdaptiveVisualProvider>
    );
    
    await waitFor(() => {
      // Should have minimal or no glow
      const text = screen.getByText('Test');
      const styles = window.getComputedStyle(text);
      expect(styles.textShadow).toBe('none');
    });
  });
  
  test('enables full features on high-end hardware', async () => {
    // Mock chrome hardware
    (hardwareProfiler.getCapabilities as jest.Mock).mockReturnValue({
      gpu: { renderer: 'NVIDIA GeForce RTX 4070' },
      vram: 12288,
      computeUnits: 5888,
      benchmarks: {
        trianglesPerSecond: 100000000,
        pixelFillRate: 10000000000
      }
    });
    
    (hardwareProfiler.getGPUScore as jest.Mock).mockReturnValue(75);
    
    const { container } = render(
      <AdaptiveVisualProvider>
        <AdaptiveNeonText importance="high">Test</AdaptiveNeonText>
      </AdaptiveVisualProvider>
    );
    
    await waitFor(() => {
      // Should have glow effect
      const text = screen.getByText('Test');
      const styles = window.getComputedStyle(text);
      expect(styles.textShadow).toContain('rgb');
    });
  });
  
  test('disables visuals in trading mode', async () => {
    const { rerender } = render(
      <AdaptiveVisualProvider config={{ defaultMode: 'balanced' }}>
        <AdaptiveNeonText importance="critical">Trade</AdaptiveNeonText>
      </AdaptiveVisualProvider>
    );
    
    // Switch to trading mode
    // Would need to expose setPerformanceMode through a test helper
  });
  
  test('adapts to performance issues', async () => {
    // Test that features reduce when FPS drops
    // Would need to simulate performance metrics
  });
});