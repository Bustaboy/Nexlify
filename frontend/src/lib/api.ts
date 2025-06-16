// frontend/src/lib/api.ts

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@stores/authStore';
import toast from 'react-hot-toast';

// Types - keeping it clean like corpo code, but with street smarts
interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}

interface RetryConfig {
  retries: number;
  retryDelay: number;
  retryCondition?: (error: AxiosError) => boolean;
}

// Create the main axios instance - our link to the neural backend
const createApiClient = (): AxiosInstance => {
  // Get the API endpoint - could be local or could be in the cloud
  const apiEndpoint = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  
  const client = axios.create({
    baseURL: `${apiEndpoint}/api/v1`,
    timeout: 30000, // 30 seconds - enough time for complex neural calculations
    headers: {
      'Content-Type': 'application/json',
      'X-Client-Version': '3.0.0',
      'X-Client-Platform': 'electron'
    }
  });

  // Request interceptor - jack in the auth token
  client.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
      // Get fresh token from store
      const token = useAuthStore.getState().token;
      
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      
      // Add request timestamp for latency tracking
      config.metadata = { startTime: Date.now() };
      
      return config;
    },
    (error) => {
      console.error('Request interceptor error:', error);
      return Promise.reject(error);
    }
  );

  // Response interceptor - handle the data stream or the flatline
  client.interceptors.response.use(
    (response) => {
      // Track request latency if we're monitoring performance
      if (response.config.metadata?.startTime) {
        const latency = Date.now() - response.config.metadata.startTime;
        if (latency > 5000) {
          console.warn(`Slow API response: ${response.config.url} took ${latency}ms`);
        }
      }
      
      return response;
    },
    async (error: AxiosError<ApiError>) => {
      // Network's down? API's flatlined? Handle it like a pro
      if (!error.response) {
        // Network error - the worst kind in Night City
        toast.error('Neural link severed - check your connection', {
          icon: '🔌',
          duration: 5000
        });
        return Promise.reject(error);
      }

      const { status, data } = error.response;
      const originalRequest = error.config;

      // 401 - Auth expired, time to jack back in
      if (status === 401 && originalRequest && !originalRequest._retry) {
        originalRequest._retry = true;
        
        const authStore = useAuthStore.getState();
        
        // Session expired - need a fresh neural handshake
        authStore.logout();
        
        toast.error('Session expired - jack in again', {
          icon: '🔒',
          duration: 4000
        });
        
        // Redirect to auth if in Electron
        if (window.nexlify) {
          window.nexlify.send('navigate', '/auth');
        }
        
        return Promise.reject(error);
      }

      // 429 - Rate limited, slow your chrome
      if (status === 429) {
        const retryAfter = error.response.headers['retry-after'];
        toast.error(`Rate limited - cooldown ${retryAfter || '60'}s`, {
          icon: '⏱️',
          duration: 5000
        });
      }

      // 500+ - Server's having a bad trip
      if (status >= 500) {
        toast.error('Neural core malfunction - try again', {
          icon: '🧠',
          duration: 4000
        });
      }

      // Generic handler for other errors
      const errorMessage = data?.message || `Error ${status}: ${error.message}`;
      console.error(`API Error [${status}]:`, errorMessage, data?.details);

      return Promise.reject(error);
    }
  );

  return client;
};

// Create the singleton instance
export const apiClient = createApiClient();

// Specialized clients for different data streams
export const marketClient = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/market`,
  timeout: 5000, // Market data needs to be snappy
});

export const analyticsClient = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/analytics`,
  timeout: 60000, // Analytics can take time to crunch
});

// Helper functions for common patterns
export const apiHelpers = {
  // Retry logic for flaky connections
  async withRetry<T>(
    fn: () => Promise<T>, 
    config: RetryConfig = { retries: 3, retryDelay: 1000 }
  ): Promise<T> {
    let lastError: any;
    
    for (let i = 0; i <= config.retries; i++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        if (i < config.retries) {
          // Check if we should retry
          if (config.retryCondition && !config.retryCondition(error as AxiosError)) {
            throw error;
          }
          
          // Exponential backoff with jitter
          const delay = config.retryDelay * Math.pow(2, i) + Math.random() * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  },

  // Batch requests to avoid rate limits
  async batchRequests<T>(
    requests: (() => Promise<T>)[],
    batchSize: number = 5,
    delay: number = 100
  ): Promise<T[]> {
    const results: T[] = [];
    
    for (let i = 0; i < requests.length; i += batchSize) {
      const batch = requests.slice(i, i + batchSize);
      const batchResults = await Promise.all(batch.map(req => req()));
      results.push(...batchResults);
      
      // Don't delay after the last batch
      if (i + batchSize < requests.length) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    return results;
  },

  // Cancel token management
  createCancelToken() {
    return axios.CancelToken.source();
  },

  // Check if error is cancellation
  isCancel: axios.isCancel
};

// WebSocket configuration for real-time data streams
export const createWebSocketConnection = (token: string) => {
  const wsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000')
    .replace('http', 'ws')
    .replace('https', 'wss');
    
  return {
    url: wsUrl,
    options: {
      auth: { token },
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 10,
      timeout: 20000
    }
  };
};

// Export types for use in components
export type { ApiError, RetryConfig };
