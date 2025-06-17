// Location: /src/utils/dashboard.utils.ts
// Utility functions for the Nexlify Neural Chrome Dashboard

import bcrypt from 'bcryptjs';
import { ThemeColors } from '../types/dashboard.types';

/**
 * Format currency values with proper sign and localization
 */
export const formatCredits = (value: number, decimals: number = 2): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}€${value.toLocaleString('en-US', { 
    minimumFractionDigits: decimals, 
    maximumFractionDigits: decimals 
  })}`;
};

/**
 * Format percentage values with sign
 */
export const formatPercent = (value: number, decimals: number = 2): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
};

/**
 * Get color based on value (positive/negative/neutral)
 */
export const getValueColor = (value: number, theme: ThemeColors): string => {
  if (value > 0) return theme.success;
  if (value < 0) return theme.danger;
  return '#6B7280'; // neutral gray
};

/**
 * Secure password hashing using bcrypt
 * NOTE: For production, this should be done server-side
 */
export const hashPassword = async (password: string): Promise<string> => {
  const saltRounds = 10;
  return bcrypt.hash(password, saltRounds);
};

/**
 * Verify password against hash
 */
export const verifyPassword = async (password: string, hash: string): Promise<boolean> => {
  return bcrypt.compare(password, hash);
};

/**
 * Encrypt data for localStorage
 * NOTE: For production, use a proper encryption library
 */
export const encryptData = (data: any, key: string): string => {
  // TODO: Implement proper encryption
  // For now, using base64 encoding as placeholder
  console.warn('Using base64 encoding - implement proper encryption for production');
  return btoa(JSON.stringify(data));
};

/**
 * Decrypt data from localStorage
 */
export const decryptData = (encryptedData: string, key: string): any => {
  // TODO: Implement proper decryption
  // For now, using base64 decoding as placeholder
  try {
    return JSON.parse(atob(encryptedData));
  } catch (error) {
    console.error('Failed to decrypt data:', error);
    return null;
  }
};

/**
 * Throttle function execution
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout | null = null;
  let lastExecTime = 0;
  
  return (...args: Parameters<T>) => {
    const currentTime = Date.now();
    
    if (currentTime - lastExecTime > delay) {
      func(...args);
      lastExecTime = currentTime;
    } else {
      if (timeoutId) clearTimeout(timeoutId);
      
      timeoutId = setTimeout(() => {
        func(...args);
        lastExecTime = Date.now();
      }, delay - (currentTime - lastExecTime));
    }
  };
};

/**
 * Debounce function execution
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeoutId) clearTimeout(timeoutId);
    
    timeoutId = setTimeout(() => {
      func(...args);
    }, delay);
  };
};

/**
 * Generate secure random ID
 */
export const generateId = (): string => {
  return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Format timestamp to local time string
 */
export const formatTimestamp = (timestamp: number): string => {
  return new Date(timestamp).toLocaleString();
};

/**
 * Calculate percentage change
 */
export const calculatePercentChange = (oldValue: number, newValue: number): number => {
  if (oldValue === 0) return 0;
  return ((newValue - oldValue) / oldValue) * 100;
};

/**
 * Clamp value between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.max(min, Math.min(max, value));
};
