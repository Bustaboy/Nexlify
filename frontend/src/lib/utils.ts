// frontend/src/lib/utils.ts

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { format, formatDistance, formatRelative, isToday, isYesterday } from 'date-fns';

// Class name merger - because even in Night City, style conflicts need resolution
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Currency formatter - counting eddies like a true Night City trader
export function formatCurrency(
  value: number,
  decimals: number = 2,
  currency: string = 'USD'
): string {
  if (value === null || value === undefined || isNaN(value)) {
    return '$0.00';
  }

  // For large numbers, use shorthand - nobody's got time for all those zeros
  if (Math.abs(value) >= 1e9) {
    return `$${(value / 1e9).toFixed(2)}B`;
  } else if (Math.abs(value) >= 1e6) {
    return `$${(value / 1e6).toFixed(2)}M`;
  } else if (Math.abs(value) >= 1e3) {
    return `$${(value / 1e3).toFixed(2)}K`;
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

// Percentage formatter - because gains and losses hit different
export function formatPercent(
  value: number,
  decimals: number = 2,
  showSign: boolean = true
): string {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00%';
  }

  const formatted = value.toFixed(decimals);
  const sign = showSign && value > 0 ? '+' : '';
  
  return `${sign}${formatted}%`;
}

// Number formatter with separators - making big numbers readable
export function formatNumber(
  value: number,
  decimals: number = 0
): string {
  if (value === null || value === undefined || isNaN(value)) {
    return '0';
  }

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

// Date formatting - time moves different in the trading matrix
export function formatDate(date: Date | string | number): string {
  const d = new Date(date);
  
  if (isToday(d)) {
    return `Today ${format(d, 'HH:mm')}`;
  } else if (isYesterday(d)) {
    return `Yesterday ${format(d, 'HH:mm')}`;
  } else if (Date.now() - d.getTime() < 7 * 24 * 60 * 60 * 1000) {
    return formatRelative(d, new Date());
  }
  
  return format(d, 'MMM dd, yyyy HH:mm');
}

// Relative time - "moments ago" hits different when you're watching charts
export function formatRelativeTime(date: Date | string | number): string {
  return formatDistance(new Date(date), new Date(), { addSuffix: true });
}

// Calculate P&L - the moment of truth for every trade
export function calculatePnL(
  entryPrice: number,
  currentPrice: number,
  amount: number,
  side: 'long' | 'short'
): { pnl: number; pnlPercent: number } {
  let pnl: number;
  let pnlPercent: number;

  if (side === 'long') {
    pnl = (currentPrice - entryPrice) * amount;
    pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;
  } else {
    pnl = (entryPrice - currentPrice) * amount;
    pnlPercent = ((entryPrice - currentPrice) / entryPrice) * 100;
  }

  return { pnl, pnlPercent };
}

// Calculate position size based on risk - because surviving Night City means managing risk
export function calculatePositionSize(
  accountBalance: number,
  riskPercent: number,
  stopLossPercent: number
): number {
  if (stopLossPercent <= 0) return 0;
  
  const riskAmount = accountBalance * (riskPercent / 100);
  const positionSize = riskAmount / (stopLossPercent / 100);
  
  return Math.floor(positionSize * 100) / 100; // Round down to 2 decimals
}

// Risk/Reward ratio - the golden metric
export function calculateRiskReward(
  entryPrice: number,
  stopLoss: number,
  takeProfit: number
): number {
  const risk = Math.abs(entryPrice - stopLoss);
  const reward = Math.abs(takeProfit - entryPrice);
  
  if (risk === 0) return 0;
  
  return reward / risk;
}

// Debounce function - because spamming buttons won't make trades execute faster
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Throttle function - rate limiting for the frontend
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  
  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Generate random ID - every entity in Night City needs an identity
export function generateId(prefix: string = 'id'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Deep clone object - because mutations in Night City can be deadly
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as any;
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as any;
  if (obj instanceof Object) {
    const clonedObj = {} as T;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key]);
      }
    }
    return clonedObj;
  }
  return obj;
}

// Sleep function - sometimes you need to slow down in the fast lane
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Retry with exponential backoff - persistence pays in Night City
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (i < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, i) + Math.random() * 1000;
        await sleep(delay);
      }
    }
  }
  
  throw lastError;
}

// Validate trading symbol - gotta make sure you're trading real chrome
export function isValidSymbol(symbol: string): boolean {
  const symbolRegex = /^[A-Z]{2,10}\/[A-Z]{2,10}$/;
  return symbolRegex.test(symbol);
}

// Format large numbers with suffix - because readability matters
export function formatLargeNumber(num: number): string {
  if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toFixed(2);
}

// Calculate time until next candle - for those watching every tick
export function timeUntilNextCandle(interval: string): number {
  const now = new Date();
  const intervals: Record<string, number> = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
  };
  
  const intervalMs = intervals[interval] || intervals['1m'];
  const elapsed = now.getTime() % intervalMs;
  
  return intervalMs - elapsed;
}

// Color for P&L - green for gains, red for pain
export function getPnLColor(value: number): string {
  if (value > 0) return 'text-neon-green';
  if (value < 0) return 'text-neon-red';
  return 'text-gray-400';
}

// Truncate address - for those long blockchain addresses
export function truncateAddress(address: string, length: number = 6): string {
  if (!address) return '';
  if (address.length <= length * 2) return address;
  
  return `${address.slice(0, length)}...${address.slice(-length)}`;
}

// Parse error message - because API errors can be cryptic
export function parseErrorMessage(error: any): string {
  if (typeof error === 'string') return error;
  
  if (error?.response?.data?.message) {
    return error.response.data.message;
  }
  
  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }
  
  if (error?.message) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
}

// Local storage helpers with error handling
export const storage = {
  get<T>(key: string, defaultValue?: T): T | null {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue || null;
    } catch (error) {
      console.error(`Error reading from storage [${key}]:`, error);
      return defaultValue || null;
    }
  },
  
  set(key: string, value: any): boolean {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error(`Error writing to storage [${key}]:`, error);
      return false;
    }
  },
  
  remove(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing from storage [${key}]:`, error);
    }
  },
  
  clear(): void {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing storage:', error);
    }
  }
};

// Check if running in Electron
export function isElectron(): boolean {
  return !!(window as any).nexlify;
}

// Get platform info
export function getPlatform(): 'windows' | 'mac' | 'linux' | 'web' {
  if (!isElectron()) return 'web';
  
  const platform = (window as any).nexlify?.system?.platform;
  
  switch (platform) {
    case 'win32': return 'windows';
    case 'darwin': return 'mac';
    case 'linux': return 'linux';
    default: return 'web';
  }
}
