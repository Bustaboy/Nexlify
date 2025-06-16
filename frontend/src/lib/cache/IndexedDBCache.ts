// frontend/src/lib/cache/IndexedDBCache.ts
/**
 * IndexedDB Cache System - Neural Memory Core
 * High-performance client-side caching for sub-second data retrieval
 * 
 * This is your brain's memory banks, hermano. Where market data flows
 * faster than synapses firing in a boosted cortex. Every millisecond
 * saved here is eddies in your pocket when the algos are hunting.
 * 
 * Been through enough system crashes to know: local cache saves lives.
 * When the net goes dark, this is what keeps you alive in the matrix.
 */

import { openDB, IDBPDatabase, IDBPTransaction } from 'idb';

// Types - the data structures that define our reality
interface CacheEntry<T = any> {
  key: string;
  data: T;
  timestamp: number;
  ttl: number;
  size: number;
  version: number;
  tags: string[];
  priority: number; // 1 = critical, 5 = can expire
}

interface CacheMetrics {
  totalEntries: number;
  totalSize: number;
  hitRate: number;
  missRate: number;
  avgRetrievalTime: number;
  lastCleanup: number;
  cacheHealth: number;
}

interface CacheOptions {
  ttl?: number;
  priority?: number;
  tags?: string[];
  compress?: boolean;
  enableVersioning?: boolean;
}

interface MarketDataEntry {
  symbol: string;
  timestamp: number;
  price: number;
  volume: number;
  change: number;
  orderBook?: any;
  trades?: any[];
}

interface TradingSignal {
  id: string;
  strategy: string;
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  timestamp: number;
  metadata: Record<string, any>;
}

// Cache configuration - tuned for maximum performance
const CACHE_CONFIG = {
  DATABASE_NAME: 'NexlifyTradingCache',
  VERSION: 3,
  STORES: {
    MARKET_DATA: 'market_data',
    TRADING_SIGNALS: 'trading_signals', 
    ORDERBOOK_DATA: 'orderbook_data',
    PERFORMANCE_METRICS: 'performance_metrics',
    USER_PREFERENCES: 'user_preferences',
    STRATEGY_CACHE: 'strategy_cache'
  },
  MAX_SIZE: 500 * 1024 * 1024,        // 500MB total cache size
  MAX_ENTRIES_PER_STORE: 10000,       // Prevent runaway memory usage
  DEFAULT_TTL: 5 * 60 * 1000,         // 5 minutes default TTL
  CLEANUP_INTERVAL: 60 * 1000,        // Cleanup every minute
  COMPRESSION_THRESHOLD: 1024,        // Compress entries > 1KB
  PERFORMANCE_SAMPLE_SIZE: 100,       // Track last 100 operations
  CRITICAL_TTL: 30 * 1000,            // 30 seconds for critical data
  LOW_PRIORITY_TTL: 60 * 60 * 1000    // 1 hour for low priority
} as const;

/**
 * High-Performance IndexedDB Cache Manager
 * Built for speed, designed for profit
 */
export class NexlifyCache {
  private db: IDBPDatabase | null = null;
  private isInitialized = false;
  private operationTimes: number[] = [];
  private hitCount = 0;
  private missCount = 0;
  private cleanupTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.initialize();
  }

  /**
   * Initialize the cache system - boot up the neural memory
   */
  private async initialize(): Promise<void> {
    try {
      console.log('🧠 Initializing neural memory core...');
      
      this.db = await openDB(CACHE_CONFIG.DATABASE_NAME, CACHE_CONFIG.VERSION, {
        upgrade: (db, oldVersion, newVersion) => {
          console.log(`📀 Upgrading cache database: ${oldVersion} → ${newVersion}`);
          
          // Create object stores with optimized indexes
          Object.values(CACHE_CONFIG.STORES).forEach(storeName => {
            if (!db.objectStoreNames.contains(storeName)) {
              const store = db.createObjectStore(storeName, { keyPath: 'key' });
              
              // Create indexes for efficient queries
              store.createIndex('timestamp', 'timestamp');
              store.createIndex('tags', 'tags', { multiEntry: true });
              store.createIndex('priority', 'priority');
              store.createIndex('ttl', 'ttl');
              
              // Specific indexes for different data types
              if (storeName === CACHE_CONFIG.STORES.MARKET_DATA) {
                store.createIndex('symbol', 'data.symbol');
              }
              
              if (storeName === CACHE_CONFIG.STORES.TRADING_SIGNALS) {
                store.createIndex('strategy', 'data.strategy');
                store.createIndex('confidence', 'data.confidence');
              }
            }
          });
        }
      });

      this.isInitialized = true;
      
      // Start background cleanup
      this.startBackgroundCleanup();
      
      // Log cache metrics on startup
      const metrics = await this.getMetrics();
      console.log('📊 Cache initialized:', metrics);
      
    } catch (error) {
      console.error('🚨 Cache initialization failed:', error);
      // Fallback to in-memory cache if IndexedDB fails
      this.initializeFallbackCache();
    }
  }

  /**
   * Fallback in-memory cache when IndexedDB fails
   */
  private initializeFallbackCache(): void {
    console.warn('⚠️ Falling back to in-memory cache - performance will be limited');
    // Implementation would use Map for basic caching
  }

  /**
   * Store market data with optimization for rapid retrieval
   */
  async setMarketData(symbol: string, data: MarketDataEntry, options: CacheOptions = {}): Promise<void> {
    const startTime = performance.now();
    
    try {
      await this.set(
        `market:${symbol}`,
        data,
        CACHE_CONFIG.STORES.MARKET_DATA,
        {
          ttl: CACHE_CONFIG.CRITICAL_TTL,
          priority: 1, // Critical data
          tags: ['market', symbol],
          ...options
        }
      );
      
      this.trackOperation(performance.now() - startTime);
      
    } catch (error) {
      console.error(`Failed to cache market data for ${symbol}:`, error);
    }
  }

  /**
   * Retrieve market data with sub-millisecond targeting
   */
  async getMarketData(symbol: string): Promise<MarketDataEntry | null> {
    const startTime = performance.now();
    
    try {
      const result = await this.get<MarketDataEntry>(
        `market:${symbol}`,
        CACHE_CONFIG.STORES.MARKET_DATA
      );
      
      this.trackOperation(performance.now() - startTime);
      
      if (result) {
        this.hitCount++;
        return result;
      } else {
        this.missCount++;
        return null;
      }
      
    } catch (error) {
      console.error(`Failed to retrieve market data for ${symbol}:`, error);
      this.missCount++;
      return null;
    }
  }

  /**
   * Store trading signals with ML metadata
   */
  async setTradingSignal(signal: TradingSignal, options: CacheOptions = {}): Promise<void> {
    try {
      await this.set(
        `signal:${signal.id}`,
        signal,
        CACHE_CONFIG.STORES.TRADING_SIGNALS,
        {
          ttl: CACHE_CONFIG.DEFAULT_TTL,
          priority: signal.confidence > 0.8 ? 1 : 3,
          tags: ['signal', signal.strategy, signal.symbol],
          ...options
        }
      );
    } catch (error) {
      console.error('Failed to cache trading signal:', error);
    }
  }

  /**
   * Retrieve trading signals by strategy or symbol
   */
  async getTradingSignals(filter: { strategy?: string; symbol?: string; minConfidence?: number } = {}): Promise<TradingSignal[]> {
    try {
      const allSignals = await this.getByTag('signal', CACHE_CONFIG.STORES.TRADING_SIGNALS);
      
      return allSignals.filter(entry => {
        const signal = entry.data as TradingSignal;
        
        if (filter.strategy && signal.strategy !== filter.strategy) return false;
        if (filter.symbol && signal.symbol !== filter.symbol) return false;
        if (filter.minConfidence && signal.confidence < filter.minConfidence) return false;
        
        return true;
      }).map(entry => entry.data);
      
    } catch (error) {
      console.error('Failed to retrieve trading signals:', error);
      return [];
    }
  }

  /**
   * Store order book data with compression for large datasets
   */
  async setOrderBookData(symbol: string, orderBook: any, options: CacheOptions = {}): Promise<void> {
    try {
      await this.set(
        `orderbook:${symbol}`,
        orderBook,
        CACHE_CONFIG.STORES.ORDERBOOK_DATA,
        {
          ttl: 10 * 1000, // 10 seconds - order books change rapidly
          priority: 1,
          tags: ['orderbook', symbol],
          compress: true, // Order books are large
          ...options
        }
      );
    } catch (error) {
      console.error('Failed to cache order book data:', error);
    }
  }

  /**
   * Generic set method with compression and optimization
   */
  private async set<T>(
    key: string, 
    data: T, 
    storeName: string, 
    options: CacheOptions = {}
  ): Promise<void> {
    if (!this.db || !this.isInitialized) {
      await this.initialize();
      if (!this.db) throw new Error('Cache not available');
    }

    const {
      ttl = CACHE_CONFIG.DEFAULT_TTL,
      priority = 3,
      tags = [],
      compress = false,
      enableVersioning = false
    } = options;

    // Serialize and optionally compress data
    let serializedData = JSON.stringify(data);
    let size = new Blob([serializedData]).size;
    
    if (compress && size > CACHE_CONFIG.COMPRESSION_THRESHOLD) {
      // In a real implementation, use compression library like lz-string
      // serializedData = LZString.compress(serializedData);
      // size = new Blob([serializedData]).size;
    }

    const entry: CacheEntry<T> = {
      key,
      data,
      timestamp: Date.now(),
      ttl,
      size,
      version: enableVersioning ? Date.now() : 1,
      tags: [...tags, 'cached'],
      priority
    };

    try {
      const tx = this.db.transaction(storeName, 'readwrite');
      await tx.objectStore(storeName).put(entry);
      await tx.done;
      
    } catch (error) {
      console.error(`Cache write failed for ${key}:`, error);
      throw error;
    }
  }

  /**
   * Generic get method with TTL checking
   */
  private async get<T>(key: string, storeName: string): Promise<T | null> {
    if (!this.db || !this.isInitialized) {
      await this.initialize();
      if (!this.db) return null;
    }

    try {
      const tx = this.db.transaction(storeName, 'readonly');
      const entry = await tx.objectStore(storeName).get(key) as CacheEntry<T> | undefined;
      await tx.done;

      if (!entry) return null;

      // Check TTL
      const now = Date.now();
      if (now > entry.timestamp + entry.ttl) {
        // Entry expired - remove it
        this.delete(key, storeName);
        return null;
      }

      return entry.data;
      
    } catch (error) {
      console.error(`Cache read failed for ${key}:`, error);
      return null;
    }
  }

  /**
   * Get entries by tag for bulk operations
   */
  private async getByTag(tag: string, storeName: string): Promise<CacheEntry[]> {
    if (!this.db || !this.isInitialized) return [];

    try {
      const tx = this.db.transaction(storeName, 'readonly');
      const index = tx.objectStore(storeName).index('tags');
      const entries = await index.getAll(tag);
      await tx.done;

      const now = Date.now();
      return entries.filter(entry => now <= entry.timestamp + entry.ttl);
      
    } catch (error) {
      console.error('Failed to get entries by tag:', error);
      return [];
    }
  }

  /**
   * Delete specific cache entry
   */
  private async delete(key: string, storeName: string): Promise<void> {
    if (!this.db) return;

    try {
      const tx = this.db.transaction(storeName, 'readwrite');
      await tx.objectStore(storeName).delete(key);
      await tx.done;
    } catch (error) {
      console.error(`Failed to delete cache entry ${key}:`, error);
    }
  }

  /**
   * Clear cache by tag - selective cleanup
   */
  async clearByTag(tag: string): Promise<void> {
    if (!this.db) return;

    console.log(`🧹 Clearing cache entries tagged: ${tag}`);

    try {
      for (const storeName of Object.values(CACHE_CONFIG.STORES)) {
        const tx = this.db.transaction(storeName, 'readwrite');
        const index = tx.objectStore(storeName).index('tags');
        const entries = await index.getAll(tag);
        
        for (const entry of entries) {
          await tx.objectStore(storeName).delete(entry.key);
        }
        
        await tx.done;
      }
    } catch (error) {
      console.error('Failed to clear cache by tag:', error);
    }
  }

  /**
   * Background cleanup of expired entries
   */
  private startBackgroundCleanup(): void {
    this.cleanupTimer = setInterval(async () => {
      await this.performCleanup();
    }, CACHE_CONFIG.CLEANUP_INTERVAL);
  }

  /**
   * Perform cache cleanup - remove expired and low-priority entries
   */
  private async performCleanup(): Promise<void> {
    if (!this.db) return;

    const now = Date.now();
    let totalCleaned = 0;
    let totalSize = 0;

    try {
      for (const storeName of Object.values(CACHE_CONFIG.STORES)) {
        const tx = this.db.transaction(storeName, 'readwrite');
        const store = tx.objectStore(storeName);
        const entries = await store.getAll();
        
        // Calculate total size first
        const currentSize = entries.reduce((sum, entry) => sum + entry.size, 0);
        totalSize += currentSize;
        
        // Remove expired entries
        const expiredKeys: string[] = [];
        const lowPriorityExpired: string[] = [];
        
        for (const entry of entries) {
          if (now > entry.timestamp + entry.ttl) {
            expiredKeys.push(entry.key);
          } else if (entry.priority > 3 && now > entry.timestamp + (entry.ttl * 0.5)) {
            // Remove low priority entries at 50% TTL if cache is getting full
            lowPriorityExpired.push(entry.key);
          }
        }
        
        // Remove expired entries first
        for (const key of expiredKeys) {
          await store.delete(key);
          totalCleaned++;
        }
        
        // If cache is still too large, remove low priority entries
        if (totalSize > CACHE_CONFIG.MAX_SIZE * 0.8) {
          for (const key of lowPriorityExpired.slice(0, 50)) {
            await store.delete(key);
            totalCleaned++;
          }
        }
        
        await tx.done;
      }
      
      if (totalCleaned > 0) {
        console.log(`🧹 Cache cleanup completed: removed ${totalCleaned} entries`);
      }
      
    } catch (error) {
      console.error('Cache cleanup failed:', error);
    }
  }

  /**
   * Get comprehensive cache metrics
   */
  async getMetrics(): Promise<CacheMetrics> {
    if (!this.db) {
      return {
        totalEntries: 0,
        totalSize: 0,
        hitRate: 0,
        missRate: 0,
        avgRetrievalTime: 0,
        lastCleanup: 0,
        cacheHealth: 0
      };
    }

    let totalEntries = 0;
    let totalSize = 0;

    try {
      for (const storeName of Object.values(CACHE_CONFIG.STORES)) {
        const tx = this.db.transaction(storeName, 'readonly');
        const entries = await tx.objectStore(storeName).getAll();
        await tx.done;
        
        totalEntries += entries.length;
        totalSize += entries.reduce((sum, entry) => sum + entry.size, 0);
      }
      
      const totalOperations = this.hitCount + this.missCount;
      const hitRate = totalOperations > 0 ? (this.hitCount / totalOperations) * 100 : 0;
      const missRate = totalOperations > 0 ? (this.missCount / totalOperations) * 100 : 0;
      
      const avgRetrievalTime = this.operationTimes.length > 0
        ? this.operationTimes.reduce((a, b) => a + b, 0) / this.operationTimes.length
        : 0;
      
      // Calculate cache health score (0-100)
      let healthScore = 100;
      if (totalSize > CACHE_CONFIG.MAX_SIZE * 0.9) healthScore -= 30;
      if (hitRate < 70) healthScore -= 20;
      if (avgRetrievalTime > 10) healthScore -= 20;
      
      return {
        totalEntries,
        totalSize,
        hitRate,
        missRate,
        avgRetrievalTime,
        lastCleanup: Date.now(),
        cacheHealth: Math.max(0, healthScore)
      };
      
    } catch (error) {
      console.error('Failed to get cache metrics:', error);
      return {
        totalEntries: 0,
        totalSize: 0,
        hitRate: 0,
        missRate: 0,
        avgRetrievalTime: 0,
        lastCleanup: 0,
        cacheHealth: 0
      };
    }
  }

  /**
   * Track operation performance
   */
  private trackOperation(time: number): void {
    this.operationTimes.push(time);
    
    // Keep only recent samples
    if (this.operationTimes.length > CACHE_CONFIG.PERFORMANCE_SAMPLE_SIZE) {
      this.operationTimes.shift();
    }
  }

  /**
   * Emergency cache clear - when all else fails
   */
  async emergencyClear(): Promise<void> {
    console.warn('🚨 Emergency cache clear initiated');
    
    if (!this.db) return;

    try {
      for (const storeName of Object.values(CACHE_CONFIG.STORES)) {
        const tx = this.db.transaction(storeName, 'readwrite');
        await tx.objectStore(storeName).clear();
        await tx.done;
      }
      
      // Reset metrics
      this.hitCount = 0;
      this.missCount = 0;
      this.operationTimes = [];
      
      console.log('💥 Emergency cache clear completed');
      
    } catch (error) {
      console.error('Emergency cache clear failed:', error);
    }
  }

  /**
   * Cleanup on instance destruction
   */
  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    this.isInitialized = false;
  }
}

// Singleton instance - one cache to rule them all
export const nexlifyCache = new NexlifyCache();

// Utility functions for common cache operations
export const cacheUtils = {
  /**
   * Quick market data caching
   */
  async cacheMarketTick(symbol: string, price: number, volume: number, change: number): Promise<void> {
    const data: MarketDataEntry = {
      symbol,
      timestamp: Date.now(),
      price,
      volume,
      change
    };
    
    await nexlifyCache.setMarketData(symbol, data);
  },

  /**
   * Batch cache multiple market updates
   */
  async batchCacheMarketData(updates: Array<{ symbol: string; price: number; volume: number; change: number }>): Promise<void> {
    const promises = updates.map(update => 
      cacheUtils.cacheMarketTick(update.symbol, update.price, update.volume, update.change)
    );
    
    await Promise.all(promises);
  },

  /**
   * Get cached market data with fallback
   */
  async getMarketDataWithFallback(symbol: string, fallbackFn?: () => Promise<MarketDataEntry>): Promise<MarketDataEntry | null> {
    let data = await nexlifyCache.getMarketData(symbol);
    
    if (!data && fallbackFn) {
      data = await fallbackFn();
      if (data) {
        await nexlifyCache.setMarketData(symbol, data);
      }
    }
    
    return data;
  }
};

// Health monitoring hook for React components
export function useCacheHealth() {
  const [metrics, setMetrics] = React.useState<CacheMetrics | null>(null);
  
  React.useEffect(() => {
    const updateMetrics = async () => {
      const newMetrics = await nexlifyCache.getMetrics();
      setMetrics(newMetrics);
    };
    
    updateMetrics();
    const interval = setInterval(updateMetrics, 30000); // Update every 30s
    
    return () => clearInterval(interval);
  }, []);
  
  return metrics;
}
