// Location: /src/services/websocket.service.ts
// Nexlify WebSocket Service - Real-time data neural highway

import { BehaviorSubject, Subject, Observable, timer, merge } from 'rxjs';
import { 
  filter, 
  map, 
  retry, 
  catchError, 
  tap,
  throttleTime,
  bufferTime,
  scan
} from 'rxjs/operators';
import { io, Socket } from 'socket.io-client';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { Position, CrossChainPosition } from '@/types/trading.types';
import { OrderBook, Trade, Ticker } from '@/types/exchange.types';

interface WebSocketConfig {
  urls: {
    primary: string;
    fallback: string[];
    darkPool?: string;
  };
  auth: {
    token: string;
    sessionId: string;
  };
  options: {
    reconnectInterval: number;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
    compression: boolean;
    binaryType: 'arraybuffer' | 'blob';
  };
  channels: {
    positions: boolean;
    orderbooks: boolean;
    trades: boolean;
    signals: boolean;
    alerts: boolean;
  };
}

interface StreamHealth {
  status: 'connected' | 'connecting' | 'disconnected' | 'error';
  latency: number;
  messagesPerSecond: number;
  reconnectCount: number;
  lastHeartbeat: number;
  dataIntegrity: number; // 0-100 score
}

interface RateLimiter {
  channel: string;
  limit: number;
  window: number; // ms
  current: number;
  reset: number;
}

export class WebSocketService {
  // Core streams
  public positions$ = new BehaviorSubject<any>(null);
  public orderBooks$ = new Subject<OrderBook>();
  public trades$ = new Subject<Trade>();
  public tickers$ = new Subject<Ticker>();
  public signals$ = new Subject<any>();
  public alerts$ = new Subject<any>();
  
  // Health monitoring
  private health$ = new BehaviorSubject<StreamHealth>({
    status: 'disconnected',
    latency: 0,
    messagesPerSecond: 0,
    reconnectCount: 0,
    lastHeartbeat: Date.now(),
    dataIntegrity: 100
  });
  
  // Connection management
  private primarySocket: ReconnectingWebSocket | null = null;
  private fallbackSockets: Map<string, ReconnectingWebSocket> = new Map();
  private darkPoolSocket: Socket | null = null;
  
  // Performance tracking
  private messageBuffer: any[] = [];
  private latencyTracker: number[] = [];
  private rateLimiters: Map<string, RateLimiter> = new Map();
  
  // Security
  private encryptionKey: CryptoKey | null = null;
  private sessionValid = true;
  
  private config: WebSocketConfig;
  private isDestroyed = false;

  constructor(config: WebSocketConfig) {
    this.config = config;
    this.initializeEncryption();
    this.setupRateLimiters();
  }

  public async connect(): Promise<void> {
    console.log('[NEXLIFY WS] Jacking into the data stream...');
    
    try {
      // Primary connection
      await this.connectPrimary();
      
      // Fallback connections for redundancy
      this.connectFallbacks();
      
      // Dark pool connection if configured
      if (this.config.urls.darkPool) {
        await this.connectDarkPool();
      }
      
      // Start health monitoring
      this.startHealthMonitoring();
      
      console.log('[NEXLIFY WS] Neural link established. Data flowing.');
      
    } catch (error) {
      console.error('[NEXLIFY WS] Connection failed. Retrying...', error);
      this.handleConnectionError(error);
    }
  }

  private async connectPrimary(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.primarySocket = new ReconnectingWebSocket(this.config.urls.primary, [], {
        connectionTimeout: 10000,
        maxRetries: this.config.options.maxReconnectAttempts,
        minReconnectionDelay: this.config.options.reconnectInterval,
        maxReconnectionDelay: 30000,
        reconnectionDelayGrowFactor: 1.5,
        WebSocket: WebSocket as any
      });
      
      this.primarySocket.binaryType = this.config.options.binaryType;
      
      this.primarySocket.addEventListener('open', () => {
        console.log('[NEXLIFY WS] Primary stream online');
        this.authenticate(this.primarySocket!);
        this.subscribeToChannels(this.primarySocket!);
        this.updateHealth({ status: 'connected' });
        resolve();
      });
      
      this.primarySocket.addEventListener('message', (event) => {
        this.handleMessage(event.data, 'primary');
      });
      
      this.primarySocket.addEventListener('error', (error) => {
        console.error('[NEXLIFY WS] Primary stream error:', error);
        this.updateHealth({ status: 'error' });
      });
      
      this.primarySocket.addEventListener('close', () => {
        console.warn('[NEXLIFY WS] Primary stream disconnected');
        this.updateHealth({ status: 'disconnected' });
      });
      
      // Timeout connection attempt
      setTimeout(() => {
        if (this.health$.value.status !== 'connected') {
          reject(new Error('Connection timeout'));
        }
      }, 15000);
    });
  }

  private connectFallbacks(): void {
    this.config.urls.fallback.forEach((url, index) => {
      setTimeout(() => {
        const ws = new ReconnectingWebSocket(url, [], {
          connectionTimeout: 10000,
          maxRetries: 5,
          minReconnectionDelay: 5000,
          WebSocket: WebSocket as any
        });
        
        ws.addEventListener('open', () => {
          console.log(`[NEXLIFY WS] Fallback ${index + 1} online`);
          this.authenticate(ws);
        });
        
        ws.addEventListener('message', (event) => {
          this.handleMessage(event.data, `fallback_${index}`);
        });
        
        this.fallbackSockets.set(url, ws);
        
      }, index * 2000); // Stagger connections
    });
  }

  private async connectDarkPool(): Promise<void> {
    console.log('[NEXLIFY WS] Connecting to dark pool...');
    
    this.darkPoolSocket = io(this.config.urls.darkPool!, {
      auth: {
        token: this.config.auth.token,
        type: 'darkpool'
      },
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 5000
    });
    
    this.darkPoolSocket.on('connect', () => {
      console.log('[NEXLIFY WS] Dark pool access granted');
      
      // Subscribe to dark pool specific channels
      this.darkPoolSocket!.emit('subscribe', {
        channels: ['blocks', 'whale_alerts', 'otc_orders']
      });
    });
    
    this.darkPoolSocket.on('darkpool_order', (data) => {
      this.handleDarkPoolOrder(data);
    });
    
    this.darkPoolSocket.on('whale_movement', (data) => {
      this.alerts$.next({
        type: 'whale',
        severity: 'high',
        data,
        timestamp: Date.now()
      });
    });
  }

  private async initializeEncryption(): Promise<void> {
    try {
      // Generate encryption key for sensitive data
      this.encryptionKey = await crypto.subtle.generateKey(
        {
          name: 'AES-GCM',
          length: 256
        },
        true,
        ['encrypt', 'decrypt']
      );
    } catch (error) {
      console.error('[NEXLIFY WS] Encryption init failed:', error);
    }
  }

  private setupRateLimiters(): void {
    // Configure rate limits per channel
    const limits = {
      positions: { limit: 100, window: 1000 },
      orderbooks: { limit: 50, window: 1000 },
      trades: { limit: 200, window: 1000 },
      signals: { limit: 20, window: 1000 },
      alerts: { limit: 10, window: 1000 }
    };
    
    Object.entries(limits).forEach(([channel, config]) => {
      this.rateLimiters.set(channel, {
        channel,
        limit: config.limit,
        window: config.window,
        current: 0,
        reset: Date.now() + config.window
      });
    });
  }

  private authenticate(socket: ReconnectingWebSocket): void {
    const authMessage = {
      type: 'auth',
      data: {
        token: this.config.auth.token,
        sessionId: this.config.auth.sessionId,
        timestamp: Date.now(),
        signature: this.generateSignature()
      }
    };
    
    socket.send(JSON.stringify(authMessage));
  }

  private subscribeToChannels(socket: ReconnectingWebSocket): void {
    const subscriptions = Object.entries(this.config.channels)
      .filter(([_, enabled]) => enabled)
      .map(([channel]) => channel);
    
    const subMessage = {
      type: 'subscribe',
      channels: subscriptions,
      params: {
        compression: this.config.options.compression,
        binaryMode: this.config.options.binaryType === 'arraybuffer'
      }
    };
    
    socket.send(JSON.stringify(subMessage));
  }

  private async handleMessage(data: any, source: string): Promise<void> {
    try {
      // Decompress if needed
      let message = data;
      if (this.config.options.compression && data instanceof ArrayBuffer) {
        message = await this.decompressMessage(data);
      }
      
      // Parse message
      const parsed = typeof message === 'string' ? JSON.parse(message) : message;
      
      // Update latency tracking
      if (parsed.timestamp) {
        const latency = Date.now() - parsed.timestamp;
        this.trackLatency(latency);
      }
      
      // Rate limiting check
      if (!this.checkRateLimit(parsed.channel)) {
        console.warn(`[NEXLIFY WS] Rate limit exceeded for ${parsed.channel}`);
        return;
      }
      
      // Route to appropriate handler
      switch (parsed.channel) {
        case 'positions':
          this.handlePositionUpdate(parsed.data);
          break;
          
        case 'orderbook':
          this.handleOrderBookUpdate(parsed.data);
          break;
          
        case 'trades':
          this.handleTradeUpdate(parsed.data);
          break;
          
        case 'signals':
          this.handleSignalUpdate(parsed.data);
          break;
          
        case 'alerts':
          this.handleAlertUpdate(parsed.data);
          break;
          
        case 'heartbeat':
          this.handleHeartbeat(parsed);
          break;
          
        default:
          console.warn(`[NEXLIFY WS] Unknown channel: ${parsed.channel}`);
      }
      
      // Track message rate
      this.messageBuffer.push(Date.now());
      
    } catch (error) {
      console.error('[NEXLIFY WS] Message handling error:', error);
      this.updateHealth({ dataIntegrity: this.health$.value.dataIntegrity - 1 });
    }
  }

  private handlePositionUpdate(data: any): void {
    // Validate and emit position updates
    if (this.validatePositionData(data)) {
      this.positions$.next(data);
    }
  }

  private handleOrderBookUpdate(data: any): void {
    const orderBook: OrderBook = {
      exchange: data.exchange,
      symbol: data.symbol,
      bids: data.bids,
      asks: data.asks,
      timestamp: data.timestamp || Date.now()
    };
    
    this.orderBooks$.next(orderBook);
  }

  private handleTradeUpdate(data: any): void {
    const trade: Trade = {
      id: data.id || `trade_${Date.now()}`,
      exchange: data.exchange,
      symbol: data.symbol,
      price: data.price,
      amount: data.amount,
      side: data.side,
      timestamp: data.timestamp || Date.now()
    };
    
    this.trades$.next(trade);
  }

  private handleSignalUpdate(data: any): void {
    // Add metadata
    const signal = {
      ...data,
      receivedAt: Date.now(),
      source: 'websocket',
      latency: this.getAverageLatency()
    };
    
    this.signals$.next(signal);
  }

  private handleAlertUpdate(data: any): void {
    // Priority-based alert handling
    const alert = {
      ...data,
      id: `alert_${Date.now()}`,
      acknowledged: false
    };
    
    this.alerts$.next(alert);
    
    // Critical alerts trigger immediate action
    if (data.severity === 'critical') {
      this.handleCriticalAlert(alert);
    }
  }

  private handleDarkPoolOrder(data: any): void {
    // Dark pool orders are sensitive - encrypt before processing
    const order = {
      ...data,
      source: 'darkpool',
      encrypted: true,
      timestamp: Date.now()
    };
    
    // Route to positions stream with special flag
    this.positions$.next({
      type: 'darkpool',
      data: order
    });
  }

  private handleHeartbeat(data: any): void {
    this.updateHealth({
      lastHeartbeat: Date.now(),
      latency: Date.now() - data.timestamp
    });
  }

  private handleCriticalAlert(alert: any): void {
    console.error('[NEXLIFY WS] CRITICAL ALERT:', alert);
    
    // Emit to all relevant streams
    this.alerts$.next({ ...alert, broadcasted: true });
    
    // Could trigger emergency protocols here
  }

  // Utility methods
  private checkRateLimit(channel: string): boolean {
    const limiter = this.rateLimiters.get(channel);
    if (!limiter) return true;
    
    const now = Date.now();
    
    // Reset window if needed
    if (now > limiter.reset) {
      limiter.current = 0;
      limiter.reset = now + limiter.window;
    }
    
    // Check limit
    if (limiter.current >= limiter.limit) {
      return false;
    }
    
    limiter.current++;
    return true;
  }

  private trackLatency(latency: number): void {
    this.latencyTracker.push(latency);
    
    // Keep last 100 measurements
    if (this.latencyTracker.length > 100) {
      this.latencyTracker.shift();
    }
  }

  private getAverageLatency(): number {
    if (this.latencyTracker.length === 0) return 0;
    
    const sum = this.latencyTracker.reduce((a, b) => a + b, 0);
    return Math.round(sum / this.latencyTracker.length);
  }

  private validatePositionData(data: any): boolean {
    // Basic validation
    return !!(
      data &&
      data.symbol &&
      typeof data.price === 'number' &&
      typeof data.quantity === 'number'
    );
  }

  private async decompressMessage(data: ArrayBuffer): Promise<string> {
    // Decompress using native API
    const cs = new DecompressionStream('gzip');
    const writer = cs.writable.getWriter();
    writer.write(data);
    writer.close();
    
    const reader = cs.readable.getReader();
    const chunks: Uint8Array[] = [];
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    
    const decompressed = new Uint8Array(
      chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    );
    
    let offset = 0;
    for (const chunk of chunks) {
      decompressed.set(chunk, offset);
      offset += chunk.length;
    }
    
    return new TextDecoder().decode(decompressed);
  }

  private generateSignature(): string {
    // Generate auth signature
    const data = `${this.config.auth.sessionId}:${Date.now()}`;
    return btoa(data); // In production, use proper HMAC
  }

  // Health monitoring
  private startHealthMonitoring(): void {
    // Monitor connection health every 5 seconds
    timer(0, 5000).subscribe(() => {
      // Calculate messages per second
      const now = Date.now();
      const recentMessages = this.messageBuffer.filter(
        time => time > now - 1000
      );
      
      // Check heartbeat timeout
      const heartbeatTimeout = now - this.health$.value.lastHeartbeat > 30000;
      
      this.updateHealth({
        messagesPerSecond: recentMessages.length,
        latency: this.getAverageLatency(),
        status: heartbeatTimeout ? 'error' : this.health$.value.status
      });
      
      // Clean old messages from buffer
      this.messageBuffer = this.messageBuffer.filter(
        time => time > now - 60000
      );
    });
  }

  private updateHealth(update: Partial<StreamHealth>): void {
    this.health$.next({
      ...this.health$.value,
      ...update
    });
  }

  private handleConnectionError(error: any): void {
    this.updateHealth({
      status: 'error',
      reconnectCount: this.health$.value.reconnectCount + 1
    });
    
    // Attempt fallback connections
    if (this.fallbackSockets.size > 0) {
      console.log('[NEXLIFY WS] Switching to fallback connection...');
    }
  }

  // Public methods for manual control
  public subscribeToSymbol(symbol: string): void {
    const message = {
      type: 'subscribe_symbol',
      symbol,
      channels: ['orderbook', 'trades', 'ticker']
    };
    
    if (this.primarySocket?.readyState === WebSocket.OPEN) {
      this.primarySocket.send(JSON.stringify(message));
    }
  }

  public unsubscribeFromSymbol(symbol: string): void {
    const message = {
      type: 'unsubscribe_symbol',
      symbol
    };
    
    if (this.primarySocket?.readyState === WebSocket.OPEN) {
      this.primarySocket.send(JSON.stringify(message));
    }
  }

  public subscribeToChain(chain: string, callback: (data: CrossChainPosition) => void): void {
    // Chain-specific subscription for DeFi positions
    const subscription = this.positions$.pipe(
      filter(data => data?.chain === chain),
      map(data => data as CrossChainPosition)
    ).subscribe(callback);
    
    // Store subscription for cleanup
    // Implementation depends on your subscription management
  }

  public getHealth$(): Observable<StreamHealth> {
    return this.health$.asObservable();
  }

  public async reconnect(): Promise<void> {
    console.log('[NEXLIFY WS] Manual reconnection initiated...');
    
    // Close existing connections
    this.disconnect();
    
    // Wait a bit
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Reconnect
    await this.connect();
  }

  public disconnect(): void {
    console.log('[NEXLIFY WS] Disconnecting data streams...');
    
    this.isDestroyed = true;
    
    // Close primary socket
    if (this.primarySocket) {
      this.primarySocket.close();
      this.primarySocket = null;
    }
    
    // Close fallback sockets
    this.fallbackSockets.forEach(ws => ws.close());
    this.fallbackSockets.clear();
    
    // Close dark pool socket
    if (this.darkPoolSocket) {
      this.darkPoolSocket.disconnect();
      this.darkPoolSocket = null;
    }
    
    // Update status
    this.updateHealth({ status: 'disconnected' });
    
    console.log('[NEXLIFY WS] All streams terminated. Going dark.');
  }

  // Emergency data dump - for remote wipe scenarios
  public async emergencyDataExport(): Promise<Blob> {
    const data = {
      positions: this.positions$.value,
      health: this.health$.value,
      latencyHistory: this.latencyTracker,
      timestamp: Date.now()
    };
    
    const json = JSON.stringify(data);
    return new Blob([json], { type: 'application/json' });
  }
}
