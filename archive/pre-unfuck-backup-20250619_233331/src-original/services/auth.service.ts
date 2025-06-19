// Location: /src/services/auth.service.ts
// Nexlify Authentication Service - Your first and last line of defense

import { BehaviorSubject, Subject, fromEvent, interval } from 'rxjs';
import { filter, map, takeUntil, throttleTime } from 'rxjs/operators';
import bcrypt from 'bcryptjs';
import * as OTPAuth from 'otpauth';
import { PublicKeyCredentialCreationOptions } from '@simplewebauthn/typescript-types';
import { startAuthentication, startRegistration } from '@simplewebauthn/browser';

interface AuthConfig {
  security: {
    maxAttempts: number;
    lockoutDuration: number; // ms
    sessionTimeout: number; // ms
    passwordMinLength: number;
    require2FA: boolean;
    biometricEnabled: boolean;
    hardwareKeyEnabled: boolean;
  };
  encryption: {
    saltRounds: number;
    keyDerivation: 'pbkdf2' | 'argon2' | 'scrypt';
    iterations: number;
  };
  monitoring: {
    logFailedAttempts: boolean;
    geoFencing: boolean;
    behavioralAnalysis: boolean;
    anomalyDetection: boolean;
  };
}

interface Session {
  id: string;
  userId: string;
  token: string;
  refreshToken: string;
  createdAt: number;
  expiresAt: number;
  lastActivity: number;
  deviceFingerprint: string;
  ipAddress: string;
  location?: {
    lat: number;
    lng: number;
    city: string;
    country: string;
  };
}

interface AuthAttempt {
  timestamp: number;
  method: 'password' | 'biometric' | 'hardware' | '2fa';
  success: boolean;
  ipAddress: string;
  deviceFingerprint: string;
  reason?: string;
}

interface BiometricData {
  type: 'fingerprint' | 'face' | 'iris';
  template: string;
  confidence: number;
  lastUsed: number;
}

export class AuthService {
  // State management
  private currentSession$ = new BehaviorSubject<Session | null>(null);
  private authAttempts$ = new Subject<AuthAttempt>();
  private securityEvents$ = new Subject<SecurityEvent>();
  
  // Security tracking
  private failedAttempts = new Map<string, number>();
  private lockedAccounts = new Map<string, number>();
  private activeSessions = new Map<string, Session>();
  private deviceFingerprints = new Map<string, string>();
  
  // Biometric simulation (in production, use WebAuthn)
  private biometricTemplates = new Map<string, BiometricData>();
  
  // Hardware key management
  private registeredKeys = new Map<string, PublicKeyCredentialCreationOptions>();
  
  // 2FA management
  private totpSecrets = new Map<string, string>();
  
  // Behavioral analysis
  private behaviorProfiles = new Map<string, BehaviorProfile>();
  
  private config: AuthConfig;
  private sessionCheckInterval: any;

  constructor(config: AuthConfig) {
    this.config = config;
    this.initializeSecurityMonitoring();
    this.startSessionManagement();
    this.setupBehavioralTracking();
  }

  // Core authentication methods
  public async authenticateUser(password: string, userId?: string): Promise<boolean> {
    const fingerprint = await this.getDeviceFingerprint();
    const ipAddress = await this.getIPAddress();
    
    // Check if account is locked
    if (this.isAccountLocked(fingerprint)) {
      this.recordAttempt('password', false, 'account_locked');
      throw new Error('Account temporarily locked. Too many failed attempts.');
    }
    
    try {
      // In production, fetch hashed password from database
      const storedHash = await this.getStoredPasswordHash(userId || fingerprint);
      
      if (!storedHash) {
        this.recordAttempt('password', false, 'user_not_found');
        return false;
      }
      
      // Verify password
      const isValid = await bcrypt.compare(password, storedHash);
      
      if (!isValid) {
        this.handleFailedAttempt(fingerprint);
        this.recordAttempt('password', false, 'invalid_password');
        return false;
      }
      
      // Check for anomalies
      if (this.config.monitoring.anomalyDetection) {
        const anomalyScore = await this.checkForAnomalies(fingerprint, ipAddress);
        if (anomalyScore > 0.8) {
          this.recordAttempt('password', false, 'anomaly_detected');
          this.securityEvents$.next({
            type: 'anomaly',
            severity: 'high',
            message: 'Suspicious login attempt detected',
            metadata: { anomalyScore, fingerprint, ipAddress }
          });
          return false;
        }
      }
      
      // Password is valid
      this.recordAttempt('password', true);
      this.resetFailedAttempts(fingerprint);
      
      return true;
      
    } catch (error) {
      console.error('[AUTH] Authentication error:', error);
      this.recordAttempt('password', false, 'system_error');
      return false;
    }
  }

  public async verify2FA(code: string, method: '2fa' | 'sms' | 'email'): Promise<boolean> {
    const fingerprint = await this.getDeviceFingerprint();
    
    try {
      if (method === '2fa') {
        // TOTP verification
        const secret = this.totpSecrets.get(fingerprint);
        if (!secret) {
          this.recordAttempt('2fa', false, 'no_2fa_setup');
          return false;
        }
        
        const totp = new OTPAuth.TOTP({
          secret: OTPAuth.Secret.fromBase32(secret),
          algorithm: 'SHA256',
          digits: 6,
          period: 30
        });
        
        const isValid = totp.validate({
          token: code,
          window: 1 // Allow 1 period before/after
        }) !== null;
        
        this.recordAttempt('2fa', isValid);
        return isValid;
        
      } else {
        // SMS/Email verification (simulated)
        // In production, check against sent codes
        const isValid = code === '123456'; // Demo only
        this.recordAttempt('2fa', isValid);
        return isValid;
      }
      
    } catch (error) {
      console.error('[AUTH] 2FA verification error:', error);
      this.recordAttempt('2fa', false, 'verification_error');
      return false;
    }
  }

  public async checkHardwareKey(): Promise<boolean> {
    try {
      // Use WebAuthn for hardware key authentication
      const fingerprint = await this.getDeviceFingerprint();
      const challenge = this.generateChallenge();
      
      const credential = await startAuthentication({
        challenge,
        rpId: window.location.hostname,
        userVerification: 'preferred',
        timeout: 60000
      });
      
      // Verify the credential (in production, verify server-side)
      const isValid = this.verifyHardwareKeyCredential(credential, fingerprint);
      
      this.recordAttempt('hardware', isValid);
      return isValid;
      
    } catch (error) {
      console.error('[AUTH] Hardware key check failed:', error);
      this.recordAttempt('hardware', false, 'hardware_key_error');
      return false;
    }
  }

  public async registerHardwareKey(userId: string): Promise<void> {
    try {
      const challenge = this.generateChallenge();
      
      const registration = await startRegistration({
        challenge,
        rp: {
          name: 'Nexlify Trading',
          id: window.location.hostname
        },
        user: {
          id: new TextEncoder().encode(userId),
          name: userId,
          displayName: `User ${userId}`
        },
        pubKeyCredParams: [
          { alg: -7, type: 'public-key' }, // ES256
          { alg: -257, type: 'public-key' } // RS256
        ],
        authenticatorSelection: {
          authenticatorAttachment: 'cross-platform',
          userVerification: 'preferred'
        },
        timeout: 60000,
        attestation: 'direct'
      });
      
      // Store registration
      this.registeredKeys.set(userId, registration as any);
      
      this.securityEvents$.next({
        type: 'hardware_key_registered',
        severity: 'info',
        message: 'Hardware key successfully registered',
        metadata: { userId }
      });
      
    } catch (error) {
      console.error('[AUTH] Hardware key registration failed:', error);
      throw error;
    }
  }

  public async createSession(userId: string): Promise<Session> {
    const fingerprint = await this.getDeviceFingerprint();
    const ipAddress = await this.getIPAddress();
    const location = await this.getLocation(ipAddress);
    
    const session: Session = {
      id: this.generateSessionId(),
      userId,
      token: this.generateSecureToken(),
      refreshToken: this.generateSecureToken(),
      createdAt: Date.now(),
      expiresAt: Date.now() + this.config.security.sessionTimeout,
      lastActivity: Date.now(),
      deviceFingerprint: fingerprint,
      ipAddress,
      location
    };
    
    // Store session
    this.activeSessions.set(session.id, session);
    this.currentSession$.next(session);
    
    // Log session creation
    this.securityEvents$.next({
      type: 'session_created',
      severity: 'info',
      message: 'New session created',
      metadata: { 
        sessionId: session.id, 
        userId, 
        location: location?.city || 'Unknown' 
      }
    });
    
    return session;
  }

  public async refreshSession(refreshToken: string): Promise<Session | null> {
    const currentSession = Array.from(this.activeSessions.values())
      .find(s => s.refreshToken === refreshToken);
    
    if (!currentSession) {
      this.securityEvents$.next({
        type: 'invalid_refresh_token',
        severity: 'warning',
        message: 'Invalid refresh token used'
      });
      return null;
    }
    
    // Check if session expired
    if (currentSession.expiresAt < Date.now()) {
      this.destroySession(currentSession.id);
      return null;
    }
    
    // Refresh the session
    currentSession.token = this.generateSecureToken();
    currentSession.expiresAt = Date.now() + this.config.security.sessionTimeout;
    currentSession.lastActivity = Date.now();
    
    this.currentSession$.next(currentSession);
    
    return currentSession;
  }

  public destroySession(sessionId: string): void {
    const session = this.activeSessions.get(sessionId);
    if (session) {
      this.activeSessions.delete(sessionId);
      
      if (this.currentSession$.value?.id === sessionId) {
        this.currentSession$.next(null);
      }
      
      this.securityEvents$.next({
        type: 'session_destroyed',
        severity: 'info',
        message: 'Session terminated',
        metadata: { sessionId, userId: session.userId }
      });
    }
  }

  // Biometric methods (simulated)
  public async registerBiometric(type: BiometricData['type']): Promise<void> {
    const fingerprint = await this.getDeviceFingerprint();
    
    // In production, use WebAuthn for biometric registration
    const template = this.generateBiometricTemplate();
    
    this.biometricTemplates.set(fingerprint, {
      type,
      template,
      confidence: 0.95,
      lastUsed: Date.now()
    });
    
    this.securityEvents$.next({
      type: 'biometric_registered',
      severity: 'info',
      message: `${type} biometric registered`,
      metadata: { type }
    });
  }

  public async verifyBiometric(type: BiometricData['type']): Promise<boolean> {
    const fingerprint = await this.getDeviceFingerprint();
    const stored = this.biometricTemplates.get(fingerprint);
    
    if (!stored || stored.type !== type) {
      return false;
    }
    
    // Simulate biometric matching with some randomness
    const matchScore = 0.85 + Math.random() * 0.15;
    const isMatch = matchScore > 0.9;
    
    if (isMatch) {
      stored.lastUsed = Date.now();
    }
    
    this.recordAttempt('biometric', isMatch);
    return isMatch;
  }

  // Security monitoring
  private initializeSecurityMonitoring(): void {
    // Monitor for suspicious patterns
    this.authAttempts$.pipe(
      throttleTime(1000),
      filter(attempt => !attempt.success)
    ).subscribe(attempt => {
      this.analyzeFailedAttempt(attempt);
    });
    
    // Monitor for brute force attempts
    interval(60000).subscribe(() => {
      this.checkForBruteForce();
    });
  }

  private startSessionManagement(): void {
    // Check session validity every minute
    this.sessionCheckInterval = setInterval(() => {
      const now = Date.now();
      
      this.activeSessions.forEach((session, id) => {
        // Check expiration
        if (session.expiresAt < now) {
          this.destroySession(id);
          return;
        }
        
        // Check inactivity
        const inactivityPeriod = now - session.lastActivity;
        if (inactivityPeriod > this.config.security.sessionTimeout / 2) {
          this.securityEvents$.next({
            type: 'session_inactive',
            severity: 'warning',
            message: 'Session approaching timeout',
            metadata: { sessionId: id, inactivityMinutes: inactivityPeriod / 60000 }
          });
        }
      });
    }, 60000);
  }

  private setupBehavioralTracking(): void {
    if (!this.config.monitoring.behavioralAnalysis) return;
    
    // Track mouse movements
    fromEvent<MouseEvent>(document, 'mousemove').pipe(
      throttleTime(100)
    ).subscribe(event => {
      this.updateBehaviorProfile('mouse', {
        x: event.clientX,
        y: event.clientY,
        time: Date.now()
      });
    });
    
    // Track typing patterns
    fromEvent<KeyboardEvent>(document, 'keydown').pipe(
      throttleTime(50)
    ).subscribe(event => {
      this.updateBehaviorProfile('typing', {
        key: event.key.length === 1 ? 'char' : event.key,
        time: Date.now()
      });
    });
  }

  // Helper methods
  private async getDeviceFingerprint(): Promise<string> {
    // Generate device fingerprint from multiple sources
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx!.textBaseline = 'top';
    ctx!.font = '14px Arial';
    ctx!.fillText('Nexlify Trading 🚀', 2, 2);
    const canvasData = canvas.toDataURL();
    
    const fingerprint = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      colorDepth: screen.colorDepth,
      screenResolution: `${screen.width}x${screen.height}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      canvas: canvasData.slice(-50), // Last 50 chars of canvas
      webgl: this.getWebGLFingerprint()
    };
    
    // Hash the fingerprint
    const str = JSON.stringify(fingerprint);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    
    return Math.abs(hash).toString(16);
  }

  private getWebGLFingerprint(): string {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) return 'no-webgl';
      
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (!debugInfo) return 'no-debug-info';
      
      return gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) + 
             gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    } catch (error) {
      return 'webgl-error';
    }
  }

  private async getIPAddress(): Promise<string> {
    // In production, get from server
    // For demo, return placeholder
    return '192.168.1.100';
  }

  private async getLocation(ipAddress: string): Promise<Session['location']> {
    // In production, use IP geolocation service
    return {
      lat: 51.9244,
      lng: 4.4777,
      city: 'Rotterdam',
      country: 'Netherlands'
    };
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }

  private generateSecureToken(): string {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
  }

  private generateChallenge(): string {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return btoa(String.fromCharCode.apply(null, Array.from(array)));
  }

  private generateBiometricTemplate(): string {
    // Simulate biometric template
    return this.generateSecureToken();
  }

  private async getStoredPasswordHash(identifier: string): Promise<string | null> {
    // In production, fetch from database
    // For demo, return a bcrypt hash of "password123"
    return '$2a$10$YourHashHere';
  }

  private verifyHardwareKeyCredential(credential: any, fingerprint: string): boolean {
    // In production, verify credential signature server-side
    return true; // Simplified for demo
  }

  private isAccountLocked(fingerprint: string): boolean {
    const lockTime = this.lockedAccounts.get(fingerprint);
    if (!lockTime) return false;
    
    if (Date.now() > lockTime) {
      this.lockedAccounts.delete(fingerprint);
      return false;
    }
    
    return true;
  }

  private handleFailedAttempt(fingerprint: string): void {
    const attempts = (this.failedAttempts.get(fingerprint) || 0) + 1;
    this.failedAttempts.set(fingerprint, attempts);
    
    if (attempts >= this.config.security.maxAttempts) {
      this.lockAccount(fingerprint);
    }
  }

  private lockAccount(fingerprint: string): void {
    const lockUntil = Date.now() + this.config.security.lockoutDuration;
    this.lockedAccounts.set(fingerprint, lockUntil);
    this.failedAttempts.delete(fingerprint);
    
    this.securityEvents$.next({
      type: 'account_locked',
      severity: 'warning',
      message: 'Account locked due to multiple failed attempts',
      metadata: { fingerprint, lockUntil }
    });
  }

  private resetFailedAttempts(fingerprint: string): void {
    this.failedAttempts.delete(fingerprint);
  }

  private recordAttempt(
    method: AuthAttempt['method'], 
    success: boolean, 
    reason?: string
  ): void {
    this.getDeviceFingerprint().then(fingerprint => {
      this.getIPAddress().then(ipAddress => {
        this.authAttempts$.next({
          timestamp: Date.now(),
          method,
          success,
          ipAddress,
          deviceFingerprint: fingerprint,
          reason
        });
      });
    });
  }

  private async checkForAnomalies(
    fingerprint: string, 
    ipAddress: string
  ): Promise<number> {
    // Simple anomaly detection
    let score = 0;
    
    // Check for new device
    if (!this.deviceFingerprints.has(fingerprint)) {
      score += 0.3;
    }
    
    // Check for unusual location (would use real geolocation)
    const lastSession = Array.from(this.activeSessions.values())
      .find(s => s.deviceFingerprint === fingerprint);
    
    if (lastSession && lastSession.ipAddress !== ipAddress) {
      score += 0.4;
    }
    
    // Check behavioral patterns
    const behavior = this.behaviorProfiles.get(fingerprint);
    if (behavior) {
      // Analyze typing speed, mouse patterns, etc.
      // Simplified for demo
      score += Math.random() * 0.3;
    }
    
    return Math.min(score, 1);
  }

  private analyzeFailedAttempt(attempt: AuthAttempt): void {
    // Look for patterns in failed attempts
    // In production, use ML for pattern recognition
  }

  private checkForBruteForce(): void {
    const threshold = 10; // Failed attempts per minute
    const window = 60000; // 1 minute
    const now = Date.now();
    
    const recentAttempts = new Map<string, number>();
    
    // Count recent failed attempts by IP
    // (Would need to store attempts properly in production)
  }

  private updateBehaviorProfile(type: string, data: any): void {
    const fingerprint = this.deviceFingerprints.get('current') || 'unknown';
    
    if (!this.behaviorProfiles.has(fingerprint)) {
      this.behaviorProfiles.set(fingerprint, {
        mousePatterns: [],
        typingPatterns: [],
        navigationPatterns: []
      });
    }
    
    const profile = this.behaviorProfiles.get(fingerprint)!;
    
    switch (type) {
      case 'mouse':
        profile.mousePatterns.push(data);
        if (profile.mousePatterns.length > 1000) {
          profile.mousePatterns.shift();
        }
        break;
        
      case 'typing':
        profile.typingPatterns.push(data);
        if (profile.typingPatterns.length > 1000) {
          profile.typingPatterns.shift();
        }
        break;
    }
  }

  // Public getters
  public getCurrentSession$() {
    return this.currentSession$.asObservable();
  }

  public getAuthAttempts$() {
    return this.authAttempts$.asObservable();
  }

  public getSecurityEvents$() {
    return this.securityEvents$.asObservable();
  }

  // Cleanup
  public destroy(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
    }
    
    // Clear all sessions
    this.activeSessions.forEach((_, id) => this.destroySession(id));
    
    console.log('[AUTH] Authentication service destroyed');
  }
}

// Type definitions
interface SecurityEvent {
  type: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  metadata?: any;
}

interface BehaviorProfile {
  mousePatterns: any[];
  typingPatterns: any[];
  navigationPatterns: any[];
}
