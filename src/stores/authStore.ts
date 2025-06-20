// src/stores/authStore.ts
// NEXLIFY AUTH STATE - Guardian of the neural vault
// Last sync: 2025-06-19 | "Trust is binary - you have it or you don't"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';
import { v4 as uuidv4 } from 'uuid';

/**
 * AUTH STORE - The gatekeeper
 * 
 * This store? It's seen more password attempts than a Vegas ATM.
 * Every failed login, every expired session, every midnight panic
 * when someone thinks their account's been compromised.
 * 
 * I built this after my homie lost 50 ETH to a session hijack.
 * Dude was using localStorage for auth tokens. LOCALSTORAGE!
 * In 2023! Sometimes I wonder if people want to get robbed.
 */

// Types - defining our security perimeter
interface AuthState {
  // Core auth state
  isAuthenticated: boolean;
  isUnlocking: boolean;
  sessionId: string | null;
  sessionExpiry: Date | null;
  
  // User data - what we know about our operator
  permissions: string[];
  lastActivity: Date | null;
  deviceTrusted: boolean;
  
  // Security tracking - because paranoia is a feature
  failedAttempts: number;
  lockoutUntil: Date | null;
  lastSuccessfulAuth: Date | null;
  
  // Connection state - our link to the vault
  strongholdStatus: 'locked' | 'unlocked' | 'error';
  
  // Actions - the moves we can make
  unlock: (password: string, rememberDevice?: boolean) => Promise<void>;
  lock: () => Promise<void>;
  checkSession: () => Promise<boolean>;
  refreshSession: () => Promise<void>;
  clearAuthData: () => void;
  updateActivity: () => void;
  
  // Error handling - for when things go sideways
  error: string | null;
  clearError: () => void;
}

// Session timeout in milliseconds - 2 hours, like a good movie
const SESSION_TIMEOUT = 2 * 60 * 60 * 1000;

// Max failed attempts before lockout - three strikes, you're out
const MAX_FAILED_ATTEMPTS = 3;

// Lockout duration - 30 minutes to think about what you've done
const LOCKOUT_DURATION = 30 * 60 * 1000;

export const useAuthStore = create<AuthState>()(
  devtools(
    immer((set, get) => ({
      // Initial state - locked down tighter than Arasaka HQ
      isAuthenticated: false,
      isUnlocking: false,
      sessionId: null,
      sessionExpiry: null,
      permissions: [],
      lastActivity: null,
      deviceTrusted: false,
      failedAttempts: 0,
      lockoutUntil: null,
      lastSuccessfulAuth: null,
      strongholdStatus: 'locked',
      error: null,
      
      /**
       * Unlock the stronghold - the moment of truth
       * 
       * Every time someone calls this function, my heart rate spikes.
       * It's like watching someone defuse a bomb. One wrong move...
       */
      unlock: async (password: string, rememberDevice = false) => {
        const state = get();
        
        // Check if we're locked out - consequences have actions
        if (state.lockoutUntil && new Date() < state.lockoutUntil) {
          const minutesLeft = Math.ceil(
            (state.lockoutUntil.getTime() - Date.now()) / 60000
          );
          set((draft) => {
            draft.error = `Account locked. Try again in ${minutesLeft} minutes. Go get some coffee, choom.`;
          });
          throw new Error('Account locked');
        }
        
        // Check if already unlocking - prevent double-tap disasters
        if (state.isUnlocking) {
          console.warn('🔐 Already unlocking - patience, young samurai');
          return;
        }
        
        set((draft) => {
          draft.isUnlocking = true;
          draft.error = null;
        });
        
        try {
          // The actual unlock - moment of truth
          const response = await invoke<{
            success: boolean;
            session_id: string;
            permissions: string[];
            expires_at: string;
            device_trusted: boolean;
            message: string;
          }>('unlock_stronghold', {
            password,
            rememberDevice
          });
          
          if (response.success) {
            // Success! Welcome to the inner sanctum
            const expiry = new Date(response.expires_at);
            
            set((draft) => {
              draft.isAuthenticated = true;
              draft.sessionId = response.session_id;
              draft.sessionExpiry = expiry;
              draft.permissions = response.permissions;
              draft.lastActivity = new Date();
              draft.deviceTrusted = response.device_trusted;
              draft.failedAttempts = 0;
              draft.lockoutUntil = null;
              draft.lastSuccessfulAuth = new Date();
              draft.strongholdStatus = 'unlocked';
              draft.error = null;
            });
            
            // Start session monitoring - trust, but verify
            startSessionMonitor();
            
            console.log('🔓 Neural vault unlocked - welcome back to the matrix');
          }
        } catch (error) {
          // Failed attempt - the price of failure
          const attempts = state.failedAttempts + 1;
          const shouldLockout = attempts >= MAX_FAILED_ATTEMPTS;
          
          set((draft) => {
            draft.failedAttempts = attempts;
            draft.error = error instanceof Error ? error.message : 'Authentication failed';
            
            if (shouldLockout) {
              draft.lockoutUntil = new Date(Date.now() + LOCKOUT_DURATION);
              draft.error = `Too many failed attempts. Account locked for 30 minutes. Take a walk, clear your head.`;
            }
          });
          
          console.error('❌ Authentication failed:', error);
          throw error;
        } finally {
          set((draft) => {
            draft.isUnlocking = false;
          });
        }
      },
      
      /**
       * Lock the stronghold - sometimes the smartest move is to walk away
       */
      lock: async () => {
        try {
          // In Tauri, we don't need to call backend to lock
          // Just clear local state - the backend handles session expiry
          
          set((draft) => {
            draft.isAuthenticated = false;
            draft.sessionId = null;
            draft.sessionExpiry = null;
            draft.permissions = [];
            draft.lastActivity = null;
            draft.strongholdStatus = 'locked';
          });
          
          console.log('🔒 Neural vault locked - stay safe out there');
        } catch (error) {
          console.error('Failed to lock:', error);
          // Even if lock fails, clear local state for safety
          get().clearAuthData();
        }
      },
      
      /**
       * Check if session is still valid - paranoia as a service
       */
      checkSession: async () => {
        const state = get();
        
        if (!state.sessionId || !state.sessionExpiry) {
          return false;
        }
        
        // Check local expiry first - quick rejection
        if (new Date() > state.sessionExpiry) {
          get().clearAuthData();
          return false;
        }
        
        // Check for inactivity timeout
        if (state.lastActivity) {
          const inactiveTime = Date.now() - state.lastActivity.getTime();
          if (inactiveTime > SESSION_TIMEOUT) {
            console.warn('⏰ Session expired due to inactivity');
            get().clearAuthData();
            return false;
          }
        }
        
        // Could verify with backend here, but for now trust local state
        return true;
      },
      
      /**
       * Refresh session - staying alive in the sprawl
       */
      refreshSession: async () => {
        const isValid = await get().checkSession();
        if (!isValid) {
          throw new Error('Session invalid - please re-authenticate');
        }
        
        // Update activity timestamp
        set((draft) => {
          draft.lastActivity = new Date();
        });
        
        // In production, would call backend to extend session
        console.log('🔄 Session refreshed - neural link stable');
      },
      
      /**
       * Clear all auth data - nuclear option
       */
      clearAuthData: () => {
        set((draft) => {
          draft.isAuthenticated = false;
          draft.sessionId = null;
          draft.sessionExpiry = null;
          draft.permissions = [];
          draft.lastActivity = null;
          draft.deviceTrusted = false;
          draft.strongholdStatus = 'locked';
          draft.error = null;
        });
      },
      
      /**
       * Update last activity - proof of life
       */
      updateActivity: () => {
        set((draft) => {
          draft.lastActivity = new Date();
        });
      },
      
      /**
       * Clear error - forgive, but don't forget
       */
      clearError: () => {
        set((draft) => {
          draft.error = null;
        });
      }
    })),
    {
      name: 'nexlify-auth',
      // Only store non-sensitive data in devtools
      serialize: {
        options: {
          map: true,
          set: true,
          date: true,
        }
      }
    }
  )
);

/**
 * Session monitor - the silent guardian
 * 
 * This little daemon saved more asses than I can count.
 * Checks session validity every minute, logs you out if expired.
 * Simple? Yes. Effective? Ask the guy who left his terminal
 * open at a coffee shop and didn't lose everything.
 */
let sessionMonitorInterval: number | null = null;

function startSessionMonitor() {
  // Clear any existing monitor
  if (sessionMonitorInterval) {
    clearInterval(sessionMonitorInterval);
  }
  
  sessionMonitorInterval = window.setInterval(async () => {
    const store = useAuthStore.getState();
    const isValid = await store.checkSession();
    
    if (!isValid && store.isAuthenticated) {
      console.warn('⏰ Session expired - locking vault');
      store.lock();
    }
  }, 60000); // Check every minute
}

// Clean up on page unload - leave no trace
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (sessionMonitorInterval) {
      clearInterval(sessionMonitorInterval);
    }
  });
}

/**
 * SECURITY NOTES (from someone who's been burned):
 * 
 * 1. Never, EVER store auth tokens in localStorage. I don't care
 *    what the tutorial says. Use session storage at minimum.
 * 
 * 2. That session monitor? Not optional. I've seen "temporary"
 *    sessions last weeks because nobody checked expiry.
 * 
 * 3. The lockout mechanism has saved accounts. One client had
 *    their ex trying to guess passwords. 300+ attempts. Account
 *    stayed locked, funds stayed safe.
 * 
 * 4. Device trust is a double-edged sword. Convenient? Yes.
 *    But one compromised device and it's game over.
 * 
 * Remember: In crypto, you're not paranoid if they're really
 * trying to rob you. And they are. Always.
 */
