// src/stores/authStore.ts
// NEXLIFY AUTHENTICATION - The gatekeeper of the neural vault
// Last sync: 2025-06-19 | "Your password is your lifeline"

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';
import { invoke } from '@tauri-apps/api/core';

// Constants - the rules of the stronghold
const MAX_LOGIN_ATTEMPTS = 3;
const LOCKOUT_DURATION = 30 * 60 * 1000; // 30 minutes in milliseconds
const SESSION_TIMEOUT = 15 * 60 * 1000; // 15 minutes of inactivity

// Types - the shape of security
interface User {
  id: string;
  username: string;
  email: string;
  roles: string[];
}

interface AuthState {
  // Authentication status - are we in or out?
  isAuthenticated: boolean;
  sessionId: string | null;
  sessionExpiry: Date | null;
  permissions: string[];
  lastActivity: Date | null;
  
  // Security measures - the vault's defenses
  failedAttempts: number;
  lockoutUntil: Date | null;
  deviceTrusted: boolean;
  strongholdStatus: 'locked' | 'unlocked' | 'pending';
  
  // Error handling - when the chrome fails
  error: string | null;
  
  // User data
  user: User | null;
  
  // Actions - the keys to the kingdom
  authenticate: (password: string) => Promise<void>;
  lock: () => Promise<void>;
  checkSession: () => Promise<boolean>;
  refreshSession: () => Promise<void>;
  clearAuthData: () => void;
  updateActivity: () => void;
  clearError: () => void;
  unlock: (password?: string) => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    immer((set, get) => ({
      // Initial state - locked down tight
      isAuthenticated: false,
      sessionId: null,
      sessionExpiry: null,
      permissions: [],
      lastActivity: null,
      failedAttempts: 0,
      lockoutUntil: null,
      deviceTrusted: false,
      strongholdStatus: 'locked',
      error: null,
      user: null,
      
      authenticate: async (password: string) => {
        const state = get();
        
        // Check lockout status - no second chances
        if (state.lockoutUntil && new Date() < new Date(state.lockoutUntil)) {
          const remaining = Math.ceil((new Date(state.lockoutUntil).getTime() - Date.now()) / 60000);
          throw new Error(`Account locked. Try again in ${remaining} minutes.`);
        }
        
        set((draft) => {
          draft.error = null;
          draft.strongholdStatus = 'pending';
        });
        
        try {
          // Call Tauri backend for authentication
          const response = await invoke<{
            session_id: string;
            expires_at: string;
            permissions: string[];
            user: User;
          }>('authenticate', { password });
          
          // Update state - welcome to the matrix
          set((draft) => {
            draft.isAuthenticated = true;
            draft.sessionId = response.session_id;
            draft.sessionExpiry = new Date(response.expires_at);
            draft.permissions = response.permissions;
            draft.lastActivity = new Date();
            draft.failedAttempts = 0;
            draft.lockoutUntil = null;
            draft.strongholdStatus = 'unlocked';
            draft.user = response.user;
          });
          
          // Start session monitoring - the silent guardian
          startSessionMonitor();
          
          console.log('🔓 Neural vault unlocked - welcome to the sprawl');
          
        } catch (error) {
          // Handle authentication failure - the chrome rejects
          set((draft) => {
            draft.failedAttempts += 1;
            draft.error = 'Invalid credentials. Check your neural link.';
            
            // Lockout after too many attempts
            if (draft.failedAttempts >= MAX_LOGIN_ATTEMPTS) {
              draft.lockoutUntil = new Date(Date.now() + LOCKOUT_DURATION);
              draft.error = `Too many failed attempts. Account locked for 30 minutes. Take a walk, clear your head.`;
            }
          });
          
          console.error('❌ Authentication failed:', error);
          throw error;
        } finally {
          set((draft) => {
            draft.strongholdStatus = 'locked';
          });
        }
      },
      
      lock: async () => {
        try {
          set((draft) => {
            draft.isAuthenticated = false;
            draft.sessionId = null;
            draft.sessionExpiry = null;
            draft.permissions = [];
            draft.lastActivity = null;
            draft.strongholdStatus = 'locked';
            draft.user = null;
          });
          console.log('🔒 Neural vault locked - stay safe out there');
        } catch (error) {
          console.error('Failed to lock:', error);
          get().clearAuthData();
        }
      },
      
      checkSession: async () => {
        const state = get();
        if (!state.sessionId || !state.sessionExpiry) {
          return false;
        }
        if (new Date() > state.sessionExpiry) {
          get().clearAuthData();
          return false;
        }
        if (state.lastActivity) {
          const inactiveTime = Date.now() - state.lastActivity.getTime();
          if (inactiveTime > SESSION_TIMEOUT) {
            console.warn('⏰ Session expired due to inactivity');
            get().clearAuthData();
            return false;
          }
        }
        return true;
      },
      
      refreshSession: async () => {
        const isValid = await get().checkSession();
        if (!isValid) {
          throw new Error('Session invalid - please re-authenticate');
        }
        set((draft) => {
          draft.lastActivity = new Date();
        });
        console.log('🔄 Session refreshed - neural link stable');
      },
      
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
          draft.user = null;
        });
      },
      
      updateActivity: () => {
        set((draft) => {
          draft.lastActivity = new Date();
        });
      },
      
      clearError: () => {
        set((draft) => {
          draft.error = null;
        });
      },
    })),
    {
      name: 'nexlify-auth',
      serialize: {
        options: {
          map: true,
          set: true,
          date: true,
        },
      },
    }
  )
);

let sessionMonitorInterval: number | null = null;

function startSessionMonitor() {
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
  }, 60000);
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (sessionMonitorInterval) {
      clearInterval(sessionMonitorInterval);
    }
  });
}