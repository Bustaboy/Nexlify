/**
 * Nexlify Auth Store
 * Handles PIN authentication, 2FA, and session management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import jwt_decode from 'jwt-decode';
import toast from 'react-hot-toast';

// API client
import { apiClient } from '../lib/api';
import { playSound } from '../lib/sounds';

interface AuthState {
  // State
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  pin: string | null;
  requires2FA: boolean;
  sessionExpiry: Date | null;
  failedAttempts: number;
  isLocked: boolean;
  lockoutUntil: Date | null;
  
  // Actions
  authenticateWithPin: (pin: string, enable2FA?: boolean) => Promise<AuthResult>;
  verify2FA: (code: string) => Promise<boolean>;
  logout: () => void;
  checkSession: () => Promise<void>;
  updateFailedAttempts: () => void;
  clearLockout: () => void;
  setPin: (newPin: string) => Promise<boolean>;
}

interface User {
  id: string;
  deviceId: string;
  createdAt: Date;
  lastLogin: Date;
  settings?: Record<string, any>;
}

interface AuthResult {
  success: boolean;
  requires2FA?: boolean;
  totpUri?: string;
  totpSecret?: string;
  error?: string;
}

interface JWTPayload {
  sub: string;
  exp: number;
  device_id: string;
}

const MAX_ATTEMPTS = 5;
const LOCKOUT_DURATION = 5 * 60 * 1000; // 5 minutes

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        isAuthenticated: false,
        token: null,
        user: null,
        pin: null,
        requires2FA: false,
        sessionExpiry: null,
        failedAttempts: 0,
        isLocked: false,
        lockoutUntil: null,

        // Authenticate with PIN
        authenticateWithPin: async (pin: string, enable2FA = false): Promise<AuthResult> => {
          const state = get();
          
          // Check if locked
          if (state.isLocked && state.lockoutUntil) {
            const now = new Date();
            if (now < state.lockoutUntil) {
              const remainingMinutes = Math.ceil(
                (state.lockoutUntil.getTime() - now.getTime()) / 60000
              );
              return {
                success: false,
                error: `Account locked. Try again in ${remainingMinutes} minutes.`
              };
            } else {
              // Lockout expired
              set((draft) => {
                draft.isLocked = false;
                draft.lockoutUntil = null;
                draft.failedAttempts = 0;
              });
            }
          }

          try {
            // Get device ID from Electron
            const systemInfo = await window.nexlify.system.getInfo();
            const deviceId = systemInfo.platform + '_' + btoa(systemInfo.paths.userData).slice(0, 8);

            const response = await apiClient.post('/auth/pin', {
              pin,
              device_id: deviceId,
              enable_2fa: enable2FA
            });

            const { access_token, requires_2fa, totp_uri, totp_secret } = response.data;

            // Decode token
            const decoded = jwt_decode<JWTPayload>(access_token);
            
            set((draft) => {
              draft.token = access_token;
              draft.sessionExpiry = new Date(decoded.exp * 1000);
              draft.failedAttempts = 0;
              draft.isLocked = false;
              draft.lockoutUntil = null;
              
              if (!requires_2fa) {
                // Fully authenticated
                draft.isAuthenticated = true;
                draft.user = {
                  id: decoded.sub,
                  deviceId: decoded.device_id,
                  createdAt: new Date(),
                  lastLogin: new Date()
                };
              } else {
                // Need 2FA
                draft.requires2FA = true;
              }
            });

            // Store token in API client
            apiClient.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

            // Play success sound
            playSound('success');

            // Show appropriate message
            if (requires_2fa) {
              toast.success('PIN verified. Please complete 2FA setup.');
            } else {
              toast.success('Welcome to Nexlify Trading Matrix');
              
              // Send notification
              window.nexlify.notification.show({
                title: 'Nexlify Active',
                body: 'Trading matrix initialized. Good hunting.',
                silent: false
              });
            }

            return {
              success: true,
              requires2FA: requires_2fa,
              totpUri: totp_uri,
              totpSecret: totp_secret
            };

          } catch (error: any) {
            const errorMessage = error.response?.data?.detail || 'Authentication failed';
            
            // Update failed attempts
            get().updateFailedAttempts();
            
            // Play error sound
            playSound('error');
            
            toast.error(errorMessage);
            
            return {
              success: false,
              error: errorMessage
            };
          }
        },

        // Verify 2FA code
        verify2FA: async (code: string): Promise<boolean> => {
          const state = get();
          
          if (!state.token) {
            toast.error('No active session');
            return false;
          }

          try {
            await apiClient.post('/auth/2fa/verify', {
              token: state.token,
              code
            });

            const decoded = jwt_decode<JWTPayload>(state.token);

            set((draft) => {
              draft.isAuthenticated = true;
              draft.requires2FA = false;
              draft.user = {
                id: decoded.sub,
                deviceId: decoded.device_id,
                createdAt: new Date(),
                lastLogin: new Date()
              };
            });

            playSound('success');
            toast.success('2FA verification successful');
            
            // Send notification
            window.nexlify.notification.show({
              title: '2FA Enabled',
              body: 'Two-factor authentication has been activated',
              silent: false
            });

            return true;

          } catch (error: any) {
            playSound('error');
            toast.error(error.response?.data?.detail || '2FA verification failed');
            return false;
          }
        },

        // Logout
        logout: () => {
          set((draft) => {
            draft.isAuthenticated = false;
            draft.token = null;
            draft.user = null;
            draft.requires2FA = false;
            draft.sessionExpiry = null;
          });

          // Clear API token
          delete apiClient.defaults.headers.common['Authorization'];
          
          // Clear any stored data
          localStorage.removeItem('nexlify-trading-store');
          
          playSound('click');
          toast.success('Logged out successfully');
        },

        // Check session validity
        checkSession: async () => {
          const state = get();
          
          if (!state.token || !state.sessionExpiry) {
            return;
          }

          const now = new Date();
          
          // Check if expired
          if (now >= state.sessionExpiry) {
            get().logout();
            toast.error('Session expired. Please login again.');
            return;
          }

          // Refresh token if close to expiry (within 1 hour)
          const oneHourFromNow = new Date(now.getTime() + 60 * 60 * 1000);
          
          if (state.sessionExpiry <= oneHourFromNow) {
            try {
              // In a real app, implement token refresh endpoint
              // For now, just warn user
              const remainingMinutes = Math.ceil(
                (state.sessionExpiry.getTime() - now.getTime()) / 60000
              );
              
              toast(`Session expires in ${remainingMinutes} minutes`, {
                icon: '⏰',
              });
              
            } catch (error) {
              console.error('Session refresh failed:', error);
            }
          }

          // Restore API token
          if (state.token) {
            apiClient.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
          }
        },

        // Update failed attempts
        updateFailedAttempts: () => {
          set((draft) => {
            draft.failedAttempts += 1;
            
            if (draft.failedAttempts >= MAX_ATTEMPTS) {
              draft.isLocked = true;
              draft.lockoutUntil = new Date(Date.now() + LOCKOUT_DURATION);
              
              toast.error(`Too many failed attempts. Account locked for ${LOCKOUT_DURATION / 60000} minutes.`, {
                duration: 6000
              });
            } else {
              const remaining = MAX_ATTEMPTS - draft.failedAttempts;
              toast.error(`Invalid PIN. ${remaining} attempts remaining.`);
            }
          });
        },

        // Clear lockout (admin function)
        clearLockout: () => {
          set((draft) => {
            draft.isLocked = false;
            draft.lockoutUntil = null;
            draft.failedAttempts = 0;
          });
          
          toast.success('Lockout cleared');
        },

        // Set new PIN
        setPin: async (newPin: string): Promise<boolean> => {
          try {
            // In real app, this would update on backend
            // For now, just update local state
            set((draft) => {
              draft.pin = newPin;
            });

            await window.nexlify.config.set('userPin', newPin);
            
            toast.success('PIN updated successfully');
            return true;
            
          } catch (error) {
            toast.error('Failed to update PIN');
            return false;
          }
        }
      })),
      {
        name: 'nexlify-auth-store',
        partialize: (state) => ({
          // Don't persist sensitive data
          isAuthenticated: state.isAuthenticated,
          user: state.user,
          sessionExpiry: state.sessionExpiry
        })
      }
    ),
    {
      name: 'NexlifyAuth'
    }
  )
);
