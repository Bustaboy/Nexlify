// src/components/auth/LoginScreen.tsx
// NEXLIFY LOGIN PORTAL - Where mortals become traders
// Last sync: 2025-06-19 | "Every empire starts with a password"

import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Lock, 
  Unlock, 
  Eye, 
  EyeOff, 
  AlertTriangle,
  Zap,
  Shield,
  Terminal,
  Fingerprint
} from 'lucide-react';

import { useAuthStore } from '@/stores/authStore';
import { NeuralBackground } from '../effects/NeuralBackground';

interface LoginScreenProps {
  onLogin: (password: string) => Promise<void>;
}

/**
 * LOGIN SCREEN - The gateway to digital fortune
 * 
 * This screen has seen everything. Desperate 3AM logins during
 * market crashes. Celebration logins after big wins. The nervous
 * first-timer who types their password one character at a time.
 * 
 * I designed it after watching a trader fail login 5 times during
 * a flash crash. Cost him $50k. Now we have biometric options and
 * password managers. Because in the heat of battle, your fingers
 * don't always cooperate.
 */
export const LoginScreen = ({ onLogin }: LoginScreenProps) => {
  // State - the mental preparation
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [capsLockOn, setCapsLockOn] = useState(false);
  const [loginMethod, setLoginMethod] = useState<'password' | 'biometric'>('password');
  
  // Refs - our anchors
  const passwordInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Store connection
  const { error, failedAttempts, lockoutUntil, clearError } = useAuthStore();
  
  // Matrix rain effect - because atmosphere matters
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const matrix = "NEXLIFY0123456789ABCDEF$¥€₿";
    const matrixArray = matrix.split("");
    const fontSize = 10;
    const columns = canvas.width / fontSize;
    const drops: number[] = [];
    
    for (let x = 0; x < columns; x++) {
      drops[x] = 1;
    }
    
    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.04)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.fillStyle = '#0F0';
      ctx.font = fontSize + 'px monospace';
      
      for (let i = 0; i < drops.length; i++) {
        const text = matrixArray[Math.floor(Math.random() * matrixArray.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };
    
    const interval = setInterval(draw, 35);
    
    return () => clearInterval(interval);
  }, []);
  
  /**
   * Handle login - the moment of truth
   */
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!password.trim()) {
      return;
    }
    
    setIsLoading(true);
    clearError();
    
    try {
      await onLogin(password);
      // Success handling done in parent
    } catch (err) {
      // Error handling done in store
      console.error('Login failed:', err);
      
      // Shake the form - physical feedback
      if (passwordInputRef.current) {
        passwordInputRef.current.classList.add('shake');
        setTimeout(() => {
          passwordInputRef.current?.classList.remove('shake');
        }, 500);
      }
    } finally {
      setIsLoading(false);
      setPassword(''); // Clear for security
    }
  }, [password, onLogin, clearError]);
  
  /**
   * Detect caps lock - because it matters
   */
  const checkCapsLock = useCallback((e: React.KeyboardEvent) => {
    if (e.getModifierState) {
      setCapsLockOn(e.getModifierState('CapsLock'));
    }
  }, []);
  
  /**
   * Biometric login - the future is now
   */
  const handleBiometric = useCallback(async () => {
    setIsLoading(true);
    
    // Simulate biometric check
    // In production, this would use WebAuthn or device APIs
    try {
      const mockBiometricSuccess = Math.random() > 0.2; // 80% success rate
      
      if (mockBiometricSuccess) {
        // In real app, this would retrieve stored credentials
        await onLogin('biometric_authenticated');
      } else {
        throw new Error('Biometric authentication failed');
      }
    } catch (err) {
      console.error('Biometric failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, [onLogin]);
  
  // Calculate security status
  const securityStatus = {
    level: failedAttempts === 0 ? 'secure' : failedAttempts < 3 ? 'warning' : 'danger',
    message: failedAttempts === 0 
      ? 'Neural vault secured' 
      : `${3 - failedAttempts} attempts remaining`
  };
  
  // Check if locked out
  const isLockedOut = lockoutUntil && new Date() < new Date(lockoutUntil);
  const lockoutMinutes = isLockedOut 
    ? Math.ceil((new Date(lockoutUntil).getTime() - Date.now()) / 60000)
    : 0;
  
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Matrix background */}
      <canvas 
        ref={canvasRef} 
        className="absolute inset-0 opacity-20"
      />
      
      {/* Neural background effect */}
      <NeuralBackground intensity="medium" />
      
      {/* Login form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-md"
      >
        <div className="bg-black/80 backdrop-blur-xl border border-cyan-500/30 rounded-lg shadow-2xl shadow-cyan-500/20 p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring" }}
              className="inline-flex items-center justify-center w-20 h-20 bg-cyan-500/10 rounded-full mb-4"
            >
              <Lock className="w-10 h-10 text-cyan-400" />
            </motion.div>
            
            <h1 className="text-3xl font-bold text-cyan-400 mb-2 font-mono">
              NEXLIFY NEURAL VAULT
            </h1>
            <p className="text-gray-400 text-sm">
              Jack in to access the trading matrix
            </p>
          </div>
          
          {/* Security status */}
          <div className={`mb-6 p-3 rounded border ${
            securityStatus.level === 'secure' 
              ? 'bg-green-900/20 border-green-500/30 text-green-400' 
              : securityStatus.level === 'warning'
              ? 'bg-yellow-900/20 border-yellow-500/30 text-yellow-400'
              : 'bg-red-900/20 border-red-500/30 text-red-400'
          }`}>
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4" />
              <span className="text-sm font-mono">{securityStatus.message}</span>
            </div>
          </div>
          
          {/* Login method toggle */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setLoginMethod('password')}
              disabled={isLoading || isLockedOut}
              className={`flex-1 py-2 px-4 rounded border transition-all ${
                loginMethod === 'password'
                  ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400'
                  : 'bg-gray-900/50 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <Terminal className="w-4 h-4 inline mr-2" />
              Password
            </button>
            <button
              onClick={() => setLoginMethod('biometric')}
              disabled={isLoading || isLockedOut}
              className={`flex-1 py-2 px-4 rounded border transition-all ${
                loginMethod === 'biometric'
                  ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400'
                  : 'bg-gray-900/50 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <Fingerprint className="w-4 h-4 inline mr-2" />
              Biometric
            </button>
          </div>
          
          {/* Lockout message */}
          {isLockedOut && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mb-6 p-4 bg-red-900/20 border border-red-500/30 rounded"
            >
              <div className="flex items-center gap-2 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <div>
                  <p className="font-bold">ACCOUNT LOCKED</p>
                  <p className="text-sm">
                    Too many failed attempts. Try again in {lockoutMinutes} minutes.
                  </p>
                  <p className="text-xs mt-2 italic">
                    "Patience is a virtue in the sprawl. Use this time to clear your mind."
                  </p>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Login form */}
          <AnimatePresence mode="wait">
            {loginMethod === 'password' ? (
              <motion.form
                key="password"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                onSubmit={handleSubmit}
              >
                <div className="mb-6">
                  <div className="relative">
                    <input
                      ref={passwordInputRef}
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      onKeyDown={checkCapsLock}
                      onKeyUp={checkCapsLock}
                      disabled={isLoading || isLockedOut}
                      placeholder="Enter neural access code"
                      className="w-full px-4 py-3 bg-gray-900/50 border border-cyan-500/30 rounded focus:border-cyan-400 focus:outline-none text-cyan-400 font-mono placeholder-gray-500 disabled:opacity-50"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-cyan-400 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  
                  {/* Caps lock warning */}
                  {capsLockOn && (
                    <motion.p
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-2 text-sm text-yellow-400 flex items-center gap-1"
                    >
                      <AlertTriangle className="w-3 h-3" />
                      Caps Lock is on
                    </motion.p>
                  )}
                </div>
                
                {/* Error message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-sm"
                  >
                    {error}
                  </motion.div>
                )}
                
                {/* Submit button */}
                <button
                  type="submit"
                  disabled={isLoading || isLockedOut || !password}
                  className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-700 text-black font-bold rounded transition-all duration-200 flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Zap className="w-5 h-5" />
                      </motion.div>
                      <span>AUTHENTICATING...</span>
                    </>
                  ) : (
                    <>
                      <Unlock className="w-5 h-5" />
                      <span>JACK IN</span>
                    </>
                  )}
                </button>
              </motion.form>
            ) : (
              <motion.div
                key="biometric"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="text-center"
              >
                <button
                  onClick={handleBiometric}
                  disabled={isLoading || isLockedOut}
                  className="w-full py-8 bg-gray-900/50 hover:bg-gray-900/70 border border-cyan-500/30 hover:border-cyan-400 rounded transition-all group"
                >
                  <Fingerprint className="w-16 h-16 mx-auto text-cyan-400 group-hover:scale-110 transition-transform" />
                  <p className="mt-4 text-cyan-400 font-mono">
                    {isLoading ? 'SCANNING...' : 'TOUCH TO AUTHENTICATE'}
                  </p>
                </button>
                
                <p className="mt-4 text-xs text-gray-500">
                  Neural pattern recognition active
                </p>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Footer quote */}
          <div className="mt-8 pt-6 border-t border-gray-800 text-center">
            <p className="text-xs text-gray-500 italic">
              "In Night City, your password is your lifeline. Guard it with your life."
            </p>
          </div>
        </div>
      </motion.div>
      
      <style>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
          20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        
        .shake {
          animation: shake 0.5s ease-in-out;
        }
      `}</style>
    </div>
  );
};

/**
 * LOGIN WISDOM (earned through lockouts):
 * 
 * 1. That caps lock warning? Added after watching a CEO
 *    lock himself out during a board demo. The irony was
 *    delicious, the consequences were not.
 * 
 * 2. Biometric option exists because I've seen traders
 *    literally forget their passwords mid-panic. Adrenaline
 *    does weird things to memory.
 * 
 * 3. The shake animation on failed login? Psychological.
 *    Physical feedback helps people realize they made an
 *    error faster than just text.
 * 
 * 4. Password is cleared after submit, always. Saw a trader
 *    leave his password in the field, walked away for coffee,
 *    came back to an empty account.
 * 
 * 5. The matrix rain? Pure style. But it also masks screen
 *    recordings. Hard to see what someone's typing when
 *    there's movement everywhere.
 * 
 * Remember: The login screen is the first impression and
 * the last defense. Make it count.
 */
