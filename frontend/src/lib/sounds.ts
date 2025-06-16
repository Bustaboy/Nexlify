// frontend/src/lib/sounds.ts

import { Howl, Howler } from 'howler';
import { useSettingsStore } from '@stores/settingsStore';

// Sound library - the heartbeat of Night City's trading floors
// Each sound tells a story, each beep a memory of deals made and lost

interface SoundConfig {
  src: string[];
  volume?: number;
  loop?: boolean;
  preload?: boolean;
  html5?: boolean;
}

interface SoundLibrary {
  [key: string]: Howl;
}

class NexlifySoundSystem {
  private sounds: SoundLibrary = {};
  private initialized = false;
  private masterVolume = 0.7;
  private enabled = true;
  
  // Sound definitions - carefully crafted like a street musician's repertoire
  private readonly soundDefinitions: Record<string, SoundConfig> = {
    // System sounds - the daily symphony of a trader's life
    startup: {
      src: ['/sounds/neural_boot.mp3', '/sounds/neural_boot.ogg'],
      volume: 0.6,
      preload: true
    },
    shutdown: {
      src: ['/sounds/power_down.mp3'],
      volume: 0.5
    },
    click: {
      src: ['/sounds/ui_click.mp3'],
      volume: 0.3,
      preload: true
    },
    hover: {
      src: ['/sounds/ui_hover.mp3'],
      volume: 0.2
    },
    
    // Trading sounds - the sweet and bitter notes of profit and loss
    trade_execute: {
      src: ['/sounds/trade_execute.mp3'],
      volume: 0.7,
      preload: true
    },
    order_placed: {
      src: ['/sounds/order_placed.mp3'],
      volume: 0.5
    },
    order_cancelled: {
      src: ['/sounds/order_cancelled.mp3'],
      volume: 0.4
    },
    position_closed: {
      src: ['/sounds/position_closed.mp3'],
      volume: 0.6
    },
    
    // Alert sounds - warnings from the digital void
    notification: {
      src: ['/sounds/notification.mp3'],
      volume: 0.6,
      preload: true
    },
    alert_low: {
      src: ['/sounds/alert_low.mp3'],
      volume: 0.5
    },
    alert_medium: {
      src: ['/sounds/alert_medium.mp3'],
      volume: 0.7
    },
    alert_high: {
      src: ['/sounds/alert_critical.mp3'],
      volume: 0.9,
      preload: true
    },
    
    // Status sounds - the pulse of your digital existence
    success: {
      src: ['/sounds/success.mp3'],
      volume: 0.5,
      preload: true
    },
    error: {
      src: ['/sounds/error.mp3'],
      volume: 0.6,
      preload: true
    },
    warning: {
      src: ['/sounds/warning.mp3'],
      volume: 0.5
    },
    
    // Profit/Loss sounds - the emotional rollercoaster
    profit: {
      src: ['/sounds/profit_ding.mp3'],
      volume: 0.6
    },
    loss: {
      src: ['/sounds/loss_buzz.mp3'],
      volume: 0.4
    },
    jackpot: {
      src: ['/sounds/major_profit.mp3'],
      volume: 0.8
    },
    
    // Ambient sounds - the atmosphere of a cyberpunk trading floor
    ambient_low: {
      src: ['/sounds/ambient_hum.mp3'],
      volume: 0.1,
      loop: true,
      html5: true // Use HTML5 for long ambient tracks
    },
    ambient_data: {
      src: ['/sounds/data_stream.mp3'],
      volume: 0.15,
      loop: true,
      html5: true
    },
    
    // Achievement sounds - celebrating the small victories
    achievement: {
      src: ['/sounds/achievement_unlock.mp3'],
      volume: 0.7
    },
    level_up: {
      src: ['/sounds/level_up.mp3'],
      volume: 0.8
    },
    milestone: {
      src: ['/sounds/milestone_reached.mp3'],
      volume: 0.7
    }
  };

  constructor() {
    // Set up global Howler settings - respecting the user's audio space
    Howler.autoUnlock = true;
    Howler.html5PoolSize = 10;
    
    // Listen for system-wide audio events
    if (typeof window !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          this.muteAll(true);
        } else {
          this.muteAll(false);
        }
      });
    }
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    console.log('🔊 Initializing Nexlify sound system...');
    
    try {
      // Load user preferences
      const settings = useSettingsStore.getState();
      this.enabled = settings.soundEnabled;
      this.masterVolume = settings.soundVolume;
      
      // Pre-load critical sounds
      const criticalSounds = ['startup', 'click', 'trade_execute', 'notification', 'error', 'success', 'alert_high'];
      
      for (const soundKey of criticalSounds) {
        const config = this.soundDefinitions[soundKey];
        if (config && config.preload) {
          await this.loadSound(soundKey, config);
        }
      }
      
      this.initialized = true;
      console.log('✅ Sound system online - Night City vibes activated');
      
      // Play startup sound if enabled
      if (this.enabled) {
        this.play('startup');
      }
      
    } catch (error) {
      console.error('Failed to initialize sound system:', error);
      // Don't let audio issues crash the whole system
      this.initialized = true;
    }
  }

  private async loadSound(key: string, config: SoundConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      const sound = new Howl({
        ...config,
        volume: (config.volume || 1) * this.masterVolume,
        onload: () => {
          this.sounds[key] = sound;
          resolve();
        },
        onloaderror: (id, error) => {
          console.warn(`Failed to load sound ${key}:`, error);
          // Don't reject - audio shouldn't break the app
          resolve();
        }
      });
    });
  }

  // Play a sound - if it exists and we're allowed
  play(soundKey: string, options?: { volume?: number; rate?: number }): void {
    if (!this.enabled || !this.initialized) return;
    
    // Lazy load sounds that weren't preloaded
    if (!this.sounds[soundKey] && this.soundDefinitions[soundKey]) {
      const config = this.soundDefinitions[soundKey];
      this.loadSound(soundKey, config).then(() => {
        this.playSound(soundKey, options);
      });
      return;
    }
    
    this.playSound(soundKey, options);
  }

  private playSound(soundKey: string, options?: { volume?: number; rate?: number }): void {
    const sound = this.sounds[soundKey];
    if (!sound) return;
    
    try {
      // Set temporary volume if specified
      if (options?.volume !== undefined) {
        sound.volume(options.volume * this.masterVolume);
      }
      
      // Set playback rate if specified (for that Matrix-style slowdown effect)
      if (options?.rate !== undefined) {
        sound.rate(options.rate);
      }
      
      sound.play();
      
      // Reset to defaults after playing
      if (options) {
        sound.once('end', () => {
          const defaultConfig = this.soundDefinitions[soundKey];
          sound.volume((defaultConfig.volume || 1) * this.masterVolume);
          sound.rate(1);
        });
      }
      
    } catch (error) {
      console.warn(`Error playing sound ${soundKey}:`, error);
    }
  }

  // Stop a specific sound or all sounds
  stop(soundKey?: string): void {
    if (soundKey && this.sounds[soundKey]) {
      this.sounds[soundKey].stop();
    } else {
      // Stop all sounds - emergency brake
      Object.values(this.sounds).forEach(sound => sound.stop());
    }
  }

  // Fade out - for those smooth transitions
  fadeOut(soundKey: string, duration: number = 1000): void {
    const sound = this.sounds[soundKey];
    if (sound && sound.playing()) {
      sound.fade(sound.volume(), 0, duration);
      sound.once('fade', () => sound.stop());
    }
  }

  // Update master volume - when the neighbors complain
  setMasterVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume));
    Howler.volume(this.masterVolume);
  }

  // Toggle sound system - for those late-night stealth sessions
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.stop(); // Stop all sounds when disabled
    }
  }

  // Mute/unmute all sounds
  muteAll(muted: boolean): void {
    Howler.mute(muted);
  }

  // Play a sequence of sounds - for complex notifications
  async playSequence(sounds: Array<{ key: string; delay?: number; options?: any }>): Promise<void> {
    for (const { key, delay = 0, options } of sounds) {
      if (delay > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
      this.play(key, options);
    }
  }

  // Create a dynamic sound effect - for that extra polish
  playDynamic(type: 'coinCollect' | 'powerUp' | 'glitch'): void {
    switch (type) {
      case 'coinCollect':
        // Ascending pitch for collecting profits
        this.play('profit', { rate: 1.0 });
        setTimeout(() => this.play('profit', { rate: 1.2 }), 100);
        setTimeout(() => this.play('profit', { rate: 1.5 }), 200);
        break;
        
      case 'powerUp':
        // Rising tone for activating features
        this.playSequence([
          { key: 'click', options: { rate: 0.8 } },
          { key: 'success', delay: 200, options: { rate: 1.2 } }
        ]);
        break;
        
      case 'glitch':
        // Distorted sound for errors
        this.play('error', { rate: 0.7, volume: 0.8 });
        setTimeout(() => this.play('error', { rate: 1.3, volume: 0.4 }), 50);
        break;
    }
  }

  // Cleanup - for when the party's over
  destroy(): void {
    this.stop();
    Object.values(this.sounds).forEach(sound => sound.unload());
    this.sounds = {};
    this.initialized = false;
  }
}

// Create singleton instance - one sound system to rule them all
export const soundSystem = new NexlifySoundSystem();

// Convenience function for components
export const playSound = (
  soundKey: string, 
  options?: { volume?: number; rate?: number }
): void => {
  soundSystem.play(soundKey, options);
};

// Export for special effects
export const playSoundSequence = soundSystem.playSequence.bind(soundSystem);
export const playDynamicSound = soundSystem.playDynamic.bind(soundSystem);

// Auto-initialize on first import
if (typeof window !== 'undefined') {
  soundSystem.initialize().catch(console.error);
}
