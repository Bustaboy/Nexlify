// Location: nexlify-dashboard/src/systems/adaptive/renderers/AudioEngine.ts
// Mission: 80-I.1 Adaptive Audio System
// Dependencies: None
// Context: Manages all audio effects with performance awareness

export interface AudioEffect {
  type: 'typewriter' | 'glitch' | 'scan' | 'boot' | 'alert' | 'success' | 'cascade';
  frequency: number;
  duration: number;
  waveform: OscillatorType;
  envelope: {
    attack: number;
    decay: number;
    sustain: number;
    release: number;
  };
  modulation?: {
    frequency: number;
    depth: number;
  };
}

export class AudioEngine {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private enabled = true;
  private volume = 0.5;
  
  // Effect presets
  private effects: Map<string, AudioEffect> = new Map();
  
  // Active sounds tracking
  private activeSounds: Set<{
    oscillator: OscillatorNode;
    gainNode: GainNode;
    stopTime: number;
  }> = new Set();
  
  constructor() {
    this.initializeEffects();
  }
  
  async initialize(): Promise<void> {
    console.log('[AUDIO] Initializing audio engine...');
    
    try {
      // Create audio context
      this.context = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Create master gain
      this.masterGain = this.context.createGain();
      this.masterGain.gain.value = this.volume;
      
      // Create compressor for consistent volume
      this.compressor = this.context.createDynamicsCompressor();
      this.compressor.threshold.value = -24;
      this.compressor.knee.value = 30;
      this.compressor.ratio.value = 12;
      this.compressor.attack.value = 0.003;
      this.compressor.release.value = 0.25;
      
      // Connect nodes
      this.compressor.connect(this.masterGain);
      this.masterGain.connect(this.context.destination);
      
      // Handle context suspension
      if (this.context.state === 'suspended') {
        console.log('[AUDIO] Context suspended, waiting for user interaction...');
        
        const resumeContext = async () => {
          if (this.context?.state === 'suspended') {
            await this.context.resume();
            console.log('[AUDIO] Context resumed');
          }
          document.removeEventListener('click', resumeContext);
          document.removeEventListener('keydown', resumeContext);
        };
        
        document.addEventListener('click', resumeContext, { once: true });
        document.addEventListener('keydown', resumeContext, { once: true });
      }
      
      console.log('[AUDIO] Audio engine initialized');
      
    } catch (err) {
      console.error('[AUDIO] Initialization failed:', err);
      this.enabled = false;
    }
  }
  
  private initializeEffects(): void {
    // Cyberpunk-themed audio effects
    
    this.effects.set('typewriter', {
      type: 'typewriter',
      frequency: 440,
      duration: 30,
      waveform: 'square',
      envelope: {
        attack: 0.001,
        decay: 0.01,
        sustain: 0.1,
        release: 0.02
      }
    });
    
    this.effects.set('glitch', {
      type: 'glitch',
      frequency: 110,
      duration: 50,
      waveform: 'sawtooth',
      envelope: {
        attack: 0.001,
        decay: 0.02,
        sustain: 0.3,
        release: 0.03
      },
      modulation: {
        frequency: 20,
        depth: 50
      }
    });
    
    this.effects.set('scan', {
      type: 'scan',
      frequency: 60,
      duration: 200,
      waveform: 'sine',
      envelope: {
        attack: 0.1,
        decay: 0.05,
        sustain: 0.2,
        release: 0.1
      }
    });
    
    this.effects.set('boot', {
      type: 'boot',
      frequency: 220,
      duration: 500,
      waveform: 'triangle',
      envelope: {
        attack: 0.2,
        decay: 0.1,
        sustain: 0.3,
        release: 0.2
      },
      modulation: {
        frequency: 5,
        depth: 10
      }
    });
    
    this.effects.set('alert', {
      type: 'alert',
      frequency: 880,
      duration: 300,
      waveform: 'square',
      envelope: {
        attack: 0.01,
        decay: 0.05,
        sustain: 0.5,
        release: 0.1
      },
      modulation: {
        frequency: 10,
        depth: 20
      }
    });
    
    this.effects.set('success', {
      type: 'success',
      frequency: 523.25, // C5
      duration: 200,
      waveform: 'sine',
      envelope: {
        attack: 0.01,
        decay: 0.1,
        sustain: 0.3,
        release: 0.1
      }
    });
    
    this.effects.set('cascade', {
      type: 'cascade',
      frequency: 150,
      duration: 1000,
      waveform: 'sawtooth',
      envelope: {
        attack: 0.5,
        decay: 0.2,
        sustain: 0.4,
        release: 0.3
      },
      modulation: {
        frequency: 2,
        depth: 100
      }
    });
  }
  
  playEffect(effectName: string, options?: {
    frequency?: number;
    duration?: number;
    volume?: number;
    pan?: number;
  }): void {
    if (!this.enabled || !this.context || !this.compressor || this.context.state === 'suspended') {
      return;
    }
    
    const effect = this.effects.get(effectName);
    if (!effect) {
      console.warn(`[AUDIO] Unknown effect: ${effectName}`);
      return;
    }
    
    try {
      const now = this.context.currentTime;
      
      // Create oscillator
      const oscillator = this.context.createOscillator();
      oscillator.type = effect.waveform;
      oscillator.frequency.value = options?.frequency || effect.frequency;
      
      // Create gain for envelope
      const gainNode = this.context.createGain();
      gainNode.gain.value = 0;
      
      // Create panner for spatial audio
      const panner = this.context.createStereoPanner();
      panner.pan.value = options?.pan || 0;
      
      // Apply modulation if specified
      if (effect.modulation) {
        const lfo = this.context.createOscillator();
        lfo.frequency.value = effect.modulation.frequency;
        
        const lfoGain = this.context.createGain();
        lfoGain.gain.value = effect.modulation.depth;
        
        lfo.connect(lfoGain);
        lfoGain.connect(oscillator.frequency);
        lfo.start(now);
        lfo.stop(now + (options?.duration || effect.duration) / 1000);
      }
      
      // Connect nodes
      oscillator.connect(gainNode);
      gainNode.connect(panner);
      panner.connect(this.compressor);
      
      // Apply envelope
      const volume = (options?.volume || 1) * 0.3; // Keep it subtle
      const duration = (options?.duration || effect.duration) / 1000;
      
      // Attack
      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(
        volume,
        now + effect.envelope.attack
      );
      
      // Decay to sustain
      gainNode.gain.linearRampToValueAtTime(
        volume * effect.envelope.sustain,
        now + effect.envelope.attack + effect.envelope.decay
      );
      
      // Release
      const releaseTime = now + duration - effect.envelope.release;
      gainNode.gain.setValueAtTime(
        volume * effect.envelope.sustain,
        releaseTime
      );
      gainNode.gain.linearRampToValueAtTime(0, now + duration);
      
      // Start and stop
      oscillator.start(now);
      oscillator.stop(now + duration);
      
      // Track active sound
      const sound = {
        oscillator,
        gainNode,
        stopTime: now + duration
      };
      
      this.activeSounds.add(sound);
      
      // Cleanup after sound finishes
      oscillator.onended = () => {
        this.activeSounds.delete(sound);
        oscillator.disconnect();
        gainNode.disconnect();
        panner.disconnect();
      };
      
    } catch (err) {
      console.error('[AUDIO] Error playing effect:', err);
    }
  }
  
  // Play multiple effects in sequence
  playSequence(effects: Array<{
    effect: string;
    delay: number;
    options?: Parameters<typeof this.playEffect>[1];
  }>): void {
    if (!this.enabled || !this.context) return;
    
    let currentDelay = 0;
    
    for (const { effect, delay, options } of effects) {
      setTimeout(() => {
        this.playEffect(effect, options);
      }, currentDelay);
      
      currentDelay += delay;
    }
  }
  
  // Play chord for more complex sounds
  playChord(
    frequencies: number[],
    duration: number,
    waveform: OscillatorType = 'sine'
  ): void {
    if (!this.enabled || !this.context || !this.compressor) return;
    
    frequencies.forEach((freq, index) => {
      this.playEffect('typewriter', {
        frequency: freq,
        duration: duration,
        volume: 0.3 / frequencies.length,
        pan: (index - frequencies.length / 2) * 0.2 // Spread across stereo
      });
    });
  }
  
  // Generate procedural cascade warning sound
  playCascadeWarning(severity: 'low' | 'medium' | 'high' | 'critical'): void {
    if (!this.enabled) return;
    
    const severityMap = {
      low: { baseFreq: 200, modFreq: 2, repeats: 2 },
      medium: { baseFreq: 300, modFreq: 4, repeats: 3 },
      high: { baseFreq: 400, modFreq: 8, repeats: 4 },
      critical: { baseFreq: 500, modFreq: 16, repeats: 6 }
    };
    
    const config = severityMap[severity];
    
    // Play escalating warning beeps
    for (let i = 0; i < config.repeats; i++) {
      setTimeout(() => {
        this.playEffect('alert', {
          frequency: config.baseFreq * (1 + i * 0.1),
          duration: 150,
          volume: 0.5 + i * 0.1
        });
      }, i * 200);
    }
    
    // Add underlying cascade sound for high/critical
    if (severity === 'high' || severity === 'critical') {
      this.playEffect('cascade', {
        frequency: config.baseFreq / 2,
        duration: config.repeats * 200,
        volume: 0.3
      });
    }
  }
  
  // UI feedback sounds
  playUIFeedback(action: 'hover' | 'click' | 'success' | 'error'): void {
    const feedbackMap = {
      hover: { effect: 'typewriter', options: { frequency: 800, duration: 20, volume: 0.2 } },
      click: { effect: 'typewriter', options: { frequency: 600, duration: 40, volume: 0.4 } },
      success: { effect: 'success', options: { volume: 0.5 } },
      error: { effect: 'alert', options: { frequency: 300, duration: 200, volume: 0.4 } }
    };
    
    const feedback = feedbackMap[action];
    if (feedback) {
      this.playEffect(feedback.effect, feedback.options);
    }
  }
  
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    
    if (!enabled) {
      // Stop all active sounds
      this.stopAll();
    }
  }
  
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume));
    
    if (this.masterGain) {
      this.masterGain.gain.setValueAtTime(
        this.volume,
        this.context?.currentTime || 0
      );
    }
  }
  
  stopAll(): void {
    if (!this.context) return;
    
    const now = this.context.currentTime;
    
    // Stop all active sounds
    for (const sound of this.activeSounds) {
      try {
        sound.gainNode.gain.cancelScheduledValues(now);
        sound.gainNode.gain.setValueAtTime(sound.gainNode.gain.value, now);
        sound.gainNode.gain.linearRampToValueAtTime(0, now + 0.1);
        sound.oscillator.stop(now + 0.1);
      } catch (err) {
        // Sound might have already stopped
      }
    }
    
    this.activeSounds.clear();
  }
  
  dispose(): void {
    console.log('[AUDIO] Disposing audio engine...');
    
    this.stopAll();
    
    if (this.context) {
      this.context.close();
    }
    
    this.context = null;
    this.masterGain = null;
    this.compressor = null;
    this.effects.clear();
  }
  
  // Get audio context state
  getState(): 'suspended' | 'running' | 'closed' | 'not-initialized' {
    if (!this.context) return 'not-initialized';
    return this.context.state;
  }
  
  // Resume context if suspended
  async resume(): Promise<void> {
    if (this.context && this.context.state === 'suspended') {
      await this.context.resume();
    }
  }
}