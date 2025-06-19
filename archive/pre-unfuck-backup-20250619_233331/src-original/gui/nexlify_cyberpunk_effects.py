"""
Nexlify Enhanced - Cyberpunk Immersion Effects
Implements Feature 27: Sound effects, holographic UI, animations, and themed elements
"""

import tkinter as tk
from tkinter import ttk, Canvas
import pygame
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import asyncio
import random
import math
from datetime import datetime
import threading
from pathlib import Path
import json

# Initialize pygame for sound
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

class SoundEffectsManager:
    """Manages all cyberpunk sound effects"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.sounds = {}
        self.volume = 0.7
        
        # Sound categories
        self.sound_map = {
            # UI Sounds
            'startup': 'neural_boot.wav',
            'shutdown': 'system_powerdown.wav',
            'click': 'cyber_click.wav',
            'hover': 'soft_beep.wav',
            'tab_switch': 'interface_switch.wav',
            
            # Trading Sounds
            'trade_execute': 'trade_confirm.wav',
            'profit': 'credits_earned.wav',
            'loss': 'warning_beep.wav',
            'order_placed': 'order_sent.wav',
            'order_filled': 'order_complete.wav',
            
            # Alert Sounds
            'alert_low': 'soft_alert.wav',
            'alert_medium': 'warning_alert.wav',
            'alert_high': 'critical_alert.wav',
            'achievement': 'achievement_unlock.wav',
            
            # Ambient
            'ambient_loop': 'cyberpunk_ambient.wav',
            'data_flow': 'data_stream.wav',
            'neural_process': 'ai_thinking.wav'
        }
        
        # Generate synthetic sounds if files don't exist
        self.generate_synthetic_sounds()
        
    def generate_synthetic_sounds(self):
        """Generate cyberpunk sounds programmatically"""
        # Startup sound - ascending digital chirps
        self.sounds['startup'] = self.create_startup_sound()
        
        # Click sound - short digital beep
        self.sounds['click'] = self.create_click_sound()
        
        # Profit sound - ascending tones
        self.sounds['profit'] = self.create_profit_sound()
        
        # Loss sound - descending tones
        self.sounds['loss'] = self.create_loss_sound()
        
        # Alert sounds
        self.sounds['alert_low'] = self.create_alert_sound(frequency=440, duration=0.2)
        self.sounds['alert_medium'] = self.create_alert_sound(frequency=880, duration=0.3)
        self.sounds['alert_high'] = self.create_alert_sound(frequency=1320, duration=0.5)
        
        # Achievement sound - victory fanfare
        self.sounds['achievement'] = self.create_achievement_sound()
        
    def create_startup_sound(self) -> pygame.mixer.Sound:
        """Create synthetic startup sound"""
        sample_rate = 22050
        duration = 1.0
        
        # Generate ascending chirps
        samples = []
        frequencies = [220, 330, 440, 550, 660, 880]
        
        for i, freq in enumerate(frequencies):
            t = np.linspace(0, 0.15, int(sample_rate * 0.15))
            
            # Envelope
            envelope = np.exp(-t * 10) * (1 - t / 0.15)
            
            # Waveform with harmonics
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.3 * np.sin(4 * np.pi * freq * t)
            wave += 0.1 * np.sin(6 * np.pi * freq * t)
            
            # Apply envelope and add delay
            sound_chunk = wave * envelope * 0.5
            
            # Add silence between chirps
            silence = np.zeros(int(sample_rate * 0.05))
            
            samples.extend(sound_chunk)
            samples.extend(silence)
            
        # Convert to pygame sound
        samples = np.array(samples)
        samples = (samples * 32767).astype(np.int16)
        
        # Stereo
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def create_click_sound(self) -> pygame.mixer.Sound:
        """Create synthetic click sound"""
        sample_rate = 22050
        duration = 0.05
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Square wave with envelope
        frequency = 1000
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
        envelope = np.exp(-t * 100)
        
        samples = wave * envelope * 0.3
        samples = (samples * 32767).astype(np.int16)
        
        # Stereo
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def create_profit_sound(self) -> pygame.mixer.Sound:
        """Create ascending profit sound"""
        sample_rate = 22050
        
        # Ascending arpeggio
        notes = [523, 659, 784, 1047]  # C, E, G, C
        samples = []
        
        for freq in notes:
            t = np.linspace(0, 0.1, int(sample_rate * 0.1))
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.ones_like(t) * 0.5
            samples.extend(wave * envelope)
            
        samples = np.array(samples)
        samples = (samples * 32767).astype(np.int16)
        
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def create_loss_sound(self) -> pygame.mixer.Sound:
        """Create descending loss sound"""
        sample_rate = 22050
        
        # Descending tones
        notes = [784, 659, 523, 392]  # G, E, C, G
        samples = []
        
        for freq in notes:
            t = np.linspace(0, 0.1, int(sample_rate * 0.1))
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 20)
            samples.extend(wave * envelope * 0.4)
            
        samples = np.array(samples)
        samples = (samples * 32767).astype(np.int16)
        
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def create_alert_sound(self, frequency: float, duration: float) -> pygame.mixer.Sound:
        """Create alert beep sound"""
        sample_rate = 22050
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Modulated sine wave
        carrier = np.sin(2 * np.pi * frequency * t)
        modulator = np.sin(2 * np.pi * 10 * t)
        
        wave = carrier * (1 + 0.3 * modulator)
        envelope = np.ones_like(t) * 0.6
        
        samples = wave * envelope
        samples = (samples * 32767).astype(np.int16)
        
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def create_achievement_sound(self) -> pygame.mixer.Sound:
        """Create achievement unlock sound"""
        sample_rate = 22050
        
        # Victory chord progression
        chord_freqs = [
            [523, 659, 784],    # C major
            [587, 740, 880],    # D major
            [659, 831, 988],    # E major
            [784, 988, 1175]    # G major
        ]
        
        samples = []
        
        for chord in chord_freqs:
            t = np.linspace(0, 0.2, int(sample_rate * 0.2))
            wave = np.zeros_like(t)
            
            for freq in chord:
                wave += np.sin(2 * np.pi * freq * t) * 0.3
                
            envelope = np.ones_like(t)
            samples.extend(wave * envelope)
            
        samples = np.array(samples)
        samples = (samples * 32767).astype(np.int16)
        
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        return pygame.sndarray.make_sound(stereo_samples)
        
    def play(self, sound_name: str, volume_override: Optional[float] = None):
        """Play a sound effect"""
        if not self.enabled:
            return
            
        if sound_name in self.sounds:
            sound = self.sounds[sound_name]
            sound.set_volume(volume_override or self.volume)
            sound.play()
            
    def set_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        
    def toggle(self):
        """Toggle sound effects on/off"""
        self.enabled = not self.enabled

class CyberpunkAnimator:
    """Handles cyberpunk UI animations"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.animations = []
        self.running = True
        
        # Start animation loop
        self.animate()
        
    def animate(self):
        """Main animation loop"""
        if not self.running:
            return
            
        # Update all active animations
        for animation in self.animations[:]:
            if not animation.update():
                self.animations.remove(animation)
                
        # Schedule next frame
        self.root.after(16, self.animate)  # ~60 FPS
        
    def add_animation(self, animation: 'Animation'):
        """Add new animation to the queue"""
        self.animations.append(animation)
        
    def create_glitch_effect(self, widget: tk.Widget, duration: float = 0.5):
        """Create glitch effect on widget"""
        animation = GlitchAnimation(widget, duration)
        self.add_animation(animation)
        
    def create_pulse_effect(self, widget: tk.Widget, color: str = '#00ff00'):
        """Create pulsing glow effect"""
        animation = PulseAnimation(widget, color)
        self.add_animation(animation)
        
    def create_matrix_rain(self, canvas: Canvas):
        """Create matrix-style falling characters"""
        animation = MatrixRainAnimation(canvas)
        self.add_animation(animation)
        
    def create_scan_line(self, widget: tk.Widget):
        """Create scanning line effect"""
        animation = ScanLineAnimation(widget)
        self.add_animation(animation)
        
    def stop(self):
        """Stop all animations"""
        self.running = False
        self.animations.clear()

class Animation:
    """Base animation class"""
    
    def __init__(self, duration: float = 1.0):
        self.duration = duration
        self.start_time = datetime.now()
        self.elapsed = 0
        
    def update(self) -> bool:
        """Update animation. Return False when complete."""
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.elapsed < self.duration

class GlitchAnimation(Animation):
    """Glitch effect animation"""
    
    def __init__(self, widget: tk.Widget, duration: float = 0.5):
        super().__init__(duration)
        self.widget = widget
        self.original_pos = (widget.winfo_x(), widget.winfo_y())
        
    def update(self) -> bool:
        if not super().update():
            # Reset position
            self.widget.place(x=self.original_pos[0], y=self.original_pos[1])
            return False
            
        # Random offset
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        
        self.widget.place(
            x=self.original_pos[0] + offset_x,
            y=self.original_pos[1] + offset_y
        )
        
        return True

class PulseAnimation(Animation):
    """Pulsing glow effect"""
    
    def __init__(self, widget: tk.Widget, color: str = '#00ff00'):
        super().__init__(float('inf'))  # Continuous
        self.widget = widget
        self.base_color = color
        self.original_bg = widget.cget('background')
        
    def update(self) -> bool:
        super().update()
        
        # Calculate pulse intensity
        intensity = (math.sin(self.elapsed * 4) + 1) / 2
        
        # Blend colors
        r1, g1, b1 = self.hex_to_rgb(self.original_bg)
        r2, g2, b2 = self.hex_to_rgb(self.base_color)
        
        r = int(r1 + (r2 - r1) * intensity * 0.3)
        g = int(g1 + (g2 - g1) * intensity * 0.3)
        b = int(b1 + (b2 - b1) * intensity * 0.3)
        
        new_color = f'#{r:02x}{g:02x}{b:02x}'
        
        try:
            self.widget.configure(background=new_color)
        except:
            return False
            
        return True
        
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class MatrixRainAnimation(Animation):
    """Matrix-style falling characters"""
    
    def __init__(self, canvas: Canvas):
        super().__init__(float('inf'))
        self.canvas = canvas
        self.width = canvas.winfo_width()
        self.height = canvas.winfo_height()
        
        # Initialize columns
        self.columns = []
        self.column_count = max(1, self.width // 20)
        
        for i in range(self.column_count):
            self.columns.append({
                'x': i * 20 + 10,
                'y': random.randint(-self.height, 0),
                'speed': random.uniform(2, 6),
                'chars': self.generate_column_chars()
            })
            
        self.text_items = []
        
    def generate_column_chars(self) -> List[str]:
        """Generate random matrix characters"""
        chars = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        return [random.choice(chars) for _ in range(20)]
        
    def update(self) -> bool:
        super().update()
        
        # Clear previous text
        for item in self.text_items:
            self.canvas.delete(item)
        self.text_items.clear()
        
        # Update and draw columns
        for col in self.columns:
            col['y'] += col['speed']
            
            # Reset column when it goes off screen
            if col['y'] > self.height + 200:
                col['y'] = -200
                col['chars'] = self.generate_column_chars()
                col['speed'] = random.uniform(2, 6)
                
            # Draw characters
            for i, char in enumerate(col['chars']):
                y_pos = col['y'] + i * 20
                
                if 0 <= y_pos <= self.height:
                    # Fade based on position
                    intensity = max(0, 1 - (i / len(col['chars'])))
                    green_value = int(255 * intensity)
                    color = f'#{0:02x}{green_value:02x}{0:02x}'
                    
                    text_item = self.canvas.create_text(
                        col['x'], y_pos,
                        text=char,
                        fill=color,
                        font=('Consolas', 12)
                    )
                    self.text_items.append(text_item)
                    
        return True

class HolographicFrame(ttk.Frame):
    """Frame with holographic cyberpunk styling"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(
            style='Holographic.TFrame',
            borderwidth=2,
            relief='ridge'
        )
        
        # Create glow effect
        self.create_glow_border()
        
    def create_glow_border(self):
        """Create glowing border effect"""
        # This would be implemented with custom canvas drawing
        pass

class CyberpunkButton(tk.Button):
    """Cyberpunk-styled button with effects"""
    
    def __init__(self, parent, sound_manager: SoundEffectsManager = None, **kwargs):
        # Default cyberpunk styling
        kwargs.setdefault('bg', '#0a0a0a')
        kwargs.setdefault('fg', '#00ff00')
        kwargs.setdefault('activebackground', '#1a1a1a')
        kwargs.setdefault('activeforeground', '#00ffff')
        kwargs.setdefault('font', ('Consolas', 10, 'bold'))
        kwargs.setdefault('bd', 2)
        kwargs.setdefault('relief', 'ridge')
        
        super().__init__(parent, **kwargs)
        
        self.sound_manager = sound_manager
        
        # Bind hover effects
        self.bind('<Enter>', self.on_hover)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def on_hover(self, event):
        """Mouse hover effect"""
        self.configure(fg='#00ffff')
        if self.sound_manager:
            self.sound_manager.play('hover', volume_override=0.3)
            
    def on_leave(self, event):
        """Mouse leave effect"""
        self.configure(fg='#00ff00')
        
    def on_click(self, event):
        """Click effect"""
        if self.sound_manager:
            self.sound_manager.play('click')

class TerminalText(scrolledtext.ScrolledText):
    """Terminal-style text widget with typing effects"""
    
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', '#0a0a0a')
        kwargs.setdefault('fg', '#00ff00')
        kwargs.setdefault('insertbackground', '#00ff00')
        kwargs.setdefault('font', ('Consolas', 10))
        kwargs.setdefault('wrap', tk.WORD)
        
        super().__init__(parent, **kwargs)
        
        # Configure tags
        self.tag_config('system', foreground='#00ffff')
        self.tag_config('error', foreground='#ff0000')
        self.tag_config('warning', foreground='#ffaa00')
        self.tag_config('success', foreground='#00ff00')
        self.tag_config('data', foreground='#ffff00')
        
        self.typing_queue = []
        self.typing_active = False
        
    def type_text(self, text: str, tag: str = None, delay: int = 20):
        """Add text with typing animation"""
        self.typing_queue.append((text, tag, delay))
        
        if not self.typing_active:
            self.process_typing_queue()
            
    def process_typing_queue(self):
        """Process queued typing animations"""
        if not self.typing_queue:
            self.typing_active = False
            return
            
        self.typing_active = True
        text, tag, delay = self.typing_queue.pop(0)
        
        self.config(state=tk.NORMAL)
        
        def type_char(index=0):
            if index < len(text):
                self.insert(tk.END, text[index], tag)
                self.see(tk.END)
                self.after(delay, lambda: type_char(index + 1))
            else:
                self.config(state=tk.DISABLED)
                self.after(100, self.process_typing_queue)
                
        type_char()

class NeuralNetworkVisualizer(Canvas):
    """Animated neural network visualization"""
    
    def __init__(self, parent, width: int = 400, height: int = 300, **kwargs):
        kwargs.setdefault('bg', '#0a0a0a')
        kwargs.setdefault('highlightthickness', 0)
        
        super().__init__(parent, width=width, height=height, **kwargs)
        
        self.nodes = []
        self.connections = []
        self.activity_level = 0.5
        
        self.setup_network()
        self.animate()
        
    def setup_network(self):
        """Create neural network structure"""
        layers = [4, 6, 6, 2]  # Input, hidden, hidden, output
        
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        
        # Create nodes
        for layer_idx, layer_size in enumerate(layers):
            layer_nodes = []
            x = (layer_idx + 1) * width / (len(layers) + 1)
            
            for node_idx in range(layer_size):
                y = (node_idx + 1) * height / (layer_size + 1)
                
                node = {
                    'x': x,
                    'y': y,
                    'activation': random.random(),
                    'circle': None
                }
                
                # Draw node
                node['circle'] = self.create_oval(
                    x - 10, y - 10, x + 10, y + 10,
                    fill='#001100',
                    outline='#00ff00',
                    width=2
                )
                
                layer_nodes.append(node)
                
            self.nodes.append(layer_nodes)
            
        # Create connections
        for i in range(len(layers) - 1):
            for node1 in self.nodes[i]:
                for node2 in self.nodes[i + 1]:
                    weight = random.random()
                    
                    line = self.create_line(
                        node1['x'], node1['y'],
                        node2['x'], node2['y'],
                        fill='#003300',
                        width=1
                    )
                    
                    self.connections.append({
                        'line': line,
                        'weight': weight,
                        'node1': node1,
                        'node2': node2
                    })
                    
    def animate(self):
        """Animate neural network activity"""
        # Update node activations
        for layer in self.nodes:
            for node in layer:
                # Random walk activation
                node['activation'] += random.uniform(-0.1, 0.1)
                node['activation'] = max(0, min(1, node['activation']))
                
                # Update node color
                intensity = int(255 * node['activation'])
                color = f'#{0:02x}{intensity:02x}{0:02x}'
                self.itemconfig(node['circle'], fill=color)
                
        # Update connection colors based on activity
        for conn in self.connections:
            # Calculate connection activity
            activity = conn['node1']['activation'] * conn['node2']['activation'] * conn['weight']
            intensity = int(100 * activity)
            color = f'#{0:02x}{intensity:02x}{0:02x}'
            self.itemconfig(conn['line'], fill=color)
            
        # Continue animation
        self.after(50, self.animate)
        
    def set_activity_level(self, level: float):
        """Set overall network activity level"""
        self.activity_level = max(0, min(1, level))

def create_cyberpunk_theme():
    """Create and configure cyberpunk ttk theme"""
    style = ttk.Style()
    
    # Configure colors
    colors = {
        'bg': '#0a0a0a',
        'fg': '#00ff00',
        'select_bg': '#1a1a1a',
        'select_fg': '#00ffff',
        'active': '#00ffff',
        'border': '#00ff00'
    }
    
    # Configure Frame
    style.configure('Cyberpunk.TFrame',
                   background=colors['bg'],
                   bordercolor=colors['border'],
                   relief='ridge')
                   
    # Configure Label
    style.configure('Cyberpunk.TLabel',
                   background=colors['bg'],
                   foreground=colors['fg'])
                   
    # Configure Button
    style.configure('Cyberpunk.TButton',
                   background=colors['bg'],
                   foreground=colors['fg'],
                   bordercolor=colors['border'],
                   focuscolor='none')
                   
    style.map('Cyberpunk.TButton',
             background=[('active', colors['select_bg'])],
             foreground=[('active', colors['active'])])
             
    # Configure Entry
    style.configure('Cyberpunk.TEntry',
                   fieldbackground=colors['bg'],
                   background=colors['bg'],
                   foreground=colors['fg'],
                   insertcolor=colors['fg'],
                   bordercolor=colors['border'])
                   
    # Configure Combobox
    style.configure('Cyberpunk.TCombobox',
                   fieldbackground=colors['bg'],
                   background=colors['bg'],
                   foreground=colors['fg'],
                   arrowcolor=colors['fg'],
                   bordercolor=colors['border'])
                   
    # Configure Notebook
    style.configure('Cyberpunk.TNotebook',
                   background=colors['bg'],
                   bordercolor=colors['border'])
                   
    style.configure('Cyberpunk.TNotebook.Tab',
                   background=colors['bg'],
                   foreground=colors['fg'])
                   
    style.map('Cyberpunk.TNotebook.Tab',
             background=[('selected', colors['select_bg'])],
             foreground=[('selected', colors['active'])])
             
    return style
