#!/usr/bin/env python3
"""
Nexlify Modern UI Effects Module
Visual and audio effects for the modern interface
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CyberpunkEffects:
    """
    Manages visual effects for the modern interface
    Note: Class name retained for backward compatibility
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.effects_enabled = self.config.get("effects_enabled", True)
        self.glow_intensity = self.config.get("glow_intensity", 20)
        self.animation_speed = self.config.get("animation_speed", 200)

        logger.info("✨ Modern UI Effects initialized")

    def apply_glow_effect(self, widget, color: str = "#2563eb"):
        """
        Apply subtle shadow effect to a widget (modern design)

        Args:
            widget: PyQt5 widget to apply effect to
            color: Hex color for the shadow (optional, uses subtle gray)
        """
        # Skip glow effects for modern design (glow_intensity should be 0)
        if not self.effects_enabled or self.glow_intensity == 0:
            return

        try:
            # Create subtle drop shadow using Qt graphics effects
            from PyQt5.QtGui import QColor
            from PyQt5.QtWidgets import QGraphicsDropShadowEffect

            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)  # Subtle blur
            shadow.setColor(QColor(0, 0, 0, 25))  # Very subtle black shadow
            shadow.setOffset(0, 2)  # Slight offset downward
            widget.setGraphicsEffect(shadow)

        except Exception as e:
            logger.debug(f"Could not apply shadow effect: {e}")

    def create_scan_line_effect(self) -> str:
        """
        Scan line effect disabled for modern design

        Returns:
            Empty string (no scan lines for modern UI)
        """
        # Scan lines disabled for modern, clean design
        return ""

    def create_pulse_animation(self, widget, color: str = "#2563eb"):
        """
        Create pulsing animation effect

        Args:
            widget: Widget to animate
            color: Color for pulse effect
        """
        if not self.effects_enabled:
            return

        try:
            from PyQt5.QtCore import (QAbstractAnimation, QEasingCurve,
                                      QPropertyAnimation)
            from PyQt5.QtWidgets import QGraphicsOpacityEffect

            # Create opacity effect
            opacity_effect = QGraphicsOpacityEffect()
            widget.setGraphicsEffect(opacity_effect)

            # Create animation
            animation = QPropertyAnimation(opacity_effect, b"opacity")
            animation.setDuration(self.animation_speed * 5)  # 1 second for full cycle
            animation.setStartValue(1.0)
            animation.setKeyValueAt(0.5, 0.3)  # Fade to 30% at midpoint
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.InOutQuad)
            animation.setLoopCount(-1)  # Infinite loop

            # Start animation
            animation.start(QAbstractAnimation.DeleteWhenStopped)

            # Store reference so it doesn't get garbage collected
            if not hasattr(widget, "_animations"):
                widget._animations = []
            widget._animations.append(animation)

        except Exception as e:
            logger.debug(f"Could not create pulse animation: {e}")

    def get_cyberpunk_palette(self) -> Dict[str, str]:
        """
        Get the cyberpunk color palette

        Returns:
            Dictionary of color names to hex values
        """
        return {
            "bg_primary": "#0a0a0a",
            "bg_secondary": "#151515",
            "bg_tertiary": "#1f1f1f",
            "accent_cyan": "#00ffff",
            "accent_magenta": "#ff00ff",
            "accent_green": "#00ff00",
            "accent_yellow": "#ffff00",
            "accent_red": "#ff0000",
            "text_primary": "#ffffff",
            "text_secondary": "#b0b0b0",
            "text_dim": "#606060",
        }

    def play_typing_sound(self):
        """Play keyboard typing sound effect"""
        # This would typically play a sound file
        # For now, it's a placeholder that can be connected later
        pass  # Sound files not included in distribution

    def play_notification_sound(self, notification_type: str = "info"):
        """Play notification sound"""
        # Map notification types to sound effects
        sound_map = {
            "info": "notification",
            "success": "profit",
            "warning": "notification",
            "error": "emergency_alarm",
        }

        sound_name = sound_map.get(notification_type, "notification")

        # This would play the sound through SoundManager
        # For now, it's a placeholder
        pass  # Sound files not included in distribution


class SoundManager:
    """
    Manages audio effects and notifications
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sounds_enabled = self.config.get("sounds_enabled", True)
        self.volume = self.config.get("volume", 0.7)
        self.sounds_dir = Path("assets/sounds")
        self.initialized = False

        # Sound file paths
        self.sound_files = {
            "trading_start": "trading_start.wav",
            "trade_executed": "trade_executed.wav",
            "profit": "profit.wav",
            "loss": "loss.wav",
            "notification": "notification.wav",
            "emergency_alarm": "emergency_alarm.wav",
            "typing": "typing.wav",
            "click": "click.wav",
        }

        logger.info("🔊 Sound Manager initialized")

    def initialize(self):
        """Initialize audio system"""
        if not self.sounds_enabled:
            logger.info("Sounds disabled in config")
            return

        try:
            # Try to import audio library
            # Using a simple approach - could use pygame, pyaudio, or playsound
            self.initialized = True
            logger.info("Audio system ready")

        except ImportError as e:
            logger.warning(f"Audio library not available: {e}")
            self.initialized = False

    def play(self, sound_name: str):
        """
        Play a sound effect

        Args:
            sound_name: Name of the sound to play
        """
        if not self.sounds_enabled or not self.initialized:
            return

        if sound_name not in self.sound_files:
            logger.warning(f"Unknown sound: {sound_name}")
            return

        sound_file = self.sounds_dir / self.sound_files[sound_name]

        # Check if file exists
        if not sound_file.exists():
            logger.debug(f"Sound file not found: {sound_file}")
            return

        try:
            # Simple sound playback using playsound (if available)
            # In production, would use a more robust audio library
            import threading

            def _play_sound():
                try:
                    # Attempt to use playsound
                    from playsound import playsound

                    playsound(str(sound_file))
                except ImportError:
                    # Fallback: use system beep or no sound
                    logger.debug("playsound not available")

            # Play in background thread to not block UI
            thread = threading.Thread(target=_play_sound, daemon=True)
            thread.start()

        except Exception as e:
            logger.debug(f"Error playing sound: {e}")

    def set_volume(self, volume: float):
        """
        Set master volume

        Args:
            volume: Volume level 0.0 to 1.0
        """
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self.volume * 100:.0f}%")

    def enable_sounds(self):
        """Enable sound effects"""
        self.sounds_enabled = True
        logger.info("Sounds enabled")

    def disable_sounds(self):
        """Disable sound effects"""
        self.sounds_enabled = False
        logger.info("Sounds disabled")

    def create_sound_assets(self):
        """
        Create placeholder sound files if they don't exist
        This would generate simple beep sounds for testing
        """
        self.sounds_dir.mkdir(parents=True, exist_ok=True)

        # In a real implementation, this would generate or download sound files
        # For now, just create empty marker files
        for sound_file in self.sound_files.values():
            file_path = self.sounds_dir / sound_file
            if not file_path.exists():
                # Create empty file as placeholder
                file_path.touch()

        logger.info(f"Created placeholder sound files in {self.sounds_dir}")


class NotificationManager:
    """
    Manages system notifications
    """

    def __init__(self, sound_manager: Optional[SoundManager] = None):
        self.sound_manager = sound_manager
        self.notification_history = []

    def notify(
        self,
        title: str,
        message: str,
        notification_type: str = "info",
        play_sound: bool = True,
    ):
        """
        Send a system notification

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type (info, warning, error, success)
            play_sound: Whether to play notification sound
        """
        try:
            # Store in history
            self.notification_history.append(
                {
                    "title": title,
                    "message": message,
                    "type": notification_type,
                    "timestamp": logger.Formatter().formatTime(
                        logger.LogRecord("", 0, "", 0, "", "", "")
                    ),
                }
            )

            # Play sound if enabled
            if play_sound and self.sound_manager:
                if notification_type == "error":
                    self.sound_manager.play("emergency_alarm")
                elif notification_type == "success":
                    self.sound_manager.play("profit")
                else:
                    self.sound_manager.play("notification")

            # Try to send OS notification
            self._send_os_notification(title, message, notification_type)

        except Exception as e:
            logger.debug(f"Notification error: {e}")

    def _send_os_notification(self, title: str, message: str, notification_type: str):
        """Send OS-level notification"""
        try:
            # Try plyer for cross-platform notifications
            from plyer import notification as plyer_notify

            plyer_notify.notify(
                title=title, message=message, app_name="Nexlify", timeout=10
            )

        except ImportError:
            # Fallback: try platform-specific methods
            import platform

            system = platform.system()

            if system == "Windows":
                self._send_windows_notification(title, message)
            elif system == "Darwin":  # macOS
                self._send_macos_notification(title, message)
            elif system == "Linux":
                self._send_linux_notification(title, message)

        except Exception as e:
            logger.debug(f"Could not send OS notification: {e}")

    def _send_windows_notification(self, title: str, message: str):
        """Send Windows notification"""
        try:
            from win10toast import ToastNotifier

            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=10, threaded=True)
        except:
            pass

    def _send_macos_notification(self, title: str, message: str):
        """Send macOS notification"""
        try:
            import subprocess

            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script])
        except:
            pass

    def _send_linux_notification(self, title: str, message: str):
        """Send Linux notification"""
        try:
            import subprocess

            subprocess.run(["notify-send", title, message])
        except:
            pass


# Convenience functions
def create_effects_manager(config: Dict = None) -> CyberpunkEffects:
    """Create and return a CyberpunkEffects instance"""
    return CyberpunkEffects(config)


def create_sound_manager(config: Dict = None) -> SoundManager:
    """Create and return a SoundManager instance"""
    return SoundManager(config)


def create_notification_manager(
    sound_manager: Optional[SoundManager] = None,
) -> NotificationManager:
    """Create and return a NotificationManager instance"""
    return NotificationManager(sound_manager)
