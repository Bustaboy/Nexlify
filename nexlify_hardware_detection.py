#!/usr/bin/env python3
"""
Nexlify Hardware Detection Module
Automatically detects system hardware and optimizes settings for performance
"""

import logging
import platform
import psutil
import os
from typing import Dict, Optional, List
from pathlib import Path
import json

from error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class HardwareDetector:
    """
    Detects system hardware capabilities and recommends optimal settings
    """

    def __init__(self):
        self.hardware_info: Dict = {}
        self.performance_profile: str = "unknown"
        self.capabilities: Dict = {}

        logger.info("🔧 Hardware Detector initializing...")
        self.detect_all()

    @handle_errors("Hardware Detection", reraise=False)
    def detect_all(self):
        """Detect all hardware components"""
        self.detect_cpu()
        self.detect_memory()
        self.detect_gpu()
        self.detect_storage()
        self.detect_network()
        self.analyze_capabilities()
        self.determine_performance_profile()

        logger.info(f"✅ Hardware detection complete - Profile: {self.performance_profile}")

    def detect_cpu(self):
        """Detect CPU information"""
        try:
            self.hardware_info['cpu'] = {
                'cores_physical': psutil.cpu_count(logical=False),
                'cores_logical': psutil.cpu_count(logical=True),
                'frequency_current': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'usage_percent': psutil.cpu_percent(interval=1)
            }

            logger.info(f"CPU: {self.hardware_info['cpu']['cores_physical']} cores, "
                       f"{self.hardware_info['cpu']['frequency_current']:.0f} MHz")

        except Exception as e:
            logger.error(f"CPU detection error: {e}")
            self.hardware_info['cpu'] = {'cores_physical': 1, 'cores_logical': 1}

    def detect_memory(self):
        """Detect RAM information"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            self.hardware_info['memory'] = {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_percent': mem.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_percent': swap.percent
            }

            logger.info(f"RAM: {self.hardware_info['memory']['total_gb']:.1f} GB total, "
                       f"{self.hardware_info['memory']['available_gb']:.1f} GB available")

        except Exception as e:
            logger.error(f"Memory detection error: {e}")
            self.hardware_info['memory'] = {'total_gb': 4, 'available_gb': 2}

    def detect_gpu(self):
        """Detect GPU information and capabilities"""
        try:
            gpu_info = {
                'nvidia_available': False,
                'amd_available': False,
                'cuda_available': False,
                'gpu_count': 0,
                'gpu_names': []
            }

            # Try to detect NVIDIA GPU via nvidia-smi
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                       '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    gpu_info['nvidia_available'] = True
                    gpu_lines = result.stdout.strip().split('\n')
                    gpu_info['gpu_count'] = len(gpu_lines)
                    gpu_info['gpu_names'] = [line.split(',')[0].strip() for line in gpu_lines]

                    # Check for CUDA
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_info['cuda_available'] = True
                            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                    except ImportError:
                        pass

            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Try to detect AMD GPU
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showproductname'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info['amd_available'] = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: Try PyOpenCL for general GPU detection
            if not gpu_info['nvidia_available'] and not gpu_info['amd_available']:
                try:
                    import pyopencl as cl
                    platforms = cl.get_platforms()
                    for platform in platforms:
                        devices = platform.get_devices()
                        for device in devices:
                            if device.type == cl.device_type.GPU:
                                gpu_info['gpu_count'] += 1
                                gpu_info['gpu_names'].append(device.name)
                except ImportError:
                    pass

            self.hardware_info['gpu'] = gpu_info

            if gpu_info['gpu_count'] > 0:
                logger.info(f"GPU: {gpu_info['gpu_count']} detected - {', '.join(gpu_info['gpu_names'])}")
            else:
                logger.info("No GPU detected - CPU-only mode")

        except Exception as e:
            logger.error(f"GPU detection error: {e}")
            self.hardware_info['gpu'] = {'nvidia_available': False, 'amd_available': False,
                                        'cuda_available': False, 'gpu_count': 0}

    def detect_storage(self):
        """Detect storage information"""
        try:
            # Get disk usage for current directory
            usage = psutil.disk_usage(os.getcwd())

            # Try to determine if SSD or HDD
            storage_type = self._detect_storage_type()

            self.hardware_info['storage'] = {
                'total_gb': usage.total / (1024**3),
                'free_gb': usage.free / (1024**3),
                'used_percent': usage.percent,
                'type': storage_type
            }

            logger.info(f"Storage: {self.hardware_info['storage']['free_gb']:.1f} GB free "
                       f"({storage_type})")

        except Exception as e:
            logger.error(f"Storage detection error: {e}")
            self.hardware_info['storage'] = {'total_gb': 100, 'free_gb': 50, 'type': 'unknown'}

    def _detect_storage_type(self) -> str:
        """Attempt to detect if storage is SSD or HDD"""
        try:
            # Linux
            if platform.system() == 'Linux':
                with open('/sys/block/sda/queue/rotational', 'r') as f:
                    is_rotational = f.read().strip() == '1'
                    return 'HDD' if is_rotational else 'SSD'

            # Windows - check via WMI
            elif platform.system() == 'Windows':
                import subprocess
                result = subprocess.run(['wmic', 'diskdrive', 'get', 'MediaType'],
                                      capture_output=True, text=True)
                if 'SSD' in result.stdout:
                    return 'SSD'
                else:
                    return 'HDD'

            # macOS - NVMe is typically SSD
            elif platform.system() == 'Darwin':
                result = subprocess.run(['diskutil', 'info', 'disk0'],
                                      capture_output=True, text=True)
                if 'Solid State' in result.stdout or 'SSD' in result.stdout:
                    return 'SSD'
                else:
                    return 'HDD'

        except Exception:
            pass

        return 'unknown'

    def detect_network(self):
        """Detect network capabilities"""
        try:
            # Get network interface stats
            net_if_stats = psutil.net_if_stats()
            net_io = psutil.net_io_counters()

            # Find fastest network interface
            max_speed = 0
            for interface, stats in net_if_stats.items():
                if stats.isup and stats.speed > max_speed:
                    max_speed = stats.speed

            self.hardware_info['network'] = {
                'max_speed_mbps': max_speed,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'is_connected': max_speed > 0
            }

            if max_speed >= 1000:
                connection_type = "Gigabit"
            elif max_speed >= 100:
                connection_type = "Fast Ethernet"
            else:
                connection_type = "Limited"

            logger.info(f"Network: {connection_type} ({max_speed} Mbps)")

        except Exception as e:
            logger.error(f"Network detection error: {e}")
            self.hardware_info['network'] = {'max_speed_mbps': 100, 'is_connected': True}

    def analyze_capabilities(self):
        """Analyze hardware capabilities for specific features"""
        self.capabilities = {
            'ml_acceleration': False,
            'parallel_processing': False,
            'high_frequency_trading': False,
            'multiple_exchanges': False,
            'advanced_charting': False,
            'real_time_analysis': False
        }

        # ML Acceleration (GPU required)
        if self.hardware_info.get('gpu', {}).get('cuda_available', False):
            self.capabilities['ml_acceleration'] = True

        # Parallel processing (4+ cores)
        if self.hardware_info.get('cpu', {}).get('cores_physical', 0) >= 4:
            self.capabilities['parallel_processing'] = True

        # High frequency trading (8+ cores, 16+ GB RAM, SSD)
        cpu_cores = self.hardware_info.get('cpu', {}).get('cores_physical', 0)
        total_ram = self.hardware_info.get('memory', {}).get('total_gb', 0)
        storage_type = self.hardware_info.get('storage', {}).get('type', 'unknown')

        if cpu_cores >= 8 and total_ram >= 16 and storage_type == 'SSD':
            self.capabilities['high_frequency_trading'] = True

        # Multiple exchanges (4+ cores, 8+ GB RAM)
        if cpu_cores >= 4 and total_ram >= 8:
            self.capabilities['multiple_exchanges'] = True

        # Advanced charting (4+ GB RAM)
        if total_ram >= 4:
            self.capabilities['advanced_charting'] = True

        # Real-time analysis (4+ cores, 8+ GB RAM)
        if cpu_cores >= 4 and total_ram >= 8:
            self.capabilities['real_time_analysis'] = True

    def determine_performance_profile(self):
        """Determine overall system performance profile"""
        cpu_cores = self.hardware_info.get('cpu', {}).get('cores_physical', 0)
        total_ram = self.hardware_info.get('memory', {}).get('total_gb', 0)
        has_gpu = self.hardware_info.get('gpu', {}).get('gpu_count', 0) > 0
        storage_type = self.hardware_info.get('storage', {}).get('type', 'unknown')

        # High-end system
        if cpu_cores >= 8 and total_ram >= 16 and has_gpu and storage_type == 'SSD':
            self.performance_profile = 'high_performance'

        # Mid-range system
        elif cpu_cores >= 4 and total_ram >= 8:
            self.performance_profile = 'balanced'

        # Low-end system
        elif cpu_cores >= 2 and total_ram >= 4:
            self.performance_profile = 'lightweight'

        # Very limited system
        else:
            self.performance_profile = 'minimal'

    def get_recommended_settings(self) -> Dict:
        """Get recommended application settings based on hardware"""
        settings = {
            'max_concurrent_trades': 5,
            'scan_interval_seconds': 300,
            'enable_ml_features': False,
            'enable_advanced_charting': True,
            'max_exchanges': 3,
            'use_testnet': True,  # Always start with testnet
            'enable_real_time_updates': True,
            'chart_update_interval_ms': 5000,
            'log_level': 'INFO'
        }

        profile = self.performance_profile

        if profile == 'high_performance':
            settings.update({
                'max_concurrent_trades': 20,
                'scan_interval_seconds': 60,
                'enable_ml_features': self.capabilities['ml_acceleration'],
                'max_exchanges': 10,
                'enable_real_time_updates': True,
                'chart_update_interval_ms': 1000,
                'log_level': 'INFO'
            })

        elif profile == 'balanced':
            settings.update({
                'max_concurrent_trades': 10,
                'scan_interval_seconds': 120,
                'enable_ml_features': False,
                'max_exchanges': 5,
                'enable_real_time_updates': True,
                'chart_update_interval_ms': 3000,
                'log_level': 'INFO'
            })

        elif profile == 'lightweight':
            settings.update({
                'max_concurrent_trades': 5,
                'scan_interval_seconds': 300,
                'enable_ml_features': False,
                'max_exchanges': 3,
                'enable_real_time_updates': True,
                'chart_update_interval_ms': 5000,
                'log_level': 'WARNING'
            })

        else:  # minimal
            settings.update({
                'max_concurrent_trades': 2,
                'scan_interval_seconds': 600,
                'enable_ml_features': False,
                'enable_advanced_charting': False,
                'max_exchanges': 1,
                'enable_real_time_updates': False,
                'chart_update_interval_ms': 10000,
                'log_level': 'ERROR'
            })

        return settings

    def get_system_report(self) -> Dict:
        """Generate comprehensive system report"""
        return {
            'hardware': self.hardware_info,
            'capabilities': self.capabilities,
            'performance_profile': self.performance_profile,
            'recommended_settings': self.get_recommended_settings(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'architecture': platform.architecture()[0]
            }
        }

    def save_hardware_profile(self, filepath: str = "config/hardware_profile.json"):
        """Save hardware profile to file"""
        try:
            profile_path = Path(filepath)
            profile_path.parent.mkdir(parents=True, exist_ok=True)

            report = self.get_system_report()

            with open(profile_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Hardware profile saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save hardware profile: {e}")
            return False

    def apply_optimal_settings(self, config_path: str = "config/neural_config.json"):
        """Apply optimal settings to configuration file"""
        try:
            recommended = self.get_recommended_settings()

            # Load existing config
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            # Update with hardware-optimized settings
            if 'trading' not in config:
                config['trading'] = {}

            config['trading']['max_concurrent_trades'] = recommended['max_concurrent_trades']

            if 'neural_net' not in config:
                config['neural_net'] = {}

            config['neural_net']['scan_interval_seconds'] = recommended['scan_interval_seconds']

            if 'environment' not in config:
                config['environment'] = {}

            config['environment']['log_level'] = recommended['log_level']

            # Add hardware profile marker
            config['hardware_profile'] = {
                'detected': True,
                'profile': self.performance_profile,
                'detection_date': logger.Formatter().formatTime(logger.LogRecord('', 0, '', 0, '', '', ''))
            }

            # Save updated config
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"✅ Applied {self.performance_profile} settings to configuration")
            return True

        except Exception as e:
            logger.error(f"Failed to apply optimal settings: {e}")
            return False


def auto_detect_and_configure():
    """
    Convenience function to detect hardware and apply optimal settings
    """
    detector = HardwareDetector()
    detector.save_hardware_profile()
    detector.apply_optimal_settings()
    return detector


if __name__ == "__main__":
    # Run hardware detection
    detector = auto_detect_and_configure()

    # Print report
    print("\n" + "="*70)
    print("NEXLIFY HARDWARE DETECTION REPORT")
    print("="*70)
    print(f"\nPerformance Profile: {detector.performance_profile.upper()}")
    print(f"\nCPU: {detector.hardware_info['cpu']['cores_physical']} cores")
    print(f"RAM: {detector.hardware_info['memory']['total_gb']:.1f} GB")
    print(f"GPU: {detector.hardware_info['gpu']['gpu_count']} detected")
    print(f"Storage: {detector.hardware_info['storage']['type']}")

    print("\nCapabilities:")
    for capability, enabled in detector.capabilities.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {capability.replace('_', ' ').title()}")

    print("\nRecommended Settings:")
    settings = detector.get_recommended_settings()
    for key, value in settings.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

    print("\n" + "="*70)
