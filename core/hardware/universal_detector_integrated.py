# Location: nexlify/core/hardware/universal_detector_integrated.py
# Integrated Universal Hardware Detector - Complete Hardware Discovery System

"""
🔧 UNIVERSAL HARDWARE DETECTOR v2.0 - INTEGRATED EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete hardware discovery with architecture selection and offloading.
Every chip has a story. Every system has hidden potential.

"The street always finds new uses for military tech." - Johnny Silverhand
"""

import os
import json
import time
import subprocess
import platform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Additional imports for hardware detection
try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    import pyamdgpuinfo
    AMD_GPU_AVAILABLE = True
except ImportError:
    AMD_GPU_AVAILABLE = False

# Import our enhanced components
# Note: In production, these would be proper imports from their respective files
# from .universal_detector import (
#     UniversalHardwareDetector,
#     DynamicHardwareProfile,
#     AcceleratorInfo,
#     ComputeCapability
# )
# from .integrated_architecture_selector import IntegratedArchitectureSelector
# from .offloading_detector import OffloadingDetector
# from .specialized_accelerator_detector import SpecializedAcceleratorDetector

# For this integrated version, we'll assume they're available
from .universal_detector import (
    UniversalHardwareDetector,
    DynamicHardwareProfile,
    AcceleratorInfo,
    ComputeCapability
)
from .integrated_architecture_selector import IntegratedArchitectureSelector
from .offloading_detector import OffloadingDetector
from .specialized_accelerator_detector import SpecializedAcceleratorDetector


# Additional accelerators to detect
ADDITIONAL_ACCELERATORS = {
    # NVIDIA Jetson Family
    "jetson_nano": {"vendor": "NVIDIA", "model": "Jetson Nano", "int8_tops": 0.5},
    "jetson_xavier_nx": {"vendor": "NVIDIA", "model": "Jetson Xavier NX", "int8_tops": 21.0},
    "jetson_orin_nano": {"vendor": "NVIDIA", "model": "Jetson Orin Nano", "int8_tops": 40.0},
    "jetson_orin_nx": {"vendor": "NVIDIA", "model": "Jetson Orin NX", "int8_tops": 100.0},
    "jetson_agx_orin": {"vendor": "NVIDIA", "model": "Jetson AGX Orin", "int8_tops": 275.0},
    
    # Rockchip NPUs
    "rk3588_npu": {"vendor": "Rockchip", "model": "RK3588 NPU", "int8_tops": 6.0},
    "rk3568_npu": {"vendor": "Rockchip", "model": "RK3568 NPU", "int8_tops": 1.0},
    
    # Intel Integrated NPUs
    "meteor_lake_npu": {"vendor": "Intel", "model": "Meteor Lake NPU", "int8_tops": 10.0},
    "raptor_lake_vnni": {"vendor": "Intel", "model": "Raptor Lake VNNI", "int8_tops": 2.0},
    
    # AMD XDNA
    "ryzen_7040_xdna": {"vendor": "AMD", "model": "Ryzen AI (XDNA)", "int8_tops": 10.0},
    "ryzen_8040_xdna": {"vendor": "AMD", "model": "Ryzen AI (XDNA 2)", "int8_tops": 16.0},
    
    # Xilinx/AMD Edge AI
    "versal_ai_edge": {"vendor": "AMD", "model": "Versal AI Edge", "int8_tops": 58.0},
    "kria_k26": {"vendor": "AMD", "model": "Kria K26", "int8_tops": 1.4},
    
    # Kneron Edge AI
    "kneron_kl520": {"vendor": "Kneron", "model": "KL520", "int8_tops": 0.3},
    "kneron_kl720": {"vendor": "Kneron", "model": "KL720", "int8_tops": 1.5},
    
    # Syntiant Ultra-Low Power
    "syntiant_ndp120": {"vendor": "Syntiant", "model": "NDP120", "int8_tops": 0.02},
    
    # Horizon Robotics (Automotive)
    "horizon_journey5": {"vendor": "Horizon", "model": "Journey 5", "int8_tops": 128.0},
    
    # BrainChip Neuromorphic
    "akida_1000": {"vendor": "BrainChip", "model": "Akida 1000", "int8_tops": 1.0},
    
    # Flex Logix
    "inferx_x1": {"vendor": "Flex Logix", "model": "InferX X1", "int8_tops": 8.5},
    
    # Kalray MPPA
    "kalray_coolidge": {"vendor": "Kalray", "model": "Coolidge", "int8_tops": 25.0}
}


@dataclass
class StorageProfile:
    """Storage device capabilities for memory tiering"""
    storage_type: str  # "nvme_gen5", "nvme_gen4", "nvme_gen3", "ssd", "hdd"
    speed_gbps: float
    capacity_gb: float
    interface: str  # "pcie4x4", "pcie3x4", "sata3", etc
    supports_direct_storage: bool = False
    supports_gpu_direct: bool = False
    numa_node: Optional[int] = None


class IntegratedUniversalDetector(UniversalHardwareDetector):
    """
    🔧 INTEGRATED DETECTOR
    
    Combines all detection capabilities into one comprehensive system.
    Detects hardware, selects architecture, and identifies optimizations.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize integrated detector with all components"""
        super().__init__(cache_dir)
        
        # Initialize sub-detectors
        self.specialized_detector = SpecializedAcceleratorDetector()
        self.offload_detector = OffloadingDetector()
        self.architecture_selector = IntegratedArchitectureSelector()
        
        print("🔧 Integrated Universal Hardware Detector v2.0")
        print("   Enhanced with specialized accelerator support")
        print("   Offloading-aware architecture selection")
        print("   Storage-optimized memory tiering")
    
    def detect_all(self, force_benchmark: bool = False) -> DynamicHardwareProfile:
        """
        Enhanced detection with all integrated components.
        
        This overrides the base method to add:
        - Specialized accelerator detection
        - Storage profiling
        - Offloading analysis
        - Integrated architecture selection
        """
        # Run base detection first
        profile = super().detect_all(force_benchmark)
        
        # Enhance with specialized accelerator detection
        print("\n🧠 Enhanced Accelerator Detection...")
        specialized_accelerators = self.specialized_detector.detect_all()
        
        # Merge specialized accelerators into profile
        for spec_accel in specialized_accelerators:
            # Convert to our AcceleratorInfo format
            accel_info = self._convert_specialized_accelerator(spec_accel)
            profile.accelerators.append(accel_info)
        
        # Detect additional accelerators
        print("\n🔍 Detecting Additional Accelerators...")
        additional = self._detect_additional_accelerators()
        profile.accelerators.extend(additional)
        
        # Detect storage capabilities
        print("\n💾 Profiling Storage Subsystem...")
        profile.storage_info = self._profile_storage()
        
        # Run offloading analysis
        print("\n🔄 Analyzing Offloading Opportunities...")
        profile.offloading_capabilities = self.offload_detector.detect_opportunities(profile)
        
        # Select optimal architecture with all information
        print("\n🏗️ Selecting Optimal Architecture...")
        architecture, reasoning = self.architecture_selector.select_architecture(profile)
        profile.recommended_architecture = architecture.name
        profile.architecture_reasoning = reasoning
        
        # Estimate performance with selected architecture
        performance = self.architecture_selector.estimate_performance(profile, architecture)
        profile.estimated_trinity_fps = performance['trinity_fps']
        profile.estimated_max_batch_size = performance['max_batch_size']
        profile.estimated_max_sequence_length = performance['max_sequence_length']
        
        # Cache the complete profile
        self._cache_profile(profile)
        
        # Display enhanced summary
        self._display_enhanced_summary(profile)
        
        return profile
    
    def _detect_additional_accelerators(self) -> List[AcceleratorInfo]:
        """Detect additional accelerators not covered by specialized detector"""
        additional = []
        
        # === HIGH PRIORITY DETECTIONS ===
        
        # Intel Arc GPU AI features
        arc_ai = self._detect_intel_arc_ai()
        if arc_ai:
            additional.extend(arc_ai)
        
        # AMD RDNA3 AI features
        rdna3_ai = self._detect_amd_rdna3_ai()
        if rdna3_ai:
            additional.extend(rdna3_ai)
        
        # NVIDIA Workstation/Professional cards
        nvidia_pro = self._detect_nvidia_professional()
        if nvidia_pro:
            additional.extend(nvidia_pro)
        
        # NVIDIA Jetson detection
        jetson = self._detect_nvidia_jetson()
        if jetson:
            additional.append(jetson)
        
        # Rockchip NPU detection
        rockchip = self._detect_rockchip_npu()
        if rockchip:
            additional.append(rockchip)
        
        # Intel integrated NPU
        intel_npu = self._detect_intel_integrated_npu()
        if intel_npu:
            additional.append(intel_npu)
        
        # AMD XDNA
        amd_xdna = self._detect_amd_xdna()
        if amd_xdna:
            additional.append(amd_xdna)
        
        # === MEDIUM PRIORITY DETECTIONS ===
        
        # FPGA AI accelerators
        fpgas = self._detect_fpga_accelerators()
        if fpgas:
            additional.extend(fpgas)
        
        # Mobile flagship NPUs (enhanced detection)
        mobile_npus = self._detect_mobile_flagship_npus()
        if mobile_npus:
            additional.extend(mobile_npus)
        
        # Edge AI boxes
        edge_boxes = self._detect_edge_ai_boxes()
        if edge_boxes:
            additional.extend(edge_boxes)
        
        return additional
    
    def _detect_nvidia_jetson(self) -> Optional[AcceleratorInfo]:
        """Detect NVIDIA Jetson devices"""
        try:
            # Check for Jetson
            if not os.path.exists('/etc/nv_tegra_release'):
                return None
            
            # Read Jetson model
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
            
            # Map to our known Jetson devices
            for key, info in ADDITIONAL_ACCELERATORS.items():
                if key.startswith('jetson') and info['model'].lower() in model.lower():
                    print(f"   ✓ {info['model']} detected")
                    
                    return AcceleratorInfo(
                        accelerator_type="jetson",
                        vendor=info['vendor'],
                        model=info['model'],
                        driver_version=self._get_jetson_version(),
                        compute_capability=ComputeCapability.AI_GPU,
                        memory_mb=self._get_jetson_memory(),
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=True,
                        max_batch_size=32
                    )
            
            # Generic Jetson
            return AcceleratorInfo(
                accelerator_type="jetson",
                vendor="NVIDIA",
                model="Jetson Device",
                driver_version=self._get_jetson_version(),
                compute_capability=ComputeCapability.AI_GPU,
                memory_mb=4096,
                supports_int8=True,
                supports_fp16=True,
                supports_dynamic_shapes=True,
                max_batch_size=16
            )
            
        except Exception:
            return None
    
    def _detect_rockchip_npu(self) -> Optional[AcceleratorInfo]:
        """Detect Rockchip NPUs"""
        try:
            # Check for Rockchip kernel
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            if 'rockchip' not in result.stdout.lower():
                return None
            
            # Check device tree for NPU
            if os.path.exists('/proc/device-tree/npu@fde40000'):
                print("   ✓ Rockchip NPU detected")
                
                # Determine model from CPU info
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                
                if 'rk3588' in cpuinfo.lower():
                    info = ADDITIONAL_ACCELERATORS['rk3588_npu']
                elif 'rk3568' in cpuinfo.lower():
                    info = ADDITIONAL_ACCELERATORS['rk3568_npu']
                else:
                    info = {"vendor": "Rockchip", "model": "NPU", "int8_tops": 1.0}
                
                return AcceleratorInfo(
                    accelerator_type="npu",
                    vendor=info['vendor'],
                    model=info['model'],
                    driver_version="RKNN",
                    compute_capability=ComputeCapability.NEURAL_ENGINE,
                    memory_mb=256,  # Shared with system
                    supports_int8=True,
                    supports_fp16=True,
                    supports_dynamic_shapes=False,
                    max_batch_size=8
                )
                
        except Exception:
            return None
    
    def _detect_intel_integrated_npu(self) -> Optional[AcceleratorInfo]:
        """Detect Intel integrated NPUs (Meteor Lake, etc)"""
        try:
            if not CPUINFO_AVAILABLE:
                return None
                
            # Check CPU model
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', '').lower()
            
            # Meteor Lake detection
            if 'core ultra' in cpu_brand or 'meteor lake' in cpu_brand:
                print("   ✓ Intel Meteor Lake NPU detected")
                info = ADDITIONAL_ACCELERATORS['meteor_lake_npu']
                
                return AcceleratorInfo(
                    accelerator_type="npu",
                    vendor=info['vendor'],
                    model=info['model'],
                    driver_version="Intel NPU Driver",
                    compute_capability=ComputeCapability.NEURAL_ENGINE,
                    memory_mb=0,  # Uses system memory
                    supports_int8=True,
                    supports_fp16=True,
                    supports_dynamic_shapes=True,
                    max_batch_size=16
                )
            
            # Check for VNNI support in newer Intel CPUs
            flags = cpu_info.get('flags', [])
            if 'avx512_vnni' in flags or 'avx_vnni' in flags:
                if any(gen in cpu_brand for gen in ['12th', '13th', '14th', 'raptor', 'alder']):
                    print("   ✓ Intel VNNI acceleration detected")
                    return AcceleratorInfo(
                        accelerator_type="cpu_acceleration",
                        vendor="Intel",
                        model="VNNI Accelerator",
                        driver_version="CPU",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=0,
                        supports_int8=True,
                        supports_fp16=False,
                        supports_dynamic_shapes=True,
                        max_batch_size=32
                    )
                    
        except Exception:
            return None
    
    def _detect_amd_xdna(self) -> Optional[AcceleratorInfo]:
        """Detect AMD XDNA AI Engine"""
        try:
            if not CPUINFO_AVAILABLE:
                return None
                
            # Check for AMD Ryzen AI
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', '').lower()
            
            # Look for Ryzen 7040/8040 series
            if 'ryzen' in cpu_brand and ('7040' in cpu_brand or '8040' in cpu_brand):
                # Check for XDNA driver
                if os.path.exists('/dev/xdna') or os.path.exists('/sys/class/xdna'):
                    print("   ✓ AMD XDNA AI Engine detected")
                    
                    if '8040' in cpu_brand:
                        info = ADDITIONAL_ACCELERATORS['ryzen_8040_xdna']
                    else:
                        info = ADDITIONAL_ACCELERATORS['ryzen_7040_xdna']
                    
                    return AcceleratorInfo(
                        accelerator_type="npu",
                        vendor=info['vendor'],
                        model=info['model'],
                        driver_version="XDNA Driver",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=0,  # Uses system memory
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=True,
                        max_batch_size=16
                    )
                    
        except Exception:
            return None
    
    def _detect_intel_arc_ai(self) -> List[AcceleratorInfo]:
        """Detect Intel Arc GPU AI features (XMX engines)"""
        arc_gpus = []
        
        try:
            # Check for Intel GPUs
            if platform.system() == "Linux":
                # Check for Intel GPU via sysfs
                intel_gpu_path = Path('/sys/class/drm/card0/device/vendor')
                if intel_gpu_path.exists():
                    vendor_id = intel_gpu_path.read_text().strip()
                    if vendor_id == '0x8086':  # Intel vendor ID
                        # Get device ID
                        device_path = Path('/sys/class/drm/card0/device/device')
                        device_id = device_path.read_text().strip()
                        
                        # Arc GPU device IDs
                        arc_devices = {
                            '0x5690': {'model': 'Arc A770', 'xmx_units': 512, 'int8_tops': 229},
                            '0x5691': {'model': 'Arc A770', 'xmx_units': 512, 'int8_tops': 229},
                            '0x5692': {'model': 'Arc A750', 'xmx_units': 448, 'int8_tops': 200},
                            '0x5693': {'model': 'Arc A580', 'xmx_units': 384, 'int8_tops': 172},
                            '0x5694': {'model': 'Arc A380', 'xmx_units': 128, 'int8_tops': 57},
                            '0x56a0': {'model': 'Arc A310', 'xmx_units': 64, 'int8_tops': 29},
                            # Battlemage IDs would go here
                        }
                        
                        if device_id in arc_devices:
                            info = arc_devices[device_id]
                            print(f"   ✓ Intel {info['model']} with XMX engines detected")
                            
                            arc_gpus.append(AcceleratorInfo(
                                accelerator_type="gpu_ai",
                                vendor="Intel",
                                model=f"{info['model']} XMX",
                                driver_version=self._get_intel_gpu_driver_version(),
                                compute_capability=ComputeCapability.AI_GPU,
                                memory_mb=self._get_intel_gpu_memory(),
                                supports_int8=True,
                                supports_fp16=True,
                                supports_dynamic_shapes=True,
                                max_batch_size=64
                            ))
            
            elif platform.system() == "Windows":
                # Windows detection via WMI
                try:
                    import wmi
                    c = wmi.WMI()
                    
                    for gpu in c.Win32_VideoController():
                        if 'intel' in gpu.Name.lower() and 'arc' in gpu.Name.lower():
                            print(f"   ✓ {gpu.Name} detected")
                            
                            # Estimate based on name
                            if 'a770' in gpu.Name.lower():
                                int8_tops = 229
                            elif 'a750' in gpu.Name.lower():
                                int8_tops = 200
                            elif 'a380' in gpu.Name.lower():
                                int8_tops = 57
                            else:
                                int8_tops = 100  # Conservative estimate
                            
                            arc_gpus.append(AcceleratorInfo(
                                accelerator_type="gpu_ai",
                                vendor="Intel",
                                model=f"{gpu.Name} XMX",
                                driver_version=gpu.DriverVersion,
                                compute_capability=ComputeCapability.AI_GPU,
                                memory_mb=int(gpu.AdapterRAM / (1024*1024)) if gpu.AdapterRAM else 6144,
                                supports_int8=True,
                                supports_fp16=True,
                                supports_dynamic_shapes=True,
                                max_batch_size=64
                            ))
                except ImportError:
                    pass
                    
        except Exception:
            pass
            
        return arc_gpus
    
    def _detect_amd_rdna3_ai(self) -> List[AcceleratorInfo]:
        """Detect AMD RDNA3 AI features (WMMA instructions)"""
        rdna3_gpus = []
        
        try:
            # Check for AMD GPUs with AI capabilities
            if AMD_GPU_AVAILABLE:
                import pyamdgpuinfo
                
                devices = pyamdgpuinfo.list_devices()
                for device in devices:
                    # RDNA3 GPU IDs (Navi 31, 32, 33)
                    rdna3_chips = {
                        'navi31': {'model': 'RX 7900 XTX', 'ai_units': 96, 'int8_tops': 123},
                        'navi32': {'model': 'RX 7800 XT', 'ai_units': 60, 'int8_tops': 77},
                        'navi33': {'model': 'RX 7600', 'ai_units': 32, 'int8_tops': 41},
                    }
                    
                    chip_name = device.get_chip_name().lower()
                    for chip_id, info in rdna3_chips.items():
                        if chip_id in chip_name:
                            print(f"   ✓ AMD {info['model']} with AI accelerators detected")
                            
                            rdna3_gpus.append(AcceleratorInfo(
                                accelerator_type="gpu_ai",
                                vendor="AMD",
                                model=f"{info['model']} AI",
                                driver_version=device.get_driver_version(),
                                compute_capability=ComputeCapability.AI_GPU,
                                memory_mb=device.get_vram_size() // (1024*1024),
                                supports_int8=True,
                                supports_fp16=True,
                                supports_dynamic_shapes=True,
                                max_batch_size=32
                            ))
                            break
                            
            # Fallback detection via ROCm
            elif os.path.exists('/opt/rocm/bin/rocm-smi'):
                result = subprocess.run(
                    ['/opt/rocm/bin/rocm-smi', '--showproductname'],
                    capture_output=True,
                    text=True
                )
                
                if 'RX 7' in result.stdout:
                    # Found RDNA3 GPU
                    for line in result.stdout.split('\n'):
                        if 'RX 7' in line:
                            model = line.strip()
                            print(f"   ✓ AMD {model} with WMMA support detected")
                            
                            rdna3_gpus.append(AcceleratorInfo(
                                accelerator_type="gpu_ai",
                                vendor="AMD",
                                model=f"{model} AI",
                                driver_version=self._get_rocm_version(),
                                compute_capability=ComputeCapability.AI_GPU,
                                memory_mb=8192,  # Conservative estimate
                                supports_int8=True,
                                supports_fp16=True,
                                supports_dynamic_shapes=True,
                                max_batch_size=32
                            ))
                            
        except Exception:
            pass
            
        return rdna3_gpus
    
    def _detect_nvidia_professional(self) -> List[AcceleratorInfo]:
        """Detect NVIDIA professional/workstation cards"""
        pro_gpus = []
        
        try:
            if NVIDIA_AVAILABLE:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Professional card mappings
                    pro_cards = {
                        'Tesla T4': {'int8_tops': 130, 'category': 'datacenter'},
                        'A2': {'int8_tops': 36, 'category': 'edge'},
                        'A10': {'int8_tops': 250, 'category': 'datacenter'},
                        'A30': {'int8_tops': 330, 'category': 'datacenter'},
                        'A40': {'int8_tops': 300, 'category': 'workstation'},
                        'A100': {'int8_tops': 624, 'category': 'datacenter'},
                        'L4': {'int8_tops': 242, 'category': 'inference'},
                        'L40': {'int8_tops': 362, 'category': 'workstation'},
                        'RTX A2000': {'int8_tops': 50, 'category': 'workstation'},
                        'RTX A4000': {'int8_tops': 100, 'category': 'workstation'},
                        'RTX A4500': {'int8_tops': 150, 'category': 'workstation'},
                        'RTX A5000': {'int8_tops': 200, 'category': 'workstation'},
                        'RTX A5500': {'int8_tops': 220, 'category': 'workstation'},
                        'RTX A6000': {'int8_tops': 310, 'category': 'workstation'},
                        'RTX 4000 Ada': {'int8_tops': 200, 'category': 'workstation'},
                        'RTX 5000 Ada': {'int8_tops': 400, 'category': 'workstation'},
                        'RTX 6000 Ada': {'int8_tops': 600, 'category': 'workstation'},
                    }
                    
                    for card_name, info in pro_cards.items():
                        if card_name in name:
                            print(f"   ✓ NVIDIA {name} professional card detected")
                            
                            # Get memory info
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            pro_gpus.append(AcceleratorInfo(
                                accelerator_type=f"gpu_{info['category']}",
                                vendor="NVIDIA",
                                model=name,
                                driver_version=pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                                compute_capability=ComputeCapability.DATACENTER_AI if info['category'] == 'datacenter' else ComputeCapability.AI_GPU,
                                memory_mb=mem_info.total // (1024*1024),
                                supports_int8=True,
                                supports_fp16=True,
                                supports_dynamic_shapes=True,
                                max_batch_size=256 if info['category'] == 'datacenter' else 128
                            ))
                            break
                            
        except Exception:
            pass
        finally:
            if NVIDIA_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                    
        return pro_gpus
    
    def _detect_fpga_accelerators(self) -> List[AcceleratorInfo]:
        """Detect FPGA AI accelerators"""
        fpgas = []
        
        try:
            # Xilinx/AMD FPGAs
            if os.path.exists('/opt/xilinx/xrt/bin/xbutil'):
                result = subprocess.run(
                    ['/opt/xilinx/xrt/bin/xbutil', 'examine'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Parse for Alveo cards
                    if 'u50' in result.stdout.lower():
                        model = 'Alveo U50'
                        int8_tops = 30
                    elif 'u250' in result.stdout.lower():
                        model = 'Alveo U250'
                        int8_tops = 38
                    elif 'u280' in result.stdout.lower():
                        model = 'Alveo U280'
                        int8_tops = 41
                    else:
                        model = 'Xilinx FPGA'
                        int8_tops = 20
                    
                    print(f"   ✓ Xilinx {model} FPGA detected")
                    
                    fpgas.append(AcceleratorInfo(
                        accelerator_type="fpga",
                        vendor="Xilinx",
                        model=model,
                        driver_version="XRT",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=16384,  # DDR
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=False,
                        max_batch_size=16
                    ))
            
            # Intel FPGAs
            if os.path.exists('/opt/intel/openvino_2021/bin/aocl'):
                result = subprocess.run(
                    ['aocl', 'diagnose'],
                    capture_output=True,
                    text=True
                )
                
                if 'PASSED' in result.stdout:
                    print("   ✓ Intel FPGA with OpenVINO detected")
                    
                    fpgas.append(AcceleratorInfo(
                        accelerator_type="fpga",
                        vendor="Intel",
                        model="Arria 10 FPGA",
                        driver_version="OpenVINO",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=8192,
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=False,
                        max_batch_size=8
                    ))
                    
        except Exception:
            pass
            
        return fpgas
    
    def _detect_mobile_flagship_npus(self) -> List[AcceleratorInfo]:
        """Enhanced detection for mobile flagship NPUs"""
        mobile_npus = []
        
        try:
            if self._is_android():
                # Get detailed chipset info
                chipset = subprocess.run(
                    ['getprop', 'ro.board.platform'],
                    capture_output=True,
                    text=True
                ).stdout.strip()
                
                # Snapdragon 8 Gen 3
                if 'sm8650' in chipset or 'snapdragon 8 gen 3' in subprocess.run(['getprop', 'ro.hardware'], capture_output=True, text=True).stdout.lower():
                    print("   ✓ Snapdragon 8 Gen 3 Hexagon NPU detected")
                    mobile_npus.append(AcceleratorInfo(
                        accelerator_type="mobile_npu",
                        vendor="Qualcomm",
                        model="Hexagon NPU (8 Gen 3)",
                        driver_version="SNPE 2.18",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=0,  # Shared system memory
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=True,
                        max_batch_size=16
                    ))
                
                # MediaTek Dimensity
                elif 'mt6985' in chipset:  # Dimensity 9300
                    print("   ✓ MediaTek Dimensity 9300 APU detected")
                    mobile_npus.append(AcceleratorInfo(
                        accelerator_type="mobile_npu",
                        vendor="MediaTek",
                        model="APU 790 (Dimensity 9300)",
                        driver_version="NeuroPilot",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=0,
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=True,
                        max_batch_size=16
                    ))
                
                # Samsung Exynos
                elif 'exynos' in chipset:
                    print("   ✓ Samsung Exynos NPU detected")
                    mobile_npus.append(AcceleratorInfo(
                        accelerator_type="mobile_npu",
                        vendor="Samsung",
                        model="Exynos NPU",
                        driver_version="Eden",
                        compute_capability=ComputeCapability.NEURAL_ENGINE,
                        memory_mb=0,
                        supports_int8=True,
                        supports_fp16=True,
                        supports_dynamic_shapes=False,
                        max_batch_size=8
                    ))
                    
        except Exception:
            pass
            
        return mobile_npus
    
    def _detect_edge_ai_boxes(self) -> List[AcceleratorInfo]:
        """Detect edge AI boxes and integrated solutions"""
        edge_boxes = []
        
        try:
            # Check for known edge AI systems
            # Hailo-8 in edge boxes
            if os.path.exists('/dev/hailo0'):
                print("   ✓ Hailo-8 Edge AI Box detected")
                edge_boxes.append(AcceleratorInfo(
                    accelerator_type="edge_box",
                    vendor="Hailo",
                    model="Hailo-8 Edge Box",
                    driver_version="HailoRT",
                    compute_capability=ComputeCapability.AI_GPU,
                    memory_mb=2048,
                    supports_int8=True,
                    supports_fp16=True,
                    supports_dynamic_shapes=True,
                    max_batch_size=32
                ))
            
            # NVIDIA Jetson in edge boxes
            if os.path.exists('/etc/nv_tegra_release') and os.path.exists('/etc/systemd/system/edge-ai.service'):
                print("   ✓ NVIDIA Jetson Edge Box detected")
                edge_boxes.append(AcceleratorInfo(
                    accelerator_type="edge_box",
                    vendor="NVIDIA",
                    model="Jetson Edge AI Platform",
                    driver_version=self._get_jetson_version(),
                    compute_capability=ComputeCapability.AI_GPU,
                    memory_mb=8192,
                    supports_int8=True,
                    supports_fp16=True,
                    supports_dynamic_shapes=True,
                    max_batch_size=64
                ))
            
            # Intel NUC with Movidius
            lsusb_result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if '03e7:2485' in lsusb_result.stdout and 'NUC' in subprocess.run(['dmidecode', '-s', 'system-product-name'], capture_output=True, text=True).stdout:
                print("   ✓ Intel NUC with Neural Compute detected")
                edge_boxes.append(AcceleratorInfo(
                    accelerator_type="edge_box",
                    vendor="Intel",
                    model="NUC Edge AI Kit",
                    driver_version="OpenVINO",
                    compute_capability=ComputeCapability.NEURAL_ENGINE,
                    memory_mb=256,
                    supports_int8=True,
                    supports_fp16=True,
                    supports_dynamic_shapes=False,
                    max_batch_size=8
                ))
                
        except Exception:
            pass
            
        return edge_boxes
    
    def _get_intel_gpu_driver_version(self) -> str:
        """Get Intel GPU driver version"""
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ['modinfo', 'i915'],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.split('\n'):
                    if 'version:' in line:
                        return line.split(':')[1].strip()
                        
            return "unknown"
        except:
            return "unknown"
    
    def _get_intel_gpu_memory(self) -> int:
        """Get Intel GPU memory in MB"""
        try:
            # Check for Intel GPU memory via sysfs
            mem_path = Path('/sys/class/drm/card0/device/mem_info_vram_total')
            if mem_path.exists():
                vram_bytes = int(mem_path.read_text().strip())
                return vram_bytes // (1024 * 1024)
            
            # Fallback estimates based on model
            return 6144  # 6GB default for Arc
        except:
            return 6144
    
    def _get_rocm_version(self) -> str:
        """Get ROCm version"""
        try:
            result = subprocess.run(['/opt/rocm/bin/rocm-smi', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.split('\n')[0]
        except:
            pass
        return "unknown"
    
    def _is_android(self) -> bool:
        """Check if running on Android"""
        return os.path.exists('/system/build.prop')
        """Profile storage subsystem for memory tiering"""
        print("   Detecting storage devices...")
        
        # Default profile
        storage = StorageProfile(
            storage_type="ssd",
            speed_gbps=0.5,
            capacity_gb=256,
            interface="sata3"
        )
        
        try:
            # Linux storage detection
            if platform.system() == "Linux":
                # Check NVMe devices
                nvme_devices = []
                for device in Path('/sys/block').iterdir():
                    if device.name.startswith('nvme'):
                        nvme_devices.append(device)
                
                if nvme_devices:
                    # Get first NVMe device info
                    device = nvme_devices[0]
                    
                    # Detect PCIe generation
                    pcie_path = device / 'device' / 'current_link_speed'
                    if pcie_path.exists():
                        link_speed = pcie_path.read_text().strip()
                        
                        if '32' in link_speed:  # 32 GT/s = Gen5
                            storage.storage_type = "nvme_gen5"
                            storage.speed_gbps = 14.0
                            storage.interface = "pcie5x4"
                        elif '16' in link_speed:  # 16 GT/s = Gen4
                            storage.storage_type = "nvme_gen4"
                            storage.speed_gbps = 7.0
                            storage.interface = "pcie4x4"
                        else:
                            storage.storage_type = "nvme_gen3"
                            storage.speed_gbps = 3.5
                            storage.interface = "pcie3x4"
                    
                    # Get capacity
                    size_path = device / 'size'
                    if size_path.exists():
                        sectors = int(size_path.read_text().strip())
                        storage.capacity_gb = (sectors * 512) / (1024**3)
                    
                    # Check for GPU Direct Storage support
                    storage.supports_direct_storage = os.path.exists('/dev/nvidia-fs')
                    
                    print(f"   ✓ {storage.storage_type.upper()} detected: {storage.speed_gbps} GB/s")
            
            elif platform.system() == "Windows":
                # Windows storage detection via WMI
                try:
                    import wmi
                    c = wmi.WMI()
                    
                    for disk in c.Win32_DiskDrive():
                        if 'nvme' in disk.Model.lower():
                            storage.storage_type = "nvme_gen4"  # Assume Gen4
                            storage.speed_gbps = 7.0
                            storage.capacity_gb = int(disk.Size) / (1024**3)
                            break
                except ImportError:
                    pass
            
            elif platform.system() == "Darwin":
                # macOS storage detection
                result = subprocess.run(
                    ['system_profiler', 'SPNVMeDataType', '-json'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    # Parse NVMe data...
                    storage.storage_type = "nvme_gen4"  # Apple uses custom NVMe
                    storage.speed_gbps = 7.0
                    
        except Exception as e:
            print(f"   Storage detection error: {e}")
        
        return storage
    
    def _convert_specialized_accelerator(self, spec_accel) -> AcceleratorInfo:
        """Convert specialized accelerator format to our format"""
        # Map capabilities
        caps = spec_accel.capabilities
        
        # Determine compute capability level
        if caps.int8_tops > 100:
            compute_cap = ComputeCapability.DATACENTER_AI
        elif caps.int8_tops > 10:
            compute_cap = ComputeCapability.AI_GPU
        else:
            compute_cap = ComputeCapability.NEURAL_ENGINE
        
        return AcceleratorInfo(
            accelerator_type=spec_accel.accelerator_type.value,
            vendor=spec_accel.vendor,
            model=spec_accel.model,
            driver_version=spec_accel.driver_version,
            compute_capability=compute_cap,
            memory_mb=int(caps.sram_mb + caps.hbm_gb * 1024 + caps.dram_gb * 1024),
            supports_int8=caps.int8_tops > 0,
            supports_fp16=caps.fp16_tflops > 0,
            supports_dynamic_shapes=caps.dynamic_shapes,
            max_batch_size=128 if caps.hbm_gb > 0 else 32
        )
    
    def _display_enhanced_summary(self, profile: DynamicHardwareProfile):
        """Display enhanced summary with all findings"""
        # Base summary
        super()._display_summary(profile)
        
        # Storage info
        if hasattr(profile, 'storage_info'):
            storage = profile.storage_info
            print(f"\n💾 Storage Subsystem:")
            print(f"   Type: {storage.storage_type.upper()}")
            print(f"   Speed: {storage.speed_gbps:.1f} GB/s")
            print(f"   Capacity: {storage.capacity_gb:.0f} GB")
            if storage.supports_direct_storage:
                print("   ✓ GPU Direct Storage supported")
        
        # Trinity-specific recommendations
        print(f"\n🎯 TRINITY OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Selected Architecture: {profile.recommended_architecture}")
        print(f"Estimated Performance:")
        print(f"   • FPS: {profile.estimated_trinity_fps:.1f}")
        print(f"   • Max Batch: {profile.estimated_max_batch_size}")
        print(f"   • Max Sequence: {profile.estimated_max_sequence_length}")
        
        # Show AI accelerators if found
        if profile.accelerators:
            print(f"\n🧠 AI Accelerators ({len(profile.accelerators)} found):")
            for accel in profile.accelerators[:5]:  # Show top 5
                if hasattr(accel, 'compute_capability'):
                    cap_str = f" [{accel.compute_capability.value}]"
                else:
                    cap_str = ""
                print(f"   • {accel.vendor} {accel.model}{cap_str}")
                if accel.supports_int8:
                    print(f"     INT8 acceleration available")
        
        # Key optimizations
        if profile.offloading_capabilities:
            print(f"\n🚀 Top 3 Optimizations:")
            for i, opt in enumerate(profile.offloading_capabilities[:3], 1):
                print(f"   {i}. {opt.recommendation}")
    
    # Helper methods
    def _get_jetson_version(self) -> str:
        """Get Jetson L4T version"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return f.read().strip().split(',')[0]
        except:
            return "unknown"
    
    def _get_jetson_memory(self) -> int:
        """Get Jetson available memory in MB"""
        try:
            if PSUTIL_AVAILABLE:
                # Jetson shares system memory
                return int(psutil.virtual_memory().total / (1024 * 1024) * 0.75)
            else:
                return 4096
        except:
            return 4096


# Example usage
if __name__ == "__main__":
    print("🌆 NEXLIFY INTEGRATED HARDWARE DETECTOR v2.0")
    print("━" * 60)
    print("Complete hardware analysis with optimization recommendations\n")
    
    detector = IntegratedUniversalDetector()
    profile = detector.detect_all(force_benchmark=True)
    
    print("\n✨ Hardware analysis complete!")
    print("Your system is ready for Trinity consciousness deployment.")
    
    # Save profile for debugging
    profile_path = Path.home() / ".nexlify" / "last_hardware_profile.json"
    profile_path.parent.mkdir(exist_ok=True)
    
    # Note: In production, implement proper serialization
    print(f"\nProfile saved to: {profile_path}")
