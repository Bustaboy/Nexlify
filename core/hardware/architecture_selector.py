# Location: nexlify/core/hardware/architecture_selector.py
# Trinity Architecture Selector - Maps Hardware Capabilities to Consciousness Configurations

"""
🏗️ TRINITY ARCHITECTURE SELECTOR v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maps measured hardware capabilities to optimal Trinity architectures.
Every configuration has a purpose, every architecture tells a story.

"The street finds its own uses for things." - William Gibson
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

from .universal_detector import DynamicHardwareProfile, ComputeCapability


@dataclass
class TrinityArchitecture:
    """Defines a Trinity consciousness architecture configuration"""
    name: str                           # Cyberpunk-themed identifier
    description: str                    # What makes this config special
    min_compute_tflops: float          # Minimum required compute
    min_memory_gb: float               # Minimum memory required
    min_bandwidth_gbps: float          # Minimum memory bandwidth
    supports_distributed: bool          # Can use multiple devices
    supports_npu: bool                 # Can leverage neural processors
    requires_tensor_cores: bool        # Needs tensor acceleration
    optimal_precision: str             # FP32, FP16, INT8
    consciousness_quality: float       # 0.0-1.0 quality factor
    power_tier: str                    # ultra, high, medium, low, emergency
    special_features: List[str]        # Unique capabilities


class ArchitectureSelector:
    """
    🎯 ARCHITECTURE SELECTOR
    
    Matches hardware capabilities to optimal Trinity configurations.
    No preconceptions - only measured truth guides selection.
    """
    
    def __init__(self):
        """Initialize with all available Trinity architectures"""
        self.architectures = self._define_architectures()
        print("🏗️ Architecture Selector initialized")
        print(f"   {len(self.architectures)} consciousness configurations available")
    
    def _define_architectures(self) -> List[TrinityArchitecture]:
        """
        Define all available Trinity architecture configurations.
        Ordered from most demanding to most accessible.
        """
        return [
            # === ULTRA TIER - The Chrome Elite ===
            TrinityArchitecture(
                name="ghost_protocol_mesh",
                description="Distributed consciousness across multiple high-end GPUs",
                min_compute_tflops=50.0,    # Multiple GPUs combined
                min_memory_gb=48.0,          # Combined VRAM
                min_bandwidth_gbps=1500.0,   # Aggregate bandwidth
                supports_distributed=True,
                supports_npu=False,
                requires_tensor_cores=True,
                optimal_precision="FP16",
                consciousness_quality=1.0,
                power_tier="ultra",
                special_features=["mesh_networking", "quantum_entanglement", "zero_latency_sync"]
            ),
            
            TrinityArchitecture(
                name="quantum_breach_ultra",
                description="Single ultra-tier GPU pushing consciousness boundaries",
                min_compute_tflops=25.0,     # RTX 4090 class
                min_memory_gb=24.0,
                min_bandwidth_gbps=900.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=True,
                optimal_precision="FP16",
                consciousness_quality=0.95,
                power_tier="ultra",
                special_features=["tensor_overdrive", "memory_overflow", "thermal_unlimited"]
            ),
            
            # === HIGH TIER - Professional Netrunners ===
            TrinityArchitecture(
                name="neon_oracle_pro",
                description="Professional consciousness for serious market prediction",
                min_compute_tflops=15.0,     # RTX 4070 Ti / 3080 Ti
                min_memory_gb=12.0,
                min_bandwidth_gbps=600.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=True,
                optimal_precision="FP16",
                consciousness_quality=0.85,
                power_tier="high",
                special_features=["market_prescience", "cascade_detection", "profit_maximizer"]
            ),
            
            TrinityArchitecture(
                name="chrome_horizon_hybrid",
                description="CPU+GPU hybrid leveraging all available silicon",
                min_compute_tflops=8.0,
                min_memory_gb=8.0,
                min_bandwidth_gbps=400.0,
                supports_distributed=True,   # CPU+GPU distribution
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP16",
                consciousness_quality=0.75,
                power_tier="high",
                special_features=["hybrid_compute", "adaptive_routing", "thermal_balance"]
            ),
            
            # === NEURAL TIER - Specialized Accelerators ===
            TrinityArchitecture(
                name="neural_chrome_adaptive",
                description="NPU-accelerated consciousness with dynamic adaptation",
                min_compute_tflops=2.0,      # NPUs measure differently
                min_memory_gb=4.0,
                min_bandwidth_gbps=100.0,
                supports_distributed=False,
                supports_npu=True,
                requires_tensor_cores=False,
                optimal_precision="INT8",
                consciousness_quality=0.7,
                power_tier="medium",
                special_features=["neural_engine", "power_efficient", "mobile_ready"]
            ),
            
            TrinityArchitecture(
                name="synaptic_mesh_lite",
                description="Distributed NPU mesh for edge consciousness",
                min_compute_tflops=1.0,
                min_memory_gb=2.0,
                min_bandwidth_gbps=50.0,
                supports_distributed=True,
                supports_npu=True,
                requires_tensor_cores=False,
                optimal_precision="INT8",
                consciousness_quality=0.65,
                power_tier="medium",
                special_features=["edge_computing", "swarm_intelligence", "low_latency"]
            ),
            
            # === MEDIUM TIER - Street Level ===
            TrinityArchitecture(
                name="street_samurai_standard",
                description="Mid-range GPU consciousness for everyday trading",
                min_compute_tflops=5.0,      # RTX 3060 / 4060 class
                min_memory_gb=6.0,
                min_bandwidth_gbps=300.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP32",
                consciousness_quality=0.6,
                power_tier="medium",
                special_features=["reliable", "thermal_stable", "cost_effective"]
            ),
            
            TrinityArchitecture(
                name="razor_edge_mobile",
                description="Laptop consciousness for netrunners on the move",
                min_compute_tflops=3.0,      # Mobile RTX class
                min_memory_gb=4.0,
                min_bandwidth_gbps=200.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP16",
                consciousness_quality=0.55,
                power_tier="medium",
                special_features=["battery_optimized", "thermal_limited", "portable"]
            ),
            
            # === LOW TIER - Entry Level ===
            TrinityArchitecture(
                name="digital_phantom_basic",
                description="Entry GPU consciousness - everyone starts somewhere",
                min_compute_tflops=1.5,      # GTX 1660 / RTX 3050 class
                min_memory_gb=4.0,
                min_bandwidth_gbps=150.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP32",
                consciousness_quality=0.45,
                power_tier="low",
                special_features=["entry_level", "upgradeable", "learning_mode"]
            ),
            
            TrinityArchitecture(
                name="wetware_enhanced_cpu",
                description="CPU-only with SIMD acceleration - biological neural patterns",
                min_compute_tflops=0.5,      # Modern CPU with AVX
                min_memory_gb=8.0,           # System RAM
                min_bandwidth_gbps=50.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP32",
                consciousness_quality=0.35,
                power_tier="low",
                special_features=["cpu_optimized", "simd_accelerated", "always_available"]
            ),
            
            # === EMERGENCY TIER - When All Else Fails ===
            TrinityArchitecture(
                name="emergency_wetware_basic",
                description="Minimal consciousness - but consciousness nonetheless",
                min_compute_tflops=0.1,      # Any CPU
                min_memory_gb=4.0,
                min_bandwidth_gbps=20.0,
                supports_distributed=False,
                supports_npu=False,
                requires_tensor_cores=False,
                optimal_precision="FP32",
                consciousness_quality=0.2,
                power_tier="emergency",
                special_features=["universal_compatible", "survival_mode", "basic_awareness"]
            ),
            
            # === EXPERIMENTAL TIER - Future Tech ===
            TrinityArchitecture(
                name="quantum_entangled_experimental",
                description="Experimental quantum-classical hybrid consciousness",
                min_compute_tflops=100.0,    # Theoretical quantum advantage
                min_memory_gb=64.0,
                min_bandwidth_gbps=2000.0,
                supports_distributed=True,
                supports_npu=True,
                requires_tensor_cores=True,
                optimal_precision="QBIT",    # Quantum precision
                consciousness_quality=2.0,   # Beyond current scale
                power_tier="experimental",
                special_features=["quantum_supremacy", "parallel_universes", "prescient_overflow"]
            )
        ]
    
    def select_architecture(self, profile: DynamicHardwareProfile) -> Tuple[TrinityArchitecture, str]:
        """
        Select optimal Trinity architecture based on measured capabilities.
        
        Returns:
            - Selected architecture
            - Reasoning for selection
        """
        # Calculate aggregate metrics
        total_compute_tflops = self._calculate_total_compute(profile)
        total_memory_gb = self._calculate_total_memory(profile)
        max_bandwidth_gbps = self._calculate_max_bandwidth(profile)
        has_multiple_gpus = len([p for p in profile.compute_profiles.items() 
                               if 'cuda' in p[0]]) > 1
        has_npu = len(profile.accelerators) > 0
        has_tensor_cores = self._has_tensor_cores(profile)
        
        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Total compute: {total_compute_tflops:.1f} TFLOPS")
        reasoning_parts.append(f"Total memory: {total_memory_gb:.1f} GB")
        reasoning_parts.append(f"Peak bandwidth: {max_bandwidth_gbps:.0f} GB/s")
        
        if has_multiple_gpus:
            reasoning_parts.append("Multiple GPUs detected - mesh capable")
        if has_npu:
            reasoning_parts.append(f"{len(profile.accelerators)} neural accelerator(s) found")
        if has_tensor_cores:
            reasoning_parts.append("Tensor cores available for acceleration")
        
        # Score each architecture
        best_architecture = None
        best_score = -1
        
        for arch in self.architectures:
            score = self._score_architecture(
                arch, 
                total_compute_tflops,
                total_memory_gb,
                max_bandwidth_gbps,
                has_multiple_gpus,
                has_npu,
                has_tensor_cores
            )
            
            if score > best_score:
                best_score = score
                best_architecture = arch
        
        # Add selection reasoning
        reasoning_parts.append(f"\nSelected: {best_architecture.name}")
        reasoning_parts.append(f"Quality factor: {best_architecture.consciousness_quality:.0%}")
        reasoning_parts.append(f"Power tier: {best_architecture.power_tier.upper()}")
        
        if best_architecture.special_features:
            reasoning_parts.append(f"Features: {', '.join(best_architecture.special_features)}")
        
        reasoning = "\n".join(reasoning_parts)
        
        return best_architecture, reasoning
    
    def _calculate_total_compute(self, profile: DynamicHardwareProfile) -> float:
        """Calculate total available compute in TFLOPS"""
        total = 0.0
        
        for device, compute in profile.compute_profiles.items():
            # Use the best measured performance
            best_gflops = max(
                compute.matrix_multiply_gflops,
                compute.convolution_gflops,
                compute.attention_gflops,
                compute.flops_fp32 / 1e9
            )
            total += best_gflops / 1000  # Convert to TFLOPS
        
        # Add NPU compute estimates
        for accel in profile.accelerators:
            if accel.compute_capability == ComputeCapability.NEURAL_ENGINE:
                # NPUs often rated in TOPS for INT8
                if accel.supports_int8:
                    total += 2.0  # Conservative NPU estimate
        
        # Apply virtualization overhead
        if profile.virtualization.is_virtualized:
            total /= profile.virtualization.overhead_factor
        
        return total
    
    def _calculate_total_memory(self, profile: DynamicHardwareProfile) -> float:
        """Calculate total available memory in GB"""
        total = 0.0
        
        # Device memory (GPUs)
        for device, memory in profile.device_memory.items():
            total += memory.total_gb
        
        # If no GPU memory, use system memory
        if total == 0:
            total = profile.system_memory.available_gb
        
        # Add accelerator memory
        for accel in profile.accelerators:
            total += accel.memory_mb / 1024
        
        return total
    
    def _calculate_max_bandwidth(self, profile: DynamicHardwareProfile) -> float:
        """Calculate maximum memory bandwidth in GB/s"""
        max_bandwidth = profile.system_memory.bandwidth_gbps
        
        # Check device memory bandwidth
        for device, memory in profile.device_memory.items():
            max_bandwidth = max(max_bandwidth, memory.bandwidth_gbps)
        
        return max_bandwidth
    
    def _has_tensor_cores(self, profile: DynamicHardwareProfile) -> bool:
        """Check if any device has tensor cores"""
        for device, compute in profile.compute_profiles.items():
            if 'cuda' in device:
                # Heuristic: If FP16 is >2x faster than FP32, likely has tensor cores
                if compute.flops_fp16 > compute.flops_fp32 * 2:
                    return True
        return False
    
    def _score_architecture(
        self,
        arch: TrinityArchitecture,
        compute_tflops: float,
        memory_gb: float,
        bandwidth_gbps: float,
        has_multiple_gpus: bool,
        has_npu: bool,
        has_tensor_cores: bool
    ) -> float:
        """
        Score how well hardware matches architecture requirements.
        Higher score = better match.
        """
        # Hard requirements check
        if compute_tflops < arch.min_compute_tflops:
            return -1
        if memory_gb < arch.min_memory_gb:
            return -1
        if bandwidth_gbps < arch.min_bandwidth_gbps:
            return -1
        if arch.supports_distributed and not has_multiple_gpus and not arch.supports_npu:
            return -1
        if arch.supports_npu and not has_npu:
            return -1
        if arch.requires_tensor_cores and not has_tensor_cores:
            return -1
        
        # Calculate match score (0-100)
        score = 0.0
        
        # Compute match (don't over-provision too much)
        compute_ratio = compute_tflops / arch.min_compute_tflops
        if compute_ratio >= 1.0 and compute_ratio <= 2.0:
            score += 40  # Perfect match
        elif compute_ratio > 2.0:
            score += 40 - (compute_ratio - 2.0) * 5  # Penalty for over-provisioning
        
        # Memory match
        memory_ratio = memory_gb / arch.min_memory_gb
        if memory_ratio >= 1.0 and memory_ratio <= 2.0:
            score += 30
        elif memory_ratio > 2.0:
            score += 30 - (memory_ratio - 2.0) * 3
        
        # Bandwidth match
        bandwidth_ratio = bandwidth_gbps / arch.min_bandwidth_gbps
        if bandwidth_ratio >= 1.0 and bandwidth_ratio <= 2.0:
            score += 20
        elif bandwidth_ratio > 2.0:
            score += 20 - (bandwidth_ratio - 2.0) * 2
        
        # Bonus for special capabilities match
        if arch.supports_distributed and has_multiple_gpus:
            score += 5
        if arch.supports_npu and has_npu:
            score += 5
        
        # Prefer higher quality architectures when hardware allows
        score += arch.consciousness_quality * 10
        
        return score
    
    def estimate_performance(
        self,
        profile: DynamicHardwareProfile,
        architecture: TrinityArchitecture
    ) -> Dict[str, float]:
        """
        Estimate Trinity performance with selected architecture.
        
        Returns performance metrics based on actual benchmarks.
        """
        # Base estimates from architecture quality
        base_fps = 30 * architecture.consciousness_quality
        base_batch = int(8 * architecture.consciousness_quality)
        base_seq_len = int(512 * architecture.consciousness_quality)
        
        # Adjust based on actual measured performance
        compute_tflops = self._calculate_total_compute(profile)
        memory_gb = self._calculate_total_memory(profile)
        
        # FPS scales with compute
        compute_multiplier = min(2.0, compute_tflops / architecture.min_compute_tflops)
        fps = base_fps * compute_multiplier
        
        # Batch size scales with memory
        memory_multiplier = min(2.0, memory_gb / architecture.min_memory_gb)
        batch_size = int(base_batch * memory_multiplier)
        
        # Sequence length affected by both
        combined_multiplier = (compute_multiplier + memory_multiplier) / 2
        seq_length = int(base_seq_len * combined_multiplier)
        
        # Apply precision benefits
        if architecture.optimal_precision == "INT8":
            fps *= 1.5
            batch_size = int(batch_size * 1.3)
        elif architecture.optimal_precision == "FP16":
            fps *= 1.2
            batch_size = int(batch_size * 1.1)
        
        return {
            "trinity_fps": fps,
            "max_batch_size": batch_size,
            "max_sequence_length": seq_length,
            "consciousness_quality": architecture.consciousness_quality,
            "power_efficiency": self._estimate_power_efficiency(profile, architecture)
        }
    
    def _estimate_power_efficiency(
        self,
        profile: DynamicHardwareProfile,
        architecture: TrinityArchitecture
    ) -> float:
        """Estimate operations per watt efficiency"""
        # This would use actual TDP measurements in production
        total_tdp = 150.0  # Placeholder
        total_compute = self._calculate_total_compute(profile)
        
        # TFLOPS per watt
        return (total_compute * 1000) / total_tdp  # GFLOPS/W


# Example usage
if __name__ == "__main__":
    from .universal_detector import UniversalHardwareDetector
    
    print("🌆 TRINITY ARCHITECTURE SELECTOR TEST")
    print("━" * 60)
    
    # Detect hardware
    detector = UniversalHardwareDetector()
    profile = detector.detect_all()
    
    # Select architecture
    selector = ArchitectureSelector()
    architecture, reasoning = selector.select_architecture(profile)
    
    print(f"\n📊 Hardware Analysis:")
    print(reasoning)
    
    # Estimate performance
    performance = selector.estimate_performance(profile, architecture)
    print(f"\n🎯 Performance Estimates:")
    print(f"   Trinity FPS: {performance['trinity_fps']:.1f}")
    print(f"   Max Batch Size: {performance['max_batch_size']}")
    print(f"   Max Sequence Length: {performance['max_sequence_length']}")
    print(f"   Consciousness Quality: {performance['consciousness_quality']:.0%}")
    print(f"   Power Efficiency: {performance['power_efficiency']:.1f} GFLOPS/W")
