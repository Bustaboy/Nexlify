# Location: nexlify/core/hardware/integrated_architecture_selector.py
# Integrated Architecture Selector - Considers Offloading Potential

"""
🏗️ INTEGRATED ARCHITECTURE SELECTOR v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Selects optimal Trinity architecture considering offloading capabilities.
Raw specs are just the beginning - synergy unlocks true potential.

"The whole is greater than the sum of its parts." - Aristotle
"Unless those parts can offload to each other." - Night City Proverb
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import re

from .architecture_selector import TrinityArchitecture, ArchitectureSelector
from .offloading_detector import OffloadingDetector, DeviceCapabilities
from .universal_detector import DynamicHardwareProfile


@dataclass
class HardwareFingerprint:
    """Specific hardware configuration fingerprint"""
    gpu_models: List[str] = field(default_factory=list)
    cpu_model: str = ""
    npu_models: List[str] = field(default_factory=list)
    total_vram_gb: float = 0.0
    total_ram_gb: float = 0.0
    storage_type: str = "unknown"  # "nvme_gen5", "nvme_gen4", "nvme_gen3", "ssd", "hdd"
    storage_speed_gbps: float = 0.0
    
    def matches_pattern(self, pattern: str) -> bool:
        """Check if hardware matches a known pattern"""
        # GPU patterns
        for gpu in self.gpu_models:
            if pattern.lower() in gpu.lower():
                return True
        
        # CPU patterns
        if pattern.lower() in self.cpu_model.lower():
            return True
            
        return False


@dataclass
class OffloadingBonus:
    """Architecture tier bonus from offloading capabilities"""
    tier_upgrade: int = 0           # How many tiers to upgrade
    compute_multiplier: float = 1.0 # Effective compute boost
    memory_multiplier: float = 1.0  # Effective memory boost
    special_features: Set[str] = field(default_factory=set)
    reasoning: str = ""


class IntegratedArchitectureSelector(ArchitectureSelector):
    """
    🎯 INTEGRATED SELECTOR
    
    Considers both raw capabilities and offloading potential.
    Specific hardware combinations unlock hidden potential.
    """
    
    def __init__(self):
        """Initialize with base architectures and known synergies"""
        super().__init__()
        self.offload_detector = OffloadingDetector()
        self.known_synergies = self._define_known_synergies()
        print("🏗️ Integrated Architecture Selector initialized")
        print(f"   Tracking {len(self.known_synergies)} hardware synergies")
    
    def select_architecture(
        self, 
        profile: DynamicHardwareProfile
    ) -> Tuple[TrinityArchitecture, str]:
        """
        Select architecture considering offloading potential.
        
        This overrides the base method to include offloading analysis.
        """
        # Step 1: Get base capabilities
        base_compute = self._calculate_total_compute(profile)
        base_memory = self._calculate_total_memory(profile)
        base_bandwidth = self._calculate_max_bandwidth(profile)
        
        # Step 2: Extract hardware fingerprint
        fingerprint = self._extract_fingerprint(profile)
        
        # Step 3: Detect offloading opportunities
        offload_ops = self.offload_detector.detect_opportunities(profile)
        
        # Step 4: Calculate offloading bonuses
        bonus = self._calculate_offloading_bonus(
            fingerprint, 
            offload_ops,
            profile
        )
        
        # Step 5: Apply bonuses to effective capabilities
        effective_compute = base_compute * bonus.compute_multiplier
        effective_memory = base_memory * bonus.memory_multiplier
        effective_bandwidth = base_bandwidth  # Bandwidth doesn't multiply
        
        # Step 6: Check for specific hardware synergies
        synergy_bonus = self._check_hardware_synergies(fingerprint, profile)
        if synergy_bonus:
            bonus = self._merge_bonuses(bonus, synergy_bonus)
            effective_compute *= synergy_bonus.compute_multiplier
            effective_memory *= synergy_bonus.memory_multiplier
        
        # Step 7: Select architecture with effective capabilities
        architecture, base_reasoning = self._select_with_effective_caps(
            effective_compute,
            effective_memory,
            effective_bandwidth,
            profile,
            bonus
        )
        
        # Step 8: Build comprehensive reasoning
        reasoning = self._build_reasoning(
            base_compute,
            base_memory,
            base_bandwidth,
            effective_compute,
            effective_memory,
            bonus,
            architecture,
            fingerprint,
            base_reasoning
        )
        
        return architecture, reasoning
    
    def _define_known_synergies(self) -> List[Tuple[str, OffloadingBonus]]:
        """Define known hardware combinations with special synergies"""
        return [
            # RTX 50 series (Blackwell) + high RAM
            (r"RTX 50[789]0.*RAM:(?:32|64|128)", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.0,
                memory_multiplier=2.5,
                special_features={"unified_memory_virtualization", "blackwell_cache"},
                reasoning="RTX 50 series massive L2 cache + high RAM enables superior tiering"
            )),
            
            # RTX 40 series (Ada) + high RAM
            (r"RTX 40[789]0.*RAM:(?:32|64|128)", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.0,
                memory_multiplier=2.2,
                special_features={"unified_memory_virtualization", "ada_streaming"},
                reasoning="RTX 40 series AV1 + streaming multiprocessors excel at offloading"
            )),
            
            # RTX 30 series + high RAM
            (r"RTX 30[789]0.*RAM:(?:32|64|128)", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.0,
                memory_multiplier=2.0,
                special_features={"unified_memory_virtualization"},
                reasoning="RTX 30 series + high RAM enables CUDA unified memory"
            )),
            
            # RTX 20 series (Turing) + high RAM
            (r"RTX 20[78]0.*RAM:(?:32|64|128)", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.0,
                memory_multiplier=1.8,
                special_features={"rt_cores_v1", "unified_memory"},
                reasoning="RTX 20 series pioneered RT cores, good memory offloading"
            )),
            
            # GTX 10 series (Pascal) + high RAM
            (r"GTX 10[78]0.*RAM:(?:32|64|128)", OffloadingBonus(
                tier_upgrade=0,
                compute_multiplier=1.0,
                memory_multiplier=1.5,
                special_features={"pascal_efficiency"},
                reasoning="GTX 10 series still capable with memory tiering"
            )),
            
            # Dual GPU same model
            (r"DUAL:RTX.*SAME", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.85,
                memory_multiplier=1.9,
                special_features={"nvlink_capable", "perfect_load_balance"},
                reasoning="Identical GPUs enable near-perfect scaling"
            )),
            
            # MacBook Pro M-series
            (r"Apple M[123].*Neural Engine", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.3,
                memory_multiplier=1.0,
                special_features={"unified_memory", "zero_copy_transfer"},
                reasoning="Apple Silicon unified memory eliminates transfer overhead"
            )),
            
            # AMD CPU + AMD GPU
            (r"AMD.*Ryzen.*AMD.*Radeon", OffloadingBonus(
                tier_upgrade=0,
                compute_multiplier=1.15,
                memory_multiplier=1.1,
                special_features={"smart_access_memory"},
                reasoning="AMD Smart Access Memory provides CPU-GPU optimization"
            )),
            
            # Intel Arc + Intel CPU
            (r"Intel.*Core.*Intel.*Arc", OffloadingBonus(
                tier_upgrade=0,
                compute_multiplier=1.2,
                memory_multiplier=1.0,
                special_features={"deep_link", "xe_matrix_engines"},
                reasoning="Intel Deep Link enables efficient CPU-GPU collaboration"
            )),
            
            # Laptop + eGPU
            (r"mobile.*external.*GPU", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.5,
                memory_multiplier=1.3,
                special_features={"thunderbolt_expansion"},
                reasoning="External GPU transforms laptop into desktop-class"
            )),
            
            # NPU + GPU combo
            (r"(Neural Engine|NPU|AIE).*GPU", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.4,
                memory_multiplier=1.0,
                special_features={"hybrid_ai_acceleration"},
                reasoning="NPU handles INT8 while GPU focuses on FP16/32"
            )),
            
            # High-end CPU + mid-range GPU
            (r"(i9|Ryzen 9|Threadripper).*RTX [34]0[67]0", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.25,
                memory_multiplier=1.0,
                special_features={"cpu_compute_assist"},
                reasoning="Powerful CPU can assist GPU with preprocessing"
            )),
            
            # Cloud/Datacenter hardware
            (r"(Tesla|A100|H100|MI\d00)", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.0,
                memory_multiplier=1.5,
                special_features={"nvlink", "infinity_fabric", "datacenter_optimized"},
                reasoning="Datacenter GPUs have optimized interconnects"
            )),
            
            # NVMe Gen5 + limited VRAM
            (r"nvme_gen5.*VRAM:[4-8]", OffloadingBonus(
                tier_upgrade=2,
                compute_multiplier=1.0,
                memory_multiplier=3.0,
                special_features={"lightning_swap", "direct_storage"},
                reasoning="PCIe 5.0 NVMe (14GB/s) enables RAM-speed memory tiering"
            )),
            
            # NVMe Gen4 + limited VRAM
            (r"nvme_gen4.*VRAM:[4-8]", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.0,
                memory_multiplier=2.5,
                special_features={"fast_swap", "direct_storage"},
                reasoning="PCIe 4.0 NVMe (7GB/s) significantly reduces swap penalty"
            )),
            
            # Multi-NVMe RAID setup
            (r"RAID.*nvme|nvme.*RAID", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.0,
                memory_multiplier=2.0,
                special_features={"raid_swap", "parallel_io"},
                reasoning="RAID NVMe array provides massive swap bandwidth"
            )),
            
            # Low VRAM + fast storage combo
            (r"VRAM:[2-6].*nvme", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.0,
                memory_multiplier=2.2,
                special_features={"aggressive_tiering"},
                reasoning="Fast storage compensates for limited VRAM"
            )),
            
            # Multi-NPU setup
            (r"MULTI:.*NPU|Neural.*Engine.*\d+", OffloadingBonus(
                tier_upgrade=1,
                compute_multiplier=1.6,
                memory_multiplier=1.0,
                special_features={"neural_mesh_network"},
                reasoning="Multiple NPUs create efficient inference mesh"
            ))
        ]
        ]
    
    def _extract_fingerprint(self, profile: DynamicHardwareProfile) -> HardwareFingerprint:
        """Extract specific hardware models from profile"""
        fingerprint = HardwareFingerprint()
        
        # GPU models
        gpu_models = []
        for device_id, compute in profile.compute_profiles.items():
            if device_id.startswith("cuda"):
                # Try to extract GPU name from profile
                # In real implementation, this would come from detection
                gpu_models.append(f"GPU_{device_id}")
        
        # Get from fingerprint if available
        if hasattr(profile.fingerprint, 'gpu_models'):
            gpu_models = profile.fingerprint.gpu_models
        
        fingerprint.gpu_models = gpu_models
        
        # CPU model
        if hasattr(profile.fingerprint, 'cpu_model'):
            fingerprint.cpu_model = profile.fingerprint.cpu_model
        
        # NPU models
        fingerprint.npu_models = [
            f"{a.vendor}_{a.model}" for a in profile.accelerators
        ]
        
        # Memory sizes
        fingerprint.total_vram_gb = sum(
            m.total_gb for d, m in profile.device_memory.items()
            if d.startswith("cuda")
        )
        fingerprint.total_ram_gb = profile.system_memory.total_gb
        
        # Storage info (would come from actual detection)
        if hasattr(profile, 'storage_info'):
            fingerprint.storage_type = profile.storage_info.storage_type
            fingerprint.storage_speed_gbps = profile.storage_info.speed_gbps
        else:
            # Estimate based on system age/class
            if any("50" in gpu for gpu in fingerprint.gpu_models):
                fingerprint.storage_type = "nvme_gen5"
                fingerprint.storage_speed_gbps = 14.0
            elif any("40" in gpu for gpu in fingerprint.gpu_models):
                fingerprint.storage_type = "nvme_gen4"
                fingerprint.storage_speed_gbps = 7.0
            else:
                fingerprint.storage_type = "nvme_gen3"
                fingerprint.storage_speed_gbps = 3.5
        
        return fingerprint
    
    def _calculate_offloading_bonus(
        self,
        fingerprint: HardwareFingerprint,
        offload_ops: List,
        profile: DynamicHardwareProfile
    ) -> OffloadingBonus:
        """Calculate architecture bonus from offloading capabilities"""
        bonus = OffloadingBonus()
        
        # Analyze offloading opportunities
        max_speedup = 1.0
        has_memory_tiering = False
        has_multi_gpu = False
        has_npu_offload = False
        has_hybrid_compute = False
        
        for op in offload_ops:
            max_speedup = max(max_speedup, op.speedup_potential)
            
            if op.task_type == "memory_tiering":
                has_memory_tiering = True
            elif op.task_type == "model_parallel":
                has_multi_gpu = True
            elif op.task_type == "int8_inference":
                has_npu_offload = True
            elif op.task_type == "hybrid_compute":
                has_hybrid_compute = True
        
        # Calculate bonuses
        if max_speedup > 2.0:
            bonus.compute_multiplier = min(max_speedup * 0.7, 2.5)  # 70% realized
            bonus.tier_upgrade += 1
        
        if has_memory_tiering and fingerprint.total_vram_gb < 12:
            # Memory tiering is huge for VRAM-limited systems
            bonus.memory_multiplier = 1.5
            bonus.special_features.add("memory_virtualization")
            
            # Scale bonus based on storage speed
            if fingerprint.storage_type == "nvme_gen5":
                bonus.memory_multiplier *= 1.5  # 14GB/s is incredible
                bonus.special_features.add("lightning_tier")
            elif fingerprint.storage_type == "nvme_gen4":
                bonus.memory_multiplier *= 1.3  # 7GB/s is very good
                bonus.special_features.add("fast_tier")
            elif fingerprint.storage_type == "nvme_gen3":
                bonus.memory_multiplier *= 1.1  # 3.5GB/s is decent
            
            # Massive RAM bonus
            if fingerprint.total_ram_gb > 32:
                bonus.tier_upgrade += 1
            if fingerprint.total_ram_gb > 64:
                bonus.memory_multiplier *= 1.2
        
        if has_multi_gpu:
            gpu_count = len(fingerprint.gpu_models)
            if gpu_count >= 2:
                bonus.compute_multiplier *= (1 + (gpu_count - 1) * 0.7)
                bonus.special_features.add("multi_gpu_mesh")
                if gpu_count >= 4:
                    bonus.tier_upgrade += 1
        
        if has_npu_offload:
            bonus.compute_multiplier *= 1.2
            bonus.special_features.add("neural_acceleration")
        
        if has_hybrid_compute:
            bonus.compute_multiplier *= 1.15
            bonus.special_features.add("heterogeneous_compute")
        
        # Build reasoning
        reasons = []
        if bonus.compute_multiplier > 1.0:
            reasons.append(f"Offloading enables {bonus.compute_multiplier:.1f}x effective compute")
        if bonus.memory_multiplier > 1.0:
            reasons.append(f"Memory tiering provides {bonus.memory_multiplier:.1f}x effective memory")
        if bonus.tier_upgrade > 0:
            reasons.append(f"Synergies unlock {bonus.tier_upgrade} tier upgrade")
        
        bonus.reasoning = "; ".join(reasons)
        
        return bonus
    
    def _check_hardware_synergies(
        self,
        fingerprint: HardwareFingerprint,
        profile: DynamicHardwareProfile
    ) -> Optional[OffloadingBonus]:
        """Check for known hardware synergies"""
        # Build pattern string for matching
        pattern_parts = []
        
        # Add GPU info
        if len(fingerprint.gpu_models) > 1:
            if len(set(fingerprint.gpu_models)) == 1:
                pattern_parts.append(f"DUAL:{fingerprint.gpu_models[0]}:SAME")
            else:
                pattern_parts.append(f"MULTI:GPU:{len(fingerprint.gpu_models)}")
        elif fingerprint.gpu_models:
            pattern_parts.append(fingerprint.gpu_models[0])
        
        # Add CPU info
        pattern_parts.append(fingerprint.cpu_model)
        
        # Add RAM info
        pattern_parts.append(f"RAM:{int(fingerprint.total_ram_gb)}")
        
        # Add VRAM info for limited VRAM detection
        if fingerprint.total_vram_gb > 0:
            pattern_parts.append(f"VRAM:{int(fingerprint.total_vram_gb)}")
        
        # Add storage info
        pattern_parts.append(fingerprint.storage_type)
        
        # Add NPU info
        if fingerprint.npu_models:
            if len(fingerprint.npu_models) > 1:
                pattern_parts.append(f"MULTI:NPU:{len(fingerprint.npu_models)}")
            else:
                pattern_parts.append(fingerprint.npu_models[0])
        
        # Check if mobile/laptop
        if "mobile" in fingerprint.cpu_model.lower() or fingerprint.total_vram_gb < 8:
            pattern_parts.append("mobile")
        
        # Create combined pattern
        hardware_string = " ".join(pattern_parts)
        
        # Check against known synergies
        for pattern, synergy_bonus in self.known_synergies:
            if re.search(pattern, hardware_string, re.IGNORECASE):
                return synergy_bonus
        
        return None
    
    def _merge_bonuses(
        self,
        bonus1: OffloadingBonus,
        bonus2: OffloadingBonus
    ) -> OffloadingBonus:
        """Merge two bonuses, taking the best of each"""
        merged = OffloadingBonus()
        
        merged.tier_upgrade = max(bonus1.tier_upgrade, bonus2.tier_upgrade)
        merged.compute_multiplier = max(bonus1.compute_multiplier, bonus2.compute_multiplier)
        merged.memory_multiplier = max(bonus1.memory_multiplier, bonus2.memory_multiplier)
        merged.special_features = bonus1.special_features | bonus2.special_features
        
        reasons = []
        if bonus1.reasoning:
            reasons.append(bonus1.reasoning)
        if bonus2.reasoning:
            reasons.append(bonus2.reasoning)
        merged.reasoning = " + ".join(reasons)
        
        return merged
    
    def _select_with_effective_caps(
        self,
        effective_compute: float,
        effective_memory: float,
        effective_bandwidth: float,
        profile: DynamicHardwareProfile,
        bonus: OffloadingBonus
    ) -> Tuple[TrinityArchitecture, str]:
        """Select architecture using effective capabilities"""
        # Get device analysis
        has_multiple_gpus = len([p for p in profile.compute_profiles.items() 
                               if 'cuda' in p[0]]) > 1
        has_npu = len(profile.accelerators) > 0
        has_tensor_cores = self._has_tensor_cores(profile)
        
        # Score each architecture with effective capabilities
        best_architecture = None
        best_score = -1
        runner_up = None
        runner_up_score = -1
        
        for arch in self.architectures:
            # Skip experimental unless we have insane hardware
            if arch.power_tier == "experimental" and effective_compute < 80:
                continue
            
            score = self._score_architecture(
                arch,
                effective_compute,
                effective_memory,
                effective_bandwidth,
                has_multiple_gpus or "multi_gpu_mesh" in bonus.special_features,
                has_npu or "neural_acceleration" in bonus.special_features,
                has_tensor_cores
            )
            
            # Apply tier upgrade bonus to score
            if bonus.tier_upgrade > 0:
                score += bonus.tier_upgrade * 20
            
            if score > best_score:
                runner_up = best_architecture
                runner_up_score = best_score
                best_score = score
                best_architecture = arch
            elif score > runner_up_score:
                runner_up = best_architecture
                runner_up_score = score
        
        # Build reasoning about why this architecture
        reasoning_parts = []
        if runner_up and best_score - runner_up_score < 10:
            reasoning_parts.append(
                f"Close call between {best_architecture.name} and {runner_up.name}"
            )
        
        reasoning = "\n".join(reasoning_parts)
        
        return best_architecture, reasoning
    
    def _build_reasoning(
        self,
        base_compute: float,
        base_memory: float,
        base_bandwidth: float,
        effective_compute: float,
        effective_memory: float,
        bonus: OffloadingBonus,
        architecture: TrinityArchitecture,
        fingerprint: HardwareFingerprint,
        additional_reasoning: str
    ) -> str:
        """Build comprehensive reasoning for architecture selection"""
        parts = []
        
        # Base capabilities
        parts.append("=== BASE HARDWARE CAPABILITIES ===")
        parts.append(f"Raw compute: {base_compute:.1f} TFLOPS")
        parts.append(f"Raw memory: {base_memory:.1f} GB")
        parts.append(f"Peak bandwidth: {base_bandwidth:.0f} GB/s")
        
        # Hardware specifics
        if fingerprint.gpu_models:
            parts.append(f"GPUs: {', '.join(fingerprint.gpu_models)}")
        if fingerprint.npu_models:
            parts.append(f"NPUs: {', '.join(fingerprint.npu_models)}")
        
        # Offloading impact
        if bonus.compute_multiplier > 1.0 or bonus.memory_multiplier > 1.0:
            parts.append("\n=== OFFLOADING ENHANCEMENTS ===")
            if effective_compute != base_compute:
                parts.append(f"Effective compute: {effective_compute:.1f} TFLOPS "
                           f"({bonus.compute_multiplier:.1f}x via offloading)")
            if effective_memory != base_memory:
                parts.append(f"Effective memory: {effective_memory:.1f} GB "
                           f"({bonus.memory_multiplier:.1f}x via tiering)")
            
            if bonus.special_features:
                parts.append(f"Enabled features: {', '.join(bonus.special_features)}")
            
            if bonus.reasoning:
                parts.append(f"Synergy bonus: {bonus.reasoning}")
        
        # Architecture selection
        parts.append(f"\n=== SELECTED ARCHITECTURE ===")
        parts.append(f"Architecture: {architecture.name}")
        parts.append(f"Description: {architecture.description}")
        parts.append(f"Quality factor: {architecture.consciousness_quality:.0%}")
        parts.append(f"Power tier: {architecture.power_tier.upper()}")
        
        if architecture.special_features:
            parts.append(f"Features: {', '.join(architecture.special_features[:3])}")
        
        # Additional reasoning
        if additional_reasoning:
            parts.append(f"\n{additional_reasoning}")
        
        # Upgrade potential
        if bonus.tier_upgrade > 0:
            parts.append(f"\n🚀 Offloading unlocked {bonus.tier_upgrade} tier upgrade!")
        
        return "\n".join(parts)


# Example usage
if __name__ == "__main__":
    from .universal_detector import UniversalHardwareDetector
    
    print("🌆 INTEGRATED ARCHITECTURE SELECTOR TEST")
    print("━" * 60)
    
    # Detect hardware
    detector = UniversalHardwareDetector()
    profile = detector.detect_all()
    
    # Select architecture with offloading awareness
    selector = IntegratedArchitectureSelector()
    architecture, reasoning = selector.select_architecture(profile)
    
    print(f"\n{reasoning}")
    
    # Show performance estimates
    performance = selector.estimate_performance(profile, architecture)
    print(f"\n🎯 Performance Estimates:")
    print(f"   Trinity FPS: {performance['trinity_fps']:.1f}")
    print(f"   Max Batch Size: {performance['max_batch_size']}")
    print(f"   Max Sequence Length: {performance['max_sequence_length']}")
    
    # Show how offloading changed the selection
    base_selector = ArchitectureSelector()
    base_arch, _ = base_selector.select_architecture(profile)
    if base_arch.name != architecture.name:
        print(f"\n✨ Offloading upgraded selection from '{base_arch.name}' to '{architecture.name}'!")
