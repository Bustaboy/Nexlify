# Location: nexlify/core/hardware/offloading_detector.py
# Offloading Opportunities Detector - Dynamic Task Distribution System

"""
🔄 OFFLOADING OPPORTUNITIES DETECTOR v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identifies bottlenecks and suggests optimal task distribution.
Every chip has strengths - use them all.

"The future is already here – it's just not evenly distributed." - William Gibson
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .universal_detector import (
    DynamicHardwareProfile, 
    ComputeProfile, 
    MemoryProfile,
    OffloadingCapability,
    AcceleratorInfo
)


class TaskType(Enum):
    """Types of computational tasks in Trinity"""
    MATRIX_MULTIPLY = "matrix_multiply"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    EMBEDDING_LOOKUP = "embedding_lookup"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    MEMORY_INTENSIVE = "memory_intensive"
    LATENCY_CRITICAL = "latency_critical"
    BATCH_PARALLEL = "batch_parallel"
    STREAMING = "streaming"


class BottleneckType(Enum):
    """Types of performance bottlenecks"""
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    BANDWIDTH_BOUND = "bandwidth_bound"
    THERMAL_THROTTLE = "thermal_throttle"
    LATENCY_BOUND = "latency_bound"
    CAPACITY_LIMIT = "capacity_limit"


@dataclass
class DeviceCapabilities:
    """Normalized device capabilities for comparison"""
    device_id: str
    compute_score: float        # Normalized 0-100
    memory_score: float         # Normalized 0-100
    bandwidth_score: float      # Normalized 0-100
    latency_score: float        # Normalized 0-100 (lower is better)
    thermal_headroom: float     # 0-100 (100 = cool, 0 = throttling)
    special_capabilities: Set[str]  # "tensor_cores", "int8", "neural_engine", etc.


@dataclass
class TaskProfile:
    """Profile of a computational task's requirements"""
    task_type: TaskType
    compute_intensity: float    # FLOPS per byte
    memory_footprint_mb: float  # Memory required
    bandwidth_requirement: float # GB/s needed
    latency_sensitivity: float  # 0-1 (1 = very sensitive)
    parallelizable: bool        # Can split across devices
    precision_flexible: bool    # Can use INT8/FP16


class OffloadingDetector:
    """
    🔄 OFFLOADING DETECTOR
    
    Analyzes hardware capabilities and suggests optimal task distribution
    for maximum Trinity consciousness performance.
    """
    
    def __init__(self):
        """Initialize the offloading detector"""
        self.task_profiles = self._define_task_profiles()
        print("🔄 Offloading Detector initialized")
        print(f"   Tracking {len(self.task_profiles)} task types")
    
    def detect_opportunities(
        self, 
        profile: DynamicHardwareProfile
    ) -> List[OffloadingCapability]:
        """
        Main entry point - detect all offloading opportunities.
        
        Returns list of actionable offloading recommendations.
        """
        opportunities = []
        
        # Step 1: Normalize device capabilities
        device_caps = self._analyze_device_capabilities(profile)
        
        # Step 2: Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(profile, device_caps)
        
        # Step 3: Generate offloading strategies
        for bottleneck in bottlenecks:
            opps = self._generate_offload_strategies(
                bottleneck, 
                device_caps, 
                profile
            )
            opportunities.extend(opps)
        
        # Step 4: Add proactive optimizations
        proactive_opps = self._suggest_proactive_offloads(device_caps, profile)
        opportunities.extend(proactive_opps)
        
        # Step 5: Sort by potential impact
        opportunities.sort(key=lambda x: x.speedup_potential, reverse=True)
        
        # Step 6: Remove redundant suggestions
        opportunities = self._deduplicate_opportunities(opportunities)
        
        return opportunities[:10]  # Top 10 most impactful
    
    def _define_task_profiles(self) -> Dict[TaskType, TaskProfile]:
        """Define computational characteristics of Trinity tasks"""
        return {
            TaskType.MATRIX_MULTIPLY: TaskProfile(
                task_type=TaskType.MATRIX_MULTIPLY,
                compute_intensity=100.0,  # Very compute heavy
                memory_footprint_mb=50.0,
                bandwidth_requirement=200.0,
                latency_sensitivity=0.3,
                parallelizable=True,
                precision_flexible=True
            ),
            TaskType.CONVOLUTION: TaskProfile(
                task_type=TaskType.CONVOLUTION,
                compute_intensity=80.0,
                memory_footprint_mb=100.0,
                bandwidth_requirement=400.0,
                latency_sensitivity=0.4,
                parallelizable=True,
                precision_flexible=True
            ),
            TaskType.ATTENTION: TaskProfile(
                task_type=TaskType.ATTENTION,
                compute_intensity=60.0,
                memory_footprint_mb=200.0,
                bandwidth_requirement=300.0,
                latency_sensitivity=0.6,  # More latency sensitive
                parallelizable=True,
                precision_flexible=True
            ),
            TaskType.EMBEDDING_LOOKUP: TaskProfile(
                task_type=TaskType.EMBEDDING_LOOKUP,
                compute_intensity=5.0,   # Memory bound
                memory_footprint_mb=500.0,
                bandwidth_requirement=800.0,
                latency_sensitivity=0.8,  # Very latency sensitive
                parallelizable=False,
                precision_flexible=False
            ),
            TaskType.NORMALIZATION: TaskProfile(
                task_type=TaskType.NORMALIZATION,
                compute_intensity=20.0,
                memory_footprint_mb=50.0,
                bandwidth_requirement=100.0,
                latency_sensitivity=0.5,
                parallelizable=True,
                precision_flexible=False
            ),
            TaskType.ACTIVATION: TaskProfile(
                task_type=TaskType.ACTIVATION,
                compute_intensity=10.0,
                memory_footprint_mb=50.0,
                bandwidth_requirement=150.0,
                latency_sensitivity=0.4,
                parallelizable=True,
                precision_flexible=True
            )
        }
    
    def _analyze_device_capabilities(
        self, 
        profile: DynamicHardwareProfile
    ) -> Dict[str, DeviceCapabilities]:
        """Normalize and analyze all device capabilities"""
        capabilities = {}
        
        # Find max values for normalization
        max_compute = max(
            (c.flops_fp32 for c in profile.compute_profiles.values()),
            default=1.0
        )
        max_memory = max(
            (m.total_gb for m in profile.device_memory.values()),
            default=profile.system_memory.total_gb
        )
        max_bandwidth = max(
            (m.bandwidth_gbps for m in profile.device_memory.values()),
            default=profile.system_memory.bandwidth_gbps
        )
        
        # Analyze CPU
        if "cpu" in profile.compute_profiles:
            cpu_compute = profile.compute_profiles["cpu"]
            capabilities["cpu"] = DeviceCapabilities(
                device_id="cpu",
                compute_score=(cpu_compute.flops_fp32 / max_compute) * 100,
                memory_score=(profile.system_memory.total_gb / max_memory) * 100,
                bandwidth_score=(profile.system_memory.bandwidth_gbps / max_bandwidth) * 100,
                latency_score=10.0,  # CPU has best latency
                thermal_headroom=90.0,  # CPUs rarely thermal throttle
                special_capabilities=self._get_cpu_capabilities(cpu_compute)
            )
        
        # Analyze GPUs
        for device_id, compute in profile.compute_profiles.items():
            if device_id.startswith("cuda"):
                memory = profile.device_memory.get(device_id, 
                    MemoryProfile(total_gb=8.0, bandwidth_gbps=200.0))
                
                # Estimate thermal headroom from TDP
                thermal_headroom = self._estimate_thermal_headroom(compute)
                
                capabilities[device_id] = DeviceCapabilities(
                    device_id=device_id,
                    compute_score=(compute.flops_fp32 / max_compute) * 100,
                    memory_score=(memory.total_gb / max_memory) * 100,
                    bandwidth_score=(memory.bandwidth_gbps / max_bandwidth) * 100,
                    latency_score=30.0,  # GPU has higher latency
                    thermal_headroom=thermal_headroom,
                    special_capabilities=self._get_gpu_capabilities(compute)
                )
        
        # Analyze NPUs/Accelerators
        for i, accel in enumerate(profile.accelerators):
            accel_id = f"npu_{i}"
            capabilities[accel_id] = DeviceCapabilities(
                device_id=accel_id,
                compute_score=40.0,  # NPUs excel at specific tasks
                memory_score=(accel.memory_mb / 1024 / max_memory) * 100,
                bandwidth_score=30.0,  # Usually lower bandwidth
                latency_score=20.0,   # Good latency
                thermal_headroom=95.0,  # Very efficient
                special_capabilities=self._get_npu_capabilities(accel)
            )
        
        return capabilities
    
    def _identify_bottlenecks(
        self,
        profile: DynamicHardwareProfile,
        device_caps: Dict[str, DeviceCapabilities]
    ) -> List[Tuple[str, BottleneckType]]:
        """Identify performance bottlenecks in current configuration"""
        bottlenecks = []
        
        # Check each device for bottlenecks
        for device_id, caps in device_caps.items():
            # Compute bottleneck - low compute relative to memory
            if caps.compute_score < 50 and caps.memory_score > 70:
                bottlenecks.append((device_id, BottleneckType.COMPUTE_BOUND))
            
            # Memory bottleneck - high compute, low memory
            if caps.compute_score > 70 and caps.memory_score < 30:
                bottlenecks.append((device_id, BottleneckType.MEMORY_BOUND))
            
            # Bandwidth bottleneck
            if caps.bandwidth_score < 40:
                bottlenecks.append((device_id, BottleneckType.BANDWIDTH_BOUND))
            
            # Thermal throttle
            if caps.thermal_headroom < 20:
                bottlenecks.append((device_id, BottleneckType.THERMAL_THROTTLE))
        
        # System-wide bottlenecks
        total_gpu_memory = sum(
            m.total_gb for d, m in profile.device_memory.items() 
            if d.startswith("cuda")
        )
        
        if total_gpu_memory < 8.0 and profile.system_memory.available_gb > 16.0:
            bottlenecks.append(("system", BottleneckType.CAPACITY_LIMIT))
        
        return bottlenecks
    
    def _generate_offload_strategies(
        self,
        bottleneck: Tuple[str, BottleneckType],
        device_caps: Dict[str, DeviceCapabilities],
        profile: DynamicHardwareProfile
    ) -> List[OffloadingCapability]:
        """Generate offloading strategies for specific bottleneck"""
        device_id, bottleneck_type = bottleneck
        strategies = []
        
        if bottleneck_type == BottleneckType.COMPUTE_BOUND:
            # Find devices with spare compute
            for target_id, target_caps in device_caps.items():
                if target_id != device_id and target_caps.compute_score > 60:
                    strategies.append(self._create_compute_offload(
                        device_id, target_id, device_caps
                    ))
        
        elif bottleneck_type == BottleneckType.MEMORY_BOUND:
            # Suggest memory overflow strategies
            if device_id.startswith("cuda") and "cpu" in device_caps:
                strategies.append(OffloadingCapability(
                    source_device=device_id,
                    target_device="system_memory",
                    task_type="memory_overflow",
                    speedup_potential=1.5,
                    overhead_ms=5.0,
                    recommendation=(
                        f"Use unified memory for {device_id} - offload cold data to "
                        f"{profile.system_memory.available_gb:.0f}GB system RAM"
                    )
                ))
        
        elif bottleneck_type == BottleneckType.THERMAL_THROTTLE:
            # Suggest thermal mitigation
            cooler_devices = [
                (did, caps) for did, caps in device_caps.items()
                if caps.thermal_headroom > 50 and did != device_id
            ]
            
            for target_id, target_caps in cooler_devices:
                strategies.append(OffloadingCapability(
                    source_device=device_id,
                    target_device=target_id,
                    task_type="thermal_migration",
                    speedup_potential=1.3,
                    overhead_ms=2.0,
                    recommendation=(
                        f"Migrate 30% of {device_id} workload to {target_id} "
                        f"(thermal headroom: {target_caps.thermal_headroom:.0f}%)"
                    )
                ))
        
        elif bottleneck_type == BottleneckType.BANDWIDTH_BOUND:
            # Suggest bandwidth optimization
            strategies.append(OffloadingCapability(
                source_device=device_id,
                target_device=device_id,
                task_type="precision_reduction",
                speedup_potential=2.0,
                overhead_ms=0.1,
                recommendation=(
                    f"Switch {device_id} to INT8 precision for bandwidth-limited "
                    f"layers - 2x throughput improvement"
                )
            ))
        
        return strategies
    
    def _suggest_proactive_offloads(
        self,
        device_caps: Dict[str, DeviceCapabilities],
        profile: DynamicHardwareProfile
    ) -> List[OffloadingCapability]:
        """Suggest proactive optimizations even without bottlenecks"""
        suggestions = []
        
        # Multi-GPU distribution
        gpu_devices = [d for d in device_caps.keys() if d.startswith("cuda")]
        if len(gpu_devices) > 1:
            suggestions.append(OffloadingCapability(
                source_device="single_gpu",
                target_device="multi_gpu",
                task_type="model_parallel",
                speedup_potential=len(gpu_devices) * 0.85,  # 85% scaling
                overhead_ms=10.0,
                recommendation=(
                    f"Enable model parallelism across {len(gpu_devices)} GPUs - "
                    f"split Trinity layers for {len(gpu_devices) * 0.85:.1f}x speedup"
                )
            ))
        
        # NPU utilization
        if profile.accelerators:
            for i, accel in enumerate(profile.accelerators):
                if accel.supports_int8:
                    suggestions.append(OffloadingCapability(
                        source_device="cuda:0",
                        target_device=f"npu_{i}",
                        task_type="int8_inference",
                        speedup_potential=2.0,
                        overhead_ms=2.0,
                        recommendation=(
                            f"Offload INT8 inference to {accel.vendor} {accel.model} - "
                            f"2x efficiency gain with {accel.memory_mb}MB dedicated memory"
                        )
                    ))
        
        # CPU+GPU hybrid
        if "cpu" in device_caps and gpu_devices:
            cpu_caps = device_caps["cpu"]
            if cpu_caps.compute_score > 20:
                suggestions.append(OffloadingCapability(
                    source_device="gpu_only",
                    target_device="cpu_gpu_hybrid",
                    task_type="hybrid_compute",
                    speedup_potential=1.3,
                    overhead_ms=3.0,
                    recommendation=(
                        "Enable CPU+GPU hybrid execution - CPU handles embeddings "
                        "and normalization while GPU focuses on attention/convolution"
                    )
                ))
        
        # Memory tiering
        total_vram = sum(
            m.total_gb for d, m in profile.device_memory.items() 
            if d.startswith("cuda")
        )
        if total_vram < 12 and profile.system_memory.available_gb > 32:
            suggestions.append(OffloadingCapability(
                source_device="gpu_memory",
                target_device="tiered_memory",
                task_type="memory_tiering",
                speedup_potential=1.4,
                overhead_ms=8.0,
                recommendation=(
                    f"Implement 3-tier memory: {total_vram:.0f}GB VRAM (hot) → "
                    f"{min(16, profile.system_memory.available_gb):.0f}GB RAM (warm) → "
                    f"Storage (cold) - 40% larger models possible"
                )
            ))
        
        return suggestions
    
    def _create_compute_offload(
        self,
        source: str,
        target: str,
        device_caps: Dict[str, DeviceCapabilities]
    ) -> OffloadingCapability:
        """Create compute offloading recommendation"""
        source_caps = device_caps[source]
        target_caps = device_caps[target]
        
        # Calculate potential speedup
        compute_ratio = target_caps.compute_score / source_caps.compute_score
        speedup = min(compute_ratio * 0.8, 10.0)  # 80% efficiency, max 10x
        
        # Estimate transfer overhead
        if source.startswith("cuda") and target.startswith("cuda"):
            overhead = 0.5  # GPU-to-GPU is fast
        elif "cpu" in source or "cpu" in target:
            overhead = 2.0  # CPU involved
        else:
            overhead = 1.0
        
        # Generate recommendation
        if target.startswith("cuda"):
            task_desc = "matrix multiplication and convolution"
        elif target == "cpu":
            task_desc = "embedding lookups and data preprocessing"
        elif "npu" in target:
            task_desc = "quantized inference layers"
        else:
            task_desc = "parallel workloads"
        
        return OffloadingCapability(
            source_device=source,
            target_device=target,
            task_type="compute",
            speedup_potential=speedup,
            overhead_ms=overhead,
            recommendation=(
                f"Offload {task_desc} from {source} to {target} - "
                f"{speedup:.1f}x potential speedup with {overhead:.1f}ms overhead"
            )
        )
    
    def _get_cpu_capabilities(self, compute: ComputeProfile) -> Set[str]:
        """Extract CPU special capabilities"""
        caps = set()
        
        # Check SIMD support based on performance ratios
        if compute.flops_fp32 > compute.compute_units * 10e9:
            caps.add("avx2")
        if compute.flops_fp32 > compute.compute_units * 15e9:
            caps.add("avx512")
        if compute.flops_int8 > compute.flops_fp32 * 1.5:
            caps.add("vnni")  # INT8 acceleration
        
        return caps
    
    def _get_gpu_capabilities(self, compute: ComputeProfile) -> Set[str]:
        """Extract GPU special capabilities"""
        caps = set()
        
        # Tensor cores if FP16 is significantly faster
        if compute.flops_fp16 > compute.flops_fp32 * 2:
            caps.add("tensor_cores")
        
        # INT8 support
        if compute.flops_int8 > compute.flops_fp32:
            caps.add("int8")
        
        # Memory size tiers
        if hasattr(compute, 'memory_mb') and compute.memory_mb > 20000:
            caps.add("high_memory")
        
        return caps
    
    def _get_npu_capabilities(self, accel: AcceleratorInfo) -> Set[str]:
        """Extract NPU/accelerator capabilities"""
        caps = {"neural_engine"}
        
        if accel.supports_int8:
            caps.add("int8")
        if accel.supports_fp16:
            caps.add("fp16")
        if accel.supports_dynamic_shapes:
            caps.add("dynamic_shapes")
        
        # Vendor-specific
        if "apple" in accel.vendor.lower():
            caps.add("unified_memory")
        elif "intel" in accel.vendor.lower():
            caps.add("openvino")
        elif "qualcomm" in accel.vendor.lower():
            caps.add("hexagon_dsp")
        
        return caps
    
    def _estimate_thermal_headroom(self, compute: ComputeProfile) -> float:
        """Estimate thermal headroom percentage"""
        if not hasattr(compute, 'current_temperature') or not hasattr(compute, 'thermal_design_power'):
            return 70.0  # Default assumption
        
        # Simple linear model: 0% at 90°C, 100% at 50°C
        temp = compute.current_temperature
        if temp >= 90:
            return 0.0
        elif temp <= 50:
            return 100.0
        else:
            return (90 - temp) / 40 * 100
    
    def _deduplicate_opportunities(
        self, 
        opportunities: List[OffloadingCapability]
    ) -> List[OffloadingCapability]:
        """Remove redundant offloading suggestions"""
        seen = set()
        unique = []
        
        for opp in opportunities:
            # Create unique key
            key = (opp.source_device, opp.target_device, opp.task_type)
            if key not in seen:
                seen.add(key)
                unique.append(opp)
        
        return unique


# Example usage
if __name__ == "__main__":
    from .universal_detector import UniversalHardwareDetector
    
    print("🌆 OFFLOADING OPPORTUNITIES DETECTOR TEST")
    print("━" * 60)
    
    # Detect hardware
    detector = UniversalHardwareDetector()
    profile = detector.detect_all()
    
    # Find offloading opportunities
    offload_detector = OffloadingDetector()
    opportunities = offload_detector.detect_opportunities(profile)
    
    print(f"\n🔄 Found {len(opportunities)} offloading opportunities:")
    print("━" * 60)
    
    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp.recommendation}")
        print(f"   Potential speedup: {opp.speedup_potential:.1f}x")
        print(f"   Transfer overhead: {opp.overhead_ms:.1f}ms")
        print(f"   Type: {opp.task_type}")
