# NEXUS-7 Neural Executive eXtended Unity System

```
███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗    ███████╗
████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ╚════██║
██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗        ██╔╝
██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║       ██╔╝ 
██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║       ██║  
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝       ╚═╝  
```

> *"Where three minds become one"*

A corpo-grade autonomous agent system combining Deep Reinforcement Learning (DRL) with Extended Long Short-Term Memory (xLSTM) for real-time predictive analytics across market manipulation, crowd behavior, and urban infrastructure domains.

## 🌃 Overview

NEXUS-7 is a cyberpunk-themed neural architecture that fuses three specialized prediction modules:

- **Market Oracle** - Financial market analysis and manipulation detection
- **Crowd Psyche** - Behavioral pattern analysis and crowd dynamics prediction  
- **City Pulse** - Urban infrastructure optimization and cascade failure prevention

These modules are integrated through a **Neural Fusion** layer that synthesizes cross-domain insights and generates actionable decisions using advanced DRL agents.

## 🔧 System Requirements

### Minimum (Back-Alley Deck)
- CPU: 2 cores @ 1.5 GHz
- RAM: 4 GB
- Storage: 10 GB
- Python 3.8+

### Recommended (Street Samurai)
- CPU: 8 cores @ 2.5 GHz
- RAM: 16 GB
- Storage: 50 GB
- Python 3.10+
- GPU: Optional (8GB VRAM)

### Optimal (Corpo Mainframe)
- CPU: 16+ cores @ 3.0 GHz
- RAM: 32+ GB
- Storage: 100 GB
- Python 3.10+
- GPU: Required (16GB+ VRAM)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/nexus-7.git
cd nexus-7

# Create virtual environment
python -m venv nexus_env
source nexus_env/bin/activate  # On Windows: nexus_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default configuration
python main.py

# Run with custom config
python main.py --config configs/custom_config.yaml

# Run in supervised mode
python main.py --mode supervised

# Resume from checkpoint
python main.py --checkpoint checkpoints/nexus7_checkpoint_1000.npz
```

## 🏗️ Architecture

### Core Components

#### 1. **xLSTM Core** (`core/xlstm.py`)
Enhanced LSTM with exponential gating for superior long-term memory:
- Exponential gating mechanism
- Stateful operation support
- Hardware-optimized computations
- Memory-efficient batch processing

#### 2. **DRL Agent** (`core/drl_agent.py`)
PPO-based reinforcement learning agent:
- Policy and value networks with xLSTM integration
- Generalized Advantage Estimation (GAE)
- Adaptive learning with meta-prediction
- Real-time decision synthesis

#### 3. **Neural Fusion** (`modules/neural_fusion.py`)
Cross-domain intelligence integration:
- Cross-modal attention mechanisms
- Conflict resolution between modules
- Cascade effect prediction
- Emergent pattern detection

### Specialized Modules

#### Market Oracle 🏦
- Real-time price prediction (1h, 6h, 24h horizons)
- Market manipulation detection (pump & dump, spoofing, etc.)
- Volatility forecasting
- Corporate attribution analysis

#### Crowd Psyche 👥
- Behavioral state classification (7 states)
- Contagion spread modeling
- Flash point prediction
- Intervention effectiveness analysis

#### City Pulse 🏙️
- Traffic flow optimization
- Resource allocation across zones
- Infrastructure health monitoring
- Cascade failure prevention

## 🔬 Technical Features

### Hardware Optimization
- Automatic hardware detection and profiling
- Adaptive computation strategies
- JIT compilation with Numba
- Intelligent memory management
- Multi-level caching system

### Real-time Processing
- Streaming data pipeline
- Asynchronous module execution
- Lock-free data structures
- Configurable FPS targets

### Predictive Capabilities
- Multi-horizon predictions (1h to 48h)
- Cross-domain cascade modeling
- Black swan event detection
- Confidence interval estimation

## 📊 Performance Metrics

The system tracks and reports:
- Prediction accuracy across domains
- Decision execution rates
- Processing latency
- Memory utilization
- Hardware efficiency scores

## 🛠️ Configuration

### Main Configuration (`configs/nexus_config.yaml`)
```yaml
system:
  mode: "autonomous"  # autonomous, supervised, test
  
modules:
  market_oracle:
    enabled: true
    prediction_horizon: 24
    
fusion:
  decision_dim: 64
  attention_heads: 8
```

### Hardware Profiles
The system automatically detects and optimizes for:
- **CORPO_MAINFRAME** - Maximum performance
- **STREET_SAMURAI** - Balanced efficiency
- **NETRUNNER_STANDARD** - Standard operations
- **BACK_ALLEY_DECK** - Survival mode

## 📁 Project Structure

```
neuralink_project/
├── core/               # Core neural components
│   ├── xlstm.py       # Extended LSTM implementation
│   ├── drl_agent.py   # DRL agent architecture
│   └── memory_bank.py # Memory management
├── modules/           # Specialized prediction modules
│   ├── market_oracle.py
│   ├── crowd_psyche.py
│   ├── city_pulse.py
│   └── neural_fusion.py
├── utils/             # Utility functions
│   ├── chrome_optimizer.py
│   ├── data_stream.py
│   └── tensor_ops.py
├── configs/           # Configuration files
├── tests/             # Test suites
├── data/              # Data directories
├── models/            # Saved models
└── main.py           # Entry point
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_xlstm.py

# Run performance benchmarks
python tests/benchmarks.py
```

## 🎯 Use Cases

1. **Financial Market Analysis**
   - High-frequency trading strategies
   - Market manipulation detection
   - Risk assessment and hedging

2. **Urban Planning**
   - Traffic optimization
   - Resource allocation
   - Infrastructure maintenance scheduling

3. **Crowd Management**
   - Event security planning
   - Emergency response coordination
   - Social unrest prediction

4. **Integrated City Operations**
   - Cross-domain threat detection
   - Cascade failure prevention
   - System-wide optimization

## ⚡ Performance Tips

1. **Memory Optimization**
   - Enable tensor pooling for repeated operations
   - Use quantization for non-critical tensors
   - Configure appropriate batch sizes

2. **Compute Optimization**
   - Enable JIT compilation
   - Use hardware-specific profiles
   - Leverage parallel processing

3. **Data Pipeline**
   - Use asynchronous data loading
   - Enable compression for network streams
   - Implement circular buffers

## 🔒 Security Considerations

- All data streams use encryption
- Model checkpoints are signed
- Access control for decision execution
- Audit logging for all critical operations

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## 📜 License

This project is licensed under the MIT License - see `LICENSE` file for details.

## 🙏 Acknowledgments

- Inspired by the cyberpunk genre and William Gibson's Neuromancer
- Built with love for the intersection of AI and speculative fiction
- Special thanks to the open-source community

---

*"The street finds its own uses for things."* - William Gibson

**Remember**: In Night City, the only limit is your hardware. Stay chrome, choombas! 🌃✨
