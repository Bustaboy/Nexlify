# /home/netrunner/neuralink_project/modules/crowd_psyche.py
"""
Crowd Psyche Module - Reading the collective unconscious
Behavioral pattern analysis and prediction engine
Trained on the streets, refined in the data streams
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numba as nb
from ..core.xlstm import xLSTMLayer, sigmoid_chrome

class BehaviorState(Enum):
    """Crowd psychological states - the mood of the streets"""
    DORMANT = "dormant"          # Baseline, everyday grind
    AGITATED = "agitated"        # Rising tension, something brewing
    VOLATILE = "volatile"        # One spark from chaos
    ERUPTING = "erupting"        # Active unrest
    DISPERSING = "dispersing"    # Cooling down
    EUPHORIC = "euphoric"        # Celebration, positive energy
    PANICKED = "panicked"        # Fear-driven behavior

@dataclass
class CrowdSignal:
    """Raw behavioral data from the urban sensors"""
    timestamp: float
    location_id: int
    density: float              # People per square meter
    movement_variance: float    # How chaotic the movement
    noise_level: float         # Ambient sound levels
    social_media_sentiment: float  # -1 to 1
    temperature: float         # Environmental factor
    police_presence: float     # 0 to 1
    economic_stress: float     # Local economic indicator

class CrowdPsyche:
    """
    Behavioral prediction engine - the city's psychoanalyst
    Predicts crowd behavior patterns and potential flashpoints
    """
    
    def __init__(self,
                 grid_size: int = 100,  # City grid dimensions
                 feature_dim: int = 48,
                 hidden_dim: int = 128,
                 prediction_window: int = 12):  # Hours ahead
        
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_window = prediction_window
        
        # Spatial-temporal processor
        self.spatial_encoder = SpatialEncoder(
            grid_size=grid_size,
            feature_dim=feature_dim
        )
        
        # Temporal pattern memory - xLSTM for behavioral sequences
        self.temporal_memory = xLSTMLayer(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            sequence_length=48,  # Two days of hourly data
            batch_size=1,
            return_sequences=True,
            stateful=True
        )
        
        # Behavior state classifier
        self.state_classifier = StateClassifier(
            input_dim=hidden_dim,
            num_states=len(BehaviorState)
        )
        
        # Contagion model - how behaviors spread
        self.contagion_model = ContagionModel(
            grid_size=grid_size,
            spread_rate=0.15
        )
        
        # Intervention predictor - what calms or escalates
        self.intervention_analyzer = InterventionAnalyzer(
            input_dim=hidden_dim
        )
        
        # Historical patterns database
        self.pattern_memory = PatternMemory(capacity=1000)
        
        # Real-time state tracking
        self.current_grid_state = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.behavior_history = []
        
    def analyze_crowd_dynamics(self, signals: List[CrowdSignal]) -> Dict:
        """
        Main analysis pipeline - from signals to predictions
        """
        # Update spatial grid with latest signals
        spatial_state = self._update_spatial_state(signals)
        
        # Encode spatial features
        spatial_features = self.spatial_encoder.encode(spatial_state)
        
        # Process through temporal memory
        temporal_features = self.temporal_memory.forward(
            spatial_features[np.newaxis, np.newaxis, :]
        )
        
        # Classify current behavioral state
        state_probs = self.state_classifier.classify(temporal_features[0, -1, :])
        current_state = BehaviorState(list(BehaviorState)[np.argmax(state_probs)])
        
        # Predict contagion spread
        contagion_map = self.contagion_model.predict_spread(
            self.current_grid_state,
            current_state
        )
        
        # Analyze potential interventions
        intervention_effects = self.intervention_analyzer.analyze(
            temporal_features[0, -1, :],
            current_state
        )
        
        # Search for similar historical patterns
        similar_patterns = self.pattern_memory.find_similar(
            spatial_features,
            k=5
        )
        
        # Compile psyche analysis
        analysis = {
            'current_state': current_state.value,
            'state_confidence': float(np.max(state_probs)),
            'state_probabilities': {
                state.value: float(prob) 
                for state, prob in zip(BehaviorState, state_probs)
            },
            'hotspots': self._identify_hotspots(contagion_map),
            'contagion_risk': float(np.max(contagion_map)),
            'predicted_evolution': self._predict_evolution(
                temporal_features[0, -1, :],
                current_state
            ),
            'intervention_recommendations': intervention_effects,
            'similar_historical_events': similar_patterns,
            'crowd_mood_index': self._calculate_mood_index(signals),
            'flash_point_probability': self._calculate_flash_point_risk(
                spatial_state, current_state
            )
        }
        
        return analysis
    
    def _update_spatial_state(self, signals: List[CrowdSignal]) -> np.ndarray:
        """Update spatial grid with new behavioral signals"""
        # Create feature grid
        grid = np.zeros((self.grid_size, self.grid_size, 9), dtype=np.float32)
        
        for signal in signals:
            # Convert location_id to grid coordinates
            x = signal.location_id // self.grid_size
            y = signal.location_id % self.grid_size
            
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Update grid cell with signal features
                grid[x, y] = [
                    signal.density,
                    signal.movement_variance,
                    signal.noise_level,
                    signal.social_media_sentiment,
                    signal.temperature,
                    signal.police_presence,
                    signal.economic_stress,
                    signal.timestamp % 86400,  # Time of day
                    1.0  # Activity indicator
                ]
        
        # Update current state
        self.current_grid_state = np.mean(grid, axis=2)
        
        return grid
    
    def _identify_hotspots(self, contagion_map: np.ndarray) -> List[Dict]:
        """Identify behavioral hotspots in the city"""
        hotspots = []
        threshold = np.percentile(contagion_map, 90)
        
        # Find high-risk areas
        hot_indices = np.argwhere(contagion_map > threshold)
        
        for idx in hot_indices:
            x, y = idx
            hotspots.append({
                'location': (int(x), int(y)),
                'intensity': float(contagion_map[x, y]),
                'radius': self._estimate_influence_radius(contagion_map, x, y)
            })
        
        # Sort by intensity
        hotspots.sort(key=lambda h: h['intensity'], reverse=True)
        
        return hotspots[:10]  # Top 10 hotspots
    
    def _estimate_influence_radius(self, contagion_map: np.ndarray, 
                                  x: int, y: int) -> int:
        """Estimate influence radius of a hotspot"""
        center_value = contagion_map[x, y]
        radius = 0
        
        for r in range(1, min(10, self.grid_size)):
            # Check surrounding cells
            surrounding_values = []
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        surrounding_values.append(contagion_map[nx, ny])
            
            if surrounding_values:
                avg_surrounding = np.mean(surrounding_values)
                if avg_surrounding < center_value * 0.5:
                    break
            
            radius = r
        
        return radius
    
    def _predict_evolution(self, state_features: np.ndarray, 
                          current_state: BehaviorState) -> List[Dict]:
        """Predict behavioral evolution over time"""
        predictions = []
        
        # Simulate forward in time
        features = state_features.copy()
        
        for hour in range(1, self.prediction_window + 1):
            # Simple evolution model (would be more complex in production)
            features = features * 0.95 + np.random.randn(*features.shape) * 0.05
            
            # Predict state
            state_probs = self.state_classifier.classify(features)
            predicted_state = BehaviorState(list(BehaviorState)[np.argmax(state_probs)])
            
            predictions.append({
                'hour': hour,
                'predicted_state': predicted_state.value,
                'confidence': float(np.max(state_probs)),
                'transition_probability': self._get_transition_probability(
                    current_state, predicted_state
                )
            })
            
            current_state = predicted_state
        
        return predictions
    
    def _get_transition_probability(self, from_state: BehaviorState, 
                                   to_state: BehaviorState) -> float:
        """Get probability of state transition"""
        # Simplified transition matrix (would be learned in production)
        if from_state == to_state:
            return 0.7
        elif (from_state == BehaviorState.DORMANT and 
              to_state == BehaviorState.AGITATED):
            return 0.2
        elif (from_state == BehaviorState.AGITATED and 
              to_state == BehaviorState.VOLATILE):
            return 0.3
        elif (from_state == BehaviorState.VOLATILE and 
              to_state == BehaviorState.ERUPTING):
            return 0.4
        else:
            return 0.1
    
    def _calculate_mood_index(self, signals: List[CrowdSignal]) -> float:
        """Calculate overall crowd mood index (-1 to 1)"""
        if not signals:
            return 0.0
        
        # Weighted combination of factors
        sentiment_weight = 0.4
        stress_weight = -0.3
        density_weight = -0.1
        police_weight = -0.2
        
        mood_components = []
        for signal in signals:
            mood = (sentiment_weight * signal.social_media_sentiment +
                   stress_weight * signal.economic_stress +
                   density_weight * min(signal.density / 10, 1.0) +
                   police_weight * signal.police_presence)
            mood_components.append(mood)
        
        return float(np.tanh(np.mean(mood_components)))
    
    def _calculate_flash_point_risk(self, spatial_state: np.ndarray, 
                                   current_state: BehaviorState) -> float:
        """Calculate probability of sudden behavioral cascade"""
        # Base risk from current state
        state_risks = {
            BehaviorState.DORMANT: 0.05,
            BehaviorState.AGITATED: 0.25,
            BehaviorState.VOLATILE: 0.60,
            BehaviorState.ERUPTING: 0.90,
            BehaviorState.DISPERSING: 0.20,
            BehaviorState.EUPHORIC: 0.15,
            BehaviorState.PANICKED: 0.70
        }
        
        base_risk = state_risks.get(current_state, 0.5)
        
        # Environmental modifiers
        avg_density = np.mean(spatial_state[:, :, 0])
        avg_variance = np.mean(spatial_state[:, :, 1])
        avg_stress = np.mean(spatial_state[:, :, 6])
        
        # Calculate modified risk
        risk = base_risk * (1 + 0.2 * avg_density) * (1 + 0.3 * avg_variance) * (1 + 0.4 * avg_stress)
        
        return float(min(risk, 0.99))

class SpatialEncoder:
    """Encode spatial behavioral patterns"""
    
    def __init__(self, grid_size: int, feature_dim: int):
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # Convolutional-style encoding (simplified)
        self.conv_weights = np.random.randn(3, 3, 9, 16).astype(np.float32) * 0.1
        self.pool_size = 2
        
        # Final encoding layer
        reduced_size = (grid_size // self.pool_size) ** 2 * 16
        self.encode_weights = np.random.randn(reduced_size, feature_dim).astype(np.float32) * 0.01
        
    def encode(self, spatial_grid: np.ndarray) -> np.ndarray:
        """Encode spatial grid into feature vector"""
        # Simplified convolution (would use proper conv in production)
        features = []
        
        for i in range(0, self.grid_size - 2, self.pool_size):
            for j in range(0, self.grid_size - 2, self.pool_size):
                # Extract patch
                patch = spatial_grid[i:i+3, j:j+3, :]
                # Flatten and apply weights
                patch_features = np.tanh(patch.flatten() @ self.conv_weights.reshape(-1, 16))
                features.append(patch_features)
        
        # Combine all features
        all_features = np.concatenate(features)
        
        # Final encoding
        encoded = np.tanh(all_features @ self.encode_weights[:len(all_features), :])
        
        return encoded

class StateClassifier:
    """Classify crowd behavioral states"""
    
    def __init__(self, input_dim: int, num_states: int):
        self.input_dim = input_dim
        self.num_states = num_states
        
        # Classification weights
        self.W = np.random.randn(input_dim, num_states).astype(np.float32) * 0.1
        self.b = np.zeros(num_states, dtype=np.float32)
        
    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify behavioral state from features"""
        logits = features @ self.W + self.b
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs

class ContagionModel:
    """Model behavioral contagion spread"""
    
    def __init__(self, grid_size: int, spread_rate: float):
        self.grid_size = grid_size
        self.spread_rate = spread_rate
        
        # Influence kernel - how behavior spreads spatially
        self.influence_kernel = self._create_influence_kernel()
        
    def _create_influence_kernel(self) -> np.ndarray:
        """Create spatial influence kernel"""
        kernel_size = 5
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-dist / 2)
        
        return kernel / np.sum(kernel)
    
    def predict_spread(self, current_state: np.ndarray, 
                      behavior: BehaviorState) -> np.ndarray:
        """Predict contagion spread pattern"""
        # Initialize contagion map
        contagion = current_state.copy()
        
        # Apply spreading based on behavior type
        spread_multipliers = {
            BehaviorState.DORMANT: 0.1,
            BehaviorState.AGITATED: 0.5,
            BehaviorState.VOLATILE: 0.8,
            BehaviorState.ERUPTING: 1.0,
            BehaviorState.DISPERSING: 0.3,
            BehaviorState.EUPHORIC: 0.7,
            BehaviorState.PANICKED: 0.9
        }
        
        multiplier = spread_multipliers.get(behavior, 0.5)
        
        # Simple contagion simulation
        for _ in range(3):  # Three iterations
            new_contagion = contagion.copy()
            
            # Apply influence kernel
            for i in range(2, self.grid_size - 2):
                for j in range(2, self.grid_size - 2):
                    neighborhood = contagion[i-2:i+3, j-2:j+3]
                    influence = np.sum(neighborhood * self.influence_kernel)
                    new_contagion[i, j] = contagion[i, j] + self.spread_rate * multiplier * influence
            
            contagion = np.clip(new_contagion, 0, 1)
        
        return contagion

class InterventionAnalyzer:
    """Analyze effectiveness of crowd interventions"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
        # Intervention effect predictors
        self.intervention_effects = {
            'police_increase': np.random.randn(input_dim).astype(np.float32) * 0.1,
            'police_decrease': np.random.randn(input_dim).astype(np.float32) * 0.1,
            'economic_relief': np.random.randn(input_dim).astype(np.float32) * 0.1,
            'media_blackout': np.random.randn(input_dim).astype(np.float32) * 0.1,
            'celebrity_appearance': np.random.randn(input_dim).astype(np.float32) * 0.1,
            'weather_modification': np.random.randn(input_dim).astype(np.float32) * 0.1
        }
        
    def analyze(self, features: np.ndarray, 
                current_state: BehaviorState) -> Dict[str, Dict]:
        """Analyze potential intervention effects"""
        recommendations = {}
        
        for intervention, effect_vector in self.intervention_effects.items():
            # Calculate effect magnitude
            effect = float(np.tanh(np.dot(features, effect_vector)))
            
            # Adjust based on current state
            effectiveness = self._calculate_effectiveness(
                intervention, current_state, effect
            )
            
            recommendations[intervention] = {
                'effectiveness': effectiveness,
                'risk': self._calculate_risk(intervention, current_state),
                'recommended': effectiveness > 0.6 and 
                             self._calculate_risk(intervention, current_state) < 0.3
            }
        
        return recommendations
    
    def _calculate_effectiveness(self, intervention: str, 
                                state: BehaviorState, base_effect: float) -> float:
        """Calculate intervention effectiveness for current state"""
        # State-specific modifiers
        if state == BehaviorState.ERUPTING:
            if intervention == 'police_increase':
                return max(0, base_effect * 0.3)  # Often backfires
            elif intervention == 'celebrity_appearance':
                return min(1, base_effect * 1.5)  # Can work well
        
        return abs(base_effect)
    
    def _calculate_risk(self, intervention: str, state: BehaviorState) -> float:
        """Calculate risk of intervention backfiring"""
        risk_matrix = {
            'police_increase': {
                BehaviorState.VOLATILE: 0.8,
                BehaviorState.ERUPTING: 0.9,
                BehaviorState.AGITATED: 0.6
            },
            'media_blackout': {
                BehaviorState.PANICKED: 0.9,
                BehaviorState.VOLATILE: 0.7
            }
        }
        
        return risk_matrix.get(intervention, {}).get(state, 0.2)

class PatternMemory:
    """Store and retrieve historical behavioral patterns"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.patterns = []
        self.outcomes = []
        
    def store(self, pattern: np.ndarray, outcome: Dict):
        """Store a pattern and its outcome"""
        if len(self.patterns) >= self.capacity:
            # Remove oldest
            self.patterns.pop(0)
            self.outcomes.pop(0)
        
        self.patterns.append(pattern)
        self.outcomes.append(outcome)
    
    def find_similar(self, query_pattern: np.ndarray, k: int = 5) -> List[Dict]:
        """Find k most similar historical patterns"""
        if not self.patterns:
            return []
        
        # Calculate similarities
        similarities = []
        for i, pattern in enumerate(self.patterns):
            similarity = np.dot(query_pattern, pattern) / (
                np.linalg.norm(query_pattern) * np.linalg.norm(pattern) + 1e-8
            )
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top k
        results = []
        for sim, idx in similarities[:k]:
            results.append({
                'similarity': float(sim),
                'outcome': self.outcomes[idx],
                'pattern_age': len(self.patterns) - idx  # How old the pattern is
            })
        
        return results
