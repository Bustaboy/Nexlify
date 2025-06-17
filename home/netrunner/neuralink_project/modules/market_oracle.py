# /home/netrunner/neuralink_project/modules/market_oracle.py
"""
Market Oracle Module - Reading the data streams like tea leaves
Financial pattern recognition with xLSTM-enhanced memory
Built for the corpo wars and street-level crypto hustle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numba as nb
from ..core.xlstm import xLSTMLayer

@dataclass
class MarketSignal:
    """Raw market pulse - straight from the data feeds"""
    timestamp: float
    price: float
    volume: float
    volatility: float
    sentiment: float  # -1 to 1, street sentiment
    manipulation_score: float  # 0 to 1, likelihood of corpo interference
    
class MarketOracle:
    """
    Financial prediction engine - sees through the corpo smoke
    Detects manipulation patterns and predicts price movements
    """
    
    def __init__(self, 
                 feature_dim: int = 32,
                 hidden_dim: int = 128,
                 prediction_horizon: int = 24,  # Hours into the future
                 sequence_length: int = 168):  # Week of hourly data
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        
        # Feature extraction network
        self.feature_extractor = FeatureExtractor(
            raw_dim=6,  # price, volume, volatility, sentiment, manipulation, time
            feature_dim=feature_dim
        )
        
        # Pattern memory - xLSTM for long-term dependencies
        self.pattern_memory = xLSTMLayer(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            batch_size=1,  # Real-time processing
            return_sequences=True,
            stateful=True
        )
        
        # Manipulation detector - identifies corpo interference
        self.manipulation_detector = ManipulationDetector(
            input_dim=hidden_dim,
            sensitivity=0.7
        )
        
        # Price predictor - the oracle's vision
        self.price_predictor = PricePredictor(
            input_dim=hidden_dim,
            horizon=prediction_horizon
        )
        
        # Adaptive learning parameters
        self.learning_rate = 0.001
        self.momentum = 0.9
        
        # Performance tracking
        self.prediction_accuracy = []
        self.manipulation_catches = []
        
    def process_market_stream(self, signals: List[MarketSignal]) -> Dict:
        """
        Process real-time market data stream
        Returns predictions and manipulation warnings
        """
        # Convert signals to feature matrix
        raw_features = self._signals_to_features(signals)
        
        # Extract deep features
        features = self.feature_extractor.extract(raw_features)
        
        # Process through pattern memory
        memory_output = self.pattern_memory.forward(features[np.newaxis, :, :])
        
        # Detect manipulation patterns
        manipulation_analysis = self.manipulation_detector.analyze(
            memory_output[0, -1, :]  # Last timestep
        )
        
        # Predict future prices
        price_predictions = self.price_predictor.predict(
            memory_output[0, -1, :],
            current_price=signals[-1].price,
            current_volatility=signals[-1].volatility
        )
        
        # Compile oracle vision
        oracle_output = {
            'price_predictions': price_predictions,
            'manipulation_detected': manipulation_analysis['detected'],
            'manipulation_type': manipulation_analysis['type'],
            'manipulation_source': manipulation_analysis['source'],
            'confidence': manipulation_analysis['confidence'],
            'risk_level': self._calculate_risk_level(
                price_predictions, manipulation_analysis
            ),
            'recommended_action': self._recommend_action(
                price_predictions, manipulation_analysis
            )
        }
        
        return oracle_output
    
    def _signals_to_features(self, signals: List[MarketSignal]) -> np.ndarray:
        """Convert market signals to feature matrix"""
        features = np.zeros((len(signals), 6), dtype=np.float32)
        
        for i, signal in enumerate(signals):
            features[i] = [
                signal.price,
                signal.volume,
                signal.volatility,
                signal.sentiment,
                signal.manipulation_score,
                signal.timestamp % 86400  # Time of day encoding
            ]
        
        return features
    
    def _calculate_risk_level(self, predictions: Dict, 
                             manipulation: Dict) -> str:
        """Calculate risk level - how hot is the action?"""
        volatility_risk = predictions['volatility_forecast']
        manipulation_risk = manipulation['confidence'] if manipulation['detected'] else 0
        
        total_risk = (volatility_risk + manipulation_risk) / 2
        
        if total_risk > 0.8:
            return "BLOOD_RED"  # Extreme danger
        elif total_risk > 0.6:
            return "NEON_ORANGE"  # High risk
        elif total_risk > 0.4:
            return "CHROME_YELLOW"  # Moderate risk
        else:
            return "ICE_BLUE"  # Low risk
    
    def _recommend_action(self, predictions: Dict, 
                         manipulation: Dict) -> str:
        """Oracle's advice - what would a street-smart trader do?"""
        if manipulation['detected'] and manipulation['confidence'] > 0.8:
            return "GHOST_PROTOCOL"  # Exit positions, go dark
        
        price_trend = predictions['price_trend']
        volatility = predictions['volatility_forecast']
        
        if price_trend > 0.2 and volatility < 0.5:
            return "FULL_CHROME"  # Strong buy
        elif price_trend > 0.1:
            return "CAUTIOUS_ENTRY"  # Small position
        elif price_trend < -0.2 and volatility > 0.7:
            return "DUMP_AND_RUN"  # Sell immediately
        elif abs(price_trend) < 0.05:
            return "FLATLINE"  # Hold position
        else:
            return "WATCH_AND_WAIT"  # Monitor closely

class FeatureExtractor:
    """Deep feature extraction from raw market data"""
    
    def __init__(self, raw_dim: int, feature_dim: int):
        self.raw_dim = raw_dim
        self.feature_dim = feature_dim
        
        # Transformation matrices
        self.W1 = np.random.randn(raw_dim, feature_dim * 2).astype(np.float32) * 0.1
        self.b1 = np.zeros(feature_dim * 2, dtype=np.float32)
        self.W2 = np.random.randn(feature_dim * 2, feature_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(feature_dim, dtype=np.float32)
        
        # Normalization parameters
        self.input_mean = np.zeros(raw_dim, dtype=np.float32)
        self.input_std = np.ones(raw_dim, dtype=np.float32)
        
    def extract(self, raw_data: np.ndarray) -> np.ndarray:
        """Extract high-level features from raw market data"""
        # Normalize input
        normalized = (raw_data - self.input_mean) / (self.input_std + 1e-8)
        
        # First layer with ReLU
        hidden = np.maximum(0, normalized @ self.W1 + self.b1)
        
        # Output layer with tanh (bounded features)
        features = np.tanh(hidden @ self.W2 + self.b2)
        
        return features
    
    def update_normalization(self, data_batch: np.ndarray):
        """Update normalization statistics"""
        alpha = 0.01
        batch_mean = np.mean(data_batch, axis=0)
        batch_std = np.std(data_batch, axis=0)
        
        self.input_mean = (1 - alpha) * self.input_mean + alpha * batch_mean
        self.input_std = (1 - alpha) * self.input_std + alpha * batch_std

class ManipulationDetector:
    """
    Detects corpo manipulation in market data
    Trained on years of Arasaka and Militech market ops
    """
    
    def __init__(self, input_dim: int, sensitivity: float = 0.7):
        self.input_dim = input_dim
        self.sensitivity = sensitivity
        
        # Pattern signatures of known manipulation techniques
        self.manipulation_patterns = {
            'pump_dump': np.random.randn(input_dim).astype(np.float32),
            'wash_trading': np.random.randn(input_dim).astype(np.float32),
            'spoofing': np.random.randn(input_dim).astype(np.float32),
            'front_running': np.random.randn(input_dim).astype(np.float32),
            'stop_hunting': np.random.randn(input_dim).astype(np.float32)
        }
        
        # Normalize pattern vectors
        for pattern in self.manipulation_patterns.values():
            pattern /= np.linalg.norm(pattern)
        
        # Source attribution network
        self.source_identifier = {
            'arasaka': np.random.randn(input_dim).astype(np.float32),
            'militech': np.random.randn(input_dim).astype(np.float32),
            'kang_tao': np.random.randn(input_dim).astype(np.float32),
            'unknown_corpo': np.random.randn(input_dim).astype(np.float32)
        }
        
    def analyze(self, market_representation: np.ndarray) -> Dict:
        """Analyze market state for manipulation"""
        # Normalize input
        norm_input = market_representation / (np.linalg.norm(market_representation) + 1e-8)
        
        # Check against known patterns
        pattern_scores = {}
        for pattern_name, pattern_vec in self.manipulation_patterns.items():
            similarity = np.dot(norm_input, pattern_vec)
            pattern_scores[pattern_name] = float(similarity)
        
        # Find strongest match
        max_pattern = max(pattern_scores, key=pattern_scores.get)
        max_score = pattern_scores[max_pattern]
        
        # Detect if manipulation threshold exceeded
        detected = max_score > self.sensitivity
        
        # Identify likely source
        source_scores = {}
        if detected:
            for source, source_vec in self.source_identifier.items():
                source_sim = np.dot(norm_input, source_vec / np.linalg.norm(source_vec))
                source_scores[source] = float(source_sim)
            
            likely_source = max(source_scores, key=source_scores.get)
        else:
            likely_source = "none"
        
        return {
            'detected': detected,
            'type': max_pattern if detected else None,
            'source': likely_source,
            'confidence': float(max_score),
            'pattern_scores': pattern_scores,
            'source_scores': source_scores if detected else {}
        }

class PricePredictor:
    """
    Multi-horizon price prediction
    Sees through the market noise to the signal beneath
    """
    
    def __init__(self, input_dim: int, horizon: int):
        self.input_dim = input_dim
        self.horizon = horizon
        
        # Prediction heads for different horizons
        self.short_term = PredictionHead(input_dim, 1)  # 1 hour
        self.medium_term = PredictionHead(input_dim, 6)  # 6 hours
        self.long_term = PredictionHead(input_dim, horizon)  # Full horizon
        
        # Volatility predictor
        self.volatility_net = np.random.randn(input_dim, 1).astype(np.float32) * 0.1
        
    def predict(self, state: np.ndarray, current_price: float, 
                current_volatility: float) -> Dict:
        """Generate multi-horizon price predictions"""
        # Get raw predictions
        short_pred = self.short_term.predict(state)
        medium_pred = self.medium_term.predict(state)
        long_pred = self.long_term.predict(state)
        
        # Predict volatility
        vol_change = np.tanh(state @ self.volatility_net).squeeze()
        future_volatility = current_volatility * (1 + vol_change * 0.2)
        
        # Scale predictions by current price
        predictions = {
            '1h': current_price * (1 + short_pred),
            '6h': current_price * (1 + medium_pred),
            f'{self.horizon}h': current_price * (1 + long_pred),
            'price_trend': float(long_pred),  # Overall trend direction
            'volatility_forecast': float(future_volatility),
            'confidence_intervals': self._calculate_confidence_intervals(
                current_price, long_pred, future_volatility
            )
        }
        
        return predictions
    
    def _calculate_confidence_intervals(self, base_price: float, 
                                      trend: float, volatility: float) -> Dict:
        """Calculate confidence intervals for predictions"""
        std_dev = base_price * volatility
        mean_price = base_price * (1 + trend)
        
        return {
            '68%': (mean_price - std_dev, mean_price + std_dev),
            '95%': (mean_price - 2*std_dev, mean_price + 2*std_dev),
            '99%': (mean_price - 3*std_dev, mean_price + 3*std_dev)
        }

class PredictionHead:
    """Single prediction head for specific time horizon"""
    
    def __init__(self, input_dim: int, horizon: int):
        self.input_dim = input_dim
        self.horizon = horizon
        
        # Simple but effective architecture
        self.W = np.random.randn(input_dim, 1).astype(np.float32) * 0.01
        self.b = np.zeros(1, dtype=np.float32)
        
        # Horizon-specific scaling
        self.horizon_scale = np.sqrt(horizon) * 0.1
        
    def predict(self, state: np.ndarray) -> float:
        """Generate prediction for this horizon"""
        raw_pred = np.tanh(state @ self.W + self.b).squeeze()
        return float(raw_pred * self.horizon_scale)
