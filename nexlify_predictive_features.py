"""
Nexlify Enhanced - Predictive Features Engine
Implements Feature 22: Volatility forecasting, liquidity prediction, and market warnings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from dataclasses import dataclass
import warnings
import asyncio
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for prediction results"""
    prediction: float
    confidence: float
    prediction_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    timestamp: datetime
    model_used: str

class GARCHModel:
    """GARCH model for volatility forecasting"""
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p  # ARCH order
        self.q = q  # GARCH order
        self.params = None
        self.residuals = None
        
    def fit(self, returns: np.array):
        """Fit GARCH model to returns data"""
        # Simplified GARCH implementation
        # In production, use arch library
        self.omega = np.var(returns) * 0.1
        self.alpha = np.array([0.1] * self.p)
        self.beta = np.array([0.8] * self.q)
        
        # Store residuals for forecasting
        self.residuals = returns - np.mean(returns)
        
    def forecast(self, horizon: int = 1) -> np.array:
        """Forecast volatility for given horizon"""
        if self.params is None:
            raise ValueError("Model not fitted")
            
        # Initialize forecast
        forecast = np.zeros(horizon)
        
        # Use GARCH recursion
        last_variance = np.var(self.residuals[-self.q:])
        
        for t in range(horizon):
            # GARCH(1,1) simplification
            forecast[t] = (self.omega + 
                         self.alpha[0] * self.residuals[-1]**2 + 
                         self.beta[0] * last_variance)
            last_variance = forecast[t]
            
        return np.sqrt(forecast)  # Return volatility (std dev)

class LiquidityPredictor(nn.Module):
    """Neural network for liquidity prediction"""
    
    def __init__(self, input_features: int = 20, hidden_size: int = 64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Final prediction
        out = self.fc(attn_out[:, -1, :])
        return out

class PredictiveEngine:
    """
    Main predictive engine combining multiple forecasting models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models
        self.volatility_model = GARCHModel()
        self.liquidity_model = LiquidityPredictor()
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.fee_predictor = RandomForestRegressor(n_estimators=100)
        
        # Scaling
        self.scalers = {
            'volatility': StandardScaler(),
            'liquidity': StandardScaler(),
            'fees': StandardScaler()
        }
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        # Alert thresholds
        self.alert_thresholds = {
            'volatility_spike': 2.0,  # 2x normal volatility
            'liquidity_drop': 0.5,    # 50% drop in liquidity
            'fee_spike': 3.0,         # 3x normal fees
            'anomaly_score': 0.8      # Anomaly threshold
        }
        
        # Historical data buffers
        self.data_buffers = {
            'volatility': deque(maxlen=1000),
            'liquidity': deque(maxlen=1000),
            'fees': deque(maxlen=1000),
            'volume': deque(maxlen=1000)
        }
        
    async def predict_volatility(self, 
                               symbol: str,
                               historical_data: pd.DataFrame,
                               horizon: int = 24) -> PredictionResult:
        """
        Predict future volatility using GARCH and ML ensemble
        
        Args:
            symbol: Trading pair symbol
            historical_data: Historical OHLCV data
            horizon: Prediction horizon in hours
            
        Returns:
            Volatility prediction with confidence intervals
        """
        # Check cache
        cache_key = f"volatility_{symbol}_{horizon}"
        if self._check_cache(cache_key):
            return self.prediction_cache[cache_key]
            
        # Calculate returns
        returns = historical_data['close'].pct_change().dropna()
        
        # Fit GARCH model
        self.volatility_model.fit(returns.values)
        garch_forecast = self.volatility_model.forecast(horizon)
        
        # Prepare features for ML model
        features = self._extract_volatility_features(historical_data)
        
        # Ensemble prediction
        ml_forecast = self._ml_volatility_forecast(features, horizon)
        
        # Combine predictions
        combined_forecast = 0.6 * garch_forecast + 0.4 * ml_forecast
        
        # Calculate confidence intervals
        confidence_interval = self._calculate_confidence_interval(
            combined_forecast,
            returns.std()
        )
        
        # Feature importance
        feature_importance = {
            'historical_volatility': 0.35,
            'volume_profile': 0.25,
            'price_momentum': 0.20,
            'market_regime': 0.20
        }
        
        result = PredictionResult(
            prediction=float(np.mean(combined_forecast)),
            confidence=0.85,  # Model confidence
            prediction_interval=confidence_interval,
            feature_importance=feature_importance,
            timestamp=datetime.now(),
            model_used='GARCH+ML Ensemble'
        )
        
        # Cache result
        self._update_cache(cache_key, result)
        
        # Check for alerts
        await self._check_volatility_alerts(symbol, result)
        
        return result
        
    async def predict_liquidity(self,
                              symbol: str,
                              order_book_data: Dict,
                              market_data: pd.DataFrame) -> PredictionResult:
        """
        Predict future liquidity using deep learning
        
        Args:
            symbol: Trading pair symbol
            order_book_data: Current order book snapshot
            market_data: Historical market data
            
        Returns:
            Liquidity prediction
        """
        # Prepare features
        features = self._prepare_liquidity_features(order_book_data, market_data)
        
        # Scale features
        scaled_features = self.scalers['liquidity'].fit_transform(features)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0)
        
        # Predict with neural network
        with torch.no_grad():
            self.liquidity_model.eval()
            prediction = self.liquidity_model(input_tensor).item()
            
        # Calculate spread-based liquidity score
        spread_score = self._calculate_spread_score(order_book_data)
        
        # Combine predictions
        final_prediction = 0.7 * prediction + 0.3 * spread_score
        
        # Confidence based on order book depth
        confidence = self._calculate_liquidity_confidence(order_book_data)
        
        result = PredictionResult(
            prediction=final_prediction,
            confidence=confidence,
            prediction_interval=(final_prediction * 0.8, final_prediction * 1.2),
            feature_importance={
                'order_book_imbalance': 0.30,
                'spread': 0.25,
                'depth': 0.25,
                'recent_volume': 0.20
            },
            timestamp=datetime.now(),
            model_used='LSTM+Attention'
        )
        
        # Check for liquidity crisis
        await self._check_liquidity_alerts(symbol, result, order_book_data)
        
        return result
        
    async def predict_fee_spikes(self,
                               exchange: str,
                               network: str = 'ethereum') -> PredictionResult:
        """
        Predict exchange fee and gas price spikes
        
        Args:
            exchange: Exchange name
            network: Blockchain network for gas prices
            
        Returns:
            Fee spike prediction
        """
        # Get historical fee data
        fee_history = await self._get_fee_history(exchange, network)
        
        # Extract features
        features = self._extract_fee_features(fee_history)
        
        # Scale features
        scaled_features = self.scalers['fees'].fit_transform(features)
        
        # Predict with Random Forest
        fee_prediction = self.fee_predictor.predict(scaled_features)
        
        # Time-based patterns (higher fees during US market hours)
        time_multiplier = self._get_time_based_fee_multiplier()
        
        # Network congestion prediction
        if network == 'ethereum':
            gas_prediction = await self._predict_gas_prices()
            fee_prediction = fee_prediction * 0.7 + gas_prediction * 0.3
            
        # Adjust for time patterns
        final_prediction = fee_prediction[0] * time_multiplier
        
        result = PredictionResult(
            prediction=final_prediction,
            confidence=0.75,
            prediction_interval=(final_prediction * 0.7, final_prediction * 1.5),
            feature_importance={
                'network_congestion': 0.35,
                'time_of_day': 0.25,
                'recent_volume': 0.20,
                'mempool_size': 0.20
            },
            timestamp=datetime.now(),
            model_used='RandomForest+TimeSeriesAnalysis'
        )
        
        # Alert if fee spike predicted
        if final_prediction > self.alert_thresholds['fee_spike']:
            await self._send_fee_alert(exchange, final_prediction)
            
        return result
        
    async def detect_market_anomalies(self,
                                    market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Detect unusual market conditions and potential issues
        
        Args:
            market_data: Market data for multiple symbols
            
        Returns:
            List of detected anomalies with severity scores
        """
        anomalies = []
        
        for symbol, data in market_data.items():
            # Prepare features for anomaly detection
            features = self._extract_anomaly_features(data)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features)
            decision_scores = self.anomaly_detector.score_samples(features)
            
            # Find anomalous periods
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            
            for idx in anomaly_indices:
                severity = abs(decision_scores[idx])
                
                if severity > self.alert_thresholds['anomaly_score']:
                    anomaly = {
                        'symbol': symbol,
                        'timestamp': data.index[idx],
                        'type': self._classify_anomaly(data.iloc[idx], features[idx]),
                        'severity': severity,
                        'description': self._describe_anomaly(data.iloc[idx]),
                        'recommended_action': self._recommend_action(severity)
                    }
                    anomalies.append(anomaly)
                    
        # Sort by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)
        
        # Send alerts for critical anomalies
        for anomaly in anomalies[:3]:  # Top 3 most severe
            await self._send_anomaly_alert(anomaly)
            
        return anomalies
        
    async def predict_exchange_issues(self,
                                    exchange: str,
                                    historical_issues: List[Dict]) -> Dict:
        """
        Predict potential exchange outages or issues
        
        Args:
            exchange: Exchange name
            historical_issues: Past issue data
            
        Returns:
            Prediction of exchange reliability
        """
        # Analyze patterns in historical issues
        issue_patterns = self._analyze_issue_patterns(historical_issues)
        
        # Current exchange metrics
        current_metrics = await self._get_exchange_metrics(exchange)
        
        # Calculate risk score
        risk_factors = {
            'api_latency': self._score_latency(current_metrics['latency']),
            'error_rate': self._score_errors(current_metrics['error_rate']),
            'time_pattern': self._score_time_pattern(issue_patterns),
            'volume_stress': self._score_volume_stress(current_metrics['volume'])
        }
        
        # Weighted risk score
        weights = {'api_latency': 0.3, 'error_rate': 0.4, 
                  'time_pattern': 0.2, 'volume_stress': 0.1}
        
        total_risk = sum(risk_factors[k] * weights[k] for k in risk_factors)
        
        # Predict probability of issues
        issue_probability = 1 / (1 + np.exp(-5 * (total_risk - 0.5)))
        
        prediction = {
            'exchange': exchange,
            'issue_probability': issue_probability,
            'risk_level': self._classify_risk_level(issue_probability),
            'risk_factors': risk_factors,
            'estimated_duration': self._estimate_issue_duration(issue_patterns),
            'recommended_actions': self._get_risk_mitigation_actions(issue_probability),
            'next_maintenance': self._predict_maintenance_window(issue_patterns)
        }
        
        # Alert if high risk
        if issue_probability > 0.7:
            await self._send_exchange_risk_alert(exchange, prediction)
            
        return prediction
        
    def _extract_volatility_features(self, data: pd.DataFrame) -> np.array:
        """Extract features for volatility prediction"""
        features = []
        
        # Historical volatility (different windows)
        for window in [5, 10, 20, 50]:
            features.append(data['close'].pct_change().rolling(window).std().iloc[-1])
            
        # Parkinson volatility (high-low)
        hl_vol = np.sqrt(np.log(data['high'] / data['low']) ** 2 / (4 * np.log(2)))
        features.append(hl_vol.rolling(20).mean().iloc[-1])
        
        # Garman-Klass volatility
        gk_vol = np.sqrt(
            0.5 * np.log(data['high'] / data['low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2
        )
        features.append(gk_vol.rolling(20).mean().iloc[-1])
        
        # Volume-based features
        features.append(data['volume'].rolling(20).std().iloc[-1])
        features.append(data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1])
        
        # Price momentum
        features.append((data['close'].iloc[-1] / data['close'].iloc[-20] - 1))
        
        # Market microstructure
        features.append((data['high'] - data['low']).rolling(20).mean().iloc[-1])
        features.append(np.abs(data['close'] - data['open']).rolling(20).mean().iloc[-1])
        
        return np.array(features).reshape(1, -1)
        
    def _prepare_liquidity_features(self, 
                                  order_book: Dict,
                                  market_data: pd.DataFrame) -> np.array:
        """Prepare features for liquidity prediction"""
        features = []
        
        # Order book features
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Spread
        if bids and asks:
            spread = (asks[0][0] - bids[0][0]) / bids[0][0]
            features.append(spread)
        else:
            features.append(0)
            
        # Depth at different levels
        for level in [0.001, 0.005, 0.01]:  # 0.1%, 0.5%, 1% from mid
            bid_depth = self._calculate_depth_at_level(bids, level, 'bid')
            ask_depth = self._calculate_depth_at_level(asks, level, 'ask')
            features.extend([bid_depth, ask_depth])
            
        # Order book imbalance
        total_bid_volume = sum(b[1] for b in bids[:10])
        total_ask_volume = sum(a[1] for a in asks[:10])
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-10)
        features.append(imbalance)
        
        # Recent volume patterns
        features.append(market_data['volume'].iloc[-1])
        features.append(market_data['volume'].rolling(24).mean().iloc[-1])
        features.append(market_data['volume'].rolling(24).std().iloc[-1])
        
        # Price volatility (affects liquidity)
        features.append(market_data['close'].pct_change().rolling(24).std().iloc[-1])
        
        # Trade count (market activity)
        if 'trade_count' in market_data.columns:
            features.append(market_data['trade_count'].iloc[-1])
        else:
            features.append(0)
            
        return np.array(features).reshape(-1, len(features))
        
    def _calculate_depth_at_level(self, orders: List, level: float, side: str) -> float:
        """Calculate order book depth at price level"""
        if not orders:
            return 0
            
        mid_price = orders[0][0]
        
        if side == 'bid':
            target_price = mid_price * (1 - level)
            depth = sum(o[1] for o in orders if o[0] >= target_price)
        else:
            target_price = mid_price * (1 + level)
            depth = sum(o[1] for o in orders if o[0] <= target_price)
            
        return depth
        
    def _calculate_confidence_interval(self, 
                                     forecast: np.array,
                                     historical_std: float) -> Tuple[float, float]:
        """Calculate prediction confidence interval"""
        # Use historical volatility of volatility
        vol_of_vol = historical_std * 0.2
        
        # 95% confidence interval
        z_score = 1.96
        margin = z_score * vol_of_vol
        
        lower = float(np.mean(forecast) - margin)
        upper = float(np.mean(forecast) + margin)
        
        return (max(0, lower), upper)
        
    def _classify_anomaly(self, data_point: pd.Series, features: np.array) -> str:
        """Classify the type of anomaly detected"""
        # Check various anomaly types
        if abs(data_point['close'].pct_change()) > 0.1:
            return 'price_spike'
        elif data_point['volume'] > data_point['volume'].rolling(20).mean() * 5:
            return 'volume_surge'
        elif features[0] > features[1] * 3:  # Volatility spike
            return 'volatility_explosion'
        else:
            return 'unknown_anomaly'
            
    def _describe_anomaly(self, data_point: pd.Series) -> str:
        """Generate human-readable anomaly description"""
        descriptions = []
        
        price_change = data_point['close'].pct_change()
        if abs(price_change) > 0.05:
            descriptions.append(f"{abs(price_change)*100:.1f}% price {'surge' if price_change > 0 else 'drop'}")
            
        volume_ratio = data_point['volume'] / data_point['volume'].rolling(20).mean()
        if volume_ratio > 3:
            descriptions.append(f"{volume_ratio:.1f}x normal volume")
            
        return ", ".join(descriptions) if descriptions else "Unusual market behavior detected"
        
    def _recommend_action(self, severity: float) -> str:
        """Recommend action based on anomaly severity"""
        if severity > 0.9:
            return "CRITICAL: Halt all trading immediately"
        elif severity > 0.7:
            return "WARNING: Reduce position sizes by 50%"
        elif severity > 0.5:
            return "CAUTION: Monitor closely, tighten stops"
        else:
            return "INFO: Anomaly detected, continue monitoring"
            
    async def _check_volatility_alerts(self, symbol: str, result: PredictionResult):
        """Check and send volatility alerts"""
        if result.prediction > self.alert_thresholds['volatility_spike']:
            alert = {
                'type': 'volatility_spike',
                'symbol': symbol,
                'predicted_volatility': result.prediction,
                'normal_range': result.prediction_interval,
                'action': 'Consider reducing position sizes'
            }
            await self._send_alert(alert)
            
    async def _send_alert(self, alert: Dict):
        """Send alert through configured channels"""
        logger.warning(f"ALERT: {alert}")
        # In production, send to Telegram, email, etc.
        
    def _check_cache(self, key: str) -> bool:
        """Check if cached prediction is still valid"""
        if key in self.prediction_cache:
            cached_result = self.prediction_cache[key]
            age = (datetime.now() - cached_result.timestamp).seconds
            return age < self.cache_ttl
        return False
        
    def _update_cache(self, key: str, result: PredictionResult):
        """Update prediction cache"""
        self.prediction_cache[key] = result
        
        # Clean old entries
        current_time = datetime.now()
        keys_to_remove = []
        for k, v in self.prediction_cache.items():
            if (current_time - v.timestamp).seconds > self.cache_ttl * 2:
                keys_to_remove.append(k)
                
        for k in keys_to_remove:
            del self.prediction_cache[k]