#!/usr/bin/env python3
"""
Nexlify Ensemble Machine Learning System
Perfect ML architecture combining multiple algorithms for optimal predictions

This system includes:
- XGBoost (gradient boosting)
- Random Forest (ensemble trees)
- LSTM (deep learning time-series)
- Transformer (attention-based)
- Linear models (ridge, lasso)
- Ensemble voting/stacking
- Auto-ML model selection
- Hardware-adaptive training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class EnsembleMLSystem:
    """
    Comprehensive ensemble ML system for crypto trading

    Combines multiple algorithms with intelligent weighting
    """

    def __init__(self, task: str = 'classification', hardware_adaptive: bool = True):
        """
        Initialize ensemble ML system

        Args:
            task: 'classification' (buy/sell/hold) or 'regression' (price prediction)
            hardware_adaptive: Auto-adapt to available hardware
        """
        self.task = task
        self.hardware_adaptive = hardware_adaptive

        # Model registry
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}

        # Training configuration
        self.config = self._get_hardware_config() if hardware_adaptive else {}

        # Feature importance
        self.feature_importance = {}

        logger.info(f"🤖 Ensemble ML System initialized ({task})")

    def _get_hardware_config(self) -> Dict:
        """Get hardware-adaptive configuration"""
        try:
            from nexlify.strategies.nexlify_adaptive_rl_agent import HardwareProfiler

            profiler = HardwareProfiler()
            hw_config = profiler.optimal_config

            # Adapt model selection based on hardware
            config = {
                'use_xgboost': True,  # Always available
                'use_random_forest': True,  # Always available
                'use_lstm': hw_config.get('use_gpu', False),  # GPU preferred
                'use_transformer': hw_config.get('use_gpu', False) and hw_config.get('model_size') in ['large', 'xlarge'],
                'use_linear': True,  # Always available
                'max_ensemble_models': 5 if hw_config.get('model_size') in ['large', 'xlarge'] else 3,
                'n_jobs': hw_config.get('num_workers', -1)
            }

            logger.info(f"Hardware-adaptive config: {config}")
            return config

        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}, using defaults")
            return {
                'use_xgboost': True,
                'use_random_forest': True,
                'use_lstm': False,
                'use_linear': True,
                'use_transformer': False,
                'max_ensemble_models': 3,
                'n_jobs': -1
            }

    def build_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Build all models in the ensemble

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("🔨 Building ensemble models...")

        # 1. XGBoost
        if self.config.get('use_xgboost', True):
            self.models['xgboost'] = self._build_xgboost(X_train, y_train)

        # 2. Random Forest
        if self.config.get('use_random_forest', True):
            self.models['random_forest'] = self._build_random_forest(X_train, y_train)

        # 3. LSTM
        if self.config.get('use_lstm', False):
            self.models['lstm'] = self._build_lstm(X_train, y_train)

        # 4. Transformer
        if self.config.get('use_transformer', False):
            self.models['transformer'] = self._build_transformer(X_train, y_train)

        # 5. Linear models
        if self.config.get('use_linear', True):
            self.models['ridge'] = self._build_linear(X_train, y_train, model_type='ridge')
            self.models['lasso'] = self._build_linear(X_train, y_train, model_type='lasso')

        logger.info(f"✅ Built {len(self.models)} models: {list(self.models.keys())}")

    def _build_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build XGBoost model"""
        try:
            import xgboost as xgb

            if self.task == 'classification':
                model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softprob',
                    num_class=3,  # buy/sell/hold
                    random_state=42,
                    n_jobs=self.config.get('n_jobs', -1),
                    tree_method='gpu_hist' if self.config.get('use_gpu') else 'auto'
                )
            else:  # regression
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=self.config.get('n_jobs', -1),
                    tree_method='gpu_hist' if self.config.get('use_gpu') else 'auto'
                )

            model.fit(X_train, y_train, verbose=False)

            logger.info("✅ XGBoost model built")
            return model

        except ImportError:
            logger.warning("XGBoost not available")
            return None
        except Exception as e:
            logger.error(f"XGBoost build error: {e}")
            return None

    def _build_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if self.task == 'classification':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=self.config.get('n_jobs', -1)
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=self.config.get('n_jobs', -1)
                )

            model.fit(X_train, y_train)

            logger.info("✅ Random Forest model built")
            return model

        except Exception as e:
            logger.error(f"Random Forest build error: {e}")
            return None

    def _build_lstm(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build LSTM model"""
        try:
            import torch
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers

                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0
                    )

                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    lstm_out, _ = self.lstm(x)
                    # Take last time step
                    last_out = lstm_out[:, -1, :]
                    output = self.fc(last_out)
                    return output

            # Reshape data for LSTM (batch, seq_len, features)
            # Use last 10 time steps as sequence
            seq_len = 10
            n_samples = len(X_train) - seq_len + 1
            n_features = X_train.shape[1]

            X_seq = np.array([X_train[i:i+seq_len] for i in range(n_samples)])
            y_seq = y_train[seq_len-1:]

            # Create model
            output_size = 3 if self.task == 'classification' else 1
            model = LSTMModel(input_size=n_features, output_size=output_size)

            # Simple wrapper to match sklearn interface
            class LSTMWrapper:
                def __init__(self, model, task):
                    self.model = model
                    self.task = task
                    self.seq_len = seq_len
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(self.device)

                def fit(self, X, y, epochs=50, batch_size=32):
                    self.model.train()
                    criterion = nn.CrossEntropyLoss() if self.task == 'classification' else nn.MSELoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

                    # Simple training loop
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    y_tensor = torch.LongTensor(y).to(self.device) if self.task == 'classification' else torch.FloatTensor(y).to(self.device)

                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        outputs = self.model(X_tensor)
                        loss = criterion(outputs, y_tensor)
                        loss.backward()
                        optimizer.step()

                def predict(self, X):
                    self.model.eval()
                    with torch.no_grad():
                        # Reshape for LSTM
                        if len(X.shape) == 2:
                            # Single sample or batch
                            if X.shape[0] < self.seq_len:
                                # Pad if needed
                                X = np.vstack([np.zeros((self.seq_len - X.shape[0], X.shape[1])), X])
                            X_seq = np.array([X[-self.seq_len:]])
                        else:
                            X_seq = X

                        X_tensor = torch.FloatTensor(X_seq).to(self.device)
                        outputs = self.model(X_tensor)

                        if self.task == 'classification':
                            return torch.argmax(outputs, dim=1).cpu().numpy()
                        else:
                            return outputs.cpu().numpy()

                def predict_proba(self, X):
                    self.model.eval()
                    with torch.no_grad():
                        if len(X.shape) == 2:
                            if X.shape[0] < self.seq_len:
                                X = np.vstack([np.zeros((self.seq_len - X.shape[0], X.shape[1])), X])
                            X_seq = np.array([X[-self.seq_len:]])
                        else:
                            X_seq = X

                        X_tensor = torch.FloatTensor(X_seq).to(self.device)
                        outputs = self.model(X_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        return probs.cpu().numpy()

            wrapper = LSTMWrapper(model, self.task)
            wrapper.fit(X_seq, y_seq, epochs=50)

            logger.info("✅ LSTM model built")
            return wrapper

        except ImportError:
            logger.warning("PyTorch not available for LSTM")
            return None
        except Exception as e:
            logger.error(f"LSTM build error: {e}")
            return None

    def _build_transformer(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build Transformer model (simplified)"""
        try:
            import torch
            import torch.nn as nn

            class TransformerModel(nn.Module):
                def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=3, dropout=0.1):
                    super(TransformerModel, self).__init__()

                    self.input_proj = nn.Linear(input_size, d_model)

                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        batch_first=True
                    )

                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.fc = nn.Linear(d_model, output_size)

                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    x = self.input_proj(x)
                    x = self.transformer(x)
                    # Global average pooling
                    x = x.mean(dim=1)
                    output = self.fc(x)
                    return output

            # Similar wrapper as LSTM
            seq_len = 20  # Longer sequence for transformer
            n_samples = len(X_train) - seq_len + 1
            n_features = X_train.shape[1]

            X_seq = np.array([X_train[i:i+seq_len] for i in range(n_samples)])
            y_seq = y_train[seq_len-1:]

            output_size = 3 if self.task == 'classification' else 1
            model = TransformerModel(input_size=n_features, output_size=output_size)

            # Reuse LSTM wrapper pattern
            from types import SimpleNamespace
            wrapper = SimpleNamespace()
            wrapper.model = model
            wrapper.seq_len = seq_len
            wrapper.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(wrapper.device)

            # Simple fit method
            def fit(X, y, epochs=30):
                model.train()
                criterion = nn.CrossEntropyLoss() if self.task == 'classification' else nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

                X_tensor = torch.FloatTensor(X).to(wrapper.device)
                y_tensor = torch.LongTensor(y).to(wrapper.device) if self.task == 'classification' else torch.FloatTensor(y).to(wrapper.device)

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            wrapper.fit = fit
            wrapper.fit(X_seq, y_seq)

            logger.info("✅ Transformer model built")
            return wrapper

        except ImportError:
            logger.warning("PyTorch not available for Transformer")
            return None
        except Exception as e:
            logger.error(f"Transformer build error: {e}")
            return None

    def _build_linear(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'ridge'):
        """Build linear model (Ridge or Lasso)"""
        try:
            from sklearn.linear_model import Ridge, Lasso, LogisticRegression
            from sklearn.preprocessing import StandardScaler

            # Standardize features for linear models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            if self.task == 'classification':
                model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    multi_class='multinomial',
                    random_state=42,
                    n_jobs=self.config.get('n_jobs', -1)
                )
            else:
                if model_type == 'ridge':
                    model = Ridge(alpha=1.0, random_state=42)
                else:
                    model = Lasso(alpha=0.1, random_state=42, max_iter=1000)

            model.fit(X_scaled, y_train)

            # Wrap with scaler
            class LinearWrapper:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler

                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict(X_scaled)

                def predict_proba(self, X):
                    if hasattr(self.model, 'predict_proba'):
                        X_scaled = self.scaler.transform(X)
                        return self.model.predict_proba(X_scaled)
                    return None

            wrapper = LinearWrapper(model, scaler)
            logger.info(f"✅ {model_type.capitalize()} model built")
            return wrapper

        except Exception as e:
            logger.error(f"{model_type} build error: {e}")
            return None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train all models in the ensemble

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("🚀 Training ensemble models...")

        # Build models
        self.build_models(X_train, y_train)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)
            self._calculate_model_weights()

        logger.info("✅ Ensemble training complete")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate all models

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of model performances
        """
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

        logger.info("📊 Evaluating models...")

        for name, model in self.models.items():
            if model is None:
                continue

            try:
                predictions = model.predict(X_test)

                if self.task == 'classification':
                    accuracy = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average='weighted')
                    self.model_performance[name] = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'score': f1  # Use F1 as primary score
                    }
                    logger.info(f"  {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                else:
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    self.model_performance[name] = {
                        'mse': mse,
                        'r2': r2,
                        'score': r2  # Use R2 as primary score
                    }
                    logger.info(f"  {name}: MSE={mse:.4f}, R2={r2:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                self.model_performance[name] = {'score': 0.0}

        return self.model_performance

    def _calculate_model_weights(self):
        """Calculate ensemble weights based on validation performance"""
        if not self.model_performance:
            # Equal weights if no performance data
            n_models = len(self.models)
            self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}
            return

        # Weight by performance (softmax of scores)
        scores = np.array([perf.get('score', 0) for perf in self.model_performance.values()])
        scores = np.maximum(scores, 0)  # Ensure non-negative

        if scores.sum() == 0:
            # Fallback to equal weights
            n_models = len(scores)
            weights = np.ones(n_models) / n_models
        else:
            # Softmax weighting
            exp_scores = np.exp(scores * 5)  # Temperature = 0.2
            weights = exp_scores / exp_scores.sum()

        self.model_weights = dict(zip(self.models.keys(), weights))

        logger.info("Model weights calculated:")
        for name, weight in self.model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")

    def predict(self, X: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """
        Make predictions using ensemble

        Args:
            X: Features
            method: 'weighted', 'voting', or 'stacking'

        Returns:
            Predictions
        """
        if method == 'weighted':
            return self._predict_weighted(X)
        elif method == 'voting':
            return self._predict_voting(X)
        else:
            return self._predict_weighted(X)

    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction"""
        if self.task == 'classification':
            # Weighted probability averaging
            proba_sum = None
            total_weight = 0

            for name, model in self.models.items():
                if model is None:
                    continue

                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        weight = self.model_weights.get(name, 1.0 / len(self.models))

                        if proba_sum is None:
                            proba_sum = proba * weight
                        else:
                            proba_sum += proba * weight

                        total_weight += weight
                except:
                    continue

            if proba_sum is not None and total_weight > 0:
                proba_avg = proba_sum / total_weight
                return np.argmax(proba_avg, axis=1)
            else:
                # Fallback: simple voting
                return self._predict_voting(X)

        else:  # regression
            predictions = []
            weights = []

            for name, model in self.models.items():
                if model is None:
                    continue

                try:
                    pred = model.predict(X)
                    weight = self.model_weights.get(name, 1.0 / len(self.models))
                    predictions.append(pred * weight)
                    weights.append(weight)
                except:
                    continue

            if predictions:
                return np.sum(predictions, axis=0) / sum(weights)
            else:
                return np.zeros(len(X))

    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Simple majority voting"""
        all_predictions = []

        for name, model in self.models.items():
            if model is None:
                continue

            try:
                pred = model.predict(X)
                all_predictions.append(pred)
            except:
                continue

        if not all_predictions:
            return np.zeros(len(X))

        # Majority vote (for classification) or median (for regression)
        predictions_array = np.array(all_predictions)

        if self.task == 'classification':
            # Mode (most common)
            from scipy.stats import mode
            return mode(predictions_array, axis=0)[0].flatten()
        else:
            # Median
            return np.median(predictions_array, axis=0)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models"""
        importance = {}

        if 'xgboost' in self.models and self.models['xgboost'] is not None:
            importance['xgboost'] = self.models['xgboost'].feature_importances_

        if 'random_forest' in self.models and self.models['random_forest'] is not None:
            importance['random_forest'] = self.models['random_forest'].feature_importances_

        return importance

    def save(self, directory: str):
        """Save all models"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            if model is None:
                continue

            try:
                model_path = dir_path / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

        # Save metadata
        metadata = {
            'task': self.task,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'config': self.config
        }

        metadata_path = dir_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Ensemble saved to {directory}")

    def load(self, directory: str):
        """Load all models"""
        dir_path = Path(directory)

        # Load metadata
        metadata_path = dir_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.task = metadata.get('task', self.task)
                self.model_weights = metadata.get('model_weights', {})
                self.model_performance = metadata.get('model_performance', {})
                self.config = metadata.get('config', {})

        # Load models
        for model_file in dir_path.glob("*.pkl"):
            name = model_file.stem
            try:
                with open(model_file, 'rb') as f:
                    self.models[name] = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

        logger.info(f"✅ Ensemble loaded from {directory}")


# Export
__all__ = ['EnsembleMLSystem']
