# nexlify/ml/neural_trader.py
"""
Nexlify Neural Trader - The Quantum Trading Brain
Transformer-based market prediction that sees patterns in the chaos
Like a techno-shaman reading the digital tea leaves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# Ray for distributed computing - spread the load across the Net
import ray
from ray import tune
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer

from config.config_manager import get_config
from database.models import Candle, Symbol, TradingSignal
from monitoring.sentinel import get_sentinel, MetricType

# Check for GPU - chrome acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NEURAL TRADER] Running on: {DEVICE}")

@dataclass
class MarketState:
    """Current market state - a snapshot of the digital battlefield"""
    timestamp: datetime
    symbol: str
    features: torch.Tensor
    price: float
    volume: float
    volatility: float
    trend_strength: float
    market_regime: str  # trending, ranging, volatile

@dataclass
class TradingSignal:
    """AI-generated trading signal - wisdom from the machine"""
    symbol: str
    timestamp: datetime
    action: str  # buy, sell, hold
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple TP levels
    risk_reward_ratio: float
    expected_return: float
    reasoning: Dict[str, Any]  # Explainable AI

class CyberTransformer(nn.Module):
    """
    The Neural Core - A transformer architecture for market prediction
    Sees patterns in chaos like a netrunner sees code in the Matrix
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        # Input projection - jack into the data stream
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding - time matters in the markets
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder - the pattern recognition neural net
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # Smoother than ReLU, like good chrome
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Multi-task heads - different perspectives on the market
        self.price_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3)  # Next candle: high, low, close
        )
        
        self.signal_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # buy, sell, hold
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)  # Volatility forecast
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        # Attention visualizer - see what the AI sees
        self.attention_weights = None
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the neural matrix
        x: [batch, seq_len, features]
        """
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Pass through transformer with attention capture
        if self.training:
            x = self.transformer_encoder(x, mask=mask)
        else:
            # Capture attention for interpretability
            x = self.transformer_encoder(x, mask=mask)
        
        # Use the last timestep for predictions
        last_hidden = x[:, -1, :]
        
        # Multi-task predictions
        price_pred = self.price_predictor(last_hidden)
        signal = self.signal_classifier(last_hidden)
        volatility = self.volatility_predictor(last_hidden)
        confidence = self.confidence_estimator(last_hidden)
        
        return {
            'price_prediction': price_pred,
            'signal': signal,
            'volatility': volatility,
            'confidence': confidence,
            'hidden_states': x
        }

class PositionalEncoding(nn.Module):
    """Positional encoding - because time is money in the markets"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MarketDataset(Dataset):
    """
    Market data loader - feeds the beast with digital nutrition
    Handles OHLCV + technical indicators + market microstructure
    """
    
    def __init__(
        self,
        candles: List[Candle],
        sequence_length: int = 100,
        prediction_horizon: int = 1,
        feature_config: Optional[Dict] = None
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_config = feature_config or self._default_features()
        
        # Convert candles to DataFrame - easier to manipulate
        self.df = self._candles_to_dataframe(candles)
        
        # Engineer features - extract signal from noise
        self.df = self._engineer_features(self.df)
        
        # Prepare sequences
        self.sequences, self.targets = self._prepare_sequences()
        
        # Fit scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = self.sequences.shape
        sequences_2d = self.sequences.reshape(-1, n_features)
        
        # Fit and transform
        self.sequences = self.feature_scaler.fit_transform(sequences_2d).reshape(
            n_samples, n_timesteps, n_features
        )
        self.targets = self.target_scaler.fit_transform(self.targets)
    
    def _default_features(self) -> Dict[str, List[str]]:
        """Default feature configuration - the standard loadout"""
        return {
            'price_features': ['open', 'high', 'low', 'close', 'vwap'],
            'volume_features': ['volume', 'volume_sma', 'volume_std'],
            'technical_indicators': [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'ema_9', 'ema_21', 'ema_50', 'atr', 'adx'
            ],
            'market_microstructure': [
                'spread', 'trades_per_minute', 'order_imbalance'
            ],
            'custom_features': []
        }
    
    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candles to DataFrame - structure the chaos"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering - extract alpha from market data
        Like a street doc upgrading your neural implants
        """
        # Price features
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Technical indicators - the cybernetic augmentations
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        for period in [9, 21, 50, 200]:
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
        
        # ATR - volatility measure
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # ADX - trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Market microstructure features
        df['spread'] = df['high'] - df['low']
        df['trades_per_minute'] = df['volume'].rolling(window=5).sum() / 5  # Approximation
        df['order_imbalance'] = (df['close'] - df['open']) / df['spread'].replace(0, 1)
        
        # Cyclical time features - market has rhythms
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Forward fill NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)  # Fill any remaining NaNs
        
        return df
    
    def _prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training - slice and dice the data"""
        feature_cols = []
        for feature_list in self.feature_config.values():
            for feature in feature_list:
                if feature in self.df.columns:
                    feature_cols.append(feature)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Prepare data
        data = self.df[feature_cols].values
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            seq = data[i:i + self.sequence_length]
            
            # Target: next candle high, low, close
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            target = [
                self.df.iloc[target_idx]['high'],
                self.df.iloc[target_idx]['low'],
                self.df.iloc[target_idx]['close']
            ]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class NeuralTrader:
    """
    The main trading brain - orchestrates the AI trading operation
    Like a master netrunner controlling multiple daemons
    """
    
    def __init__(self, symbol: str, model_name: str = "CyberTransformer-v1"):
        self.symbol = symbol
        self.model_name = model_name
        self.config = get_config()
        self.sentinel = get_sentinel()
        self.logger = logging.getLogger(f"nexlify.ml.{symbol}")
        
        # Model configuration
        self.model_config = {
            'input_dim': 64,  # Will be set based on features
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'max_seq_length': self.config.ml.feature_window
        }
        
        # Initialize model
        self.model = None
        self.dataset = None
        self.is_trained = False
        
        # Trading parameters
        self.min_confidence = self.config.ml.confidence_threshold
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Model paths
        self.model_dir = Path(self.config.ml.model_checkpoint_dir) / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, candles: List[Candle]) -> MarketDataset:
        """Prepare market data for the neural net - feed the beast"""
        self.logger.info(f"Preparing {len(candles)} candles for training")
        
        dataset = MarketDataset(
            candles=candles,
            sequence_length=self.config.ml.feature_window,
            prediction_horizon=1
        )
        
        # Update input dimension based on actual features
        self.model_config['input_dim'] = dataset.sequences.shape[-1]
        
        return dataset
    
    def build_model(self) -> CyberTransformer:
        """Build the neural architecture - construct the digital brain"""
        self.logger.info(f"Building {self.model_name} with config: {self.model_config}")
        
        model = CyberTransformer(**self.model_config)
        model = model.to(DEVICE)
        
        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model built: {total_params:,} total params, {trainable_params:,} trainable")
        
        return model
    
    async def train(
        self,
        train_candles: List[Candle],
        val_candles: Optional[List[Candle]] = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None
    ):
        """
        Train the neural trader - teach it the ways of the market
        Like training a young netrunner in the ways of the Net
        """
        epochs = epochs or self.config.ml.epochs
        batch_size = batch_size or self.config.ml.batch_size
        learning_rate = learning_rate or self.config.ml.learning_rate
        
        self.logger.info(f"Starting training for {self.symbol}")
        
        # Prepare datasets
        train_dataset = self.prepare_data(train_candles)
        val_dataset = self.prepare_data(val_candles) if val_candles else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) if val_dataset else None
        
        # Build model
        self.model = self.build_model()
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss functions - multi-task learning
        price_criterion = nn.MSELoss()
        signal_criterion = nn.CrossEntropyLoss()
        volatility_criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = self.config.ml.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(sequences)
                
                # Calculate losses
                price_loss = price_criterion(outputs['price_prediction'], targets)
                
                # Generate signal targets based on price movement
                price_change = targets[:, 2] - sequences[:, -1, 3]  # Close price change
                signal_targets = torch.zeros(len(price_change), dtype=torch.long).to(DEVICE)
                signal_targets[price_change > 0.001] = 0  # Buy
                signal_targets[price_change < -0.001] = 1  # Sell
                # Hold = 2 (default)
                
                signal_loss = signal_criterion(outputs['signal'], signal_targets)
                
                # Volatility target (using price range)
                volatility_target = (targets[:, 0] - targets[:, 1]).unsqueeze(1)  # High - Low
                volatility_loss = volatility_criterion(outputs['volatility'], volatility_target)
                
                # Combined loss
                total_loss = price_loss + 0.5 * signal_loss + 0.3 * volatility_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.sentinel.record_metric(
                        MetricType.ML_MODEL,
                        f"training_loss_{self.symbol}",
                        {
                            'epoch': epoch,
                            'batch': batch_idx,
                            'loss': total_loss.item()
                        }
                    )
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences = sequences.to(DEVICE)
                        targets = targets.to(DEVICE)
                        
                        outputs = self.model(sequences)
                        
                        # Calculate validation loss
                        val_price_loss = price_criterion(outputs['price_prediction'], targets)
                        val_losses.append(val_price_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(epoch, optimizer.state_dict(), best_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch metrics
            avg_train_loss = np.mean(train_losses)
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}" if val_loader else f"Train Loss: {avg_train_loss:.4f}"
            )
            
            # Update monitoring metrics
            self.sentinel.ml_accuracy.labels(model=self.model_name).set(1.0 - avg_train_loss)
        
        self.is_trained = True
        self.logger.info(f"Training complete for {self.symbol}")
    
    def save_checkpoint(self, epoch: int, optimizer_state: dict, val_loss: float):
        """Save model checkpoint - preserve the neural state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'val_loss': val_loss,
            'model_config': self.model_config,
            'symbol': self.symbol,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if it's the best
        if epoch > 0:
            best_path = self.model_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """Load model from checkpoint - resurrect the neural state"""
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / 'best_model.pt'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Rebuild model with saved config
        self.model_config = checkpoint['model_config']
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    async def predict(
        self,
        recent_candles: List[Candle],
        current_price: float
    ) -> TradingSignal:
        """
        Generate trading signal - the moment of truth
        When the AI speaks, we listen
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained - can't predict the future without training")
        
        # Start inference timer
        start_time = time.time()
        
        # Prepare data
        dataset = self.prepare_data(recent_candles)
        
        if len(dataset) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Get the latest sequence
        sequence = torch.FloatTensor(dataset.sequences[-1:]).to(DEVICE)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequence)
        
        # Extract predictions
        price_pred = outputs['price_prediction'][0].cpu().numpy()
        signal_probs = F.softmax(outputs['signal'][0], dim=0).cpu().numpy()
        volatility = outputs['volatility'][0].item()
        confidence = outputs['confidence'][0].item()
        
        # Denormalize price predictions
        price_pred = dataset.target_scaler.inverse_transform(price_pred.reshape(1, -1))[0]
        predicted_high, predicted_low, predicted_close = price_pred
        
        # Determine trading action
        action_map = {0: 'buy', 1: 'sell', 2: 'hold'}
        action_idx = np.argmax(signal_probs)
        action = action_map[action_idx]
        
        # Calculate risk management levels
        atr = dataset.df['atr'].iloc[-1] if 'atr' in dataset.df.columns else volatility
        
        if action == 'buy':
            entry_price = current_price
            stop_loss = entry_price - 2 * atr
            take_profits = [
                entry_price + 1.5 * atr,  # TP1: 1.5x ATR
                entry_price + 3 * atr,    # TP2: 3x ATR
                entry_price + 5 * atr     # TP3: 5x ATR
            ]
        elif action == 'sell':
            entry_price = current_price
            stop_loss = entry_price + 2 * atr
            take_profits = [
                entry_price - 1.5 * atr,  # TP1: 1.5x ATR
                entry_price - 3 * atr,    # TP2: 3x ATR
                entry_price - 5 * atr     # TP3: 5x ATR
            ]
        else:  # hold
            entry_price = current_price
            stop_loss = current_price
            take_profits = [current_price]
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profits[0] - entry_price) if take_profits else 0
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Expected return based on confidence and predicted price
        expected_return = ((predicted_close - current_price) / current_price) * confidence
        
        # Generate reasoning - explainable AI
        reasoning = {
            'price_prediction': {
                'high': float(predicted_high),
                'low': float(predicted_low),
                'close': float(predicted_close)
            },
            'signal_probabilities': {
                'buy': float(signal_probs[0]),
                'sell': float(signal_probs[1]),
                'hold': float(signal_probs[2])
            },
            'volatility_forecast': float(volatility),
            'technical_indicators': {
                'rsi': float(dataset.df['rsi'].iloc[-1]) if 'rsi' in dataset.df.columns else None,
                'macd': float(dataset.df['macd'].iloc[-1]) if 'macd' in dataset.df.columns else None,
                'bb_position': float(dataset.df['bb_position'].iloc[-1]) if 'bb_position' in dataset.df.columns else None
            },
            'market_regime': self._detect_market_regime(dataset.df)
        }
        
        # Record inference time
        inference_time = time.time() - start_time
        self.sentinel.ml_inference_time.labels(model=self.model_name).observe(inference_time)
        
        # Create trading signal
        signal = TradingSignal(
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc),
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            risk_reward_ratio=risk_reward_ratio,
            expected_return=expected_return,
            reasoning=reasoning
        )
        
        # Log prediction
        self.sentinel.ml_predictions.labels(
            model=self.model_name,
            signal_type=action
        ).inc()
        
        self.logger.info(
            f"Signal generated: {action} with {confidence:.2%} confidence - "
            f"Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, TP1: {take_profits[0]:.2f}"
        )
        
        return signal
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime - understand the battlefield
        """
        # Simple regime detection based on recent price action and volatility
        recent_data = df.tail(20)
        
        # Calculate trend
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        volatility = recent_data['atr'].mean() / recent_data['close'].mean() if 'atr' in df.columns else 0.02
        
        if abs(price_change) > 0.05 and volatility < 0.03:
            return "trending"
        elif volatility > 0.05:
            return "volatile"
        else:
            return "ranging"
    
    async def backtest(
        self,
        historical_candles: List[Candle],
        initial_capital: float = 10000,
        position_size: float = 0.1  # 10% per trade
    ) -> Dict[str, Any]:
        """
        Backtest the strategy - test in the digital past
        Like running a braindance of market history
        """
        self.logger.info(f"Starting backtest for {self.symbol}")
        
        # Prepare dataset
        dataset = self.prepare_data(historical_candles)
        
        # Initialize portfolio
        capital = initial_capital
        positions = []
        trades = []
        equity_curve = [initial_capital]
        
        # Run through historical data
        for i in range(self.config.ml.feature_window, len(historical_candles) - 1):
            # Get candles for prediction
            candle_window = historical_candles[i - self.config.ml.feature_window:i]
            current_candle = historical_candles[i]
            next_candle = historical_candles[i + 1]
            
            # Generate signal
            try:
                signal = await self.predict(candle_window, float(current_candle.close))
                
                # Only trade on high confidence signals
                if signal.confidence >= self.min_confidence and signal.action != 'hold':
                    # Calculate position size
                    risk_amount = capital * self.risk_per_trade
                    stop_distance = abs(signal.entry_price - signal.stop_loss)
                    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                    
                    # Execute trade (simulated)
                    if signal.action == 'buy' and len(positions) == 0:
                        positions.append({
                            'type': 'long',
                            'entry': signal.entry_price,
                            'size': position_size,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit[0],
                            'entry_time': current_candle.timestamp
                        })
                    
                    elif signal.action == 'sell' and len(positions) == 0:
                        positions.append({
                            'type': 'short',
                            'entry': signal.entry_price,
                            'size': position_size,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit[0],
                            'entry_time': current_candle.timestamp
                        })
                
                # Check existing positions
                for position in positions[:]:
                    # Simulate order execution with next candle
                    if position['type'] == 'long':
                        # Check stop loss
                        if float(next_candle.low) <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            pnl = (exit_price - position['entry']) * position['size']
                            capital += pnl
                            trades.append({
                                'type': 'long',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'return': pnl / (position['entry'] * position['size']),
                                'duration': (next_candle.timestamp - position['entry_time']).total_seconds() / 3600
                            })
                            positions.remove(position)
                        
                        # Check take profit
                        elif float(next_candle.high) >= position['take_profit']:
                            exit_price = position['take_profit']
                            pnl = (exit_price - position['entry']) * position['size']
                            capital += pnl
                            trades.append({
                                'type': 'long',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'return': pnl / (position['entry'] * position['size']),
                                'duration': (next_candle.timestamp - position['entry_time']).total_seconds() / 3600
                            })
                            positions.remove(position)
                    
                    elif position['type'] == 'short':
                        # Check stop loss
                        if float(next_candle.high) >= position['stop_loss']:
                            exit_price = position['stop_loss']
                            pnl = (position['entry'] - exit_price) * position['size']
                            capital += pnl
                            trades.append({
                                'type': 'short',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'return': pnl / (position['entry'] * position['size']),
                                'duration': (next_candle.timestamp - position['entry_time']).total_seconds() / 3600
                            })
                            positions.remove(position)
                        
                        # Check take profit
                        elif float(next_candle.low) <= position['take_profit']:
                            exit_price = position['take_profit']
                            pnl = (position['entry'] - exit_price) * position['size']
                            capital += pnl
                            trades.append({
                                'type': 'short',
                                'entry': position['entry'],
                                'exit': exit_price,
                                'pnl': pnl,
                                'return': pnl / (position['entry'] * position['size']),
                                'duration': (next_candle.timestamp - position['entry_time']).total_seconds() / 3600
                            })
                            positions.remove(position)
                
            except Exception as e:
                self.logger.warning(f"Prediction failed at index {i}: {e}")
            
            # Record equity
            equity_curve.append(capital)
        
        # Calculate performance metrics
        if len(trades) > 0:
            returns = [t['return'] for t in trades]
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            # Sharpe ratio (assuming 0% risk-free rate)
            returns_series = pd.Series(returns)
            sharpe = np.sqrt(252) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_series[returns_series < 0]
            sortino = np.sqrt(252) * returns_series.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Max drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            results = {
                'total_return': (capital - initial_capital) / initial_capital,
                'final_capital': capital,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades),
                'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else 0,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'equity_curve': equity_curve,
                'trades': trades
            }
        else:
            results = {
                'total_return': 0,
                'final_capital': initial_capital,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'equity_curve': equity_curve,
                'trades': []
            }
        
        self.logger.info(
            f"Backtest complete: {results['total_return']:.2%} return, "
            f"{results['win_rate']:.2%} win rate, "
            f"Sharpe: {results['sharpe_ratio']:.2f}"
        )
        
        return results

# Ray-based distributed training for multiple symbols
@ray.remote(num_gpus=0.25)  # Allocate GPU resources
class DistributedNeuralTrader(NeuralTrader):
    """
    Distributed version of Neural Trader - spread across the Net
    Like having multiple netrunners working in parallel
    """
    
    async def train_distributed(self, *args, **kwargs):
        """Train with Ray's distributed computing"""
        return await self.train(*args, **kwargs)

class NeuralTradingOrchestrator:
    """
    Orchestrates multiple neural traders - the puppet master
    Manages an army of AI traders across different markets
    """
    
    def __init__(self):
        self.config = get_config()
        self.traders: Dict[str, NeuralTrader] = {}
        self.is_ray_initialized = False
    
    def initialize_ray(self):
        """Initialize Ray for distributed computing"""
        if not self.is_ray_initialized:
            ray.init(ignore_reinit_error=True)
            self.is_ray_initialized = True
    
    async def train_all_symbols(
        self,
        symbols: List[str],
        candles_dict: Dict[str, List[Candle]],
        distributed: bool = True
    ):
        """
        Train models for all symbols - teach the whole crew
        """
        if distributed:
            self.initialize_ray()
            
            # Create distributed traders
            futures = []
            for symbol in symbols:
                if symbol in candles_dict:
                    trader = DistributedNeuralTrader.remote(symbol)
                    future = trader.train_distributed.remote(candles_dict[symbol])
                    futures.append(future)
            
            # Wait for all training to complete
            results = await asyncio.gather(*[ray.get(f) for f in futures])
            
        else:
            # Sequential training
            for symbol in symbols:
                if symbol in candles_dict:
                    trader = NeuralTrader(symbol)
                    await trader.train(candles_dict[symbol])
                    self.traders[symbol] = trader
    
    def get_trader(self, symbol: str) -> Optional[NeuralTrader]:
        """Get a specific trader - find your specialist"""
        return self.traders.get(symbol)

# Global orchestrator instance
_orchestrator: Optional[NeuralTradingOrchestrator] = None

def get_neural_orchestrator() -> NeuralTradingOrchestrator:
    """Get or create the neural orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = NeuralTradingOrchestrator()
    return _orchestrator
