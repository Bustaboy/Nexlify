"""
Model Manifest System

Manages model metadata, training configuration, and trading capabilities.
Ensures models only make trades they were trained for.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class TradingCapabilities:
    """
    Defines what a model is capable of trading

    This ensures the model only trades on assets, timeframes, and strategies
    it was actually trained on.
    """

    # Trading pairs the model was trained on
    symbols: List[str] = field(default_factory=list)

    # Timeframes used in training
    timeframes: List[str] = field(default_factory=list)

    # Base currencies (e.g., BTC, ETH, USDT)
    base_currencies: Set[str] = field(default_factory=set)

    # Quote currencies
    quote_currencies: Set[str] = field(default_factory=set)

    # Exchanges trained on
    exchanges: List[str] = field(default_factory=list)

    # Strategy types
    strategies: List[str] = field(default_factory=list)

    # Market conditions (bull, bear, sideways, volatile)
    market_conditions: List[str] = field(default_factory=list)

    # Maximum position size (as fraction of balance)
    max_position_size: float = 0.1

    # Minimum confidence for trades
    min_confidence: float = 0.7

    # Maximum concurrent trades
    max_concurrent_trades: int = 5

    # DeFi capabilities
    defi_protocols: List[str] = field(default_factory=list)  # uniswap, aave, pancakeswap
    defi_networks: List[str] = field(default_factory=list)  # ethereum, polygon, bsc
    defi_strategies: List[str] = field(default_factory=list)  # yield_farming, liquidity_provision, lending
    defi_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['base_currencies'] = list(self.base_currencies)
        data['quote_currencies'] = list(self.quote_currencies)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingCapabilities':
        """Create from dictionary"""
        # Convert lists back to sets where needed
        if 'base_currencies' in data:
            data['base_currencies'] = set(data['base_currencies'])
        if 'quote_currencies' in data:
            data['quote_currencies'] = set(data['quote_currencies'])
        return cls(**data)

    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if model can trade this symbol"""
        if symbol in self.symbols:
            return True

        # Check if base/quote match
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            return base in self.base_currencies and quote in self.quote_currencies

        return False

    def can_trade_timeframe(self, timeframe: str) -> bool:
        """Check if model can trade this timeframe"""
        return timeframe in self.timeframes

    def can_trade_exchange(self, exchange: str) -> bool:
        """Check if model was trained on this exchange"""
        return exchange in self.exchanges

    def can_use_defi_protocol(self, protocol: str) -> bool:
        """Check if model can use this DeFi protocol"""
        return self.defi_enabled and protocol.lower() in [p.lower() for p in self.defi_protocols]

    def can_use_defi_network(self, network: str) -> bool:
        """Check if model was trained on this DeFi network"""
        return self.defi_enabled and network.lower() in [n.lower() for n in self.defi_networks]

    def can_execute_defi_strategy(self, strategy: str) -> bool:
        """Check if model can execute this DeFi strategy"""
        return self.defi_enabled and strategy.lower() in [s.lower() for s in self.defi_strategies]


@dataclass
class TrainingMetadata:
    """
    Metadata about how the model was trained
    """

    # Training method (e.g., 'walk_forward', 'standard', 'ensemble')
    method: str = 'walk_forward'

    # Walk-forward validation parameters
    total_episodes: int = 0
    train_size: int = 0
    test_size: int = 0
    step_size: int = 0
    mode: str = 'rolling'

    # Number of folds completed
    num_folds: int = 0

    # Training start and end times
    training_start: str = ''
    training_end: str = ''

    # Training duration in seconds
    duration_seconds: float = 0.0

    # RL agent configuration
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 64
    architecture: str = 'medium'
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01

    # Performance on training data
    training_metrics: Dict[str, float] = field(default_factory=dict)

    # Performance on validation data
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Best fold information
    best_fold_id: int = 0
    best_fold_metric: str = 'sharpe_ratio'
    best_fold_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelManifest:
    """
    Complete manifest for a trained model

    Contains all metadata about training, capabilities, and usage guidelines.
    """

    # Model identification
    model_id: str = ''
    model_name: str = ''
    version: str = '1.0.0'
    created_at: str = ''

    # Model file paths
    model_path: str = ''
    checkpoint_path: str = ''

    # What this model can do
    capabilities: TradingCapabilities = field(default_factory=TradingCapabilities)

    # How it was trained
    training: TrainingMetadata = field(default_factory=TrainingMetadata)

    # Performance summary
    performance_summary: Dict[str, Any] = field(default_factory=dict)

    # Risk parameters
    risk_parameters: Dict[str, float] = field(default_factory=dict)

    # Tags for organization
    tags: List[str] = field(default_factory=list)

    # Notes/description
    description: str = ''

    # Is this model approved for live trading?
    approved_for_live: bool = False

    # Minimum performance thresholds
    min_sharpe_ratio: float = 1.0
    min_win_rate: float = 0.55
    max_drawdown: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert nested dataclasses
        data['capabilities'] = self.capabilities.to_dict()
        data['training'] = self.training.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelManifest':
        """Create from dictionary"""
        # Parse nested structures
        if 'capabilities' in data:
            data['capabilities'] = TradingCapabilities.from_dict(data['capabilities'])
        if 'training' in data:
            data['training'] = TrainingMetadata.from_dict(data['training'])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved model manifest to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ModelManifest':
        """Load manifest from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate_trade(
        self,
        symbol: str,
        timeframe: str,
        exchange: Optional[str] = None,
        defi_protocol: Optional[str] = None,
        defi_network: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Validate if this model can execute a trade

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            exchange: Optional exchange
            defi_protocol: Optional DeFi protocol (uniswap, aave, etc.)
            defi_network: Optional DeFi network (ethereum, polygon, etc.)

        Returns:
            (is_valid, reason)
        """
        # Check symbol
        if not self.capabilities.can_trade_symbol(symbol):
            return False, f"Model not trained on symbol {symbol}"

        # Check timeframe
        if not self.capabilities.can_trade_timeframe(timeframe):
            return False, f"Model not trained on timeframe {timeframe}"

        # Check exchange
        if exchange and not self.capabilities.can_trade_exchange(exchange):
            return False, f"Model not trained on exchange {exchange}"

        # Check DeFi protocol
        if defi_protocol:
            if not self.capabilities.defi_enabled:
                return False, "Model not trained for DeFi trading"
            if not self.capabilities.can_use_defi_protocol(defi_protocol):
                return False, f"Model not trained on DeFi protocol {defi_protocol}"

        # Check DeFi network
        if defi_network:
            if not self.capabilities.defi_enabled:
                return False, "Model not trained for DeFi trading"
            if not self.capabilities.can_use_defi_network(defi_network):
                return False, f"Model not trained on DeFi network {defi_network}"

        # Check if approved for live trading
        if not self.approved_for_live:
            return False, "Model not approved for live trading"

        return True, "Trade validated"

    def meets_performance_thresholds(self) -> bool:
        """Check if model meets minimum performance requirements"""
        metrics = self.validation_metrics or self.performance_summary

        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        drawdown = abs(metrics.get('max_drawdown', 1.0))

        if sharpe < self.min_sharpe_ratio:
            logger.warning(f"Sharpe ratio {sharpe:.2f} below threshold {self.min_sharpe_ratio}")
            return False

        if win_rate < self.min_win_rate:
            logger.warning(f"Win rate {win_rate:.2%} below threshold {self.min_win_rate:.2%}")
            return False

        if drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown {drawdown:.2%} exceeds threshold {self.max_drawdown:.2%}")
            return False

        return True

    @property
    def validation_metrics(self) -> Dict[str, float]:
        """Get validation metrics"""
        return self.training.validation_metrics


class ModelManager:
    """
    Manages multiple trained models and their manifests

    Features:
    - Load and validate models
    - Select appropriate model for trading scenario
    - Enforce trading boundaries
    - Support multiple specialized models
    """

    def __init__(self, models_dir: Path = None):
        """
        Initialize model manager

        Args:
            models_dir: Directory containing models and manifests
        """
        self.models_dir = models_dir or Path('models')
        self.manifests: Dict[str, ModelManifest] = {}
        self.active_model: Optional[str] = None

        logger.info(f"ModelManager initialized with directory: {self.models_dir}")

    def register_model(
        self,
        model_id: str,
        manifest: ModelManifest
    ) -> None:
        """Register a model with its manifest"""
        self.manifests[model_id] = manifest
        logger.info(f"Registered model: {model_id}")

    def load_manifest(self, manifest_path: Path) -> ModelManifest:
        """Load a model manifest from file"""
        manifest = ModelManifest.load(manifest_path)
        self.register_model(manifest.model_id, manifest)
        return manifest

    def scan_models_directory(self) -> int:
        """
        Scan models directory for manifests

        Returns:
            Number of models found
        """
        count = 0
        for manifest_file in self.models_dir.rglob('*_manifest.json'):
            try:
                self.load_manifest(manifest_file)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load manifest {manifest_file}: {e}")

        logger.info(f"Scanned models directory: found {count} models")
        return count

    def get_model(self, model_id: str) -> Optional[ModelManifest]:
        """Get model manifest by ID"""
        return self.manifests.get(model_id)

    def list_models(self) -> List[ModelManifest]:
        """List all registered models"""
        return list(self.manifests.values())

    def set_active_model(self, model_id: str) -> bool:
        """
        Set the active model for trading

        Args:
            model_id: Model ID to activate

        Returns:
            True if successful
        """
        if model_id not in self.manifests:
            logger.error(f"Model {model_id} not found")
            return False

        manifest = self.manifests[model_id]

        # Validate model meets performance thresholds
        if not manifest.meets_performance_thresholds():
            logger.error(f"Model {model_id} does not meet performance thresholds")
            return False

        self.active_model = model_id
        logger.info(f"Activated model: {model_id}")
        return True

    def get_active_manifest(self) -> Optional[ModelManifest]:
        """Get the currently active model manifest"""
        if self.active_model:
            return self.manifests.get(self.active_model)
        return None

    def validate_trade(
        self,
        symbol: str,
        timeframe: str,
        exchange: Optional[str] = None,
        defi_protocol: Optional[str] = None,
        defi_network: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Validate trade against active model capabilities

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            exchange: Optional exchange
            defi_protocol: Optional DeFi protocol
            defi_network: Optional DeFi network

        Returns:
            (is_valid, reason)
        """
        manifest = self.get_active_manifest()
        if not manifest:
            return False, "No active model selected"

        return manifest.validate_trade(symbol, timeframe, exchange, defi_protocol, defi_network)

    def find_models_for_symbol(
        self,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> List[ModelManifest]:
        """
        Find all models that can trade a given symbol

        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe filter

        Returns:
            List of compatible models
        """
        compatible = []

        for manifest in self.manifests.values():
            if not manifest.capabilities.can_trade_symbol(symbol):
                continue

            if timeframe and not manifest.capabilities.can_trade_timeframe(timeframe):
                continue

            compatible.append(manifest)

        # Sort by performance (Sharpe ratio)
        compatible.sort(
            key=lambda m: m.validation_metrics.get('sharpe_ratio', 0),
            reverse=True
        )

        return compatible

    def get_best_model_for_trade(
        self,
        symbol: str,
        timeframe: str,
        exchange: Optional[str] = None
    ) -> Optional[ModelManifest]:
        """
        Get the best performing model for a specific trade

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            exchange: Optional exchange

        Returns:
            Best model manifest or None
        """
        models = self.find_models_for_symbol(symbol, timeframe)

        # Filter by exchange if specified
        if exchange:
            models = [
                m for m in models
                if m.capabilities.can_trade_exchange(exchange)
            ]

        # Filter by live trading approval
        models = [m for m in models if m.approved_for_live]

        # Filter by performance thresholds
        models = [m for m in models if m.meets_performance_thresholds()]

        if not models:
            return None

        # Return best performing
        return models[0]

    def export_models_summary(self, output_path: Path) -> None:
        """Export summary of all models to JSON"""
        summary = {
            'total_models': len(self.manifests),
            'active_model': self.active_model,
            'models': []
        }

        for model_id, manifest in self.manifests.items():
            summary['models'].append({
                'model_id': model_id,
                'name': manifest.model_name,
                'version': manifest.version,
                'created_at': manifest.created_at,
                'symbols': manifest.capabilities.symbols,
                'timeframes': manifest.capabilities.timeframes,
                'approved_for_live': manifest.approved_for_live,
                'sharpe_ratio': manifest.validation_metrics.get('sharpe_ratio', 0),
                'win_rate': manifest.validation_metrics.get('win_rate', 0),
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported models summary to {output_path}")
