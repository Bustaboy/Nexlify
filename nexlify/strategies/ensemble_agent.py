#!/usr/bin/env python3
"""
Nexlify Ensemble Agent - Model Ensemble System for Robust Trading Predictions

Implements ensemble learning for DQN agents with multiple strategies:
- Simple averaging
- Weighted averaging (by validation performance)
- Voting (for discrete actions)
- Stacking (meta-model on top)

Provides uncertainty estimation through prediction variance across models.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nexlify.strategies.nexlify_rl_agent import DQNAgent
from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class EnsembleStrategy:
    """Enum for ensemble strategies"""
    SIMPLE_AVG = "simple_avg"
    WEIGHTED_AVG = "weighted_avg"
    VOTING = "voting"
    STACKING = "stacking"


class StackingMetaModel(nn.Module):
    """
    Meta-model for stacking ensemble

    Takes Q-values from all base models and learns optimal combination
    """

    def __init__(self, num_models: int, action_size: int, hidden_size: int = 64):
        super(StackingMetaModel, self).__init__()

        input_size = num_models * action_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, stacked_q_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stacked_q_values: Shape [batch_size, num_models * action_size]

        Returns:
            Combined Q-values: Shape [batch_size, action_size]
        """
        return self.network(stacked_q_values)


class ModelInfo:
    """Information about an ensemble member model"""

    def __init__(
        self,
        model_path: str,
        agent: DQNAgent,
        validation_score: float = 0.0,
        weight: float = 1.0
    ):
        self.model_path = model_path
        self.agent = agent
        self.validation_score = validation_score
        self.weight = weight
        self.prediction_count = 0
        self.total_prediction_time = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_path": self.model_path,
            "validation_score": self.validation_score,
            "weight": self.weight,
            "prediction_count": self.prediction_count,
            "avg_prediction_time": (
                self.total_prediction_time / self.prediction_count
                if self.prediction_count > 0 else 0.0
            )
        }


class EnsembleManager:
    """
    Ensemble Manager for combining multiple DQN agents

    Supports multiple ensemble strategies:
    - Simple averaging: Average Q-values from all models
    - Weighted averaging: Weight by validation performance
    - Voting: Majority vote on actions
    - Stacking: Meta-model learns optimal combination

    Provides uncertainty estimation through prediction variance.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        strategy: str = EnsembleStrategy.WEIGHTED_AVG,
        ensemble_size: int = 3,
        diversity_penalty: float = 0.1,
        config: Optional[Dict] = None
    ):
        """
        Initialize Ensemble Manager

        Args:
            state_size: Size of state space
            action_size: Size of action space
            strategy: Ensemble strategy (simple_avg, weighted_avg, voting, stacking)
            ensemble_size: Target number of models in ensemble
            diversity_penalty: Penalty for similar predictions (0-1)
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.strategy = strategy
        self.ensemble_size = ensemble_size
        self.diversity_penalty = diversity_penalty
        self.config = config or {}

        # Ensemble members
        self.models: List[ModelInfo] = []

        # Stacking meta-model (if using stacking strategy)
        self.meta_model: Optional[StackingMetaModel] = None
        self.meta_optimizer: Optional[torch.optim.Optimizer] = None

        # Device
        self.device = self._get_device()

        # Performance tracking
        self.ensemble_predictions = 0
        self.uncertainty_history = []
        self.disagreement_history = []

        # Statistics
        self.stats = {
            "total_predictions": 0,
            "high_uncertainty_count": 0,
            "disagreement_count": 0,
            "avg_uncertainty": 0.0,
            "avg_disagreement": 0.0
        }

        logger.info(f"🎯 Ensemble Manager initialized")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Target ensemble size: {ensemble_size}")
        logger.info(f"   Diversity penalty: {diversity_penalty}")

    def _get_device(self) -> str:
        """Detect available compute device"""
        try:
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        return "cpu"

    def add_model(
        self,
        model_path: str,
        validation_score: float = 0.0,
        agent_config: Optional[Dict] = None
    ) -> bool:
        """
        Add a trained model to the ensemble

        Args:
            model_path: Path to saved model checkpoint
            validation_score: Validation performance metric (higher is better)
            agent_config: Configuration for creating agent instance

        Returns:
            True if model added successfully
        """
        try:
            # Check if model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Create agent instance
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config=agent_config or self.config
            )

            # Load model weights
            agent.load(model_path)

            # Set to evaluation mode
            if hasattr(agent.model, 'eval'):
                agent.model.eval()

            # Calculate weight based on validation score
            weight = self._calculate_weight(validation_score)

            # Create model info
            model_info = ModelInfo(
                model_path=model_path,
                agent=agent,
                validation_score=validation_score,
                weight=weight
            )

            self.models.append(model_info)

            logger.info(
                f"✅ Added model to ensemble: {Path(model_path).name} "
                f"(score: {validation_score:.4f}, weight: {weight:.4f})"
            )

            # Re-normalize weights
            self._normalize_weights()

            # Initialize stacking meta-model if needed
            if self.strategy == EnsembleStrategy.STACKING and len(self.models) >= 2:
                if self.meta_model is None:
                    self._initialize_meta_model()

            return True

        except Exception as e:
            logger.error(f"Failed to add model {model_path}: {e}")
            return False

    def _calculate_weight(self, validation_score: float) -> float:
        """
        Calculate model weight based on validation score

        Uses softmax-like weighting to emphasize better models
        """
        if validation_score <= 0:
            return 1.0

        # Exponential weighting
        return np.exp(validation_score)

    def _normalize_weights(self):
        """Normalize weights so they sum to 1.0"""
        if not self.models:
            return

        total_weight = sum(m.weight for m in self.models)
        if total_weight > 0:
            for model in self.models:
                model.weight /= total_weight

    def _initialize_meta_model(self):
        """Initialize stacking meta-model"""
        num_models = len(self.models)

        self.meta_model = StackingMetaModel(
            num_models=num_models,
            action_size=self.action_size,
            hidden_size=self.config.get('meta_model_hidden_size', 64)
        ).to(self.device)

        learning_rate = self.config.get('meta_learning_rate', 0.001)
        self.meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=learning_rate
        )

        logger.info(f"🧠 Initialized stacking meta-model ({num_models} base models)")

    def load_ensemble_from_directory(
        self,
        models_dir: str,
        validation_scores: Optional[Dict[str, float]] = None,
        max_models: Optional[int] = None
    ) -> int:
        """
        Load multiple models from a directory

        Args:
            models_dir: Directory containing saved models
            validation_scores: Dict mapping model names to validation scores
            max_models: Maximum number of models to load (None = load all)

        Returns:
            Number of models successfully loaded
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            logger.error(f"Models directory not found: {models_dir}")
            return 0

        # Find all model files
        model_files = list(models_path.glob("*.pt")) + list(models_path.glob("*.pth"))

        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return 0

        # Sort by validation score if available
        if validation_scores:
            model_files = sorted(
                model_files,
                key=lambda f: validation_scores.get(f.stem, 0.0),
                reverse=True
            )

        # Limit number of models
        if max_models:
            model_files = model_files[:max_models]

        # Load models
        loaded_count = 0
        for model_file in model_files:
            score = validation_scores.get(model_file.stem, 0.0) if validation_scores else 0.0

            if self.add_model(str(model_file), validation_score=score):
                loaded_count += 1

        logger.info(f"📦 Loaded {loaded_count}/{len(model_files)} models into ensemble")

        return loaded_count

    def select_diverse_models(
        self,
        candidate_models: List[Tuple[str, float]],
        validation_states: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Select diverse models for ensemble using diversity penalty

        Prefers models that make different predictions to increase robustness

        Args:
            candidate_models: List of (model_path, validation_score) tuples
            validation_states: States to evaluate diversity (if None, skip diversity)

        Returns:
            List of selected model paths
        """
        if len(candidate_models) <= self.ensemble_size:
            return [path for path, _ in candidate_models]

        # Sort by validation score
        sorted_candidates = sorted(candidate_models, key=lambda x: x[1], reverse=True)

        # Always include best model
        selected = [sorted_candidates[0][0]]
        selected_scores = [sorted_candidates[0][1]]

        # If no validation states, just select top K by score
        if validation_states is None or len(validation_states) == 0:
            return [path for path, _ in sorted_candidates[:self.ensemble_size]]

        # Iteratively select models that balance performance and diversity
        for candidate_path, candidate_score in sorted_candidates[1:]:
            if len(selected) >= self.ensemble_size:
                break

            # Calculate diversity score
            diversity_score = self._calculate_diversity(
                candidate_path,
                selected,
                validation_states
            )

            # Combined score: performance + diversity bonus
            combined_score = (
                (1 - self.diversity_penalty) * candidate_score +
                self.diversity_penalty * diversity_score
            )

            # Add if good combined score
            if combined_score > min(selected_scores) or len(selected) < self.ensemble_size:
                selected.append(candidate_path)
                selected_scores.append(combined_score)

        logger.info(f"🎯 Selected {len(selected)} diverse models for ensemble")

        return selected[:self.ensemble_size]

    def _calculate_diversity(
        self,
        candidate_path: str,
        selected_paths: List[str],
        validation_states: np.ndarray
    ) -> float:
        """
        Calculate prediction diversity between candidate and selected models

        Higher diversity = more different predictions
        """
        # Load candidate model temporarily
        candidate_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=self.config
        )
        candidate_agent.load(candidate_path)
        candidate_agent.model.eval()

        # Get predictions from candidate
        candidate_preds = []
        for state in validation_states:
            action = candidate_agent.act(state, training=False)
            candidate_preds.append(action)

        # Calculate average disagreement with selected models
        total_disagreement = 0.0

        for selected_path in selected_paths:
            selected_agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config=self.config
            )
            selected_agent.load(selected_path)
            selected_agent.model.eval()

            # Get predictions from selected model
            selected_preds = []
            for state in validation_states:
                action = selected_agent.act(state, training=False)
                selected_preds.append(action)

            # Calculate disagreement rate
            disagreement = np.mean([
                1.0 if c != s else 0.0
                for c, s in zip(candidate_preds, selected_preds)
            ])

            total_disagreement += disagreement

        # Average disagreement across all selected models
        diversity_score = total_disagreement / len(selected_paths)

        return diversity_score

    def predict(
        self,
        state: np.ndarray,
        return_uncertainty: bool = True
    ) -> Tuple[int, Optional[float]]:
        """
        Make ensemble prediction

        Args:
            state: Current state
            return_uncertainty: If True, return uncertainty estimate

        Returns:
            (action, uncertainty) where uncertainty is std of Q-values
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        self.stats["total_predictions"] += 1

        if self.strategy == EnsembleStrategy.SIMPLE_AVG:
            action, uncertainty = self._predict_simple_avg(state)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVG:
            action, uncertainty = self._predict_weighted_avg(state)
        elif self.strategy == EnsembleStrategy.VOTING:
            action, uncertainty = self._predict_voting(state)
        elif self.strategy == EnsembleStrategy.STACKING:
            action, uncertainty = self._predict_stacking(state)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Track uncertainty
        if uncertainty is not None:
            self.uncertainty_history.append(uncertainty)
            self.stats["avg_uncertainty"] = np.mean(self.uncertainty_history[-1000:])

            # Check for high uncertainty
            uncertainty_threshold = self.config.get('high_uncertainty_threshold', 0.5)
            if uncertainty > uncertainty_threshold:
                self.stats["high_uncertainty_count"] += 1
                logger.debug(f"⚠️  High uncertainty: {uncertainty:.4f} (action: {action})")

        if return_uncertainty:
            return action, uncertainty
        else:
            return action, None

    def _predict_simple_avg(self, state: np.ndarray) -> Tuple[int, float]:
        """Simple averaging of Q-values"""
        all_q_values = []

        for model_info in self.models:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = model_info.agent.model(state_tensor)
                all_q_values.append(q_values.cpu().numpy()[0])

        # Average Q-values
        avg_q_values = np.mean(all_q_values, axis=0)

        # Best action
        action = int(np.argmax(avg_q_values))

        # Uncertainty: std deviation across models
        uncertainty = float(np.std([q[action] for q in all_q_values]))

        return action, uncertainty

    def _predict_weighted_avg(self, state: np.ndarray) -> Tuple[int, float]:
        """Weighted averaging based on validation scores"""
        all_q_values = []
        weights = []

        for model_info in self.models:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = model_info.agent.model(state_tensor)
                all_q_values.append(q_values.cpu().numpy()[0])
                weights.append(model_info.weight)

        # Weighted average Q-values
        weights = np.array(weights)
        avg_q_values = np.average(all_q_values, axis=0, weights=weights)

        # Best action
        action = int(np.argmax(avg_q_values))

        # Uncertainty: weighted std deviation
        uncertainty = float(np.sqrt(
            np.average(
                [(q[action] - avg_q_values[action])**2 for q in all_q_values],
                weights=weights
            )
        ))

        return action, uncertainty

    def _predict_voting(self, state: np.ndarray) -> Tuple[int, float]:
        """Majority voting on actions"""
        votes = defaultdict(float)
        action_q_values = defaultdict(list)

        for model_info in self.models:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = model_info.agent.model(state_tensor).cpu().numpy()[0]
                action = int(np.argmax(q_values))

                # Weight vote by model weight
                votes[action] += model_info.weight
                action_q_values[action].append(q_values[action])

        # Get action with most votes
        action = max(votes.items(), key=lambda x: x[1])[0]

        # Uncertainty: disagreement rate
        vote_counts = list(votes.values())
        max_votes = max(vote_counts)
        disagreement = 1.0 - (max_votes / sum(vote_counts))

        # Track disagreement
        self.disagreement_history.append(disagreement)
        self.stats["avg_disagreement"] = np.mean(self.disagreement_history[-1000:])

        if disagreement > 0.5:
            self.stats["disagreement_count"] += 1

        uncertainty = float(disagreement)

        return action, uncertainty

    def _predict_stacking(self, state: np.ndarray) -> Tuple[int, float]:
        """Stacking with meta-model"""
        if self.meta_model is None:
            # Fallback to weighted average if meta-model not trained
            logger.warning("Meta-model not trained, using weighted average")
            return self._predict_weighted_avg(state)

        # Get Q-values from all base models
        all_q_values = []

        for model_info in self.models:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = model_info.agent.model(state_tensor)
                all_q_values.append(q_values)

        # Stack Q-values
        stacked = torch.cat(all_q_values, dim=1)  # [1, num_models * action_size]

        # Meta-model prediction
        with torch.no_grad():
            meta_q_values = self.meta_model(stacked).cpu().numpy()[0]

        action = int(np.argmax(meta_q_values))

        # Uncertainty: variance in base model Q-values for chosen action
        base_q_for_action = [
            q[0][action % self.action_size].item()
            for q in all_q_values
        ]
        uncertainty = float(np.std(base_q_for_action))

        return action, uncertainty

    def train_meta_model(
        self,
        training_states: np.ndarray,
        training_targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train stacking meta-model

        Args:
            training_states: States for training
            training_targets: Target Q-values
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.meta_model is None:
            self._initialize_meta_model()

        logger.info(f"🧠 Training stacking meta-model ({epochs} epochs)...")

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(training_states), batch_size):
                batch_states = training_states[i:i+batch_size]
                batch_targets = training_targets[i:i+batch_size]

                # Get predictions from base models
                all_q_values = []
                for state in batch_states:
                    q_vals = []
                    for model_info in self.models:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            q_values = model_info.agent.model(state_tensor)
                            q_vals.append(q_values)
                    stacked = torch.cat(q_vals, dim=1)
                    all_q_values.append(stacked)

                # Stack batch
                batch_stacked = torch.cat(all_q_values, dim=0)
                batch_targets_tensor = torch.FloatTensor(batch_targets).to(self.device)

                # Forward pass
                predictions = self.meta_model(batch_stacked)
                loss = criterion(predictions, batch_targets_tensor)

                # Backward pass
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            if (epoch + 1) % 10 == 0:
                logger.info(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

        logger.info("✅ Meta-model training complete")

    def remove_model(self, model_index: int) -> bool:
        """
        Remove a model from the ensemble

        Args:
            model_index: Index of model to remove

        Returns:
            True if removed successfully
        """
        if 0 <= model_index < len(self.models):
            removed = self.models.pop(model_index)
            logger.info(f"Removed model: {Path(removed.model_path).name}")

            # Re-normalize weights
            self._normalize_weights()

            return True

        return False

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about ensemble"""
        return {
            "num_models": len(self.models),
            "strategy": self.strategy,
            "ensemble_size": self.ensemble_size,
            "diversity_penalty": self.diversity_penalty,
            "models": [m.to_dict() for m in self.models],
            "stats": self.stats,
            "has_meta_model": self.meta_model is not None
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        return {
            **self.stats,
            "num_models": len(self.models),
            "recent_uncertainty": (
                float(np.mean(self.uncertainty_history[-100:]))
                if self.uncertainty_history else 0.0
            ),
            "recent_disagreement": (
                float(np.mean(self.disagreement_history[-100:]))
                if self.disagreement_history else 0.0
            )
        }

    def save_ensemble_config(self, filepath: str):
        """Save ensemble configuration"""
        config = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "strategy": self.strategy,
            "ensemble_size": self.ensemble_size,
            "diversity_penalty": self.diversity_penalty,
            "models": [
                {
                    "path": m.model_path,
                    "validation_score": m.validation_score,
                    "weight": m.weight
                }
                for m in self.models
            ],
            "stats": self.stats
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        # Save meta-model if using stacking
        if self.meta_model is not None:
            meta_path = Path(filepath).parent / f"{Path(filepath).stem}_meta.pt"
            torch.save({
                'model_state_dict': self.meta_model.state_dict(),
                'optimizer_state_dict': self.meta_optimizer.state_dict()
            }, meta_path)

        logger.info(f"💾 Ensemble config saved to {filepath}")

    def load_ensemble_config(self, filepath: str):
        """Load ensemble configuration"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        # Load models
        for model_config in config['models']:
            self.add_model(
                model_path=model_config['path'],
                validation_score=model_config['validation_score']
            )

        self.stats = config.get('stats', self.stats)

        # Load meta-model if exists
        meta_path = Path(filepath).parent / f"{Path(filepath).stem}_meta.pt"
        if meta_path.exists() and self.strategy == EnsembleStrategy.STACKING:
            self._initialize_meta_model()
            checkpoint = torch.load(meta_path)
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Loaded meta-model")

        logger.info(f"📂 Ensemble config loaded from {filepath}")


# Convenience function
def create_ensemble(
    state_size: int,
    action_size: int,
    models_dir: str,
    strategy: str = EnsembleStrategy.WEIGHTED_AVG,
    ensemble_size: int = 3,
    validation_scores: Optional[Dict[str, float]] = None,
    config: Optional[Dict] = None
) -> EnsembleManager:
    """
    Create ensemble from a directory of models

    Args:
        state_size: State space size
        action_size: Action space size
        models_dir: Directory containing trained models
        strategy: Ensemble strategy
        ensemble_size: Number of models to use
        validation_scores: Optional validation scores for each model
        config: Configuration dictionary

    Returns:
        Configured ensemble manager
    """
    manager = EnsembleManager(
        state_size=state_size,
        action_size=action_size,
        strategy=strategy,
        ensemble_size=ensemble_size,
        config=config
    )

    manager.load_ensemble_from_directory(
        models_dir=models_dir,
        validation_scores=validation_scores,
        max_models=ensemble_size
    )

    return manager


__all__ = [
    "EnsembleManager",
    "EnsembleStrategy",
    "ModelInfo",
    "StackingMetaModel",
    "create_ensemble"
]
