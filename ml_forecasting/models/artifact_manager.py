"""
Model artifact management system for saving/loading trained models.
Prevents retraining on every prediction and enables model versioning.
"""

import os
import pickle
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from sklearn.preprocessing import MinMaxScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class ModelArtifactManager:
    """
    Manages saving, loading, and versioning of trained models.
    """

    def __init__(self, base_dir='models/saved'):
        """
        Args:
            base_dir: Base directory for saving model artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / 'registry.json'
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load the model registry from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_registry(self):
        """Save the model registry to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            warnings.warn(f"Failed to save registry: {e}")

    def _generate_model_id(self, symbol, model_type, data_hash):
        """Generate unique model ID"""
        combined = f"{symbol}_{model_type}_{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _generate_data_hash(self, data):
        """Generate hash of training data for versioning"""
        try:
            if isinstance(data, pd.Series):
                data_str = f"{len(data)}_{data.iloc[0]:.6f}_{data.iloc[-1]:.6f}_{data.mean():.6f}"
            elif isinstance(data, pd.DataFrame):
                data_str = f"{len(data)}_{data.iloc[0, 0]:.6f}_{data.iloc[-1, 0]:.6f}"
            else:
                data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()[:8]
        except Exception:
            return "default"

    def save_model(self, model, symbol, model_type, metadata=None):
        """
        Save a trained model with metadata.

        Args:
            model: The trained model object (sklearn, statsmodels, or custom)
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            model_type: Type of model (e.g., 'arima', 'rf', 'lstm')
            metadata: Additional metadata dictionary

        Returns:
            model_id: Unique identifier for the saved model
        """
        try:
            # Generate model ID
            data_hash = metadata.get('data_hash', 'default') if metadata else 'default'
            model_id = self._generate_model_id(symbol, model_type, data_hash)

            # Create model directory
            model_dir = self.base_dir / model_id
            model_dir.mkdir(exist_ok=True)

            # Save model based on type
            model_path = model_dir / 'model.pkl'

            if _HAS_JOBLIB:
                joblib.dump(model, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # Save metadata
            full_metadata = {
                'model_id': model_id,
                'symbol': symbol,
                'model_type': model_type,
                'created_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                **(metadata or {})
            }

            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)

            # Update registry
            registry_key = f"{symbol}_{model_type}"
            if registry_key not in self.registry:
                self.registry[registry_key] = []

            self.registry[registry_key].append({
                'model_id': model_id,
                'created_at': full_metadata['created_at'],
                'data_hash': data_hash
            })

            self._save_registry()

            return model_id

        except Exception as e:
            warnings.warn(f"Failed to save model: {e}")
            return None

    def load_model(self, symbol, model_type, model_id=None):
        """
        Load a trained model.

        Args:
            symbol: Stock symbol
            model_type: Type of model
            model_id: Specific model ID (if None, loads latest)

        Returns:
            (model, metadata) tuple or (None, None) if not found
        """
        try:
            registry_key = f"{symbol}_{model_type}"

            # Get model ID
            if model_id is None:
                # Load latest model
                if registry_key not in self.registry or len(self.registry[registry_key]) == 0:
                    return None, None
                model_id = self.registry[registry_key][-1]['model_id']

            # Load model
            model_dir = self.base_dir / model_id
            if not model_dir.exists():
                return None, None

            model_path = model_dir / 'model.pkl'
            metadata_path = model_dir / 'metadata.json'

            if not model_path.exists():
                return None, None

            # Load model
            if _HAS_JOBLIB:
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            return model, metadata

        except Exception as e:
            warnings.warn(f"Failed to load model: {e}")
            return None, None

    def model_exists(self, symbol, model_type, data_hash=None):
        """
        Check if a model exists for given parameters.

        Args:
            symbol: Stock symbol
            model_type: Type of model
            data_hash: Optional data hash to check for exact match

        Returns:
            Boolean indicating if model exists
        """
        registry_key = f"{symbol}_{model_type}"

        if registry_key not in self.registry:
            return False

        if data_hash is None:
            return len(self.registry[registry_key]) > 0

        # Check for specific data hash
        for entry in self.registry[registry_key]:
            if entry.get('data_hash') == data_hash:
                return True

        return False

    def save_scaler(self, scaler, symbol, model_type):
        """
        Save a data scaler/normalizer.

        Args:
            scaler: Fitted scaler object (e.g., MinMaxScaler)
            symbol: Stock symbol
            model_type: Associated model type

        Returns:
            Boolean indicating success
        """
        try:
            scaler_dir = self.base_dir / 'scalers'
            scaler_dir.mkdir(exist_ok=True)

            scaler_path = scaler_dir / f"{symbol}_{model_type}_scaler.pkl"

            if _HAS_JOBLIB:
                joblib.dump(scaler, scaler_path)
            else:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)

            return True

        except Exception as e:
            warnings.warn(f"Failed to save scaler: {e}")
            return False

    def load_scaler(self, symbol, model_type):
        """
        Load a data scaler.

        Args:
            symbol: Stock symbol
            model_type: Associated model type

        Returns:
            Scaler object or None if not found
        """
        try:
            scaler_path = self.base_dir / 'scalers' / f"{symbol}_{model_type}_scaler.pkl"

            if not scaler_path.exists():
                return None

            if _HAS_JOBLIB:
                return joblib.load(scaler_path)
            else:
                with open(scaler_path, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            warnings.warn(f"Failed to load scaler: {e}")
            return None

    def save_training_results(self, symbol, model_type, results):
        """
        Save training/validation results.

        Args:
            symbol: Stock symbol
            model_type: Model type
            results: Dictionary with training metrics

        Returns:
            Boolean indicating success
        """
        try:
            results_dir = self.base_dir / 'training_results'
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = results_dir / f"{symbol}_{model_type}_{timestamp}.json"

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return True

        except Exception as e:
            warnings.warn(f"Failed to save training results: {e}")
            return False

    def list_models(self, symbol=None, model_type=None):
        """
        List available models.

        Args:
            symbol: Filter by symbol (optional)
            model_type: Filter by model type (optional)

        Returns:
            List of model metadata dictionaries
        """
        models = []

        for key, entries in self.registry.items():
            sym, mtype = key.rsplit('_', 1)

            # Apply filters
            if symbol and sym != symbol:
                continue
            if model_type and mtype != model_type:
                continue

            for entry in entries:
                models.append({
                    'symbol': sym,
                    'model_type': mtype,
                    **entry
                })

        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_model(self, model_id):
        """
        Delete a model and its artifacts.

        Args:
            model_id: Model ID to delete

        Returns:
            Boolean indicating success
        """
        try:
            model_dir = self.base_dir / model_id

            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)

            # Update registry
            for key in self.registry:
                self.registry[key] = [
                    e for e in self.registry[key]
                    if e.get('model_id') != model_id
                ]

            self._save_registry()
            return True

        except Exception as e:
            warnings.warn(f"Failed to delete model: {e}")
            return False

    def cleanup_old_models(self, keep_latest=3):
        """
        Remove old model versions, keeping only the latest N.

        Args:
            keep_latest: Number of versions to keep per symbol/model_type

        Returns:
            Number of models deleted
        """
        deleted = 0

        for key in self.registry:
            entries = self.registry[key]
            if len(entries) <= keep_latest:
                continue

            # Sort by date and keep latest
            entries_sorted = sorted(entries, key=lambda x: x.get('created_at', ''), reverse=True)
            to_delete = entries_sorted[keep_latest:]

            for entry in to_delete:
                if self.delete_model(entry['model_id']):
                    deleted += 1

        return deleted

    def get_cache_key(self, data, symbol, model_type, steps):
        """
        Generate cache key for prediction caching.

        Args:
            data: Training data
            symbol: Stock symbol
            model_type: Model type
            steps: Forecast steps

        Returns:
            Cache key string
        """
        data_hash = self._generate_data_hash(data)
        return f"{symbol}_{model_type}_{data_hash}_{steps}"

    def is_model_stale(self, symbol, model_type, max_age_days=7):
        """
        Check if the latest model is stale.

        Args:
            symbol: Stock symbol
            model_type: Model type
            max_age_days: Maximum age in days

        Returns:
            Boolean indicating if model is stale
        """
        registry_key = f"{symbol}_{model_type}"

        if registry_key not in self.registry or len(self.registry[registry_key]) == 0:
            return True

        latest = self.registry[registry_key][-1]
        created_at = datetime.fromisoformat(latest['created_at'])
        age = (datetime.now() - created_at).days

        return age > max_age_days
