"""
Quick test script for walk-forward validation and model artifacts
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test 1: Walk-forward validation
print("=" * 60)
print("TEST 1: Walk-Forward Validation")
print("=" * 60)

try:
    from ml_forecasting.models.validation import WalkForwardValidator

    # Create synthetic data
    dates = pd.date_range(start='2020-01-01', periods=600, freq='D')
    prices = 100 + np.cumsum(np.random.randn(600) * 2)
    test_data = pd.Series(prices, index=dates)

    # Simple test model function
    def simple_model(train_data, steps):
        """Simple moving average model for testing"""
        return np.full(steps, train_data.tail(10).mean())

    # Run validation
    validator = WalkForwardValidator(
        initial_train_size=400,
        step_size=50,
        forecast_horizon=7
    )

    print("Running walk-forward validation...")
    results = validator.validate(test_data, simple_model, verbose=True)

    if 'error' in results:
        print(f"❌ Validation failed: {results['error']}")
    else:
        print("\n✅ Walk-forward validation successful!")
        print(f"   Number of folds: {results.get('num_folds', 0)}")
        print(f"   Mean MAPE: {results.get('mape_mean', 0):.2f}%")
        print(f"   Direction Accuracy: {results.get('direction_accuracy_mean', 0):.2f}%")

except Exception as e:
    print(f"❌ Walk-forward validation test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Model Artifact Manager
print("=" * 60)
print("TEST 2: Model Artifact Management")
print("=" * 60)

try:
    from ml_forecasting.models.artifact_manager import ModelArtifactManager

    # Initialize manager
    manager = ModelArtifactManager(base_dir='models/saved')
    print("✅ Artifact manager initialized")

    # Create a dummy model (simple dict for testing)
    dummy_model = {
        'type': 'test_model',
        'parameters': {'alpha': 0.5},
        'trained_at': datetime.now().isoformat()
    }

    # Test save
    print("\nTesting model save...")
    metadata = {
        'data_hash': 'test123',
        'validation': {
            'mape_mean': 5.5,
            'direction_accuracy_mean': 72.0
        }
    }
    model_id = manager.save_model(
        dummy_model,
        symbol='TEST.NS',
        model_type='test',
        metadata=metadata
    )

    if model_id:
        print(f"✅ Model saved successfully! ID: {model_id}")
    else:
        print("❌ Model save failed")

    # Test load
    print("\nTesting model load...")
    loaded_model, loaded_metadata = manager.load_model('TEST.NS', 'test')

    if loaded_model is not None:
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {loaded_model.get('type')}")
        print(f"   Metadata keys: {list(loaded_metadata.keys())}")
    else:
        print("❌ Model load failed")

    # Test list models
    print("\nTesting model listing...")
    models = manager.list_models()
    print(f"✅ Found {len(models)} model(s) in registry")

    # Test staleness check
    print("\nTesting staleness check...")
    is_stale = manager.is_model_stale('TEST.NS', 'test', max_age_days=7)
    print(f"✅ Model is {'stale' if is_stale else 'fresh'}")

except Exception as e:
    print(f"❌ Artifact management test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Integration with EnsembleModel
print("=" * 60)
print("TEST 3: EnsembleModel Integration")
print("=" * 60)

try:
    from ml_forecasting.models.ensemble_model import EnsembleModel

    # Create test data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 1.5), index=dates)

    # Test with caching enabled
    print("Testing EnsembleModel with caching enabled...")
    model_cached = EnsembleModel(use_cached_models=True)
    print(f"✅ EnsembleModel initialized (caching: enabled)")
    print(f"   Artifact manager available: {model_cached.artifact_manager is not None}")

    # Test prediction
    print("\nTesting prediction...")
    result = model_cached.predict(prices, steps=7, symbol='TEST.NS')

    if result and 'predictions' in result:
        print(f"✅ Prediction successful!")
        print(f"   Predictions: {len(result['predictions'])} steps")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Risk score: {result['risk_score']}")
        print(f"   Method: {result['method']}")
    else:
        print("❌ Prediction failed")

    # Test without caching
    print("\nTesting EnsembleModel with caching disabled...")
    model_no_cache = EnsembleModel(use_cached_models=False)
    print(f"✅ EnsembleModel initialized (caching: disabled)")
    print(f"   Artifact manager: {model_no_cache.artifact_manager}")

except Exception as e:
    print(f"❌ EnsembleModel integration test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All basic tests completed. Check for ✅ (success) or ❌ (failure) above.")
print("\nTo test with real data, run:")
print("  python ml_forecasting/train_models.py --symbol RELIANCE.NS --period 1y")
print("=" * 60)
