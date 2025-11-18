"""
Training pipeline for ML models.
Separates training from inference and saves artifacts for reuse.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_forecasting.models.artifact_manager import ModelArtifactManager
from ml_forecasting.models.validation import WalkForwardValidator
from utils.data_fetcher import fetch_stock_data

# Optional imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


class ModelTrainer:
    """
    Centralized training pipeline for all model types.
    """

    def __init__(self, artifact_manager=None):
        """
        Args:
            artifact_manager: ModelArtifactManager instance
        """
        self.artifact_manager = artifact_manager or ModelArtifactManager()
        self.validator = WalkForwardValidator(
            initial_train_size=500,
            step_size=30,
            forecast_horizon=7
        )

    def train_arima(self, data, symbol, validate=True):
        """Train and save ARIMA model"""
        if not _HAS_STATSMODELS:
            print("statsmodels not available, skipping ARIMA")
            return None

        print(f"Training ARIMA model for {symbol}...")

        try:
            series = pd.Series(data.values, index=data.index)

            # Determine differencing order
            try:
                adf_p = adfuller(series.dropna().values, autolag='AIC')[1]
                d = 0 if adf_p < 0.05 else 1
            except Exception:
                d = 1

            # Grid search for best parameters
            best_aic = np.inf
            best_order = (1, d, 1)
            best_model = None

            for p in [0, 1, 2]:
                for q in [0, 1, 2]:
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        res = model.fit(method_kwargs={"warn_convergence": False})
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p, d, q)
                            best_model = res
                    except Exception:
                        continue

            if best_model is None:
                print("Failed to train ARIMA model")
                return None

            # Validate if requested
            validation_results = {}
            if validate:
                print("Running walk-forward validation...")

                def arima_predict_func(train_data, steps):
                    try:
                        temp_series = pd.Series(train_data.values)
                        temp_model = ARIMA(temp_series, order=best_order)
                        temp_res = temp_model.fit(method_kwargs={"warn_convergence": False})
                        return temp_res.forecast(steps=steps)
                    except Exception:
                        return np.full(steps, train_data.iloc[-1])

                validation_results = self.validator.validate(
                    data, arima_predict_func, verbose=True
                )

            # Save model
            metadata = {
                'order': best_order,
                'aic': float(best_aic),
                'data_points': len(data),
                'data_hash': self.artifact_manager._generate_data_hash(data),
                'validation': validation_results
            }

            model_id = self.artifact_manager.save_model(
                best_model, symbol, 'arima', metadata
            )

            if validation_results and 'mape_mean' in validation_results:
                print(f"ARIMA trained successfully! Model ID: {model_id}")
                print(f"  Validation MAPE: {validation_results['mape_mean']:.2f}%")
                print(f"  Direction Accuracy: {validation_results['direction_accuracy_mean']:.2f}%")

            return model_id

        except Exception as e:
            print(f"Error training ARIMA: {e}")
            return None

    def train_random_forest(self, data, symbol, validate=True):
        """Train and save Random Forest model"""
        if not _HAS_SKLEARN:
            print("sklearn not available, skipping Random Forest")
            return None

        print(f"Training Random Forest model for {symbol}...")

        try:
            # Build lag features
            look_back = 15
            X, y = self._build_lag_features(data.values, look_back)

            if len(y) < 50:
                print("Insufficient data for Random Forest")
                return None

            # Train model
            rf = RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)

            # Validate if requested
            validation_results = {}
            if validate:
                print("Running walk-forward validation...")

                def rf_predict_func(train_data, steps):
                    try:
                        X_train, y_train = self._build_lag_features(train_data.values, look_back)
                        if len(y_train) < 20:
                            return np.full(steps, train_data.iloc[-1])

                        temp_rf = RandomForestRegressor(
                            n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
                        )
                        temp_rf.fit(X_train, y_train)

                        window = list(X_train[-1])
                        preds = []
                        for _ in range(steps):
                            inp = np.array(window[-look_back:]).reshape(1, -1)
                            yhat = float(temp_rf.predict(inp)[0])
                            preds.append(yhat)
                            window.append(yhat)
                        return np.array(preds)
                    except Exception:
                        return np.full(steps, train_data.iloc[-1])

                validation_results = self.validator.validate(
                    data, rf_predict_func, verbose=True
                )

            # Save model
            metadata = {
                'n_estimators': 150,
                'max_depth': 8,
                'look_back': look_back,
                'data_points': len(data),
                'data_hash': self.artifact_manager._generate_data_hash(data),
                'validation': validation_results
            }

            model_id = self.artifact_manager.save_model(
                rf, symbol, 'random_forest', metadata
            )

            if validation_results and 'mape_mean' in validation_results:
                print(f"Random Forest trained successfully! Model ID: {model_id}")
                print(f"  Validation MAPE: {validation_results['mape_mean']:.2f}%")
                print(f"  Direction Accuracy: {validation_results['direction_accuracy_mean']:.2f}%")

            return model_id

        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return None

    def train_lstm(self, data, symbol, validate=True):
        """Train and save LSTM model"""
        if not _HAS_TF or not _HAS_SKLEARN:
            print("TensorFlow/sklearn not available, skipping LSTM")
            return None

        print(f"Training LSTM model for {symbol}...")

        try:
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

            # Build sequences
            look_back = 20
            X, y = [], []
            for i in range(look_back, len(scaled)):
                X.append(scaled[i - look_back:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)

            if len(y) < 60:
                print("Insufficient data for LSTM")
                return None

            # Split data
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_val, y_val = X[split:], y[split:]

            X_train = X_train.reshape((-1, look_back, 1))
            X_val = X_val.reshape((-1, look_back, 1))

            # Build model
            tf.random.set_seed(42)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(look_back, 1)),
                tf.keras.layers.LSTM(32, activation='tanh'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

            # Train
            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                verbose=1,
                validation_data=(X_val, y_val)
            )

            # Validate if requested
            validation_results = {}
            if validate:
                print("Running walk-forward validation...")

                def lstm_predict_func(train_data, steps):
                    try:
                        temp_scaler = MinMaxScaler(feature_range=(0, 1))
                        temp_scaled = temp_scaler.fit_transform(
                            train_data.values.reshape(-1, 1)
                        ).flatten()

                        temp_X, temp_y = [], []
                        for i in range(look_back, len(temp_scaled)):
                            temp_X.append(temp_scaled[i - look_back:i])
                            temp_y.append(temp_scaled[i])
                        temp_X, temp_y = np.array(temp_X), np.array(temp_y)

                        if len(temp_y) < 20:
                            return np.full(steps, train_data.iloc[-1])

                        temp_X = temp_X.reshape((-1, look_back, 1))

                        temp_model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(look_back, 1)),
                            tf.keras.layers.LSTM(32, activation='tanh'),
                            tf.keras.layers.Dense(1)
                        ])
                        temp_model.compile(optimizer='adam', loss='mse')
                        temp_model.fit(temp_X, temp_y, epochs=5, batch_size=32, verbose=0)

                        window = temp_scaled[-look_back:].tolist()
                        preds_scaled = []
                        for _ in range(steps):
                            arr = np.array(window[-look_back:]).reshape((1, look_back, 1))
                            yhat = float(temp_model.predict(arr, verbose=0)[0, 0])
                            preds_scaled.append(yhat)
                            window.append(yhat)

                        preds = temp_scaler.inverse_transform(
                            np.array(preds_scaled).reshape(-1, 1)
                        ).flatten()
                        return preds
                    except Exception:
                        return np.full(steps, train_data.iloc[-1])

                validation_results = self.validator.validate(
                    data, lstm_predict_func, verbose=True
                )

            # Save model and scaler
            metadata = {
                'look_back': look_back,
                'epochs': 10,
                'data_points': len(data),
                'data_hash': self.artifact_manager._generate_data_hash(data),
                'validation': validation_results
            }

            model_id = self.artifact_manager.save_model(
                model, symbol, 'lstm', metadata
            )
            self.artifact_manager.save_scaler(scaler, symbol, 'lstm')

            if validation_results and 'mape_mean' in validation_results:
                print(f"LSTM trained successfully! Model ID: {model_id}")
                print(f"  Validation MAPE: {validation_results['mape_mean']:.2f}%")
                print(f"  Direction Accuracy: {validation_results['direction_accuracy_mean']:.2f}%")

            return model_id

        except Exception as e:
            print(f"Error training LSTM: {e}")
            return None

    def _build_lag_features(self, series, look_back=10):
        """Build lag features for supervised learning"""
        values = np.asarray(series, dtype=float)
        X, y = [], []
        for i in range(look_back, len(values)):
            X.append(values[i - look_back:i])
            y.append(values[i])
        return np.array(X), np.array(y)

    def train_all_models(self, symbol, period='2y', validate=True):
        """
        Train all available models for a given stock.

        Args:
            symbol: Stock symbol
            period: Data period to fetch
            validate: Whether to run walk-forward validation

        Returns:
            Dictionary with model IDs
        """
        print(f"\n{'='*60}")
        print(f"Training pipeline for {symbol}")
        print(f"{'='*60}\n")

        # Fetch data
        print(f"Fetching {period} of data for {symbol}...")
        data = fetch_stock_data(symbol, period=period)

        if data is None or data.empty:
            print(f"Failed to fetch data for {symbol}")
            return {}

        if 'Close' in data.columns:
            close_data = data['Close']
        else:
            close_data = data.iloc[:, -1]

        close_data = close_data.dropna()

        print(f"Loaded {len(close_data)} data points\n")

        # Train models
        results = {}

        if _HAS_STATSMODELS:
            results['arima'] = self.train_arima(close_data, symbol, validate)

        if _HAS_SKLEARN:
            results['random_forest'] = self.train_random_forest(close_data, symbol, validate)

        if _HAS_TF and _HAS_SKLEARN:
            results['lstm'] = self.train_lstm(close_data, symbol, validate)

        print(f"\n{'='*60}")
        print(f"Training complete for {symbol}")
        print(f"Models trained: {[k for k, v in results.items() if v is not None]}")
        print(f"{'='*60}\n")

        return results


def main():
    """Command-line interface for training"""
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., RELIANCE.NS)')
    parser.add_argument('--period', type=str, default='2y', help='Data period (default: 2y)')
    parser.add_argument('--models', type=str, default='all', help='Models to train: all, arima, rf, lstm')
    parser.add_argument('--no-validate', action='store_true', help='Skip walk-forward validation')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup old models (keep latest 3)')

    args = parser.parse_args()

    # Initialize
    artifact_manager = ModelArtifactManager()
    trainer = ModelTrainer(artifact_manager)

    # Cleanup if requested
    if args.cleanup:
        print("Cleaning up old models...")
        deleted = artifact_manager.cleanup_old_models(keep_latest=3)
        print(f"Deleted {deleted} old model(s)\n")

    # Train models
    validate = not args.no_validate

    if args.models == 'all':
        trainer.train_all_models(args.symbol, args.period, validate)
    else:
        # Fetch data
        data = fetch_stock_data(args.symbol, period=args.period)
        if data is None or data.empty:
            print(f"Failed to fetch data for {args.symbol}")
            return

        close_data = data['Close'] if 'Close' in data.columns else data.iloc[:, -1]
        close_data = close_data.dropna()

        # Train specific model
        if args.models == 'arima':
            trainer.train_arima(close_data, args.symbol, validate)
        elif args.models == 'rf':
            trainer.train_random_forest(close_data, args.symbol, validate)
        elif args.models == 'lstm':
            trainer.train_lstm(close_data, args.symbol, validate)
        else:
            print(f"Unknown model type: {args.models}")


if __name__ == '__main__':
    main()
