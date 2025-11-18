"""
Walk-forward validation and time-series cross-validation utilities
Prevents data leakage and provides realistic performance estimates
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Implements walk-forward validation for time-series models.
    Prevents look-ahead bias and data leakage.
    """

    def __init__(self, initial_train_size=500, step_size=30, forecast_horizon=7):
        """
        Args:
            initial_train_size: Minimum number of data points for initial training
            step_size: How many days to move forward between validations
            forecast_horizon: Number of days to forecast ahead
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon

    def validate(self, data, model_func, verbose=False):
        """
        Perform walk-forward validation on a model.

        Args:
            data: Time series data (pandas Series or DataFrame with Close column)
            model_func: Function that takes (train_data, steps) and returns predictions
            verbose: Print progress information

        Returns:
            Dictionary with validation metrics and fold results
        """
        if isinstance(data, pd.DataFrame):
            data = data['Close'] if 'Close' in data.columns else data.iloc[:, -1]

        data = data.dropna()

        if len(data) < self.initial_train_size + self.forecast_horizon:
            return {
                'error': 'Insufficient data for walk-forward validation',
                'required': self.initial_train_size + self.forecast_horizon,
                'available': len(data)
            }

        fold_results = []
        train_end_dates = []

        # Walk forward through the data
        for train_end in range(self.initial_train_size,
                               len(data) - self.forecast_horizon + 1,
                               self.step_size):

            # Split data
            train_data = data[:train_end]
            test_start = train_end
            test_end = min(train_end + self.forecast_horizon, len(data))
            test_data = data[test_start:test_end]

            if len(test_data) == 0:
                continue

            try:
                # Generate predictions using only training data
                predictions = model_func(train_data, steps=len(test_data))

                # Handle different return types
                if isinstance(predictions, dict) and 'predictions' in predictions:
                    pred_values = predictions['predictions']
                else:
                    pred_values = predictions

                pred_values = np.array(pred_values).flatten()
                actual_values = test_data.values

                # Ensure same length
                min_len = min(len(pred_values), len(actual_values))
                pred_values = pred_values[:min_len]
                actual_values = actual_values[:min_len]

                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(actual_values, pred_values)
                fold_metrics['train_end'] = train_end
                fold_metrics['test_size'] = len(actual_values)
                fold_metrics['train_size'] = len(train_data)

                if hasattr(data, 'index'):
                    fold_metrics['train_end_date'] = data.index[train_end - 1]

                fold_results.append(fold_metrics)
                train_end_dates.append(train_end)

                if verbose:
                    print(f"Fold {len(fold_results)}: Train={len(train_data)}, Test={len(test_data)}, "
                          f"MAPE={fold_metrics['mape']:.2f}%, Direction Acc={fold_metrics['direction_accuracy']:.2f}%")

            except Exception as e:
                if verbose:
                    print(f"Fold at {train_end} failed: {str(e)}")
                continue

        if len(fold_results) == 0:
            return {
                'error': 'No successful validation folds',
                'folds_attempted': len(range(self.initial_train_size, len(data) - self.forecast_horizon, self.step_size))
            }

        # Aggregate results across all folds
        aggregated = self._aggregate_folds(fold_results)
        aggregated['fold_results'] = fold_results
        aggregated['num_folds'] = len(fold_results)
        aggregated['validation_type'] = 'walk_forward'

        return aggregated

    def _calculate_metrics(self, actual, predicted):
        """Calculate comprehensive metrics for a single fold"""
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Basic error metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)

        # Percentage errors
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100

        # Direction accuracy (most important for trading)
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(predicted) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 50.0

        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Mean percentage error (bias detection)
        mpe = np.mean((actual - predicted) / (actual + 1e-10)) * 100

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mpe': float(mpe),
            'direction_accuracy': float(direction_accuracy),
            'r_squared': float(r_squared)
        }

    def _aggregate_folds(self, fold_results):
        """Aggregate metrics across all folds"""
        metrics = {}

        # Calculate mean and std for each metric
        for metric in ['mae', 'mse', 'rmse', 'mape', 'mpe', 'direction_accuracy', 'r_squared']:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if len(values) > 0:
                metrics[f'{metric}_mean'] = float(np.mean(values))
                metrics[f'{metric}_std'] = float(np.std(values))
                metrics[f'{metric}_min'] = float(np.min(values))
                metrics[f'{metric}_max'] = float(np.max(values))

        return metrics

    def time_series_cv_split(self, data, n_splits=5):
        """
        Generate train/test splits for time-series cross-validation.

        Args:
            data: Time series data
            n_splits: Number of splits to generate

        Yields:
            (train_indices, test_indices) tuples
        """
        if isinstance(data, pd.DataFrame):
            data_len = len(data)
        else:
            data_len = len(data)

        # Calculate split points
        min_train = max(self.initial_train_size, data_len // (n_splits + 1))

        for i in range(n_splits):
            train_end = min_train + (i * (data_len - min_train) // n_splits)
            test_end = min(train_end + self.forecast_horizon, data_len)

            if test_end > train_end:
                train_indices = list(range(train_end))
                test_indices = list(range(train_end, test_end))
                yield train_indices, test_indices


class ModelBacktester:
    """
    Backtesting utility for evaluating model performance over time.
    """

    def __init__(self, transaction_cost=0.001):
        """
        Args:
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
        """
        self.transaction_cost = transaction_cost

    def backtest_trading_strategy(self, data, predictions, confidence_threshold=0.6):
        """
        Backtest a simple trading strategy based on model predictions.

        Args:
            data: Historical price data
            predictions: Model predictions with confidence scores
            confidence_threshold: Minimum confidence to take a position

        Returns:
            Dictionary with backtest results
        """
        if len(predictions) == 0:
            return {'error': 'No predictions provided'}

        # Simple strategy: buy if predicted increase > 1% with high confidence
        trades = []
        portfolio_value = [100.0]  # Start with $100
        positions = []

        for i, (pred, conf) in enumerate(zip(predictions.get('predictions', []),
                                             predictions.get('confidences', []))):
            if i >= len(data) - 1:
                break

            current_price = data.iloc[i]
            next_price = data.iloc[i + 1] if i + 1 < len(data) else current_price

            predicted_return = (pred - current_price) / current_price

            # Trading logic
            if conf >= confidence_threshold and abs(predicted_return) > 0.01:
                # Enter position
                direction = 1 if predicted_return > 0 else -1

                # Calculate return (simplified)
                actual_return = (next_price - current_price) / current_price
                strategy_return = direction * actual_return - self.transaction_cost

                portfolio_value.append(portfolio_value[-1] * (1 + strategy_return))

                trades.append({
                    'date': data.index[i] if hasattr(data, 'index') else i,
                    'direction': 'long' if direction > 0 else 'short',
                    'entry_price': current_price,
                    'exit_price': next_price,
                    'return': strategy_return,
                    'confidence': conf
                })
            else:
                # Hold position
                portfolio_value.append(portfolio_value[-1])

        # Calculate strategy metrics
        total_return = (portfolio_value[-1] - 100.0) / 100.0
        num_trades = len(trades)
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0

        # Calculate Sharpe ratio (simplified)
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

        return {
            'total_return': float(total_return * 100),
            'final_portfolio_value': float(portfolio_value[-1]),
            'num_trades': num_trades,
            'win_rate': float(win_rate * 100),
            'sharpe_ratio': float(sharpe),
            'trades': trades,
            'portfolio_curve': portfolio_value
        }


def validate_model_robustness(model, data, n_iterations=10, noise_level=0.01):
    """
    Test model robustness by adding noise to input data.

    Args:
        model: Model function to test
        data: Time series data
        n_iterations: Number of noise iterations
        noise_level: Standard deviation of noise as fraction of data std

    Returns:
        Robustness metrics
    """
    baseline_pred = model(data, steps=7)
    if isinstance(baseline_pred, dict):
        baseline_pred = baseline_pred['predictions']
    baseline_pred = np.array(baseline_pred)

    predictions = []

    for i in range(n_iterations):
        # Add noise to data
        noise = np.random.randn(len(data)) * data.std() * noise_level
        noisy_data = data + noise

        # Get prediction
        pred = model(noisy_data, steps=7)
        if isinstance(pred, dict):
            pred = pred['predictions']
        predictions.append(np.array(pred))

    predictions = np.array(predictions)

    # Calculate stability metrics
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    # Coefficient of variation
    cv = np.mean(pred_std / (pred_mean + 1e-10))

    return {
        'prediction_stability_cv': float(cv),
        'prediction_std_mean': float(np.mean(pred_std)),
        'prediction_std_max': float(np.max(pred_std)),
        'robustness_score': float(max(0, 1 - cv))  # Higher is better
    }
