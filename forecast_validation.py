from __future__ import annotations

"""
Rolling validation for air-quality forecasting models.

Why this file exists:
- A single train/test split is useful for an initial baseline, but it is not a strong
  validation protocol for time-series forecasting.
- This module compares ARIMA and SARIMA using rolling one-step-ahead evaluation
  with an expanding training window.
- The output is intended to be reusable by later reporting or dashboard layers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import logging
import math
import warnings

import pandas as pd

from forecast_dataset import DatasetError, load_city_series

try:
    # These imports are isolated so dependency failures are explicit and actionable.
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecast_validation.py. "
        "Install it with: pip install statsmodels"
    ) from exc


# Centralized logging is important because rolling validation performs repeated model
# fits. When one step fails, the log must make it obvious which model and which step
# caused the failure.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("air_quality_forecast_validation")


class ValidationError(Exception):
    """
    Why this exists:
    - Validation-stage failures should be isolated from data-loading failures and
      model-specific exceptions.
    """
    pass


@dataclass(frozen=True)
class ValidationConfig:
    """
    Why this exists:
    - Validation parameters must be explicit and immutable so that results remain
      reproducible across reruns and future comparisons.
    """
    city_name: str = "Dhaka"
    database_path: str = "air_quality.db"
    initial_train_size: int = 120
    validation_horizon: int = 24
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    predictions_output_csv: str = "forecast_validation_predictions.csv"
    metrics_output_csv: str = "forecast_validation_metrics.csv"


@dataclass(frozen=True)
class ModelMetrics:
    """
    Why this exists:
    - A typed metrics object makes it easier to keep comparison output structured
      and consistent between models.
    """
    model_name: str
    city_name: str
    initial_train_size: int
    validation_horizon: int
    rmse: float
    mae: float


def validate_series(series: pd.Series, config: ValidationConfig) -> pd.Series:
    """
    Why this exists:
    - Rolling validation needs a chronologically ordered, hourly-indexed signal.
    - Setting an explicit frequency avoids downstream ambiguity in forecasting APIs.
    """
    if series.empty:
        raise ValidationError("The input time series is empty.")

    if not series.index.is_monotonic_increasing:
        raise ValidationError("The input time series must be sorted chronologically.")

    # Assigning an explicit hourly frequency eliminates ambiguity and prevents model
    # code from inferring the cadence implicitly at fit time.
    series = series.asfreq("h")

    if series.isna().any():
        missing_count = int(series.isna().sum())
        raise ValidationError(
            f"The time series contains {missing_count} missing hourly observations after frequency assignment. "
            f"Gap handling must be defined before rolling validation."
        )

    minimum_required = config.initial_train_size + config.validation_horizon
    if len(series) < minimum_required:
        raise ValidationError(
            f"Insufficient observations for rolling validation. "
            f"series_length={len(series)}, required_minimum={minimum_required}"
        )

    return series


def compute_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Why this exists:
    - MAE is a standard, interpretable measure of average absolute forecast error.
    """
    if len(actual) != len(predicted):
        raise ValidationError("MAE computation requires equal-length actual and predicted sequences.")

    if not actual:
        raise ValidationError("MAE computation received empty input.")

    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return float(sum(absolute_errors) / len(absolute_errors))


def compute_rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Why this exists:
    - RMSE penalizes larger misses more strongly and is useful for comparing model quality.
    """
    if len(actual) != len(predicted):
        raise ValidationError("RMSE computation requires equal-length actual and predicted sequences.")

    if not actual:
        raise ValidationError("RMSE computation received empty input.")

    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return float(math.sqrt(sum(squared_errors) / len(squared_errors)))


def forecast_next_arima(
    history: pd.Series,
    order: Tuple[int, int, int],
) -> float:
    """
    Why this exists:
    - ARIMA one-step forecasting is isolated so that model-specific fitting logic
      does not pollute the validation loop.
    """
    try:
        model = ARIMA(history, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=1)
        return float(forecast.iloc[0])

    except Exception as exc:
        raise ValidationError(f"ARIMA one-step forecast failed: {exc}") from exc


def forecast_next_sarima(
    history: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> float:
    """
    Why this exists:
    - SARIMA one-step forecasting is isolated from the orchestration layer so seasonal
      model complexity remains encapsulated.
    """
    try:
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted_model = model.fit(disp=False)
        forecast_result = fitted_model.get_forecast(steps=1)
        return float(forecast_result.predicted_mean.iloc[0])

    except Exception as exc:
        raise ValidationError(f"SARIMA one-step forecast failed: {exc}") from exc


def run_rolling_validation_for_model(
    model_name: str,
    series: pd.Series,
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Why this exists:
    - This function implements expanding-window one-step-ahead validation for one model.
    - Each model is evaluated under the exact same protocol, which is necessary for
      a fair comparison.
    """
    records: List[Dict[str, object]] = []

    validation_start = config.initial_train_size
    validation_end = config.initial_train_size + config.validation_horizon

    for step_index in range(validation_start, validation_end):
        history = series.iloc[:step_index]
        actual_timestamp = series.index[step_index]
        actual_value = float(series.iloc[step_index])

        LOGGER.info(
            "Rolling validation | model=%s | step=%d/%d | history_size=%d | forecast_timestamp=%s",
            model_name,
            step_index - validation_start + 1,
            config.validation_horizon,
            len(history),
            actual_timestamp.isoformat(),
        )

        # Convergence and frequency warnings from repeated fitting are suppressed here
        # because the validation output is intended to focus on model comparison rather
        # than verbose solver internals. Genuine exceptions still propagate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if model_name == "ARIMA":
                predicted_value = forecast_next_arima(
                    history=history,
                    order=config.arima_order,
                )
            elif model_name == "SARIMA":
                predicted_value = forecast_next_sarima(
                    history=history,
                    order=config.sarima_order,
                    seasonal_order=config.sarima_seasonal_order,
                )
            else:
                raise ValidationError(f"Unsupported model_name={model_name}")

        records.append(
            {
                "model_name": model_name,
                "timestamp_utc": actual_timestamp,
                "actual_pm2_5": actual_value,
                "predicted_pm2_5": predicted_value,
                "forecast_error": actual_value - predicted_value,
            }
        )

    return pd.DataFrame(records)


def build_metrics_dataframe(
    city_name: str,
    initial_train_size: int,
    validation_horizon: int,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Why this exists:
    - Summary metrics must be derived in one place so reporting remains consistent
      and reproducible across models.
    """
    if predictions_df.empty:
        raise ValidationError("Predictions DataFrame is empty; cannot compute metrics.")

    metrics_rows: List[Dict[str, object]] = []

    for model_name, group_df in predictions_df.groupby("model_name", sort=True):
        actual = group_df["actual_pm2_5"].tolist()
        predicted = group_df["predicted_pm2_5"].tolist()

        metrics_rows.append(
            {
                "model_name": model_name,
                "city_name": city_name,
                "initial_train_size": initial_train_size,
                "validation_horizon": validation_horizon,
                "rmse": compute_rmse(actual, predicted),
                "mae": compute_mae(actual, predicted),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return metrics_df


def save_dataframe(df: pd.DataFrame, output_path: str) -> Path:
    """
    Why this exists:
    - Validation results should persist as artifacts so they can be consumed later
      by reports, dashboards, or Git commits.
    """
    path = Path(output_path)
    df.to_csv(path, index=False)
    return path


def run_validation_pipeline(config: ValidationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Why this exists:
    - This function orchestrates the end-to-end validation workflow while keeping
      each major responsibility modular and testable.
    """
    series = load_city_series(
        city_name=config.city_name,
        db_path=config.database_path,
    )
    series = validate_series(series=series, config=config)

    LOGGER.info(
        "Loaded validation dataset for city=%s with total_obs=%d initial_train_size=%d validation_horizon=%d",
        config.city_name,
        len(series),
        config.initial_train_size,
        config.validation_horizon,
    )

    arima_predictions = run_rolling_validation_for_model(
        model_name="ARIMA",
        series=series,
        config=config,
    )

    sarima_predictions = run_rolling_validation_for_model(
        model_name="SARIMA",
        series=series,
        config=config,
    )

    predictions_df = pd.concat(
        [arima_predictions, sarima_predictions],
        axis=0,
        ignore_index=True,
    )

    metrics_df = build_metrics_dataframe(
        city_name=config.city_name,
        initial_train_size=config.initial_train_size,
        validation_horizon=config.validation_horizon,
        predictions_df=predictions_df,
    )

    predictions_path = save_dataframe(predictions_df, config.predictions_output_csv)
    metrics_path = save_dataframe(metrics_df, config.metrics_output_csv)

    LOGGER.info("Saved rolling validation predictions to %s", predictions_path.resolve())
    LOGGER.info("Saved rolling validation metrics to %s", metrics_path.resolve())
    LOGGER.info("Metrics summary:\n%s", metrics_df.to_string(index=False))

    return predictions_df, metrics_df


def main() -> None:
    """
    Why this exists:
    - This entrypoint keeps the validation workflow directly runnable while preserving
      clean imports for later reuse.
    """
    config = ValidationConfig()

    try:
        predictions_df, metrics_df = run_validation_pipeline(config=config)

        print("\nRolling Validation Summary")
        print("--------------------------")
        print(f"City               : {config.city_name}")
        print(f"Initial Train Size : {config.initial_train_size}")
        print(f"Validation Horizon : {config.validation_horizon}")
        print("\nModel Metrics")
        print(metrics_df.to_string(index=False))

        best_model = metrics_df.iloc[0]["model_name"]
        print(f"\nBest Model by RMSE : {best_model}")

        print("\nPrediction Preview")
        print(predictions_df.head(10).to_string(index=False))

    except (DatasetError, ValidationError, ValueError) as exc:
        LOGGER.exception("Forecast validation pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()