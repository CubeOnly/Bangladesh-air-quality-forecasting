from __future__ import annotations

"""
Multi-city rolling validation for air-quality forecasting models.

Why this file exists:
- The project has already validated the forecasting pipeline on Dhaka.
- The next correct step is to generalize evaluation across all configured Bangladesh cities.
- This module runs the same rolling one-step-ahead validation protocol for each city
  and compares ARIMA vs SARIMA under identical conditions.
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
    # These imports are isolated so environment failures are explicit and actionable.
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecast_validation_multicity.py. "
        "Install it with: pip install statsmodels"
    ) from exc


# Centralized logging is important because multi-city rolling validation performs many
# repeated fits. When one city or one model fails, logs must make the failure location explicit.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("air_quality_forecast_validation_multicity")


class ValidationError(Exception):
    """
    Why this exists:
    - Validation-stage failures should be isolated from data-loading and model-fitting errors.
    - This makes partial diagnostics much easier when evaluating several cities in one run.
    """
    pass


@dataclass(frozen=True)
class ValidationConfig:
    """
    Why this exists:
    - Validation parameters should be explicit and immutable so experiment runs remain reproducible.
    - The city list is part of domain scope, so it is kept in configuration instead of hardcoded
      inside control flow.
    """
    cities: Tuple[str, ...] = (
        "Dhaka",
        "Chattogram",
        "Khulna",
        "Rajshahi",
        "Sylhet",
        "Barishal",
        "Rangpur",
        "Mymensingh",
    )
    database_path: str = "air_quality.db"
    initial_train_size: int = 120
    validation_horizon: int = 24
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    predictions_output_csv: str = "multicity_validation_predictions.csv"
    metrics_output_csv: str = "multicity_validation_metrics.csv"
    best_models_output_csv: str = "multicity_best_models.csv"


def validate_series(series: pd.Series, config: ValidationConfig, city_name: str) -> pd.Series:
    """
    Why this exists:
    - Rolling validation requires a chronological hourly series with no hidden gaps.
    - Explicit validation here prevents model-stage failures from being misdiagnosed as solver issues.
    """
    if series.empty:
        raise ValidationError(f"The input time series is empty for city={city_name}.")

    if not series.index.is_monotonic_increasing:
        raise ValidationError(f"The input time series is not sorted for city={city_name}.")

    # Assigning hourly frequency makes the time index explicit for repeated forecasting.
    series = series.asfreq("h")

    if series.isna().any():
        missing_count = int(series.isna().sum())
        raise ValidationError(
            f"The time series for city={city_name} contains {missing_count} missing hourly observations "
            f"after frequency assignment."
        )

    minimum_required = config.initial_train_size + config.validation_horizon
    if len(series) < minimum_required:
        raise ValidationError(
            f"Insufficient observations for city={city_name}. "
            f"series_length={len(series)}, required_minimum={minimum_required}"
        )

    return series


def compute_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Why this exists:
    - MAE provides an interpretable average absolute deviation for model comparison.
    """
    if len(actual) != len(predicted):
        raise ValidationError("MAE computation requires equal-length actual and predicted sequences.")

    if len(actual) == 0:
        raise ValidationError("MAE computation received empty input.")

    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return float(sum(absolute_errors) / len(absolute_errors))


def compute_rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Why this exists:
    - RMSE penalizes larger misses more strongly and is useful for ranking models consistently.
    """
    if len(actual) != len(predicted):
        raise ValidationError("RMSE computation requires equal-length actual and predicted sequences.")

    if len(actual) == 0:
        raise ValidationError("RMSE computation received empty input.")

    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return float(math.sqrt(sum(squared_errors) / len(squared_errors)))


def forecast_next_arima(
    history: pd.Series,
    order: Tuple[int, int, int],
) -> float:
    """
    Why this exists:
    - ARIMA one-step logic is isolated so model-specific behavior does not leak into orchestration.
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
    - SARIMA one-step logic is isolated so seasonal fitting complexity remains modular.
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
    city_name: str,
    model_name: str,
    series: pd.Series,
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Why this exists:
    - This function runs the exact same expanding-window validation protocol for one
      city-model pair, which is necessary for fair model comparison.
    """
    records: List[Dict[str, object]] = []

    validation_start = config.initial_train_size
    validation_end = config.initial_train_size + config.validation_horizon

    for step_index in range(validation_start, validation_end):
        history = series.iloc[:step_index]
        actual_timestamp = series.index[step_index]
        actual_value = float(series.iloc[step_index])

        LOGGER.info(
            "Rolling validation | city=%s | model=%s | step=%d/%d | history_size=%d | forecast_timestamp=%s",
            city_name,
            model_name,
            step_index - validation_start + 1,
            config.validation_horizon,
            len(history),
            actual_timestamp.isoformat(),
        )

        # Repeated fitting can emit non-critical warnings that would drown out the useful logs.
        # Real fitting failures still raise exceptions and are not suppressed.
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
                "city_name": city_name,
                "model_name": model_name,
                "timestamp_utc": actual_timestamp,
                "actual_pm2_5": actual_value,
                "predicted_pm2_5": predicted_value,
                "forecast_error": actual_value - predicted_value,
            }
        )

    return pd.DataFrame(records)


def build_metrics_dataframe(
    predictions_df: pd.DataFrame,
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Why this exists:
    - Summary metrics should be derived in one place so city-model comparisons are
      consistent and reproducible.
    """
    if predictions_df.empty:
        raise ValidationError("Predictions DataFrame is empty; cannot compute metrics.")

    metrics_rows: List[Dict[str, object]] = []

    grouped = predictions_df.groupby(["city_name", "model_name"], sort=True)

    for (city_name, model_name), group_df in grouped:
        actual = group_df["actual_pm2_5"].tolist()
        predicted = group_df["predicted_pm2_5"].tolist()

        metrics_rows.append(
            {
                "city_name": city_name,
                "model_name": model_name,
                "initial_train_size": config.initial_train_size,
                "validation_horizon": config.validation_horizon,
                "rmse": compute_rmse(actual, predicted),
                "mae": compute_mae(actual, predicted),
            }
        )

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["city_name", "rmse"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return metrics_df


def build_best_models_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Why this exists:
    - A best-model summary is the cleanest artifact for final project reporting.
    - It reduces per-city comparison to one defensible model choice.
    """
    if metrics_df.empty:
        raise ValidationError("Metrics DataFrame is empty; cannot determine best models.")

    best_models_df = (
        metrics_df.sort_values(["city_name", "rmse"], ascending=[True, True])
        .groupby("city_name", as_index=False)
        .first()
        .sort_values("city_name")
        .reset_index(drop=True)
    )

    return best_models_df


def save_dataframe(df: pd.DataFrame, output_path: str) -> Path:
    """
    Why this exists:
    - Validation outputs should persist as artifacts for reporting, plotting, and Git tracking.
    """
    path = Path(output_path)
    df.to_csv(path, index=False)
    return path


def run_validation_pipeline(config: ValidationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Why this exists:
    - This is the multi-city orchestration entrypoint.
    - It coordinates dataset loading, model validation, metrics generation, and artifact persistence.
    """
    all_predictions: List[pd.DataFrame] = []
    city_failures: List[str] = []

    for city_name in config.cities:
        try:
            LOGGER.info("Starting validation for city=%s", city_name)

            series = load_city_series(
                city_name=city_name,
                db_path=config.database_path,
            )
            series = validate_series(series=series, config=config, city_name=city_name)

            arima_predictions = run_rolling_validation_for_model(
                city_name=city_name,
                model_name="ARIMA",
                series=series,
                config=config,
            )

            sarima_predictions = run_rolling_validation_for_model(
                city_name=city_name,
                model_name="SARIMA",
                series=series,
                config=config,
            )

            city_predictions = pd.concat(
                [arima_predictions, sarima_predictions],
                axis=0,
                ignore_index=True,
            )
            all_predictions.append(city_predictions)

            LOGGER.info(
                "Completed validation for city=%s with prediction_rows=%d",
                city_name,
                len(city_predictions),
            )

        except (DatasetError, ValidationError) as exc:
            # Failures are collected and reported explicitly so one problematic city does not
            # destroy the whole experiment run unless all cities fail.
            LOGGER.exception("Validation failed for city=%s: %s", city_name, exc)
            city_failures.append(city_name)

    if not all_predictions:
        raise ValidationError(
            "Validation failed for all cities. No prediction artifacts were produced."
        )

    predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    metrics_df = build_metrics_dataframe(predictions_df=predictions_df, config=config)
    best_models_df = build_best_models_dataframe(metrics_df=metrics_df)

    predictions_path = save_dataframe(predictions_df, config.predictions_output_csv)
    metrics_path = save_dataframe(metrics_df, config.metrics_output_csv)
    best_models_path = save_dataframe(best_models_df, config.best_models_output_csv)

    LOGGER.info("Saved multi-city validation predictions to %s", predictions_path.resolve())
    LOGGER.info("Saved multi-city validation metrics to %s", metrics_path.resolve())
    LOGGER.info("Saved best-model summary to %s", best_models_path.resolve())

    if city_failures:
        LOGGER.warning("Validation failed for these cities: %s", ", ".join(city_failures))

    LOGGER.info("Best model summary:\n%s", best_models_df.to_string(index=False))

    return predictions_df, metrics_df, best_models_df


def main() -> None:
    """
    Why this exists:
    - Keeps the workflow directly runnable while preserving importability for future reuse.
    """
    config = ValidationConfig()

    try:
        _, metrics_df, best_models_df = run_validation_pipeline(config=config)

        print("\nMulti-City Rolling Validation Summary")
        print("-------------------------------------")
        print(f"Cities Evaluated     : {len(config.cities)}")
        print(f"Initial Train Size   : {config.initial_train_size}")
        print(f"Validation Horizon   : {config.validation_horizon}")

        print("\nPer-City Model Metrics")
        print(metrics_df.to_string(index=False))

        print("\nBest Model Per City")
        print(best_models_df.to_string(index=False))

    except (DatasetError, ValidationError, ValueError) as exc:
        LOGGER.exception("Multi-city validation pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()