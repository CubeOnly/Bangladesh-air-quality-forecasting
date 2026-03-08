from __future__ import annotations

"""
Seasonal forecasting baseline for the air-quality project.

Why this file exists:
- The earlier ARIMA baseline underfit the hourly PM2.5 series because it did not
  model daily seasonality explicitly.
- This module introduces a SARIMA-style baseline using statsmodels SARIMAX with a
  24-hour seasonal cycle, which is a better structural match for hourly pollution data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import logging
import math

import pandas as pd

from forecast_dataset import DatasetError, load_city_series

try:
    # This import is isolated so environment issues fail clearly and do not create
    # ambiguous runtime errors later in the pipeline.
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecast_model_sarima.py. "
        "Install it with: pip install statsmodels"
    ) from exc


# Centralized logging is retained because model execution is becoming an operational
# workflow. This makes fit failures and data issues diagnosable once automation is added.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("air_quality_forecast_sarima")


class ModelError(Exception):
    """
    Why this exists:
    - Model-stage failures should be isolated from data-access failures.
    - A dedicated exception class keeps troubleshooting precise.
    """
    pass


@dataclass(frozen=True)
class ForecastConfig:
    """
    Why this exists:
    - Forecast parameters should be explicit, immutable, and reproducible.
    - This prevents accidental parameter drift while comparing baseline models.
    """
    city_name: str = "Dhaka"
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    test_size: int = 24
    database_path: str = "air_quality.db"
    output_csv_path: str = "forecast_results_dhaka_sarima.csv"


@dataclass(frozen=True)
class ForecastEvaluation:
    """
    Why this exists:
    - A typed evaluation object makes reporting and later dashboard integration cleaner.
    """
    city_name: str
    train_size: int
    test_size: int
    rmse: float
    mae: float


def validate_series(series: pd.Series, test_size: int, seasonal_period: int) -> None:
    """
    Why this exists:
    - Seasonal models need enough data to support differencing, holdout evaluation,
      and at least a minimal number of seasonal cycles.
    """
    if series.empty:
        raise ModelError("The input time series is empty.")

    if len(series) <= test_size:
        raise ModelError(
            f"Insufficient observations for holdout evaluation. "
            f"series_length={len(series)}, test_size={test_size}"
        )

    if len(series) < seasonal_period * 3:
        raise ModelError(
            f"Insufficient observations for a stable seasonal model. "
            f"series_length={len(series)}, required_minimum={seasonal_period * 3}"
        )

    if not series.index.is_monotonic_increasing:
        raise ModelError("The input time series index must be sorted chronologically.")

    if series.isna().any():
        raise ModelError("The input time series contains null values after dataset loading.")


def split_train_test(series: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    """
    Why this exists:
    - Time-series evaluation must preserve chronology. Random splitting would create
      leakage and invalidate performance metrics.
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    if train.empty or test.empty:
        raise ModelError("Chronological split produced an empty train or test set.")

    return train, test


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Why this exists:
    - MAE is easy to interpret and robust for pollution forecasting error reporting.
    """
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")

    if aligned_actual.empty:
        raise ModelError("No overlapping observations found when computing MAE.")

    return float((aligned_actual - aligned_predicted).abs().mean())


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Why this exists:
    - RMSE penalizes large misses more strongly and is a standard forecasting metric.
    """
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")

    if aligned_actual.empty:
        raise ModelError("No overlapping observations found when computing RMSE.")

    return float(math.sqrt(((aligned_actual - aligned_predicted) ** 2).mean()))


def fit_sarima_and_forecast(
    train: pd.Series,
    forecast_horizon: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> pd.Series:
    """
    Why this exists:
    - This encapsulates the seasonal baseline contract: fit on the historical window,
      then generate an out-of-sample forecast for the holdout horizon.
    """
    try:
        # enforce_stationarity and enforce_invertibility are relaxed here because
        # short environmental series can otherwise fail to fit despite being useful
        # for baseline comparison.
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted_model = model.fit(disp=False)

        # get_forecast is used because the official statsmodels state-space API
        # supports it for out-of-sample forecasting and it exposes predicted_mean cleanly.
        forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
        predicted = forecast_result.predicted_mean

        if not isinstance(predicted, pd.Series):
            predicted = pd.Series(predicted)

        return predicted

    except Exception as exc:
        raise ModelError(f"SARIMA fit/forecast failed: {exc}") from exc


def build_results_dataframe(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """
    Why this exists:
    - Side-by-side actual vs predicted output is the most practical artifact for
      manual validation before dashboard integration.
    """
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")

    if aligned_actual.empty:
        raise ModelError("No overlapping observations found when building results DataFrame.")

    results_df = pd.DataFrame(
        {
            "timestamp_utc": aligned_actual.index,
            "actual_pm2_5": aligned_actual.values,
            "predicted_pm2_5": aligned_predicted.values,
        }
    )

    results_df["forecast_error"] = (
        results_df["actual_pm2_5"] - results_df["predicted_pm2_5"]
    )

    return results_df


def save_results(results_df: pd.DataFrame, output_csv_path: str) -> Path:
    """
    Why this exists:
    - Persisting evaluation results creates a reusable artifact for later comparison,
      charting, and dashboard display.
    """
    output_path = Path(output_csv_path)
    results_df.to_csv(output_path, index=False)
    return output_path


def run_forecast_pipeline(config: ForecastConfig) -> ForecastEvaluation:
    """
    Why this exists:
    - This orchestrates the seasonal modeling stage end-to-end while keeping the
      major responsibilities modular and testable.
    """
    series = load_city_series(
        city_name=config.city_name,
        db_path=config.database_path,
    )

    validate_series(
        series=series,
        test_size=config.test_size,
        seasonal_period=config.seasonal_order[3],
    )

    train, test = split_train_test(series=series, test_size=config.test_size)

    LOGGER.info(
        "Prepared seasonal forecast dataset for city=%s with total_obs=%d train_obs=%d test_obs=%d",
        config.city_name,
        len(series),
        len(train),
        len(test),
    )

    predicted = fit_sarima_and_forecast(
        train=train,
        forecast_horizon=len(test),
        order=config.order,
        seasonal_order=config.seasonal_order,
    )

    # The forecast must be indexed to the holdout timestamps; otherwise evaluation
    # and exported comparison artifacts would be structurally invalid.
    predicted.index = test.index

    rmse = compute_rmse(actual=test, predicted=predicted)
    mae = compute_mae(actual=test, predicted=predicted)

    results_df = build_results_dataframe(actual=test, predicted=predicted)
    output_path = save_results(results_df=results_df, output_csv_path=config.output_csv_path)

    LOGGER.info("Saved SARIMA forecast results to %s", output_path.resolve())
    LOGGER.info("Forecast preview:\n%s", results_df.head(10).to_string(index=False))
    LOGGER.info("Evaluation metrics | RMSE=%.4f | MAE=%.4f", rmse, mae)

    return ForecastEvaluation(
        city_name=config.city_name,
        train_size=len(train),
        test_size=len(test),
        rmse=rmse,
        mae=mae,
    )


def main() -> None:
    """
    Why this exists:
    - Keeps the module directly executable while preserving clean import boundaries
      for future automation and dashboard integration.
    """
    config = ForecastConfig()

    try:
        evaluation = run_forecast_pipeline(config=config)

        print("\nSARIMA Forecast Evaluation Summary")
        print("---------------------------------")
        print(f"City            : {evaluation.city_name}")
        print(f"Train Size      : {evaluation.train_size}")
        print(f"Test Size       : {evaluation.test_size}")
        print(f"RMSE            : {evaluation.rmse:.4f}")
        print(f"MAE             : {evaluation.mae:.4f}")
        print(f"Model Order     : {config.order}")
        print(f"Seasonal Order  : {config.seasonal_order}")

    except (DatasetError, ModelError, ValueError) as exc:
        LOGGER.exception("SARIMA forecast pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()