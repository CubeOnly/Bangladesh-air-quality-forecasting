from __future__ import annotations

"""
Baseline forecasting model for the air-quality project.

Why this file exists:
- The project now has a validated persistence layer and a clean dataset loader.
- This module creates the first forecasting baseline on top of that foundation.
- The purpose is to validate the end-to-end modeling path before adding rolling
  validation, multi-city scaling, or dashboard integration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import logging
import math

import pandas as pd

from forecast_dataset import DatasetError, load_city_series

try:
    # This import is isolated so the failure mode is explicit and operationally clear
    # if the environment does not yet contain the modeling dependency.
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecast_model.py. "
        "Install it with: pip install statsmodels"
    ) from exc


# Logging is centralized because model training and evaluation will eventually be
# automated. Clear logs make model failures diagnosable once the project moves
# beyond manual execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("air_quality_forecast_model")


class ModelError(Exception):
    """
    Why this exists:
    - Modeling failures should be distinguishable from dataset or database failures.
    - A dedicated exception boundary makes the pipeline easier to debug and extend.
    """


@dataclass(frozen=True)
class ForecastConfig:
    """
    Why this exists:
    - Forecast configuration must be explicit and immutable so that model runs are
      reproducible and parameter drift is avoided during development.
    """
    city_name: str = "Dhaka"
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    test_size: int = 24
    database_path: str = "air_quality.db"
    output_csv_path: str = "forecast_results_dhaka.csv"


@dataclass(frozen=True)
class ForecastEvaluation:
    """
    Why this exists:
    - Returning evaluation metadata in a typed structure keeps the output contract
      clean and makes downstream reporting easier.
    """
    city_name: str
    train_size: int
    test_size: int
    rmse: float
    mae: float


def validate_series(series: pd.Series, test_size: int) -> None:
    """
    Why this exists:
    - Forecasting requires enough historical data to support both model fitting
      and a holdout evaluation window.
    - Failing early here prevents obscure downstream model errors.
    """
    if series.empty:
        raise ModelError("The input time series is empty.")

    if len(series) <= test_size:
        raise ModelError(
            f"Insufficient observations for holdout evaluation. "
            f"series_length={len(series)}, test_size={test_size}"
        )

    if not series.index.is_monotonic_increasing:
        raise ModelError("The input time series index must be sorted chronologically.")

    if series.isna().any():
        raise ModelError("The input time series contains null values after dataset loading.")


def split_train_test(series: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    """
    Why this exists:
    - Time-series splits must preserve chronology. Random splitting would create
      temporal leakage and invalid evaluation.
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    if train.empty or test.empty:
        raise ModelError("Chronological split produced an empty train or test set.")

    return train, test


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Why this exists:
    - Mean Absolute Error is robust and interpretable for pollution forecasting.
    - It is implemented locally to avoid unnecessary dependency expansion.
    """
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")

    if aligned_actual.empty:
        raise ModelError("No overlapping observations found when computing MAE.")

    absolute_errors = (aligned_actual - aligned_predicted).abs()
    return float(absolute_errors.mean())


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Why this exists:
    - RMSE penalizes larger forecasting misses and is a standard diagnostic for
      regression-style forecasting evaluation.
    """
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")

    if aligned_actual.empty:
        raise ModelError("No overlapping observations found when computing RMSE.")

    squared_errors = (aligned_actual - aligned_predicted) ** 2
    return float(math.sqrt(squared_errors.mean()))


def fit_arima_and_forecast(
    train: pd.Series,
    forecast_horizon: int,
    order: Tuple[int, int, int],
) -> pd.Series:
    """
    Why this exists:
    - This function encapsulates the baseline forecasting contract:
      fit once on the training series, then forecast the holdout horizon.
    - Keeping this isolated makes later replacement with SARIMA or Prophet-like
      models straightforward.
    """
    try:
        # A non-seasonal ARIMA baseline is intentionally chosen first because the
        # goal at this stage is pipeline validation, not exhaustive model tuning.
        model = ARIMA(train, order=order)
        fitted_model = model.fit()

        # The official statsmodels ARIMA API supports out-of-sample forecasting
        # using forecast/get_forecast on the fitted results object.
        forecast = fitted_model.forecast(steps=forecast_horizon)

        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast, index=train.index[-forecast_horizon:])

        return forecast

    except Exception as exc:
        raise ModelError(f"ARIMA fit/forecast failed: {exc}") from exc


def build_results_dataframe(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    """
    Why this exists:
    - A side-by-side comparison table is the most useful debugging artifact for
      validating the model output before any dashboard is built.
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

    # This explicit error column is useful for model diagnostics and later charting.
    results_df["forecast_error"] = (
        results_df["actual_pm2_5"] - results_df["predicted_pm2_5"]
    )

    return results_df


def save_results(results_df: pd.DataFrame, output_csv_path: str) -> Path:
    """
    Why this exists:
    - Persisting evaluation output creates an artifact that can be reused by the
      dashboard layer and supports traceability across model iterations.
    """
    output_path = Path(output_csv_path)
    results_df.to_csv(output_path, index=False)
    return output_path


def run_forecast_pipeline(config: ForecastConfig) -> ForecastEvaluation:
    """
    Why this exists:
    - This is the orchestration entrypoint for the modeling stage.
    - It coordinates dataset loading, splitting, forecasting, evaluation, and output generation.
    """
    series = load_city_series(
        city_name=config.city_name,
        db_path=config.database_path,
    )

    validate_series(series=series, test_size=config.test_size)

    train, test = split_train_test(series=series, test_size=config.test_size)

    LOGGER.info(
        "Prepared time series for city=%s with total_obs=%d train_obs=%d test_obs=%d",
        config.city_name,
        len(series),
        len(train),
        len(test),
    )

    predicted = fit_arima_and_forecast(
        train=train,
        forecast_horizon=len(test),
        order=config.arima_order,
    )

    # The forecast must align to the holdout timestamps, otherwise the evaluation
    # would be structurally invalid even if numeric values were produced.
    predicted.index = test.index

    rmse = compute_rmse(actual=test, predicted=predicted)
    mae = compute_mae(actual=test, predicted=predicted)

    results_df = build_results_dataframe(actual=test, predicted=predicted)
    output_path = save_results(results_df=results_df, output_csv_path=config.output_csv_path)

    LOGGER.info("Saved forecast results to %s", output_path.resolve())
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
    - This keeps the module directly executable while preserving clean importability
      for future automation or dashboard integration.
    """
    config = ForecastConfig()

    try:
        evaluation = run_forecast_pipeline(config=config)

        print("\nForecast Evaluation Summary")
        print("---------------------------")
        print(f"City       : {evaluation.city_name}")
        print(f"Train Size : {evaluation.train_size}")
        print(f"Test Size  : {evaluation.test_size}")
        print(f"RMSE       : {evaluation.rmse:.4f}")
        print(f"MAE        : {evaluation.mae:.4f}")

    except (DatasetError, ModelError, ValueError) as exc:
        LOGGER.exception("Forecast pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()