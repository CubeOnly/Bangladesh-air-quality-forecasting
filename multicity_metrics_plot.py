from __future__ import annotations

"""
Visualization of multi-city forecast validation metrics.

Why this file exists:
- Model validation already produced a metrics artifact.
- This script reads that artifact and produces summary visualizations.
- Plotting is intentionally separated from model execution to maintain
  experiment reproducibility.
"""

from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("multicity_metrics_plot")


class PlotError(Exception):
    """Raised when plotting fails due to missing files or invalid structure."""


def load_metrics(path: str) -> pd.DataFrame:
    """
    Load validation metrics artifact.

    Why:
    - Ensures the CSV exists and contains required columns before plotting.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise PlotError(f"Metrics file not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = {
        "city_name",
        "model_name",
        "rmse",
        "mae",
    }

    if not required_columns.issubset(df.columns):
        raise PlotError("Metrics file missing required columns")

    return df


def plot_metric(df: pd.DataFrame, metric: str, output_file: str) -> None:
    """
    Create bar chart comparing ARIMA vs SARIMA across cities.

    Why:
    - Pivoting the table makes model comparison clearer.
    - Bar charts highlight differences across discrete locations.
    """

    pivot = df.pivot(index="city_name", columns="model_name", values=metric)

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(f"{metric.upper()} Comparison by City")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("City")

    plt.xticks(rotation=45)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(output_file)

    LOGGER.info("Saved %s plot to %s", metric, output_file)

    plt.close()


def main() -> None:
    """
    Entry point for metrics visualization.
    """

    try:

        df = load_metrics("multicity_validation_metrics.csv")

        LOGGER.info("Loaded metrics dataset with %d rows", len(df))

        plot_metric(df, "rmse", "multicity_rmse_comparison.png")
        plot_metric(df, "mae", "multicity_mae_comparison.png")

    except PlotError as exc:

        LOGGER.exception("Plotting failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()