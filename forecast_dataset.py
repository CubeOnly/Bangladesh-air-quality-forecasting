from __future__ import annotations

"""
Dataset loader for forecasting models.

Why this file exists:
- Forecasting models should not directly access raw database queries.
- This module provides a clean, validated time series extracted from SQLite.
- The output is a chronologically ordered PM2.5 time series for a selected city.
"""

import sqlite3
from typing import Optional

import pandas as pd


class DatasetError(Exception):
    """
    Raised when dataset preparation fails due to missing data,
    invalid timestamps, or empty query results.
    """
    pass


def load_city_series(
    city_name: str,
    db_path: str = "air_quality.db",
) -> pd.Series:
    """
    Load PM2.5 time series for a specific city.

    Why this function exists:
    - Forecasting algorithms require a continuous chronological signal.
    - This function extracts a city-specific time series from the database
      and ensures correct ordering and type safety.

    Parameters
    ----------
    city_name : str
        City name stored in the database.
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    pandas.Series
        Time-indexed PM2.5 series ready for modeling.
    """

    query = """
    SELECT
        timestamp_utc,
        pm2_5
    FROM air_quality_data
    WHERE city_name = ?
    ORDER BY timestamp_utc ASC
    """

    try:
        with sqlite3.connect(db_path) as connection:
            df = pd.read_sql_query(query, connection, params=(city_name,))

    except sqlite3.Error as exc:
        raise DatasetError(f"Database query failed: {exc}") from exc

    if df.empty:
        raise DatasetError(f"No data found for city={city_name}")

    # Convert timestamp column to pandas datetime
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Remove rows with missing PM2.5 values
    df = df.dropna(subset=["pm2_5"])

    if df.empty:
        raise DatasetError(f"No valid PM2.5 data available for city={city_name}")

    # Ensure chronological ordering
    df = df.sort_values("timestamp_utc")

    # Set time index
    df = df.set_index("timestamp_utc")

    # Return as Series
    series = df["pm2_5"]

    return series


def preview_city_series(city_name: str) -> None:
    """
    Debug utility to inspect the generated time series.

    Why this exists:
    - Allows quick inspection of the dataset before modeling.
    - Prevents model errors caused by malformed input.
    """

    series = load_city_series(city_name)

    print("\nSeries preview:")
    print(series.head())

    print("\nSeries info:")
    print(series.describe())

    print("\nTotal observations:", len(series))


if __name__ == "__main__":
    preview_city_series("Dhaka")