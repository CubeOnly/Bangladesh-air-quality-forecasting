from __future__ import annotations

"""
This module implements the multi-city Bangladesh air-quality ingestion pipeline
with SQLite persistence.

Why this file exists:
- The project has completed source ingestion validation and now needs persistent storage.
- SQLite is sufficient for the current portfolio stage because it introduces SQL persistence
  and idempotent writes without premature infrastructure complexity.
- This file replaces the previous ingestion-only script and becomes the new pipeline entrypoint.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import os
import sqlite3

import pandas as pd
import requests


# Logging is centralized because this script is transitioning from a prototype
# into an operational pipeline. Once automation is added, logs become the primary
# artifact for diagnosing request failures, data-quality issues, and database problems.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("bd_air_quality_pipeline")


@dataclass(frozen=True)
class CityConfig:
    """
    Why this exists:
    - Each city must have a durable identity and coordinate pair.
    - This structure keeps geographic scope separate from ingestion logic,
      which makes the pipeline easy to extend later.
    """
    city_name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class AppConfig:
    """
    Why this exists:
    - Runtime behavior should be configurable without editing business logic.
    - This supports safe local development now and smoother transition to scheduled execution later.
    """
    cities: List[CityConfig]
    database_path: str = "air_quality.db"
    timezone_name: str = "auto"
    request_timeout_seconds: int = 30
    source_name: str = "open_meteo"
    past_days: int = 2


@dataclass(frozen=True)
class AirQualityRecord:
    """
    Why this exists:
    - A typed record forms a stable contract between ingestion and persistence.
    - Pollutant-level fields are retained instead of collapsing to AQI because
      that preserves analytical flexibility and supports future feature engineering.
    """
    city_name: str
    timestamp_utc: datetime
    latitude: float
    longitude: float
    pm2_5: Optional[float]
    pm10: Optional[float]
    carbon_monoxide: Optional[float]
    nitrogen_dioxide: Optional[float]
    ozone: Optional[float]
    source_name: str
    ingestion_time_utc: datetime


class APIClientError(Exception):
    """
    Why this exists:
    - Network and API failures must be distinguishable from programming defects.
    - This allows the orchestration layer to fail clearly and predictably.
    """


class DataValidationError(Exception):
    """
    Why this exists:
    - Invalid source data should never silently enter persistence.
    - Failing fast here protects downstream modeling integrity.
    """


class RepositoryError(Exception):
    """
    Why this exists:
    - Storage-layer failures need a dedicated exception boundary so they can be
      logged and handled separately from ingestion or validation issues.
    """


class OpenMeteoAirQualityClient:
    """
    Why this exists:
    - The API adapter is isolated from normalization and storage concerns.
    - This preserves replaceability if the project later adds OpenAQ or another source.
    """

    BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, config: AppConfig) -> None:
        """
        Why this exists:
        - A shared requests session reduces repeated connection overhead across city calls.
        """
        self._config = config
        self._session = requests.Session()

    def fetch_hourly_air_quality_for_city(self, city: CityConfig) -> Dict[str, Any]:
        """
        Why this exists:
        - Each city is fetched independently so that one failed city does not force
          structural changes to the pipeline.
        """
        params = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "timezone": self._config.timezone_name,
            "past_days": self._config.past_days,
            "hourly": ",".join(
                [
                    "pm2_5",
                    "pm10",
                    "carbon_monoxide",
                    "nitrogen_dioxide",
                    "ozone",
                ]
            ),
        }

        try:
            LOGGER.info(
                "Requesting air-quality data for city=%s lat=%s lon=%s",
                city.city_name,
                city.latitude,
                city.longitude,
            )

            response = self._session.get(
                self.BASE_URL,
                params=params,
                timeout=self._config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload:
                raise APIClientError(
                    f"Unexpected API response for city={city.city_name}: missing 'hourly' section."
                )

            return payload

        except requests.Timeout as exc:
            raise APIClientError(
                f"Air-quality API request timed out for city={city.city_name}."
            ) from exc
        except requests.RequestException as exc:
            raise APIClientError(
                f"Air-quality API request failed for city={city.city_name}: {exc}"
            ) from exc
        except ValueError as exc:
            raise APIClientError(
                f"Air-quality API returned malformed JSON for city={city.city_name}."
            ) from exc


class AirQualityNormalizer:
    """
    Why this exists:
    - Source payloads are not suitable internal contracts.
    - This layer transforms API arrays into row-wise records that are easier to validate and store.
    """

    @staticmethod
    def to_records(
        payload: Dict[str, Any],
        city: CityConfig,
        source_name: str,
    ) -> List[AirQualityRecord]:
        """
        Why this exists:
        - Row-wise normalization creates one record per city-hour observation,
          which is the correct grain for both SQL storage and time-series modeling.
        """
        hourly = payload.get("hourly", {})
        timestamps = hourly.get("time", [])

        pm2_5_values = hourly.get("pm2_5", [])
        pm10_values = hourly.get("pm10", [])
        co_values = hourly.get("carbon_monoxide", [])
        no2_values = hourly.get("nitrogen_dioxide", [])
        ozone_values = hourly.get("ozone", [])

        column_lengths = {
            "time": len(timestamps),
            "pm2_5": len(pm2_5_values),
            "pm10": len(pm10_values),
            "carbon_monoxide": len(co_values),
            "nitrogen_dioxide": len(no2_values),
            "ozone": len(ozone_values),
        }

        expected_length = len(timestamps)
        inconsistent_columns = {
            key: value for key, value in column_lengths.items() if value != expected_length
        }

        if inconsistent_columns:
            raise DataValidationError(
                f"Inconsistent hourly array lengths for city={city.city_name}: {inconsistent_columns}"
            )

        ingestion_time = datetime.now(timezone.utc)
        records: List[AirQualityRecord] = []

        for idx, ts in enumerate(timestamps):
            # Timestamps are normalized to timezone-aware UTC values so that all cities
            # share one consistent temporal basis inside the database.
            timestamp_utc = pd.to_datetime(ts, utc=True).to_pydatetime()

            records.append(
                AirQualityRecord(
                    city_name=city.city_name,
                    timestamp_utc=timestamp_utc,
                    latitude=city.latitude,
                    longitude=city.longitude,
                    pm2_5=AirQualityNormalizer._coerce_float(pm2_5_values[idx]),
                    pm10=AirQualityNormalizer._coerce_float(pm10_values[idx]),
                    carbon_monoxide=AirQualityNormalizer._coerce_float(co_values[idx]),
                    nitrogen_dioxide=AirQualityNormalizer._coerce_float(no2_values[idx]),
                    ozone=AirQualityNormalizer._coerce_float(ozone_values[idx]),
                    source_name=source_name,
                    ingestion_time_utc=ingestion_time,
                )
            )

        return records

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """
        Why this exists:
        - Controlled numeric coercion prevents silent type pollution.
        - Null values are preserved because source incompleteness is a data-quality fact,
          not a parsing error by default.
        """
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid numeric value encountered: {value}") from exc


class AirQualityValidator:
    """
    Why this exists:
    - The validator enforces dataset integrity before records reach the database.
    - This protects the persistence layer from malformed records and duplicate keys.
    """

    @staticmethod
    def validate(records: Sequence[AirQualityRecord]) -> None:
        """
        Why this exists:
        - The pipeline should fail early on structural data errors rather than storing
          inconsistent rows and discovering the issue later during modeling.
        """
        if not records:
            raise DataValidationError("No air-quality records were produced after normalization.")

        seen_keys = set()

        for record in records:
            composite_key = (record.city_name, record.timestamp_utc)

            if composite_key in seen_keys:
                raise DataValidationError(
                    f"Duplicate city/timestamp detected: city={record.city_name}, "
                    f"timestamp={record.timestamp_utc.isoformat()}"
                )
            seen_keys.add(composite_key)

            # PM2.5 is the intended primary target variable for the forecasting pipeline.
            # Logging nulls now exposes source-quality issues without suppressing them.
            if record.pm2_5 is None:
                LOGGER.warning(
                    "Null pm2_5 detected for city=%s timestamp=%s",
                    record.city_name,
                    record.timestamp_utc.isoformat(),
                )


class SQLiteAirQualityRepository:
    """
    Why this exists:
    - Persistence must be isolated behind a repository boundary so database logic
      does not leak into ingestion orchestration.
    - This makes the eventual move from SQLite to PostgreSQL significantly easier.
    """

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        """
        Why this exists:
        - Schema creation belongs in the repository because storage contracts should
          be owned by the persistence layer, not by the ingestion service.
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS air_quality_data (
            city_name TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            pm2_5 REAL,
            pm10 REAL,
            carbon_monoxide REAL,
            nitrogen_dioxide REAL,
            ozone REAL,
            source_name TEXT NOT NULL,
            ingestion_time_utc TEXT NOT NULL,
            PRIMARY KEY (city_name, timestamp_utc)
        );
        """

        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_air_quality_timestamp
        ON air_quality_data(timestamp_utc);
        """

        try:
            with sqlite3.connect(self._database_path) as connection:
                connection.execute(create_table_sql)
                connection.execute(create_index_sql)
                connection.commit()

            LOGGER.info("SQLite repository initialized successfully at path=%s", self._database_path)

        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to initialize SQLite schema: {exc}") from exc

    def upsert_records(self, records: Sequence[AirQualityRecord]) -> int:
        """
        Why this exists:
        - UPSERT semantics provide idempotency. Re-running the ingestion job should
          refresh existing rows, not duplicate them.
        - Batch insertion improves throughput compared with one-row-at-a-time writes.
        """
        if not records:
            return 0

        upsert_sql = """
        INSERT INTO air_quality_data (
            city_name,
            timestamp_utc,
            latitude,
            longitude,
            pm2_5,
            pm10,
            carbon_monoxide,
            nitrogen_dioxide,
            ozone,
            source_name,
            ingestion_time_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(city_name, timestamp_utc) DO UPDATE SET
            latitude = excluded.latitude,
            longitude = excluded.longitude,
            pm2_5 = excluded.pm2_5,
            pm10 = excluded.pm10,
            carbon_monoxide = excluded.carbon_monoxide,
            nitrogen_dioxide = excluded.nitrogen_dioxide,
            ozone = excluded.ozone,
            source_name = excluded.source_name,
            ingestion_time_utc = excluded.ingestion_time_utc;
        """

        payload = [self._record_to_row(record) for record in records]

        try:
            with sqlite3.connect(self._database_path) as connection:
                connection.executemany(upsert_sql, payload)
                connection.commit()

            LOGGER.info("Upserted %d records into SQLite database.", len(records))
            return len(records)

        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to upsert records into SQLite: {exc}") from exc

    def fetch_row_counts_by_city(self) -> pd.DataFrame:
        """
        Why this exists:
        - A row-count summary is the fastest integrity check after ingestion.
        - This gives immediate visibility into whether the database reflects the expected city coverage.
        """
        query = """
        SELECT city_name, COUNT(*) AS row_count
        FROM air_quality_data
        GROUP BY city_name
        ORDER BY city_name;
        """

        try:
            with sqlite3.connect(self._database_path) as connection:
                return pd.read_sql_query(query, connection)

        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to fetch row counts by city: {exc}") from exc

    def fetch_preview(self, limit: int = 20) -> pd.DataFrame:
        """
        Why this exists:
        - Reading a preview back from the database confirms that persistence,
          not just in-memory normalization, succeeded.
        """
        query = """
        SELECT
            city_name,
            timestamp_utc,
            latitude,
            longitude,
            pm2_5,
            pm10,
            carbon_monoxide,
            nitrogen_dioxide,
            ozone,
            source_name,
            ingestion_time_utc
        FROM air_quality_data
        ORDER BY city_name, timestamp_utc
        LIMIT ?;
        """

        try:
            with sqlite3.connect(self._database_path) as connection:
                return pd.read_sql_query(query, connection, params=(limit,))

        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to fetch preview rows: {exc}") from exc

    @staticmethod
    def _record_to_row(record: AirQualityRecord) -> Tuple[Any, ...]:
        """
        Why this exists:
        - SQLite persistence should receive simple serializable values rather than
          domain objects. ISO-8601 text preserves time ordering and readability.
        """
        return (
            record.city_name,
            record.timestamp_utc.isoformat(),
            record.latitude,
            record.longitude,
            record.pm2_5,
            record.pm10,
            record.carbon_monoxide,
            record.nitrogen_dioxide,
            record.ozone,
            record.source_name,
            record.ingestion_time_utc.isoformat(),
        )


class AirQualityIngestionService:
    """
    Why this exists:
    - The orchestration layer coordinates source access, normalization, validation,
      and storage without collapsing those concerns into one class.
    """

    def __init__(
        self,
        client: OpenMeteoAirQualityClient,
        repository: SQLiteAirQualityRepository,
        config: AppConfig,
    ) -> None:
        self._client = client
        self._repository = repository
        self._config = config

    def run(self) -> List[AirQualityRecord]:
        """
        Why this exists:
        - This becomes the reusable entrypoint for local execution now and scheduled
          execution later.
        """
        self._repository.initialize()

        all_records: List[AirQualityRecord] = []

        for city in self._config.cities:
            payload = self._client.fetch_hourly_air_quality_for_city(city=city)
            city_records = AirQualityNormalizer.to_records(
                payload=payload,
                city=city,
                source_name=self._config.source_name,
            )
            AirQualityValidator.validate(city_records)

            LOGGER.info(
                "City ingestion complete for city=%s with %d records",
                city.city_name,
                len(city_records),
            )
            all_records.extend(city_records)

        AirQualityValidator.validate(all_records)
        self._repository.upsert_records(all_records)

        LOGGER.info(
            "Multi-city ingestion and persistence completed successfully with total_records=%d",
            len(all_records),
        )
        return all_records


def load_config_from_env() -> AppConfig:
    """
    Why this exists:
    - Externalized configuration keeps runtime settings out of business logic.
    - City definitions remain explicit in code because they are domain configuration,
      not secrets.
    """
    try:
        request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
        past_days = int(os.getenv("PAST_DAYS", "2"))
        timezone_name = os.getenv("TIMEZONE", "auto")
        database_path = os.getenv("DATABASE_PATH", "air_quality.db")
    except ValueError as exc:
        raise ValueError("Invalid environment variable value for numeric configuration.") from exc

    bd_cities = [
        CityConfig(city_name="Dhaka", latitude=23.8103, longitude=90.4125),
        CityConfig(city_name="Chattogram", latitude=22.3569, longitude=91.7832),
        CityConfig(city_name="Khulna", latitude=22.8456, longitude=89.5403),
        CityConfig(city_name="Rajshahi", latitude=24.3745, longitude=88.6042),
        CityConfig(city_name="Sylhet", latitude=24.8949, longitude=91.8687),
        CityConfig(city_name="Barishal", latitude=22.7010, longitude=90.3535),
        CityConfig(city_name="Rangpur", latitude=25.7439, longitude=89.2752),
        CityConfig(city_name="Mymensingh", latitude=24.7471, longitude=90.4203),
    ]

    return AppConfig(
        cities=bd_cities,
        database_path=database_path,
        timezone_name=timezone_name,
        request_timeout_seconds=request_timeout_seconds,
        source_name="open_meteo",
        past_days=past_days,
    )


def records_to_dataframe(records: Sequence[AirQualityRecord]) -> pd.DataFrame:
    """
    Why this exists:
    - Returning a DataFrame for the in-memory records remains useful for debugging
      and consistency checks before the modeling phase is added.
    """
    return pd.DataFrame(
        [
            {
                "city_name": record.city_name,
                "timestamp_utc": record.timestamp_utc,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "pm2_5": record.pm2_5,
                "pm10": record.pm10,
                "carbon_monoxide": record.carbon_monoxide,
                "nitrogen_dioxide": record.nitrogen_dioxide,
                "ozone": record.ozone,
                "source_name": record.source_name,
                "ingestion_time_utc": record.ingestion_time_utc,
            }
            for record in records
        ]
    )


def main() -> None:
    """
    Why this exists:
    - The script entrypoint preserves direct executability while keeping all major
      components importable and testable.
    """
    try:
        config = load_config_from_env()
        client = OpenMeteoAirQualityClient(config=config)
        repository = SQLiteAirQualityRepository(database_path=config.database_path)
        service = AirQualityIngestionService(
            client=client,
            repository=repository,
            config=config,
        )

        records = service.run()

        in_memory_df = records_to_dataframe(records)
        LOGGER.info("In-memory dataset preview:\n%s", in_memory_df.head(10).to_string(index=False))

        db_preview = repository.fetch_preview(limit=20)
        LOGGER.info("Database preview:\n%s", db_preview.to_string(index=False))

        city_counts = repository.fetch_row_counts_by_city()
        LOGGER.info("Database row counts by city:\n%s", city_counts.to_string(index=False))

    except (APIClientError, DataValidationError, RepositoryError, ValueError) as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()