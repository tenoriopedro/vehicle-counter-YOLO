import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

# Standard I/O infrastructure logger
logger = logging.getLogger("S3ParquetWriter")


class S3ParquetWriter:
    """
    Engineered Parquet writer using PyArrow's C++ S3FileSystem interface.
    Guarantees strict type schema mapping and input payload immutability.
    """

    def __init__(
        self,
        bucket_name: str,
        region: str | None = None,
        endpoint_override: str | None = None,
    ) -> None:
        if not bucket_name or not bucket_name.strip():
            msg = "Configuration Error: 'bucket_name' must be a non-empty string."
            raise ValueError(
                msg
            )

        self.bucket_name: str = bucket_name.strip()
        s3_kwargs: dict[str, Any] = {}

        env_region: str | None = region or os.getenv("AWS_DEFAULT_REGION")
        if env_region:
            s3_kwargs["region"] = env_region

        env_endpoint: str | None = endpoint_override or os.getenv("AWS_ENDPOINT_URL")
        if env_endpoint:
            s3_kwargs["endpoint_override"] = env_endpoint

        try:
            # Bind to abstract FileSystem class to satisfy Pylance
            self.s3: pafs.FileSystem = pafs.S3FileSystem(  # type: ignore[reportPrivateImportUsage]
                **s3_kwargs
            )
        except Exception:
            logger.critical("Failed to initialize PyArrow S3FileSystem.", exc_info=True)
            raise

        self.schema: pa.Schema = pa.schema(
            [
                pa.field("track_id", pa.int32()),
                pa.field("class_id", pa.int8()),
                pa.field("confidence", pa.float32()),
                pa.field("direction", pa.int8()),
                pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            ]
        )

    def write_batch(self, events: list[dict[str, Any]]) -> None:
        """
        Converts a list of dictionary events into a PyArrow Table and writes a Parquet
        file to S3 using a temporal partitioning directory layout (YYYY/MM).

        Does NOT mutate input dictionaries.
        """
        if not events:
            logger.warning(
                "write_batch invoked with empty event payload. Operation skipped."
            )
            return

        now: datetime = datetime.now(UTC)
        run_id: str = uuid.uuid4().hex[:8]
        year: str = now.strftime("%Y")
        month: str = now.strftime("%m")
        time_str: str = now.strftime("%H%M%S")

        processed_events: list[dict[str, Any]] = []
        for raw_event in events:
            clean_event: dict[str, Any] = raw_event.copy()
            ts_value: Any = clean_event.get("timestamp")

            if isinstance(ts_value, str):
                try:
                    clean_event["timestamp"] = datetime.fromisoformat(ts_value)
                except ValueError:
                    logger.error("Malformed ISO timestamp received.", exc_info=True)
                    raise

            processed_events.append(clean_event)

        try:
            table: pa.Table = pa.Table.from_pylist(processed_events, schema=self.schema)
            file_key: str = (
                f"{self.bucket_name}/{year}/{month}/{time_str}_{run_id}.parquet"
            )

            pq.write_table(table, file_key, filesystem=self.s3)

        except Exception:
            logger.error("I/O Failure writing Parquet batch to S3.", exc_info=True)
            raise
