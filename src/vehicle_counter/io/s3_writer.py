import uuid
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


class S3ParquetWriter:
    def __init__(self, bucket_name: str | None) -> None:
        self.bucket_name = bucket_name

        self.s3 = pafs.S3FileSystem()  # type: ignore

        self.schema = pa.schema(
            [
                pa.field("track_id", pa.int32()),
                pa.field("class_id", pa.int8()),
                pa.field("confidence", pa.float32()),
                pa.field("direction", pa.int8()),
                pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            ]
        )

    def write_batch(self, events: list[dict[Any, Any]]) -> None:
        run_id = uuid.uuid4().hex[:8]
        timestamp_id = datetime.now(UTC)

        year = timestamp_id.strftime("%Y")
        month = timestamp_id.strftime("%m")
        time = timestamp_id.strftime("%H%M%S")

        for row in events:
            if isinstance(row.get("timestamp"), str):
                row["timestamp"] = datetime.fromisoformat(row["timestamp"])

        new_list = pa.Table.from_pylist(events, schema=self.schema)

        path_s3 = f"{self.bucket_name}/{year}/{month}/{time}_{run_id}.parquet"

        pq.write_table(new_list, path_s3, filesystem=self.s3)

        print(f"Table saved successfully in: s3://{path_s3}")
