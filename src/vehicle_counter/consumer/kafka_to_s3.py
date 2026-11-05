import json
import logging
import os
import signal
import sys
import time
from types import FrameType
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message
from dotenv import load_dotenv

from vehicle_counter.io.s3_writer import S3ParquetWriter

# Standard logging configuration for containerized workloads
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TelemetryConsumer")


class TelemetryConsumer:
    """
    Production-grade Kafka telemetry consumer.
    Implements At-Least-Once processing guarantees by manually committing
    Kafka offsets only after successful S3 Parquet batch persistence.
    """

    def __init__(
        self,
        bootstrap_server: str,
        topic: str,
        group_id: str,
        writer: S3ParquetWriter,
        batch_size: int = 100,
        flush_timeout: float = 30.0,
    ) -> None:
        self.bootstrap_server: str = bootstrap_server
        self.topic: str = topic
        self.group_id: str = group_id
        self.writer: S3ParquetWriter = writer

        self.batch_size: int = batch_size
        self.flush_timeout: float = flush_timeout

        self.buffer: list[dict[str, Any]] = []
        self.timer: float = time.time()
        self.running: bool = True

        self.conf: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_server,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        self.consumer: Consumer = Consumer(self.conf)
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        """Attaches OS signal listeners for graceful container shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: FrameType | None) -> None:
        logger.info(
            "Shutdown signal received (%s). Stopping consumption loop...", signum
        )
        self.running = False

    def start_consuming(self) -> None:
        """
        Main polling loop with strict offset control and interval-based batching.
        """
        self.consumer.subscribe([self.topic])

        try:
            while self.running:
                message: Message | None = self.consumer.poll(timeout=1.0)
                elapsed_time: float = time.time() - self.timer

                if message is not None:
                    err = message.error()
                    if err is not None:
                        if err.code() != KafkaError._PARTITION_EOF:  # noqa: SLF001
                            logger.error("Kafka consumer error. %s", err)
                        continue

                    raw_value: bytes | None = message.value()
                    if raw_value is not None:
                        try:
                            payload: dict[str, Any] = json.loads(
                                raw_value.decode("utf-8")
                            )
                            self.buffer.append(payload)
                        except json.JSONDecodeError:
                            logger.error(
                                "Failed to deserialize payload.", exc_info=True
                            )

                if self._should_flush(elapsed_time):
                    self._flush_and_commit()

        finally:
            self._cleanup()

    def _should_flush(self, elapsed_time: float) -> bool:
        """Evaluates whether the memory buffer satisfies the flushing criteria."""
        if not self.buffer:
            return False
        return len(self.buffer) >= self.batch_size or elapsed_time >= self.flush_timeout

    def _flush_and_commit(self) -> None:
        """
        Persists buffered records to S3 and synchronously commits Kafka offsets.
        Guarantees that messages are never marked as consumed if S3 I/O fails.
        """
        try:
            self.writer.write_batch(self.buffer)
            self.consumer.commit(asynchronous=False)

            self.buffer.clear()
            self.timer = time.time()

        except Exception:
            logger.error(
                "Failed to flush batch to S3 or commit offsets.", exc_info=True
            )

    def _cleanup(self) -> None:
        """Flushes remaining records and safely releases consumer resources."""
        if self.buffer:
            self._flush_and_commit()

        self.consumer.close()
        logger.info("Kafka consumer closed successfully.")


if __name__ == "__main__":
    load_dotenv()

    bucket_name: str = os.getenv("S3_BUCKET_NAME", "my-data-lake-telemetry-123")
    bootstrap_server: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    batch_size: int = int(os.getenv("CONSUMER_BATCH_SIZE", "20"))
    flush_timeout: float = float(os.getenv("CONSUMER_FLUSH_TIMEOUT", "15.0"))

    s3_writer: S3ParquetWriter = S3ParquetWriter(bucket_name)

    consumer: TelemetryConsumer = TelemetryConsumer(
        bootstrap_server=bootstrap_server,
        topic="vehicle-telemetry",
        group_id="s3-ingestion-group",
        writer=s3_writer,
        batch_size=batch_size,
        flush_timeout=flush_timeout,
    )
    consumer.start_consuming()
