import json
import logging
import os
from typing import Any

from confluent_kafka import KafkaError, Message, Producer

from vehicle_counter.domain.events import VehicleEvent

# Standard infrastructure logger
logger = logging.getLogger("TelemetrySink")


def delivery_report(err: KafkaError | None, msg: Message) -> None:
    """
    Callback invoked by librdkafka after a message delivery attempt.
    Ensures traceability and visibility over broker ingestion.
    """
    if err is not None:
        logger.error("Failed to deliver message to Kafka: %s", err)


class TelemetrySink:
    """
    Kafka-based telemetry sink.
    Fires domain events to a distributed message broker in real-time,
    completely decoupled from local storage mechanisms.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str = "vehicle-telemetry",
    ) -> None:
        self.topic: str = topic
        kafka_host: str = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )

        # Production configuration for resilience and delivery guarantees
        producer_conf: dict[str, Any] = {
            "bootstrap.servers": kafka_host,
            "acks": "all",
            "retries": 5,
            "retry.backoff.ms": 300,
            "queue.buffering.max.messages": 100000,
        }

        try:
            self.producer: Producer = Producer(producer_conf)

        except Exception as exc:
            logger.critical(
                "Failed to initialize Kafka Producer: %s", exc, exc_info=True
            )
            msg = f"Kafka Producer Initialization Error: {exc}"
            raise RuntimeError(msg) from exc

    def add(self, event: VehicleEvent) -> None:
        """
        Serializes the domain event and dispatches it asynchronously.
        """
        # Safe extraction of the numeric value from the domain Enum
        class_id_val: int = (
            event.class_id.value
            if hasattr(event.class_id, "value")
            else int(event.class_id)
        )

        payload: dict[str, Any] = {
            "timestamp": event.timestamp.isoformat(),
            "track_id": event.track_id,
            "class_id": class_id_val,
            "confidence": round(event.confidence, 3),
            "direction": event.direction.value,
        }

        try:
            event_bytes: bytes = json.dumps(payload).encode("utf-8")
            self.producer.produce(
                self.topic,
                value=event_bytes,
                on_delivery=delivery_report,
            )
            # Process pending callback events in the queue
            self.producer.poll(0)
        except Exception as exc:
            logger.error(
                "Failed to enqueue message in Kafka Producer: %s", exc, exc_info=True
            )
            raise

    def close(self) -> None:
        """
        Blocks the application shutdown until all queued messages are delivered.
        """
        remaining: int = self.producer.flush(timeout=10.0)

        if remaining > 0:
            logger.warning(
                "%d messages were not delivered within the timeout.", remaining
            )
        else:
            logger.info(
                "All messages in the queue have been acknowledged by the broker."
            )
