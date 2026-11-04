import json
import os

from confluent_kafka import Producer

from vehicle_counter.domain.events import VehicleEvent


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
        self.topic = topic
        kafka_host = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )

        # Additional configuration for retries and acks
        # should be added here for production.
        self.producer = Producer(
            {
                "bootstrap.servers": kafka_host,
            }
        )

    def add(self, event: VehicleEvent) -> None:
        """
        Serializes the domain event and dispatches it asynchronously.
        """
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "track_id": event.track_id,
            "class_id": event.class_id,
            "confidence": round(event.confidence, 3),
            "direction": event.direction.value,
        }

        event_bytes = json.dumps(payload).encode("utf-8")

        self.producer.produce(self.topic, value=event_bytes)
        self.producer.poll(0)

    def close(self) -> None:
        """
        Blocks the application shutdown until all queued messages are delivered.
        """
        self.producer.flush()
