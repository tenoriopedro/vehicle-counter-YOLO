import json
import os
import time

from confluent_kafka import Consumer
from dotenv import load_dotenv

from vehicle_counter.io.s3_writer import S3ParquetWriter


class TelemetryConsumer:
    def __init__(
        self, bootstrap_server: str, topic: str, group_id: str, writer: S3ParquetWriter
    ) -> None:

        self.bootstrap_server = bootstrap_server
        self.topic = topic
        self.group_id = group_id

        self.buffer = []
        self.batch_size = 50000
        self.flush_timeout = 300

        self.writer = writer

        self.timer = time.time()

        self.conf = {
            "bootstrap.servers": self.bootstrap_server,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
        }

        self.consumer = Consumer(self.conf)

    def start_consuming(self) -> None:
        self.consumer.subscribe([self.topic])

        try:
            while True:
                message = self.consumer.poll(timeout=1.0)

                current_time = time.time() - self.timer
                if self._is_batch_ready(current_time):
                    self.writer.write_batch(self.buffer)
                    self.buffer.clear()
                    self.timer = time.time()

                if message is None:
                    continue

                if message.error():
                    print(f"Erro while consuming message: {message.error()}")
                    continue

                raw_value = message.value()
                if raw_value is None:
                    continue

                raw_string = raw_value.decode("utf-8")
                data = json.loads(raw_string)

                self.buffer.append(data)

        except KeyboardInterrupt:
            pass

        finally:
            self.consumer.close()

    def _is_batch_ready(self, current_time: float) -> bool:

        if len(self.buffer) >= self.batch_size or (
            current_time >= self.flush_timeout and len(self.buffer) > 0
        ):
            print(f"Processing batch with {len(self.buffer)} records..")
            return True
        return False


if __name__ == "__main__":
    load_dotenv()
    bucket_name = os.getenv("S3_BUCKET_NAME")

    writer = S3ParquetWriter(bucket_name)

    print("Starting telemetry consumer..")

    consumer = TelemetryConsumer(
        bootstrap_server="localhost:9092",
        topic="vehicle-telemetry",
        group_id="s3-ingestion-group",
        writer=writer,
    )
    consumer.start_consuming()
