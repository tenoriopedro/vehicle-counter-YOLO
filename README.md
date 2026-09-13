# 🚦 Traffic Telemetry Pipeline: Edge-to-Cloud Data Lake (YOLOv8 + Kafka + AWS S3)

## 🚀 The Engineering Challenge & Solution

**The Problem:** Monitoring traffic in real-time typically presents two catastrophic infrastructural bottlenecks. Sending raw video streams to the cloud drains network bandwidth and incurs massive compute costs. Conversely, processing data locally on edge devices usually leads to Out-Of-Memory (OOM) crashes and disk I/O bottlenecks.

**The Solution:** This system decouples visual inference from data persistence. It operates directly on edge hardware, extracting only vector movement metadata (dropping the heavy video frames). It then micro-batches these events in memory and streams them asynchronously to an AWS Data Lake in a highly compressed columnar format.

The architecture is built upon a strict separation of concerns:

| Layer                    | Technology        | Responsibility                                                                               |
| :----------------------- | :---------------- | :------------------------------------------------------------------------------------------- |
| **Vision (Producer)**    | YOLOv8 + OpenVINO | Spatial tracking and inference, with asynchronous data offloading to the network interface.  |
| **Network (Broker)**     | Confluent Kafka   | Traffic spike absorption and chronological event immobilization (Topic).                     |
| **Ingestion (Consumer)** | PyArrow + Boto3   | Network subscription, RAM micro-batching, and columnar conversion to the cloud (AWS S3).     |
| **Domain (Contracts)**   | Pydantic          | Strict type guarantee for all JSON events, preventing orphaned or corrupted data structures. |

### 🎯 Core Business Value

- **Zero Cloud Compute for Vision:** Heavy GPU/CPU bounding box math is done at the edge; the cloud only receives clean analytics.
- **Cost-Optimized Storage:** Direct RAM-to-S3 injection in **Parquet** format means AWS Athena queries cost fractions of a cent compared to raw JSON/CSV scans.
- **Fail-Safe Streaming:** The visual engine never blocks waiting for I/O operations. Network failures are caught by Kafka's underlying C buffer, guaranteeing maximum FPS and zero data loss.

---

## ⚙️ Target Architecture & Installation

The pipeline is entirely encapsulated by [uv](https://github.com/astral-sh/uv), requiring Python 3.11.9+

### 1. Primary Profile: Linux / Intel Hardware (Recommended)

To enable native acceleration on integrated GPUs and prevent CPU bottlenecking during H.264 decoding:

```bash
sudo apt update
sudo apt install intel-opencl-icd ffmpeg
```

_(If operating on macOS, Windows, or with dedicated NVIDIA hardware, ignore this step; the `uv` installer manages the runtime environment)._

### 2. Environment Synchronization

Clone the repository and enforce strict virtual environment synchronization:

```bash
git clone [https://github.com/tenoriopedro/vehicle-counter-YOLO.git](https://github.com/tenoriopedro/vehicle-counter-YOLO.git)
cd vehicle-counter
uv sync
```

### 3. Credentials Vault & Infrastructure

Direct cloud ingestion requires programmatic authorization. Create a `.env` file at the project root (never committed) with the following structure:

```ini
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-telemetry-bucket
```

## 📂 Data Contract (Mandatory)

The source video must be isolated within a **Context** directory inside `data/raw/`. The system strictly rejects unstructured files to guarantee telemetry integrity.

```text
data/raw/
└── north_avenue/
    └── north_avenue.mp4  <-- The video file must assume this exact name
```

## 🛠️ Execution Flow (Microservices)

The architecture mandates the execution of isolated processes. Boot up the network infrastructure first:

```bash
docker-compose up -d
```

### Service A: Visual Orchestrator (The Producer)

Initiates the interactive spatial calibration window, followed by targeted tracking. Telemetry is asynchronously emitted to port `9092` under the `vehicle-telemetry` topic.

```bash
# 1. Calibrate the counting line
uv run vehicle-calibrate --context <context_name>

# 2. Initiate inference and network streaming
uv run vehicle-counter --context <context_name> -m <path/to/model/weights>
```

### Service B: Ingestion Engine (The Consumer)

In a parallel terminal session, launch the daemon that subscribes to the network interface, aggregates events into micro-batches, and orchestrates the columnar conversion to S3.

```bash
# (Temporary development command)
uv run python src/vehicle_counter/consumer/kafka_to_s3.py
```
