# 🚦 Traffic Telemetry Pipeline: Edge-to-Cloud Data Lake (YOLOv8 + Kafka + AWS S3)

## 🚀 The Engineering Challenge & Solution

**The Problem:** Monitoring traffic in real-time typically presents two catastrophic infrastructural bottlenecks. Sending raw video streams to the cloud drains network bandwidth and incurs massive compute costs. Conversely, processing data locally on edge devices usually leads to Out-Of-Memory (OOM) crashes and disk I/O bottlenecks.

**The Solution:** This system decouples visual inference from data persistence using an Event-Driven Hexagonal Architecture. It operates directly on edge hardware, extracting only vector movement metadata (dropping the heavy video frames). It then micro-batches these events in memory and streams them asynchronously to an AWS Data Lake in a highly compressed columnar format.

The architecture is built upon a strict separation of concerns:

| Layer                    | Technology         | Responsibility                                                                              |
| :----------------------- | :----------------- | :------------------------------------------------------------------------------------------ |
| **Vision (Producer)**    | YOLOv8 + OpenVINO  | Spatial tracking and inference, with asynchronous data offloading to the network interface. |
| **Network (Broker)**     | Confluent Kafka    | Traffic spike absorption and chronological event immobilization (Topic).                    |
| **Ingestion (Consumer)** | PyArrow + Boto3    | Network subscription, RAM micro-batching, and columnar conversion to the cloud (AWS S3).    |
| **Domain (Contracts)**   | Python DataClasses | Strict type guarantee for all internal events, preventing orphaned or corrupted data.       |

### 🎯 Core Business Value

- **Zero Cloud Compute for Vision:** Heavy GPU/CPU bounding box math is done at the edge; the cloud only receives clean analytics.
- **Cost-Optimized Storage:** Hardcoded internal batching limits (10,000 records / 120s) ensure injection into AWS S3 in highly compressed **Parquet** format. AWS Athena queries cost fractions of a cent compared to raw JSON/CSV scans, bypassing the "small files problem".
- **Fail-Safe Streaming:** The visual engine never blocks waiting for I/O operations. Network failures are caught by Kafka's underlying C buffer, guaranteeing maximum FPS and zero data loss.
- **Infrastructure as Code:** Fully containerized via Docker, allowing seamless deployment from local laptops to cloud servers.

---

## ⚙️ Target Architecture & Installation

The pipeline relies on [uv](https://github.com/astral-sh/uv) for local development and dependency resolution, and **Docker Compose** for production execution.

### 1. Environment Synchronization (Local Development)

Clone the repository and enforce strict virtual environment synchronization for local tools (like the visual calibrator):

```bash
git clone [https://github.com/tenoriopedro/vehicle-counter-YOLO.git](https://github.com/tenoriopedro/vehicle-counter-YOLO.git)
cd vehicle-counter-YOLO
uv sync
```

### 2. Credentials Vault & Infrastructure

Direct cloud ingestion requires programmatic authorization. Create a `.env` file at the project root (never committed) with the following structure:

```ini
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-telemetry-bucket
# Optional: Override the default data directory (defaults to ./data/raw)
# VEHICLE_DATA_DIR=/mnt/external_drive/data
```

## 📂 Data Contract (Convention over Configuration)

The system resolves physical files dynamically based on the context name. The source video must be isolated within a directory that matches its name. The root directory is managed by the `$VEHICLE_DATA_DIR` environment variable (falling back to `./data/raw/` for zero-config local development).

```text
$VEHICLE_DATA_DIR/
└── north_avenue/
    ├── north_avenue.mp4  <-- The video file must assume this exact name
    └── north_avenue.json <-- Generated automatically by the calibration CLI
```

## 🛠️ Execution Flow (Microservices)

The architecture mandates the execution of isolated processes.

### Step 1: Spatial Calibration (Local)

Launch the interactive UI to draw the counting line for a specific context. This generates the required `.json` spatial matrix.

```bash
uv run vehicle-calibrate -c north_avenue
```

### Step 2: Visual Debugging (Local - Optional)

Before triggering the data pipeline, validate the YOLO bounding boxes and the spatial line alignment in real-time.

```bash
uv run vehicle-debug -c north_avenue -m models/yolov8n_openvino_model
```

### Step 3: Production Pipeline (Dockerized)

Launch the background network infrastructure (Zookeeper, Kafka broker) and the S3 Ingestion Engine (Consumer).

```bash
docker compose up -d --build
```

Once the broker is healthy, trigger the Visual Orchestrator (Producer). This container runs the heavy inference and pipes data to the network, destroying itself cleanly when the video ends.

```bash
docker compose run --rm producer -c north_avenue -m models/yolov8n_openvino_model
```
