# 🏸 High-Speed Badminton Match AI Analyser (Backend)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-Run-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

This repository houses the **computer vision and machine learning backend** for the Badminton Match AI Analyser. It is a containerised Flask application designed to process high-speed sports footage, extracting player movements, shuttlecock trajectories, and classifying shots using a multi-stage deep learning pipeline.

---

## 🏗️ Architecture & Pipeline

This backend is designed for **concurrent execution** to minimise processing time. When a video is received, the pipeline splits into parallel threads before converging for sequential analysis.



### 5-Stage Pipeline:

1.  **👥 Player Tracking (Parallel):**
    * **Model:** **YOLOv11** (Detection) + **StrongSORT** (Tracking).
    * **Function:** Detects players in every frame and assigns unique IDs (ReID) to track them consistently throughout the rally, handling occlusions and crossovers.
    * **Output:** `tracks.csv` containing bounding boxes and IDs.

2.  **🏸 Shuttlecock Tracking (Parallel):**
    * **Model:** **TrackNetV3**.
    * **Function:** A specialised deep learning model designed to track small, fast-moving objects (the shuttlecock) against complex backgrounds.
    * **Output:** `ball.csv` containing X, Y coordinates and visibility confidence.

3.  **💥 Contact Point Detection (Sequential):**
    * **Model:** **Custom 1D CNN**.
    * **Function:** Analyses the shuttlecock's trajectory (velocity, acceleration, and directional changes) to identify the exact frames where a racket strikes the shuttle.
    * **Output:** A list of "hit frames" (timestamps).

4.  **🎬 Action Recognition (Sequential):**
    * **Model:** **SlowFast (ResNet-101 backbone)**.
    * **Function:** For every detected contact point, this model examines a 2-second temporal window. It uses a "Slow" pathway for spatial detail and a "Fast" pathway for motion context to classify the shot (e.g., *Smash, Drop, Clear, Lift*).
    * **Output:** `events.json` containing shot labels, timestamps, and confidence scores.

5.  **🎨 Overlay & Visualisation:**
    * **Tools:** **OpenCV** & **FFmpeg**.
    * **Function:** Merges all data sources to render a final MP4 video with bounding boxes, shuttle trails, and shot labels overlaid for user analysis.
    * ***Examples: Several examples of the final output can be found in the 'example_output' folder.***

---

## 📂 Project Structure

The codebase has been refactored for modularity and scalability:

```text
mds06-backend/
├── app/
│   ├── api/             # Flask Route definitions (/health, /process)
│   ├── core/            # Pipeline orchestrator & Singleton Model Loader
│   ├── services/        # Logic for each pipeline step (tracking, shuttle, action)
│   ├── utils/           # Helpers for Video, GCS, and Geometry
│   ├── config.py        # Centralised configuration & Env vars
│   └── schemas.py       # Data classes for strict typing
├── models/              # Directory for weights (populated at build time)
├── TrackNetV3/          # Submodule for TrackNet logic
├── Dockerfile           # Multi-stage build definition
├── main.py              # Application entry point (Gunicorn target)
└── requirements.txt     # Python dependencies

```

---

## 🤖 The Models

We utilise a suite of State-of-the-Art (SOTA) models, trained and fine-tuned using our own data.

| Task | Model | Params | Description | Training |
| --- | --- | --- | --- | --- |
| **Player Detection** | **YOLOv11s** | 9.4 million | Lightweight, real-time object detection tuned for humans in sports courts. | Fine-tuned on 685 high frame rate and standard frame rate images. |
| **Player Tracking** | **osnet_x1_0** | 2.2 million | Uses OSNet for Re-Identification (ReID) using StrongSort algorithm to keep track of players even when they cross paths. | Fine-tuned on 26 high frame rate and standard frame rate clips (with an average of 700 frames each). |
| **Shuttle Tracking** | **TrackNetV3** | NA | A segmentation-based network capable of tracking the shuttlecock even with motion blur. | Used out-of-the-box; peak performance. |
| **Action Classification** | **SlowFast r101** | 62.83 million | A two-stream CNN architecture that captures the high-speed motion of badminton strokes better than standard 3D CNNs. | Fine-tuned on a total of 4621 clips across 13 classes (high and standard frame rate). |
| **Contact Detection** | **1D CNN** | 106,626 | A custom lightweight heuristic model trained on trajectory data to identify contact frames and filter false positives. | Trained from scratch on 32 rallies, 585 shots. |

> **Note on Storage:** To keep the repository light, large model weights (`.pt`, `.pth`) are **not stored in Git**. They are downloaded automatically from GitHub Releases during the Docker build process.

---

## 🔗 Project Structure & Repositories

This project is divided into three main repositories to handle the frontend, backend logic, and model development separately:

* **Frontend:** [https://github.com/bryanlje/mds06-frontend](https://github.com/bryanlje/mds06-frontend)
* **Backend API (This Repo):** [https://github.com/bryanlje/mds06-backend](https://github.com/bryanlje/mds06-backend)
* **Model Training & Data Preparation:** [https://github.com/bryanlje/mds06-ml](https://github.com/bryanlje/mds06-ml)

---

## 🚀 Getting Started (Local Development)

### Prerequisites

* Docker Desktop installed.
* GPU (Optional but recommended). If running on CPU, inference will be significantly slower.

### 1. Clone the Repository

```bash
git clone [https://github.com/bryanlje/mds06-backend.git](https://github.com/bryanlje/mds06-backend.git)
cd mds06-backend

```

### 2. Build the Docker Image

This step includes downloading the model weights (~500MB+), so it may take a few minutes.

```bash
docker build -t badminton-analyser .

```

### 3. Run the Container

```bash
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e WORKERS=1 \
  badminton-analyser

```

### 4. Test the API

You can send a test request using `curl`:

```bash
curl -X POST http://localhost:8080/process \
  -F "video=@/path/to/your/test_match.mp4" \
  -F "batch_size=16"

```

---

## ☁️ Deployment (Google Cloud Run)

This container is optimised for **Cloud Run** with GPU support (if available) or high-memory CPU instances.

1. **Authenticate & Config:**
```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]

```


2. **Submit Build to Container Registry:**
```bash
gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/badminton-analyser

```


3. **Deploy to Cloud Run:**
```bash
gcloud run deploy badminton-backend \
  --image gcr.io/[YOUR_PROJECT_ID]/badminton-analyser \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 900  # Analysis takes time!

```



---

## 🔌 API Reference

### 1. Health Check

Ensures the container is running and models are loaded into memory.

* **Endpoint:** `GET /api/health`
* **Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": { "yolo": true, "slowfast": true, ... }
}

```



### 2. Process Video

Triggers the full analysis pipeline.

* **Endpoint:** `POST /api/process`
* **Content-Type:** `multipart/form-data`
* **Parameters:**
* `video`: The video file (binary).
* `gcs_uri` (Optional): Alternatively, provide a `gs://` link to a video file.
* `output_bucket` (Optional): A GCS bucket name to automatically upload results to.
* `job_id`: Unique identifier for the job (for tracking).
* `user_id`: Unique identifier for the user.


* **Response:**
```json
{
  "job": { "job_id": "123", "user_id": "user_1" },
  "events": [
    { "track_id": 1, "label": "Smash", "t0": 145, "t1": 155, "p": 0.98 }
  ],
  "outputs": {
    "tracks_csv": "gs://bucket/outputs/.../tracks.csv",
    "overlay_video": "gs://bucket/outputs/.../overlay.mp4"
  },
  "timing": { "total_time": 145.5, "time_saved": 55.0 }
}

```



---

## 🌐 Frontend Integration

This backend is designed to work statelessly with a React frontend:

1. **Upload:** User drags a video to the frontend.
2. **Request:** Frontend gets a **Signed URL** (if using GCS directly) or streams the file to this backend.
3. **Processing:** The backend runs the pipeline. This can take 1-3 minutes depending on video length.
4. **Display:** The backend returns the `events` JSON array immediately. The frontend uses this to generate the interactive statistics dashboard and timeline while the overlay video loads asynchronously.

---

## 📝 License

---

## 👥 Team

This project was developed by **Team MDS06**:

* **Bryan Leong Jing Ern**
* **Phua Yee Yen**
* **Ting Shu Hui**
* **Lee Jian Jun Thomas**

---

## 📧 Contact

Email - [2025mds06@gmail.com](mailto:2025mds06@gmail.com)

Project Link: [https://github.com/bryanlje/mds06-frontend](https://github.com/bryanlje/mds06-frontend)
