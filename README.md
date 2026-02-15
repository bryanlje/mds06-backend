# Badminton Video Analyzer - Cloud Run

AI-powered badminton video analysis with concurrent processing:
- 🏸 Shuttlecock tracking (TrackNet)
- 👥 Player tracking (YOLO + StrongSort)
- 💥 Contact detection
- 🎬 Action recognition (SlowFast)
- 🎨 Overlay video rendering

## 🚀 Quick Start

### Prerequisites
- Docker installed
- GCP account with billing enabled
- gcloud CLI installed

### Setup

1. **Clone and prepare:**
```bash
git clone 
cd badminton-analyzer
```

2. **Add model weights:**
Place your model files in:
- `models/` - YOLO, Contact, SlowFast, ReID models
- `TrackNetV3/ckpts/` - TrackNet and InpaintNet models

3. **Configure:**
```bash
cp .env.example .env
# Edit .env with your GCP project ID
```

4. **Deploy:**
```bash
chmod +x deploy.sh
./deploy.sh
```

## 📋 API Endpoints

### Health Check
```bash
curl https://your-service-url/health
```

### Process Video
```bash
curl -X POST https://your-service-url/process_video \
  -F "video=@your_video.mp4" \
  -F "batch_size=16" \
  -F "eval_mode=weight" \
  -F "use_inpaint=true"
```

## 🏗️ Architecture

Pipeline runs Steps 1 & 2 in parallel:
1. Player Tracking (StrongSort) - ~60s
2. Shuttlecock Tracking (TrackNet) - ~80s

Then sequential:
3. Contact Detection - ~10s
4. Action Recognition - ~30s
5. Overlay Rendering - ~20s

**Total: ~140s (vs 200s sequential)**

## 📊 Performance

- Parallel processing saves ~40-60% time
- Handles 1080p video @ 30fps
- ~8GB memory per instance
- Auto-scales 0-10 instances

## 🔧 Local Development
```bash
# Build locally
docker build -t badminton-analyzer .

# Run locally
docker run -p 8080:8080 \
  -e PORT=8080 \
  badminton-analyzer

# Test
curl -X POST http://localhost:8080/process_video \
  -F "video=@test.mp4"
```

## 📝 License

MIT License