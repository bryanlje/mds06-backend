#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "Badminton Analyzer - Cloud Run Deploy"
echo -e "======================================${NC}\n"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-badminton-analyzer}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}⚠ Not logged in. Running gcloud auth login...${NC}"
    gcloud auth login
fi

# Set project
echo -e "${YELLOW}Setting project to: ${PROJECT_ID}${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "\n${YELLOW}Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# Check if model files exist
echo -e "\n${YELLOW}Checking model files...${NC}"
REQUIRED_FILES=(
    "models/yolo_weights.pt"
    "models/contact_model.pth"
    "models/slowfast_model.pth"
    "models/osnet_x0_25_msmt17.pt"
    "TrackNetV3/ckpts/tracknet_weights.pt"
    "TrackNetV3/ckpts/inpaintnet_weights.pt"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "\n${RED}❌ Missing model files:${NC}"
    printf '%s\n' "${MISSING_FILES[@]}"
    echo -e "\n${YELLOW}Run ./scripts/download_models.sh to download missing models${NC}"
    exit 1
fi

# Build container
echo -e "\n${YELLOW}Building container image...${NC}"
echo "This may take 5-10 minutes..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=20m \
    --machine-type=e2-highcpu-8

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"

# Deploy to Cloud Run
echo -e "\n${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --max-instances 10 \
    --min-instances 0 \
    --concurrency 1 \
    --port 8080 \
    --set-env-vars "USE_INPAINTNET=true,USE_CONTACT_DETECTION=true,USE_SLOWFAST=true,USE_YOLO=true" \
    --allow-unauthenticated \
    --no-cpu-throttling

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo -e "\n${GREEN}======================================"
echo "✅ Deployment Successful!"
echo "======================================${NC}"
echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo -e "\nTest the service:"
echo -e "  Health check: ${GREEN}curl ${SERVICE_URL}/health${NC}"
echo -e "  Process video: ${GREEN}curl -X POST -F 'video=@video.mp4' ${SERVICE_URL}/process_video${NC}"
echo ""

6. .env.example
bash# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
SERVICE_NAME=badminton-analyzer

# Model Configuration (optional, defaults set in app.py)
TRACKNET_WEIGHTS_PATH=/app/TrackNetV3/ckpts/tracknet_weights.pt
INPAINTNET_WEIGHTS_PATH=/app/TrackNetV3/ckpts/inpaintnet_weights.pt
CONTACT_WEIGHTS_PATH=/app/models/contact_model.pth
SLOWFAST_WEIGHTS_PATH=/app/models/slowfast_model.pth
YOLO_WEIGHTS_PATH=/app/models/yolo_weights.pt
REID_WEIGHTS_PATH=/app/models/osnet_x0_25_msmt17.pt

# Feature Flags
USE_INPAINTNET=true
USE_CONTACT_DETECTION=true
USE_SLOWFAST=true
USE_YOLO=true

# Application Configuration
PORT=8080
LOG_LEVEL=info