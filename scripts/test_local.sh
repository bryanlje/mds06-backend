#!/bin/bash

# Test script for local development

SERVICE_URL="http://localhost:8080"

echo "Testing Badminton Analyzer..."

# Test health endpoint
echo -e "\n1. Testing health endpoint..."
curl -s ${SERVICE_URL}/health | jq

# Test video processing (if test video exists)
if [ -f "tests/test_video.mp4" ]; then
    echo -e "\n2. Testing video processing..."
    curl -X POST ${SERVICE_URL}/process_video \
        -F "video=@tests/test_video.mp4" \
        -F "batch_size=8" \
        | jq
else
    echo -e "\n2. Skipping video test (no test video found)"
fi

echo -e "\nTest complete!"