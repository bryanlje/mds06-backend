# deploy.ps1 - Complete Cloud Run Deployment with GPU
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [string]$Region = "us-central1",
    [string]$ServiceName = "badminton-analyzer",
    [string]$ImageName = "badminton-analyzer"
)

$ErrorActionPreference = "Stop"

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "  DEPLOYING BADMINTON ANALYZER TO GCP CLOUD RUN WITH GPU" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

Write-Host "`n📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Project ID    : $ProjectId" -ForegroundColor White
Write-Host "   Region        : $Region" -ForegroundColor White
Write-Host "   Service Name  : $ServiceName" -ForegroundColor White
Write-Host "   GPU Type      : nvidia-l4" -ForegroundColor White
Write-Host "   Memory        : 32Gi" -ForegroundColor White
Write-Host "   CPU Cores     : 8" -ForegroundColor White
Write-Host ""

# Confirmation
$confirm = Read-Host "Continue with deployment? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

try {
    # ===========================================
    # STEP 1: Configure GCP Project
    # ===========================================
    Write-Host "`n[1/6] Configuring GCP project..." -ForegroundColor Yellow
    gcloud config set project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Failed to set project" }
    Write-Host "      ✓ Project set to: $ProjectId" -ForegroundColor Green

    # ===========================================
    # STEP 2: Enable Required APIs
    # ===========================================
    Write-Host "`n[2/6] Enabling required APIs..." -ForegroundColor Yellow
    Write-Host "      This may take 1-2 minutes..." -ForegroundColor Gray
    
    $apis = @(
        "cloudbuild.googleapis.com",
        "run.googleapis.com",
        "artifactregistry.googleapis.com",
        "compute.googleapis.com"
    )
    
    foreach ($api in $apis) {
        Write-Host "      Enabling $api..." -ForegroundColor Gray
        gcloud services enable $api --project=$ProjectId 2>$null
    }
    Write-Host "      ✓ APIs enabled" -ForegroundColor Green

    # ===========================================
    # STEP 3: Create Artifact Registry Repository
    # ===========================================
    Write-Host "`n[3/6] Setting up Artifact Registry..." -ForegroundColor Yellow
    
    $repoCheck = gcloud artifacts repositories describe cloud-run-images `
        --location=$Region `
        --project=$ProjectId 2>$null
    
    if (-not $repoCheck) {
        Write-Host "      Creating repository..." -ForegroundColor Gray
        gcloud artifacts repositories create cloud-run-images `
            --repository-format=docker `
            --location=$Region `
            --project=$ProjectId `
            --description="Docker images for Cloud Run services"
        
        if ($LASTEXITCODE -ne 0) { throw "Failed to create repository" }
        Write-Host "      ✓ Repository created" -ForegroundColor Green
    } else {
        Write-Host "      ✓ Repository already exists" -ForegroundColor Green
    }

    # ===========================================
    # STEP 4: Build Container Image
    # ===========================================
    Write-Host "`n[4/6] Building container image..." -ForegroundColor Yellow
    Write-Host "      This will take 15-25 minutes" -ForegroundColor Gray
    Write-Host "      ☕ Perfect time for a coffee break!`n" -ForegroundColor Gray
    
    $imageUri = "${Region}-docker.pkg.dev/${ProjectId}/cloud-run-images/${ImageName}:latest"
    $buildStart = Get-Date
    
    # Check required files
    $requiredFiles = @(
        "Dockerfile",
        "app.py",
        "requirements.txt",
        "TrackNetV3\predict.py",
        "models\yolo_weights.pt"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            throw "Required file missing: $file"
        }
    }
    
    Write-Host "      ✓ All required files present" -ForegroundColor Green
    Write-Host "      Starting build...`n" -ForegroundColor Gray
    
    gcloud builds submit `
        --tag=$imageUri `
        --project=$ProjectId `
        --timeout=30m `
        --machine-type=e2-highcpu-32 `
        --disk-size=200
    
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    
    $buildEnd = Get-Date
    $buildDuration = $buildEnd - $buildStart
    Write-Host "`n      ✓ Build completed in $($buildDuration.ToString('mm\:ss'))" -ForegroundColor Green

    # ===========================================
    # STEP 5: Deploy to Cloud Run with GPU
    # ===========================================
    Write-Host "`n[5/6] Deploying to Cloud Run with GPU..." -ForegroundColor Yellow
    Write-Host "      This may take 3-5 minutes..." -ForegroundColor Gray
    
    $deployStart = Get-Date
    
    gcloud run deploy $ServiceName `
        --image=$imageUri `
        --platform=managed `
        --region=$Region `
        --project=$ProjectId `
        --gpu=1 `
        --gpu-type=nvidia-l4 `
        --memory=32Gi `
        --cpu=8 `
        --min-instances=0 `
        --max-instances=10 `
        --concurrency=1 `
        --timeout=3600 `
        --no-cpu-throttling `
        --set-env-vars="USE_YOLO=true,USE_CONTACT_DETECTION=true,USE_SLOWFAST=true,USE_INPAINTNET=false" `
        --allow-unauthenticated `
        --port=8080 `
        --no-traffic `
        --tag=latest
    
    if ($LASTEXITCODE -ne 0) { throw "Deployment failed" }
    
    # Route all traffic to latest
    gcloud run services update-traffic $ServiceName `
        --region=$Region `
        --project=$ProjectId `
        --to-latest
    
    $deployEnd = Get-Date
    $deployDuration = $deployEnd - $deployStart
    Write-Host "      ✓ Deployment completed in $($deployDuration.ToString('mm\:ss'))" -ForegroundColor Green

    # ===========================================
    # STEP 6: Get Service Information
    # ===========================================
    Write-Host "`n[6/6] Retrieving service information..." -ForegroundColor Yellow
    
    $serviceUrl = gcloud run services describe $ServiceName `
        --region=$Region `
        --project=$ProjectId `
        --format='value(status.url)'
    
    # Save URL to file
    $serviceUrl | Out-File "service_url.txt" -Encoding UTF8
    
    Write-Host "`n" + "="*80 -ForegroundColor Green
    Write-Host "  ✅ DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Green
    
    Write-Host "`n🌐 Service URL:" -ForegroundColor Cyan
    Write-Host "   $serviceUrl" -ForegroundColor White
    
    Write-Host "`n📝 Service Details:" -ForegroundColor Cyan
    Write-Host "   Project    : $ProjectId" -ForegroundColor Gray
    Write-Host "   Region     : $Region" -ForegroundColor Gray
    Write-Host "   Service    : $ServiceName" -ForegroundColor Gray
    Write-Host "   GPU        : NVIDIA L4 x1" -ForegroundColor Gray
    Write-Host "   Memory     : 32Gi" -ForegroundColor Gray
    Write-Host "   CPU        : 8 cores" -ForegroundColor Gray
    
    Write-Host "`n🧪 Test Commands:" -ForegroundColor Cyan
    Write-Host "   # Health check" -ForegroundColor Gray
    Write-Host "   Invoke-RestMethod `"${serviceUrl}/health`"" -ForegroundColor White
    
    Write-Host "`n   # Process video" -ForegroundColor Gray
    Write-Host "   .\test_cloudrun.ps1" -ForegroundColor White
    
    Write-Host "`n📊 Monitor Service:" -ForegroundColor Cyan
    $monitorUrl = "https://console.cloud.google.com/run/detail/${Region}/${ServiceName}?project=${ProjectId}"
    Write-Host "   $monitorUrl" -ForegroundColor White
    
    Write-Host "`n💰 Cost Information:" -ForegroundColor Cyan
    Write-Host "   GPU (L4)       : ~`$0.80/hour when running" -ForegroundColor Yellow
    Write-Host "   Memory (32GB)  : ~`$0.12/hour when running" -ForegroundColor Yellow
    Write-Host "   Total          : ~`$0.92/hour when running" -ForegroundColor Yellow
    Write-Host "   Idle Cost      : `$0.00/hour (min-instances=0)" -ForegroundColor Green
    Write-Host "   Note           : You only pay when processing videos!" -ForegroundColor Gray
    
    Write-Host "`n📁 Files Created:" -ForegroundColor Cyan
    Write-Host "   service_url.txt    - Service URL for testing" -ForegroundColor Gray
    Write-Host "   test_cloudrun.ps1  - Test script (will be created)" -ForegroundColor Gray
    
    # Create test script
    CreateTestScript -ServiceUrl $serviceUrl
    
    Write-Host "`n" + "="*80 -ForegroundColor Cyan
    Write-Host ""
    
} catch {
    Write-Host "`n" + "="*80 -ForegroundColor Red
    Write-Host "  ❌ DEPLOYMENT FAILED" -ForegroundColor Red
    Write-Host "="*80 -ForegroundColor Red
    Write-Host "`nError: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nFor detailed logs, run:" -ForegroundColor Yellow
    Write-Host "   gcloud builds list --limit=1 --project=$ProjectId" -ForegroundColor White
    Write-Host ""
    exit 1
}

function CreateTestScript {
    param([string]$ServiceUrl)
    
    $testScript = @"
# test_cloudrun.ps1 - Test Cloud Run GPU Service
param([string]`$VideoPath = "test_videos\shi_vit_rally_3.mp4")

Write-Host "``n🌐 TESTING CLOUD RUN GPU SERVICE" -ForegroundColor Cyan
Write-Host "=" * 80

`$serviceUrl = "$ServiceUrl"
Write-Host "``n🔗 Service: `$serviceUrl" -ForegroundColor Gray

# Check video
if (-not (Test-Path `$VideoPath)) {
    Write-Host "❌ Video not found: `$VideoPath" -ForegroundColor Red
    exit 1
}

`$videoSize = (Get-Item `$VideoPath).Length / 1MB
Write-Host "📹 Video: `$VideoPath (`$(`$videoSize.ToString('F2')) MB)" -ForegroundColor Green

# Health check
Write-Host "``n🏥 Testing health endpoint..." -ForegroundColor Yellow
try {
    `$health = Invoke-RestMethod "`$serviceUrl/health" -TimeoutSec 60
    Write-Host "   ✓ Service healthy" -ForegroundColor Green
    Write-Host "   ✓ Device: `$(`$health.device)" -ForegroundColor Green
    Write-Host "   ✓ GPU Available: `$(if(`$health.device -eq 'cuda'){'Yes'}else{'No'})" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Health check failed: `$_" -ForegroundColor Red
    exit 1
}

# Process video
Write-Host "``n🎬 Processing video with GPU..." -ForegroundColor Yellow
Write-Host "   ⚡ GPU is 5-10x faster than CPU!" -ForegroundColor Gray
Write-Host "   ⏳ Expected time: 30-90 seconds...``n" -ForegroundColor Gray

`$startTime = Get-Date

try {
    `$form = @{
        video = Get-Item `$VideoPath
        batch_size = "32"
        eval_mode = "weight"
    }
    
    `$response = Invoke-RestMethod ``
        -Uri "`$serviceUrl/process_video" ``
        -Method Post ``
        -Form `$form ``
        -TimeoutSec 3600
    
    `$duration = (Get-Date) - `$startTime
    
    Write-Host "✅ SUCCESS!" -ForegroundColor Green
    Write-Host "⏱️  Total Time: `$(`$duration.ToString('mm\:ss'))" -ForegroundColor Cyan
    
    `$response | ConvertTo-Json -Depth 10 | Out-File "gpu_results.json"
    
    Write-Host "``n📊 RESULTS" -ForegroundColor Cyan
    Write-Host "=" * 80
    
    `$r = `$response.results
    `$t = `$response.timing
    
    Write-Host "``n💥 Detections:" -ForegroundColor Yellow
    Write-Host "   Contact Frames : `$(`$r.contact_frames_count)" -ForegroundColor White
    Write-Host "   Action Events  : `$(`$r.events_count)" -ForegroundColor White
    
    Write-Host "``n⚡ GPU Performance:" -ForegroundColor Yellow
    Write-Host "   Player Tracking    : `$(`$t.strongsort.ToString('F1'))s" -ForegroundColor Gray
    Write-Host "   Shuttle Tracking   : `$(`$t.tracknet.ToString('F1'))s" -ForegroundColor Gray
    Write-Host "   Parallel Time      : `$(`$t.parallel_time.ToString('F1'))s" -ForegroundColor Green
    Write-Host "   Contact Detection  : `$(`$t.contact_detection.ToString('F1'))s" -ForegroundColor Gray
    Write-Host "   Action Recognition : `$(`$t.action_recognition.ToString('F1'))s" -ForegroundColor Gray
    Write-Host "   Video Rendering    : `$(`$t.overlay_rendering.ToString('F1'))s" -ForegroundColor Gray
    Write-Host "   Total              : `$(`$t.total_time.ToString('F1'))s" -ForegroundColor Cyan
    Write-Host "   Speedup            : `$(`$t.time_saved.ToString('F1'))s" -ForegroundColor Green
    
    `$costPerSec = 0.92 / 3600
    `$cost = `$t.total_time * `$costPerSec
    Write-Host "``n💰 Estimated Cost: ~```$`$(`$cost.ToString('F4'))" -ForegroundColor Yellow
    
    Write-Host "``n💾 Results saved to: gpu_results.json" -ForegroundColor Cyan
    Write-Host "``n" + "=" * 80
    
} catch {
    `$duration = (Get-Date) - `$startTime
    Write-Host "❌ ERROR after `$(`$duration.ToString('mm\:ss'))" -ForegroundColor Red
    Write-Host `$_.Exception.Message -ForegroundColor Red
}
"@
    
    $testScript | Out-File "test_cloudrun.ps1" -Encoding UTF8
    Write-Host "   ✓ Created test_cloudrun.ps1" -ForegroundColor Green
}