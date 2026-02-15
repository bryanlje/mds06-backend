# deploy_now.ps1
$ErrorActionPreference = "Stop"

Write-Host "`n🚀 DEPLOYING TO CLOUD RUN" -ForegroundColor Cyan
Write-Host "="*70

$ProjectId = "fyp-mds06"
$Region = "asia-southeast1"
$ServiceName = "badminton-analyzer"
$ImageName = "asia-southeast1-docker.pkg.dev/fyp-mds06/cloud-run-images/badminton-analyzer:latest"

# Check if image exists
Write-Host "`n1️⃣ Verifying image..." -ForegroundColor Yellow
$imageCheck = gcloud artifacts docker images describe $ImageName 2>$null

if ($imageCheck) {
    Write-Host "   ✓ Image found: $ImageName" -ForegroundColor Green
} else {
    Write-Host "   ❌ Image not found!" -ForegroundColor Red
    Write-Host "   Build the image first with: gcloud builds submit" -ForegroundColor Yellow
    exit 1
}

# Deploy to Cloud Run
Write-Host "`n2️⃣ Deploying to Cloud Run..." -ForegroundColor Yellow
Write-Host "   Region: $Region" -ForegroundColor Gray
Write-Host "   GPU: NVIDIA L4 x1" -ForegroundColor Gray
Write-Host "   Memory: 32Gi, CPU: 8 cores`n" -ForegroundColor Gray

$deployStart = Get-Date

gcloud run deploy $ServiceName `
    --image=$ImageName `
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
    --port=8080

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Deployment failed!" -ForegroundColor Red
    exit 1
}

$deployDuration = (Get-Date) - $deployStart
Write-Host "`n✓ Deployment complete in $($deployDuration.ToString('mm\:ss'))" -ForegroundColor Green

# Get service URL
Write-Host "`n3️⃣ Getting service information..." -ForegroundColor Yellow

$serviceUrl = gcloud run services describe $ServiceName `
    --region=$Region `
    --project=$ProjectId `
    --format='value(status.url)'

$serviceUrl | Out-File "service_url.txt" -Encoding UTF8

Write-Host "`n" + "="*70 -ForegroundColor Green
Write-Host "  ✅ DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green

Write-Host "`n🌐 Service URL:" -ForegroundColor Cyan
Write-Host "   $serviceUrl" -ForegroundColor White

Write-Host "`n⏳ Waiting for service to warm up (45 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 45

# Test health
Write-Host "`n4️⃣ Testing service health..." -ForegroundColor Yellow

try {
    $health = Invoke-RestMethod "${serviceUrl}/health" -TimeoutSec 60
    
    Write-Host "`n📊 Health Check Results:" -ForegroundColor Cyan
    Write-Host "   Status    : $($health.status)" -ForegroundColor $(if($health.status -eq 'healthy'){'Green'}else{'Red'})
    Write-Host "   Device    : $($health.device)" -ForegroundColor $(if($health.device -eq 'cuda'){'Green'}else{'Yellow'})
    Write-Host "   Threading : $($health.threading)" -ForegroundColor Gray
    
    Write-Host "`n   Models:" -ForegroundColor Gray
    Write-Host "     • YOLO       : $($health.models_loaded.yolo)" -ForegroundColor White
    Write-Host "     • StrongSort : $($health.models_loaded.strongsort)" -ForegroundColor White
    Write-Host "     • Contact    : $($health.models_loaded.contact)" -ForegroundColor White
    Write-Host "     • SlowFast   : $($health.models_loaded.slowfast)" -ForegroundColor White
    
    if ($health.device -eq 'cuda') {
        Write-Host "`n   🎉 GPU IS WORKING! ⚡" -ForegroundColor Green
        Write-Host "   Videos will process 5-10x faster!" -ForegroundColor Green
    } else {
        Write-Host "`n   ⚠️  GPU NOT DETECTED - Running on CPU" -ForegroundColor Yellow
        Write-Host "   Check build logs to ensure PyTorch CUDA was installed" -ForegroundColor Gray
    }
    
    Write-Host "`n▶️  Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Test with video: .\test_service.ps1" -ForegroundColor White
    Write-Host "   2. View logs: gcloud run services logs read $ServiceName --region=$Region" -ForegroundColor White
    Write-Host "   3. Monitor: https://console.cloud.google.com/run/detail/$Region/$ServiceName?project=$ProjectId" -ForegroundColor White
    
} catch {
    Write-Host "`n⚠️  Health check failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "`n   Service may still be starting (cold start)..." -ForegroundColor Gray
    Write-Host "   Wait 30 seconds and try:" -ForegroundColor Gray
    Write-Host "   Invoke-RestMethod `"${serviceUrl}/health`"" -ForegroundColor White
}

Write-Host "`n" + "="*70
Write-Host ""