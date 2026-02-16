# Quantitative Betting API Test Script (PowerShell)

# Base URL (change for production)
$BASE_URL = if ($env:BASE_URL) { $env:BASE_URL } else { "http://localhost:8080" }

Write-Host "`n=== Quantitative Betting API Test Script ===`n" -ForegroundColor Blue

# Health check
Write-Host "1. Checking health..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get
$response | ConvertTo-Json
Write-Host "`n"

# Get status
Write-Host "2. Getting system status..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$BASE_URL/status" -Method Get
$response | ConvertTo-Json
Write-Host "`n"

# Trigger workflow (dry run)
Write-Host "3. Triggering workflow (dry run - auto_place_bets: false)..." -ForegroundColor Cyan
$body = @{
    min_ev_threshold = 0.08
    min_confidence = 0.65
    max_daily_stake = 5.0
    auto_place_bets = $false
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$BASE_URL/run-workflow" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json
Write-Host "`n"

# Wait a bit
Write-Host "4. Waiting 5 seconds..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Check workflow status
Write-Host "5. Checking workflow status..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$BASE_URL/workflow-status" -Method Get
$response | ConvertTo-Json
Write-Host "`n"

# Get bet history
Write-Host "6. Getting bet history (last 7 days)..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$BASE_URL/bet-history?days=7&limit=10" -Method Get
$response | ConvertTo-Json
Write-Host "`n"

Write-Host "=== Test complete ===" -ForegroundColor Green
