# Cloud Run Deployment Guide

## Overview
This guide covers deploying the Quantitative Betting API to Google Cloud Run with Cloud Scheduler for periodic bet placement.

## Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed (for local testing)

## Architecture
```
Cloud Scheduler → Cloud Run API → Matchbook Exchange
                      ↓
                  DuckDB Database
```

## Deploy to Cloud Run

### 1. Set up Google Cloud project
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
```

### 2. Build and deploy
```bash
# Deploy to Cloud Run (builds automatically)
gcloud run deploy betting-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 900 \
  --max-instances 1 \
  --set-env-vars "API_FOOTBALL_KEY=your_api_key_here,MATCHBOOK_SESSION_TOKEN=your_token_here"

# Note: Adjust memory/CPU based on your needs
# Note: For production, use --no-allow-unauthenticated and set up authentication
```

### 3. Get your service URL
```bash
gcloud run services describe betting-api \
  --region us-central1 \
  --format 'value(status.url)'
```

## Set up Cloud Scheduler

### 1. Create a scheduler job to run workflow every 4 hours
```bash
# Get your Cloud Run service URL
export SERVICE_URL=$(gcloud run services describe betting-api \
  --region us-central1 \
  --format 'value(status.url)')

# Create scheduler job
gcloud scheduler jobs create http betting-workflow-runner \
  --location us-central1 \
  --schedule "0 */4 * * *" \
  --uri "${SERVICE_URL}/run-workflow" \
  --http-method POST \
  --headers "Content-Type=application/json" \
  --message-body '{"min_ev_threshold":0.08,"min_confidence":0.65,"max_daily_stake":5.0,"auto_place_bets":true}' \
  --oidc-service-account-email your-service-account@your-project.iam.gserviceaccount.com
```

### 2. Schedule variations

**Every hour during active betting hours (10am-10pm):**
```bash
gcloud scheduler jobs create http betting-workflow-hourly \
  --location us-central1 \
  --schedule "0 10-22 * * *" \
  --uri "${SERVICE_URL}/run-workflow" \
  --http-method POST \
  --message-body '{"auto_place_bets":true}'
```

**Twice daily (conservative):**
```bash
gcloud scheduler jobs create http betting-workflow-daily \
  --location us-central1 \
  --schedule "0 9,18 * * *" \
  --uri "${SERVICE_URL}/run-workflow" \
  --http-method POST \
  --message-body '{"auto_place_bets":true}'
```

### 3. Create cache cleanup job (daily at 3am)
```bash
gcloud scheduler jobs create http cache-cleanup \
  --location us-central1 \
  --schedule "0 3 * * *" \
  --uri "${SERVICE_URL}/cleanup-cache" \
  --http-method POST
```

## API Endpoints

### Trigger Workflow
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"min_ev_threshold":0.08,"min_confidence":0.65,"auto_place_bets":true}' \
  https://your-service-url/run-workflow
```

### Check Workflow Status
```bash
curl https://your-service-url/workflow-status
```

### Get Bet History
```bash
curl https://your-service-url/bet-history?days=7&limit=100
```

### Get System Status
```bash
curl https://your-service-url/status
```

### Health Check
```bash
curl https://your-service-url/health
```

## Testing Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export API_FOOTBALL_KEY="your_api_key"
export MATCHBOOK_SESSION_TOKEN="your_token"
```

### 3. Run the API server
```bash
python api_server.py
```

### 4. Test endpoints
```bash
# Health check
curl http://localhost:8080/health

# Trigger workflow
curl -X POST http://localhost:8080/run-workflow

# Get status
curl http://localhost:8080/status
```

### 5. View API documentation
Open browser: http://localhost:8080/docs

## Persistent Storage

### Option 1: Mount Cloud Storage bucket (recommended)
```bash
# Create a bucket
gsutil mb gs://your-project-betting-data

# Deploy with Cloud Storage volume
gcloud run deploy betting-api \
  --source . \
  --region us-central1 \
  --execution-environment gen2 \
  --add-volume name=betting-data,type=cloud-storage,bucket=your-project-betting-data \
  --add-volume-mount volume=betting-data,mount-path=/app/data

# Update code to use /app/data/football_data.duckdb
```

### Option 2: Use Cloud SQL (for production)
```bash
# Create Cloud SQL instance
gcloud sql instances create betting-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Connect from Cloud Run
gcloud run services update betting-api \
  --add-cloudsql-instances your-project:us-central1:betting-db
```

## Monitoring

### View logs
```bash
gcloud run services logs read betting-api \
  --region us-central1 \
  --limit 50
```

### Stream logs
```bash
gcloud run services logs tail betting-api \
  --region us-central1
```

### View scheduler job status
```bash
gcloud scheduler jobs describe betting-workflow-runner \
  --location us-central1
```

## Security Best Practices

### 1. Use Secret Manager for sensitive data
```bash
# Create secrets
echo -n "your-api-key" | gcloud secrets create api-football-key --data-file=-
echo -n "your-token" | gcloud secrets create matchbook-token --data-file=-

# Deploy with secrets
gcloud run deploy betting-api \
  --source . \
  --region us-central1 \
  --update-secrets API_FOOTBALL_KEY=api-football-key:latest \
  --update-secrets MATCHBOOK_SESSION_TOKEN=matchbook-token:latest
```

### 2. Require authentication
```bash
# Deploy with authentication required
gcloud run deploy betting-api \
  --source . \
  --region us-central1 \
  --no-allow-unauthenticated

# Create service account for scheduler
gcloud iam service-accounts create betting-scheduler

# Grant invoker role
gcloud run services add-iam-policy-binding betting-api \
  --member=serviceAccount:betting-scheduler@your-project.iam.gserviceaccount.com \
  --role=roles/run.invoker \
  --region=us-central1

# Update scheduler to use service account
gcloud scheduler jobs update http betting-workflow-runner \
  --oidc-service-account-email betting-scheduler@your-project.iam.gserviceaccount.com
```

## Cost Optimization

1. **Set max instances to 1** - Prevents concurrent runs
2. **Use minimum CPU/memory** - Start with 1 CPU, 1Gi RAM
3. **Set appropriate timeout** - 900s for workflow execution
4. **Use Cloud Storage for DuckDB** - Persistent data without keeping instance running

## Troubleshooting

### Check service health
```bash
curl https://your-service-url/health
```

### View recent errors
```bash
gcloud run services logs read betting-api \
  --region us-central1 \
  --limit 100 | grep ERROR
```

### Test scheduler job manually
```bash
gcloud scheduler jobs run betting-workflow-runner \
  --location us-central1
```

### Debug Cloud Run deployment
```bash
# Get service details
gcloud run services describe betting-api --region us-central1

# Get revisions
gcloud run revisions list --service betting-api --region us-central1
```

## Environment Variables

Required:
- `API_FOOTBALL_KEY` - API Football API key
- `MATCHBOOK_SESSION_TOKEN` - Matchbook session token

Optional:
- `PORT` - Server port (default: 8080)
- `LOG_LEVEL` - Logging level (default: INFO)

## Next Steps

1. Deploy to Cloud Run using the commands above
2. Set up Cloud Scheduler for periodic execution
3. Configure Cloud Storage for persistent DuckDB
4. Set up monitoring and alerting
5. Test with dry-run mode before enabling auto-betting
