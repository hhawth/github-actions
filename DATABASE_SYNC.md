# Database Sync with GCS

Automatic bi-directional sync between local DuckDB database and Google Cloud Storage.

## How It Works

1. **On Startup**: Downloads `football_data.duckdb` from GCS bucket
2. **During Runtime**: App reads/writes to local database (fast)
3. **Every Hour**: Uploads modified database back to GCS (automatic backup)

## Local Development

```bash
# Download database from GCS
python database_sync.py download

# Upload database to GCS
python database_sync.py upload

# Check status
./manual_sync.sh status
```

## Cloud Run Deployment

The Dockerfile automatically:
1. Installs gcloud CLI
2. Downloads database on container startup
3. Starts background thread for periodic uploads
4. Runs the API server

## Architecture

```
┌─────────────────────────────────────┐
│  Cloud Run Container                │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  api_server.py               │  │
│  │  - Starts sync worker        │  │
│  │  - Handles API requests      │  │
│  └──────────────────────────────┘  │
│              │                      │
│              v                      │
│  ┌──────────────────────────────┐  │
│  │  Local ephemeral storage     │  │
│  │  football_data.duckdb        │  │
│  └──────────────────────────────┘  │
│         │              ^            │
│         │              │            │
│     read/write    every hour        │
│         │              │            │
└─────────┼──────────────┼────────────┘
          │              │
          v              │
    ┌─────────────────────────────┐
    │  Google Cloud Storage       │
    │  gs://PROJECT-betting-data/ │
    │  football_data.duckdb       │
    └─────────────────────────────┘
```

## Configuration

**Environment Variables:**
- `GOOGLE_CLOUD_PROJECT` - GCP project ID (auto-set on Cloud Run)
- `GOOGLE_PROJECT` - Alternative project ID variable

**Bucket Name:** `${PROJECT_ID}-betting-data`

**Sync Interval:** 3600 seconds (1 hour) - configurable in `database_sync.py`

## Manual Sync

Use the helper script:

```bash
# Download from GCS
./manual_sync.sh download

# Upload to GCS
./manual_sync.sh upload

# Check sync status
./manual_sync.sh status
```

## Benefits

✅ **Fast local access**: DuckDB operates on local filesystem  
✅ **Automatic backups**: Changes synced to GCS every hour  
✅ **Small container images**: Database not baked into image  
✅ **Multiple instances**: All instances can access latest data  
✅ **Manual control**: Can trigger sync on demand

## Troubleshooting

**Database not found:**
```bash
# Check if bucket exists
gsutil ls gs://${GOOGLE_PROJECT}-betting-data/

# Create bucket if needed
gsutil mb -l europe-west2 gs://${GOOGLE_PROJECT}-betting-data/

# Upload initial database
gsutil cp football_data.duckdb gs://${GOOGLE_PROJECT}-betting-data/
```

**Permission errors:**
Ensure Cloud Run service account has Storage Object Admin role:
```bash
gcloud projects add-iam-policy-binding ${GOOGLE_PROJECT} \
    --member="serviceAccount:${GOOGLE_PROJECT}@appspot.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

## Files

- `database_sync.py` - Main sync utility with background worker
- `manual_sync.sh` - Helper script for manual sync operations  
- `api_server.py` - Modified to start sync worker on boot
- `Dockerfile` - Updated with gcloud CLI and sync command
