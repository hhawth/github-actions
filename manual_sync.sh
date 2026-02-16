#!/bin/bash
# Manual database sync utility

case "$1" in
    download)
        echo "📥 Downloading database from GCS..."
        python database_sync.py download
        ;;
    upload)
        echo "📤 Uploading database to GCS..."
        python database_sync.py upload
        ;;
    status)
        echo "📊 Database status:"
        if [ -f "football_data.duckdb" ]; then
            SIZE=$(du -h football_data.duckdb | cut -f1)
            MTIME=$(stat -c %y football_data.duckdb 2>/dev/null || stat -f "%Sm" football_data.duckdb)
            echo "  Local: ${SIZE} (modified: ${MTIME})"
        else
            echo "  Local: Not found"
        fi
        
        if [ -n "$GOOGLE_PROJECT" ]; then
            BUCKET="${GOOGLE_PROJECT}-betting-data"
            echo "  GCS: Checking gs://${BUCKET}/football_data.duckdb..."
            gsutil ls -lh "gs://${BUCKET}/football_data.duckdb" 2>/dev/null || echo "  GCS: Not found"
        fi
        ;;
    *)
        echo "Usage: $0 {download|upload|status}"
        echo ""
        echo "Commands:"
        echo "  download  - Download database from GCS"
        echo "  upload    - Upload database to GCS"
        echo "  status    - Show database status (local and GCS)"
        exit 1
        ;;
esac
