#!/usr/bin/env python3
"""
Database sync utility for Cloud Run
- Downloads database from GCS on startup
- Periodically uploads local changes back to GCS
"""
import os
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime

DB_FILE = "football_data.duckdb"
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT")
BUCKET_NAME = f"{PROJECT_ID}-betting-data" if PROJECT_ID else None
GCS_PATH = f"gs://{BUCKET_NAME}/{DB_FILE}" if BUCKET_NAME else None

# Sync interval (seconds)
SYNC_INTERVAL = 3600  # 1 hour

def run_gsutil_command(cmd):
    """Run gsutil command with error handling"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except FileNotFoundError:
        return False, "", "gsutil not found"
    except Exception as e:
        return False, "", str(e)

def download_from_gcs():
    """Download database from GCS to local"""
    if not GCS_PATH:
        print("⚠️  No GOOGLE_CLOUD_PROJECT set, skipping GCS download")
        return False
    
    db_file = Path(DB_FILE)
    
    # Check if database already exists locally
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"✅ Database already exists locally ({size_mb:.1f} MB)")
        return True
    
    print(f"📥 Downloading database from {GCS_PATH}...")
    
    success, stdout, stderr = run_gsutil_command(["gsutil", "cp", GCS_PATH, DB_FILE])
    
    if success:
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"✅ Database downloaded successfully ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Failed to download database: {stderr}")
        return False

def upload_to_gcs():
    """Upload local database to GCS"""
    if not GCS_PATH:
        print("⚠️  No GOOGLE_CLOUD_PROJECT set, skipping GCS upload")
        return False
    
    db_file = Path(DB_FILE)
    
    if not db_file.exists():
        print("⚠️  No local database to upload")
        return False
    
    size_mb = db_file.stat().st_size / (1024 * 1024)
    print(f"📤 Uploading database to {GCS_PATH} ({size_mb:.1f} MB)...")
    
    # Use -h flag to disable caching
    success, stdout, stderr = run_gsutil_command([
        "gsutil", "-h", "Cache-Control:no-cache, max-age=0", 
        "cp", DB_FILE, GCS_PATH
    ])
    
    if success:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ Database uploaded successfully at {timestamp}")
        return True
    else:
        print(f"❌ Failed to upload database: {stderr}")
        return False

def get_local_db_mtime():
    """Get last modified time of local database"""
    db_file = Path(DB_FILE)
    if db_file.exists():
        return db_file.stat().st_mtime
    return None

def periodic_upload_worker():
    """Background worker that periodically uploads database to GCS"""
    print(f"🔄 Starting periodic upload worker (interval: {SYNC_INTERVAL}s)")
    
    last_mtime = get_local_db_mtime()
    
    while True:
        try:
            time.sleep(SYNC_INTERVAL)
            
            current_mtime = get_local_db_mtime()
            
            # Only upload if file has been modified
            if current_mtime and current_mtime != last_mtime:
                print("📝 Database has been modified, uploading to GCS...")
                if upload_to_gcs():
                    last_mtime = current_mtime
            else:
                print("✓ Database unchanged, skipping upload")
                
        except KeyboardInterrupt:
            print("🛑 Upload worker stopped")
            break
        except Exception as e:
            print(f"❌ Error in upload worker: {e}")
            time.sleep(60)  # Wait a bit before retrying

def start_periodic_upload(daemon=True):
    """Start background thread for periodic uploads"""
    thread = threading.Thread(target=periodic_upload_worker, daemon=daemon)
    thread.start()
    return thread

def ensure_database_exists():
    """Ensure database exists, download from GCS or create empty"""
    import duckdb
    
    db_file = Path(DB_FILE)
    
    # Try to download from GCS first
    if not db_file.exists():
        download_from_gcs()
    
    # If still doesn't exist, create empty database
    if not db_file.exists():
        print("📝 Creating empty database...")
        try:
            # Try to import and create tables
            from duckdb.tables import create_all_tables
            conn = duckdb.connect(str(db_file))
            create_all_tables(conn)
            conn.close()
            print("✅ Empty database created with schema")
            
            # Upload the new database to GCS
            upload_to_gcs()
        except ImportError:
            # Fallback: create minimal database
            conn = duckdb.connect(str(db_file))
            conn.close()
            print("✅ Empty database created (minimal)")
    
    return db_file.exists()

if __name__ == "__main__":
    import sys
    
    command = sys.argv[1] if len(sys.argv) > 1 else "setup"
    
    if command == "download":
        success = download_from_gcs()
        sys.exit(0 if success else 1)
    
    elif command == "upload":
        success = upload_to_gcs()
        sys.exit(0 if success else 1)
    
    elif command == "sync-loop":
        # Run continuous sync loop
        ensure_database_exists()
        periodic_upload_worker()
    
    elif command == "setup":
        # Initial setup (download or create)
        success = ensure_database_exists()
        sys.exit(0 if success else 1)
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python database_sync.py [download|upload|sync-loop|setup]")
        sys.exit(1)
