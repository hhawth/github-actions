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
BUCKET_NAME = os.getenv("GOOGLE_BUCKET") or (f"{PROJECT_ID}-bucket" if PROJECT_ID else None)
GCS_PATH = f"gs://{BUCKET_NAME}/{DB_FILE}" if BUCKET_NAME else None

# Debug environment variables
print(f"🔧 Debug: GOOGLE_PROJECT={PROJECT_ID}")
print(f"🔧 Debug: GOOGLE_BUCKET={BUCKET_NAME}")
print(f"🔧 Debug: GCS_PATH={GCS_PATH}")

# Sync interval (seconds)
SYNC_INTERVAL = 3600  # 1 hour

def run_gsutil_command(cmd):
    """Run gsutil command with error handling"""
    print(f"🔧 Debug: Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        print(f"🔧 Debug: Command exit code: {result.returncode}")
        if result.stdout:
            print(f"🔧 Debug: Stdout: {result.stdout[:200]}...")
        if result.stderr:
            print(f"🔧 Debug: Stderr: {result.stderr[:200]}...")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("❌ Debug: Command timed out")
        return False, "", "Command timed out"
    except FileNotFoundError as e:
        print(f"❌ Debug: gsutil not found: {e}")
        return False, "", "gsutil not found"
    except Exception as e:
        print(f"❌ Debug: Command failed: {e}")
        return False, "", str(e)

def download_from_gcs():
    """Download database from GCS to local"""
    print("🔧 Debug: Attempting GCS download...")
    print(f"🔧 Debug: GCS_PATH = {GCS_PATH}")
    print(f"🔧 Debug: BUCKET_NAME = {BUCKET_NAME}")
    print(f"🔧 Debug: DB_FILE = {DB_FILE}")
    print(f"🔧 Debug: Current working directory: {os.getcwd()}")
    
    if not GCS_PATH:
        print("❌ No GOOGLE_CLOUD_PROJECT or GOOGLE_BUCKET set, skipping GCS download")
        print(f"🔧 Debug: PROJECT_ID={PROJECT_ID}, BUCKET_NAME={BUCKET_NAME}")
        return False
    
    db_file = Path(DB_FILE)
    
    # Check if database already exists locally and remove it to force fresh download
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"🔧 Debug: Local database exists ({size_mb:.1f} MB), removing for fresh download")
        try:
            db_file.unlink()
            print("🔧 Debug: Removed existing local database")
        except Exception as e:
            print(f"❌ Failed to remove existing database: {e}")
            return False
    
    print(f"📥 Starting download from {GCS_PATH}...")
    
    # Try to test gsutil first
    test_success, test_stdout, test_stderr = run_gsutil_command(["gsutil", "version"])
    if not test_success:
        print(f"❌ gsutil not available: {test_stderr}")
        return False
        
    print(f"✅ gsutil is available: {test_stdout[:100]}...")
    
    # Check if file exists in GCS first
    print("🔧 Debug: Checking if file exists in GCS...")
    ls_success, ls_stdout, ls_stderr = run_gsutil_command(["gsutil", "ls", "-l", GCS_PATH])
    if ls_success:
        print(f"🔧 Debug: GCS file info: {ls_stdout.strip()}")
    else:
        print(f"❌ File not found in GCS: {ls_stderr}")
        return False
    
    # Download with progress
    success, stdout, stderr = run_gsutil_command(["gsutil", "-m", "cp", GCS_PATH, DB_FILE])
    
    if success:
        print("🔧 Debug: Download command succeeded")
        if db_file.exists():
            size_mb = db_file.stat().st_size / (1024 * 1024)
            print(f"✅ Database downloaded successfully ({size_mb:.1f} MB)")
            
            # Verify the downloaded database  
            try:
                import duckdb
                conn = duckdb.connect(str(db_file))
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0].lower() for table in tables]
                conn.close()
                print(f"🔧 Debug: Downloaded database contains tables: {table_names}")
                print(f"🔧 Debug: Table count: {len(table_names)}")
            except Exception as e:
                print(f"⚠️  Warning: Could not verify downloaded database: {e}")
            
            return True
        else:
            print("❌ Download succeeded but file doesn't exist locally")
            return False
    else:
        print(f"❌ Failed to download database from {GCS_PATH}")
        print(f"❌ Error details: {stderr}")
        print(f"❌ Stdout: {stdout}")
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
    """Ensure database exists and is valid, download from GCS if incomplete"""
    import duckdb
    
    print("🔧 Debug: Starting ensure_database_exists()")
    print(f"🔧 Debug: Looking for database at: {DB_FILE}")
    print(f"🔧 Debug: Current working directory: {os.getcwd()}")
    
    db_file = Path(DB_FILE)
    needs_download = False
    
    # Check if local database exists and is complete
    if not db_file.exists():
        print("📥 No local database found")
        needs_download = True
    else:
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"🔧 Debug: Local database found ({size_mb:.1f} MB)")
        
        # Validate database has required schema
        try:
            print("🔧 Debug: Connecting to database for validation...")
            conn = duckdb.connect(str(db_file))
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0].lower() for table in tables]
            conn.close()
            
            print(f"🔧 Debug: Database validation - found {len(tables)} tables")
            print(f"🔧 Debug: Table names: {table_names}")
            
            required_tables = ['bet_history', 'fixtures', 'odds', 'predictions']
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                print(f"⚠️  Local database missing required tables: {missing_tables}")
                print(f"📊 Available tables: {table_names}")
                print("🔄 Will replace incomplete database with complete version from GCS...")
                needs_download = True
            else:
                print(f"✅ Local database is complete - Found tables: {table_names}")
                return True
                
        except Exception as e:
            print(f"❌ Error validating local database: {e}")
            print("🔄 Will download fresh database from GCS...")
            needs_download = True
    
    # Download from GCS if needed (this will overwrite incomplete local file)
    if needs_download:
        print("📥 Downloading complete database from GCS...")
        if not download_from_gcs():
            print("❌ CRITICAL: Could not download database from GCS")
            if db_file.exists():
                print("⚠️  Will use incomplete local database - workflow may create missing tables")
                return True
            else:
                print("❌ No local database available - cannot start")
                return False
        
        # Validate the downloaded database
        try:
            print("🔧 Debug: Validating downloaded database...")
            conn = duckdb.connect(str(db_file))
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0].lower() for table in tables]
            conn.close()
            
            required_tables = ['bet_history', 'fixtures', 'odds', 'predictions']
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                print(f"⚠️  Downloaded database still missing tables: {missing_tables}")
                print(f"📊 Available tables: {table_names}")
                print("📝 Tables will be created by workflow on first run")
            else:
                print(f"✅ Downloaded database validated successfully - Found tables: {table_names}")
            
            return True
            
        except Exception as e:
            print(f"❌ CRITICAL: Downloaded database validation failed: {e}")
            return False
    
    return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Database validation failed: {e}")
        print("❌ Database may be corrupted")
        return False

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
