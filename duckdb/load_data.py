import duckdb
import json
from pathlib import Path

def populate_fixtures_from_json(json_file_path):
    conn = duckdb.connect('football_data.duckdb')
    
    print(f"Processing: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'response' in data:
        fixtures_list = data['response']  # API response format
    elif 'fixtures' in data:
        fixtures_list = data['fixtures']  # Your merged format
    elif isinstance(data, list):
        fixtures_list = data  # Direct array format
    else:
        fixtures_list = [data]  # Single fixture format
    
    print(f"Found {len(fixtures_list)} fixtures to process")
    inserted_count = 0
    batch_size = 100  # Process in smaller batches
    batch_count = 0
    
    for i, fixture in enumerate(fixtures_list):
        try:
            # Debug: Check if fixture is the right type
            if not isinstance(fixture, dict):
                print(f"⚠️  Skipping non-dict fixture: {type(fixture)} - {fixture}")
                continue
                
            # Extract data with safe access
            fixture_id = fixture['fixture']['id']
            date = fixture['fixture']['date'][:19]  # Get YYYY-MM-DDTHH:MM:SS part only
            status = fixture['fixture']['status']['long']
            league_id = fixture['league']['id']
            league_name = fixture['league']['name'].replace("'", "''")
            league_country = fixture['league']['country'].replace("'", "''")
            season = fixture['league']['season']
            home_team_id = fixture['teams']['home']['id']
            home_team_name = fixture['teams']['home']['name'].replace("'", "''")
            away_team_id = fixture['teams']['away']['id']
            away_team_name = fixture['teams']['away']['name'].replace("'", "''")
            
            # Handle potentially null scores with safe type checking
            score_data = fixture.get('score', {})
            if isinstance(score_data, dict):
                halftime_data = score_data.get('halftime', {})
                fulltime_data = score_data.get('fulltime', {})
                if isinstance(halftime_data, dict) and isinstance(fulltime_data, dict):
                    score_ht_home = halftime_data.get('home')
                    score_ht_away = halftime_data.get('away')
                    score_ft_home = fulltime_data.get('home')
                    score_ft_away = fulltime_data.get('away')
                else:
                    score_ht_home = score_ht_away = score_ft_home = score_ft_away = None
            else:
                score_ht_home = score_ht_away = score_ft_home = score_ft_away = None
            
            # Convert None to 0, keep numbers as-is
            goals_half_time_home = 0 if score_ht_home is None else score_ht_home
            goals_half_time_away = 0 if score_ht_away is None else score_ht_away
            goals_full_time_home = 0 if score_ft_home is None else score_ft_home
            goals_full_time_away = 0 if score_ft_away is None else score_ft_away

            # Extract odds and predictions data
            odds_data = fixture.get('odds', {"odds": "N/A"})
            if fixture.get('predictions') is None or isinstance(fixture.get('predictions'), list):
                home_predictions_data = {"prediction": "N/A"}
                away_predictions_data = {"prediction": "N/A"}
            else:
                home_predictions_data = fixture.get('predictions', {}).get('teams', {}).get('home', {})
                away_predictions_data = fixture.get('predictions', {}).get('teams', {}).get('away', {})

            # Insert into fixtures table
            conn.execute("""
                INSERT OR REPLACE INTO fixtures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fixture_id, date, status, league_id, league_name, league_country, season,
                home_team_id, home_team_name, away_team_id, away_team_name,
                goals_half_time_home, goals_half_time_away, goals_full_time_home, goals_full_time_away
            ])
            
            # Insert into odds table
            conn.execute("""
                INSERT OR REPLACE INTO odds VALUES (?, ?, ?, ?, ?, ?)
            """, [
                fixture_id, date, league_id, league_name, league_country, json.dumps(odds_data)
            ])
            
            # Insert into predictions table
            conn.execute("""
                INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fixture_id, date, league_id, league_name, league_country,
                home_team_id, home_team_name, json.dumps(home_predictions_data),
                away_team_id, away_team_name, json.dumps(away_predictions_data)
            ])
            inserted_count += 1
            
            # Progress indicator for large files
            if (i + 1) % batch_size == 0:
                batch_count += 1
                print(f"  Processed {i + 1}/{len(fixtures_list)} fixtures (batch {batch_count})")
            
        except KeyError as e:
            fixture_id = "unknown"
            if isinstance(fixture, dict):
                fixture_id = fixture.get('fixture', {}).get('id', 'unknown') if isinstance(fixture.get('fixture'), dict) else 'unknown'
            print(f"❌ Missing field {e} in fixture {fixture_id}")
        except Exception as e:
            fixture_id = "unknown"  
            if isinstance(fixture, dict):
                fixture_id = fixture.get('fixture', {}).get('id', 'unknown') if isinstance(fixture.get('fixture'), dict) else 'unknown'
            print(f"❌ Error processing fixture {fixture_id}: {e}")
            # Don't print debug info for every error to avoid spam
            if inserted_count < 10:  # Only show details for first few errors
                print(f"   Fixture type: {type(fixture)}")
                if isinstance(fixture, dict):
                    print(f"   Available keys: {list(fixture.keys())}")
    
    conn.close()
    print(f"✅ Loaded {inserted_count} fixtures from: {Path(json_file_path).name}")

def load_all_fixture_files():
    """Load all fixture JSON files in the current directory"""
    current_dir = Path('.')
    
    # Find all fixture JSON files
    fixture_patterns = [
        'api_football_merged_*.json'
    ]
    
    all_files = []
    for pattern in fixture_patterns:
        all_files.extend(current_dir.glob(pattern))
    
    if not all_files:
        print("❌ No fixture JSON files found!")
        print("Looking for files matching: api_football_fixtures_*.json or api_football_merged_*.json")
        return
    
    print(f"📁 Found {len(all_files)} fixture files to process")
    
    total_loaded = 0
    for file_path in sorted(all_files):
        try:
            populate_fixtures_from_json(str(file_path))
            total_loaded += 1
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
    
    print(f"\n🎉 Completed! Processed {total_loaded}/{len(all_files)} files")
    
    # Show final stats
    conn = duckdb.connect('football_data.duckdb')
    total_fixtures = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
    print(f"📊 Total fixtures in database: {total_fixtures}")
    conn.close()

if __name__ == "__main__":
    load_all_fixture_files()