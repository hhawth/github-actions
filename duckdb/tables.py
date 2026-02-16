
import duckdb

def create_tables():
    # Create persistent database file
    conn = duckdb.connect('football_data.duckdb')
    
    # Create table schemas
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
            status TEXT NOT NULL,
            league_id INTEGER NOT NULL,
            league_name TEXT NOT NULL,
            league_country TEXT NOT NULL,
            season INTEGER NOT NULL,
            home_team_id INTEGER NOT NULL,
            home_team_name TEXT NOT NULL,
            away_team_id INTEGER NOT NULL,
            away_team_name TEXT NOT NULL,
            goals_half_time_home INTEGER NOT NULL,
            goals_half_time_away INTEGER NOT NULL,
            goals_full_time_home INTEGER NOT NULL,
            goals_full_time_away INTEGER NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            fixture_id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
            league_id INTEGER NOT NULL,
            league_name TEXT NOT NULL,
            league_country TEXT NOT NULL,
            odds json NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            fixture_id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
            league_id INTEGER NOT NULL,
            league_name TEXT NOT NULL,
            league_country TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            home_team_name TEXT NOT NULL,
            home_team_prediction json NOT NULL,
            away_team_id INTEGER NOT NULL,
            away_team_name TEXT NOT NULL,
            away_team_prediction json NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bet_history (
            runner_id BIGINT PRIMARY KEY,
            match_name TEXT,
            league TEXT,
            market TEXT,
            runner_name TEXT,
            stake REAL,
            odds REAL,
            expected_value REAL,
            confidence REAL,
            placed_at TIMESTAMP
        );
    """)
    conn.close()
    print("✅ Database saved successfully to: football_data.duckdb")
    print("✅ Tables created: fixtures, odds, predictions, bet_history")
    print("\nNext steps to populate tables:")
    print("- Use INSERT statements to load data from your JSON files")
    print("- Or use DuckDB's direct JSON loading with your table schema")

def populate_fixtures_from_json():
    """Example function to load JSON data into the flattened table"""
    conn = duckdb.connect('football_data.duckdb')
    
    try:
        # Load and insert data from JSON files
        conn.execute("""
            INSERT INTO fixtures 
            SELECT 
                UNNEST(fixtures).fixture.id as fixture_id,
                UNNEST(fixtures).fixture.date::DATE as date,
                UNNEST(fixtures).fixture.status.long as status,
                UNNEST(fixtures).league.id as league_id,
                UNNEST(fixtures).league.name as league_name,
                UNNEST(fixtures).league.country as league_country,
                UNNEST(fixtures).league.season as season,
                UNNEST(fixtures).teams.home.id as home_team_id,
                UNNEST(fixtures).teams.home.name as home_team_name,
                UNNEST(fixtures).teams.away.id as away_team_id,
                UNNEST(fixtures).teams.away.name as away_team_name,
                COALESCE(UNNEST(fixtures).score.halftime.home, 0) as goals_half_time_home,
                COALESCE(UNNEST(fixtures).score.halftime.away, 0) as goals_half_time_away,
                COALESCE(UNNEST(fixtures).score.fulltime.home, 0) as goals_full_time_home,
                COALESCE(UNNEST(fixtures).score.fulltime.away, 0) as goals_full_time_away
            FROM 'api_football_fixtures_2026-02-01.json'
            ON CONFLICT (fixture_id) DO NOTHING;
        """)
        
        count = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
        print(f"✅ Loaded {count} fixtures into database")
        
    except Exception as e:
        print(f"❌ Error loading fixtures: {e}")
    
    conn.close()

if __name__ == "__main__":
    create_tables()
    # Uncomment to populate with data:
    # populate_fixtures_from_json()