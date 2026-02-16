import duckdb

conn = duckdb.connect('football_data.duckdb')

# Check for El-Ittihad / Al Ittihad variations
print("=" * 70)
print("Checking for Ittihad fixtures (all variations):")
print("=" * 70)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE (home_team_name LIKE '%Ittihad%' OR away_team_name LIKE '%Ittihad%')
    AND date >= CURRENT_DATE
    ORDER BY date 
    LIMIT 10
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No Ittihad fixtures found")

# Check for Future FC / Modern Sport
print("\n" + "=" * 70)
print("Checking for Future FC / Modern Sport fixtures:")
print("=" * 70)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE (home_team_name LIKE '%Future%' OR away_team_name LIKE '%Future%'
           OR home_team_name LIKE '%Modern%' OR away_team_name LIKE '%Modern%')
    AND date >= CURRENT_DATE
    ORDER BY date 
    LIMIT 10
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No Future FC / Modern Sport fixtures found")

# Check for Kahraba Ismailia
print("\n" + "=" * 70)
print("Checking for Kahraba Ismailia fixtures:")
print("=" * 70)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE (home_team_name LIKE '%Kahraba%' OR away_team_name LIKE '%Kahraba%')
    AND date >= CURRENT_DATE
    ORDER BY date 
    LIMIT 10
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No Kahraba fixtures found")

# Check all upcoming Egyptian league fixtures
print("\n" + "=" * 70)
print("All upcoming Egyptian league fixtures:")
print("=" * 70)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, league_name, status 
    FROM fixtures 
    WHERE league_name LIKE '%Egypt%'
    AND date >= CURRENT_DATE
    ORDER BY date 
    LIMIT 20
""").fetchall()

if result:
    for r in result:
        print(f"Date: {r[1]}, {r[2]} vs {r[3]}, League: {r[4]}")
else:
    print("No Egyptian league fixtures found")

conn.close()
