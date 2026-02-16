import duckdb

conn = duckdb.connect('football_data.duckdb')

# Check for Al Ittihad
print("=" * 60)
print("Checking for Al Ittihad fixtures:")
print("=" * 60)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE home_team_name LIKE '%Ittihad%' OR away_team_name LIKE '%Ittihad%'
    ORDER BY date DESC 
    LIMIT 5
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No Al Ittihad fixtures found")

# Check for Smouha SC
print("\n" + "=" * 60)
print("Checking for Smouha SC fixtures:")
print("=" * 60)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE home_team_name LIKE '%Smouha%' OR away_team_name LIKE '%Smouha%'
    ORDER BY date DESC 
    LIMIT 5
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No Smouha SC fixtures found")

# Check upcoming fixtures with both teams
print("\n" + "=" * 60)
print("Checking for Al Ittihad vs Smouha SC (upcoming):")
print("=" * 60)
result = conn.execute("""
    SELECT fixture_id, date, home_team_name, away_team_name, status 
    FROM fixtures 
    WHERE date >= CURRENT_DATE
    AND (
        (home_team_name LIKE '%Ittihad%' AND away_team_name LIKE '%Smouha%')
        OR (home_team_name LIKE '%Smouha%' AND away_team_name LIKE '%Ittihad%')
    )
    LIMIT 5
""").fetchall()

if result:
    for r in result:
        print(f"ID: {r[0]}, Date: {r[1]}, {r[2]} vs {r[3]}, Status: {r[4]}")
else:
    print("No upcoming Al Ittihad vs Smouha SC fixtures found")

# Count total upcoming fixtures
print("\n" + "=" * 60)
print("Total upcoming fixtures in database:")
print("=" * 60)
count = conn.execute("""
    SELECT COUNT(*) FROM fixtures WHERE date >= CURRENT_DATE
""").fetchone()[0]
print(f"Total: {count} upcoming fixtures")

conn.close()
