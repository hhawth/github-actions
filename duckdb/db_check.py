import duckdb

def check_database():
    conn = duckdb.connect('football_data.duckdb')

    # Check if tables exist
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [table[0] for table in tables]
    if 'fixtures' in table_names:
        print("✅ 'fixtures' table exists")
    else:
        print("❌ 'fixtures' table is missing")

    print(conn.execute("SELECT * FROM fixtures where fixture_id = 1517388").fetchall())  # Example query to check data
    print(conn.execute("SELECT * FROM odds where fixture_id = 1517388").fetchall())  # Check total number of records
    print(conn.execute("SELECT * FROM predictions where fixture_id = 1517388").fetchall())  # Check total number of records
    conn.close()

if __name__ == "__main__":
    check_database()