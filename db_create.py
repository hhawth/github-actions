import sqlite3 as sl

con = sl.connect("my-test.db")
cur = con.cursor()
cur.execute("CREATE TABLE teams(team)")
cur.execute(
    "CREATE TABLE goals(team, scored_home, scored_away, conceded_home, conceded_away, cleansheet_home, cleansheet_away, year)"
)
cur.execute(
    "CREATE TABLE stats(team, wins_home, wins_away, draws_home, draws_away, loses_home, loses_away, year)"
)
