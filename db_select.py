import sqlite3 as sl

con = sl.connect("my-test.db")
cur = con.cursor()
cur.execute("SELECT * from goals")
records = cur.fetchall()
for row in records:
    print(row)
