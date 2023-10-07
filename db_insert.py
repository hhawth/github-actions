import sqlite3 as sl

con = sl.connect('my-test.db')
cur = con.cursor()
# teams = [
# "Arsenal"
# "Aston Villa",
# "Bournemouth",
# "Brentford",
# "Brighton and Hove Albion",
# "Burnley",
# "Chelsea",
# "Crystal Palace",
# "Everton",
# "Fulham",
# "Liverpool",
# "Luton Town",
# "Manchester City",
# "Manchester United",
# "Newcastle United",
# "Nottingham Forest",
# "Sheffield United",
# "Tottenham Hotspur",
# "West Ham United",
# "Wolverhampton Wanderers",]
# for x in teams:
cur.execute("INSERT into goals(team, scored_home, scored_away, conceded_home, conceded_away, cleansheet_home, cleansheet_away, year, max_scored_home, max_scored_away, max_conceded_home, max_conceded_away) VALUES('Liverpool',2.42, 1.53, 0.89, 1.58, 0.47, 0.26, '22/23', 9, 6, 3, 4)")
# cur.execute(f"UPDATE goals set max_scored_home = 3, max_scored_away = 2, max_conceded_home = 4, max_conceded_away = 6 WHERE team = 'Wolverhampton Wanderers' AND year = '22/23'")
# cur.execute("ALTER TABLE goals ADD max_scored_home, max_scored_away, max_conceded_home, max_conceded_away")
con.commit()