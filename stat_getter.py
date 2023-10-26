import requests
from bs4 import BeautifulSoup
import sqlite3 as sl

teams = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton and Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sheffield United",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]
def get_stats():
    url = "https://www.soccerstats.com/table.asp?league=england&tid=f"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="html.parser")
    scored_data_side = soup.find_all(id="btable")[1:3]
    con = sl.connect("my-test.db")
    cur = con.cursor()
    for i, side in enumerate(scored_data_side):
        home = True
        if i == 1:
            home = False
        scored_data_row = side.find_all("tr")[2:22]
        for j, row in enumerate(scored_data_row):
            gf_4plus, gf_3, gf_2, gf_1, gf_0, gf_avg, team,  ga_avg, ga_0, ga_1, ga_2, ga_3, ga_4plus = row.find_all("td")
            goals_for = [gf_4plus.text.strip(), gf_3.text.strip(), gf_2.text.strip(), gf_1.text.strip(), gf_0.text.strip(), gf_avg.text.strip()]
            max_goals_for = 0
            for i, stat in enumerate(goals_for):
                try:
                    int(stat)
                    max_goals_for = 4 - int(i)
                    break
                except:
                    continue
            goals_against = [ga_4plus.text.strip(), ga_3.text.strip(), ga_2.text.strip(), ga_1.text.strip(), ga_0.text.strip(), ga_avg.text.strip()]
            max_goals_against = 0
            for i, stat in enumerate(goals_against):
                try:
                    int(stat)
                    max_goals_against = 4 - int(i)
                    break
                except:
                    continue
            # print(team.text.strip(), teams[j], max_goals_for, gf_avg.text.strip(), max_goals_against, ga_avg.text.strip())
            if home:
                cur.execute(
                    f'UPDATE goals set scored_home = {float(gf_avg.text.strip())}, conceded_home = {float(ga_avg.text.strip())}, max_scored_home = {int(max_goals_for)}, max_conceded_home = {int(max_goals_against)} WHERE team = "{teams[j]}" AND year = "23/24"'
                )
            else:
                cur.execute(
                    f'UPDATE goals set scored_away = {float(gf_avg.text.strip())}, conceded_away = {float(ga_avg.text.strip())}, max_scored_away = {int(max_goals_for)}, max_conceded_away = {int(max_goals_against)} WHERE team = "{teams[j]}" AND year = "23/24"'
                )
            con.commit()
