import requests
from bs4 import BeautifulSoup
import sqlite3 as sl

teams_url = {
    "Manchester City": "https://footystats.org/clubs/manchester-city-fc-93",
    "Tottenham Hotspur": "https://footystats.org/clubs/tottenham-hotspur-fc-92",
    "Liverpool": "https://footystats.org/clubs/liverpool-fc-151",
    "West Ham United": "https://footystats.org/clubs/west-ham-united-fc-153",
    "Arsenal": "https://footystats.org/clubs/arsenal-fc-59",
    "Brighton and Hove Albion": "https://footystats.org/clubs/brighton-hove-albion-fc-209",
    "Crystal Palace": "https://footystats.org/clubs/crystal-palace-fc-143",
    "Brentford": "https://footystats.org/clubs/brentford-fc-218",
    "Nottingham Forest": "https://footystats.org/clubs/nottingham-forest-fc-211",
    "Aston Villa": "https://footystats.org/clubs/aston-villa-fc-158",
    "Manchester United": "https://footystats.org/clubs/manchester-united-fc-149",
    "Chelsea": "https://footystats.org/clubs/chelsea-fc-152",
    "Fulham": "https://footystats.org/clubs/fulham-fc-162",
    "Newcastle United": "https://footystats.org/clubs/newcastle-united-fc-157",
    "Wolverhampton Wanderers": "https://footystats.org/clubs/wolverhampton-wanderers-fc-223",
    "Bournemouth": "https://footystats.org/clubs/afc-bournemouth-148",
    "Sheffield United": "https://footystats.org/clubs/sheffield-united-fc-251",
    "Everton": "https://footystats.org/clubs/everton-fc-144",
    "Luton Town": "https://footystats.org/clubs/luton-town-fc-271",
    "Burnley": "https://footystats.org/clubs/burnley-fc-145",
}


# get all teams current stats
def get_stats():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    con = sl.connect("my-test.db")
    cur = con.cursor()
    for team, url in teams_url.items():
        # print(team)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, features="html.parser")
        scored_data = soup.find(id="scored-0").find("table").find_all("td")
        chunk_size = 4
        while scored_data:
            chunk, scored_data = scored_data[:chunk_size], scored_data[chunk_size:]
            title, overall, home, away = chunk
            if title.text == "Scored / Match":
                scored_home = float(home.text)
                scored_away = float(away.text)
                # print(title.text, overall.text, home.text, away.text)

            if title.text == "Highest Scored in a Match":
                max_scored_home = int(home.text.split(" ")[0])
                max_scored_away = int(away.text.split(" ")[0])
                # print(title.text, overall.text, home.text, away.text)

        conceded_data = soup.find(id="conceded-0").find("table").find_all("td")
        chunk_size = 4
        while conceded_data:
            chunk, conceded_data = (
                conceded_data[:chunk_size],
                conceded_data[chunk_size:],
            )
            title, overall, home, away = chunk
            if title.text == "Conceded / Match":
                conceded_home = float(home.text)
                conceded_away = float(away.text)
                # print(title.text, overall.text, home.text, away.text)
            if title.text == "Highest Conceded in a Match":
                max_conceded_home = int(home.text.split(" ")[0])
                max_conceded_away = int(away.text.split(" ")[0])
                # print(title.text, overall.text, home.text, away.text)
            if title.text == "Clean Sheets %":
                cleansheet_home = float(home.text.strip("%")) / 100
                cleansheet_away = float(away.text.strip("%")) / 100
                # print(title.text, overall.text, cleansheet_home, cleansheet_away)

        # conceded_data = soup.find(id="overview-0").find("table").find_all("td")
        # chunk_size = 4
        # while conceded_data:
        #     chunk, conceded_data = conceded_data[:chunk_size], conceded_data[chunk_size:]
        #     title, overall, home, away = chunk
        #     if title.text == "Wins":
        #         print(title.text, overall.text, home.text, away.text)

        #     if title.text == "Draws":
        #         print(title.text, overall.text, home.text, away.text)

        #     if title.text == "Losses":
        #         print(title.text, overall.text, home.text, away.text)

        #     if title.text == "Clean Sheets %":
        #         print(title.text, overall.text, home.text, away.text)

        cur.execute(
            f'UPDATE goals set scored_home = {scored_home}, scored_away = {scored_away}, conceded_home = {conceded_home}, conceded_away = {conceded_away}, cleansheet_home = {cleansheet_home}, cleansheet_away = {cleansheet_away}, max_scored_home = {max_scored_home}, max_scored_away = {max_scored_away}, max_conceded_home = {max_conceded_home}, max_conceded_away = {max_conceded_away} WHERE team = "{team}" AND year = "23/24"'
        )
        con.commit()
