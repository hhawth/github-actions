import requests
from bs4 import BeautifulSoup
import sqlite3 as sl

from scipy.stats import skewnorm
import numpy as np
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

PROMOTED_TEAMS = ["Burnley", "Luton Town", "Sheffield United"]

FIXTURES_LAST_CALL = 0

FIXTURES = None

def filter_db_results(records, home=True):
    for row in records:
        if home:
            scored_avg = row[1]
            scored_max = row[8]
            conceded = row[3]
            conceded_max = row[10]
        else:
            scored_avg = row[2]
            scored_max = row[9]
            conceded = row[4]
            conceded_max = row[11]
    return (
        scored_avg,
        scored_max,
        conceded,
        conceded_max
    )

def normal_dist_calc(avg, range, skew):
    value = skewnorm.rvs(skew, loc=avg, scale=(range / 4), size=1000)
    value = value - min(value)  # Shift the set so the minimum value is equal to zero.
    value = value / max(value)  # Standadize all the vlues between 0 and 1.
    value = value * range  # Multiply the standardized values by the maximum value.
    return np.mean(value)

def estimate_goals_scored(
    team_scored_avg,
    team_scored_max,
    other_team_conceded_avg,
    other_team_conceded_max,
    win_percentage
):
    win_threshold = 0.5
    scaler = 1/win_threshold
    if win_percentage > win_threshold:
        scored_skew = -(1/(1 - win_percentage)*scaler) ** 2
        conceded_skew = -(1/(1 - win_percentage)*scaler) ** 2
    else:
        scored_skew = (1/(win_percentage*scaler)) ** 2
        conceded_skew = (1/(win_percentage*scaler)) ** 2


    home_mean = normal_dist_calc(team_scored_avg, team_scored_max, scored_skew)

    away_mean = normal_dist_calc(other_team_conceded_avg, other_team_conceded_max, conceded_skew)

    return (home_mean + away_mean) / 2


def get_sky_sports_odds():
    global FIXTURES

    LOGGER.info("Getting sky sports odds")
    response = requests.get("https://www.skysports.com/premier-league-fixtures")
    soup = BeautifulSoup(response.text, features="html.parser")
    FIXTURES = soup.find_all("div", {"class": "fixres__item"})[0:10]


def get_fixtures():
    global FIXTURES

    results = {}

    con = sl.connect("my-test.db")
    cur = con.cursor()

    for fixture in FIXTURES:
        teams = fixture.find_all("span", {"class": "swap-text__target"})
        fixture_text = teams[0].text + " vs " + teams[1].text
        home_team = teams[0].text
        away_team = teams[1].text
        odds = fixture.find_all("span", {"class": "matches__betting-odds"})
        try:
            odds[0].text.split(" ")[1].split("/"), odds[1].text.split("/"), odds[
                2
            ].text.split(" ")[1].split("/")
            home_wins_odds = odds[0].text.split(" ")[1].split("/")
            home_team_wins_odds_as_percent = int(home_wins_odds[1]) / (
                int(home_wins_odds[0]) + int(home_wins_odds[1])
            )

            draw_odds = odds[1].text.split("/")
            draw_odds_as_percent = int(draw_odds[1]) / (
                int(draw_odds[0]) + int(draw_odds[1])
            )

            away_wins_odds = odds[2].text.split(" ")[1].split("/")
            away_team_wins_odds_as_percent = int(away_wins_odds[1]) / (
                int(away_wins_odds[0]) + int(away_wins_odds[1])
            )
            sky_sports_profit = (home_team_wins_odds_as_percent + draw_odds_as_percent + away_team_wins_odds_as_percent) - 1
            home_team_wins_odds_as_percent = home_team_wins_odds_as_percent - (sky_sports_profit/3)
            draw_odds_as_percent = draw_odds_as_percent - (sky_sports_profit/3)
            away_team_wins_odds_as_percent = away_team_wins_odds_as_percent - (sky_sports_profit/3)
        except:
            home_team_wins_odds_as_percent = None
            draw_odds_as_percent = None
            away_team_wins_odds_as_percent = None
        

        cur.execute(f"SELECT * from goals WHERE team = '{home_team}' AND year = '23/24'")
        records = cur.fetchall()
        (
            home_team_scored_avg_at_home,
            home_team_scored_max_at_home,
            home_team_conceded_at_home,
            home_team_conceded_at_home_max,
        ) = filter_db_results(records)

        cur.execute(f"SELECT * from goals WHERE team = '{away_team}' AND year = '23/24'")
        records = cur.fetchall()
        (
            away_team_scored_avg_away,
            away_team_scored_max_away,
            away_team_conceded_away,
            away_team_conceded_away_max,
        ) = filter_db_results(records, home=False)

        try:
            home_score_guess = estimate_goals_scored(
                home_team_scored_avg_at_home,
                home_team_scored_max_at_home,
                away_team_conceded_away,
                away_team_conceded_away_max,
                home_team_wins_odds_as_percent
            )
            away_score_guess = estimate_goals_scored(
                away_team_scored_avg_away,
                away_team_scored_max_away,
                home_team_conceded_at_home,
                home_team_conceded_at_home_max,
                away_team_wins_odds_as_percent
            )

            results[fixture_text] = [
                home_team,
                round(home_team_wins_odds_as_percent * 100, 1),
                round(home_score_guess, 1),
                round(draw_odds_as_percent * 100, 1),
                away_team,
                round(away_team_wins_odds_as_percent * 100, 1),
                round(away_score_guess, 1),
            ]
        except:
            results[fixture_text] = [
                home_team,
                "N/A",
                "Sky Sports havent given odds",
                "N/A",
                away_team,
                "N/A",
                "Sky Sports havent given odds",
            ]

    return results
