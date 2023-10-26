import requests
from bs4 import BeautifulSoup
import sqlite3 as sl

from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import numpy as np

PROMOTED_TEAMS = ["Burnley", "Luton Town", "Sheffield United"]

FIXTURES = None


def year_weightings(
    records, home=True, previous_year_weight=0.4, current_year_weight=0.6
):
    old = None
    current = None
    for row in records:
        if row[0] in PROMOTED_TEAMS:
            previous_year_weight = 0.2
            current_year_weight = 0.8
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
        if row[7] == "22/23":
            old = [scored_avg, scored_max, conceded, conceded_max]
        else:
            current = [scored_avg, scored_max, conceded, conceded_max]

    scored_avg_weighted = (
        old[0] * previous_year_weight + current[0] * current_year_weight
    )
    scored_max_weighted = (
        old[1] * previous_year_weight + current[1] * current_year_weight
    )
    conceded_avg_weighted = (
        old[2] * previous_year_weight + current[2] * current_year_weight
    )
    conceded_max_weighted = (
        old[3] * previous_year_weight + current[3] * current_year_weight
    )
    return (
        scored_avg_weighted,
        scored_max_weighted,
        conceded_avg_weighted,
        conceded_max_weighted,
    )


def normal_dist_calc(avg, range, skew):
    # print(range)
    value = skewnorm.rvs(skew, loc=avg, scale=(range / 4), size=100)
    value = value - min(value)  # Shift the set so the minimum value is equal to zero.
    value = value / max(value)  # Standadize all the vlues between 0 and 1.
    value = value * range  # Multiply the standardized values by the maximum value.
    np.mean(value)
    return np.mean(value)


def find_skew(percentage, conceded):
    # less skewed
    # if conceded:
    #     skew = (percentage * 10) - 5
    #     if percentage < 0.35:
    #         skew = percentage * -10
    # else:
    #     skew = (percentage * -10) + 5
    #     if percentage < 0.35:
    #         skew = percentage * 10

    # more skewed
    if conceded:
        skew = percentage * 10
        if percentage < 0.35:
            skew = -10 + (percentage * 10)
    else:
        skew = (percentage * - 10)
        if percentage < 0.35:
            skew = 10 - (percentage * 10)
    return skew


def estimate_value(
    home_stat,
    home_stat_max,
    away_stat,
    away_stat_max,
    home_win_percentage,
    away_win_percentage,
    home=True,
):
    if home:
        # home team to score againest away conceded
        home_skew = find_skew(home_win_percentage, conceded=False)
        away_skew = find_skew(away_win_percentage, conceded=True)
    else:
        # away team to score againest home conceded
        home_skew = find_skew(home_win_percentage, conceded=True)
        away_skew = find_skew(away_win_percentage, conceded=False)

    # print(home_skew, away_skew)

    home_mean = normal_dist_calc(home_stat, home_stat_max, home_skew)

    away_mean = normal_dist_calc(away_stat, away_stat_max, away_skew)

    return (home_mean + away_mean) / 2


def get_sky_sports_odds():
    global FIXTURES
    response = requests.get("https://www.skysports.com/premier-league-fixtures")
    soup = BeautifulSoup(response.text, features="html.parser")
    # breakpoint()
    FIXTURES = soup.find_all("div", {"class": "fixres__item"})[0:10]


def get_fixtures():
    global FIXTURES
    results = {}

    con = sl.connect("my-test.db")
    cur = con.cursor()

    for fixture in FIXTURES:
        teams = fixture.find_all("span", {"class": "swap-text__target"})
        # print(teams[0].text,"v", teams[1].text)
        fixture_text = teams[0].text + " vs " + teams[1].text
        home_team = teams[0].text
        away_team = teams[1].text
        odds = fixture.find_all("span", {"class": "matches__betting-odds"})
        print(teams)
        try:
            odds[0].text.split(" ")[1].split("/"), odds[1].text.split("/"), odds[
                2
            ].text.split(" ")[1].split("/")
            home_wins_odds = odds[0].text.split(" ")[1].split("/")
            home_wins_odds_as_percent = int(home_wins_odds[1]) / (
                int(home_wins_odds[0]) + int(home_wins_odds[1])
            )

            draw_odds = odds[1].text.split("/")
            draw_odds_as_percent = int(draw_odds[1]) / (
                int(draw_odds[0]) + int(draw_odds[1])
            )

            away_wins_odds = odds[2].text.split(" ")[1].split("/")
            away_wins_odds_as_percent = int(away_wins_odds[1]) / (
                int(away_wins_odds[0]) + int(away_wins_odds[1])
            )
        except:
            home_wins_odds_as_percent = None
            draw_odds_as_percent = None
            away_wins_odds_as_percent = None

        # print(home_wins_odds_as_percent, draw_odds_as_percent, away_wins_odds_as_percent)

        cur.execute(f"SELECT * from goals WHERE team = '{home_team}'")
        records = cur.fetchall()
        (
            home_scored_avg,
            home_scored_max,
            conceded_home,
            conceded_home_max,
        ) = year_weightings(records)

        cur.execute(f"SELECT * from goals WHERE team = '{away_team}'")
        records = cur.fetchall()
        (
            away_scored_avg,
            away_scored_max,
            conceded_away,
            conceded_away_max,
        ) = year_weightings(records, home=False)

        try:
            home_score_guess = estimate_value(
                home_scored_avg,
                home_scored_max,
                conceded_away,
                conceded_away_max,
                home_wins_odds_as_percent,
                away_wins_odds_as_percent,
            )
            away_score_guess = estimate_value(
                away_scored_avg,
                away_scored_max,
                conceded_home,
                conceded_home_max,
                home_wins_odds_as_percent,
                away_wins_odds_as_percent,
                home=False,
            )

            results[fixture_text] = [
                home_team,
                round(home_wins_odds_as_percent * 100, 1),
                round(home_score_guess, 1),
                away_team,
                round(away_wins_odds_as_percent * 100, 1),
                round(away_score_guess, 1),
            ]
        except:
            results[fixture_text] = [
                home_team,
                "N/A",
                "Sky Sports havent given odds",
                away_team,
                "N/A",
                "Sky Sports havent given odds",
            ]

    return results
