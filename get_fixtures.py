import requests
from bs4 import BeautifulSoup

import logging
from stat_getter import get_stats, get_top_booked, get_top_scorers, cached_function, get_form, get_relative_performance
from calculations import calculate_form_score, calculate_win_probability, simulate_multiple_matches

# Create a cache with a time-to-live (TTL) of 1 hour (3600 seconds)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

FIXTURES_LAST_CALL = 0

FIXTURES = None

@cached_function(maxsize=100, ttl=3600)
def get_fixtures_and_odds():
    LOGGER.info("Getting sky sports odds")
    response = requests.get(
        "https://www.sportytrader.com/en/odds/football/england/premier-league-49/"
    )
    soup = BeautifulSoup(response.text, features="html.parser")
    return soup.find_all(
        "div",
        {
            "class": "cursor-pointer border rounded-md mb-4 px-1 py-2 flex flex-col lg:flex-row relative"
        },
    )[0:10]


def get_fixtures():
    fixtures = get_fixtures_and_odds()

    results = {}
    goal_stats = get_stats()
    top_scorers = get_top_scorers()
    top_booked = get_top_booked()
    form = get_form()
    performance = get_relative_performance()

    for fixture in fixtures:
        teams = fixture.find_next("a").text.strip().split(" - ")
        home_team = teams[0]
        away_team = teams[1]
        fixture_text = home_team + " vs " + away_team
        odds = fixture.find_all(
            "span",
            {
                "class": "px-1 h-booklogosm font-bold bg-primary-yellow text-white leading-8 rounded-r-md w-14 md:w-18 flex justify-center items-center text-base"
            },
        )
        try:
            home_wins_odds = float(odds[0].text)
            home_team_wins_odds_as_percent = (1 / home_wins_odds) * 100

            draw_odds = float(odds[1].text)
            draw_odds_as_percent = (1 / draw_odds) * 100

            away_wins_odds = float(odds[2].text)
            away_team_wins_odds_as_percent = (1 / away_wins_odds) * 100
            broker_profit = (
                home_team_wins_odds_as_percent
                + draw_odds_as_percent
                + away_team_wins_odds_as_percent
            ) - 100
            broker_home_team_wins_odds_as_percent = round(home_team_wins_odds_as_percent - (
                broker_profit / 3
            ),1)
            broker_draw_odds_as_percent = round(draw_odds_as_percent - (broker_profit / 3),1)
            broker_away_team_wins_odds_as_percent = round(away_team_wins_odds_as_percent - (
                broker_profit / 3
            ), 1 )
            home_team_form_score = calculate_form_score(form.get(home_team), performance.get(home_team))
            away_team_form_score = calculate_form_score(form.get(away_team), performance.get(away_team))
            home_team_wins_odds_as_percent, away_team_wins_odds_as_percent = calculate_win_probability(home_team_form_score, away_team_form_score, broker_home_team_wins_odds_as_percent, broker_away_team_wins_odds_as_percent)


            # home_team_wins_odds_as_percent, away_team_wins_odds_as_percent, draw_odds_as_percent = calculate_probabilities_with_performance_index(form.get(home_team), form.get(away_team), performance.get(home_team), performance.get(away_team))
        except:
            home_team_wins_odds_as_percent = None
            draw_odds_as_percent = None
            away_team_wins_odds_as_percent = None

        try:
            average_home_score, average_away_score = simulate_multiple_matches(
                    goal_stats,
                    home_team,
                    away_team,
                    home_team_wins_odds_as_percent,
                    away_team_wins_odds_as_percent,
                )
            results[fixture_text]= {
                "home_team": home_team,
                "away_team": away_team,
                "home_team_form": form.get(home_team),
                "broker_home_win_percentage": broker_home_team_wins_odds_as_percent,
                "home_goals": round(average_home_score),
                "likely_home_scorers": top_scorers[top_scorers['team_name'] == home_team].to_dict(orient='records'),
                "likely_home_booked": top_booked[top_booked['team_name'] == home_team].to_dict(orient='records'),
                "broker_draw_percentage": broker_draw_odds_as_percent,
                "broker_away_win_percentage": broker_away_team_wins_odds_as_percent,
                "away_goals": round(average_away_score),
                "away_team_form": form.get(away_team),
                "likely_away_scorers": top_scorers[top_scorers['team_name'] == away_team].to_dict(orient='records'),
                "likely_away_booked": top_booked[top_booked['team_name'] == away_team].to_dict(orient='records'),
                "avg_goals_home": goal_stats[home_team]["home"]["goals_for"][-1],
                "avg_goals_away": goal_stats[away_team]["away"]["goals_for"][-1],
            }
        except Exception as e:
            print(e.with_traceback())
            results[fixture_text]= {
                "home_team": home_team,
                "away_team": away_team,
                "home_win_percentage": "N/A",
                "draw_percentage": "N/A",
                "away_win_percentage": "N/A",
                "home_goals": "N/A",
                "away_goals": "N/A",
                "likely_home_scorers": [],
                "likely_away_scorers": [],
                "broker_home_win_percentage":"N/A",
                "broker_away_win_percentage":"N/A",
                "broker_draw_percentage":"N/A",
            }

    return results