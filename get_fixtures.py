import requests
from bs4 import BeautifulSoup

from scipy.stats import mode, skewnorm
import numpy as np
import logging
from stat_getter import get_stats, get_top_booked, get_top_scorers, cached_function, get_form

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

def new_func_skewnorm_with_prob(occurrences_of_scores, probability):
    scores = np.array([4, 3, 2, 1, 0])  # Possible scores
    occurrences = np.array(occurrences_of_scores[:5])  # First 5 occurrences

    # Calculate base mean
    total_occurrences = np.sum(occurrences)
    base_mean = (
        np.sum(scores * occurrences) / total_occurrences if total_occurrences > 0 else 1
    )

    # Calculate standard deviation
    variance = np.sum(occurrences * (scores - base_mean) ** 2) / total_occurrences
    std_dev = np.sqrt(variance) if total_occurrences > 0 else 1

    # Cap probability between 0% and 100%
    probability = min(probability, 100)

    # Adjust skewness: neutral at 33%, scale it and cap at -3 to +3
    skew = (probability - 33) / 11.17  # Maps probability from -3 to +3
    skew = max(skew, -3)  # Cap skew at a minimum of -3
    skew = min(skew, 3)   # Cap skew at a maximum of 3

    # Generate skewed normally-distributed goals
    random_goals = skewnorm.rvs(skew, loc=base_mean, scale=std_dev, size=1000)
    random_goals = np.round(random_goals)  # Round to nearest whole number
    random_goals = np.clip(random_goals, 0, 4)  # Cap values between 0 and 4

    # Return the mode of generated goals
    return mode(random_goals).mode


def simulate_match(
    goal_stats,
    home_team,
    away_team,
    home_team_wins_odds_as_percent,
    away_team_wins_odds_as_percent,
):
    home_team_estimates_goals_scored = new_func_skewnorm_with_prob(
        goal_stats[home_team]["home"]["goals_for"], home_team_wins_odds_as_percent
    )
    away_team_estimates_goals_conceded = new_func_skewnorm_with_prob(
        goal_stats[away_team]["away"]["goals_against"],
        home_team_wins_odds_as_percent,
    )

    # Calculate rough estimate for home team
    rough_estimate_home_team_scored = (
        home_team_estimates_goals_scored + away_team_estimates_goals_conceded
    ) / 2

    # Calculate expected goals scored by the away team
    away_team_estimates_goals_scored = new_func_skewnorm_with_prob(
        goal_stats[away_team]["away"]["goals_for"], away_team_wins_odds_as_percent
    )
    home_team_estimates_goals_conceded = new_func_skewnorm_with_prob(
        goal_stats[home_team]["home"]["goals_against"],
        away_team_wins_odds_as_percent,
    )

    # Calculate rough estimate for away team
    rough_estimate_away_team_scored = (
        away_team_estimates_goals_scored + home_team_estimates_goals_conceded
    ) / 2

    return round(rough_estimate_home_team_scored), round(rough_estimate_away_team_scored)

# Function to calculate winning probabilities based on form
def calculate_probabilities(arsenal_form, liverpool_form):
    # Calculate total form score
    total_form = arsenal_form + liverpool_form
    
    # Calculate draw probability as a function of form (this can be adjusted)
    draw_probability = (0.6 * min(arsenal_form, liverpool_form)) / total_form  # Adjust draw probability factor as needed
    
    # Calculate winning probabilities using relative form
    arsenal_win_probability = (arsenal_form / total_form) * (1 - draw_probability)
    liverpool_win_probability = (liverpool_form / total_form) * (1 - draw_probability)
    
    return arsenal_win_probability * 100, liverpool_win_probability* 100, draw_probability* 100

# Calculate probabilities
# probabilities = calculate_probabilities(arsenal_form, liverpool_form)


def get_fixtures():
    fixtures = get_fixtures_and_odds()

    results = {}
    goal_stats = get_stats()
    top_scorers = get_top_scorers()
    top_booked = get_top_booked()
    form = get_form()

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

            home_team_wins_odds_as_percent, away_team_wins_odds_as_percent, draw_odds_as_percent = calculate_probabilities(form.get(home_team), form.get(away_team))
        except:
            home_team_wins_odds_as_percent = None
            draw_odds_as_percent = None
            away_team_wins_odds_as_percent = None

        try:
            average_home_score, average_away_score = simulate_match(
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
                "home_win_percentage": round(home_team_wins_odds_as_percent, 1),
                "broker_home_win_percentage": broker_home_team_wins_odds_as_percent,
                "home_goals": round(average_home_score),
                "likely_home_scorers": top_scorers[top_scorers['team_name'] == home_team].to_dict(orient='records'),
                "likely_home_booked": top_booked[top_booked['team_name'] == home_team].to_dict(orient='records'),
                "draw_percentage": round(draw_odds_as_percent, 1),
                "broker_draw_percentage": broker_draw_odds_as_percent,
                "away_win_percentage": round(away_team_wins_odds_as_percent, 1),
                "broker_away_win_percentage": broker_away_team_wins_odds_as_percent,
                "away_goals": round(average_away_score),
                "away_team_form": form.get(away_team),
                "likely_away_scorers": top_scorers[top_scorers['team_name'] == away_team].to_dict(orient='records'),
                "likely_away_booked": top_booked[top_booked['team_name'] == away_team].to_dict(orient='records'),
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