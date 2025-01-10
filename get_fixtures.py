import logging
from stat_getter import (
    get_stats,
    get_top_booked,
    get_top_scorers,
    get_form,
    get_relative_performance,
)
from calculations import (
    calculate_form_score,
    calculate_win_probability,
    simulate_multiple_matches,
)
from selenium.webdriver.chrome.options import Options
import cachetools
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


def cached_function(ttl=3600, maxsize=100):
    """Decorator to create a unique cache for each function."""

    def decorator(func):
        cache = cachetools.TTLCache(
            maxsize=maxsize, ttl=ttl
        )  # Create a unique cache for each function

        @cachetools.cached(cache)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Random User-Agent


# Create a cache with a time-to-live (TTL) of 1 hour (3600 seconds)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

FIXTURES_LAST_CALL = 0

FIXTURES = None


# Set up the browser


@cached_function(maxsize=100, ttl=3600)
def get_fixtures_and_odds():
    LOGGER.info("Getting Sky Sports odds")

    # Set up Chrome options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")


    # Start the WebDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    try:
        # Open the website
        driver.get("https://www.ukclubsport.com/football/premier-league/")

        # Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "time-table__match"))
        )

        # Get page content
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find fixtures
        fixtures = soup.find_all("div", class_="time-table__match")

        # Extract relevant data
        results = []
        for fixture in fixtures[:10]:  # Limit to top 10 fixtures
            match_info = fixture.get_text(strip=True)
            results.append(match_info)

        LOGGER.info(f"Found {len(results)} fixtures")
        return results

    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")
        return []

    finally:
        driver.quit()


def get_fixtures():
    fixtures = get_fixtures_and_odds()
    results = {}
    goal_stats = get_stats()
    top_scorers = get_top_scorers()
    top_booked = get_top_booked()
    form = get_form()
    performance = get_relative_performance()

    LOGGER.info(f"Length of fixutes: {len(fixtures)}")

    for fixture in fixtures:
        teams = fixture.find_all("div", {"time-table__team-title"})
        home_team = teams[0].text
        away_team = teams[1].text
        fixture_text = home_team + " vs " + away_team
        LOGGER.info(fixture_text)
        odds = fixture.find_all(
            "div",
            {"class": "ratio__wrap"},
        )
        try:
            home_wins_odds = float(odds[0].text.split()[1])
            home_team_wins_odds_as_percent = (1 / home_wins_odds) * 100

            draw_odds = float(odds[1].text.split()[1])
            draw_odds_as_percent = (1 / draw_odds) * 100

            away_wins_odds = float(odds[2].text.split()[1])
            away_team_wins_odds_as_percent = (1 / away_wins_odds) * 100
            broker_profit = (
                home_team_wins_odds_as_percent
                + draw_odds_as_percent
                + away_team_wins_odds_as_percent
            ) - 100
            broker_home_team_wins_odds_as_percent = round(
                home_team_wins_odds_as_percent - (broker_profit / 3), 1
            )
            broker_draw_odds_as_percent = round(
                draw_odds_as_percent - (broker_profit / 3), 1
            )
            broker_away_team_wins_odds_as_percent = round(
                away_team_wins_odds_as_percent - (broker_profit / 3), 1
            )
            home_team_form_score = calculate_form_score(
                form.get(home_team), performance.get(home_team)
            )
            away_team_form_score = calculate_form_score(
                form.get(away_team), performance.get(away_team)
            )
            home_team_wins_odds_as_percent, away_team_wins_odds_as_percent = (
                calculate_win_probability(
                    home_team_form_score,
                    away_team_form_score,
                    broker_home_team_wins_odds_as_percent,
                    broker_away_team_wins_odds_as_percent,
                )
            )

            # home_team_wins_odds_as_percent, away_team_wins_odds_as_percent, draw_odds_as_percent = calculate_probabilities_with_performance_index(form.get(home_team), form.get(away_team), performance.get(home_team), performance.get(away_team))
        except Exception:
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
            results[fixture_text] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_team_form": form.get(home_team),
                "broker_home_win_percentage": broker_home_team_wins_odds_as_percent,
                "home_goals": round(average_home_score),
                "likely_home_scorers": top_scorers[
                    top_scorers["team_name"] == home_team
                ].to_dict(orient="records"),
                "likely_home_booked": top_booked[
                    top_booked["team_name"] == home_team
                ].to_dict(orient="records"),
                "broker_draw_percentage": broker_draw_odds_as_percent,
                "broker_away_win_percentage": broker_away_team_wins_odds_as_percent,
                "away_goals": round(average_away_score),
                "away_team_form": form.get(away_team),
                "likely_away_scorers": top_scorers[
                    top_scorers["team_name"] == away_team
                ].to_dict(orient="records"),
                "likely_away_booked": top_booked[
                    top_booked["team_name"] == away_team
                ].to_dict(orient="records"),
                "avg_goals_home": goal_stats[home_team]["home"]["goals_for"][-1],
                "avg_goals_away": goal_stats[away_team]["away"]["goals_for"][-1],
            }
        except Exception:
            results[fixture_text] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_win_percentage": "N/A",
                "draw_percentage": "N/A",
                "away_win_percentage": "N/A",
                "home_goals": "N/A",
                "away_goals": "N/A",
                "likely_home_scorers": [],
                "likely_away_scorers": [],
                "broker_home_win_percentage": "N/A",
                "broker_away_win_percentage": "N/A",
                "broker_draw_percentage": "N/A",
            }

    return results


get_fixtures_and_odds()
