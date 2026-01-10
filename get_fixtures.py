

import logging
from stat_getter import (
    get_stats, get_top_booked, get_top_scorers, 
    get_soccerstats_team_mapping, get_rankings_from_clubelo, 
    get_fixtures_from_clubelo
)
from calculations import simulate_multiple_matches

# Random User-Agent


# Create a cache with a time-to-live (TTL) of 1 hour (3600 seconds)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

FIXTURES_LAST_CALL = 0

FIXTURES = None


def get_fixtures():
    fixtures = get_fixtures_from_clubelo()
    rankings = get_rankings_from_clubelo()
    results = {}
    goal_stats = get_stats()
    top_scorers = get_top_scorers()
    top_booked = get_top_booked()

    for _, row in fixtures.iterrows():
        home_team = row["Home"]
        away_team = row["Away"]
        
        # Get team mapping for validation
        team_mapping = get_soccerstats_team_mapping()
        valid_teams = set(team_mapping.values())
        
        # Skip if teams aren't in our valid set
        if home_team not in valid_teams or away_team not in valid_teams:
            continue
            
        fixture_text = home_team + " vs " + away_team
        try:
            home_team_wins_odds_as_percent = row["Home Win"] * 100

            draw_odds_as_percent = row["Draw"] * 100

            away_team_wins_odds_as_percent = row["Away Win"] * 100

            likely_outcome = row["Likely Outcome"].strip("R:")
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
            # home_team_form_score = calculate_form_score(form.get(home_team), performance.get(home_team))
            # away_team_form_score = calculate_form_score(form.get(away_team), performance.get(away_team))
            # home_team_wins_odds_as_percent, away_team_wins_odds_as_percent = calculate_win_probability(home_team_form_score, away_team_form_score, broker_home_team_wins_odds_as_percent, broker_away_team_wins_odds_as_percent)


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
            results[fixture_text]= {
                "home_team": home_team,
                "away_team": away_team,
                "likely_outcome": likely_outcome,
                "broker_home_win_percentage": broker_home_team_wins_odds_as_percent,
                "home_goals": max(round(average_home_score),0),
                "likely_home_scorers": top_scorers[top_scorers['team_name'] == home_team].to_dict(orient='records'),
                "likely_home_booked": top_booked[top_booked['team_name'] == home_team].to_dict(orient='records'),
                "broker_draw_percentage": broker_draw_odds_as_percent,
                "broker_away_win_percentage": broker_away_team_wins_odds_as_percent,
                "away_goals": max(round(average_away_score),0),
                "likely_away_scorers": top_scorers[top_scorers['team_name'] == away_team].to_dict(orient='records'),
                "likely_away_booked": top_booked[top_booked['team_name'] == away_team].to_dict(orient='records'),
                "avg_goals_home": goal_stats[home_team]["home"]["goals_for"][-1],
                "avg_goals_away": goal_stats[away_team]["away"]["goals_for"][-1],
            }
        except Exception:
            results[fixture_text]= {
                "home_team": home_team,
                "away_team": away_team,
                "likely_outcome": "N/A",
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