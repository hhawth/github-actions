from scipy.stats import mode, skewnorm
import numpy as np

def new_func_skewnorm_with_prob(occurrences_of_scores, probability):
    scores = np.array([4, 3, 2, 1, 0])  # Possible scores
    occurrences = np.array(occurrences_of_scores[:5])  # First 5 occurrences

    # Calculate base mean
    total_occurrences = np.sum(occurrences)

    base_mean = occurrences_of_scores[-1]

    # Calculate standard deviation
    variance = np.sum(occurrences * (scores - base_mean) ** 2) / total_occurrences
    std_dev = np.sqrt(variance) if total_occurrences > 0 else 1

    # Cap probability between 0% and 100%
    probability = min(probability, 100)

    # Adjust skewness: neutral at 50%, scale it and cap at -3 to +3
    skew = (probability - 50) / 16.67  # Adjust the scaling for a 50% center
    skew = max(skew, -3)  # Cap skew at a minimum of -3
    skew = min(skew, 3)   # Cap skew at a maximum of 3

    # Generate skewed normally-distributed goals
    random_goals = skewnorm.rvs(skew, loc=base_mean, scale=std_dev, size=1000)
    random_goals = np.round(random_goals)  # Round to nearest whole number

    # Return the mode of generated goals
    return mode(random_goals).mode  # Updated to return the first mode value


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

def calculate_form_score(ppg, relative_performance, weight_ppg=1, weight_relative=0.2):
    """
    Calculate the form score for a team based on average points per game (PPG) and raw relative performance.
    """
    # Calculate form score using weights
    form_score = (ppg * weight_ppg) + (relative_performance * weight_relative)
    return form_score

def calculate_win_probability(team_a_score, team_b_score):
    """
    Calculate win probabilities for both teams based on their form scores.
    """
    # Calculate total score
    total_score = team_a_score + team_b_score
    
    # Calculate win probabilities for each team
    team_a_win_prob = team_a_score / total_score
    team_b_win_prob = team_b_score / total_score
    
    return team_a_win_prob * 100, team_b_win_prob * 100