from scipy.stats import mode, skewnorm
import numpy as np

def new_func_skewnorm_with_prob(occurrences_of_scores, probability, skew_factor=1.5):
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

    # Adjust skewness: neutral at 50%, scale it, cap at -4 to +4, then apply skew_factor
    skew = ((probability - 50) / 12.5) * skew_factor  # Adjusted scaling for -4 to +4 range and skew factor
    skew = max(skew, -4)  # Cap skew at a minimum of -4
    skew = min(skew, 4)   # Cap skew at a maximum of 4

    # Generate skewed normally-distributed goals
    random_goals = skewnorm.rvs(skew, loc=base_mean, scale=std_dev, size=10000)
    random_goals = np.round(random_goals)  # Round to nearest whole number

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


def simulate_multiple_matches(
    goal_stats,
    home_team,
    away_team,
    home_team_wins_odds_as_percent,
    away_team_wins_odds_as_percent,
    simulations=10
):
    # List to store results of each simulation
    match_results = []

    for _ in range(simulations):
        # Run the simulation once
        home_score, away_score = simulate_match(
            goal_stats,
            home_team,
            away_team,
            home_team_wins_odds_as_percent,
            away_team_wins_odds_as_percent
        )
        # Append the result as a tuple (home_score, away_score)
        match_results.append((home_score, away_score))

    # Find the mode of the results
    most_common_result = mode(match_results).mode

    return most_common_result


def calculate_form_score(ppg, relative_performance, weight_ppg=1, weight_relative=0.2):
    """
    Calculate the form score for a team based on average points per game (PPG) and raw relative performance.
    """
    # Calculate form score using weights
    form_score = (ppg * weight_ppg) + (relative_performance * weight_relative)
    return form_score

# def calculate_win_probability(team_a_score, team_b_score):
#     """
#     Calculate win probabilities for both teams based on their form scores.
#     """
#     # Calculate total score
#     total_score = team_a_score + team_b_score
    
#     # Calculate win probabilities for each team
#     team_a_win_prob = team_a_score / total_score
#     team_b_win_prob = team_b_score / total_score
    
#     return team_a_win_prob * 100, team_b_win_prob * 100

def calculate_win_probability(team_a_score, team_b_score, bookmaker_a_prob, bookmaker_b_prob, bookmaker_weight=0.75):
    """
    Calculate win probabilities for both teams by blending form scores with bookmaker probabilities.
    """
    # Normalize team scores to a 0-1 range by dividing by 3 (the max score possible)
    normalized_team_a_score = team_a_score / 3
    normalized_team_b_score = team_b_score / 3

    # Calculate form-based win probabilities
    total_score = normalized_team_a_score + normalized_team_b_score
    form_team_a_win_prob = normalized_team_a_score / total_score
    form_team_b_win_prob = normalized_team_b_score / total_score

    # Convert bookmaker probabilities from percentages to 0-1 scale
    bookmaker_a_prob /= 100
    bookmaker_b_prob /= 100

    # Blend form-based probabilities with bookmaker probabilities
    team_a_win_prob = (bookmaker_weight * bookmaker_a_prob) + ((1 - bookmaker_weight) * form_team_a_win_prob)
    team_b_win_prob = (bookmaker_weight * bookmaker_b_prob) + ((1 - bookmaker_weight) * form_team_b_win_prob)

    # Normalize to ensure they add up to 1
    normalization_factor = team_a_win_prob + team_b_win_prob
    team_a_win_prob /= normalization_factor
    team_b_win_prob /= normalization_factor
    
    return team_a_win_prob * 100, team_b_win_prob * 100