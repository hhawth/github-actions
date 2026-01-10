from scipy.stats import mode, skewnorm
import numpy as np

def generate_goals_with_skewed_distribution(occurrences_of_scores, probability, skew_factor=1.5):
    """
    Generate goals using skewed normal distribution based on historical data and probability.
    
    Args:
        occurrences_of_scores: List of score occurrences with base mean as last element
        probability: Win probability percentage (0-100)
        skew_factor: Factor to amplify skewness (default: 1.5)
    
    Returns:
        int: Most likely number of goals (mode of distribution)
    """
    scores = np.array([4, 3, 2, 1, 0])  # Possible scores
    occurrences = np.array(occurrences_of_scores[:5])  # First 5 occurrences

    # Calculate base mean
    total_occurrences = np.sum(occurrences)
    base_mean = occurrences_of_scores[-1]

    # Calculate standard deviation
    variance = np.sum(occurrences * (scores - base_mean) ** 2) / total_occurrences
    std_dev = np.sqrt(variance) if total_occurrences > 0 else 1

    # Cap probability between 0% and 100%
    probability = np.clip(probability, 0, 100)

    # Adjust skewness: neutral at 50%, scale it, cap at -4 to +4, then apply skew_factor
    skew = ((probability - 50) / 12.5) * skew_factor
    skew = np.clip(skew, -4, 4)  # Cap skew between -4 and 4

    # Generate skewed normally-distributed goals
    random_goals = skewnorm.rvs(skew, loc=base_mean, scale=std_dev, size=10000)
    random_goals = np.round(random_goals)  # Round to nearest whole number

    # Return the mode of generated goals
    return int(mode(random_goals).mode) 



def simulate_match(
    goal_stats,
    home_team,
    away_team,
    home_team_wins_odds_as_percent,
    away_team_wins_odds_as_percent,
):
    """
    Simulate a single match between two teams based on their goal statistics and win probabilities.
    
    Returns:
        tuple: (home_team_goals, away_team_goals)
    """
    home_team_estimated_goals_scored = generate_goals_with_skewed_distribution(
        goal_stats[home_team]["home"]["goals_for"], home_team_wins_odds_as_percent
    )
    away_team_estimated_goals_conceded = generate_goals_with_skewed_distribution(
        goal_stats[away_team]["away"]["goals_against"],
        home_team_wins_odds_as_percent,
    )

    # Calculate rough estimate for home team
    rough_estimate_home_team_scored = (
        home_team_estimated_goals_scored + away_team_estimated_goals_conceded
    ) / 2

    # Calculate expected goals scored by the away team
    away_team_estimated_goals_scored = generate_goals_with_skewed_distribution(
        goal_stats[away_team]["away"]["goals_for"], away_team_wins_odds_as_percent
    )
    home_team_estimated_goals_conceded = generate_goals_with_skewed_distribution(
        goal_stats[home_team]["home"]["goals_against"],
        away_team_wins_odds_as_percent,
    )
    # Calculate rough estimate for away team
    rough_estimate_away_team_scored = (
        away_team_estimated_goals_scored + home_team_estimated_goals_conceded
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
    """
    Simulate multiple matches and return the most common result.
    
    Args:
        goal_stats: Dictionary containing team goal statistics
        home_team: Name of the home team
        away_team: Name of the away team
        home_team_wins_odds_as_percent: Home team win probability (0-100)
        away_team_wins_odds_as_percent: Away team win probability (0-100)
        simulations: Number of simulations to run (default: 10)
    
    Returns:
        tuple: Most common match result (home_score, away_score)
    """
    # Store results of each simulation
    match_results = [
        simulate_match(
            goal_stats,
            home_team,
            away_team,
            home_team_wins_odds_as_percent,
            away_team_wins_odds_as_percent
        )
        for _ in range(simulations)
    ]

    # Find the mode of the results
    return mode(match_results).mode


def calculate_form_score(ppg, relative_performance, weight_ppg=1, weight_relative=0.2):
    """
    Calculate the form score for a team based on average points per game (PPG) and relative performance.
    
    Args:
        ppg: Average points per game
        relative_performance: Raw relative performance metric
        weight_ppg: Weight for PPG in calculation (default: 1)
        weight_relative: Weight for relative performance (default: 0.2)
    
    Returns:
        float: Weighted form score
    """
    return (ppg * weight_ppg) + (relative_performance * weight_relative)


def calculate_win_probability(team_a_score, team_b_score, bookmaker_a_prob, bookmaker_b_prob, bookmaker_weight=0.75):
    """
    Calculate win probabilities for both teams by blending form scores with bookmaker probabilities.
    
    Args:
        team_a_score: Form score for team A
        team_b_score: Form score for team B
        bookmaker_a_prob: Bookmaker probability for team A (percentage)
        bookmaker_b_prob: Bookmaker probability for team B (percentage)
        bookmaker_weight: Weight for bookmaker probabilities (default: 0.75)
    
    Returns:
        tuple: (team_a_win_probability, team_b_win_probability) as percentages
    """
    # Normalize team scores to a 0-1 range (assuming max score is 3)
    normalized_team_a_score = team_a_score / 3
    normalized_team_b_score = team_b_score / 3

    # Calculate form-based win probabilities
    total_score = normalized_team_a_score + normalized_team_b_score
    form_team_a_win_prob = normalized_team_a_score / total_score
    form_team_b_win_prob = normalized_team_b_score / total_score

    # Convert bookmaker probabilities from percentages to 0-1 scale
    bookmaker_a_prob_normalized = bookmaker_a_prob / 100
    bookmaker_b_prob_normalized = bookmaker_b_prob / 100

    # Blend form-based probabilities with bookmaker probabilities
    team_a_win_prob = (bookmaker_weight * bookmaker_a_prob_normalized) + ((1 - bookmaker_weight) * form_team_a_win_prob)
    team_b_win_prob = (bookmaker_weight * bookmaker_b_prob_normalized) + ((1 - bookmaker_weight) * form_team_b_win_prob)

    # Normalize to ensure they add up to 1
    normalization_factor = team_a_win_prob + team_b_win_prob
    team_a_win_prob /= normalization_factor
    team_b_win_prob /= normalization_factor
    
    return team_a_win_prob * 100, team_b_win_prob * 100