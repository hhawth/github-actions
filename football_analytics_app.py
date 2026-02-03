import streamlit as st
import json
from datetime import datetime
import sys

# Add current directory to path to import our betting system
sys.path.append(".")
from working_betting_system import (
    convert_fractional_to_decimal,
    AccumulatorBettingModel,
)
from football_prediction_model import FootballPredictionModel
from best_match_outcomes import BestMatchOutcomeSelector

import time
from functools import lru_cache


def ttl_cache(ttl_seconds):
    def decorator(fn):
        @lru_cache(maxsize=None)
        def cached_fn(*args, _ttl_marker=None, **kwargs):
            return fn(*args, **kwargs)

        def wrapper(*args, **kwargs):
            ttl_marker = int(time.time() / ttl_seconds)
            return cached_fn(*args, _ttl_marker=ttl_marker, **kwargs)

        return wrapper

    return decorator


# Configure the page
st.set_page_config(
    page_title="Football Analytics & Betting", page_icon="⚽", layout="wide"
)


# Load data functions
@ttl_cache(ttl_seconds=300)
def load_merged_data():
    """Load merged match data"""
    try:
        with open("merged_match_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("merged_match_data.json not found. Please run the data merger first.")
        return []


@ttl_cache(ttl_seconds=300)
def load_predictions():
    """Load match predictions"""
    try:
        with open("match_predictions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("match_predictions.json not found. Only showing basic match data.")
        return []


@ttl_cache(ttl_seconds=300)
def load_best_match_outcomes():
    """Load best match outcomes data"""
    try:
        selector = BestMatchOutcomeSelector()
        ranked_outcomes = selector.select_best_outcomes_per_match()
        return ranked_outcomes
    except Exception as e:
        st.error(f"Error loading best outcomes: {e}")
        return []


@ttl_cache(ttl_seconds=300)
def load_match_data():
    from soccerway_scraper import main as soccerway_main

    soccerway_main()
    from soccerstats_scraper import main as soccerstats_main

    soccerstats_main()
    from match_data_merger import main as merger_main

    merger_main()
    from football_prediction_model import main as prediction_main

    prediction_main()
    from best_match_outcomes import main as best_outcomes_main

    best_outcomes_main()


def main():
    """Main Streamlit app"""

    # App title
    st.title("⚽ Football Analytics & Betting Platform")

    load_match_data()

    st.markdown("---")

    # Create tabs
    tab1, tab2 = st.tabs(["🏆 Match Predictions", "💰 Betting Accumulators"])

    with tab1:
        display_match_predictions()

    # with tab2:
    #     display_betting_accumulators()

    # Footer
    st.markdown("---")
    st.markdown("*Data updated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*")


def generate_betting_analysis():
    """Generate betting analysis using our system"""
    try:
        merged_data = load_merged_data()
        if not merged_data:
            return None

        betting_model = AccumulatorBettingModel(min_probability=0.25, max_odds=6.0)
        selections = []

        for match in merged_data:
            if not isinstance(match, dict):
                continue

            home_team = match.get("home_team", "Unknown")
            away_team = match.get("away_team", "Unknown")
            league = match.get("league", "Unknown")

            # Convert odds
            odds_1_decimal = convert_fractional_to_decimal(match.get("odds_1"))
            odds_x_decimal = convert_fractional_to_decimal(match.get("odds_x"))
            odds_2_decimal = convert_fractional_to_decimal(match.get("odds_2"))

            # Create selections for each outcome
            outcomes = [
                ("home_win", odds_1_decimal, "1"),
                ("draw", odds_x_decimal, "X"),
                ("away_win", odds_2_decimal, "2"),
            ]

            for outcome_name, decimal_odds, symbol in outcomes:
                if decimal_odds and decimal_odds > 1.0:
                    implied_prob = 1.0 / decimal_odds

                    selection = {
                        "outcome": outcome_name,
                        "probability": implied_prob,
                        "odds": decimal_odds,
                        "confidence": 0.4,
                        "source": "odds_only",
                        "expected_value": (implied_prob * decimal_odds) - 1,
                        "risk_score": betting_model._calculate_risk_score(
                            implied_prob, 0.4, decimal_odds
                        ),
                        "match_info": {
                            "home_team": home_team,
                            "away_team": away_team,
                            "league": league,
                            "time": match.get("time", "TBD"),
                            "date": match.get("date", "TBD"),
                        },
                        "selection_symbol": symbol,
                    }

                    if (
                        selection["probability"] >= betting_model.min_probability
                        and selection["odds"] <= betting_model.max_odds
                    ):
                        selections.append(selection)

        # Add confidence score for ranking
        for selection in selections:
            selection["confidence_score"] = selection["confidence"]

        # Rank selections and generate accumulators
        ranked_selections = betting_model.rank_selections(selections)
        accumulators = betting_model.generate_multiple_accumulators(
            ranked_selections, max_fold=6
        )

        return {
            "total_matches": len(merged_data),
            "total_selections": len(selections),
            "ranked_selections": ranked_selections,
            "accumulators": accumulators,
            "best_single_bet": ranked_selections[0] if ranked_selections else None,
            "best_accumulator": accumulators[0] if accumulators else None,
        }
    except Exception as e:
        st.error(f"Error generating betting analysis: {e}")
        return None


def generate_live_prediction(match_data):
    """Generate prediction for a single match using the football prediction model"""
    try:
        prediction_model = FootballPredictionModel()
        prediction = prediction_model.ensemble_prediction(match_data)

        # Extract expected goals from statistical prediction
        if "statistical" in prediction.get("predictions", {}):
            stat_pred = prediction["predictions"]["statistical"]

            # Calculate expected goals from the model
            home_stats = match_data.get("home_team_stats", {})
            away_stats = match_data.get("away_team_stats", {})

            if home_stats and away_stats:
                home_xg, away_xg = prediction_model.calculate_expected_goals(
                    home_stats, away_stats
                )

                # Add score prediction and xG analysis to the prediction
                prediction["score_prediction"] = {
                    "predicted_home_goals": home_xg,
                    "predicted_away_goals": away_xg,
                }

                prediction["xg_analysis"] = {"home_xg": home_xg, "away_xg": away_xg}

                # Add most likely score from probabilities
                if "score_probabilities" in stat_pred:
                    sorted_scores = sorted(
                        stat_pred["score_probabilities"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    if sorted_scores:
                        most_likely_score = sorted_scores[0][0]  # e.g., "2-1"
                        home_score, away_score = most_likely_score.split("-")
                        prediction["most_likely_score"] = {
                            "home_goals": int(home_score),
                            "away_goals": int(away_score),
                            "probability": sorted_scores[0][1],
                        }

        return prediction
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return None


def format_team_stats_table(stats, team_name):
    """Format team statistics as a nice table display"""
    if not stats or stats == "N/A":
        st.write("No stats available")
        return

    if isinstance(stats, dict):
        # Key statistics to highlight
        key_stats = {
            "Games Played": stats.get("gp", "N/A"),
            "Win %": stats.get("w%", "N/A"),
            "Points/Game": stats.get("points_per_game", "N/A"),
            "Goals For": stats.get("gf", "N/A"),
            "Goals Against": stats.get("ga", "N/A"),
            "Total Goals": stats.get("tg", "N/A"),
            "Clean Sheets %": stats.get("cs", "N/A"),
            "Both Teams Score %": stats.get("bts", "N/A"),
            "Over 2.5 Goals %": stats.get("2.5+", "N/A"),
        }

        # Create two columns for stats
        col1, col2 = st.columns(2)

        stats_items = list(key_stats.items())
        mid_point = len(stats_items) // 2

        with col1:
            for key, value in stats_items[:mid_point]:
                if value and value != "N/A":
                    st.metric(key, value)

        with col2:
            for key, value in stats_items[mid_point:]:
                if value and value != "N/A":
                    st.metric(key, value)
    else:
        st.write("No structured stats available")


def display_match_predictions():
    """Display matches with predictions and stats"""
    st.header("⚽ Match Predictions & Statistics")

    merged_data = load_merged_data()
    predictions = load_predictions()

    # Create predictions lookup
    predictions_dict = {}
    for pred in predictions:
        if isinstance(pred, dict):
            key = f"{pred.get('home_team', '')} vs {pred.get('away_team', '')}"
            predictions_dict[key] = pred

    if not merged_data:
        st.error("No match data available")
        return

    # Filter controls
    col1, col2 = st.columns(2)

    with col1:
        # Create combined region-league options
        league_combinations = []
        for match in merged_data:
            if isinstance(match, dict):
                region = match.get("region", "Unknown")
                league = match.get("league", "Unknown")
                combination = f"{region} - {league}"
                if combination not in league_combinations:
                    league_combinations.append(combination)

        league_combinations.sort()
        selected_league_combinations = st.multiselect(
            "Filter by Region - League",
            league_combinations,
            default=league_combinations,  # Show ALL leagues by default
        )

    # Filter matches
    filtered_matches = []
    for match in merged_data:
        if not isinstance(match, dict):
            continue

        league = match.get("league", "Unknown")
        region = match.get("region", "Unknown")
        combination = f"{region} - {league}"

        if combination not in selected_league_combinations:
            continue

        match_key = f"{match.get('home_team', '')} vs {match.get('away_team', '')}"

        filtered_matches.append(match)

    st.write(f"Showing {len(filtered_matches)} matches")

    # Group matches by time segments
    def get_time_segment(time_str):
        """Convert time string to time segment"""
        if not time_str or time_str == "TBD":
            return "⏰ Time TBD"
        if time_str == "Canc" or time_str == "Postp":
            return "❌ Cancelled"
        if time_str == "Full-time":
            return "✅ Finished"

        try:
            # Handle different time formats
            time_str_stripped = time_str.strip()
            if ":" in time_str_stripped:
                return time_str
            else:
                return "Live Match"
        except (ValueError, TypeError):
            return "⏰ Time TBD"

    # Group matches by time
    time_groups = {}
    for match in filtered_matches:
        time_segment = get_time_segment(match.get("time"))
        if time_segment not in time_groups:
            time_groups[time_segment] = []
        time_groups[time_segment].append(match)

    time_groups = dict(sorted(time_groups.items(), key=lambda x: x[0]))
    for time_key in time_groups.keys():
        matches_in_segment = time_groups[time_key]

        # Sort matches within segment by actual time
        matches_in_segment.sort(key=lambda m: m.get("time", "ZZ:ZZ"))

        st.subheader(f"{time_key} ({len(matches_in_segment)} matches)")

        # Display matches in this time segment
        for match in matches_in_segment:
            home_team = match.get("home_team", "Unknown")
            away_team = match.get("away_team", "Unknown")
            league = match.get("league", "Unknown")
            region = match.get("region", "Unknown")
            time = match.get("time", "TBD")

            match_key = f"{home_team} vs {away_team}"
            prediction = predictions_dict.get(match_key)

            # Create expandable match card
            with st.expander(f"🏆 {home_team} vs {away_team} ({league}) - {time}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Match Info")
                    st.write(f"**League:** {league}")
                    st.write(f"**Region:** {region}")
                    st.write(f"**Time:** {time}")

                    # Show scores if available
                    home_score = match.get("home_score")
                    away_score = match.get("away_score")
                    if home_score is not None and away_score is not None:
                        st.write(f"**Score:** {home_score} - {away_score}")

                    # Show odds
                    odds_1 = match.get("odds_1")
                    odds_x = match.get("odds_x")
                    odds_2 = match.get("odds_2")

                    if odds_1 or odds_x or odds_2:
                        st.write("**Odds:**")
                        if odds_1:
                            st.write(f"Home: {odds_1}")
                        if odds_x:
                            st.write(f"Draw: {odds_x}")
                        if odds_2:
                            st.write(f"Away: {odds_2}")

                with col2:
                    st.subheader("Team Statistics")

                    home_stats = match.get("home_team_stats", {})
                    away_stats = match.get("away_team_stats", {})

                    # Home team stats
                    st.write(f"**🏠 {home_team}**")
                    if home_stats and home_stats != "N/A":
                        format_team_stats_table(home_stats, home_team)
                    else:
                        st.write("No stats available")

                    st.markdown("---")

                    # Away team stats
                    st.write(f"**✈️ {away_team}**")
                    if away_stats and away_stats != "N/A":
                        format_team_stats_table(away_stats, away_team)
                    else:
                        st.write("No stats available")

                with col3:
                    st.subheader("Predictions")

                    match_key = f"{home_team} vs {away_team}"
                    prediction = predictions_dict.get(match_key)

                    # If no prediction exists but we have team stats, generate one
                    if (
                        not prediction
                        and match.get("home_team_stats")
                        and match.get("away_team_stats")
                    ):
                        with st.spinner("Generating live prediction..."):
                            prediction = generate_live_prediction(match)
                            if prediction:
                                st.success("✨ Live prediction generated!")

                    if prediction:
                        # Show score prediction prominently at top
                        if "score_prediction" in prediction:
                            score_pred = prediction["score_prediction"]
                            pred_home = score_pred.get("predicted_home_goals", "N/A")
                            pred_away = score_pred.get("predicted_away_goals", "N/A")
                            if pred_home != "N/A" and pred_away != "N/A":
                                st.success(
                                    f"⚽ **Predicted Score: {pred_home:.1f} - {pred_away:.1f}**"
                                )

                        # Show most likely exact score
                        if "most_likely_score" in prediction:
                            likely_score = prediction["most_likely_score"]
                            home_goals = likely_score["home_goals"]
                            away_goals = likely_score["away_goals"]
                            probability = likely_score["probability"]
                            st.info(
                                f"🎯 **Most Likely Score: {home_goals}-{away_goals}** ({probability:.1%} chance)"
                            )

                        # Show expected goals prominently
                        if "xg_analysis" in prediction:
                            xg = prediction["xg_analysis"]
                            home_xg = xg.get("home_xg", "N/A")
                            away_xg = xg.get("away_xg", "N/A")
                            if isinstance(home_xg, (int, float)) and isinstance(
                                away_xg, (int, float)
                            ):
                                st.info(
                                    f"📊 **Expected Goals: {home_xg:.2f} - {away_xg:.2f}**"
                                )

                        # Show prediction results
                        preds = prediction.get("predictions", {})

                        if "ensemble" in preds:
                            ensemble = preds["ensemble"]
                            st.write("**📊 Outcome Probabilities:**")

                            outcomes = ensemble.get("outcome_probabilities", {})

                            # Use columns for probabilities
                            prob_col1, prob_col2, prob_col3 = st.columns(3)

                            with prob_col1:
                                if "home_win" in outcomes:
                                    st.metric(
                                        "🏠 Home Win", f"{outcomes['home_win']:.1%}"
                                    )

                            with prob_col2:
                                if "draw" in outcomes:
                                    st.metric("🤝 Draw", f"{outcomes['draw']:.1%}")

                            with prob_col3:
                                if "away_win" in outcomes:
                                    st.metric(
                                        "✈️ Away Win", f"{outcomes['away_win']:.1%}"
                                    )

                            confidence = ensemble.get("confidence_score", 0)
                            if confidence > 0:
                                st.write(f"**🎯 Confidence:** {confidence:.1%}")

                        elif "statistical" in preds:
                            stats_pred = preds["statistical"]
                            st.write("**📈 Statistical Prediction:**")

                            outcomes = stats_pred.get("outcome_probabilities", {})
                            prob_col1, prob_col2, prob_col3 = st.columns(3)

                            with prob_col1:
                                if "home_win" in outcomes:
                                    st.metric(
                                        "🏠 Home Win", f"{outcomes['home_win']:.1%}"
                                    )
                            with prob_col2:
                                if "draw" in outcomes:
                                    st.metric("🤝 Draw", f"{outcomes['draw']:.1%}")
                            with prob_col3:
                                if "away_win" in outcomes:
                                    st.metric(
                                        "✈️ Away Win", f"{outcomes['away_win']:.1%}"
                                    )

                        elif "odds_based" in preds:
                            odds_pred = preds["odds_based"]
                            st.write("**💰 Odds-based Prediction:**")

                            outcomes = odds_pred.get("outcome_probabilities", {})
                            prob_col1, prob_col2, prob_col3 = st.columns(3)

                            with prob_col1:
                                if "home_win" in outcomes:
                                    st.metric(
                                        "🏠 Home Win", f"{outcomes['home_win']:.1%}"
                                    )
                            with prob_col2:
                                if "draw" in outcomes:
                                    st.metric("🤝 Draw", f"{outcomes['draw']:.1%}")
                            with prob_col3:
                                if "away_win" in outcomes:
                                    st.metric(
                                        "✈️ Away Win", f"{outcomes['away_win']:.1%}"
                                    )

                    else:
                        if not match.get("home_team_stats") or not match.get(
                            "away_team_stats"
                        ):
                            st.write("❌ No team statistics available for prediction")
                        else:
                            st.write("🔄 Click to generate prediction")


if __name__ == "__main__":
    main()
