import json
from typing import Dict, List, Tuple
from scipy.stats import poisson
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballPredictionModel:
    """
    Advanced football match prediction model using team statistics and betting odds
    
    Features:
    - Expected Goals (xG) calculation based on team stats
    - Poisson distribution for score prediction
    - Betting odds integration for market confidence
    - Multiple prediction methods with ensemble approach
    """
    
    def __init__(self):
        self.home_advantage = 0.3  # Home advantage factor
        self.model_weights = {
            'statistical': 0.6,    # Weight for statistical predictions
            'odds_based': 0.4      # Weight for odds-based predictions
        }
    
    def parse_percentage(self, percentage_str: str) -> float:
        """Convert percentage string to float (e.g., '50%' -> 0.5)"""
        try:
            if isinstance(percentage_str, str) and '%' in percentage_str:
                return float(percentage_str.replace('%', '')) / 100
            elif isinstance(percentage_str, (int, float)):
                return float(percentage_str) / 100 if percentage_str > 1 else float(percentage_str)
            else:
                return 0.0
        except:
            return 0.0
    
    def parse_decimal(self, decimal_str: str) -> float:
        """Convert decimal string to float"""
        try:
            return float(decimal_str) if decimal_str else 0.0
        except:
            return 0.0
    
    def odds_to_probability(self, odds_str: str) -> float:
        """Convert fractional odds to probability"""
        try:
            if not odds_str or odds_str == '-':
                return 0.0
            
            # Handle fractional odds like "2/1", "10/11"
            if '/' in odds_str:
                numerator, denominator = map(float, odds_str.split('/'))
                decimal_odds = (numerator / denominator) + 1
                return 1 / decimal_odds
            else:
                # If it's already decimal
                decimal_odds = float(odds_str)
                return 1 / decimal_odds
        except:
            return 0.0
    
    def calculate_team_strength(self, team_stats: Dict) -> Dict[str, float]:
        """Calculate team strength metrics from statistics"""
        if not team_stats:
            return {'attack': 0.5, 'defense': 0.5, 'form': 0.5}
        
        # Parse key statistics
        gf = self.parse_decimal(team_stats.get('GF', '0'))  # Goals for per game
        ga = self.parse_decimal(team_stats.get('GA', '0'))  # Goals against per game
        win_pct = self.parse_percentage(team_stats.get('W%', '0%'))
        cs_pct = self.parse_percentage(team_stats.get('CS', '0%'))  # Clean sheet %
        fts_pct = self.parse_percentage(team_stats.get('FTS', '0%'))  # Failed to score %
        
        # Calculate strength metrics (0-1 scale)
        attack_strength = min(gf / 3.0, 1.0)  # Normalize to max 3 goals per game
        defense_strength = max(0, 1 - (ga / 3.0))  # Better defense = fewer goals conceded
        form_strength = win_pct
        
        # Adjust for clean sheets and scoring consistency
        defense_strength = (defense_strength + cs_pct) / 2
        attack_strength = attack_strength * (1 - fts_pct)  # Penalize for failing to score
        
        return {
            'attack': max(0.1, attack_strength),    # Minimum 0.1 to avoid zero
            'defense': max(0.1, defense_strength),  # Minimum 0.1 to avoid zero
            'form': max(0.1, form_strength),        # Minimum 0.1 to avoid zero
            'goals_for': gf,
            'goals_against': ga
        }
    
    def calculate_expected_goals(self, home_stats: Dict, away_stats: Dict) -> Tuple[float, float]:
        """Calculate expected goals for both teams using team strengths"""
        home_strength = self.calculate_team_strength(home_stats)
        away_strength = self.calculate_team_strength(away_stats)
        
        # Base expected goals from historical averages
        league_avg_goals = 2.5  # Average goals per game in most leagues
        
        # Calculate expected goals using team strengths
        home_attack = home_strength['attack']
        away_defense = away_strength['defense']
        away_attack = away_strength['attack']
        home_defense = home_strength['defense']
        
        # Expected goals calculation with home advantage
        home_xg = (home_attack / away_defense) * league_avg_goals * 0.5 * (1 + self.home_advantage)
        away_xg = (away_attack / home_defense) * league_avg_goals * 0.5
        
        # Ensure reasonable bounds
        home_xg = max(0.1, min(5.0, home_xg))
        away_xg = max(0.1, min(5.0, away_xg))
        
        return home_xg, away_xg
    
    def poisson_prediction(self, home_xg: float, away_xg: float, max_goals: int = 6) -> Dict:
        """Generate score probabilities using Poisson distribution"""
        probabilities = {}
        total_prob = 0
        
        # Calculate probability for each possible score
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
                probabilities[f"{home_goals}-{away_goals}"] = prob
                total_prob += prob
        
        # Normalize probabilities
        for score in probabilities:
            probabilities[score] /= total_prob
        
        # Calculate outcome probabilities
        home_win_prob = sum(prob for score, prob in probabilities.items() 
                           if int(score.split('-')[0]) > int(score.split('-')[1]))
        draw_prob = sum(prob for score, prob in probabilities.items() 
                       if int(score.split('-')[0]) == int(score.split('-')[1]))
        away_win_prob = sum(prob for score, prob in probabilities.items() 
                           if int(score.split('-')[0]) < int(score.split('-')[1]))
        
        # Most likely scores
        sorted_scores = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'score_probabilities': probabilities,
            'outcome_probabilities': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'most_likely_scores': sorted_scores[:5],
            'expected_goals': {'home': home_xg, 'away': away_xg}
        }
    
    def odds_based_prediction(self, odds_1: str, odds_x: str, odds_2: str) -> Dict:
        """Generate predictions based on betting odds"""
        home_prob = self.odds_to_probability(odds_1)
        draw_prob = self.odds_to_probability(odds_x)
        away_prob = self.odds_to_probability(odds_2)
        
        # Normalize probabilities (remove bookmaker margin)
        total_prob = home_prob + draw_prob + away_prob
        if total_prob > 0:
            home_prob /= total_prob
            draw_prob /= total_prob
            away_prob /= total_prob
        
        # Estimate expected goals from odds-implied probabilities
        # Teams with higher win probability typically score more
        home_xg_odds = 1.5 * home_prob + 0.5
        away_xg_odds = 1.5 * away_prob + 0.5
        
        return {
            'outcome_probabilities': {
                'home_win': home_prob,
                'draw': draw_prob,
                'away_win': away_prob
            },
            'expected_goals': {'home': home_xg_odds, 'away': away_xg_odds},
            'bookmaker_margin': max(0, total_prob - 1)
        }
    
    def ensemble_prediction(self, match_data: Dict) -> Dict:
        """Combine statistical and odds-based predictions"""
        
        home_stats = match_data.get('home_team_stats')
        away_stats = match_data.get('away_team_stats')
        odds_1 = match_data.get('odds_1', '')
        odds_x = match_data.get('odds_x', '')
        odds_2 = match_data.get('odds_2', '')
        
        prediction = {
            'match_info': {
                'home_team': match_data.get('home_team'),
                'away_team': match_data.get('away_team'),
                'league': match_data.get('league'),
                'region': match_data.get('region'),
                'time': match_data.get('time')
            },
            'has_stats': bool(home_stats and away_stats),
            'has_odds': bool(odds_1 and odds_x and odds_2),
            'predictions': {}
        }
        
        # Statistical prediction (if stats available)
        if home_stats and away_stats:
            home_xg, away_xg = self.calculate_expected_goals(home_stats, away_stats)
            stat_pred = self.poisson_prediction(home_xg, away_xg)
            prediction['predictions']['statistical'] = stat_pred
            
            # Add team strength analysis
            prediction['team_analysis'] = {
                'home_strength': self.calculate_team_strength(home_stats),
                'away_strength': self.calculate_team_strength(away_stats)
            }
        
        # Odds-based prediction (if odds available)
        if odds_1 and odds_x and odds_2:
            odds_pred = self.odds_based_prediction(odds_1, odds_x, odds_2)
            prediction['predictions']['odds_based'] = odds_pred
        
        # Ensemble prediction (combine both if available)
        if 'statistical' in prediction['predictions'] and 'odds_based' in prediction['predictions']:
            stat_pred = prediction['predictions']['statistical']
            odds_pred = prediction['predictions']['odds_based']
            
            # Weighted combination of outcome probabilities
            ensemble_outcomes = {
                'home_win': (stat_pred['outcome_probabilities']['home_win'] * self.model_weights['statistical'] +
                           odds_pred['outcome_probabilities']['home_win'] * self.model_weights['odds_based']),
                'draw': (stat_pred['outcome_probabilities']['draw'] * self.model_weights['statistical'] +
                        odds_pred['outcome_probabilities']['draw'] * self.model_weights['odds_based']),
                'away_win': (stat_pred['outcome_probabilities']['away_win'] * self.model_weights['statistical'] +
                           odds_pred['outcome_probabilities']['away_win'] * self.model_weights['odds_based'])
            }
            
            # Weighted combination of expected goals
            ensemble_xg = {
                'home': (stat_pred['expected_goals']['home'] * self.model_weights['statistical'] +
                        odds_pred['expected_goals']['home'] * self.model_weights['odds_based']),
                'away': (stat_pred['expected_goals']['away'] * self.model_weights['statistical'] +
                        odds_pred['expected_goals']['away'] * self.model_weights['odds_based'])
            }
            
            # Generate new score predictions with ensemble xG
            ensemble_score_pred = self.poisson_prediction(ensemble_xg['home'], ensemble_xg['away'])
            
            prediction['predictions']['ensemble'] = {
                'outcome_probabilities': ensemble_outcomes,
                'expected_goals': ensemble_xg,
                'score_probabilities': ensemble_score_pred['score_probabilities'],
                'most_likely_scores': ensemble_score_pred['most_likely_scores']
            }
            
            # Add confidence score
            prediction['confidence'] = self._calculate_confidence(prediction)
        
        # Add betting value analysis
        if 'odds_based' in prediction['predictions'] and 'ensemble' in prediction['predictions']:
            prediction['betting_analysis'] = self._analyze_betting_value(prediction)
        
        return prediction
    
    def _calculate_confidence(self, prediction: Dict) -> Dict:
        """Calculate prediction confidence based on agreement between methods"""
        if 'ensemble' not in prediction['predictions']:
            return {'overall': 0.5}
        
        stat_outcomes = prediction['predictions']['statistical']['outcome_probabilities']
        odds_outcomes = prediction['predictions']['odds_based']['outcome_probabilities']
        
        # Calculate agreement between statistical and odds predictions
        agreement = 1 - sum(abs(stat_outcomes[outcome] - odds_outcomes[outcome]) 
                          for outcome in ['home_win', 'draw', 'away_win']) / 3
        
        # Higher agreement = higher confidence
        confidence = {
            'overall': agreement,
            'level': 'High' if agreement > 0.8 else 'Medium' if agreement > 0.6 else 'Low'
        }
        
        return confidence
    
    def _analyze_betting_value(self, prediction: Dict) -> Dict:
        """Analyze potential betting value based on model vs odds"""
        ensemble_probs = prediction['predictions']['ensemble']['outcome_probabilities']
        odds_probs = prediction['predictions']['odds_based']['outcome_probabilities']
        
        # Calculate expected value for each outcome
        value_analysis = {}
        outcomes = ['home_win', 'draw', 'away_win']
        
        for outcome in outcomes:
            model_prob = ensemble_probs[outcome]
            odds_prob = odds_probs[outcome]
            
            # Value = model probability - odds probability
            value = model_prob - odds_prob
            value_analysis[outcome] = {
                'model_probability': model_prob,
                'odds_probability': odds_prob,
                'value': value,
                'recommendation': 'BET' if value > 0.1 else 'AVOID' if value < -0.1 else 'NEUTRAL'
            }
        
        return value_analysis
    
    def predict_matches(self, merged_data_file: str, output_file: str = None) -> List[Dict]:
        """Predict outcomes for all matches in merged dataset"""
        
        with open(merged_data_file, 'r') as f:
            matches = json.load(f)
        
        predictions = []
        stats_available = 0
        
        logger.info(f"Generating predictions for {len(matches)} matches")
        
        for match in matches:
            if match.get('home_team_stats'):
                stats_available += 1
            
            prediction = self.ensemble_prediction(match)
            predictions.append(prediction)
        
        logger.info(f"Generated predictions for {len(matches)} matches")
        logger.info(f"Stats available for {stats_available} matches ({round(stats_available/len(matches)*100, 1)}%)")
        
        # Save predictions if output file specified
        if output_file:
            output_data = {
                'generated_at': datetime.now().isoformat(),
                'total_matches': len(matches),
                'matches_with_stats': stats_available,
                'model_info': {
                    'home_advantage': self.home_advantage,
                    'weights': self.model_weights
                },
                'predictions': predictions
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Predictions saved to: {output_file}")
        
        return predictions
    
    def generate_summary_report(self, predictions: List[Dict]) -> Dict:
        """Generate a summary report of predictions"""
        total_matches = len(predictions)
        with_stats = sum(1 for p in predictions if p['has_stats'])
        with_odds = sum(1 for p in predictions if p['has_odds'])
        ensemble_predictions = sum(1 for p in predictions if 'ensemble' in p['predictions'])
        
        # Confidence distribution
        confidence_levels = {'High': 0, 'Medium': 0, 'Low': 0, 'None': 0}
        for prediction in predictions:
            conf_level = prediction.get('confidence', {}).get('level', 'None')
            confidence_levels[conf_level] += 1
        
        # Most confident predictions
        confident_predictions = [
            p for p in predictions 
            if p.get('confidence', {}).get('overall', 0) > 0.8 and 'ensemble' in p['predictions']
        ]
        
        return {
            'summary': {
                'total_matches': total_matches,
                'with_statistics': with_stats,
                'with_odds': with_odds,
                'ensemble_predictions': ensemble_predictions,
                'coverage_percentage': round(ensemble_predictions / total_matches * 100, 2)
            },
            'confidence_distribution': confidence_levels,
            'high_confidence_predictions': len(confident_predictions),
            'sample_predictions': confident_predictions[:5] if confident_predictions else []
        }

def main():
    """Example usage"""
    predictor = FootballPredictionModel()
    
    # Generate predictions
    predictions = predictor.predict_matches(
        'merged_match_data.json',
        'match_predictions.json'
    )
    
    # Generate summary report
    report = predictor.generate_summary_report(predictions)
    
    print("\\n=== FOOTBALL PREDICTION MODEL REPORT ===")
    print(f"Total matches: {report['summary']['total_matches']}")
    print(f"With statistics: {report['summary']['with_statistics']}")
    print(f"With odds: {report['summary']['with_odds']}")
    print(f"Ensemble predictions: {report['summary']['ensemble_predictions']}")
    print(f"Coverage: {report['summary']['coverage_percentage']}%")
    
    print("\\n=== CONFIDENCE DISTRIBUTION ===")
    for level, count in report['confidence_distribution'].items():
        print(f"{level}: {count}")
    
    print(f"\\nHigh confidence predictions: {report['high_confidence_predictions']}")
    
    # Save report
    with open('prediction_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Show sample high-confidence predictions
    if report['sample_predictions']:
        print("\\n=== SAMPLE HIGH-CONFIDENCE PREDICTIONS ===")
        for i, pred in enumerate(report['sample_predictions'], 1):
            match_info = pred['match_info']
            if 'ensemble' in pred['predictions']:
                outcomes = pred['predictions']['ensemble']['outcome_probabilities']
                most_likely = max(outcomes.items(), key=lambda x: x[1])
                xg = pred['predictions']['ensemble']['expected_goals']
                
                print(f"\\n{i}. {match_info['home_team']} vs {match_info['away_team']}")
                print(f"   League: {match_info['league']}")
                print(f"   Most likely: {most_likely[0]} ({most_likely[1]:.1%})")
                print(f"   Expected Goals: {xg['home']:.1f} - {xg['away']:.1f}")
                print(f"   Confidence: {pred.get('confidence', {}).get('level', 'N/A')}")

if __name__ == "__main__":
    main()