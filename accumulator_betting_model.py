import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

class AccumulatorBettingModel:
    """
    Advanced accumulator betting model that generates X-fold accumulator bets
    based on match predictions and statistical analysis.
    """
    
    def __init__(self, min_probability: float = 0.35, max_odds: float = 4.0):
        """
        Initialize the betting model with risk parameters
        
        Args:
            min_probability: Minimum win probability to consider a bet
            max_odds: Maximum odds to prevent extremely risky bets
        """
        self.min_probability = min_probability
        self.max_odds = max_odds
    
    def extract_betting_selection(self, prediction: Dict) -> Optional[Dict]:
        """Extract the best betting selection from a match prediction"""
        best_selections = []
        
        # Check ensemble prediction first (most reliable)
        if 'ensemble' in prediction.get('predictions', {}):
            ensemble = prediction['predictions']['ensemble']
            
            # Get the highest confidence outcome
            if 'outcome_probabilities' in ensemble:
                outcomes = ensemble['outcome_probabilities']
                confidence = ensemble.get('confidence_score', 0)
                
                # Find best outcome
                best_outcome = max(outcomes.items(), key=lambda x: x[1])
                outcome_name, probability = best_outcome
                
                # Map outcome to odds (this would normally come from merged data)
                odds_mapping = self._get_odds_for_outcome(prediction, outcome_name)
                if odds_mapping:
                    odds = odds_mapping['odds']
                    expected_value = (probability * odds) - 1
                    risk_score = self._calculate_risk_score(probability, confidence, odds)
                    
                    best_selections.append({
                        'outcome': outcome_name,
                        'probability': probability,
                        'odds': odds,
                        'confidence': confidence,
                        'source': 'ensemble',
                        'expected_value': expected_value,
                        'risk_score': risk_score
                    })
        
        # Check odds-based prediction as fallback
        if 'odds_based' in prediction.get('predictions', {}) and not best_selections:
            odds_pred = prediction['predictions']['odds_based']
            outcomes = odds_pred['outcome_probabilities']
            
            best_outcome = max(outcomes.items(), key=lambda x: x[1])
            outcome_name, probability = best_outcome
            
            odds_mapping = self._get_odds_for_outcome(prediction, outcome_name)
            if odds_mapping:
                odds = odds_mapping['odds']
                expected_value = (probability * odds) - 1
                risk_score = self._calculate_risk_score(probability, 0.5, odds)  # Default confidence
                
                best_selections.append({
                    'outcome': outcome_name,
                    'probability': probability,
                    'odds': odds,
                    'confidence': 0.5,
                    'source': 'odds_based',
                    'expected_value': expected_value,
                    'risk_score': risk_score
                })
        
        # Return best selection based on expected value and confidence
        if best_selections:
            return max(best_selections, key=lambda x: x['confidence'] * x['probability'])
        
        return None
    
    def _get_odds_for_outcome(self, prediction: Dict, outcome: str) -> Optional[Dict]:
        """Extract odds for specific outcome from prediction data"""
        # This is a placeholder - we need to access the original merged data
        # to get the actual odds. For now, simulate reasonable odds.
        
        # Map outcome names to typical odds ranges
        outcome_odds = {
            'home_win': 2.0,  # Placeholder
            'draw': 3.2,      # Placeholder  
            'away_win': 2.5   # Placeholder
        }
        
        if outcome in outcome_odds:
            return {
                'outcome': outcome,
                'odds': outcome_odds[outcome]
            }
        
        return None
    
    def _calculate_risk_score(self, probability: float, confidence: float, odds: float) -> float:
        """Calculate risk score (lower = safer bet)"""
        # Risk factors:
        # 1. Low probability = high risk
        # 2. Low confidence = high risk  
        # 3. High odds = high risk
        # 4. Poor expected value = high risk
        
        prob_risk = 1 - probability  # Higher probability = lower risk
        confidence_risk = 1 - confidence  # Higher confidence = lower risk
        odds_risk = min(odds / 5.0, 1.0)  # Normalize odds risk
        
        # Weighted risk score (0 = no risk, 1 = maximum risk)
        risk_score = (prob_risk * 0.4 + confidence_risk * 0.3 + odds_risk * 0.3)
        return min(max(risk_score, 0), 1)
    
    def rank_selections(self, selections: List[Dict]) -> List[Dict]:
        """Rank selections by betting attractiveness"""
        
        # Filter selections based on minimum criteria
        filtered_selections = [
            sel for sel in selections 
            if (sel['probability'] >= self.min_probability and 
                sel['odds'] <= self.max_odds and
                sel['risk_score'] <= 0.7)  # Max acceptable risk
        ]
        
        # Sort by composite score: probability * confidence * (1 - risk_score)
        def selection_score(selection):
            return (selection['probability'] * 
                   selection['confidence'] * 
                   (1 - selection['risk_score']))
        
        ranked_selections = sorted(filtered_selections, 
                                 key=selection_score, 
                                 reverse=True)
        
        # Add rank to each selection
        for i, selection in enumerate(ranked_selections, 1):
            selection['rank'] = i
            selection['composite_score'] = selection_score(selection)
        
        return ranked_selections
    
    def build_accumulator(self, ranked_selections: List[Dict], fold_size: int) -> Dict:
        """Build an accumulator bet from top ranked selections"""
        if fold_size > len(ranked_selections):
            fold_size = len(ranked_selections)
        
        top_selections = ranked_selections[:fold_size]
        
        if not top_selections:
            return {'error': 'No suitable selections found'}
        
        # Calculate accumulator details
        total_odds = 1.0
        combined_probability = 1.0
        total_risk = 0.0
        
        for selection in top_selections:
            total_odds *= selection['odds']
            combined_probability *= selection['probability']
            total_risk = max(total_risk, selection['risk_score'])  # Use highest individual risk
        
        # Calculate potential returns for different stakes
        stake_options = [1, 5, 10, 20, 50, 100]
        potential_returns = {stake: (stake * total_odds) - stake for stake in stake_options}
        
        accumulator = {
            'fold_size': fold_size,
            'selections': top_selections,
            'total_odds': round(total_odds, 2),
            'combined_probability': round(combined_probability, 4),
            'expected_value': round((combined_probability * total_odds) - 1, 4),
            'risk_score': round(total_risk, 3),
            'risk_level': self._get_risk_level(total_risk),
            'potential_returns': potential_returns,
            'recommendation': self._get_recommendation(combined_probability, total_odds, total_risk)
        }
        
        return accumulator
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to human-readable level"""
        if risk_score <= 0.3:
            return 'Low'
        elif risk_score <= 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _get_recommendation(self, probability: float, odds: float, risk: float) -> str:
        """Generate betting recommendation"""
        expected_value = (probability * odds) - 1
        
        if expected_value > 0.2 and risk <= 0.4:
            return 'STRONG BET'
        elif expected_value > 0.1 and risk <= 0.6:
            return 'GOOD BET'
        elif expected_value > 0 and risk <= 0.7:
            return 'FAIR BET'
        else:
            return 'AVOID'
    
    def generate_multiple_accumulators(self, ranked_selections: List[Dict], max_fold: int = 8) -> List[Dict]:
        """Generate accumulators of different sizes"""
        accumulators = []
        
        for fold_size in range(2, min(max_fold + 1, len(ranked_selections) + 1)):
            accumulator = self.build_accumulator(ranked_selections, fold_size)
            if 'error' not in accumulator:
                accumulators.append(accumulator)
        
        return accumulators
    
    def generate_betting_report_from_merged_data(self, merged_data_file: str, predictions_file: str = None, output_file: str = None) -> Dict:
        """Generate betting report using merged match data (preferred method)"""
        
        # Load merged match data (contains odds and stats)
        with open(merged_data_file, 'r') as f:
            merged_data = json.load(f)
        
        # Load predictions if available
        predictions_dict = {}
        if predictions_file:
            try:
                with open(predictions_file, 'r') as f:
                    predictions_data = json.load(f)
                    # Convert to dictionary keyed by match identifier
                    for pred in predictions_data:
                        match_key = f"{pred['home_team']} vs {pred['away_team']}"
                        predictions_dict[match_key] = pred
            except FileNotFoundError:
                print(f"Predictions file {predictions_file} not found, using odds-only analysis")
        
        # Process matches and extract betting selections
        selections = []
        
        for match in merged_data:
            if not isinstance(match, dict):
                continue
                
            # Create match info
            match_info = {
                'home_team': match.get('home_team', 'Unknown'),
                'away_team': match.get('away_team', 'Unknown'),
                'league': match.get('league', 'Unknown'),
                'time': match.get('time', 'Unknown'),
                'date': match.get('date', 'Unknown')
            }
            
            # Get match predictions if available
            match_key = f"{match_info['home_team']} vs {match_info['away_team']}"
            prediction = predictions_dict.get(match_key)
            
            # Extract odds directly from merged data
            odds_data = self._extract_odds_from_merged_data(match)
            
            if odds_data:
                # Create betting selections from odds
                for outcome, odds_info in odds_data.items():
                    odds = odds_info['odds']
                    implied_prob = 1.0 / odds if odds > 0 else 0.1
                    
                    # Use prediction data if available, otherwise use implied probabilities
                    if prediction:
                        selection = self.extract_betting_selection(prediction)
                        if selection and selection['outcome'] == outcome:
                            # Use prediction probability but real odds
                            selection['odds'] = odds
                            selection['expected_value'] = (selection['probability'] * odds) - 1
                            selection['risk_score'] = self._calculate_risk_score(
                                selection['probability'], 
                                selection['confidence'], 
                                odds
                            )
                    else:
                        # Create selection from odds only
                        selection = {
                            'outcome': outcome,
                            'probability': implied_prob,
                            'odds': odds,
                            'confidence': 0.4,  # Lower confidence for odds-only
                            'source': 'odds_only',
                            'expected_value': (implied_prob * odds) - 1,
                            'risk_score': self._calculate_risk_score(implied_prob, 0.4, odds)
                        }
                    
                    # Add match info to selection
                    selection['match_info'] = match_info
                    
                    # Filter by basic criteria
                    if (selection['probability'] >= self.min_probability and 
                        selection['odds'] <= self.max_odds):
                        selections.append(selection)
        
        # Process selections and generate accumulators
        return self._generate_betting_report_from_selections(selections, output_file)
    
    def _extract_odds_from_merged_data(self, match: Dict) -> Dict:
        """Extract odds data from merged match data"""
        odds_data = {}
        
        # Map common odds fields to outcomes
        odds_mapping = {
            'home_odds': 'home_win',
            'draw_odds': 'draw', 
            'away_odds': 'away_win',
            'home_win_odds': 'home_win',
            'draw_win_odds': 'draw',
            'away_win_odds': 'away_win'
        }
        
        for odds_field, outcome in odds_mapping.items():
            if odds_field in match and match[odds_field]:
                try:
                    odds_value = float(match[odds_field])
                    if odds_value > 1.0:  # Valid odds
                        odds_data[outcome] = {
                            'odds': odds_value,
                            'field': odds_field
                        }
                except (ValueError, TypeError):
                    continue
        
        return odds_data
    
    def generate_betting_report(self, predictions_file: str, output_file: str = None) -> Dict:
        """Generate betting report from match predictions"""
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        # Extract betting selections
        selections = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
                
            # Create match info
            match_info = {
                'home_team': prediction.get('home_team', 'Unknown'),
                'away_team': prediction.get('away_team', 'Unknown'),
                'league': prediction.get('league', 'Unknown'),
                'time': prediction.get('time', 'Unknown')
            }
            
            # Extract best betting selection
            selection = self.extract_betting_selection(prediction)
            
            if selection:
                # Add match info
                selection['match_info'] = match_info
                selections.append(selection)
        
        return self._generate_betting_report_from_selections(selections, output_file)
    
    def _generate_betting_report_from_selections(self, selections: List[Dict], output_file: str = None) -> Dict:
        """Generate complete betting report from selections"""
        
        # Add confidence score to selections for ranking
        for selection in selections:
            selection['confidence_score'] = selection['confidence']
        
        # Rank selections
        ranked_selections = self.rank_selections(selections)
        
        # Generate accumulators of different sizes
        accumulators = self.generate_multiple_accumulators(ranked_selections)
        
        # Find best accumulator
        best_accumulator = None
        if accumulators:
            # Sort by expected value, then by win probability
            best_accumulator = max(accumulators, 
                                 key=lambda x: (x['expected_value'], x['combined_probability']))
        
        # Calculate summary statistics
        total_matches = len(selections)
        qualified_selections = len(ranked_selections)
        
        # Calculate risk distribution
        risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
        avg_odds = 0
        avg_confidence = 0
        
        if ranked_selections:
            for sel in ranked_selections:
                risk_level = self._get_risk_level(sel['risk_score'])
                risk_levels[risk_level] += 1
                avg_odds += sel['odds']
                avg_confidence += sel['confidence']
            
            avg_odds /= len(ranked_selections)
            avg_confidence /= len(ranked_selections)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_matches_analyzed': total_matches,
            'qualified_selections': qualified_selections,
            'top_10_selections': ranked_selections[:10],
            'recommended_accumulators': accumulators,
            'betting_summary': {
                'total_selections': qualified_selections,
                'average_odds': round(avg_odds, 2),
                'average_confidence': round(avg_confidence, 3)
            },
            'risk_distribution': risk_levels,
            'best_single_bet': ranked_selections[0] if ranked_selections else None,
            'best_accumulator': best_accumulator,
            'total_accumulators_generated': len(accumulators)
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Betting report saved to {output_file}")
        
        return report
    
    def print_betting_recommendations(self, report: Dict):
        """Print formatted betting recommendations"""
        print("\\n" + "="*60)
        print("           ACCUMULATOR BETTING RECOMMENDATIONS")
        print("="*60)
        
        summary = report['betting_summary']
        print(f"\\nMatches Analyzed: {report['total_matches_analyzed']}")
        print(f"Qualified Selections: {report['qualified_selections']}")
        print(f"Accumulators Generated: {summary['total_accumulators_generated']}")
        
        # Best single bet
        if summary['best_single_bet']:
            bet = summary['best_single_bet']
            print(f"\\n🏆 BEST SINGLE BET:")
            print(f"   {bet['match_info']['home_team']} vs {bet['match_info']['away_team']}")
            print(f"   Selection: {bet['outcome'].replace('_', ' ').title()}")
            print(f"   Probability: {bet['probability']:.1%} | Odds: {bet['odds']:.2f}")
            print(f"   Confidence: {bet['confidence_score']:.1%} | Risk: {bet['risk_score']:.2f}")
        
        # Best accumulator
        if summary['best_accumulator']:
            acc = summary['best_accumulator']
            print(f"\\n🎯 RECOMMENDED ACCUMULATOR ({acc['fold_size']}-fold):")
            print(f"   Total Odds: {acc['total_odds']:.2f}")
            print(f"   Win Probability: {acc['combined_probability']:.1%}")
            print(f"   Expected Value: {acc['expected_value']:.3f}")
            print(f"   Risk Level: {acc['risk_level']}")
            print(f"   Recommendation: {acc['recommendation']}")
            
            print(f"\\n   Potential Returns (stake → return):")
            for stake, returns in acc['potential_returns'].items():
                if stake <= 50:  # Show smaller stakes
                    print(f"   £{stake} → £{returns:.2f}")
        
        # Top selections
        print(f"\\n📊 TOP 5 SELECTIONS:")
        for i, selection in enumerate(report['top_10_selections'][:5], 1):
            match = selection['match_info']
            print(f"   {i}. {match['home_team']} vs {match['away_team']}")
            print(f"      {selection['outcome'].replace('_', ' ').title()} @ {selection['odds']:.2f}")
            print(f"      Prob: {selection['probability']:.1%} | Score: {selection['composite_score']:.3f}")
            print(f"      League: {match['league']} | Time: {match['time']}")
            print()

def main():
    """Example usage"""
    betting_model = AccumulatorBettingModel()
    
    # Generate betting report
    report = betting_model.generate_betting_report(
        'match_predictions.json',
        'betting_report.json'
    )
    
    # Print recommendations
    betting_model.print_betting_recommendations(report)
    
    # Show additional accumulator options
    print("\\n" + "="*60)
    print("           ACCUMULATOR OPTIONS")
    print("="*60)
    
    for i, acc in enumerate(report['recommended_accumulators'], 1):
        print(f"\\n{i}. {acc['fold_size']}-FOLD ACCUMULATOR:")
        print(f"   Odds: {acc['total_odds']:.2f} | Probability: {acc['combined_probability']:.2%}")
        print(f"   Risk: {acc['risk_level']} | EV: {acc['expected_value']:.3f}")
        print(f"   £10 stake → £{acc['potential_returns'][10]:.2f} return")
        print(f"   Recommendation: {acc['recommendation']}")

if __name__ == "__main__":
    main()