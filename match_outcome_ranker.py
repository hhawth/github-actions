import json
from typing import List, Dict
from football_prediction_model import FootballPredictionModel

class MatchOutcomeRanker:
    """
    Advanced system for ranking match outcomes from most likely to worst.
    Combines multiple factors including probability, confidence, and value.
    """
    
    def __init__(self):
        self.prediction_model = FootballPredictionModel()
        
    def calculate_outcome_score(self, outcome: Dict) -> float:
        """
        Calculate comprehensive score for an outcome considering:
        - Predicted probability (40% weight)
        - Confidence level (25% weight)  
        - Value/Edge (20% weight)
        - Risk adjustment (15% weight)
        """
        prob_weight = 0.40
        conf_weight = 0.25
        value_weight = 0.20
        risk_weight = 0.15
        
        # Normalize probability (0-1)
        probability = outcome.get('probability', 0)
        
        # Confidence score (0-1)
        confidence = outcome.get('confidence', 0.5)
        
        # Value/edge score (can be negative)
        edge = outcome.get('edge', 0)
        value_score = max(0, min(1, edge * 10))  # Normalize edge to 0-1
        
        # Risk adjustment (lower risk = higher score)
        risk_score = outcome.get('risk_score', 0.5)
        risk_adjustment = 1 - min(1, risk_score)
        
        # Calculate weighted score
        total_score = (
            probability * prob_weight +
            confidence * conf_weight +
            value_score * value_weight +
            risk_adjustment * risk_weight
        )
        
        return total_score
    
    def rank_all_outcomes(self, merged_data_file: str = 'merged_match_data.json') -> List[Dict]:
        """
        Analyze all matches and rank ALL possible outcomes from most likely to worst
        """
        print("🔍 Analyzing all match outcomes for comprehensive ranking...")
        
        # Load match data
        try:
            with open(merged_data_file, 'r') as f:
                merged_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {merged_data_file} not found")
            return []
        
        all_outcomes = []
        processed = 0
        
        for match in merged_data:
            if not isinstance(match, dict):
                continue
            
            processed += 1
            
            # Skip matches without complete data
            if not match.get('home_team') or not match.get('away_team'):
                continue
            
            # Generate predictions for this match
            prediction = self.prediction_model.ensemble_prediction(match)
            
            # Extract probabilities
            predicted_probs = self._extract_probabilities(prediction)
            
            if not any(predicted_probs.values()):
                continue
            
            # Get odds for value calculation
            home_odds = self._convert_odds(match.get('odds_1'))
            draw_odds = self._convert_odds(match.get('odds_x'))
            away_odds = self._convert_odds(match.get('odds_2'))
            
            # Create outcome entries
            match_info = {
                'home_team': match.get('home_team', 'Unknown'),
                'away_team': match.get('away_team', 'Unknown'),
                'league': match.get('league', 'Unknown'),
                'time': match.get('time', 'TBD'),
                'date': match.get('date', 'TBD')
            }
            
            # Add all three outcomes for this match
            outcomes_data = [
                ('home_win', predicted_probs['home_win'], home_odds, 'Home Win', '1'),
                ('draw', predicted_probs['draw'], draw_odds, 'Draw', 'X'),
                ('away_win', predicted_probs['away_win'], away_odds, 'Away Win', '2')
            ]
            
            for outcome_type, prob, odds, display_name, symbol in outcomes_data:
                if prob > 0:
                    # Calculate implied probability and edge
                    implied_prob = (1.0 / odds) if odds > 1 else 0
                    edge = prob - implied_prob
                    
                    # Determine confidence based on prediction method
                    confidence = 0.8 if 'statistical' in prediction.get('predictions', {}) else 0.6
                    
                    # Calculate risk score
                    risk_score = self._calculate_outcome_risk(prob, confidence, odds)
                    
                    outcome = {
                        'match_info': match_info,
                        'outcome_type': outcome_type,
                        'outcome_display': display_name,
                        'symbol': symbol,
                        'probability': prob,
                        'implied_probability': implied_prob,
                        'edge': edge,
                        'odds': odds,
                        'confidence': confidence,
                        'risk_score': risk_score,
                        'expected_value': (prob * odds) - 1 if odds > 1 else 0,
                        'match_description': f"{match_info['home_team']} vs {match_info['away_team']}"
                    }
                    
                    # Calculate comprehensive ranking score
                    outcome['ranking_score'] = self.calculate_outcome_score(outcome)
                    
                    all_outcomes.append(outcome)
            
            # Progress update
            if processed % 50 == 0:
                print(f"  📊 Processed {processed} matches...")
        
        # Sort by ranking score (highest first = most likely)
        ranked_outcomes = sorted(all_outcomes, key=lambda x: x['ranking_score'], reverse=True)
        
        # Add ranking positions
        for i, outcome in enumerate(ranked_outcomes, 1):
            outcome['rank'] = i
            outcome['rank_percentile'] = (1 - (i-1) / len(ranked_outcomes)) * 100
        
        print(f"\\n✅ Ranking complete! Analyzed {processed} matches, ranked {len(ranked_outcomes)} outcomes")
        
        return ranked_outcomes
    
    def display_top_outcomes(self, ranked_outcomes: List[Dict], top_n: int = 20):
        """Display the top N most likely outcomes in a formatted way"""
        
        print(f"\\n🏆 TOP {top_n} MOST LIKELY MATCH OUTCOMES")
        print("=" * 80)
        
        for i, outcome in enumerate(ranked_outcomes[:top_n], 1):
            match = outcome['match_info']
            
            print(f"\\n{i:2d}. {outcome['match_description']}")
            print(f"     📊 Outcome: {outcome['outcome_display']} ({outcome['symbol']})")
            print(f"     🎯 Probability: {outcome['probability']:.1%}")
            print(f"     💰 Odds: {outcome['odds']:.2f}")
            print(f"     🔥 Confidence: {outcome['confidence']:.1%}")
            print(f"     📈 Ranking Score: {outcome['ranking_score']:.3f}")
            
            if outcome['edge'] > 0:
                print(f"     💎 Value Edge: +{outcome['edge']:.1%}")
            
            print(f"     🏆 Rank Percentile: {outcome['rank_percentile']:.1f}%")
            print(f"     📅 {match.get('date', 'TBD')} {match.get('time', 'TBD')}")
    
    def display_worst_outcomes(self, ranked_outcomes: List[Dict], bottom_n: int = 10):
        """Display the bottom N least likely outcomes"""
        
        print(f"\\n💀 BOTTOM {bottom_n} LEAST LIKELY OUTCOMES")
        print("=" * 80)
        
        worst_outcomes = ranked_outcomes[-bottom_n:]
        
        for i, outcome in enumerate(worst_outcomes, len(ranked_outcomes) - bottom_n + 1):
            match = outcome['match_info']
            
            print(f"\\n{i:2d}. {outcome['match_description']}")
            print(f"     📊 Outcome: {outcome['outcome_display']} ({outcome['symbol']})")
            print(f"     🎯 Probability: {outcome['probability']:.1%}")
            print(f"     💰 Odds: {outcome['odds']:.2f}")
            print(f"     📈 Ranking Score: {outcome['ranking_score']:.3f}")
    
    def create_ranking_categories(self, ranked_outcomes: List[Dict]) -> Dict:
        """Categorize outcomes into different likelihood tiers"""
        
        total = len(ranked_outcomes)
        
        categories = {
            'very_likely': [],      # Top 10%
            'likely': [],           # 10-30%
            'moderate': [],         # 30-60%
            'unlikely': [],         # 60-85%
            'very_unlikely': []     # Bottom 15%
        }
        
        for i, outcome in enumerate(ranked_outcomes):
            percentile = (i / total) * 100
            
            if percentile <= 10:
                categories['very_likely'].append(outcome)
            elif percentile <= 30:
                categories['likely'].append(outcome)
            elif percentile <= 60:
                categories['moderate'].append(outcome)
            elif percentile <= 85:
                categories['unlikely'].append(outcome)
            else:
                categories['very_unlikely'].append(outcome)
        
        return categories
    
    def save_ranking_report(self, ranked_outcomes: List[Dict], filename: str = 'outcome_rankings_report.json'):
        """Save comprehensive ranking report"""
        
        categories = self.create_ranking_categories(ranked_outcomes)
        
        report = {
            'generated_at': '2026-01-22T00:00:00',
            'analysis_type': 'comprehensive_outcome_ranking',
            'total_outcomes_analyzed': len(ranked_outcomes),
            'ranking_methodology': {
                'probability_weight': '40%',
                'confidence_weight': '25%',
                'value_edge_weight': '20%',
                'risk_adjustment_weight': '15%'
            },
            'top_10_outcomes': ranked_outcomes[:10],
            'bottom_10_outcomes': ranked_outcomes[-10:],
            'category_breakdown': {
                'very_likely': len(categories['very_likely']),
                'likely': len(categories['likely']),
                'moderate': len(categories['moderate']),
                'unlikely': len(categories['unlikely']),
                'very_unlikely': len(categories['very_unlikely'])
            },
            'categories': categories,
            'all_ranked_outcomes': ranked_outcomes[:100]  # Top 100 for file size
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n📄 Comprehensive ranking report saved to: {filename}")
    
    def _extract_probabilities(self, prediction: Dict) -> Dict:
        """Extract outcome probabilities from prediction"""
        predicted_probs = {'home_win': 0, 'draw': 0, 'away_win': 0}
        
        if 'statistical' in prediction.get('predictions', {}):
            stat_pred = prediction['predictions']['statistical']
            outcomes = stat_pred.get('outcome_probabilities', {})
            predicted_probs.update(outcomes)
        elif 'odds_based' in prediction.get('predictions', {}):
            odds_pred = prediction['predictions']['odds_based']
            outcomes = odds_pred.get('outcome_probabilities', {})
            predicted_probs.update(outcomes)
        
        return predicted_probs
    
    def _convert_odds(self, odds_str: str) -> float:
        """Convert fractional odds to decimal"""
        try:
            if not odds_str or odds_str == 'N/A':
                return 2.0  # Default odds
            
            if '/' in str(odds_str):
                # Fractional odds like "3/1"
                numerator, denominator = map(float, str(odds_str).split('/'))
                return (numerator / denominator) + 1
            else:
                # Already decimal
                return float(odds_str)
        except:
            return 2.0
    
    def _calculate_outcome_risk(self, probability: float, confidence: float, odds: float) -> float:
        """Calculate risk score for an outcome (lower = better)"""
        # Higher probability = lower risk
        prob_risk = 1 - probability
        
        # Higher confidence = lower risk  
        conf_risk = 1 - confidence
        
        # Odds risk (very high or very low odds increase risk)
        odds_risk = abs(odds - 2.5) / 10  # Risk increases as odds deviate from 2.5
        
        return (prob_risk * 0.5) + (conf_risk * 0.3) + (odds_risk * 0.2)

def main():
    """Main function to run outcome ranking analysis"""
    ranker = MatchOutcomeRanker()
    
    # Rank all outcomes
    ranked_outcomes = ranker.rank_all_outcomes()
    
    if not ranked_outcomes:
        print("❌ No outcomes to rank. Check your match data.")
        return
    
    # Display results
    ranker.display_top_outcomes(ranked_outcomes, top_n=15)
    ranker.display_worst_outcomes(ranked_outcomes, bottom_n=10)
    
    # Show category breakdown
    categories = ranker.create_ranking_categories(ranked_outcomes)
    
    print("\\n📊 OUTCOME LIKELIHOOD CATEGORIES")
    print("=" * 50)
    print(f"🔥 Very Likely (Top 10%): {len(categories['very_likely'])} outcomes")
    print(f"✅ Likely (10-30%): {len(categories['likely'])} outcomes")
    print(f"⚖️ Moderate (30-60%): {len(categories['moderate'])} outcomes")
    print(f"❓ Unlikely (60-85%): {len(categories['unlikely'])} outcomes")
    print(f"💀 Very Unlikely (Bottom 15%): {len(categories['very_unlikely'])} outcomes")
    
    # Save report
    ranker.save_ranking_report(ranked_outcomes)
    
    # Show some statistics
    total_value_bets = len([o for o in ranked_outcomes if o['edge'] > 0.05])
    high_prob_bets = len([o for o in ranked_outcomes if o['probability'] > 0.6])
    
    print("\\n📈 QUICK STATS:")
    print(f"  💰 Value opportunities (5%+ edge): {total_value_bets}")
    print(f"  🎯 High probability outcomes (60%+): {high_prob_bets}")
    print(f"  📊 Total outcomes ranked: {len(ranked_outcomes)}")

if __name__ == "__main__":
    main()