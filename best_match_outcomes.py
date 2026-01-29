#!/usr/bin/env python3
"""
Best Match Outcomes - Select the single best outcome from each match and rank them
"""
import json
from typing import List, Dict
from football_prediction_model import FootballPredictionModel

class BestMatchOutcomeSelector:
    """
    System for selecting the best outcome from each match and ranking matches
    from best to worst betting opportunities.
    """
    
    def __init__(self):
        self.prediction_model = FootballPredictionModel()
        
    def _convert_odds(self, odds_str: str) -> float:
        """Convert odds string to decimal odds"""
        if not odds_str or odds_str == 'N/A':
            return 1.0
        
        try:
            if '/' in odds_str:
                num, den = map(int, odds_str.split('/'))
                return (num / den) + 1.0
            else:
                return float(odds_str)
        except (ValueError, ZeroDivisionError):
            return 1.0
    
    def _extract_probabilities(self, prediction: Dict) -> Dict[str, float]:
        """Extract probabilities from prediction result"""
        probs = {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
        
        if not prediction or 'predictions' not in prediction:
            return probs
        
        preds = prediction['predictions']
        
        # Try to get probabilities from different prediction methods
        # Prefer ensemble, then statistical, then odds_based
        for method in ['ensemble', 'statistical', 'odds_based']:
            if method in preds and 'outcome_probabilities' in preds[method]:
                method_probs = preds[method]['outcome_probabilities']
                
                # Extract probabilities with the correct key format
                for outcome_key in ['home_win', 'draw', 'away_win']:
                    if outcome_key in method_probs:
                        probs[outcome_key] = max(probs[outcome_key], method_probs[outcome_key])
        
        return probs
    
    def _calculate_outcome_quality(self, probability: float, odds: float, confidence: float = 0.7) -> float:
        """
        Calculate the quality score for an outcome considering:
        - Expected value (40% weight)
        - Probability strength (30% weight)  
        - Value edge (20% weight)
        - Confidence level (10% weight)
        """
        if odds <= 1.0:
            return 0.0
            
        # Calculate components
        expected_value = (probability * odds) - 1
        value_edge = probability - (1.0 / odds)
        
        # Normalize scores (0-1 range)
        ev_score = max(0, min(1, expected_value + 1))  # Shift to positive range
        prob_score = probability
        edge_score = max(0, min(1, value_edge * 10))  # Scale edge appropriately
        conf_score = confidence
        
        # Weighted combination
        quality_score = (
            ev_score * 0.40 +
            prob_score * 0.30 +
            edge_score * 0.20 +
            conf_score * 0.10
        )
        
        return quality_score
    
    def select_best_outcomes_per_match(self, merged_data_file: str = 'merged_match_data.json') -> List[Dict]:
        """
        For each match, select the single best outcome and rank all matches
        from best to worst opportunities.
        """
        print("🎯 Selecting best outcome for each match...")
        
        # Load match data
        try:
            with open(merged_data_file, 'r') as f:
                merged_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {merged_data_file} not found")
            return []
        
        best_match_outcomes = []
        processed = 0
        no_predictions = 0
        no_outcomes = 0
        
        for match in merged_data:
            if not isinstance(match, dict):
                continue
            
            processed += 1
            
            # Skip matches without complete data
            if not match.get('home_team') or not match.get('away_team'):
                continue
            
            # Generate predictions for this match
            prediction = self.prediction_model.ensemble_prediction(match)
            predicted_probs = self._extract_probabilities(prediction)
            
            # Debug: Check if we're getting predictions
            if not any(predicted_probs.values()):
                no_predictions += 1
                if processed <= 3:  # Debug first few matches
                    print(f"  ⚠️ No probabilities for {match.get('home_team')} vs {match.get('away_team')}")
                    print(f"     Prediction: {prediction}")
                continue
            
            # Get odds for all outcomes
            home_odds = self._convert_odds(match.get('odds_1'))
            draw_odds = self._convert_odds(match.get('odds_x'))
            away_odds = self._convert_odds(match.get('odds_2'))
            score = f"{match.get('home_score', None)} - {match.get('away_score', None)}"
            
            # Match information
            match_info = {
                'home_team': match.get('home_team', 'Unknown'),
                'away_team': match.get('away_team', 'Unknown'),
                'league': match.get('league', 'Unknown'),
                'time': match.get('time', 'TBD'),
                'date': match.get('date', 'TBD'),
                'score': score
            }
            
            # Evaluate all three outcomes for this match
            outcomes_to_evaluate = [
                ('home_win', predicted_probs['home_win'], home_odds, 'Home Win', '1'),
                ('draw', predicted_probs['draw'], draw_odds, 'Draw', 'X'),
                ('away_win', predicted_probs['away_win'], away_odds, 'Away Win', '2')
            ]
            
            best_outcome = None
            best_quality = 0
            
            for outcome_type, prob, odds, display_name, symbol in outcomes_to_evaluate:
                if prob > 0 and odds > 1:
                    # Calculate confidence based on prediction method
                    confidence = 0.8 if 'statistical' in prediction.get('predictions', {}) else 0.6
                    
                    # Calculate quality score for this outcome
                    quality = self._calculate_outcome_quality(prob, odds, confidence)
                    
                    if quality > best_quality:
                        best_quality = quality
                        
                        # Calculate additional metrics
                        implied_prob = 1.0 / odds
                        edge = prob - implied_prob
                        expected_value = (prob * odds) - 1
                        
                        best_outcome = {
                            'match_info': match_info,
                            'match_description': f"{match_info['home_team']} vs {match_info['away_team']}",
                            'outcome_type': outcome_type,
                            'outcome_display': display_name,
                            'symbol': symbol,
                            'probability': prob,
                            'implied_probability': implied_prob,
                            'edge': edge,
                            'odds': odds,
                            'confidence': confidence,
                            'expected_value': expected_value,
                            'quality_score': quality,
                            'recommendation_strength': self._get_recommendation_strength(quality)
                        }
            
            # Add the best outcome for this match
            if best_outcome:
                best_match_outcomes.append(best_outcome)
            else:
                no_outcomes += 1
                if processed <= 3:  # Debug first few matches
                    print(f"  ⚠️ No valid outcome for {match.get('home_team')} vs {match.get('away_team')}")
                    print(f"     Probs: {predicted_probs}")
                    print(f"     Odds: H:{home_odds} D:{draw_odds} A:{away_odds}")
            
            # Progress update
            if processed % 50 == 0:
                print(f"  📊 Processed {processed} matches...")
        
        # Sort by quality score (highest first = best opportunities)
        ranked_best_outcomes = sorted(best_match_outcomes, key=lambda x: x['quality_score'], reverse=True)
        
        # Add ranking positions
        for i, outcome in enumerate(ranked_best_outcomes, 1):
            outcome['rank'] = i
            outcome['rank_percentile'] = (1 - (i-1) / len(ranked_best_outcomes)) * 100
        
        print(f"\\n✅ Analysis complete! Processed {processed} matches")
        print(f"📊 Found best outcomes for {len(ranked_best_outcomes)} matches")
        print(f"🚫 Skipped {no_predictions} matches (no predictions)")
        print(f"❌ Skipped {no_outcomes} matches (no valid outcomes)")
        
        return ranked_best_outcomes
    
    def _get_recommendation_strength(self, quality_score: float) -> str:
        """Convert quality score to recommendation strength"""
        if quality_score >= 0.8:
            return "🔥 STRONG"
        elif quality_score >= 0.6:
            return "✅ GOOD"
        elif quality_score >= 0.4:
            return "⚠️ MODERATE"
        else:
            return "❌ WEAK"
    
    def display_ranked_best_outcomes(self, ranked_outcomes: List[Dict], show_count: int = 20):
        """Display the ranked best outcomes from each match"""
        
        print(f"\\n🏆 TOP {show_count} BEST MATCH OPPORTUNITIES")
        print("=" * 80)
        print("Each match shows its single BEST outcome opportunity\\n")
        
        for i, outcome in enumerate(ranked_outcomes[:show_count], 1):
            match = outcome['match_info']
            
            print(f"{i:2d}. {outcome['match_description']}")
            print(f"    📊 Best Outcome: {outcome['outcome_display']} ({outcome['symbol']})")
            print(f"    🎯 Probability: {outcome['probability']:.1%} @ {outcome['odds']:.2f} odds")
            print(f"    💎 Quality Score: {outcome['quality_score']:.3f}")
            print(f"    {outcome['recommendation_strength']}")
            
            if outcome['edge'] > 0:
                print(f"    💰 Value Edge: +{outcome['edge']:.1%}")
            print(f"    📈 Expected Value: {outcome['expected_value']:.3f}")
            print(f"    📅 {match.get('date', 'TBD')} {match.get('time', 'TBD')}")
            print(f"    🏆 Match Rank: #{outcome['rank']} ({outcome['rank_percentile']:.1f}% percentile)")
            print()
    
    def create_match_categories(self, ranked_outcomes: List[Dict]) -> Dict:
        """Categorize matches by opportunity quality"""
        
        total = len(ranked_outcomes)
        
        # Define category thresholds
        excellent_threshold = int(total * 0.1)   # Top 10%
        good_threshold = int(total * 0.25)       # Top 25%
        decent_threshold = int(total * 0.50)     # Top 50%
        
        categories = {
            'excellent': ranked_outcomes[:excellent_threshold],
            'good': ranked_outcomes[excellent_threshold:good_threshold],
            'decent': ranked_outcomes[good_threshold:decent_threshold],
            'poor': ranked_outcomes[decent_threshold:]
        }
        
        return categories
    
    def display_category_summary(self, categories: Dict):
        """Display summary of match categories"""
        
        print("\\n📊 MATCH OPPORTUNITY CATEGORIES")
        print("=" * 40)
        print(f"🔥 Excellent (Top 10%):    {len(categories['excellent']):3d} matches")
        print(f"✅ Good (10-25%):          {len(categories['good']):3d} matches")
        print(f"⚠️ Decent (25-50%):        {len(categories['decent']):3d} matches")
        print(f"❌ Poor (Bottom 50%):      {len(categories['poor']):3d} matches")
        print()
        
        # Show average quality scores per category
        for category_name, matches in categories.items():
            if matches:
                avg_quality = sum(m['quality_score'] for m in matches) / len(matches)
                avg_edge = sum(max(0, m['edge']) for m in matches) / len(matches)
                print(f"   {category_name.title()}: Avg Quality {avg_quality:.3f} | Avg Edge {avg_edge:.1%}")

def main():
    """Main function to run the best match outcomes analysis"""
    
    print("🎯 BEST MATCH OUTCOMES ANALYZER")
    print("=" * 60)
    print("Finding the single best betting opportunity for each match...")
    print()
    
    selector = BestMatchOutcomeSelector()
    
    # Get best outcomes for each match, ranked from best to worst
    ranked_outcomes = selector.select_best_outcomes_per_match()
    
    if not ranked_outcomes:
        print("❌ No match data found or processed")
        return
    
    # Display results
    selector.display_ranked_best_outcomes(ranked_outcomes, show_count=15)
    
    # Show category breakdown
    categories = selector.create_match_categories(ranked_outcomes)
    selector.display_category_summary(categories)
    
    # Save results
    print("\\n💾 Saving results...")
    with open('best_match_outcomes_report.json', 'w') as f:
        json.dump(ranked_outcomes, f, indent=2)
    print("   📁 Saved to: best_match_outcomes_report.json")

if __name__ == "__main__":
    main()