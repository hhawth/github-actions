import json
from working_betting_system import convert_fractional_to_decimal, AccumulatorBettingModel
from football_prediction_model import FootballPredictionModel

def create_value_betting_system():
    """Create a betting system that actually finds value by comparing predictions to odds"""
    
    # Load merged data
    with open('merged_match_data.json', 'r') as f:
        merged_data = json.load(f)
    
    # More aggressive betting model
    betting_model = AccumulatorBettingModel(
        min_probability=0.35,  # Require reasonable confidence
        max_odds=4.0           # Avoid extremely long shots
    )
    
    # Initialize prediction model
    prediction_model = FootballPredictionModel()
    
    selections = []
    processed = 0
    value_bets_found = 0
    
    print(f"🔍 Searching for value bets in {len(merged_data)} matches...")
    
    for match in merged_data:
        if not isinstance(match, dict):
            continue
            
        processed += 1
        
        # Skip matches without team stats (can't generate predictions)
        if not match.get('home_team_stats') or not match.get('away_team_stats'):
            continue
        
        # Extract team names and match info
        home_team = match.get('home_team', 'Unknown')
        away_team = match.get('away_team', 'Unknown')
        league = match.get('league', 'Unknown')
        
        # Convert odds from fractional to decimal
        odds_1_decimal = convert_fractional_to_decimal(match.get('odds_1'))  # Home win
        odds_x_decimal = convert_fractional_to_decimal(match.get('odds_x'))  # Draw
        odds_2_decimal = convert_fractional_to_decimal(match.get('odds_2'))  # Away win
        
        # Generate prediction using statistical model
        prediction = prediction_model.ensemble_prediction(match)
        
        # Extract predicted probabilities (these could differ from implied odds probabilities)
        predicted_probs = {}
        if 'statistical' in prediction.get('predictions', {}):
            stat_pred = prediction['predictions']['statistical']
            outcomes = stat_pred.get('outcome_probabilities', {})
            predicted_probs = {
                'home_win': outcomes.get('home_win', 0),
                'draw': outcomes.get('draw', 0), 
                'away_win': outcomes.get('away_win', 0)
            }
        elif 'odds_based' in prediction.get('predictions', {}):
            odds_pred = prediction['predictions']['odds_based']
            outcomes = odds_pred.get('outcome_probabilities', {})
            predicted_probs = {
                'home_win': outcomes.get('home_win', 0),
                'draw': outcomes.get('draw', 0),
                'away_win': outcomes.get('away_win', 0)
            }
        
        # Compare predictions to available odds to find value
        opportunities = [
            ('home_win', odds_1_decimal, predicted_probs.get('home_win', 0), '1'),
            ('draw', odds_x_decimal, predicted_probs.get('draw', 0), 'X'), 
            ('away_win', odds_2_decimal, predicted_probs.get('away_win', 0), '2')
        ]
        
        for outcome_name, decimal_odds, predicted_prob, symbol in opportunities:
            if not decimal_odds or decimal_odds <= 1.0 or predicted_prob <= 0:
                continue
            
            # Calculate implied probability from odds
            implied_prob = 1.0 / decimal_odds
            
            # Look for value: when our prediction is higher than implied probability
            edge = predicted_prob - implied_prob
            
            if edge > 0.05:  # Require at least 5% edge
                expected_value = (predicted_prob * decimal_odds) - 1
                
                # Use higher confidence for stat-based predictions
                confidence = 0.7 if 'statistical' in prediction.get('predictions', {}) else 0.5
                
                selection = {
                    'outcome': outcome_name,
                    'probability': predicted_prob,  # Use our prediction, not implied probability
                    'implied_probability': implied_prob,
                    'edge': edge,
                    'odds': decimal_odds,
                    'confidence': confidence,
                    'source': 'value_betting',
                    'expected_value': expected_value,
                    'risk_score': betting_model._calculate_risk_score(predicted_prob, confidence, decimal_odds),
                    'match_info': {
                        'home_team': home_team,
                        'away_team': away_team,
                        'league': league,
                        'time': match.get('time', 'TBD'),
                        'date': match.get('date', 'TBD')
                    },
                    'selection_symbol': symbol
                }
                
                # Apply betting filters
                if (selection['probability'] >= betting_model.min_probability and 
                    selection['odds'] <= betting_model.max_odds and
                    selection['expected_value'] > 0.1):  # Require positive EV
                    selections.append(selection)
                    value_bets_found += 1
                    
                    print(f"💰 Value found: {home_team} vs {away_team}")
                    print(f"    {outcome_name.replace('_', ' ').title()}: {predicted_prob:.1%} vs {implied_prob:.1%} implied (+{edge:.1%} edge)")
                    print(f"    Odds: {decimal_odds:.2f}, Expected Value: {expected_value:.3f}")
        
        # Show progress every 25 matches
        if processed % 25 == 0:
            print(f"  📊 Processed {processed} matches, found {value_bets_found} value opportunities")
    
    print("\\n🎯 Value Betting Analysis Complete:")
    print(f"  📈 Matches analyzed: {processed}")
    print(f"  💎 Value opportunities found: {value_bets_found}")
    print(f"  ✅ Qualifying value bets: {len(selections)}")
    
    if len(selections) < 2:
        print("  ⚠️ Limited value opportunities found")
        if len(selections) > 0:
            print("  💡 Try single bets instead of accumulators")
        return selections
    
    # Enhanced ranking system
    def calculate_comprehensive_score(selection):
        """Calculate comprehensive ranking score"""
        probability_weight = 0.35
        edge_weight = 0.25
        ev_weight = 0.20
        confidence_weight = 0.15
        risk_weight = 0.05
        
        prob_score = selection['probability']
        edge_score = max(0, selection['edge']) * 10  # Normalize edge
        ev_score = max(0, selection['expected_value'])
        conf_score = selection['confidence']
        risk_score = 1 - min(1, selection['risk_score'])  # Lower risk = higher score
        
        return (prob_score * probability_weight + 
                edge_score * edge_weight +
                ev_score * ev_weight + 
                conf_score * confidence_weight +
                risk_score * risk_weight)
    
    # Add comprehensive scores
    for selection in selections:
        selection['confidence_score'] = selection['confidence']
        selection['comprehensive_score'] = calculate_comprehensive_score(selection)
    
    # Rank selections by comprehensive score
    def value_score(selection):
        return selection['comprehensive_score']
    
    ranked_selections = sorted(selections, key=value_score, reverse=True)
    
    # Add ranks and percentiles
    total_selections = len(ranked_selections)
    for i, selection in enumerate(ranked_selections, 1):
        selection['rank'] = i
        selection['rank_percentile'] = (1 - (i-1) / total_selections) * 100 if total_selections > 0 else 100
        selection['value_score'] = value_score(selection)
        selection['likelihood_tier'] = self._get_likelihood_tier(selection['rank_percentile'])
    
    def _get_likelihood_tier(percentile):
        """Assign likelihood tier based on percentile ranking"""
        if percentile >= 90:
            return "🔥 Extremely Likely"
        elif percentile >= 70:
            return "✅ Very Likely" 
        elif percentile >= 50:
            return "⚖️ Likely"
        elif percentile >= 30:
            return "❓ Moderate"
        else:
            return "💀 Unlikely"
    
    # Generate accumulators
    accumulators = betting_model.generate_multiple_accumulators(ranked_selections, max_fold=5)
    
    print("\\n💰 VALUE BETTING OPPORTUNITIES")
    print("="*60)
    
    # Show top 5 value bets
    print("\n🏆 TOP 5 VALUE BETS (RANKED BY LIKELIHOOD):")
    for i, selection in enumerate(ranked_selections[:5], 1):
        match = selection['match_info']
        print(f"\n{i}. {match['home_team']} vs {match['away_team']} ({match['league']})")
        print(f"   Selection: {selection['outcome'].replace('_', ' ').title()} @ {selection['odds']:.2f}")
        print(f"   Our Prediction: {selection['probability']:.1%} vs {selection['implied_probability']:.1%} implied")
        print(f"   Edge: +{selection['edge']:.1%} | Expected Value: {selection['expected_value']:.3f}")
        print(f"   Comprehensive Score: {selection['comprehensive_score']:.3f}")
        print(f"   Likelihood Tier: {selection['likelihood_tier']}")
        print(f"   Rank: #{selection['rank']} ({selection['rank_percentile']:.1f}% percentile)")
    
    # Show best accumulator if available
    if accumulators:
        best_acc = accumulators[0]
        print(f"\\n🎯 BEST VALUE ACCUMULATOR ({best_acc['fold_size']}-fold):")
        print(f"   Total Odds: {best_acc['total_odds']:.2f}")
        print(f"   Combined Probability: {best_acc['combined_probability']:.1%}")
        print(f"   Expected Value: {best_acc['expected_value']:.3f}")
        print(f"   Risk Level: {best_acc['risk_level']}")
        
        print("\\n   💰 Potential Returns:")
        for stake in [5, 10, 20]:
            returns = best_acc['potential_returns'][stake]
            profit = returns
            print(f"      £{stake} → £{stake + returns:.2f} (£{profit:.2f} profit)")
        
        print("\\n   📋 Selections:")
        for i, sel in enumerate(best_acc['selections'], 1):
            match = sel['match_info'] 
            print(f"      {i}. {match['home_team']} vs {match['away_team']}")
            print(f"         {sel['outcome'].replace('_', ' ').title()} @ {sel['odds']:.2f} (+{sel['edge']:.1%} edge)")
    
    # Save comprehensive report
    report = {
        'generated_at': '2026-01-20T01:00:00',
        'analysis_type': 'value_betting',
        'total_matches_analyzed': processed,
        'value_opportunities_found': value_bets_found,
        'qualified_value_bets': len(ranked_selections),
        'top_value_selections': ranked_selections[:10],
        'value_accumulators': accumulators[:3],
        'best_single_value_bet': ranked_selections[0] if ranked_selections else None,
        'best_value_accumulator': accumulators[0] if accumulators else None
    }
    
    with open('value_betting_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\\n📄 Detailed report saved to: value_betting_report.json")
    
    return ranked_selections

if __name__ == "__main__":
    create_value_betting_system()