import json
from accumulator_betting_model_fixed import AccumulatorBettingModel

def debug_merged_data():
    """Debug what's happening with merged data processing"""
    
    # Load merged data
    with open('merged_match_data.json', 'r') as f:
        merged_data = json.load(f)
    
    print(f"Total matches in merged data: {len(merged_data)}")
    
    # Look at first few matches
    for i, match in enumerate(merged_data[:3]):
        print(f"\\nMatch {i+1}:")
        print(f"  Type: {type(match)}")
        print(f"  Keys: {list(match.keys()) if isinstance(match, dict) else 'Not a dict'}")
        
        if isinstance(match, dict):
            print(f"  Home team: {match.get('home_team', 'Missing')}")
            print(f"  Away team: {match.get('away_team', 'Missing')}")
            print(f"  League: {match.get('league', 'Missing')}")
            
            # Check for odds
            odds_fields = [k for k in match.keys() if 'odds' in k.lower()]
            print(f"  Odds fields: {odds_fields}")
            
            for field in odds_fields:
                print(f"    {field}: {match[field]}")

def create_simple_betting_selections():
    """Create betting selections manually from merged data"""
    
    with open('merged_match_data.json', 'r') as f:
        merged_data = json.load(f)
    
    betting_model = AccumulatorBettingModel()
    selections = []
    
    for match in merged_data[:10]:  # Test with first 10 matches
        if not isinstance(match, dict):
            continue
            
        # Check if match has odds
        has_odds = any('odds' in k.lower() for k in match.keys())
        
        if has_odds:
            print(f"\\nProcessing: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")
            
            # Extract home win odds
            home_odds = None
            for field in ['home_odds', 'home_win_odds', '1_odds']:
                if field in match and match[field]:
                    try:
                        home_odds = float(match[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if home_odds and home_odds > 1.0:
                implied_prob = 1.0 / home_odds
                
                selection = {
                    'outcome': 'home_win',
                    'probability': implied_prob,
                    'odds': home_odds,
                    'confidence': 0.4,  # Odds-only confidence
                    'source': 'odds_only',
                    'expected_value': (implied_prob * home_odds) - 1,
                    'risk_score': betting_model._calculate_risk_score(implied_prob, 0.4, home_odds),
                    'match_info': {
                        'home_team': match.get('home_team', 'Unknown'),
                        'away_team': match.get('away_team', 'Unknown'),
                        'league': match.get('league', 'Unknown'),
                        'time': match.get('time', 'Unknown'),
                        'date': match.get('date', 'Unknown')
                    }
                }
                
                print(f"  Created selection: {selection['outcome']} @ {selection['odds']:.2f} (EV: {selection['expected_value']:.3f})")
                
                # Filter by betting criteria
                if (selection['probability'] >= betting_model.min_probability and 
                    selection['odds'] <= betting_model.max_odds):
                    selections.append(selection)
                    print("  ✓ Qualifies for betting")
                else:
                    print(f"  ✗ Doesn't meet criteria (prob: {selection['probability']:.1%}, odds: {selection['odds']:.2f})")
    
    print(f"\\nTotal qualifying selections: {len(selections)}")
    
    if selections:
        # Test accumulator building
        ranked_selections = betting_model.rank_selections(selections)
        accumulators = betting_model.generate_multiple_accumulators(ranked_selections, max_fold=4)
        
        print(f"Ranked selections: {len(ranked_selections)}")
        print(f"Accumulators generated: {len(accumulators)}")
        
        if accumulators:
            best_acc = accumulators[0]
            print(f"\\nBest accumulator ({best_acc['fold_size']}-fold):")
            print(f"  Total odds: {best_acc['total_odds']:.2f}")
            print(f"  Win probability: {best_acc['combined_probability']:.1%}")
            print(f"  Expected value: {best_acc['expected_value']:.3f}")
            print(f"  £10 stake → £{best_acc['potential_returns'][10]:.2f} return")

if __name__ == "__main__":
    print("=== DEBUGGING MERGED DATA ===")
    debug_merged_data()
    
    print("\\n\\n=== CREATING SIMPLE BETTING SELECTIONS ===")
    create_simple_betting_selections()