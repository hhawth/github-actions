#!/usr/bin/env python3
"""
Debug Smart Team Name Matching System with detailed logging
"""

import json
from improved_ev_betting_model import ImprovedEVBettingModel

def debug_smart_matching():
    print("🔬 DEBUG SMART TEAM NAME MATCHING")
    print("=" * 50)
    
    # Initialize model
    model = ImprovedEVBettingModel()
    
    # Load Matchbook data
    with open('matchbook_football_events_simplified.json', 'r', encoding='utf-8') as f:
        matchbook_data = json.load(f)
    
    model.matchbook_data = matchbook_data
    
    # Test case: Mingəçevir vs Zaqatala -> Mingachevir vs FK Zaqatala  
    test_case = {
        'home_team': 'Mingəçevir',
        'away_team': 'Zaqatala',
        'matchbook_target': 'Mingachevir vs FK Zaqatala'
    }
    
    # Create a simple prediction
    predictions_lookup = {
        f"{test_case['home_team']} vs {test_case['away_team']}": {
            'fixture_id': 'test_1',
            'home_team': test_case['home_team'],
            'away_team': test_case['away_team'],
            'timestamp': 1770940800,  # Same as Matchbook for time matching
            'predictions': {
                'double_chance': {
                    '1x_prob': 0.6,
                    'x2_prob': 0.4,
                    '12_prob': 0.7
                },
                'goals': {
                    'over_25_prob': 0.5,
                    'under_25_prob': 0.5
                }
            }
        }
    }
    
    matchbook_name = test_case['matchbook_target']
    print(f"🎯 Testing: {matchbook_name}")
    print(f"   Looking for: {test_case['home_team']} vs {test_case['away_team']}")
    
    # Manual step-by-step debugging
    matchbook_event = model.matchbook_data.get(matchbook_name)
    if not matchbook_event:
        print(f"   ❌ No Matchbook event found for {matchbook_name}")
        return
        
    print("   ✅ Found Matchbook event")
    
    # Parse matchbook fixture name 
    if ' vs ' not in matchbook_name:
        print("   ❌ Invalid matchbook name format")
        return
        
    mb_home, mb_away = matchbook_name.split(' vs ', 1)
    print(f"   🏠 Matchbook Home: '{mb_home}'")
    print(f"   🛫 Matchbook Away: '{mb_away}'")
    
    matchbook_time = matchbook_event.get('start', '')
    print(f"   ⏰ Matchbook Time: {matchbook_time}")
    
    # Test time matching
    time_matches = model._find_predictions_by_time(matchbook_time, predictions_lookup)
    print(f"   🔢 Time matches found: {len(time_matches)}")
    
    if not time_matches:
        print("   ❌ No time matches - checking prediction timestamps")
        for pred_name, pred in predictions_lookup.items():
            print(f"      Prediction '{pred_name}' has timestamp: {pred.get('timestamp')}")
        return
    
    # Test similarity for the time match
    for pred in time_matches:
        pred_name = f"{pred['home_team']} vs {pred['away_team']}"
        print(f"\n   🔍 Testing prediction: {pred_name}")
        
        if ' vs ' not in pred_name:
            print("      ❌ Invalid prediction name format")
            continue
            
        pred_home, pred_away = pred_name.split(' vs ', 1)
        print(f"      🏠 Prediction Home: '{pred_home}'")
        print(f"      🛫 Prediction Away: '{pred_away}'")
        
        # Calculate normalized similarity scores
        home_score = model._normalized_team_similarity(mb_home, pred_home)
        away_score = model._normalized_team_similarity(mb_away, pred_away)
        
        # Also try reversed order
        home_score_rev = model._normalized_team_similarity(mb_home, pred_away)  
        away_score_rev = model._normalized_team_similarity(mb_away, pred_home)
        
        print(f"      📊 Home Score: {home_score:.3f}")
        print(f"      📊 Away Score: {away_score:.3f}") 
        print(f"      📊 Home Score (Rev): {home_score_rev:.3f}")
        print(f"      📊 Away Score (Rev): {away_score_rev:.3f}")
        
        total_score = max(home_score + away_score, home_score_rev + away_score_rev)
        print(f"      🎯 Total Score: {total_score:.3f}")
        
        min_constraint = min(max(home_score, home_score_rev), max(away_score, away_score_rev))
        print(f"      ⚖️ Min Constraint: {min_constraint:.3f} (needs >= 0.5)")
        
        meets_total_threshold = total_score >= 1.0
        meets_min_constraint = min_constraint >= 0.5
        
        print(f"      ✅ Meets total threshold (>= 1.0): {meets_total_threshold}")
        print(f"      ✅ Meets min constraint (>= 0.5): {meets_min_constraint}")
        
        if meets_total_threshold and meets_min_constraint:
            print("      🎉 MATCH SHOULD BE FOUND!")
        else:
            print("      ❌ MATCH CRITERIA NOT MET")

if __name__ == "__main__":
    debug_smart_matching()