#!/usr/bin/env python3
"""
Odds Extraction Debug - Check why real odds aren't being read
"""

import json
from datetime import datetime

def debug_odds_extraction():
    print("🔍 Odds Extraction Debug Analysis")
    print("=" * 50)
    
    # Load merged fixtures  
    today = datetime.now().strftime('%Y-%m-%d')
    fixtures_file = f'api_football_merged_{today}.json'
    
    with open(fixtures_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixtures = data['fixtures']
    upcoming = [f for f in fixtures if f.get('fixture', {}).get('status', {}).get('short') == 'NS']
    
    # Focus on the matches we know have Matchbook data
    matchbook_matches = [
        "Corinthians vs Red Bull Bragantino",
        "Nacional de Patos vs Sousa EC", 
        "Lobos UPNFM vs Club Deportivo Victoria",
        "Once Caldas vs Junior"
    ]
    
    print(f"📊 Found {len(upcoming)} upcoming fixtures")
    print("🎯 Looking for matches that should have both API Football and Matchbook data...")
    
    # Check first upcoming fixture with odds
    for i, fixture in enumerate(upcoming[:10]):
        try:
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name'] 
            match_name = f"{home_team} vs {away_team}"
            
            print(f"\n🏆 Fixture {i+1}: {match_name}")
            
            # Check if this is one of our matched fixtures
            is_matchbook_match = any(mb_match in match_name or match_name in mb_match for mb_match in matchbook_matches)
            if is_matchbook_match:
                print("✅ This is a Matchbook matched fixture!")
            
            # Check odds structure in detail
            if 'odds' not in fixture:
                print("❌ No 'odds' key in fixture")
                continue
                
            odds_data = fixture['odds']
            print(f"📊 Odds data type: {type(odds_data)}")
            print(f"📏 Odds data length: {len(odds_data) if isinstance(odds_data, list) else 'Not a list'}")
            
            if not odds_data:
                print("❌ Empty odds data")
                continue
                
            if isinstance(odds_data, list) and len(odds_data) > 0:
                first_bookmaker = odds_data[0]
                print("📚 First bookmaker structure:")
                print(f"   Type: {type(first_bookmaker)}")
                print(f"   Keys: {list(first_bookmaker.keys()) if isinstance(first_bookmaker, dict) else 'Not a dict'}")
                
                if 'bets' in first_bookmaker:
                    bets = first_bookmaker['bets']
                    print(f"   Bets count: {len(bets)}")
                    
                    for j, bet in enumerate(bets[:3]):  # Check first 3 bets
                        bet_name = bet.get('name', 'Unknown')
                        print(f"   Bet {j+1}: {bet_name}")
                        
                        if bet_name in ['Match Winner', 'Match Result']:
                            print("   ✅ Found Match Winner bet!")
                            values = bet.get('values', [])
                            print(f"   Values count: {len(values)}")
                            
                            for value in values:
                                outcome = value.get('value', 'Unknown')
                                odd_str = value.get('odd', 'Missing')
                                print(f"      {outcome}: {odd_str}")
                                
                                # Test the conversion
                                try:
                                    odd_float = float(odd_str)
                                    print(f"      ✅ Converts to: {odd_float}")
                                except:
                                    print(f"      ❌ Failed to convert '{odd_str}' to float")
                            break
                else:
                    print("❌ No 'bets' in first bookmaker")
                    
            # Now test our extraction function
            print("\n🧪 Testing actual extraction function...")
            
            # Simulate the _extract_odds_features logic
            odds_features = test_extract_odds_features(fixture)
            
            print("📊 Extracted odds features:")
            print(f"   has_odds: {odds_features['has_odds']}")
            print(f"   home_odds: {odds_features['home_odds']}")
            print(f"   draw_odds: {odds_features['draw_odds']}")
            print(f"   away_odds: {odds_features['away_odds']}")
            
            if odds_features['has_odds']:
                print("✅ Odds extraction SUCCESSFUL!")
            else:
                print("❌ Odds extraction FAILED!")
                
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error processing fixture {i+1}: {e}")
            continue

def test_extract_odds_features(fixture):
    """Test version of the odds extraction logic"""
    odds_features = {
        'has_odds': False,
        'home_odds': 3.0,
        'draw_odds': 3.0, 
        'away_odds': 3.0,
    }
    
    try:
        print(f"   🔍 Checking if 'odds' in fixture: {'odds' in fixture}")
        
        if 'odds' in fixture and fixture['odds']:
            odds_list = fixture['odds']
            print(f"   🔍 odds_list type: {type(odds_list)}, length: {len(odds_list) if isinstance(odds_list, list) else 'N/A'}")
            
            if isinstance(odds_list, list) and len(odds_list) > 0:
                odds_data = odds_list[0]
                print(f"   🔍 First bookmaker keys: {list(odds_data.keys()) if isinstance(odds_data, dict) else 'Not a dict'}")
                
                if isinstance(odds_data, dict) and 'bets' in odds_data:
                    bets_list = odds_data['bets']
                    print(f"   🔍 bets_list type: {type(bets_list)}, length: {len(bets_list) if isinstance(bets_list, list) else 'N/A'}")
                    
                    for bet in bets_list:
                        bet_name = bet.get('name', '')
                        print(f"   🔍 Checking bet: '{bet_name}'")
                        
                        if bet_name in ['Match Winner', 'Match Result']:
                            print(f"   ✅ Found matching bet: '{bet_name}'")
                            values = bet.get('values', [])
                            print(f"   🔍 values type: {type(values)}, length: {len(values) if isinstance(values, list) else 'N/A'}")
                            
                            home_odds = None
                            draw_odds = None
                            away_odds = None
                            
                            for value in values:
                                value_name = value.get('value', '')
                                odd_str = value.get('odd', '')
                                print(f"   🔍 Processing value: '{value_name}' = '{odd_str}'")
                                
                                try:
                                    odd_float = float(odd_str)
                                    if value_name == 'Home':
                                        home_odds = odd_float
                                        print(f"   ✅ Set home_odds = {odd_float}")
                                    elif value_name == 'Draw':
                                        draw_odds = odd_float
                                        print(f"   ✅ Set draw_odds = {odd_float}")
                                    elif value_name == 'Away':
                                        away_odds = odd_float
                                        print(f"   ✅ Set away_odds = {odd_float}")
                                except ValueError as ve:
                                    print(f"   ❌ Failed to convert '{odd_str}' to float: {ve}")
                            
                            print(f"   🔍 Final odds: home={home_odds}, draw={draw_odds}, away={away_odds}")
                            
                            if all(x is not None for x in [home_odds, draw_odds, away_odds]):
                                print("   ✅ All odds found, updating features...")
                                odds_features.update({
                                    'has_odds': True,
                                    'home_odds': home_odds,
                                    'draw_odds': draw_odds,
                                    'away_odds': away_odds,
                                })
                                print("   ✅ Features updated successfully!")
                            else:
                                print("   ❌ Missing some odds values")
                            break
                    else:
                        print(f"   ❌ No Match Winner bet found in {len(bets_list)} bets")
                else:
                    print("   ❌ No 'bets' found or not a dict")
            else:
                print("   ❌ odds not a list or empty")
        else:
            print("   ❌ No 'odds' key or empty odds")
            
    except Exception as e:
        print(f"   ❌ Exception in extraction: {e}")
        import traceback
        traceback.print_exc()
        
    return odds_features

if __name__ == "__main__":
    debug_odds_extraction()