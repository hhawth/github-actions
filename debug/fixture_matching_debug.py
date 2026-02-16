#!/usr/bin/env python3
"""
Fixture Matching Debug - Show which fixtures we have and why they're not matching
"""

import json
from datetime import datetime
from improved_ev_betting_model import ImprovedEVBettingModel

def main():
    print("🔍 FIXTURE MATCHING DEBUG")
    print("=" * 50)
    
    # Load Matchbook data
    with open('matchbook_football_events_simplified.json', 'r', encoding='utf-8') as f:
        matchbook_data = json.load(f)
    
    print("🏢 MATCHBOOK FIXTURES:")
    print("-" * 30)
    for i, match_name in enumerate(matchbook_data.keys(), 1):
        print(f"{i}. {match_name}")
    
    # Load API Football fixtures
    today = datetime.now().strftime('%Y-%m-%d')
    fixtures_file = f'api_football_merged_{today}.json'
    
    try:
        with open(fixtures_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        fixtures = data['fixtures']
        upcoming = [f for f in fixtures if f.get('fixture', {}).get('status', {}).get('short') == 'NS']
        
        print(f"\n⚽ API FOOTBALL UPCOMING FIXTURES ({len(upcoming)} total):")
        print("-" * 40)
        
        for i, fixture in enumerate(upcoming[:15], 1):  # Show first 15
            home_team = fixture.get('teams', {}).get('home', {}).get('name', 'Unknown')
            away_team = fixture.get('teams', {}).get('away', {}).get('name', 'Unknown')
            print(f"{i}. {home_team} vs {away_team}")
            
            # Check if this could match any Matchbook fixture
            for mb_name in matchbook_data.keys():
                if (home_team.lower() in mb_name.lower() or away_team.lower() in mb_name.lower() or
                    any(word in mb_name.lower() for word in home_team.lower().split()) or
                    any(word in mb_name.lower() for word in away_team.lower().split())):
                    print(f"   🎯 Potential match: {mb_name}")
                    break
    
    except Exception as e:
        print(f"❌ Could not load API Football fixtures: {e}")
        return
    
    print("\n" + "="*60)
    print("🔍 TESTING MANUAL FIXTURE MATCHING")
    print("="*60)
    
    # Try to manually create some matches for the Matchbook fixtures
    model = ImprovedEVBettingModel()
    
    # Test direct matching
    test_fixtures = [
        ("Anápolis FC", "Vila Nova FC"),
        ("Mixto EC Women", "CR Flamengo Women"),
        ("Anápolis", "Vila Nova"),
        ("Mixto", "Flamengo"),
    ]
    
    for home, away in test_fixtures:
        matchbook_event = model._get_matchbook_odds(home, away)
        if matchbook_event:
            print(f"✅ Found Matchbook data for: {home} vs {away}")
            
            # Extract some odds to show data is available
            match_odds = model._extract_matchbook_market_odds(matchbook_event, 'match_odds')
            if match_odds:
                print(f"   📊 Match Odds: Home {match_odds.get('home')}, Draw {match_odds.get('draw')}, Away {match_odds.get('away')}")
            
            double_chance_odds = model._extract_matchbook_market_odds(matchbook_event, 'double_chance')
            if double_chance_odds:
                print(f"   📊 Double Chance: 1X {double_chance_odds.get('1x')}, 12 {double_chance_odds.get('12')}, X2 {double_chance_odds.get('x2')}")
        else:
            print(f"❌ No Matchbook data found for: {home} vs {away}")

if __name__ == "__main__":
    main()