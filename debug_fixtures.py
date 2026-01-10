#!/usr/bin/env python3
"""
Debug script to check why fixtures/predictions aren't showing up
"""

import traceback
import pandas as pd

def debug_fixtures():
    print("🔍 Debugging Fixtures and Predictions...")
    print("=" * 60)
    
    # Test 1: Check ClubElo fixtures
    print("\n1️⃣ Testing ClubElo Fixtures API:")
    try:
        from stat_getter import get_fixtures_from_clubelo
        fixtures_df = get_fixtures_from_clubelo()
        print(f"   ✅ ClubElo fixtures: {len(fixtures_df)} total fixtures")
        
        if not fixtures_df.empty:
            print("   📋 Sample fixtures:")
            print(fixtures_df.head(3)[['Home', 'Away']].to_string(index=False))
            
            # Check for English fixtures
            if 'Country' in fixtures_df.columns:
                eng_fixtures = fixtures_df[fixtures_df['Country'] == 'ENG']
                print(f"   🏴󠁧󠁢󠁥󠁮󠁧󠁿 English fixtures: {len(eng_fixtures)}")
            else:
                print("   ⚠️ No 'Country' column found")
        else:
            print("   ⚠️ No fixtures returned from ClubElo")
            
    except Exception as e:
        print(f"   ❌ Error with ClubElo: {e}")
        traceback.print_exc()
    
    # Test 2: Check team mapping
    print("\n2️⃣ Testing Team Mapping:")
    try:
        from stat_getter import get_official_team_names, get_soccerstats_team_mapping
        
        official_teams = get_official_team_names()
        team_mapping = get_soccerstats_team_mapping()
        
        print(f"   ✅ Official teams: {len(official_teams)}")
        print(f"   ✅ Team mapping: {len(team_mapping)} mappings")
        print(f"   📝 Sample teams: {list(official_teams)[:5]}")
        
    except Exception as e:
        print(f"   ❌ Error with team mapping: {e}")
        traceback.print_exc()
    
    # Test 3: Check get_fixtures function
    print("\n3️⃣ Testing get_fixtures() function:")
    try:
        from get_fixtures import get_fixtures
        fixtures = get_fixtures()
        
        print(f"   ✅ get_fixtures() returned: {len(fixtures)} fixtures")
        
        if fixtures:
            print("   📋 Sample fixture keys:")
            for i, key in enumerate(list(fixtures.keys())[:3]):
                print(f"      {i+1}. {key}")
            
            # Check a sample fixture
            sample_fixture = list(fixtures.values())[0]
            print(f"   📊 Sample fixture data keys: {list(sample_fixture.keys())}")
        else:
            print("   ⚠️ No fixtures returned from get_fixtures()")
            
    except Exception as e:
        print(f"   ❌ Error with get_fixtures: {e}")
        traceback.print_exc()
    
    # Test 4: Check goal stats
    print("\n4️⃣ Testing Goal Statistics:")
    try:
        from stat_getter import get_stats
        goal_stats = get_stats()
        
        print(f"   ✅ Goal stats for {len(goal_stats)} teams")
        
        if goal_stats:
            sample_team = list(goal_stats.keys())[0]
            print(f"   📊 Sample team ({sample_team}) data:")
            team_data = goal_stats[sample_team]
            if 'home' in team_data:
                print(f"      Home goals: {team_data['home']['goals_for'][:3]}...")
                
    except Exception as e:
        print(f"   ❌ Error with goal stats: {e}")
        traceback.print_exc()
    
    # Test 5: Check if ClubElo has current season data
    print("\n5️⃣ Checking ClubElo Data Currency:")
    try:
        import requests
        from io import StringIO
        
        response = requests.get("http://api.clubelo.com/Fixtures")
        df = pd.read_csv(StringIO(response.text))
        
        print(f"   📅 Total fixtures in API: {len(df)}")
        
        if 'Date' in df.columns:
            print(f"   📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        if 'Country' in df.columns:
            countries = df['Country'].value_counts()
            print(f"   🌍 Countries: {dict(countries.head())}")
            
        eng_fixtures = df[df['Country'] == 'ENG'] if 'Country' in df.columns else pd.DataFrame()
        print(f"   🏴󠁧󠁢󠁥󠁮󠁧󠁿 English fixtures: {len(eng_fixtures)}")
        
        if not eng_fixtures.empty and 'Date' in eng_fixtures.columns:
            print(f"   📅 English fixture dates: {eng_fixtures['Date'].min()} to {eng_fixtures['Date'].max()}")
            
    except Exception as e:
        print(f"   ❌ Error checking ClubElo data: {e}")

if __name__ == "__main__":
    debug_fixtures()