#!/usr/bin/env python3
"""
Test script for stat_getter.py team mapping functionality
"""

import sys
import traceback
from stat_getter import (
    get_official_team_names, 
    create_team_mapping, 
    get_soccerstats_team_mapping,
    get_stats,
    get_form,
    get_top_scorers,
    get_fixtures_from_clubelo,
    get_rankings_from_clubelo
)

def test_official_team_names():
    """Test fetching official team names from FPL API"""
    print("Testing get_official_team_names()...")
    try:
        teams = get_official_team_names()
        print(f"✅ Successfully fetched {len(teams)} official team names")
        print("Official teams:", teams[:5], "..." if len(teams) > 5 else "")
        return True
    except Exception as e:
        print(f"❌ Error fetching official team names: {e}")
        traceback.print_exc()
        return False

def test_team_mapping():
    """Test team mapping functionality"""
    print("\nTesting create_team_mapping()...")
    try:
        official_teams = get_official_team_names()
        test_source_teams = ["Man City", "Leeds Utd", "Arsenal", "Wolverhampton", "Brighton"]
        
        mapping = create_team_mapping(test_source_teams, official_teams)
        print("✅ Team mapping created successfully")
        print("Sample mappings:")
        for source, official in mapping.items():
            print(f"  {source} → {official}")
        return True
    except Exception as e:
        print(f"❌ Error creating team mapping: {e}")
        traceback.print_exc()
        return False

def test_soccerstats_mapping():
    """Test SoccerStats team mapping"""
    print("\nTesting get_soccerstats_team_mapping()...")
    try:
        mapping = get_soccerstats_team_mapping()
        print("✅ SoccerStats team mapping created successfully")
        print(f"Mapped {len(mapping)} teams")
        print("Sample mappings:")
        for i, (source, official) in enumerate(mapping.items()):
            if i < 5:
                print(f"  {source} → {official}")
        return True
    except Exception as e:
        print(f"❌ Error creating SoccerStats mapping: {e}")
        traceback.print_exc()
        return False

def test_get_stats():
    """Test the get_stats function with new team mapping"""
    print("\nTesting get_stats()...")
    try:
        stats = get_stats()
        print("✅ Successfully fetched goal statistics")
        print(f"Retrieved stats for {len(stats)} teams")
        
        # Show a sample team's data
        if stats:
            sample_team = list(stats.keys())[0]
            print(f"Sample data for {sample_team}:")
            if 'home' in stats[sample_team]:
                print(f"  Home goals for: {stats[sample_team]['home']['goals_for'][:3]}...")
            if 'away' in stats[sample_team]:
                print(f"  Away goals for: {stats[sample_team]['away']['goals_for'][:3]}...")
        return True
    except Exception as e:
        print(f"❌ Error fetching stats: {e}")
        traceback.print_exc()
        return False

def test_get_form():
    """Test the get_form function"""
    print("\nTesting get_form()...")
    try:
        form_data = get_form()
        print("✅ Successfully fetched form data")
        print(f"Retrieved form for {len(form_data)} teams")
        
        # Show sample data
        if form_data:
            sample_items = list(form_data.items())[:3]
            print("Sample form data:")
            for team, form in sample_items:
                print(f"  {team}: {form}")
        return True
    except Exception as e:
        print(f"❌ Error fetching form data: {e}")
        traceback.print_exc()
        return False

def test_get_top_scorers():
    """Test the get_top_scorers function"""
    print("\nTesting get_top_scorers()...")
    try:
        scorers = get_top_scorers()
        print("✅ Successfully fetched top scorers")
        print(f"Retrieved data for {len(scorers)} players")
        
        # Show sample data
        if not scorers.empty:
            print("Sample top scorers:")
            print(scorers.head(3)[['first_name', 'second_name', 'team_name', 'goals_scored']].to_string(index=False))
        return True
    except Exception as e:
        print(f"❌ Error fetching top scorers: {e}")
        traceback.print_exc()
        return False

def test_get_clubelo_fixtures():
    """Test the ClubElo fixtures function"""
    print("\nTesting get_fixtures_from_clubelo()...")
    try:
        fixtures = get_fixtures_from_clubelo()
        print("✅ Successfully fetched ClubElo fixtures")
        print(f"Retrieved {len(fixtures)} fixtures")
        
        # Show sample data and check team names
        if not fixtures.empty:
            print("Sample fixtures with team names:")
            sample_fixtures = fixtures[['Home', 'Away', 'Home Win', 'Draw', 'Away Win']].head(3)
            print(sample_fixtures.to_string(index=False))
            
            # Check if team names look like official names
            unique_teams = set(fixtures['Home'].tolist() + fixtures['Away'].tolist())
            print(f"Unique teams found: {len(unique_teams)}")
            print("Sample team names:", list(unique_teams)[:5])
        return True
    except Exception as e:
        print(f"❌ Error fetching ClubElo fixtures: {e}")
        traceback.print_exc()
        return False

def test_get_clubelo_rankings():
    """Test the ClubElo rankings function"""
    print("\nTesting get_rankings_from_clubelo()...")
    try:
        rankings = get_rankings_from_clubelo()
        print("✅ Successfully fetched ClubElo rankings")
        print(f"Retrieved rankings for {len(rankings)} clubs")
        
        # Show sample data and check team names
        if not rankings.empty:
            # Filter for English teams if possible
            eng_teams = rankings[rankings.get('Country', '') == 'ENG'] if 'Country' in rankings.columns else rankings
            if len(eng_teams) > 0:
                print("Sample English club rankings:")
                sample_rankings = eng_teams[['Club', 'Elo']].head(5) if 'Elo' in eng_teams.columns else eng_teams[['Club']].head(5)
                print(sample_rankings.to_string(index=False))
            else:
                print("Sample club rankings:")
                sample_rankings = rankings[['Club']].head(5) if 'Club' in rankings.columns else rankings.head(5)
                print(sample_rankings.to_string(index=False))
        return True
    except Exception as e:
        print(f"❌ Error fetching ClubElo rankings: {e}")
        traceback.print_exc()
        return False

def test_cross_source_consistency():
    """Test that team names are consistent across all data sources"""
    print("\nTesting cross-source team name consistency...")
    try:
        # Get team names from each source
        official_teams = set(get_official_team_names())
        
        stats_data = get_stats()
        stats_teams = set(stats_data.keys())
        
        form_data = get_form()
        form_teams = set(form_data.keys())
        
        scorers_data = get_top_scorers()
        scorer_teams = set(scorers_data['team_name'].unique()) if not scorers_data.empty else set()
        
        try:
            fixtures_data = get_fixtures_from_clubelo()
            clubelo_teams = set(fixtures_data['Home'].tolist() + fixtures_data['Away'].tolist()) if not fixtures_data.empty else set()
        except:
            clubelo_teams = set()
            print("⚠️  ClubElo fixtures not available, skipping...")
        
        print("✅ Team name consistency check:")
        print(f"Official FPL teams: {len(official_teams)}")
        print(f"Stats teams: {len(stats_teams)}")
        print(f"Form teams: {len(form_teams)}")
        print(f"Scorer teams: {len(scorer_teams)}")
        print(f"ClubElo teams: {len(clubelo_teams)}")
        
        # Check for consistency
        all_consistent = True
        
        if stats_teams - official_teams:
            print(f"❌ Stats has extra teams: {stats_teams - official_teams}")
            all_consistent = False
        
        if form_teams - official_teams:
            print(f"❌ Form has extra teams: {form_teams - official_teams}")
            all_consistent = False
            
        if scorer_teams - official_teams:
            print(f"❌ Scorers has extra teams: {scorer_teams - official_teams}")
            all_consistent = False
        
        # ClubElo might have teams not in current Premier League
        prem_clubelo = clubelo_teams & official_teams
        if len(prem_clubelo) > 0:
            print(f"✅ ClubElo has {len(prem_clubelo)} Premier League teams with matching names")
        
        if all_consistent:
            print("✅ All data sources use consistent team names!")
        
        return all_consistent
    except Exception as e:
        print(f"❌ Error checking consistency: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing stat_getter.py team mapping system\n")
    print("=" * 60)
    
    tests = [
        test_official_team_names,
        test_team_mapping,
        test_soccerstats_mapping,
        test_get_stats,
        test_get_form,
        test_get_top_scorers,
        test_get_clubelo_fixtures,
        test_get_clubelo_rankings,
        test_cross_source_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Team mapping system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)