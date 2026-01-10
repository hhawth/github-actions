#!/usr/bin/env python3
from stat_getter import get_fixtures_from_clubelo

def test_fixtures():
    print("🔍 Testing ClubElo fixtures API...")
    df = get_fixtures_from_clubelo()
    
    if not df.empty:
        print(f"✅ Found {len(df)} fixtures from {df['Country'].nunique()} countries")
        print("\n🌍 Country breakdown:")
        print(df['Country'].value_counts())
        print("\n📋 Sample fixtures:")
        print(df[['Country', 'Home', 'Away', 'Home Win', 'Draw', 'Away Win']].head())
    else:
        print("❌ No fixtures found")

if __name__ == "__main__":
    test_fixtures()