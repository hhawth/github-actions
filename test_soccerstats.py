#!/usr/bin/env python3
import sys
sys.path.append('/root/git/github-actions')

from stat_getter import get_fixtures_from_soccerstats

def test_soccerstats_scraping():
    print("🔍 Testing SoccerStats.com scraping...")
    df = get_fixtures_from_soccerstats()
    
    if not df.empty:
        print(f"✅ Found {len(df)} fixtures")
        print(f"🌍 Countries: {df['Country'].unique()}")
        print(f"📊 Country breakdown: {dict(df['Country'].value_counts())}")
        print("\n📋 Sample fixtures:")
        sample_columns = ['Country', 'League', 'Time', 'Home', 'Away', 'Home Win', 'Draw', 'Away Win']
        available_columns = [col for col in sample_columns if col in df.columns]
        print(df[available_columns].head(10))
    else:
        print("❌ No fixtures found")

if __name__ == "__main__":
    test_soccerstats_scraping()