#!/usr/bin/env python3
import sys
sys.path.append('/root/git/github-actions')

from stat_getter import get_todays_fixtures

def test_combined_fixtures():
    print("🔍 Testing combined fixture sources...")
    df = get_todays_fixtures()
    
    if not df.empty:
        print(f"✅ Found {len(df)} total fixtures")
        print(f"🌍 Countries: {list(df['Country'].unique())}")
        print("📊 Country breakdown:")
        for country, count in df['Country'].value_counts().items():
            print(f"   {country}: {count} fixtures")
        
        print("\n📋 Sample fixtures by country:")
        for country in df['Country'].unique()[:5]:  # Show top 5 countries
            country_fixtures = df[df['Country'] == country]
            print(f"\n🇺🇳 {country} ({len(country_fixtures)} fixtures):")
            if 'Time' in country_fixtures.columns:
                sample_cols = ['Time', 'Home', 'Away', 'Home Win', 'Draw', 'Away Win']
            else:
                sample_cols = ['Home', 'Away', 'Home Win', 'Draw', 'Away Win']
            available_cols = [col for col in sample_cols if col in country_fixtures.columns]
            print(country_fixtures[available_cols].head(3))
    else:
        print("❌ No fixtures found")

if __name__ == "__main__":
    test_combined_fixtures()