#!/usr/bin/env python3
import sys
sys.path.append('/root/git/github-actions')

from stat_getter import get_todays_fixtures

def test_time_sorting():
    print("🕐 Testing Time Sorting...")
    df = get_todays_fixtures()
    
    if df.empty:
        print("❌ No fixtures found")
        return
    
    print("📅 First 10 fixtures by time:")
    print(df[['Time', 'Country', 'Home', 'Away']].head(10).to_string(index=False))
    
    print(f"\n⏰ Time range: {df['Time'].iloc[0]} → {df['Time'].iloc[-1]}")
    print(f"📊 Total fixtures: {len(df)}")

if __name__ == "__main__":
    test_time_sorting()