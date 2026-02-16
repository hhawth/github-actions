#!/usr/bin/env python3
"""
Debug Fixture Status Issue
===========================
Check why no predictions are being generated
"""

import json
from datetime import datetime

def debug_fixture_status():
    """Debug why 0 predictions are being generated"""
    
    print("🔍 DEBUGGING FIXTURE STATUS ISSUE")
    print("="*50)
    
    # Load fixture data
    try:
        with open('api_football_merged_2026-02-16.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract fixtures array
        fixtures_data = data.get('fixtures', [])
        
        print(f"📊 Total fixtures loaded: {len(fixtures_data)}")
        
        # Check fixture statuses
        status_counts = {}
        upcoming_count = 0
        
        for i, fixture in enumerate(fixtures_data):
            if i >= 10:  # Only check first 10
                break
                
            try:
                status = fixture.get('fixture', {}).get('status', {}).get('short', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if status == 'NS':
                    upcoming_count += 1
                    
                print(f"Fixture {i+1}:")
                print(f"   Status: {status}")
                print(f"   Date: {fixture.get('fixture', {}).get('date', 'Unknown')}")
                print(f"   Teams: {fixture.get('teams', {}).get('home', {}).get('name', '?')} vs {fixture.get('teams', {}).get('away', {}).get('name', '?')}")
                print()
                
            except Exception as e:
                print(f"   Error reading fixture {i+1}: {e}")
        
        print("📈 Status Summary (first 10 fixtures):")
        for status, count in sorted(status_counts.items()):
            print(f"   {status}: {count} fixtures")
        
        print(f"🎯 Upcoming fixtures (NS): {upcoming_count}")
        
        # Check for today's date matches
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"📅 Looking for fixtures on {today_str}...")
        
        today_fixtures = 0
        for fixture in fixtures_data:
            fixture_date = fixture.get('fixture', {}).get('date', '')[:10]  # Get YYYY-MM-DD part
            if fixture_date == today_str:
                today_fixtures += 1
                status = fixture.get('fixture', {}).get('status', {}).get('short', 'UNKNOWN')
                print(f"   Today fixture: {fixture.get('teams', {}).get('home', {}).get('name', '?')} vs {fixture.get('teams', {}).get('away', {}).get('name', '?')} - Status: {status}")
        
        print(f"📊 Total fixtures today: {today_fixtures}")
        
    except Exception as e:
        print(f"❌ Error loading fixture data: {e}")

if __name__ == "__main__":
    debug_fixture_status()