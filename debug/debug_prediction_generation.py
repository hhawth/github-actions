#!/usr/bin/env python3
"""
Debug Prediction Generation
===========================
Show why predict_fixture() is failing for all fixtures
"""

import sys
sys.path.append('.')

import json
from improved_ev_betting_model import ImprovedEVBettingModel

def debug_prediction_generation():
    """Debug why predict_fixture is failing"""
    
    print("🔍 DEBUGGING PREDICTION GENERATION")
    print("="*50)
    
    # Initialize EV model
    model = ImprovedEVBettingModel()
    
    # Load fixture data
    with open('api_football_merged_2026-02-16.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixtures_data = data.get('fixtures', [])
    print(f"📊 Total fixtures: {len(fixtures_data)}")
    
    # Find upcoming fixtures
    upcoming_fixtures = []
    for fixture in fixtures_data:
        if fixture.get('fixture', {}).get('status', {}).get('short') == 'NS':
            upcoming_fixtures.append(fixture)
    
    print(f"🎯 Upcoming fixtures (NS): {len(upcoming_fixtures)}")
    
    if len(upcoming_fixtures) == 0:
        print("❌ No upcoming fixtures found!")
        return
    
    # Test prediction generation on first few upcoming fixtures
    print("\n🧪 Testing prediction generation on first 5 upcoming fixtures...")
    
    predictions_generated = 0
    
    for i, fixture in enumerate(upcoming_fixtures[:5]):
        try:
            teams = fixture.get('teams', {})
            home_team = teams.get('home', {}).get('name', 'Unknown')
            away_team = teams.get('away', {}).get('name', 'Unknown')
            
            print(f"\nFixture {i+1}: {home_team} vs {away_team}")
            
            # Try to generate prediction
            prediction = model.predict_fixture(fixture)
            
            if prediction:
                if 'error' not in prediction:
                    predictions_generated += 1
                    print("   ✅ Prediction generated successfully")
                    print(f"   🎯 Home prob: {prediction.get('home_prob', 'N/A')}")
                    print(f"   🎯 Draw prob: {prediction.get('draw_prob', 'N/A')}")
                    print(f"   🎯 Away prob: {prediction.get('away_prob', 'N/A')}")
                else:
                    print(f"   ⚠️ Prediction has error: {prediction.get('error')}")
            else:
                print("   ❌ No prediction returned (None)")
                
        except Exception as e:
            print(f"   💥 Exception in predict_fixture: {str(e)}")
            print(f"   🔍 Exception type: {type(e).__name__}")
            import traceback
            print(f"   📋 Traceback: {traceback.format_exc()}")
    
    print("\n📊 SUMMARY:")
    print(f"   🎯 Upcoming fixtures: {len(upcoming_fixtures)}")
    print(f"   ✅ Predictions generated: {predictions_generated}")
    print(f"   ❌ Failed predictions: {5 - predictions_generated}")
    
    if predictions_generated == 0:
        print("\n🚨 CRITICAL: predict_fixture() is failing for all fixtures!")
        print("💡 This explains why the workflow generates 0 predictions")
    else:
        print("\n✅ predict_fixture() works, issue may be elsewhere")

if __name__ == "__main__":
    debug_prediction_generation()