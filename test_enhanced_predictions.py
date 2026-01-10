#!/usr/bin/env python3
import sys
sys.path.append('/root/git/github-actions')

from stat_getter import get_todays_fixtures, predict_match_score

def test_enhanced_predictions():
    print("🎯 Testing Enhanced Predictions...")
    
    # Get fixtures
    df = get_todays_fixtures()
    if df.empty:
        print("❌ No fixtures found")
        return
    
    print(f"📊 Testing with {len(df)} fixtures")
    
    # Test first 3 predictions
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        home_team = row['Home']
        away_team = row['Away']
        
        print(f"\n⚽ Match: {home_team} vs {away_team}")
        print(f"🕐 Time: {row.get('Time', 'TBD')} | 🏆 League: {row.get('Country', 'Unknown')}")
        
        # Get enhanced prediction
        prediction = predict_match_score(home_team, away_team, df)
        
        print(f"🎯 Predicted Score: {prediction['home_goals']}-{prediction['away_goals']}")
        print(f"📈 Confidence: {prediction['confidence']:.1%}")
        print(f"🏠 Home Win: {prediction['home_win_prob']:.1%}")
        print(f"⚖️ Draw: {prediction['draw_prob']:.1%}")
        print(f"🛣️ Away Win: {prediction['away_win_prob']:.1%}")
        
        if prediction['reasoning']:
            print("🧠 Reasoning:")
            for reason in prediction['reasoning'][:3]:  # Show first 3 reasons
                print(f"   • {reason}")

if __name__ == "__main__":
    test_enhanced_predictions()