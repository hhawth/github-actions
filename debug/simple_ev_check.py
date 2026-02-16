#!/usr/bin/env python3
"""
Simple EV Values Checker - Shows what EVs are actually being calculated
"""

import json
from datetime import datetime
from automated_betting_workflow import AutomatedBettingWorkflow

def main():
    print("🔍 EV Values Analysis")
    print("=" * 40)
    
    # Create workflow instance
    workflow = AutomatedBettingWorkflow()
    workflow.config = {
        'min_ev': 0.001,  # 0.1% - extremely low threshold
        'stake': 0.10, 
        'daily_limit': 2.0,
        'auto_place': False
    }
    
    # Load model
    print("📈 Loading improved model...")
    if not workflow.step_2_prepare_ai_models(use_enhanced_ai=True):
        print("❌ Failed to load model")
        return
    
    # Load fixtures
    today = datetime.now().strftime('%Y-%m-%d')
    fixtures_file = f'api_football_merged_{today}.json'
    
    with open(fixtures_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixtures = data['fixtures']
    upcoming = [f for f in fixtures if f.get('fixture', {}).get('status', {}).get('short') == 'NS']
    
    print(f"📊 Analyzing {len(upcoming)} upcoming fixtures...")
    print("🎯 Using extremely low EV threshold: 0.1%")
    print("\n" + "="*60)
    
    # Get a few predictions to examine
    predictions = []
    for fixture in upcoming[:5]:  # Just analyze 5 fixtures
        try:
            prediction = workflow.ev_model.predict_fixture(fixture)
            if prediction and 'error' not in prediction:
                predictions.append(prediction)
        except Exception:
            continue
    
    if not predictions:
        print("❌ No predictions generated")
        return
        
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Now try to find value bets with extremely low threshold
    print("\n🔍 Checking for ANY positive EV opportunities...")
    
    # Temporarily modify model threshold
    original_min_ev = workflow.ev_model.config['min_ev']
    workflow.ev_model.config['min_ev'] = 0.001  # 0.1%
    
    opportunities = workflow.ev_model.find_value_bets(predictions)
    
    if opportunities:
        print(f"✅ Found {len(opportunities)} opportunities with 0.1% threshold!")
        for i, opp in enumerate(opportunities[:3]):
            print(f"\n🎯 Opportunity {i+1}:")
            print(f"   🏆 {opp['home_team']} vs {opp['away_team']}")
            print(f"   📊 Market: {opp['market']}")
            print(f"   📈 EV: {opp['expected_value']:+.4f} ({opp['roi_pct']:+.2f}%)")
            print(f"   🎲 Probability: {opp['prediction_prob']:.1%}")
            print(f"   💰 Estimated Odds: {opp['odds']:.2f}")
    else:
        print("❌ NO opportunities found even at 0.1% threshold!")
        print("\n🧐 This means:")
        print("   • All calculated EVs are negative or zero")
        print("   • Markets are priced efficiently or better than model predictions")
        print("   • Model may be too conservative")
        
    # Restore original threshold
    workflow.ev_model.config['min_ev'] = original_min_ev
    
    # Show prediction sample
    if predictions:
        print("\n📊 SAMPLE PREDICTION DATA:")
        print("-" * 40)
        pred = predictions[0]
        print(f"🏆 {pred['home_team']} vs {pred['away_team']}")
        
        dc = pred['predictions']['double_chance']
        print("Double Chance Probabilities:")
        print(f"   1X (Home/Draw): {dc['1x_prob']:.1%}")
        print(f"   X2 (Draw/Away): {dc['x2_prob']:.1%}")
        print(f"   12 (No Draw): {dc['12_prob']:.1%}")
        
        btts = pred['predictions']['btts']
        print(f"BTTS Probability: {btts['probability']:.1%}")
        
        goals = pred['predictions']['goals']
        print("Goals Predictions:")
        print(f"   Expected Total: {goals['expected_total']:.2f}")
        print(f"   Over 2.5: {goals['over_25_prob']:.1%}")
        print(f"   Under 2.5: {goals['under_25_prob']:.1%}")
        
        odds_data = pred['odds']
        print("Model's Estimated Odds:")
        print(f"   Home: {odds_data['home_odds']:.2f}")
        print(f"   Draw: {odds_data['draw_odds']:.2f}")
        print(f"   Away: {odds_data['away_odds']:.2f}")

if __name__ == "__main__":
    main()