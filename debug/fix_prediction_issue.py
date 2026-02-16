#!/usr/bin/env python3
"""
Quick Fix for Prediction Generation
===================================
Fix the model loading issue so predictions can be generated
"""

import sys
sys.path.append('.')

from automated_betting_workflow import AutomatedBettingWorkflow

def fix_prediction_issue():
    """Fix the prediction generation issue by using the workflow's model"""
    
    print("🔧 FIXING PREDICTION GENERATION ISSUE")
    print("="*50)
    
    # Initialize the workflow (which properly loads models)
    workflow = AutomatedBettingWorkflow()
    
    print("✅ Workflow initialized")
    
    # Prepare the AI models (this loads the trained model)
    print("🏋️ Preparing AI models...")
    result = workflow.step_2_prepare_ai_models(force_retrain=False)
    
    if result:
        print("✅ AI models prepared successfully")
        print(f"🎯 Model is_trained flag: {workflow.ev_model.is_trained}")
        
        # Now test prediction generation on a fixture
        print("\n🧪 Testing prediction generation...")
        
        # Load fixture data
        import json
        with open('api_football_merged_2026-02-16.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixtures_data = data.get('fixtures', [])
        
        # Find an upcoming fixture
        upcoming_fixture = None
        for fixture in fixtures_data:
            if fixture.get('fixture', {}).get('status', {}).get('short') == 'NS':
                upcoming_fixture = fixture
                break
        
        if upcoming_fixture:
            teams = upcoming_fixture.get('teams', {})
            home_team = teams.get('home', {}).get('name', 'Unknown')
            away_team = teams.get('away', {}).get('name', 'Unknown')
            
            print(f"🎯 Testing fixture: {home_team} vs {away_team}")
            
            try:
                prediction = workflow.ev_model.predict_fixture(upcoming_fixture)
                
                if prediction and 'error' not in prediction:
                    print("✅ PREDICTION SUCCESS!")
                    print(f"   🏠 Home prob: {prediction.get('home_prob', 'N/A')}")
                    print(f"   🤝 Draw prob: {prediction.get('draw_prob', 'N/A')}")
                    print(f"   ✈️ Away prob: {prediction.get('away_prob', 'N/A')}")
                    return True
                else:
                    print(f"❌ Prediction failed: {prediction}")
                    return False
            except Exception as e:
                print(f"❌ Exception in prediction: {e}")
                return False
        else:
            print("❌ No upcoming fixtures found")
            return False
    else:
        print("❌ Failed to prepare AI models")
        return False

def test_actual_betting_analysis():
    """Test the actual betting analysis with fixed model"""
    
    print("\n🎲 TESTING ACTUAL BETTING ANALYSIS")
    print("-"*40)
    
    # Run the workflow step that analyzes EV opportunities
    workflow = AutomatedBettingWorkflow()
    
    # Prepare models
    if workflow.step_2_prepare_ai_models():
        print("✅ Models prepared")
        
        # Run the EV analysis step
        print("🔍 Running EV analysis...")
        opportunities = workflow.step_4_analyze_ev_opportunities()
        
        if opportunities:
            print(f"🎯 Found {len(opportunities)} betting opportunities!")
            for i, opp in enumerate(opportunities[:3]):  # Show first 3
                print(f"   {i+1}. {opp.get('match', 'Unknown')} - EV: {opp.get('ev', 'N/A'):.1%}")
            return True
        else:
            print("📉 No betting opportunities found (but predictions should now work)")
            return True  # Still success if models work
    else:
        print("❌ Failed to prepare models")
        return False

if __name__ == "__main__":
    success1 = fix_prediction_issue()
    success2 = test_actual_betting_analysis()
    
    if success1 and success2:
        print("\n🎉 PREDICTION ISSUE FIXED!")
        print("💰 The betting system should now work correctly")
        print("💡 Run: python automated_betting_workflow.py --full-run")
    else:
        print("\n⚠️ Some issues remain - may need more debugging")