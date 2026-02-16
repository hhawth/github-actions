#!/usr/bin/env python3
"""
Diagnose why no bets are getting through the filter
"""

import glob
import json
from improved_ev_betting_model import ImprovedEVBettingModel
from market_filter import should_bet
from improved_ev_calculator import ImprovedEVCalculator

def diagnose_market_availability():
    """Check what markets are actually being generated"""
    
    print("🔍 DIAGNOSING MARKET AVAILABILITY")
    print("="*50)
    
    # Load the EV model to see what it predicts
    model = ImprovedEVBettingModel()
    
    # Try to find the latest model
    model_files = glob.glob("*ev_model*.pkl") + ["improved_ev_model.pkl"]
    if model_files:
        latest_model = sorted(model_files, key=lambda x: x)[-1]
        print(f"📈 Loading model: {latest_model}")
        try:
            model.load_model(latest_model)
            print("✅ Model loaded successfully")
        except:
            print("❌ Could not load model")
            return
    else:
        print("❌ No model files found")
        return
    
    # Load recent API Football data
    data_files = glob.glob("api_football_merged_*.json")
    if not data_files:
        print("❌ No API Football data found")
        return
    
    latest_data = sorted(data_files)[-1]
    print(f"📊 Using data: {latest_data}")
    
    with open(latest_data, 'r', encoding='utf-8') as f:
        fixtures = json.load(f)
    
    print(f"📅 Analyzing {len(fixtures)} fixtures...")
    
    # Analyze market types being generated
    market_counts = {}
    btts_opportunities = []
    double_chance_opportunities = []
    
    calc = ImprovedEVCalculator()
    
    for i, fixture in enumerate(fixtures[:10]):  # Check first 10 fixtures
        if i >= 10:  # Limit for diagnostic
            break
            
        print(f"\n🏆 {fixture.get('teams', {}).get('home', {}).get('name', 'Unknown')} vs {fixture.get('teams', {}).get('away', {}).get('name', 'Unknown')}")
        
        try:
            # Get model predictions for this fixture
            predictions = model.predict_fixture(fixture)
            
            if not predictions:
                print("   ⚠️ No predictions generated")
                continue
                
            print(f"   📊 Generated {len(predictions)} predictions:")
            
            for pred in predictions:
                market = pred.get('market', 'Unknown')
                prob = pred.get('probability', 0)
                odds = pred.get('odds', 0)
                
                market_counts[market] = market_counts.get(market, 0) + 1
                print(f"      • {market}: {prob:.1%} @ {odds:.2f}")
                
                # Test if this would pass our filters
                if 'both teams to score' in market.lower() or 'btts' in market.lower():
                    
                    # Calculate with enhanced EV
                    ev_result = calc.calculate_expected_value(market, prob, odds)
                    should_place, reason = should_bet(market, ev_result.get('calibrated_edge', ev_result['edge']), ev_result.get('calibrated_probability', ev_result['adjusted_probability']))
                    
                    btts_opportunities.append({
                        'fixture': f"{fixture.get('teams', {}).get('home', {}).get('name', 'Unknown')} vs {fixture.get('teams', {}).get('away', {}).get('name', 'Unknown')}",
                        'market': market,
                        'probability': prob,
                        'calibrated_prob': ev_result.get('calibrated_probability', ev_result['adjusted_probability']),
                        'edge': ev_result.get('calibrated_edge', ev_result['edge']),
                        'should_bet': should_place,
                        'reason': reason
                    })
                    
                elif any(dc in market.lower() for dc in ['1x', 'x2', '12', 'double chance']):
                    
                    # Calculate with enhanced EV
                    ev_result = calc.calculate_expected_value(market, prob, odds)
                    should_place, reason = should_bet(market, ev_result.get('calibrated_edge', ev_result['edge']), ev_result.get('calibrated_probability', ev_result['adjusted_probability']))
                    
                    double_chance_opportunities.append({
                        'fixture': f"{fixture.get('teams', {}).get('home', {}).get('name', 'Unknown')} vs {fixture.get('teams', {}).get('away', {}).get('name', 'Unknown')}",
                        'market': market,
                        'probability': prob,
                        'calibrated_prob': ev_result.get('calibrated_probability', ev_result['adjusted_probability']),
                        'edge': ev_result.get('calibrated_edge', ev_result['edge']),
                        'should_bet': should_place,
                        'reason': reason
                    })
                
        except Exception as e:
            print(f"   ❌ Error processing fixture: {e}")
            continue
    
    print("\n📊 MARKET TYPE SUMMARY:")
    print("-" * 30)
    for market, count in sorted(market_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {market}: {count} opportunities")
    
    print(f"\n🎯 BTTS ANALYSIS ({len(btts_opportunities)} opportunities):")
    print("-" * 40)
    if btts_opportunities:
        approved = [b for b in btts_opportunities if b['should_bet']]
        print(f"   ✅ Approved: {len(approved)}")
        print(f"   ❌ Rejected: {len(btts_opportunities) - len(approved)}")
        
        for opp in btts_opportunities:
            status = "✅" if opp['should_bet'] else "❌"
            print(f"   {status} {opp['fixture']} - {opp['market']}")
            print(f"      Prob: {opp['probability']:.1%} → {opp['calibrated_prob']:.1%}, Edge: {opp['edge']:.1%}")
            print(f"      {opp['reason']}")
    else:
        print("   ⚠️ NO BTTS opportunities found in sample")
        print("   💡 This might be why no bets are being placed!")
    
    print(f"\n⚖️ DOUBLE CHANCE ANALYSIS ({len(double_chance_opportunities)} opportunities):")
    print("-" * 50)
    if double_chance_opportunities:
        approved = [d for d in double_chance_opportunities if d['should_bet']]
        print(f"   ✅ Approved: {len(approved)}")
        print(f"   ❌ Rejected: {len(double_chance_opportunities) - len(approved)}")
        
        # Show why they're failing
        rejection_reasons = {}
        for opp in double_chance_opportunities:
            if not opp['should_bet']:
                reason = opp['reason'].split('(')[0].strip()  # Get main reason
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        print("\n   📊 Rejection Reasons:")
        for reason, count in rejection_reasons.items():
            print(f"      • {reason}: {count} times")
    
    return market_counts, btts_opportunities, double_chance_opportunities

def suggest_adjustments(btts_opps, dc_opps):
    """Suggest filter adjustments based on analysis"""
    
    print("\n💡 RECOMMENDATIONS:")
    print("="*50)
    
    if not btts_opps:
        print("🚨 CRITICAL: No BTTS opportunities detected!")
        print("   1. Check if BTTS markets are in Matchbook data")
        print("   2. Verify model is generating BTTS predictions")
        print("   3. Make sure BTTS market recognition is working")
    else:
        btts_approved = len([b for b in btts_opps if b['should_bet']])
        if btts_approved == 0:
            avg_edge = sum(b['edge'] for b in btts_opps) / len(btts_opps) if btts_opps else 0
            avg_prob = sum(b['calibrated_prob'] for b in btts_opps) / len(btts_opps) if btts_opps else 0
            print(f"⚠️ BTTS: 0/{len(btts_opps)} approved")
            print(f"   Average edge: {avg_edge:.1%}, Average prob: {avg_prob:.1%}")
            print("   💡 Consider lowering BTTS requirements slightly")
    
    if dc_opps:
        dc_approved = len([d for d in dc_opps if d['should_bet']])
        avg_edge = sum(d['edge'] for d in dc_opps) / len(dc_opps) if dc_opps else 0
        avg_prob = sum(d['calibrated_prob'] for d in dc_opps) / len(dc_opps) if dc_opps else 0
        
        print(f"⚖️ Double Chance: {dc_approved}/{len(dc_opps)} approved")
        print(f"   Average edge: {avg_edge:.1%}, Average prob: {avg_prob:.1%}")
        
        if dc_approved == 0 and avg_edge > 0.02 and avg_prob > 0.55:
            print("   💡 Consider lowering Double Chance requirements:")
            print("     - Reduce win rate requirement from 58% to 55%")
            print("     - Reduce edge requirement from 4% to 3%")

if __name__ == "__main__":
    market_counts, btts_opps, dc_opps = diagnose_market_availability()
    suggest_adjustments(btts_opps, dc_opps)