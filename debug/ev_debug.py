#!/usr/bin/env python3
"""
EV Debug Tool - Show actual calculated EV values for analysis
"""

import json
import numpy as np
from datetime import datetime
from improved_ev_betting_model import ImprovedEVBettingModel

def main():
    print("🔍 EV Calculation Debug Analysis")
    print("=" * 50)
    
    try:
        # Initialize model
        model = ImprovedEVBettingModel()
        
        # Load model
        try:
            model.load_model()
            print("✅ Model loaded successfully")
        except:
            print("❌ Could not load model")
            return
        
        # Load fixtures
        today = datetime.now().strftime('%Y-%m-%d')
        fixtures_file = f'api_football_merged_{today}.json'
        
        with open(fixtures_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixtures = data['fixtures']
        upcoming = [f for f in fixtures if f.get('fixture', {}).get('status', {}).get('short') == 'NS']
        
        print(f"📊 Analyzing {len(upcoming)} upcoming fixtures...")
        print(f"🎯 Current EV threshold: {model.config['min_ev']:.1%}")
        print("\n" + "="*80)
        
        sample_count = 0
        total_analyzed = 0
        ev_values = []
        
        for fixture in upcoming[:10]:  # Sample 10 fixtures
            try:
                # Get prediction
                prediction = model.predict_fixture(fixture)
                if not prediction or 'error' in prediction:
                    continue
                
                total_analyzed += 1
                
                fixture_id = prediction['fixture_id']
                home_team = prediction['home_team']
                away_team = prediction['away_team']
                
                # Extract probabilities
                dc = prediction['predictions']['double_chance']
                btts_prob = prediction['predictions']['btts']['probability']
                goals = prediction['predictions']['goals']
                
                print(f"\n🏆 {home_team} vs {away_team}")
                print("-" * 60)
                
                # Calculate and display EVs for Double Chance markets
                markets_evs = []
                
                # 1X Market (Home Win or Draw)
                prob_1x = dc['1x_prob']
                estimated_odds_1x = 1 / max(prob_1x, 0.01) if prob_1x > 0 else 10.0
                ev_1x = model.calculate_expected_value(prob_1x, estimated_odds_1x)
                markets_evs.append(('1X (Home/Draw)', prob_1x, estimated_odds_1x, ev_1x))
                ev_values.append(ev_1x)
                
                # X2 Market (Draw or Away Win)  
                prob_x2 = dc['x2_prob']
                estimated_odds_x2 = 1 / max(prob_x2, 0.01) if prob_x2 > 0 else 10.0
                ev_x2 = model.calculate_expected_value(prob_x2, estimated_odds_x2)
                markets_evs.append(('X2 (Draw/Away)', prob_x2, estimated_odds_x2, ev_x2))
                ev_values.append(ev_x2)
                
                # 12 Market (Home or Away Win)
                prob_12 = dc['12_prob']
                estimated_odds_12 = 1 / max(prob_12, 0.01) if prob_12 > 0 else 10.0
                ev_12 = model.calculate_expected_value(prob_12, estimated_odds_12)
                markets_evs.append(('12 (No Draw)', prob_12, estimated_odds_12, ev_12))
                ev_values.append(ev_12)
                
                # BTTS Market
                estimated_btts_odds = 1 / max(btts_prob, 0.01) if btts_prob > 0 else 10.0
                ev_btts = model.calculate_expected_value(btts_prob, estimated_btts_odds)
                markets_evs.append(('BTTS Yes', btts_prob, estimated_btts_odds, ev_btts))
                ev_values.append(ev_btts)
                
                # Over/Under 2.5 Goals
                over_25_prob = goals['over_25_prob']
                under_25_prob = goals['under_25_prob']
                
                estimated_over_odds = 1 / max(over_25_prob, 0.01) if over_25_prob > 0 else 10.0
                estimated_under_odds = 1 / max(under_25_prob, 0.01) if under_25_prob > 0 else 10.0
                
                ev_over = model.calculate_expected_value(over_25_prob, estimated_over_odds)
                ev_under = model.calculate_expected_value(under_25_prob, estimated_under_odds)
                
                markets_evs.append(('Over 2.5', over_25_prob, estimated_over_odds, ev_over))
                markets_evs.append(('Under 2.5', under_25_prob, estimated_under_odds, ev_under))
                ev_values.extend([ev_over, ev_under])
                
                # Display results
                best_ev = max(markets_evs, key=lambda x: x[3])
                
                for market, prob, odds, ev in sorted(markets_evs, key=lambda x: x[3], reverse=True):
                    color = "🟢" if ev > 0.005 else "🟡" if ev > 0 else "🔴"
                    print(f"   {color} {market:<15}: {ev:+.3f} EV ({ev*100:+.1f}%) | Prob: {prob:.1%} | Est. Odds: {odds:.2f}")
                
                sample_count += 1
                if sample_count >= 5:  # Show detailed analysis for first 5
                    print("   ...")
                
            except Exception:
                continue
        
        # Summary statistics
        if ev_values:
            print("\n📊 SUMMARY STATISTICS")
            print("=" * 50)
            print(f"🔍 Fixtures analyzed: {total_analyzed}")
            print(f"📈 Total EV calculations: {len(ev_values)}")
            print(f"📊 Average EV: {np.mean(ev_values):+.4f} ({np.mean(ev_values)*100:+.2f}%)")
            print(f"📈 Max EV found: {max(ev_values):+.4f} ({max(ev_values)*100:+.2f}%)")
            print(f"📉 Min EV found: {min(ev_values):+.4f} ({min(ev_values)*100:+.2f}%)")
            print(f"📊 Standard deviation: {np.std(ev_values):.4f}")
            
            positive_evs = [ev for ev in ev_values if ev > 0]
            print(f"✅ Positive EVs: {len(positive_evs)}/{len(ev_values)} ({len(positive_evs)/len(ev_values)*100:.1f}%)")
            
            if positive_evs:
                print(f"📈 Best positive EV: {max(positive_evs):+.4f} ({max(positive_evs)*100:+.2f}%)")
            else:
                print("❌ No positive EVs found across all markets!")
                
            # Check different thresholds
            for threshold in [0.005, 0.01, 0.02, 0.03]:
                count = sum(1 for ev in ev_values if ev >= threshold)
                print(f"🎯 EVs ≥ {threshold:.1%}: {count}/{len(ev_values)} opportunities")
                
        print("\n🧐 ANALYSIS:")
        if not ev_values or max(ev_values) < 0:
            print("🎯 Markets are extremely efficient today!")
            print("📊 Model predictions closely match market expectations")
            print("💡 Consider: Live betting, different time periods, or niche markets")
        elif max(ev_values) < 0.005:
            print("🎯 Very efficient markets - largest EV under 0.5%")
            print("📊 Model finds only marginal value opportunities")
        else:
            print(f"🎯 Best opportunity: {max(ev_values)*100:.2f}% EV")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()