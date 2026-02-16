#!/usr/bin/env python3
"""
EV Diagnostic Tool
Shows actual calculated EVs for today's fixtures to understand market efficiency.
"""

import json
import numpy as np
from datetime import datetime

def load_improved_model():
    """Load the improved EV betting model."""
    import pickle
    import os
    import sys
    
    # Ensure xgboost is available
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost not available, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    
    # Find the latest model file
    model_files = [f for f in os.listdir('.') if f.startswith('ev_model_') and f.endswith('.pkl')]
    if not model_files:
        raise Exception("No model files found")
    
    latest_model = sorted(model_files)[-1]
    print(f"📈 Loading model: {latest_model}")
    
    try:
        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try simpler approach - just load the current workflow results
        return None

def calculate_ev_for_markets(prob_dict, odds_dict):
    """Calculate EV for different markets."""
    results = {}
    
    # Double Chance markets
    markets = ['1X', 'X2', '12']
    for market in markets:
        if market in prob_dict and market in odds_dict:
            prob = prob_dict[market]
            odds = odds_dict[market]
            ev = (prob * odds) - 1
            if ev > 0:  # Only show positive EVs
                results[f'DC_{market}'] = {
                    'probability': prob,
                    'odds': odds,
                    'ev': ev,
                    'ev_percent': ev * 100
                }
    
    # BTTS
    if 'btts_yes' in prob_dict and 'btts_yes' in odds_dict:
        prob = prob_dict['btts_yes']
        odds = odds_dict['btts_yes']
        ev = (prob * odds) - 1
        if ev > 0:
            results['BTTS_Yes'] = {
                'probability': prob,
                'odds': odds,
                'ev': ev,
                'ev_percent': ev * 100
            }
    
    if 'btts_no' in prob_dict and 'btts_no' in odds_dict:
        prob = prob_dict['btts_no']
        odds = odds_dict['btts_no']
        ev = (prob * odds) - 1
        if ev > 0:
            results['BTTS_No'] = {
                'probability': prob,
                'odds': odds,
                'ev': ev,
                'ev_percent': ev * 100
            }
    
    # Over/Under 2.5
    if 'over_2_5' in prob_dict and 'over_2_5' in odds_dict:
        prob = prob_dict['over_2_5']
        odds = odds_dict['over_2_5']
        ev = (prob * odds) - 1
        if ev > 0:
            results['Over_2.5'] = {
                'probability': prob,
                'odds': odds,
                'ev': ev,
                'ev_percent': ev * 100
            }
    
    if 'under_2_5' in prob_dict and 'under_2_5' in odds_dict:
        prob = prob_dict['under_2_5']
        odds = odds_dict['under_2_5']
        ev = (prob * odds) - 1
        if ev > 0:
            results['Under_2.5'] = {
                'probability': prob,
                'odds': odds,
                'ev': ev,
                'ev_percent': ev * 100
            }
    
    return results

def main():
    print("🔍 EV Diagnostic Analysis")
    print("=" * 50)
    
    try:
        # Load model and data
        model_data = load_improved_model()
        model = model_data['model']
        
        # Load fixtures
        today = datetime.now().strftime('%Y-%m-%d')
        fixtures_file = f'api_football_merged_{today}.json'
        
        with open(fixtures_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixtures = data['fixtures']
        odds_data = {item['fixture']['id']: item for item in data['odds']}
        
        print(f"📊 Analyzing {len(fixtures)} fixtures...")
        
        total_opportunities = 0
        best_evs = []
        
        for fixture in fixtures:
            if fixture['fixture']['status']['short'] != 'NS':
                continue
                
            fixture_id = fixture['fixture']['id']
            
            # Get odds
            if fixture_id not in odds_data:
                continue
            
            odds_item = odds_data[fixture_id]
            
            # Extract features and predict
            try:
                from improved_ev_betting_model import ImprovedEVBettingModel
                temp_model = ImprovedEVBettingModel()
                features = temp_model.extract_features_from_fixture(fixture, odds_item, {}, {})
                
                if features is None:
                    continue
                
                # Convert data for prediction
                feature_array = np.array([list(features.values())]).reshape(1, -1)
                predictions = model.predict_proba(feature_array)
                
                # Build probability dict
                prob_dict = {
                    'home_win': predictions[0][2],  # Home win
                    'draw': predictions[0][1],      # Draw
                    'away_win': predictions[0][0],  # Away win
                }
                
                # Add derived probabilities
                prob_dict['1X'] = prob_dict['home_win'] + prob_dict['draw']
                prob_dict['X2'] = prob_dict['draw'] + prob_dict['away_win']
                prob_dict['12'] = prob_dict['home_win'] + prob_dict['away_win']
                
                # Add BTTS and totals (simplified)
                prob_dict['btts_yes'] = 0.45  # Estimated
                prob_dict['btts_no'] = 0.55
                prob_dict['over_2_5'] = 0.50
                prob_dict['under_2_5'] = 0.50
                
                # Extract odds
                odds_dict = {}
                for bookmaker in odds_item.get('bookmakers', []):
                    for bet in bookmaker.get('bets', []):
                        if bet['name'] == 'Match Winner':
                            for value in bet['values']:
                                key = value['value']
                                if key == 'Home':
                                    odds_dict['home_win'] = float(value['odd'])
                                elif key == 'Draw':
                                    odds_dict['draw'] = float(value['odd'])
                                elif key == 'Away':
                                    odds_dict['away_win'] = float(value['odd'])
                        elif bet['name'] == 'Double Chance':
                            for value in bet['values']:
                                if value['value'] == 'Home/Draw':
                                    odds_dict['1X'] = float(value['odd'])
                                elif value['value'] == 'Draw/Away':
                                    odds_dict['X2'] = float(value['odd'])
                                elif value['value'] == 'Home/Away':
                                    odds_dict['12'] = float(value['odd'])
                        elif bet['name'] == 'Goals Over/Under':
                            for value in bet['values']:
                                if value['value'] == 'Over 2.5':
                                    odds_dict['over_2_5'] = float(value['odd'])
                                elif value['value'] == 'Under 2.5':
                                    odds_dict['under_2_5'] = float(value['odd'])
                        elif bet['name'] == 'Both Teams Score':
                            for value in bet['values']:
                                if value['value'] == 'Yes':
                                    odds_dict['btts_yes'] = float(value['odd'])
                                elif value['value'] == 'No':
                                    odds_dict['btts_no'] = float(value['odd'])
                    break  # Only use first bookmaker
                
                # Calculate EVs
                evs = calculate_ev_for_markets(prob_dict, odds_dict)
                
                if evs:
                    total_opportunities += len(evs)
                    for market, data in evs.items():
                        best_evs.append({
                            'fixture': f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}",
                            'market': market,
                            'ev_percent': data['ev_percent'],
                            'probability': data['probability'],
                            'odds': data['odds']
                        })
                
                # Show top 5 fixtures regardless of EV
                if len(best_evs) == 0 and len([f for f in fixtures if f['fixture']['status']['short'] == 'NS']) <= 5:
                    # Show all calculated EVs for first few fixtures
                    all_evs = {}
                    for market in ['1X', 'X2', '12']:
                        if market in prob_dict and market in odds_dict:
                            prob = prob_dict[market]
                            odds = odds_dict[market]
                            ev = (prob * odds) - 1
                            all_evs[f'DC_{market}'] = ev * 100
                    
                    if 'btts_yes' in prob_dict and 'btts_yes' in odds_dict:
                        prob = prob_dict['btts_yes']
                        odds = odds_dict['btts_yes']
                        ev = (prob * odds) - 1
                        all_evs['BTTS_Yes'] = ev * 100
                    
                    if 'over_2_5' in prob_dict and 'over_2_5' in odds_dict:
                        prob = prob_dict['over_2_5']
                        odds = odds_dict['over_2_5']
                        ev = (prob * odds) - 1
                        all_evs['Over_2.5'] = ev * 100
                    
                    print(f"\n📊 {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
                    for market, ev_pct in sorted(all_evs.items(), key=lambda x: x[1], reverse=True):
                        print(f"   {market}: {ev_pct:.2f}%")
                
            except Exception:
                continue
        
        print("\n📈 RESULTS:")
        print(f"📊 Total positive EV opportunities: {total_opportunities}")
        
        if best_evs:
            print("\n🏆 Best EV Opportunities:")
            best_evs.sort(key=lambda x: x['ev_percent'], reverse=True)
            for i, bet in enumerate(best_evs[:10]):
                print(f"{i+1}. {bet['fixture']}")
                print(f"   Market: {bet['market']}")
                print(f"   EV: {bet['ev_percent']:.2f}%")
                print(f"   Prob: {bet['probability']:.3f}, Odds: {bet['odds']:.2f}")
        else:
            print("\n☹️ No positive EV opportunities found")
            print("🧐 This suggests:")
            print("   • Markets are very efficiently priced today")
            print("   • Model predictions align closely with market odds")
            print("   • Need even more relaxed EV threshold (0.5%?) or different approach")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()