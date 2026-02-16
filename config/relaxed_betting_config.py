#!/usr/bin/env python3
"""
Relaxed Betting Configuration
============================
Moderate settings to allow some bets while still being protective
"""

def update_betting_config():
    """Update betting configuration to moderate protection"""
    
    config = {
        # BTTS - The only profitable market (66.7% win rate historically)
        'btts_requirements': {
            'min_win_rate': 0.55,      # Reduce from 60% to 55%
            'min_edge': 0.02,          # Reduce from 3% to 2%  
            'enabled': True
        },
        
        # Double Chance - Marginal performance (58.5% win rate but losing units)
        'double_chance_requirements': {
            'min_win_rate': 0.60,      # Increase from 58% to 60% (be more selective)
            'min_edge': 0.035,         # Reduce from 4% to 3.5%
            'enabled': True
        },
        
        # Over/Under - Keep banned (43.3% win rate - clearly losing money)
        'over_under_banned': True,
        
        # General settings
        'min_ev': 0.08,               # 8% minimum EV (down from 15% but still higher than original 5%)
        'max_stake': 0.05,            # £0.05 maximum stake (5p)
        'daily_limit': 0.25,          # £0.25 daily limit (down from £0.50)
        
        # Safety limits
        'max_bets_per_day': 5,        # Maximum 5 bets per day
        'stop_loss_daily': -0.50,    # Stop if daily loss exceeds 50p
    }
    
    print("📊 MODERATE BETTING CONFIG:")
    print(f"   ✅ BTTS: {config['btts_requirements']['min_win_rate']*100:.0f}%+ WR, {config['btts_requirements']['min_edge']*100:.0f}%+ edge")
    print(f"   ⚖️ Double Chance: {config['double_chance_requirements']['min_win_rate']*100:.0f}%+ WR, {config['double_chance_requirements']['min_edge']*100:.1f}%+ edge")
    print("   🚫 Over/Under: BANNED")
    print(f"   📈 Min EV: {config['min_ev']*100:.0f}%")
    print(f"   💰 Max stake: £{config['max_stake']:.2f}")
    print(f"   🛡️ Daily limit: £{config['daily_limit']:.2f}")
    
    return config

if __name__ == "__main__":
    config = update_betting_config()
    
    print("\n💡 This should allow some BTTS bets while staying protective!")
    print("🎯 To test: Run the betting workflow with these relaxed settings")