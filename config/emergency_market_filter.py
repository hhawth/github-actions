#!/usr/bin/env python3
"""
Emergency Market Filter
======================
Immediately disable losing markets and focus on profitable ones
"""

import json

def apply_emergency_market_filter():
    """Apply emergency filter based on bet analysis results"""
    
    # Based on your settled_bets.json analysis:
    PROFITABLE_MARKETS = [
        'Both Teams To Score',  # 66.7% win rate - ONLY profitable market
        'BTTS',
        'Both Teams Score'
    ]
    
    BANNED_MARKETS = [
        'Total',                # 43.3% win rate
        'Over/Under 2.5 Goals', 
        'Over/Under 1.5 Goals',
        'Under 2.5',
        'Over 2.5',
        'O/U',
        'Totals'
    ]
    
    market_filter = {
        "emergency_mode": True,
        "allowed_markets": PROFITABLE_MARKETS,
        "banned_markets": BANNED_MARKETS,
        "min_ev_threshold": 0.15,  # INCREASE from 5% to 15%
        "max_confidence": 0.85,    # REDUCE overconfidence 
        "max_stake_multiplier": 0.2,  # REDUCE stakes by 80%
        "reason": "Emergency filter after -2.03 unit loss analysis",
        "created": "2026-02-16 urgent retraining"
    }
    
    # Save emergency market filter
    with open('emergency_market_filter.json', 'w') as f:
        json.dump(market_filter, f, indent=2)
    
    print("🚨 EMERGENCY MARKET FILTER APPLIED")
    print("="*40)
    print("✅ ALLOWED (Profitable):")
    for market in PROFITABLE_MARKETS:
        print(f"   ✓ {market}")
    print("\n🚫 BANNED (Losing Money):")
    for market in BANNED_MARKETS:
        print(f"   ✗ {market}")
    print("\n📈 Minimum EV requirement: 15%")
    print("🎯 Maximum confidence: 85%")
    print("💰 Stakes reduced by 80%")
    
    # Update automated betting workflow if it exists
    update_workflow_config()

def update_workflow_config():
    """Update the main workflow to use emergency settings"""
    
    workflow_config = {
        "force_emergency_mode": True,
        "market_filter_file": "emergency_market_filter.json",
        "recalibration_factor": 0.85,  # Reduce overconfidence by 15%
        "retraining_triggers": {
            "win_rate_threshold": 0.55,
            "loss_streak_limit": 5,
            "roi_emergency_stop": -0.10
        }
    }
    
    with open('emergency_workflow_config.json', 'w') as f:
        json.dump(workflow_config, f, indent=2)
    
    print("\n⚙️ Emergency workflow config saved")

if __name__ == "__main__":
    apply_emergency_market_filter()
    print("\n💡 Next steps:")
    print("   1. Run: python emergency_retrain.py")
    print("   2. Test with small stakes")
    print("   3. Monitor performance closely")