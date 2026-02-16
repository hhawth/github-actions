#!/usr/bin/env python3
"""
Show Current EV Edge Requirements
=================================
"""

from automated_betting_workflow import AutomatedBettingWorkflow

print("🎯 CURRENT EV EDGE REQUIREMENTS")
print("=" * 50)

# Default configuration (what you ran)
print("\n📊 DEFAULT CONFIGURATION:")
workflow_default = AutomatedBettingWorkflow()
print(f"   📉 Minimum EV: {workflow_default.config['min_ev']:.1%} (must beat this to qualify)")
print(f"   📈 Maximum EV: {workflow_default.config['max_ev']:.1%} (reject if higher - likely error)")
print("   🎲 Min Probability: 15% (reject extreme longshots)")
print(f"   💰 Stake per bet: £{workflow_default.config['stake_amount']:.2f}")
print(f"   🛡️ Daily stake limit: £{workflow_default.config['max_daily_stake']:.2f}")

print("\n🔍 ENHANCED COMMAND LINE OPTIONS:")
print("   --min-ev 0.05     → 5% minimum edge (more selective)")
print("   --min-ev 0.02     → 2% minimum edge (more opportunities)")  
print("   --min-ev 0.01     → 1% minimum edge (many opportunities)")
print("   --conservative    → 5% minimum edge + extra safety")

print("\n⚡ CONSERVATIVE MODE (--conservative):")
config_conservative = {
    'min_ev': 0.05,
    'max_ev': 0.10, 
    'stake_amount': 0.05,
    'max_daily_stake': 0.50,
}
print(f"   📉 Minimum EV: {config_conservative['min_ev']:.1%}")
print(f"   📈 Maximum EV: {config_conservative['max_ev']:.1%}")
print(f"   💰 Stake per bet: £{config_conservative['stake_amount']:.2f}")
print(f"   🛡️ Daily stake limit: £{config_conservative['max_daily_stake']:.2f}")

print("\n🎯 YOUR LAST RUN USED:")
print("   📉 Minimum EV: 3.0% (from --min-ev default)")
print("   📈 Maximum EV: 15.0%") 
print("   💰 Stake: £0.10 per bet")
print("   🛡️ Daily limit: £2.00")

print("\n💡 EDGE INTERPRETATION:")
print("   2% edge = Expect £2 profit per £100 wagered long-term")
print("   5% edge = Expect £5 profit per £100 wagered long-term") 
print("   10% edge = Expect £10 profit per £100 wagered long-term")

print("\n⚖️ RISK vs OPPORTUNITY:")
print("   1-2% edge: More opportunities, but smaller edges")
print("   3-5% edge: Balanced approach (CURRENT)")
print("   5%+ edge: Very selective, only clear advantages")