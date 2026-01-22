#!/usr/bin/env python3
"""
Quick Match Rankings - Run this to see your best match outcomes ranked from best to worst
"""
from match_outcome_ranker import MatchOutcomeRanker
from best_match_outcomes import BestMatchOutcomeSelector

def show_quick_rankings():
    """Show a quick summary of ranked match outcomes"""
    
    print("🚀 FOOTBALL MATCH OUTCOME RANKINGS")
    print("=" * 60)
    print("Choose your ranking approach:")
    print("1. Best outcome per match (recommended)")
    print("2. All outcomes from all matches")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        show_best_per_match()
    else:
        show_all_outcomes()

def show_best_per_match():
    """Show best outcome per match ranked from best to worst"""
    print("\n🎯 BEST OUTCOMES PER MATCH")
    print("=" * 60)
    print("Finding the single best opportunity for each match...")
    print()
    
    selector = BestMatchOutcomeSelector()
    ranked_outcomes = selector.select_best_outcomes_per_match()
    
    if not ranked_outcomes:
        print("❌ No match data found. Make sure you have merged_match_data.json")
        return
    
    # Show top 10 best match opportunities
    print("\n🏆 TOP 10 BEST MATCH OPPORTUNITIES")
    print("=" * 50)
    
    for i, outcome in enumerate(ranked_outcomes[:10], 1):
        match_desc = outcome['match_description']
        outcome_type = outcome['outcome_display']
        probability = outcome['probability']
        odds = outcome['odds']
        edge = outcome['edge']
        quality = outcome['quality_score']
        strength = outcome['recommendation_strength']
        
        print(f"{i:2d}. {match_desc}")
        print(f"    📊 {outcome_type} ({outcome['symbol']}) - {probability:.1%} @ {odds:.2f}")
        print(f"    {strength} | Quality: {quality:.3f}", end="")
        if edge > 0:
            print(f" | 💰 Edge: +{edge:.1%}")
        else:
            print()
        print()
    
    # Show categories summary
    categories = selector.create_match_categories(ranked_outcomes)
    selector.display_category_summary(categories)

def show_all_outcomes():
    """Show all outcomes from all matches"""
    print("\n📊 ALL MATCH OUTCOMES")
    print("=" * 60)
    print("Analyzing every possible outcome for ranking...")
    print()
    
    # Create ranker and analyze outcomes
    ranker = MatchOutcomeRanker()
    ranked_outcomes = ranker.rank_all_outcomes()
    
    if not ranked_outcomes:
        print("❌ No match data found. Make sure you have merged_match_data.json")
        return
    
    # Show top 10 most likely outcomes
    print("\n🏆 TOP 10 HIGHEST SCORING OUTCOMES")
    print("=" * 50)
    
    for i, outcome in enumerate(ranked_outcomes[:10], 1):
        match_desc = outcome['match_description']
        outcome_type = outcome['outcome_display']
        probability = outcome['probability']
        odds = outcome['odds']
        edge = outcome['edge']
        score = outcome['ranking_score']
        
        print(f"{i:2d}. {match_desc}")
        print(f"    📊 {outcome_type} - {probability:.1%} probability @ {odds:.2f} odds")
        print(f"    🎯 Ranking Score: {score:.3f}", end="")
        if edge > 0:
            print(f" | 💰 Value Edge: +{edge:.1%}")
        else:
            print()
        print()
    
    # Show categories summary
    categories = ranker.create_ranking_categories(ranked_outcomes)
    
    print("📊 LIKELIHOOD CATEGORIES")
    print("=" * 30)
    print(f"🔥 Very Likely (Top 10%):     {len(categories['very_likely']):3d} outcomes")
    print(f"✅ Likely (10-30%):          {len(categories['likely']):3d} outcomes")
    print(f"⚖️ Moderate (30-60%):         {len(categories['moderate']):3d} outcomes")
    print(f"❓ Unlikely (60-85%):         {len(categories['unlikely']):3d} outcomes")
    print(f"💀 Very Unlikely (85-100%):  {len(categories['very_unlikely']):3d} outcomes")
    print(f"\n📈 Total Outcomes Analyzed:   {len(ranked_outcomes):3d}")
    
    # Show value opportunities
    value_outcomes = [o for o in ranked_outcomes if o['edge'] > 0.05]
    high_prob_outcomes = [o for o in ranked_outcomes if o['probability'] > 0.6]
    
    print(f"\n💰 Value Opportunities (5%+ edge): {len(value_outcomes)}")
    print(f"🎯 High Probability (60%+):        {len(high_prob_outcomes)}")
    
    # Save the report
    ranker.save_ranking_report(ranked_outcomes, 'quick_rankings_report.json')
    
    print("\n📄 Full report saved to: quick_rankings_report.json")
    print("\n🎲 Want to see value betting opportunities? Run the value betting system:")
    print("   python value_betting_system.py")

def show_best_and_worst():
    """Show just the best and worst outcomes side by side"""
    
    ranker = MatchOutcomeRanker()
    ranked_outcomes = ranker.rank_all_outcomes()
    
    if not ranked_outcomes:
        print("❌ No match data found.")
        return
    
    print("\n🏆 BEST vs 💀 WORST OUTCOMES")
    print("=" * 60)
    
    # Show top 5 and bottom 5 side by side
    top_5 = ranked_outcomes[:5]
    bottom_5 = ranked_outcomes[-5:]
    
    print("\n🏆 TOP 5 MOST LIKELY          💀 BOTTOM 5 LEAST LIKELY")
    print("-" * 60)
    
    for i in range(5):
        # Top outcome
        top = top_5[i]
        top_desc = f"{top['match_description'][:25]}..." if len(top['match_description']) > 25 else top['match_description']
        top_line = f"{top_desc:28} {top['probability']:5.1%}"
        
        # Bottom outcome
        bottom = bottom_5[i]
        bottom_desc = f"{bottom['match_description'][:25]}..." if len(bottom['match_description']) > 25 else bottom['match_description']
        bottom_line = f"{bottom_desc:28} {bottom['probability']:5.1%}"
        
        print(f"{i+1}. {top_line} | {bottom_line}")
    
    print(f"\nTotal outcomes ranked: {len(ranked_outcomes)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--comparison":
        show_best_and_worst()
    else:
        show_quick_rankings()