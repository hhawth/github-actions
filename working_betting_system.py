import json
from accumulator_betting_model_fixed import AccumulatorBettingModel

def convert_fractional_to_decimal(fractional_odds):
    """Convert fractional odds (e.g., '5/2') to decimal odds (e.g., 3.5)"""
    if not fractional_odds or fractional_odds == '':
        return None
        
    try:
        # Handle decimal odds that are already in decimal format
        if isinstance(fractional_odds, (int, float)):
            return float(fractional_odds)
            
        # Convert string to float if it's already decimal
        try:
            return float(fractional_odds)
        except ValueError:
            pass
            
        # Handle fractional format
        if '/' in str(fractional_odds):
            parts = str(fractional_odds).split('/')
            if len(parts) == 2:
                numerator = float(parts[0])
                denominator = float(parts[1])
                return (numerator / denominator) + 1.0
        
        return None
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def create_working_betting_system():
    """Create a fully working betting system with proper odds conversion"""
    
    # Load merged data
    with open('merged_match_data.json', 'r') as f:
        merged_data = json.load(f)
    
    betting_model = AccumulatorBettingModel(
        min_probability=0.25,  # Lower threshold to get more selections
        max_odds=6.0           # Higher threshold for more variety
    )
    
    selections = []
    processed = 0
    
    print(f"Processing {len(merged_data)} matches...")
    
    for match in merged_data:
        if not isinstance(match, dict):
            continue
            
        processed += 1
        
        # Extract team names and match info
        home_team = match.get('home_team', 'Unknown')
        away_team = match.get('away_team', 'Unknown')
        league = match.get('league', 'Unknown')
        
        # Convert odds from fractional to decimal
        odds_1_decimal = convert_fractional_to_decimal(match.get('odds_1'))  # Home win
        odds_x_decimal = convert_fractional_to_decimal(match.get('odds_x'))  # Draw
        odds_2_decimal = convert_fractional_to_decimal(match.get('odds_2'))  # Away win
        
        # Create selections for each outcome
        outcomes = [
            ('home_win', odds_1_decimal, '1'),
            ('draw', odds_x_decimal, 'X'), 
            ('away_win', odds_2_decimal, '2')
        ]
        
        for outcome_name, decimal_odds, symbol in outcomes:
            if decimal_odds and decimal_odds > 1.0:
                implied_prob = 1.0 / decimal_odds
                
                selection = {
                    'outcome': outcome_name,
                    'probability': implied_prob,
                    'odds': decimal_odds,
                    'confidence': 0.4,  # Odds-only confidence
                    'source': 'odds_only',
                    'expected_value': (implied_prob * decimal_odds) - 1,
                    'risk_score': betting_model._calculate_risk_score(implied_prob, 0.4, decimal_odds),
                    'match_info': {
                        'home_team': home_team,
                        'away_team': away_team,
                        'league': league,
                        'time': match.get('time', 'TBD'),
                        'date': match.get('date', 'TBD')
                    },
                    'selection_symbol': symbol
                }
                
                # Filter by betting criteria
                if (selection['probability'] >= betting_model.min_probability and 
                    selection['odds'] <= betting_model.max_odds):
                    selections.append(selection)
        
        # Show progress every 50 matches
        if processed % 50 == 0:
            print(f"  Processed {processed} matches, found {len(selections)} qualifying selections")
    
    print("\\nFinal results:")
    print(f"  Matches processed: {processed}")
    print(f"  Qualifying selections found: {len(selections)}")
    
    if len(selections) < 5:
        print("  ⚠️ Not enough selections for meaningful accumulators")
        return
    
    # Add confidence score for ranking
    for selection in selections:
        selection['confidence_score'] = selection['confidence']
    
    # Rank selections
    ranked_selections = betting_model.rank_selections(selections)
    print(f"  Selections after ranking: {len(ranked_selections)}")
    
    # Generate accumulators
    accumulators = betting_model.generate_multiple_accumulators(ranked_selections, max_fold=6)
    print(f"  Accumulators generated: {len(accumulators)}")
    
    # Create comprehensive report
    report = {
        'generated_at': '2026-01-20T01:00:00',
        'total_matches_analyzed': processed,
        'total_selections_found': len(selections),
        'qualified_selections': len(ranked_selections),
        'top_10_selections': ranked_selections[:10],
        'recommended_accumulators': accumulators[:5],
        'all_accumulators': accumulators,
        'best_single_bet': ranked_selections[0] if ranked_selections else None,
        'best_accumulator': accumulators[0] if accumulators else None,
        'total_accumulators_generated': len(accumulators)
    }
    
    # Save comprehensive report
    with open('comprehensive_betting_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\\n" + "="*70)
    print("                    🎯 BETTING ANALYSIS RESULTS")
    print("="*70)
    
    # Show best single bet
    if report['best_single_bet']:
        bet = report['best_single_bet']
        match = bet['match_info']
        print("\\n🏆 BEST SINGLE BET:")
        print(f"   {match['home_team']} vs {match['away_team']} ({match['league']})")
        print(f"   Selection: {bet['outcome'].replace('_', ' ').title()} [{bet['selection_symbol']}]")
        print(f"   Odds: {bet['odds']:.2f} | Probability: {bet['probability']:.1%}")
        print(f"   Expected Value: {bet['expected_value']:.3f}")
        print(f"   Risk Score: {bet['risk_score']:.2f}/1.0")
    
    # Show best accumulator
    if report['best_accumulator']:
        acc = report['best_accumulator']
        print(f"\\n🎯 RECOMMENDED ACCUMULATOR ({acc['fold_size']}-fold):")
        print(f"   Total Odds: {acc['total_odds']:.2f}")
        print(f"   Win Probability: {acc['combined_probability']:.1%}")
        print(f"   Expected Value: {acc['expected_value']:.3f}")
        print(f"   Risk Level: {acc['risk_level']}")
        print(f"   Recommendation: {acc['recommendation']}")
        
        print("\\n   💰 Potential Returns:")
        for stake in [5, 10, 20, 50]:
            returns = acc['potential_returns'][stake]
            profit = returns
            print(f"      £{stake} stake → £{stake + returns:.2f} return (£{profit:.2f} profit)")
        
        print("\\n   📋 Accumulator Selections:")
        for i, sel in enumerate(acc['selections'], 1):
            match = sel['match_info'] 
            print(f"      {i}. {match['home_team']} vs {match['away_team']}")
            print(f"         {sel['outcome'].replace('_', ' ').title()} @ {sel['odds']:.2f}")
    
    # Show top individual selections
    print("\\n📊 TOP 5 INDIVIDUAL SELECTIONS:")
    for i, selection in enumerate(ranked_selections[:5], 1):
        match = selection['match_info']
        print(f"   {i}. {match['home_team']} vs {match['away_team']} ({match['league']})")
        print(f"      {selection['outcome'].replace('_', ' ').title()} @ {selection['odds']:.2f}")
        print(f"      Probability: {selection['probability']:.1%} | EV: {selection['expected_value']:.3f}")
        print(f"      Composite Score: {selection['composite_score']:.3f}")
    
    # Show accumulator options
    if len(accumulators) > 1:
        print("\\n⚽ ACCUMULATOR OPTIONS:")
        for i, acc in enumerate(accumulators[:3], 1):
            print(f"   {i}. {acc['fold_size']}-fold: {acc['total_odds']:.2f} odds, {acc['combined_probability']:.1%} chance")
            print(f"      EV: {acc['expected_value']:.3f} | Risk: {acc['risk_level']} | {acc['recommendation']}")
            print(f"      £10 → £{10 + acc['potential_returns'][10]:.2f} ({acc['potential_returns'][10]:.2f} profit)")
    
    print("\\n" + "="*70)
    print("Report saved to: comprehensive_betting_report.json")
    
    return report

if __name__ == "__main__":
    # Test fractional odds conversion
    test_odds = ["5/2", "3/1", "4/5", "6/4", "11/10", "2/3"]
    print("Testing odds conversion:")
    for odds in test_odds:
        decimal = convert_fractional_to_decimal(odds)
        print(f"  {odds} → {decimal:.2f}")
    
    print("\\n" + "="*70)
    
    # Create the working betting system
    create_working_betting_system()