from accumulator_betting_model_fixed import AccumulatorBettingModel

def test_betting_model():
    """Test the betting model with our actual data"""
    
    betting_model = AccumulatorBettingModel()
    
    # Check what files we have
    import os
    files = os.listdir('.')
    print("Available files:", [f for f in files if f.endswith('.json')])
    
    # Use merged data if available
    merged_files = [f for f in files if 'merged' in f and f.endswith('.json')]
    prediction_files = [f for f in files if 'prediction' in f and f.endswith('.json')]
    
    if merged_files:
        print(f"\\nUsing merged data file: {merged_files[0]}")
        
        # Try the merged data method
        try:
            report = betting_model.generate_betting_report_from_merged_data(
                merged_files[0],
                prediction_files[0] if prediction_files else None,
                'betting_report.json'
            )
            
            print("\\nBetting Analysis Results:")
            print(f"Total matches analyzed: {report['total_matches_analyzed']}")
            print(f"Qualified selections: {report['qualified_selections']}")
            print(f"Accumulators generated: {report['total_accumulators_generated']}")
            
            if report['best_single_bet']:
                bet = report['best_single_bet']
                match = bet['match_info']
                print("\\nBest single bet:")
                print(f"  {match['home_team']} vs {match['away_team']}")
                print(f"  Outcome: {bet['outcome']} @ {bet['odds']:.2f}")
                print(f"  Probability: {bet['probability']:.1%}")
                print(f"  Expected Value: {bet['expected_value']:.3f}")
            
            if report['best_accumulator']:
                acc = report['best_accumulator']
                print(f"\\nBest accumulator ({acc['fold_size']}-fold):")
                print(f"  Total odds: {acc['total_odds']:.2f}")
                print(f"  Win probability: {acc['combined_probability']:.1%}")
                print(f"  Expected value: {acc['expected_value']:.3f}")
                print(f"  Risk level: {acc['risk_level']}")
                print(f"  £10 stake → £{acc['potential_returns'][10]:.2f} return")
            
            # Show top selections
            if report['top_10_selections']:
                print("\\nTop 3 selections:")
                for i, sel in enumerate(report['top_10_selections'][:3], 1):
                    match = sel['match_info']
                    print(f"  {i}. {match['home_team']} vs {match['away_team']}")
                    print(f"     {sel['outcome']} @ {sel['odds']:.2f} (Prob: {sel['probability']:.1%})")
                    
        except Exception as e:
            print(f"Error with merged data method: {e}")
            print("Trying with predictions only...")
            
            if prediction_files:
                report = betting_model.generate_betting_report(
                    prediction_files[0],
                    'betting_report.json'
                )
                print("Generated report using predictions only")
    else:
        print("No suitable data files found")

if __name__ == "__main__":
    test_betting_model()