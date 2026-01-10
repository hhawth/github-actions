#!/usr/bin/env python3
"""
Mock data generator for when ClubElo doesn't have current fixtures from various leagues
"""

import pandas as pd
from datetime import datetime, timedelta
import random

def generate_mock_fixtures():
    """Generate mock fixtures from multiple European leagues for demonstration"""
    
    # Teams from different leagues
    leagues = {
        'ENG': ['Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Man Utd', 'Newcastle', 'Tottenham', 'Brighton'],
        'ESP': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Real Sociedad', 'Villarreal', 'Valencia', 'Athletic Club'],
        'FRA': ['PSG', 'Marseille', 'Monaco', 'Lyon', 'Lille', 'Rennes', 'Nice', 'Lens'],
        'GER': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt', 'Union Berlin', 'Freiburg', 'Wolfsburg'],
        'ITA': ['Juventus', 'AC Milan', 'Inter Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta', 'Fiorentina']
    }
    
    fixtures = []
    fixture_date = datetime.now() + timedelta(days=1)
    
    # Generate fixtures from each league
    for country, teams in leagues.items():
        # Generate 3-4 fixtures per league
        num_fixtures = random.randint(3, 4)
        used_teams = set()
        
        for i in range(num_fixtures):
            # Pick home team
            available_home = [t for t in teams if t not in used_teams]
            if len(available_home) < 2:
                used_teams.clear()  # Reset if we run out
                available_home = teams
            
            home_team = random.choice(available_home)
            used_teams.add(home_team)
            
            # Pick away team
            available_away = [t for t in teams if t != home_team and t not in used_teams]
            if not available_away:
                available_away = [t for t in teams if t != home_team]
            
            away_team = random.choice(available_away)
            used_teams.add(away_team)
            
            # Generate realistic probabilities that sum to ~100%
            home_win = random.uniform(0.2, 0.6)
            away_win = random.uniform(0.15, 0.5) 
            draw = 1.0 - home_win - away_win
            
            # Adjust to make them sum to ~1 with some bookmaker edge
            total = home_win + draw + away_win
            home_win /= total
            draw /= total  
            away_win /= total
            
            # Add some bookmaker edge
            edge = random.uniform(1.05, 1.12)
            home_win *= edge
            draw *= edge
            away_win *= edge
            
            # Generate likely score outcomes
            score_outcomes = {
                'R:0-0': draw * 0.15,
                'R:1-0': home_win * 0.25,
                'R:0-1': away_win * 0.25,
                'R:1-1': draw * 0.4,
                'R:2-0': home_win * 0.2,
                'R:0-2': away_win * 0.2,
                'R:2-1': home_win * 0.3,
                'R:1-2': away_win * 0.3,
                'R:2-2': draw * 0.25,
                'R:3-0': home_win * 0.1,
                'R:0-3': away_win * 0.1,
                'R:3-1': home_win * 0.15,
                'R:1-3': away_win * 0.15,
            }
            
            # Pick most likely outcome
            likely_outcome = max(score_outcomes.items(), key=lambda x: x[1])[0]
            
            fixture = {
                'Date': (fixture_date + timedelta(days=i//2)).strftime('%Y-%m-%d'),
                'Country': country,
                'Home': home_team,
                'Away': away_team,
                'Home Win': home_win,
                'Draw': draw, 
                'Away Win': away_win,
                'Likely Outcome': likely_outcome,
                # Add goal difference probabilities
                'GD=-5': away_win * 0.02,
                'GD=-4': away_win * 0.03,
                'GD=-3': away_win * 0.08,
                'GD=-2': away_win * 0.15,
                'GD=-1': away_win * 0.30,
                'GD=0': draw,
                'GD=1': home_win * 0.30,
                'GD=2': home_win * 0.15,
                'GD=3': home_win * 0.08,
                'GD=4': home_win * 0.03,
                'GD=5': home_win * 0.02,
            }
            
            # Add score outcome probabilities
            fixture.update(score_outcomes)
            
            fixtures.append(fixture)
    
    return pd.DataFrame(fixtures)

def get_mock_fixtures_with_fallback():
    """Get fixtures with fallback to mock data"""
    
    try:
        # First try to get real fixtures
        from stat_getter import get_fixtures_from_clubelo
        real_fixtures = get_fixtures_from_clubelo()
        
        # Check if we have any fixtures
        if not real_fixtures.empty and len(real_fixtures) > 0:
            print(f"📊 Using {len(real_fixtures)} real fixtures from multiple leagues")
            return real_fixtures
        
        print("⚠️ No fixtures found in ClubElo, generating mock data from multiple leagues...")
        return generate_mock_fixtures()
        
    except Exception as e:
        print(f"❌ Error fetching fixtures, using mock data: {e}")
        return generate_mock_fixtures()

if __name__ == "__main__":
    print("🔮 Testing Mock Fixtures from Multiple Leagues...")
    fixtures = get_mock_fixtures_with_fallback()
    print(f"Generated {len(fixtures)} fixtures")
    print(f"Countries included: {', '.join(fixtures['Country'].unique())}")
    print("\nSample fixtures:")
    print(fixtures[['Country', 'Home', 'Away', 'Home Win', 'Draw', 'Away Win']].head())