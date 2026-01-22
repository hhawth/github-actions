import requests
from bs4 import BeautifulSoup
import json
# import pandas as pd

def fetch_sportinglife_data(url="https://www.soccerstats.com/matches.asp?matchday=1&matchdayn=1"):
    """
    Fetches and parses soccer match data from the given URL.
    
    Data field definitions:
    - Name: Team name
    - Time: Match time
    - Scope: Home/Away designation
    - GP: Number of matches played
    - W%: Percentage of matches won
    - FTS: % Failed to score (no color coding)
    - CS: % Clean sheet (no color coding)
    - BTS: % of matches where both teams scored (no color coding)
    - TG: Average total goals scored + conceded per match
    - GF: Goals scored per match (on this page: in green, no color coding)
    - GA: Goals conceded per match (on this page: in red, no color coding)
    - 1.5+: % of matches over 1.5 goals (scored + conceded)
    - 2.5+: % of matches over 2.5 goals (scored + conceded) - values above 50% displayed in blue, others in red
    - 3.5+: % of matches over 3.5 goals (scored + conceded)
    - Points Per Game: Points per game (no color coding)
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tr_elements = soup.find_all('tr')
        match_elements = tr_elements[8].find_all('tr', class_=['parent', 'team1row', 'team2row'])
        
        # Data structure to store all leagues and matches
        structured_data = {
            "leagues": []
        }
        
        current_league = None
        league_matches = []
        
        def process_league_matches(league_name, matches):
            """Process matches for a league and pair teams"""
            if not matches:
                return
            
            def has_sufficient_data(team_data):
                """Check if team has sufficient statistical data for predictions"""
                # Key stats that must be present for meaningful predictions
                required_stats = ["GP", "W%", "GF", "GA", "CS"]
                
                for stat in required_stats:
                    value = team_data.get(stat, "")
                    # Check if value is empty, blank, or N/A
                    if not value or str(value).strip() == "" or str(value).strip().upper() in ["N/A", "NA", "-"]:
                        return False
                
                # Additional check: GP should be a reasonable number (> 0)
                try:
                    gp = float(team_data.get("GP", 0))
                    if gp <= 0:
                        return False
                except (ValueError, TypeError):
                    return False
                
                return True
            
            # Define column headers with their meanings for AI agent reference
            headers = ["Name", "Time", "Scope", "GP", "W%", "FTS", "CS", "BTS", "TG", "GF", "GA", "1.5+", "2.5+", "3.5+", "Points Per Game"]
            
            # Field definitions for reference:
            field_definitions = {
                "Name": "Team name",
                "Time": "Match time", 
                "Scope": "Home/Away designation",
                "GP": "Number of matches played",
                "W%": "Percentage of matches won",
                "FTS": "% Failed to score",
                "CS": "% Clean sheet", 
                "BTS": "% of matches where both teams scored",
                "TG": "Average total goals scored + conceded per match",
                "GF": "Goals scored per match",
                "GA": "Goals conceded per match", 
                "1.5+": "% of matches over 1.5 goals (scored + conceded)",
                "2.5+": "% of matches over 2.5 goals (scored + conceded)",
                "3.5+": "% of matches over 3.5 goals (scored + conceded)",
                "Points Per Game": "Points per game"
            }
            
            league_data = {
                "league_name": league_name,
                "matches": [],
                "field_definitions": field_definitions  # Include field definitions for AI agent reference
            }
            
            current_match_teams = []
            
            for match in matches:
                # Extract text from all td elements more carefully
                cells = match.find_all('td')
                cell_texts = []
                
                for cell in cells:
                    # Get text and clean it up
                    cell_text = cell.get_text(separator=' ', strip=True)
                    # Remove extra whitespace and newlines
                    cell_text = ' '.join(cell_text.split())
                    cell_texts.append(cell_text)
                
                # Skip rows that contain 'analysis' as they're not team data
                if any('analysis' in cell.lower() for cell in cell_texts):
                    continue
                
                # Only proceed if we have enough cells for a valid team row
                if len(cell_texts) >= 3:  # At least name, scope, GP
                    team_data = {}
                    
                    # First cell is always the team name
                    team_data["Name"] = cell_texts[0]
                    
                    # Determine if this is home or away team based on data structure
                    # Home teams have: Name, Time, Scope, GP, W%, FTS, CS, BTS, TG, GF, GA, 1.5+, 2.5+, 3.5+, [empty], Points Per Game
                    # Away teams have: Name, Scope, GP, W%, FTS, CS, BTS, TG, GF, GA, 1.5+, 2.5+, 3.5+, [empty], Points Per Game
                    
                    # Check if second cell looks like a time (contains : or is "pp." etc.)
                    is_home_team = len(cell_texts) > 1 and (':' in str(cell_texts[1]) or cell_texts[1] in ['pp.', 'PP'])
                    
                    if is_home_team:
                        # Home team structure: Name, Time, Scope, GP, W%, FTS, CS, BTS, TG, GF, GA, 1.5+, 2.5+, 3.5+, [empty], Points Per Game
                        mapping = {
                            "Time": 1,
                            "Scope": 2, 
                            "GP": 3,
                            "W%": 4,
                            "FTS": 5,
                            "CS": 6,
                            "BTS": 7,
                            "TG": 8,
                            "GF": 9,
                            "GA": 10,
                            "1.5+": 11,
                            "2.5+": 12,
                            "3.5+": 13,
                            "Points Per Game": 15  # Skip index 14 which is often empty
                        }
                    else:
                        # Away team structure: Name, Scope, GP, W%, FTS, CS, BTS, TG, GF, GA, 1.5+, 2.5+, 3.5+, [empty], Points Per Game
                        mapping = {
                            "Time": None,  # Away teams don't have time
                            "Scope": 1,
                            "GP": 2,
                            "W%": 3,
                            "FTS": 4,
                            "CS": 5,
                            "BTS": 6,
                            "TG": 7,
                            "GF": 8,
                            "GA": 9,
                            "1.5+": 10,
                            "2.5+": 11,
                            "3.5+": 12,
                            "Points Per Game": 14  # Skip index 13 which is often empty
                        }
                    
                    # Map the data using the correct structure
                    for field, index in mapping.items():
                        if index is not None and index < len(cell_texts):
                            team_data[field] = cell_texts[index]
                        else:
                            team_data[field] = ""
                    
                    # Add team to current match
                    current_match_teams.append(team_data)

                    # If we have 2 teams, create a match pair (only if both teams have sufficient data)
                    if len(current_match_teams) == 2:
                        team1_valid = has_sufficient_data(current_match_teams[0])
                        team2_valid = has_sufficient_data(current_match_teams[1])
                        
                        if team1_valid and team2_valid:
                            match_pair = {
                                "team1": current_match_teams[0],
                                "team2": current_match_teams[1]
                            }
                            league_data["matches"].append(match_pair)
                        
                        current_match_teams = []  # Reset for next match
            
            # Handle case where last team doesn't have a pair
            if current_match_teams:
                # Only add unpaired team if it has sufficient data
                if has_sufficient_data(current_match_teams[0]):
                    match_pair = {
                        "team1": current_match_teams[0],
                        "team2": None
                    }
                    league_data["matches"].append(match_pair)
            
            # Only add league to structured data if it has valid matches
            if league_data["matches"]:
                structured_data["leagues"].append(league_data)
        
        for elem in match_elements:
            if elem.has_attr('class') and 'parent' in elem['class']:
                # Process previous league if it exists
                if current_league and league_matches:
                    process_league_matches(current_league, league_matches)
                    
                # Start a new league
                current_league = elem.find('font').get_text(strip=True) if elem.find('font') else 'No name'
                league_matches = []
                
            elif elem.has_attr('class') and ('team1row' in elem['class'] or 'team2row' in elem['class']):
                # Add this team row to the current league
                if current_league:
                    league_matches.append(elem)
        
        # Process the last league if it exists
        if current_league and league_matches:
            process_league_matches(current_league, league_matches)

        return structured_data
    except Exception:
        return None

def save_results_json(data, filename=None):
    """Save scraper results to JSON file"""
    if data is None:
        return None
    
    if filename is None:
        filename = "soccerstats_matches.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return None

def main():
    data = fetch_sportinglife_data()
    if data:
        save_results_json(data)
        print(f"Scraped {len(data.get('leagues', []))} leagues")

# Main execution
if __name__ == "__main__":
    main()