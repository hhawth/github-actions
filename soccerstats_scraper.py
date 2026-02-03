import requests
from bs4 import BeautifulSoup
import json
import pickle
import os
# import pandas as pd

def get_authenticated_session():
    """Create or load an authenticated session with persistent cookies"""
    session = requests.Session()
    cookie_file = "soccerstats_cookies.pkl"
    
    # Load existing cookies if available
    if os.path.exists(cookie_file):
        try:
            with open(cookie_file, 'rb') as f:
                session.cookies.update(pickle.load(f))
            print("Loaded existing session cookies")
        except Exception as e:
            print(f"Could not load cookies: {e}")
    
    # Set browser headers
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",  # Simplified encoding
        "Referer": "https://www.soccerstats.com/members.asp",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0, i"
    })
    
    return session, cookie_file

def save_session_cookies(session, cookie_file):
    """Save session cookies to file"""
    try:
        with open(cookie_file, 'wb') as f:
            pickle.dump(session.cookies, f)
        print("Session cookies saved")
    except Exception as e:
        print(f"Could not save cookies: {e}")

def set_manual_cookies(session):
    """Set cookies manually if automated login fails"""
    # Fresh cookies from user's browser - updated with new tokens
    cookie_string = "vpl=1; cf_clearance=vG1lqpMylTxlR_.RwAsHfupaq.GeLOOAgLbwdSQ4CY0-1770113677-1.2.1.1-wgeBqkQj3UmvlT6fBKxWrNMzQsvFxdEEWKNyRcu7Gc9MAOuQH2d4fyXBKsU_Zv3nrN5KmzIer7MjQaDqd_6cIkbNY8TUabRpdc9T_zbNPxAZgFF.nXPz2XWeHq.6nFY5PMJE2MGkVG71N.DbWSP3fnyDm4K0fDr38le5sYp7L755Blkt_4tp0JrWb4z5VmOyOQGcj.4ytYoqG2fKuuZrJUzOvj3HGWstkjT4V6_kmLM; ASPSESSIONIDAGAQBSAB=HKFAHNNADMJEFNKKIJPGLMBN; tz=0; myhtmltickerlive=61; dmmode=1; steady-token=dWxpTHY3c0hIQy9jSEZBTmJPMXU0UT09; mmode=1; mmoderec=1; mmoderem=1"
    
    # Parse cookie string and add to session
    for cookie in cookie_string.split('; '):
        if '=' in cookie:
            name, value = cookie.split('=', 1)
            session.cookies.set(name, value, domain='.soccerstats.com')
    
    print("✅ Fresh cookies set with new tokens!")
    print("Updated cf_clearance and steady-token for full member access")

def check_authentication_status(session, url):
    """Check if we're properly authenticated by testing the response"""
    try:
        response = session.get(url)
        content = response.text.lower()
        
        # Signs that authentication failed
        auth_failed_indicators = [
            'you are not logged in',
            'please login', 
            'access denied',
            'subscription required',
            'premium members only',
            'sign up',
            'register now'
        ]
        
        for indicator in auth_failed_indicators:
            if indicator in content:
                return False, f"Authentication failed: found '{indicator}'"
        
        # Signs that we might be authenticated
        auth_success_indicators = [
            'welcome back',
            'logout',
            'my account',
            'member since'
        ]
        
        for indicator in auth_success_indicators:
            if indicator in content:
                return True, f"Authentication likely successful: found '{indicator}'"
        
        return None, "Authentication status unclear"
        
    except Exception as e:
        return False, f"Error checking authentication: {e}"

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
        # Get authenticated session with persistent cookies
        session, cookie_file = get_authenticated_session()
        
        # Try request with existing cookies first
        response = session.get(url)
        
        # If we get a login page or 401/403, set manual cookies
        if "login" in response.url.lower() or response.status_code in [401, 403]:
            print("Session expired, setting manual cookies...")
            set_manual_cookies(session)
            response = session.get(url)
        
        response.raise_for_status()
        
        # Save cookies after successful request
        save_session_cookies(session, cookie_file)
        
        # Handle encoding properly to avoid character replacement warnings
        response.encoding = 'utf-8'  # Force UTF-8 encoding
        soup = BeautifulSoup(response.text, 'html.parser')
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
        # Get absolute path to ensure proper file writing in cloud environments
        abs_filename = os.path.abspath(filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_filename = abs_filename + ".tmp"
        
        with open(temp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename
        os.rename(temp_filename, abs_filename)
        
        print(f"Results saved to {abs_filename}")
        return filename
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        # Clean up temp file if it exists
        temp_filename = os.path.abspath(filename) + ".tmp"
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        return None

def main():
    session, cookie_file = get_authenticated_session()
    
    # Force fresh cookies to ensure we have member access
    print("Setting fresh cookies for full member access...")
    set_manual_cookies(session)
    
    # Test authentication on members page
    print("Testing authentication status...")
    auth_status, message = check_authentication_status(session, "https://www.soccerstats.com/members.asp")
        
    if auth_status is True:
        print(f"✅ {message}")
    elif auth_status is None:
        print(f"❓ {message} - proceeding anyway")
    else:
        print(f"❌ {message}")
    
    # Test if we can find "falcon" now with member access
    print("Testing member content access...")
    response = session.get("https://www.soccerstats.com/matches.asp")
    if 'falcon' in response.text.lower():
        print("✅ Found 'falcon' - member content accessible!")
    else:
        print("❓ 'falcon' not found in current matches")
    
    save_session_cookies(session, cookie_file)
    
    # Run the scraper
    print("\nRunning scraper with member access...")
    data = fetch_sportinglife_data()
    if data:
        save_results_json(data)
        print(f"Scraped {len(data.get('leagues', []))} leagues")
        
        # Show what we got and search for falcon
        total_matches = 0
        falcon_found = False
        for league in data.get('leagues', []):
            matches_count = len(league['matches'])
            total_matches += matches_count
            print(f"  {league['league_name']}: {matches_count} matches")
            
            # Search for falcon in team names
            for match in league['matches']:
                if match['team1'] and 'falcon' in match['team1']['Name'].lower():
                    print(f"    🦅 Found falcon team: {match['team1']['Name']}")
                    falcon_found = True
                if match['team2'] and 'falcon' in match['team2']['Name'].lower():
                    print(f"    🦅 Found falcon team: {match['team2']['Name']}")
                    falcon_found = True
        
        print(f"Total matches: {total_matches}")
        if falcon_found:
            print("🎉 Falcon teams found in member data!")
        else:
            print("❓ No falcon teams in today's matches")
    else:
        print("No data scraped - authentication may have failed")

# Main execution
if __name__ == "__main__":
    main()