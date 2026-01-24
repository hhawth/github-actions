from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoccerwayScraper:
    """Scraper for soccerway.com to extract football match data"""
    
    def __init__(self, headless=True):
        self.base_url = "https://uk.soccerway.com"
        self.headless = headless
        self.matches = []
        self.driver = None
    
    def init_driver(self):
        """Initialize headless Selenium driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            # Clear webdriver cache to force fresh download
            import os
            import shutil
            
            cache_dir = os.path.expanduser("~/.wdm")
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    logger.info("Cleared webdriver cache")
                except Exception:
                    pass  # Ignore cache clear errors
            
            # Use webdriver_manager to automatically download latest chromedriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Selenium driver initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing driver: {e}")
            logger.info("Try: pip install --upgrade webdriver-manager selenium")
            raise
    
    def close_driver(self):
        """Close the driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Driver closed")
    
    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a webpage using Selenium"""
        try:
            logger.info(f"Loading page: {url}")
            self.driver.get(url)
            
            # Wait for match elements to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "event__match")))
            
            # Additional wait to ensure all elements are rendered
            time.sleep(2)
            
            page_source = self.driver.page_source
            return BeautifulSoup(page_source, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _clean_time_text(self, time_text: str) -> str:
        """Clean unwanted text from time strings"""
        if not time_text:
            return ""
        
        # Remove common unwanted suffixes/text
        unwanted_terms = [
            'Analysis', 'analysis', 'ANALYSIS',
            'Preview', 'preview', 'PREVIEW',
            'Stats', 'stats', 'STATS',
            'Report', 'report', 'REPORT',
            'Lineups', 'lineups', 'LINEUPS'
        ]
        
        # Clean the text by removing unwanted terms
        cleaned = time_text
        for term in unwanted_terms:
            cleaned = cleaned.replace(term, '')
        
        # Clean up any remaining spaces and return
        return cleaned.strip()
    
    def extract_match_data(self, match_element, league: str = None) -> Dict:
        """Extract data from a single match element"""
        try:
            # Debug: Print the HTML structure of the first few matches
            if not hasattr(self, '_debug_printed'):
                logger.info("=== DEBUGGING FIRST MATCH ELEMENT ===")
                logger.info(str(match_element)[:500] + "...")
                self._debug_printed = True
            
            match_data = {
                'region': None,
                'league': league,
                'time': None,
                'home_team': None,
                'away_team': None,
                'home_score': None,
                'away_score': None,
                'odds_1': None,
                'odds_x': None,
                'odds_2': None,
                'status': None
            }
            
            # Extract league name and region from headerLeague__titleWrapper if not provided
            if not league:
                league_header = match_element.find_previous('div', class_='headerLeague__titleWrapper')
                if league_header:
                    # Look for the specific span with headerLeague__title-text class
                    title_span = league_header.find('span', class_='headerLeague__title-text')
                    region_span = league_header.find('span', class_='headerLeague__category-text')
                    
                    if title_span:
                        match_data['league'] = title_span.get_text(strip=True)
                    
                    if region_span:
                        match_data['region'] = region_span.get_text(strip=True)
                    
                    # If specific spans not found, try fallback methods
                    if not match_data['league']:
                        # Fallback to h2 or general text
                        league_title_elem = league_header.find('h2')
                        if league_title_elem:
                            match_data['league'] = league_title_elem.get_text(strip=True)
                        else:
                            # Try to find any text within the title wrapper
                            title_link = league_header.find('a', class_='headerLeague__title')
                            if title_link:
                                match_data['league'] = title_link.get_text(strip=True)
                            else:
                                full_text = league_header.get_text(strip=True)
                                # Try to parse combined text like "Premier LeagueENGLAND:"
                                if ':' in full_text:
                                    parts = full_text.split(':')
                                    if len(parts) >= 2:
                                        # Extract region and league from combined text
                                        combined = parts[0].strip()
                                        # Try to split based on common patterns
                                        import re
                                        match = re.match(r'(.+?)([A-Z]{2,}|[A-Z][a-z]+)$', combined)
                                        if match:
                                            match_data['league'] = match.group(1).strip()
                                            match_data['region'] = match.group(2).strip()
                                        else:
                                            match_data['league'] = combined
                                else:
                                    match_data['league'] = full_text
                else:
                    # Fallback to h4
                    league_header = match_element.find_previous('h4')
                    if league_header:
                        league_text = league_header.get_text(strip=True)
                        match_data['league'] = league_text
            
            # First extract scores (needed for time extraction logic)
            # Look for home score with specific class structure (span elements)
            home_score_elem = match_element.find('span', class_=lambda x: x and 'wcl-matchRowScore_fWR-Z' in x and 'event__score--home' in x)
            if home_score_elem:
                match_data['home_score'] = home_score_elem.get_text(strip=True)
            else:
                # Fallback to simpler selector
                home_score_elem = match_element.find('span', class_=lambda x: x and 'event__score--home' in x)
                if home_score_elem:
                    match_data['home_score'] = home_score_elem.get_text(strip=True)
            
            # Look for away score with specific class structure (span elements)
            away_score_elem = match_element.find('span', class_=lambda x: x and 'wcl-matchRowScore_fWR-Z' in x and 'event__score--away' in x)
            if away_score_elem:
                match_data['away_score'] = away_score_elem.get_text(strip=True)
            else:
                # Fallback to simpler selector
                away_score_elem = match_element.find('span', class_=lambda x: x and 'event__score--away' in x)
                if away_score_elem:
                    match_data['away_score'] = away_score_elem.get_text(strip=True)
            
            # Extract time/status with comprehensive approach
            time_found = False
            
            # Method 1: Check event__time
            time_elem = match_element.find('div', class_='event__time')
            if time_elem:
                time_text = time_elem.get_text(strip=True)
                # Clean up unwanted text like "Analysis"
                time_text = self._clean_time_text(time_text)
                if time_text:
                    match_data['time'] = time_text
                    match_data['status'] = time_text
                    time_found = True
            
            # Method 2: Check event__stage--block (for live/finished games)
            if not time_found:
                time_elem = match_element.find('div', class_='event__stage--block')
                if time_elem:
                    time_text = time_elem.get_text(strip=True)
                    time_text = time_text.replace('\xa0', '').replace('&nbsp;', '').strip()
                    time_text = self._clean_time_text(time_text)
                    if time_text:
                        match_data['time'] = time_text
                        match_data['status'] = time_text
                        time_found = True
            
            # Method 3: Check any event__stage element
            if not time_found:
                stage_elems = match_element.find_all('div', class_=lambda x: x and 'event__stage' in x)
                for elem in stage_elems:
                    text = elem.get_text(strip=True).replace('\xa0', '').replace('&nbsp;', '').strip()
                    text = self._clean_time_text(text)
                    if text and text not in ['', ' ']:
                        match_data['time'] = text
                        match_data['status'] = text
                        time_found = True
                        break
            
            # Method 4: Look for time-related text patterns anywhere in the match element
            if not time_found:
                all_text = match_element.get_text()
                time_patterns = ['Full-time', 'full-time', 'FULL-TIME', 'FT', 'Half-time', 'HT', 
                               'Live', 'LIVE', 'Postponed', 'POSTP', 'Postp', 'AET', 'Pen', 
                               'HT', '90+', 'ET', 'Extra Time']
                
                for pattern in time_patterns:
                    if pattern in all_text:
                        match_data['time'] = pattern
                        match_data['status'] = pattern
                        time_found = True
                        break
            
            # Method 5: For finished games with scores, assume Full-time if no time found
            if not time_found and match_data.get('home_score') and match_data.get('away_score'):
                # Check if scores are actual numbers (not "-" or null)
                try:
                    home_score = match_data.get('home_score')
                    away_score = match_data.get('away_score')
                    if (home_score and home_score != '-' and away_score and away_score != '-' and 
                        home_score.isdigit() and away_score.isdigit()):
                        match_data['time'] = 'Full-time'
                        match_data['status'] = 'Full-time'
                        time_found = True
                except:
                    pass
            
            # Extract home team from event__homeParticipant
            home_team_elem = match_element.find('div', class_='event__homeParticipant')
            if home_team_elem:
                # Look for team name in specific span with wcl-name_jjfMf class
                team_name_span = home_team_elem.find('span', class_=lambda x: x and 'wcl-name_jjfMf' in x)
                if team_name_span:
                    match_data['home_team'] = team_name_span.get_text(strip=True)
                else:
                    # Look for team name in wcl-participants_ASufu child div
                    team_name_div = home_team_elem.find('div', class_='wcl-participants_ASufu')
                    if team_name_div:
                        match_data['home_team'] = team_name_div.get_text(strip=True)
                    else:
                        # Fallback to link or direct text
                        team_link = home_team_elem.find('a')
                        if team_link:
                            match_data['home_team'] = team_link.get_text(strip=True)
                        else:
                            match_data['home_team'] = home_team_elem.get_text(strip=True)
            else:
                # Try alternative selectors
                home_team_elem = match_element.find('div', class_=lambda x: x and 'home' in x.lower())
                if home_team_elem:
                    team_name_div = home_team_elem.find('div', class_='wcl-participants_ASufu')
                    if team_name_div:
                        match_data['home_team'] = team_name_div.get_text(strip=True)
                    else:
                        team_link = home_team_elem.find('a')
                        if team_link:
                            match_data['home_team'] = team_link.get_text(strip=True)
                        else:
                            match_data['home_team'] = home_team_elem.get_text(strip=True)
                else:
                    # Final fallback to generic search
                    home_team_elem = match_element.find('a', class_=lambda x: x and 'home' in x.lower())
                    if not home_team_elem:
                        teams = match_element.find_all('a')
                        if teams:
                            match_data['home_team'] = teams[0].get_text(strip=True)
            
            # Extract away team from event__awayParticipant
            away_team_elem = match_element.find('div', class_='event__awayParticipant')
            if away_team_elem:
                # Look for team name in specific span with wcl-name_jjfMf class
                team_name_span = away_team_elem.find('span', class_=lambda x: x and 'wcl-name_jjfMf' in x)
                if team_name_span:
                    match_data['away_team'] = team_name_span.get_text(strip=True)
                else:
                    # Look for team name in wcl-participants_ASufu child div
                    team_name_div = away_team_elem.find('div', class_='wcl-participants_ASufu')
                    if team_name_div:
                        match_data['away_team'] = team_name_div.get_text(strip=True)
                    else:
                        # Fallback to link or direct text
                        team_link = away_team_elem.find('a')
                        if team_link:
                            match_data['away_team'] = team_link.get_text(strip=True)
                        else:
                            match_data['away_team'] = away_team_elem.get_text(strip=True)
            else:
                # Try alternative selectors
                away_team_elem = match_element.find('div', class_=lambda x: x and 'away' in x.lower())
                if away_team_elem:
                    team_name_div = away_team_elem.find('div', class_='wcl-participants_ASufu')
                    if team_name_div:
                        match_data['away_team'] = team_name_div.get_text(strip=True)
                    else:
                        team_link = away_team_elem.find('a')
                        if team_link:
                            match_data['away_team'] = team_link.get_text(strip=True)
                        else:
                            match_data['away_team'] = away_team_elem.get_text(strip=True)
                else:
                    # Final fallback to generic search
                    away_team_elem = match_element.find('a', class_=lambda x: x and 'away' in x.lower())
                    if not away_team_elem:
                        teams = match_element.find_all('a')
                        if len(teams) > 1:
                            match_data['away_team'] = teams[1].get_text(strip=True)
            
            # Debug: Log what we found
            if match_data['home_team'] or match_data['away_team']:
                logger.info(f"Found teams: {match_data['home_team']} vs {match_data['away_team']}")
            else:
                logger.warning("No teams found in match element")
            
            # Additional fallback score extraction if primary methods failed
            if not match_data['home_score'] or not match_data['away_score']:
                # Look for event__score class
                score_elem = match_element.find('div', class_='event__score')
                if not score_elem:
                    score_elem = match_element.find('div', class_=lambda x: x and 'score' in x.lower())
                
                if score_elem:
                    score_text = score_elem.get_text(strip=True)
                    # Try different score patterns
                    if ' - ' in score_text:
                        try:
                            scores = score_text.split(' - ')
                            if not match_data['home_score']:
                                match_data['home_score'] = scores[0].strip()
                            if not match_data['away_score']:
                                match_data['away_score'] = scores[1].strip()
                        except:
                            pass
                    elif '-' in score_text and len(score_text.split('-')) == 2:
                        try:
                            scores = score_text.split('-')
                            if not match_data['home_score']:
                                match_data['home_score'] = scores[0].strip()
                            if not match_data['away_score']:
                                match_data['away_score'] = scores[1].strip()
                        except:
                            pass
            
            # Extract odds (1X2 format)
            # Home odds (odd1)
            odds_1_elem = match_element.find('div', class_=lambda x: x and 'event__odd--odd1' in x)
            if odds_1_elem:
                match_data['odds_1'] = odds_1_elem.get_text(strip=True)
            
            # Draw odds (odd2)
            odds_x_elem = match_element.find('div', class_=lambda x: x and 'event__odd--odd2' in x)
            if odds_x_elem:
                match_data['odds_x'] = odds_x_elem.get_text(strip=True)
            
            # Away odds (odd3)
            odds_2_elem = match_element.find('div', class_=lambda x: x and 'event__odd--odd3' in x)
            if odds_2_elem:
                match_data['odds_2'] = odds_2_elem.get_text(strip=True)

            if match_data['odds_1'] == "-" and match_data['odds_x'] == "-" and match_data['odds_2'] == "-":
                match_data["time"] = "Full-time"

            
            # Fallback: if specific odds not found, try generic search
            if not match_data['odds_1'] or not match_data['odds_x'] or not match_data['odds_2']:
                if match_data['odds_1'] == "-" and match_data['odds_x'] == "-" and match_data['odds_2'] == "-":
                    raise ValueError("Odds are all '-'")
                odds_elements = match_element.find_all('div', class_=lambda x: x and 'odd' in x.lower())
                if odds_elements:
                    odds_list = [odd.get_text(strip=True) for odd in odds_elements[:3]]
                    if len(odds_list) >= 1 and not match_data['odds_1']:
                        match_data['odds_1'] = odds_list[0]
                    if len(odds_list) >= 2 and not match_data['odds_x']:
                        match_data['odds_x'] = odds_list[1]
                    if len(odds_list) >= 3 and not match_data['odds_2']:
                        match_data['odds_2'] = odds_list[2]
            else:
                if match_data['odds_1'] == "-" and match_data['odds_x'] == "-" and match_data['odds_2'] == "-":
                    match_data["time"] = "Full-time"
                    match_data["status"] = "Full-time"
            
            return match_data
        except Exception as e:
            logger.error(f"Error extracting match data: {e}")
            return None
    
    def scrape_matches(self, league_url: str = None) -> List[Dict]:
        """Scrape matches from the homepage or a specific league"""
        url = league_url or self.base_url
        logger.info(f"Scraping matches from: {url}")
        
        soup = self.fetch_page(url)
        if not soup:
            return []
        
        matches = []
        
        # Find all match elements by class "event__match"
        match_elements = soup.find_all('div', class_=lambda x: x and 'event__match' in x)
        
        logger.info(f"Found {len(match_elements)} match elements")
        
        # Debug: Print classes of first few elements
        for i, elem in enumerate(match_elements[:3]):
            logger.info(f"Match {i+1} classes: {elem.get('class', [])}")
        
        if match_elements:
            # Process all matches directly without trying to group by league sections
            logger.info("Processing all match elements directly")
            for i, match_elem in enumerate(match_elements):  # Process all matches, not just 10
                logger.info(f"Processing match element {i+1}/{len(match_elements)}")
                match_data = self.extract_match_data(match_elem)
                if match_data and (match_data['home_team'] or match_data['away_team']):
                    matches.append(match_data)
                    logger.info(f"Added match: {match_data['home_team']} vs {match_data['away_team']}")
                else:
                    logger.warning(f"Match data incomplete for element {i+1}")
        else:
            logger.warning("No match elements found")
        
        self.matches.extend(matches)
        logger.info(f"Extracted {len(matches)} matches")
        return matches
    
    def scrape_league(self, league_name: str) -> List[Dict]:
        """Scrape matches from a specific league"""
        # Common leagues and their paths
        leagues = {
            'premier-league': '/england/premier-league/',
            'laliga': '/spain/laliga/',
            'serie-a': '/italy/serie-a/',
            'bundesliga': '/germany/bundesliga/',
            'ligue-1': '/france/ligue-1/',
            'champions-league': '/europe/champions-league/',
        }
        
        league_path = leagues.get(league_name.lower())
        if not league_path:
            logger.warning(f"League {league_name} not found in predefined list")
            league_path = f"/{league_name}/"
        
        url = self.base_url + league_path
        return self.scrape_matches(url)
    
    def save_to_json(self, filename: str = None) -> str:
        """Save scraped matches to JSON file"""
        if not filename:
            filename = "soccerway_matches.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.matches, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.matches)} matches to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return None
    
    def print_matches(self, limit: int = 10):
        """Print matches to console"""
        print(f"\n{'='*100}")
        print(f"Scraped Matches (showing {min(limit, len(self.matches))} of {len(self.matches)})")
        print(f"{'='*100}\n")
        
        for i, match in enumerate(self.matches[:limit], 1):
            print(f"{i}. {match['league']}")
            print(f"   {match['home_team']} vs {match['away_team']}")
            
            # Display score prominently for finished matches
            if match['home_score'] is not None and match['away_score'] is not None:
                print(f"   ⚽ FINAL SCORE: {match['home_score']} - {match['away_score']} ⚽")
            elif match['status'] and 'full-time' in match['status'].lower():
                print(f"   Status: {match['status']} (Score not available)")
            elif match['time']:
                print(f"   Time: {match['time']}")
            
            if match['status'] and match['status'] != match['time']:
                print(f"   Status: {match['status']}")
            
            if match['odds_1'] or match['odds_x'] or match['odds_2']:
                odds_display = f"{match['odds_1']} / {match['odds_x']} / {match['odds_2']}"
                if odds_display != "None / None / None":
                    print(f"   Odds (1/X/2): {odds_display}")
            print()


def main():
    """Main function to run the scraper"""
    scraper = SoccerwayScraper(headless=True)
    
    try:
        # Initialize driver
        scraper.init_driver()
        
        # Scrape homepage
        logger.info("Starting Soccerway scraper...")
        matches = scraper.scrape_matches()
        
        if matches:
            # Print summary
            scraper.print_matches(limit=5)
            
            # Save to JSON
            json_file = scraper.save_to_json()
            
            print(f"\nTotal matches scraped: {len(matches)}")
            print(f"Saved to: {json_file}")
        else:
            logger.warning("No matches found")
    
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    
    finally:
        # Always close the driver
        scraper.close_driver()


if __name__ == "__main__":
    main()
