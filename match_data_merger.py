import json
import os
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchDataMerger:
    """
    Merges data from soccerway (match info + odds) with soccerstats (team statistics)
    """
    
    def __init__(self):
        self.team_name_mappings = self._init_team_name_mappings()
    
    def _init_team_name_mappings(self) -> Dict[str, str]:
        """Initialize common team name variations for better matching"""
        return {
            # Common abbreviations and variations
            "Man City": "Manchester City",
            "Man Utd": "Manchester United", 
            "Man United": "Manchester United",
            "Newcastle": "Newcastle United",
            "West Ham": "West Ham United",
            "Tottenham": "Tottenham Hotspur",
            "Brighton": "Brighton & Hove Albion",
            "Wolves": "Wolverhampton Wanderers",
            "Leicester": "Leicester City",
            "Crystal Palace": "Crystal Palace",
            "Sheffield Utd": "Sheffield United",
            "Sheffield United": "Sheffield United",
            "Norwich": "Norwich City",
            "Birmingham": "Birmingham City",
            "QPR": "Queens Park Rangers",
            "Preston": "Preston North End",
            # Add more mappings as needed
        }
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for better matching"""
        if not team_name:
            return ""
        
        # Remove extra spaces and convert to title case
        normalized = ' '.join(team_name.strip().split())
        
        # Check if there's a mapping
        if normalized in self.team_name_mappings:
            return self.team_name_mappings[normalized]
        
        return normalized
    
    def normalize_league_name(self, league_name: str) -> str:
        """Normalize league names for matching"""
        if not league_name:
            return ""
        
        # Remove common prefixes and suffixes
        league = league_name.lower()
        league = re.sub(r'^(england|italy|spain|germany|france|europe)\s*-?\s*', '', league)
        league = re.sub(r'\s*(league|division|serie|liga|bundesliga|ligue)?\s*\d*$', '', league)
        
        # Common league mappings
        league_mappings = {
            'premier league': 'Premier League',
            'championship': 'Championship', 
            'league one': 'League One',
            'league two': 'League Two',
            'serie a': 'Serie A',
            'la liga': 'LaLiga',
            'ligue 1': 'Ligue 1',
            'bundesliga': 'Bundesliga',
            'champions league': 'Champions League',
            'europa league': 'Europa League'
        }
        
        return league_mappings.get(league.strip(), league_name)
    
    def group_matches_by_time(self, data_source: List[Dict]) -> Dict[str, List[Dict]]:
        """Group matches by their kick-off time for more efficient matching"""
        time_groups = {}
        
        for match in data_source:
            # Extract clean time from the match
            if 'time' in match:  # Soccerway format
                time_key = self._extract_clean_time(match.get('time', ''))
            elif 'team1' in match and 'Time' in match.get('team1', {}):  # Soccerstats format
                time_key = self._extract_clean_time(match['team1'].get('Time', ''))
            else:
                time_key = 'unknown'
            
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(match)
        
        return time_groups
    
    def _extract_clean_time(self, time_str: str) -> str:
        """Extract clean time format (HH:MM) from various time strings"""
        if not time_str:
            return 'unknown'
        
        # Handle common time formats and extract HH:MM
        import re
        time_match = re.search(r'(\d{1,2}:\d{2})', str(time_str))
        if time_match:
            return time_match.group(1)
        
        # Handle finished matches, live matches, etc.
        if time_str.lower() in ['full-time', 'ft', 'finished', 'live', 'ht', 'half-time']:
            return 'finished_or_live'
        
        return 'unknown'
    
    def find_matching_stats_by_time(self, soccerway_match: Dict, time_grouped_stats: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Find matching team statistics using time-based grouping for better accuracy"""
        sw_league = self.normalize_league_name(soccerway_match.get('league', ''))
        sw_home = self.normalize_team_name(soccerway_match.get('home_team', ''))
        sw_away = self.normalize_team_name(soccerway_match.get('away_team', ''))
        sw_time = self._extract_clean_time(soccerway_match.get('time', ''))
        
        # Get matches for the same time slot
        time_matches = time_grouped_stats.get(sw_time, [])
        
        # If no matches for exact time, try 'unknown' group as fallback
        if not time_matches and sw_time != 'unknown':
            time_matches = time_grouped_stats.get('unknown', [])
        
        # Search through time-filtered matches
        for match_data in time_matches:
            # Extract league info from match_data structure
            league_name = ""
            team1 = {}
            team2 = {}
            
            # Handle the nested structure from soccerstats
            if 'league_name' in match_data:  # Direct league match
                league_name = match_data.get('league_name', '')
                continue  # Skip league headers
            elif 'team1' in match_data and 'team2' in match_data:  # Individual match
                team1 = match_data.get('team1', {})
                team2 = match_data.get('team2', {})
                # We need to get league name from context - we'll pass it separately
            
            if not team1 or not team2:
                continue
                
            ss_team1_name = self.normalize_team_name(team1.get('Name', ''))
            ss_team2_name = self.normalize_team_name(team2.get('Name', ''))
            
            # Check if teams match (considering home/away order)
            home_away_match = (
                self._teams_match(sw_home, ss_team1_name) and 
                self._teams_match(sw_away, ss_team2_name)
            )
            
            away_home_match = (
                self._teams_match(sw_home, ss_team2_name) and 
                self._teams_match(sw_away, ss_team1_name)
            )
            
            if home_away_match or away_home_match:
                return {
                    'team1_stats': team1,
                    'team2_stats': team2,
                    'home_is_team1': home_away_match
                }
        
        return None
    
    def _leagues_match(self, league1: str, league2: str) -> bool:
        """Check if two league names refer to the same league"""
        if not league1 or not league2:
            return False
        
        l1 = league1.lower().strip()
        l2 = league2.lower().strip()
        
        # Exact match
        if l1 == l2:
            return True
        
        # Partial match (one contains the other)
        if l1 in l2 or l2 in l1:
            return True
        
        return False
    
    def _teams_match(self, team1: str, team2: str) -> bool:
        """Check if two team names refer to the same team"""
        if not team1 or not team2:
            return False
        
        t1 = team1.lower().strip()
        t2 = team2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return True
        
        # Check if one is contained in the other (handles abbreviations)
        if len(t1) >= 3 and len(t2) >= 3:
            if t1 in t2 or t2 in t1:
                return True
        
        return False
    
    def _times_match(self, time1: str, time2: str) -> bool:
        """Check if two times match (if both are provided)"""
        if not time1 or not time2:
            return True  # If time not available, don't use it for matching
        
        # Extract time from strings like "19:45" or "19:45Analysis"
        def extract_time(time_str):
            match = re.search(r'(\d{1,2}:\d{2})', str(time_str))
            return match.group(1) if match else ""
        
        t1 = extract_time(time1)
        t2 = extract_time(time2)
        
        return t1 == t2 if t1 and t2 else True
    
    def merge_match_data(self, soccerway_file: str, soccerstats_file: str, output_file: str = None) -> List[Dict]:
        """Merge data from both sources using time-based grouping for better matching"""
        
        # Load data
        with open(soccerway_file, 'r') as f:
            soccerway_data = json.load(f)
        
        with open(soccerstats_file, 'r') as f:
            soccerstats_data = json.load(f)
        
        # Flatten soccerstats data and group by time
        flattened_stats = []
        for league_data in soccerstats_data.get('leagues', []):
            league_name = league_data.get('league_name', '')
            for match in league_data.get('matches', []):
                # Add league context to each match
                match_with_league = match.copy()
                match_with_league['source_league'] = league_name
                flattened_stats.append(match_with_league)
        
        # Group soccerstats matches by time
        time_grouped_stats = self.group_matches_by_time(flattened_stats)
        
        merged_matches = []
        stats_found = 0
        
        logger.info(f"Processing {len(soccerway_data)} matches from soccerway data")
        logger.info(f"Grouped soccerstats into {len(time_grouped_stats)} time slots")
        
        # Print time group summary
        for time_key, matches in time_grouped_stats.items():
            logger.info(f"  {time_key}: {len(matches)} matches")
        
        for sw_match in soccerway_data:
            # Create enhanced match data starting with soccerway data
            enhanced_match = sw_match.copy()
            
            # Try to find matching statistics using time-based approach
            stats_match = self.find_matching_stats_by_time(sw_match, time_grouped_stats)
            
            if stats_match:
                # Add team statistics to the match
                if stats_match['home_is_team1']:
                    enhanced_match['home_team_stats'] = stats_match['team1_stats']
                    enhanced_match['away_team_stats'] = stats_match['team2_stats']
                else:
                    enhanced_match['home_team_stats'] = stats_match['team2_stats']
                    enhanced_match['away_team_stats'] = stats_match['team1_stats']
                
                enhanced_match['stats_source'] = 'soccerstats'
                stats_found += 1
                
                logger.debug(f"✓ Found stats for: {sw_match.get('home_team')} vs {sw_match.get('away_team')} at {sw_match.get('time')}")
            else:
                enhanced_match['home_team_stats'] = None
                enhanced_match['away_team_stats'] = None
                enhanced_match['stats_source'] = None
                
                logger.debug(f"✗ No stats found for: {sw_match.get('home_team')} vs {sw_match.get('away_team')} at {sw_match.get('time')}")
            
            merged_matches.append(enhanced_match)
        
        logger.info(f"Successfully merged {stats_found}/{len(soccerway_data)} matches with statistics")
        logger.info(f"Improvement: {round((stats_found / len(soccerway_data) * 100), 1)}% coverage")
        
        # Save merged data if output file specified
        if output_file:
            try:
                # Get absolute path to ensure proper file writing in cloud environments
                abs_filename = os.path.abspath(output_file)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
                
                # Write to temporary file first, then rename (atomic operation)
                temp_filename = abs_filename + ".tmp"
                
                with open(temp_filename, 'w') as f:
                    json.dump(merged_matches, f, indent=2)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic rename
                os.rename(temp_filename, abs_filename)
                
                logger.info(f"Merged data saved to: {abs_filename}")
            except Exception as e:
                logger.error(f"Error saving merged data: {e}")
                # Clean up temp file if it exists
                temp_filename = os.path.abspath(output_file) + ".tmp"
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
        
        return merged_matches
    
    def generate_match_report(self, merged_matches: List[Dict]) -> Dict:
        """Generate a summary report of the merged data"""
        total_matches = len(merged_matches)
        with_stats = sum(1 for m in merged_matches if m.get('home_team_stats'))
        
        leagues_with_stats = {}
        for match in merged_matches:
            league = match.get('league', 'Unknown')
            if league not in leagues_with_stats:
                leagues_with_stats[league] = {'total': 0, 'with_stats': 0}
            
            leagues_with_stats[league]['total'] += 1
            if match.get('home_team_stats'):
                leagues_with_stats[league]['with_stats'] += 1
        
        return {
            'summary': {
                'total_matches': total_matches,
                'matches_with_stats': with_stats,
                'coverage_percentage': round((with_stats / total_matches * 100), 2) if total_matches > 0 else 0
            },
            'by_league': leagues_with_stats,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Example usage"""
    merger = MatchDataMerger()
    
    # Merge the data
    merged_matches = merger.merge_match_data(
        'soccerway_matches.json',
        'soccerstats_matches.json', 
        'merged_match_data.json'
    )
    
    # Generate report
    report = merger.generate_match_report(merged_matches)
    
    print("\\n=== MATCH DATA MERGER REPORT ===")
    print(f"Total matches processed: {report['summary']['total_matches']}")
    print(f"Matches with statistics: {report['summary']['matches_with_stats']}")
    print(f"Coverage: {report['summary']['coverage_percentage']}%")
    
    print("\\n=== BY LEAGUE ===")
    for league, stats in report['by_league'].items():
        coverage = round((stats['with_stats'] / stats['total'] * 100), 1) if stats['total'] > 0 else 0
        print(f"{league}: {stats['with_stats']}/{stats['total']} ({coverage}%)")
    
    # Save report
    try:
        # Get absolute path to ensure proper file writing in cloud environments
        abs_filename = os.path.abspath('merger_report.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_filename = abs_filename + ".tmp"
        
        with open(temp_filename, 'w') as f:
            json.dump(report, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename
        os.rename(temp_filename, abs_filename)
    except Exception as e:
        logger.error(f"Error saving merger report: {e}")
        # Clean up temp file if it exists
        temp_filename = os.path.abspath('merger_report.json') + ".tmp"
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

if __name__ == "__main__":
    main()