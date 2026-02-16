#!/usr/bin/env python3
"""
DuckDB Data Pipeline for Advanced Quant Model
============================================

High-performance data pipeline that migrates existing JSON data 
into DuckDB for quantitative analysis.

Based on empirical findings:
- BTTS: 66.7% WR, +0.30 units (PROFITABLE)
- O/U 2.5: 43.3% WR, -1.15 units (BANNED)
- O/U 1.5: New focus as 2.5 replacement
- Handicap: New market exploration
"""

import duckdb
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional

class DuckDBDataPipeline:
    """
    Efficient data pipeline for quantitative betting analysis
    """
    
    def __init__(self, db_path: str = "football_data.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"📊 DuckDB Pipeline initialized: {db_path}")
    
    def migrate_json_data(self, data_directory: str = "."):
        """
        Migrate existing JSON data files into DuckDB
        """
        print("🔄 Starting JSON to DuckDB migration...")
        
        data_path = Path(data_directory)
        
        # Find all data files
        fixture_files = list(data_path.glob("api_football_fixtures_*.json"))
        merged_files = list(data_path.glob("api_football_merged_*.json"))
        odds_files = list(data_path.glob("api_football_odds_*.json"))
        
        print(f"📁 Found {len(fixture_files)} fixture files")
        print(f"📁 Found {len(merged_files)} merged files")
        print(f"📁 Found {len(odds_files)} odds files")
        
        # Process each file type
        if merged_files:
            self._process_merged_files(merged_files)
        elif fixture_files:
            self._process_fixture_files(fixture_files)
            
        if odds_files:
            self._process_odds_files(odds_files)
        
        # Generate team statistics
        self._generate_team_statistics()
        
        print("✅ Migration completed successfully")
    
    def _process_merged_files(self, merged_files: List[Path]):
        """
        Process merged data files (fixtures + odds combined)
        """
        print("🔧 Processing merged data files...")
        
        all_fixtures = []
        all_odds = []
        
        for file_path in merged_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'response' in data:
                    fixtures_data = data['response']
                else:
                    fixtures_data = data if isinstance(data, list) else [data]
                
                for fixture in fixtures_data:
                    # Extract fixture info
                    fixture_record = self._extract_fixture_data(fixture)
                    if fixture_record:
                        all_fixtures.append(fixture_record)
                    
                    # Extract odds info
                    odds_records = self._extract_odds_data(fixture)
                    all_odds.extend(odds_records)
                
                print(f"   ✅ Processed {file_path.name}")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {e}")
        
        # Bulk insert fixtures
        if all_fixtures:
            fixtures_df = pd.DataFrame(all_fixtures)
            self._bulk_insert_fixtures(fixtures_df)
        
        # Bulk insert odds
        if all_odds:
            odds_df = pd.DataFrame(all_odds)
            self._bulk_insert_odds(odds_df)
    
    def _extract_fixture_data(self, fixture: Dict) -> Optional[Dict]:
        """
        Extract fixture information from API response
        """
        try:
            fixture_info = fixture.get('fixture', {})
            teams_info = fixture.get('teams', {})
            goals_info = fixture.get('goals', {})
            league_info = fixture.get('league', {})
            
            # Only process completed matches
            if fixture_info.get('status', {}).get('short') != 'FT':
                return None
            
            return {
                'id': fixture_info.get('id'),
                'date': fixture_info.get('date', '').split('T')[0],  # Extract date only
                'home_team': teams_info.get('home', {}).get('name', ''),
                'away_team': teams_info.get('away', {}).get('name', ''),
                'league_id': league_info.get('id'),
                'season': league_info.get('season'),
                'home_goals': goals_info.get('home'),
                'away_goals': goals_info.get('away'),
                'status': 'FT'
            }
        
        except Exception:
            return None
    
    def _extract_odds_data(self, fixture: Dict) -> List[Dict]:
        """
        Extract odds data with focus on target markets
        """
        odds_records = []
        fixture_id = fixture.get('fixture', {}).get('id')
        
        if not fixture_id:
            return odds_records
        
        # Target markets based on empirical findings
        target_markets = {
            'Both Teams Score': 'btts',
            'Goals Over/Under': 'over_under',
            'Asian Handicap': 'handicap'
        }
        
        bookmakers = fixture.get('bookmakers', [])
        
        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get('name', 'Unknown')
            markets = bookmaker.get('bets', [])
            
            for market in markets:
                market_name = market.get('name', '')
                
                if market_name in target_markets:
                    market_type = target_markets[market_name]
                    
                    for outcome in market.get('values', []):
                        odds_record = {
                            'fixture_id': fixture_id,
                            'market_type': market_type,
                            'selection': outcome.get('value', ''),
                            'odds': outcome.get('odd'),
                            'bookmaker': bookmaker_name
                        }
                        
                        # Special handling for Over/Under markets
                        if market_type == 'over_under':
                            selection = outcome.get('value', '')
                            # Focus on 1.5 goals (replacement for failing 2.5)
                            if '1.5' in selection:
                                odds_record['market_type'] = 'over_under_15'
                                odds_records.append(odds_record)
                            elif '2.5' in selection:
                                # Still collect 2.5 data for comparison
                                odds_record['market_type'] = 'over_under_25'
                                odds_records.append(odds_record)
                        else:
                            odds_records.append(odds_record)
        
        return odds_records
    
    def _bulk_insert_fixtures(self, df: pd.DataFrame):
        """
        Efficiently insert fixture data
        """
        print(f"📊 Inserting {len(df)} fixtures...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Insert using DuckDB's efficient bulk insert
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO fixtures SELECT * FROM df"
            )
            print(f"   ✅ {len(df)} fixtures inserted")
        except Exception as e:
            print(f"   ❌ Error inserting fixtures: {e}")
    
    def _bulk_insert_odds(self, df: pd.DataFrame):
        """
        Efficiently insert odds data
        """
        print(f"📊 Inserting {len(df)} odds records...")
        
        # Clean odds data
        df = df.dropna(subset=['odds'])
        df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
        df = df.dropna(subset=['odds'])
        
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO market_odds (fixture_id, market_type, selection, odds, bookmaker) SELECT fixture_id, market_type, selection, odds, bookmaker FROM df"
            )
            print(f"   ✅ {len(df)} odds records inserted")
        except Exception as e:
            print(f"   ❌ Error inserting odds: {e}")
    
    def _generate_team_statistics(self):
        """
        Generate team performance statistics for model features
        """
        print("📈 Generating team statistics...")
        
        # Home team statistics
        home_stats_query = """
        INSERT OR REPLACE INTO team_stats (
            team_name, league_id, season,
            goals_scored_home, goals_conceded_home,
            btts_frequency_home, avg_match_goals
        )
        SELECT 
            home_team as team_name,
            league_id,
            season,
            AVG(CAST(home_goals AS DECIMAL)) as goals_scored_home,
            AVG(CAST(away_goals AS DECIMAL)) as goals_conceded_home,
            AVG(CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1.0 ELSE 0.0 END) as btts_frequency_home,
            AVG(CAST(home_goals + away_goals AS DECIMAL)) as avg_match_goals
        FROM fixtures 
        WHERE status = 'FT' 
        AND home_goals IS NOT NULL 
        AND away_goals IS NOT NULL
        GROUP BY home_team, league_id, season
        """
        
        # Away team statistics  
        away_stats_query = """
        INSERT OR REPLACE INTO team_stats (
            team_name, league_id, season,
            goals_scored_away, goals_conceded_away,
            btts_frequency_away
        )
        SELECT 
            away_team as team_name,
            league_id,
            season,
            AVG(CAST(away_goals AS DECIMAL)) as goals_scored_away,
            AVG(CAST(home_goals AS DECIMAL)) as goals_conceded_away,
            AVG(CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1.0 ELSE 0.0 END) as btts_frequency_away
        FROM fixtures 
        WHERE status = 'FT' 
        AND home_goals IS NOT NULL 
        AND away_goals IS NOT NULL
        GROUP BY away_team, league_id, season
        ON CONFLICT (team_name, league_id, season) 
        DO UPDATE SET
            goals_scored_away = EXCLUDED.goals_scored_away,
            goals_conceded_away = EXCLUDED.goals_conceded_away,
            btts_frequency_away = EXCLUDED.btts_frequency_away
        """
        
        try:
            self.conn.execute(home_stats_query)
            self.conn.execute(away_stats_query)
            
            # Generate handicap performance statistics
            handicap_query = """
            UPDATE team_stats SET
                handicap_performance = (
                    SELECT AVG(CASE WHEN h.home_goals > h.away_goals THEN 1.0 ELSE 0.0 END)
                    FROM fixtures h 
                    WHERE h.home_team = team_stats.team_name 
                    AND h.league_id = team_stats.league_id
                ),
                margin_of_victory = (
                    SELECT AVG(ABS(h.home_goals - h.away_goals))
                    FROM fixtures h 
                    WHERE (h.home_team = team_stats.team_name OR h.away_team = team_stats.team_name)
                    AND h.league_id = team_stats.league_id
                )
            """
            
            self.conn.execute(handicap_query)
            
            # Get statistics count
            result = self.conn.execute("SELECT COUNT(*) as count FROM team_stats").fetchone()
            print(f"   ✅ Generated statistics for {result[0]} teams")
            
        except Exception as e:
            print(f"   ❌ Error generating team statistics: {e}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of data in DuckDB
        """
        summary = {}
        
        try:
            # Fixtures summary
            fixture_result = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_fixtures,
                    COUNT(DISTINCT league_id) as leagues,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    AVG(home_goals + away_goals) as avg_total_goals,
                    AVG(CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1.0 ELSE 0.0 END) as btts_frequency
                FROM fixtures WHERE status = 'FT'
            """).fetchone()
            
            summary['fixtures'] = {
                'total': fixture_result[0],
                'leagues': fixture_result[1], 
                'date_range': f"{fixture_result[2]} to {fixture_result[3]}",
                'avg_total_goals': round(fixture_result[4] or 0, 2),
                'btts_frequency': round(fixture_result[5] or 0, 3)
            }
            
            # Odds summary
            odds_result = self.conn.execute("""
                SELECT 
                    market_type,
                    COUNT(*) as records,
                    COUNT(DISTINCT fixture_id) as fixtures_covered
                FROM market_odds 
                GROUP BY market_type
                ORDER BY records DESC
            """).fetchall()
            
            summary['odds'] = [{               
                'market': row[0],
                'records': row[1],
                'fixture_coverage': row[2]
            } for row in odds_result]
            
            # Team stats summary
            team_result = self.conn.execute("SELECT COUNT(*) FROM team_stats").fetchone()
            summary['team_stats'] = {'teams': team_result[0]}
            
        except Exception as e:
            print(f"❌ Error getting data summary: {e}")
        
        return summary
    
    def export_training_data(self, output_file: str = "training_data.parquet"):
        """
        Export processed training data for model training
        """
        print(f"📤 Exporting training data to {output_file}..")
        
        query = """
        SELECT 
            f.*,
            -- Market outcomes
            CASE WHEN f.home_goals > 0 AND f.away_goals > 0 THEN 1 ELSE 0 END as btts_result,
            CASE WHEN f.home_goals + f.away_goals > 1.5 THEN 1 ELSE 0 END as over_15_result,
            CASE WHEN f.home_goals + f.away_goals < 1.5 THEN 1 ELSE 0 END as under_15_result,
            CASE WHEN f.home_goals - f.away_goals > 0.5 THEN 1 ELSE 0 END as home_handicap_minus_05,
            
            -- Team statistics
            hs.goals_scored_home,
            hs.goals_conceded_home,
            hs.btts_frequency_home,
            hs.handicap_performance as home_handicap_perf,
            
            as_.goals_scored_away,
            as_.goals_conceded_away, 
            as_.btts_frequency_away,
            as_.handicap_performance as away_handicap_perf
            
        FROM fixtures f
        LEFT JOIN team_stats hs ON f.home_team = hs.team_name AND f.league_id = hs.league_id
        LEFT JOIN team_stats as_ ON f.away_team = as_.team_name AND f.league_id = as_.league_id
        WHERE f.status = 'FT' 
        AND f.home_goals IS NOT NULL 
        AND f.away_goals IS NOT NULL
        ORDER BY f.date DESC
        """
        
        try:
            df = self.conn.execute(query).fetchdf()
            df.to_parquet(output_file)
            print(f"✅ Exported {len(df)} training records to {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error exporting training data: {e}")
            return None
    
    def close_connection(self):
        """
        Close DuckDB connection
        """
        if self.conn:
            self.conn.close()
            print("📊 DuckDB connection closed")


def main():
    """
    Main pipeline execution
    """
    print("📊 DUCKDB DATA PIPELINE")
    print("=" * 50)
    
    pipeline = DuckDBDataPipeline()
    
    try:
        # Migrate JSON data
        pipeline.migrate_json_data()
        
        # Show data summary
        summary = pipeline.get_data_summary()
        
        print("\n📈 DATA SUMMARY:")
        print(f"Fixtures: {summary.get('fixtures', {}).get('total', 0)}")
        print(f"Date Range: {summary.get('fixtures', {}).get('date_range', 'N/A')}")
        print(f"Avg Goals: {summary.get('fixtures', {}).get('avg_total_goals', 0)}")
        print(f"BTTS Frequency: {summary.get('fixtures', {}).get('btts_frequency', 0):.1%}")
        
        print("\n📊 MARKET COVERAGE:")
        for market in summary.get('odds', []):
            print(f"  {market['market']}: {market['records']} records")
        
        # Export training data
        training_file = pipeline.export_training_data()
        if training_file:
            print(f"\n✅ Ready for quantitative model training with {training_file}")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    
    finally:
        pipeline.close_connection()


if __name__ == "__main__":
    main()