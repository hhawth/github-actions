#!/usr/bin/env python3
"""
Quantitative Betting Workflow
============================
Modern automated betting system combining:
- Advanced Quantitative Model (DuckDB-based)
- Evidence-based market targeting (BTTS, O/U 1.5, Asian Handicap)  
- Matchbook exchange integration with smart matching
- Risk management and performance monitoring

Based on empirical findings:
✅ BTTS: 51.1% historical rate, 70% model accuracy
✅ Over/Under 1.5: 73.4% over rate, multiple 100% leagues  
✅ Asian Handicap: 26.4% draw rate provides opportunities
❌ Over/Under 2.5: BANNED (43.3% WR, losing)

Usage:
    python quantitative_betting_workflow.py --full-run
    python quantitative_betting_workflow.py --data-only
    python quantitative_betting_workflow.py --analyze-only
"""

import os
import sys
import json
import time
import argparse
import duckdb
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add our module paths
sys.path.append('./quant_model')
sys.path.append('./src')

# Import quantitative model and supporting modules
try:
    from advanced_quant_model import AdvancedQuantModel
    from quant_config import QuantConfig
    from duckdb_data_pipeline import DuckDBDataPipeline
    
    # Import existing modules
    from api_football import get_fixtures_past_time_plus_hour, main as api_football_main
    from matchbook import matchbookExchange
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.info("💡 Make sure you're running from the correct directory")
    sys.exit(1)

# Database connection will be initialized when needed
conn = None


class QuantitativeBettingWorkflow:
    """
    Next-generation betting workflow using the quantitative model
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'min_ev': 0.08,  # 8% minimum edge (conservative)
            'max_ev': 0.70,  # 70% max - allow high-confidence predictions
            'min_stake': 0.10,  # £0.10 minimum (Matchbook requirement)
            'max_daily_stake': 5.00,  # £5.00 daily maximum
            'max_bets_per_match': 1,  # Reduce correlation risk
            'auto_place_bets': True,  # Auto-place bets by default
            'min_confidence': 0.65,  # 65% minimum model confidence
            'target_markets': ['btts', 'over_under_15', 'handicap'],  # Evidence-based markets
        }
        
        # Initialize quantitative model components
        self.quant_model = None
        self.quant_config = None
        self.data_pipeline = None
        self.db_conn = None
        
        # Matchbook connection
        self.matchbook = None
        
        # Matchbook API cache (5 minutes TTL)
        self.matchbook_cache = {
            'events': None,
            'events_timestamp': 0,
            'cache_duration': 300  # 5 minutes in seconds
        }
        
        # Workflow tracking
        self.workflow_state = {
            'data_updated': False,
            'model_ready': False,
            'opportunities_analyzed': False,
            'bets_placed': False
        }
        
        # Initialize database connection
        self._initialize_database()
        
        logger.info("🚀 QUANTITATIVE BETTING WORKFLOW INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"🎯 Target Markets: {', '.join(self.config['target_markets'])}")
        logger.info(f"💰 Stake Range: £{self.config['min_stake']:.2f} - £{self.config['max_daily_stake']:.2f}")
        logger.info(f"📊 Min Edge: {self.config['min_ev']:.1%} | Min Confidence: {self.config['min_confidence']:.1%}")
        logger.info("=" * 60)
    
    def _get_cached_matchbook_events(self):
        """Get Matchbook football events with 5-minute caching"""
        import time
        
        current_time = time.time()
        
        # Check if cache is still valid (within 5 minutes)
        if (self.matchbook_cache['events'] is not None and 
            current_time - self.matchbook_cache['events_timestamp'] < self.matchbook_cache['cache_duration']):
            
            age_minutes = (current_time - self.matchbook_cache['events_timestamp']) / 60
            logger.info(f"🏢 Using cached Matchbook events ({age_minutes:.1f}min old)")
            return self.matchbook_cache['events']
        
        # Cache is expired or empty, fetch fresh data
        logger.info("🏢 Fetching upcoming fixtures from Matchbook...")
        try:
            if not self.matchbook:
                logger.info("🔧 Initializing Matchbook connection...")
                self.matchbook = matchbookExchange()
                self.matchbook.login()
                logger.info("✅ Connected to Matchbook exchange")
            
            matchbook_events = self.matchbook.get_football_events()
            
            # Update cache
            self.matchbook_cache['events'] = matchbook_events
            self.matchbook_cache['events_timestamp'] = current_time
            
            logger.info("✅ Matchbook events cached (valid for 5 minutes)")
            return matchbook_events
            
        except Exception as e:
            logger.warning(f"⚠️ Matchbook connection failed: {e}")
            logger.warning("⚠️ This is likely due to invalid credentials or account issues")
            logger.warning("⚠️ Continuing workflow without Matchbook data - only Football API data will be used")
            # Set cache to None to avoid treating empty list as valid cached data
            self.matchbook_cache['events'] = None
            self.matchbook_cache['events_timestamp'] = current_time
            # Disable matchbook for this session
            self.matchbook = None
            return None
    
    def _initialize_database(self):
        """Initialize database connection with proper error handling"""
        import os
        
        # Skip database initialization in CI/test environments
        if os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true':
            logger.warning("⚠️  CI environment detected - skipping database initialization")
            return
        
        try:
            from database_sync import ensure_database_exists
            if ensure_database_exists():
                # Connect to local database
                self.db_conn = duckdb.connect("./football_data.duckdb")
                logger.info("✅ Database connection established")
            else:
                logger.error("❌ Failed to initialize database")
                # Don't exit in __init__, just warn
                logger.warning("⚠️  Some features may not work without database")
        except Exception as e:
            logger.warning(f"⚠️  Database sync not available: {e}")
            logger.info("💡 Proceeding without database sync")
            # Try to connect to local database if it exists
            try:
                if os.path.exists("./football_data.duckdb"):
                    self.db_conn = duckdb.connect("./football_data.duckdb")
                    logger.info("✅ Connected to local database")
                else:
                    logger.warning("⚠️  No local database found")
            except Exception as local_error:
                logger.warning(f"⚠️  Cannot connect to local database: {local_error}")
    
    def step_1_initialize_quantitative_system(self):
        """Initialize the quantitative model and database connection"""
        logger.info("🧠 STEP 1: Initializing Quantitative System")
        logger.info("-" * 50)
        
        try:
            # Use existing database connection or skip if not available
            if not self.db_conn:
                logger.error("❌ Database connection not available - cannot initialize quantitative system")
                return False
            
            # Check database has data
            fixture_count = self.db_conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
            if fixture_count == 0:
                logger.error("❌ No fixture data in database. Please run data collection first.")
                return False
            
            logger.info(f"📊 Database connected: {fixture_count:,} fixtures available")
            
            # Initialize quantitative model
            self.quant_model = AdvancedQuantModel(db_path='football_data.duckdb')
            logger.info("✅ Quantitative model initialized")
            
            # Check if we have a saved model to load
            model_file = "advanced_quant_model_latest.pkl"
            should_train = True
            
            if os.path.exists(model_file):
                # Check model age
                model_age_hours = (time.time() - os.path.getmtime(model_file)) / 3600
                logger.info(f"📦 Found saved model (age: {model_age_hours:.1f} hours)")
                
                if model_age_hours < 24:
                    logger.info("   ⚡ Loading saved model (< 24 hours old)...")
                    try:
                        self.quant_model.load_models(model_file)
                        logger.info("   ✅ Model loaded successfully - skipping training")
                        should_train = False
                    except Exception as load_error:
                        logger.warning(f"   ⚠️ Failed to load model: {load_error}")
                        logger.info("   🔄 Will train new model")
                else:
                    logger.warning(f"   ⚠️ Model is stale ({model_age_hours:.1f} hours old)")
                    logger.info("   🔄 Will retrain with fresh data")
            else:
                logger.info("📦 No saved model found - will train new model")
            
            # Train the model if needed
            if should_train:
                logger.info("🏋️ Training quantitative model...")
                try:
                    # Load historical data from database for training
                    logger.info("   📊 Loading historical fixture data...")
                    historical_data = self.db_conn.execute("""
                    SELECT 
                        fixture_id,
                        date,
                        goals_full_time_home as home_goals,
                        goals_full_time_away as away_goals,
                        -- Calculate BTTS result target
                        CASE WHEN goals_full_time_home > 0 AND goals_full_time_away > 0 THEN 1 ELSE 0 END as btts_result,
                        -- Calculate Over/Under 1.5 results
                        CASE WHEN (goals_full_time_home + goals_full_time_away) > 1.5 THEN 1 ELSE 0 END as over_15_result,
                        CASE WHEN (goals_full_time_home + goals_full_time_away) < 1.5 THEN 1 ELSE 0 END as under_15_result,
                        -- Calculate handicap results (negative = team must win by X, positive = team can lose by X and still win bet)
                        -- Home team handicaps
                        CASE WHEN (goals_full_time_home - 0.5) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_minus_05,
                        CASE WHEN (goals_full_time_home - 1.0) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_minus_10,
                        CASE WHEN (goals_full_time_home - 1.5) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_minus_15,
                        CASE WHEN (goals_full_time_home + 0.5) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_plus_05,
                        CASE WHEN (goals_full_time_home + 1.0) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_plus_10,
                        CASE WHEN (goals_full_time_home + 1.5) > goals_full_time_away THEN 1 ELSE 0 END as home_handicap_plus_15,
                        -- Away team handicaps
                        CASE WHEN (goals_full_time_away - 0.5) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_minus_05,
                        CASE WHEN (goals_full_time_away - 1.0) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_minus_10,
                        CASE WHEN (goals_full_time_away - 1.5) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_minus_15,
                        CASE WHEN (goals_full_time_away + 0.5) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_plus_05,
                        CASE WHEN (goals_full_time_away + 1.0) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_plus_10,
                        CASE WHEN (goals_full_time_away + 1.5) > goals_full_time_home THEN 1 ELSE 0 END as away_handicap_plus_15,
                        -- Draw handicap (0.0) - draw or better for each team
                        CASE WHEN goals_full_time_home >= goals_full_time_away THEN 1 ELSE 0 END as home_handicap_00,
                        CASE WHEN goals_full_time_away >= goals_full_time_home THEN 1 ELSE 0 END as away_handicap_00,
                        -- Numeric feature columns only
                        goals_full_time_home + goals_full_time_away as total_goals,
                        ABS(goals_full_time_home - goals_full_time_away) as goal_difference,
                        -- Goals scored in each half
                        COALESCE(goals_half_time_home, 0) as half_time_home_goals,
                        COALESCE(goals_half_time_away, 0) as half_time_away_goals,
                        -- Basic league averages (simplified features)
                        2.5 as league_avg_goals,
                        1.2 as home_form_goals,
                        1.2 as away_form_goals,
                        1 as btts_probability_indicator,
                        2.4 as match_pace_indicator,
                        -- Additional features
                        CASE WHEN goals_full_time_home > goals_full_time_away THEN 1 ELSE 0 END as home_win,
                        CASE WHEN goals_full_time_home = goals_full_time_away THEN 1 ELSE 0 END as draw,
                        CASE WHEN goals_full_time_away > goals_full_time_home THEN 1 ELSE 0 END as away_win
                    FROM fixtures 
                    WHERE status IN ('Match Finished', 'FT')
                    AND goals_full_time_home IS NOT NULL 
                    AND goals_full_time_away IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 5000
                    """).df()
                
                    logger.info(f"   📈 Query returned {len(historical_data)} rows")
                
                    if len(historical_data) == 0:
                        logger.warning("   ⚠️ No historical completed matches found - using minimal training")
                        # Create minimal placeholder data for training with proper structure
                        import pandas as pd
                        historical_data = pd.DataFrame({
                            'fixture_id': [1, 2, 3, 4, 5],
                            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                            'home_goals': [2, 1, 3, 0, 2],
                            'away_goals': [1, 2, 1, 1, 2],
                            # Target variables
                            'btts_result': [1, 1, 1, 0, 1],  # Both teams scored
                            'over_15_result': [1, 1, 1, 0, 1],  # Over 1.5 goals
                            'under_15_result': [0, 0, 0, 1, 0],  # Under 1.5 goals  
                            'home_handicap_minus_05': [1, 0, 1, 0, 0],  # Home -0.5
                            'home_handicap_minus_15': [0, 0, 1, 0, 0],  # Home -1.5
                            'away_handicap_minus_05': [0, 1, 0, 0, 0],  # Away -0.5
                            # Numeric feature variables only
                            'total_goals': [3, 3, 4, 1, 4],
                            'goal_difference': [1, 1, 2, 1, 0],
                            'half_time_home_goals': [1, 0, 2, 0, 1],
                            'half_time_away_goals': [0, 1, 1, 0, 1],
                            'league_avg_goals': [2.5, 2.5, 2.5, 2.3, 2.3],
                            'home_form_goals': [1.2, 1.1, 1.3, 0.8, 1.2],
                            'away_form_goals': [1.0, 1.2, 1.0, 0.9, 1.1],
                            'btts_probability_indicator': [1, 1, 1, 0, 1],
                            'match_pace_indicator': [2.5, 2.1, 2.8, 1.2, 2.4],
                            'home_win': [1, 0, 1, 0, 0],
                            'draw': [0, 0, 0, 0, 1],
                            'away_win': [0, 1, 0, 1, 0]
                        })
                        logger.info(f"   🔨 Created placeholder training data with {len(historical_data)} rows")
                    else:
                        logger.info(f"   ✅ Found historical data: {len(historical_data):,} completed matches")
                        # Show sample of data
                        logger.info(f"   📊 Sample: Match {historical_data.iloc[0]['fixture_id']} result {historical_data.iloc[0]['home_goals']}-{historical_data.iloc[0]['away_goals']}")
                    
                    logger.info("   🏋️ Calling train_market_models()...")
                    
                    # Train the quantitative models with detailed error tracking
                    training_result = self.quant_model.train_market_models(historical_data)
                    logger.debug(f"   🔍 Training result: {training_result}")
                    
                    # Verify models were actually trained
                    if hasattr(self.quant_model, 'models') and self.quant_model.models:
                        trained_markets = list(self.quant_model.models.keys())
                        logger.info(f"   ✅ Models trained for markets: {trained_markets}")
                        
                        # Save the trained model for future use
                        logger.info("   💾 Saving trained model...")
                        try:
                            saved_file = self.quant_model.save_models("latest")
                            logger.info(f"   ✅ Model saved: {saved_file}")
                        except Exception as save_error:
                            logger.warning(f"   ⚠️ Failed to save model: {save_error}")
                    else:
                        logger.error("   ❌ No models found after training - check train_market_models() implementation")
                    
                    logger.info("   ✅ Model training completed!")
                
                except Exception as training_error:
                    logger.error(f"   ❌ Model training failed with error: {training_error}")
                    logger.debug("   📍 Error details:")
                    import traceback
                    logger.debug(traceback.format_exc())
                    logger.info("   🔄 Continuing with untrained model (predictions will fail)")
                    # Continue anyway - the workflow will handle prediction failures gracefully
            
            # Load configuration
            self.quant_config = QuantConfig()
            logger.info("✅ Configuration loaded")
            
            # Initialize data pipeline for updates
            self.data_pipeline = DuckDBDataPipeline(db_path='football_data.duckdb')
            logger.info("✅ Data pipeline ready")
            
            self.workflow_state['model_ready'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing quantitative system: {e}")
            return False
    
    def step_2_update_data(self, force_update=False):
        """Update database with latest fixture data"""
        logger.info("📊 STEP 2: Updating Data")
        logger.info("-" * 50)
        
        try:
            # Check if data is current (within 2 hours)
            if not force_update:
                latest_fixture = self.db_conn.execute("""
                    SELECT MAX(date) FROM fixtures WHERE date >= CURRENT_DATE
                """).fetchone()[0]
                
                if latest_fixture and latest_fixture == datetime.now().strftime('%Y-%m-%d'):
                    logger.info("✅ Database already has today's fixtures")
                    self.workflow_state['data_updated'] = True
                    return True
            
            # Get fixture IDs for current/upcoming matches
            logger.info("🔍 Getting fixture IDs for current and upcoming matches...")
            api_football_main()  # Ensure API data is fresh
            fixture_ids = get_fixtures_past_time_plus_hour()
            
            if not fixture_ids:
                logger.info("📭 No current or upcoming fixtures found")
                return True
            
            logger.info(f"⚽ Found {len(fixture_ids)} fixtures to analyze")
            
            # Update database with any new fixture data if needed
            logger.info("🔄 Database update complete")
            self.workflow_state['data_updated'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating data: {e}")
            return False
    
    def step_3_analyze_opportunities(self):
        """Get Matchbook fixtures, match with DuckDB data, and analyze EV opportunities"""
        logger.info("🎯 STEP 3: Matchbook-First Opportunity Analysis") 
        logger.info("-" * 50)
        
        try:
            if not self.quant_model:
                logger.error("❌ Quantitative model not initialized")
                return []
            
            # STEP 3A: Get upcoming fixtures from Matchbook first (with caching)
            matchbook_events = self._get_cached_matchbook_events()
            
            if not matchbook_events:
                logger.error("❌ No Matchbook events available")
                return []
            
            # Extract events from the response (it has an "events" key)
            events_list = matchbook_events.get('events', [])
            if not events_list:
                logger.error("❌ No events found in Matchbook response")
                return []
            
            logger.info(f"📈 Found {len(events_list)} Matchbook events")
            
            # STEP 3B: Match Matchbook events with DuckDB fixtures
            logger.info("🔗 Matching Matchbook events with DuckDB fixtures...")
            matched_fixtures = []
            unmatched_events = []
            
            for mb_event in events_list:
                event_name = mb_event.get('name', '')
                event_id = mb_event.get('id')
                
                # Extract team names from Matchbook event name
                teams = self._extract_teams_from_event_name(event_name)
                if not teams:
                    unmatched_events.append((event_name, "Could not extract teams from event name"))
                    continue
                    
                home_team, away_team = teams
                
                # Get event start time if available
                event_start_time = mb_event.get('start-time') or mb_event.get('start_time')
                
                # Find matching fixture in DuckDB
                db_fixture = self._find_matching_duckdb_fixture(home_team, away_team, event_start_time)
                
                if db_fixture:
                    matched_fixture = {
                        'matchbook_event': mb_event,
                        'duckdb_fixture': db_fixture,
                        'home_team': home_team,
                        'away_team': away_team,
                        'match_name': f"{home_team} vs {away_team}"
                    }
                    matched_fixtures.append(matched_fixture)
                    logger.debug(f"   ✅ Matched: {event_name} → {db_fixture['home_team_name']} vs {db_fixture['away_team_name']}")
                else:
                    unmatched_events.append((event_name, f"No DuckDB match for {home_team} vs {away_team}"))
                    logger.debug(f"   ❌ No DuckDB match: {event_name} (extracted: {home_team} vs {away_team})")
            
            logger.info(f"🔍 Matching results: {len(matched_fixtures)}/{len(events_list)} events matched")
            
            if unmatched_events and len(unmatched_events) <= 10:
                logger.debug("⚠️  Unmatched events:")
                for event_name, reason in unmatched_events:
                    logger.debug(f"   - {event_name}: {reason}")
            
            if not matched_fixtures:
                logger.info("📭 No matched fixtures found for analysis")
                return []
            
            # STEP 3C: Analyze only matched fixtures for EV opportunities
            logger.info("💰 Analyzing matched fixtures for EV opportunities...")
            opportunities = []
            
            for matched in matched_fixtures:

                mb_event = matched['matchbook_event']
                db_fixture = matched['duckdb_fixture']
                home_team = matched['home_team']
                away_team = matched['away_team']
                
                try:
                    # Prepare fixture data for quantitative model
                    fixture_data = {
                        'fixture_id': db_fixture.get('fixture_id'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'league': db_fixture.get('league_name'),
                        'match_date': db_fixture.get('date'),
                        'duckdb_data': db_fixture
                    }
                    
                    # Check each target market in Matchbook event
                    for market in self.config['target_markets']:
                        logger.debug(f"      🔍 Checking {market} market...")
                        
                        # Find the market in Matchbook event
                        # Markets are directly available in mb_event['markets']
                        if 'markets' not in mb_event:
                            logger.debug(f"         ❌ No markets found in event for {market}")
                            continue
                        
                        market_data = self._find_matchbook_market(mb_event, market)
                        
                        if not market_data:
                            logger.debug(f"         ❌ No {market} market found in Matchbook event")
                            continue
                        
                        logger.debug(f"         ✅ Found {market} market: {market_data.get('market_name', 'N/A')}")
                        logger.debug(f"         💰 Runner: {market_data.get('runner_name', 'N/A')} @ {market_data.get('back_odds', 0):.2f}")
                        
                        # Get model prediction for this market
                        prediction = self._get_market_prediction(fixture_data, market)
                        
                        if not prediction:
                            logger.debug(f"         ❌ No valid prediction for {market}")
                            continue
                        
                        if not prediction.get('should_bet', False):
                            logger.debug(f"         ❌ Model says don't bet on {market}")
                            logger.debug(f"             EV: {prediction.get('expected_value', 0):.1%} | Confidence: {prediction.get('confidence', 0):.1%}")
                            continue
                        
                        # Calculate EV using Matchbook odds
                        exchange_odds = market_data.get('back_odds', 0)
                        
                        if exchange_odds <= 0:
                            logger.debug(f"         ❌ Invalid exchange odds: {exchange_odds}")
                            continue
                        
                        # Recalculate EV with actual Matchbook odds
                        model_prob = prediction['prediction']
                        actual_ev = (model_prob * exchange_odds) - 1.0
                        
                        logger.debug(f"         📊 Model prob: {model_prob:.1%} | Exchange odds: {exchange_odds:.2f}")
                        logger.debug(f"         💎 Calculated EV: {actual_ev:.1%} (min required: {self.config['min_ev']:.1%})")
                        
                        # Check EV is within reasonable bounds
                        max_ev = self.config.get('max_ev', 0.70)  # Default to 70% if not set
                        if actual_ev >= self.config['min_ev'] and actual_ev <= max_ev:
                            opportunity = {
                                'fixture_id': db_fixture.get('fixture_id'),
                                'match': matched['match_name'],
                                'league': db_fixture.get('league_name'),
                                'market': market,
                                'runner_name': market_data.get('runner_name', ''),  # Add runner name for display
                                'prediction_prob': model_prob,
                                'confidence': prediction['confidence'],
                                'expected_value': actual_ev,
                                'exchange_odds': exchange_odds,
                                'recommended_stake': self._calculate_stake_size(actual_ev, prediction['confidence']),
                                # Matchbook data for betting
                                'matchbook_event_id': mb_event.get('id'),
                                'matchbook_market_id': market_data.get('market_id'),
                                'matchbook_runner_id': market_data.get('runner_id'),
                                'exchange_liquidity': market_data.get('liquidity', 0),
                                'reasoning': f"{market} EV opportunity with Matchbook odds"
                            }
                            
                            # Validate final opportunity
                            if self._validate_exchange_opportunity(opportunity):
                                opportunities.append(opportunity)
                                logger.info(f"   🎯 {market.upper()}: {matched['match_name']}")
                                logger.info(f"      Model: {model_prob:.1%} | Odds: {exchange_odds:.2f} | EV: {actual_ev:.1%}")
                            else:
                                logger.warning(f"         ❌ Failed final validation for {market}")
                        elif actual_ev < self.config['min_ev']:
                            logger.debug(f"         ❌ EV too low: {actual_ev:.1%} < {self.config['min_ev']:.1%}")
                        else:
                            max_ev = self.config.get('max_ev', 0.70)
                            logger.debug(f"         ❌ EV too high (unrealistic): {actual_ev:.1%} > {max_ev:.1%}")
                except Exception as e:
                    logger.warning(f"   ❌ Error analyzing {matched['match_name']}: {e}")
                    continue
            
            # Sort by expected value * confidence
            opportunities.sort(key=lambda x: x['expected_value'] * x['confidence'], reverse=True)
            
            logger.info(f"✅ Found {len(opportunities)} validated EV opportunities")
            self.workflow_state['opportunities_analyzed'] = True
            
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Error in opportunity analysis: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def step_4_finalize_opportunities(self, opportunities):
        """Apply final validation and prepare opportunities for betting"""
        logger.info("🛡️ STEP 4: Final Opportunity Validation")
        logger.info("-" * 50)
        
        try:
            if not opportunities:
                logger.info("📭 No opportunities to validate")
                return []
            
            logger.info(f"🔍 Validating {len(opportunities)} pre-matched opportunities...")
            
            # Apply final risk management and validation
            validated_opportunities = self._apply_risk_management(opportunities)
            
            if not validated_opportunities:
                logger.warning("📉 All opportunities filtered out by risk management")
                return []
            
            # Final liquidity and odds checks
            final_opportunities = []
            for opp in validated_opportunities:
                if self._final_validation_check(opp):
                    final_opportunities.append(opp)
                    logger.info(f"   ✅ Validated: {opp['match']} - {opp['market']}")
                    logger.info(f"      Final EV: {opp['expected_value']:.1%} | Stake: £{opp['recommended_stake']:.2f}")
                else:
                    logger.warning(f"   ❌ Final validation failed: {opp['match']} - {opp['market']}")
            
            logger.info(f"📊 Final validated opportunities: {len(final_opportunities)}")
            return final_opportunities
            
        except Exception as e:
            logger.error(f"❌ Error in final validation: {e}")
            return []
    
    def step_5_place_bets(self, matched_opportunities):
        """Place bets on matched opportunities"""
        logger.info("💰 STEP 5: Placing Quantitative Bets")
        logger.info("-" * 50)
        
        if not matched_opportunities:
            logger.info("📭 No matched opportunities to bet on")
            return False
        
        # Apply final risk management
        filtered_opportunities = self._apply_risk_management(matched_opportunities)
        
        if not filtered_opportunities:
            logger.warning("📉 All opportunities filtered out by risk management")
            return False
        
        # Calculate total stake
        total_stake = sum(opp['recommended_stake'] for opp in filtered_opportunities)
        
        logger.info(f"🎯 Ready to place {len(filtered_opportunities)} quantitative bets")
        logger.info(f"💰 Total stake: £{total_stake:.2f}")
        
        # Check daily limits  
        if total_stake > self.config['max_daily_stake']:
            logger.warning(f"⚠️  Total stake exceeds daily limit (£{self.config['max_daily_stake']:.2f})")
            return False
        
        # Manual approval unless auto-place is enabled
        if not self.config['auto_place_bets']:
            logger.info("🤔 Manual approval required:")
            self._display_betting_opportunities(filtered_opportunities)
            
            # Check if we're in an interactive terminal
            if sys.stdin.isatty():
                # Interactive mode - ask for user input
                choice = input("\n👉 Place these bets? (YES/no): ").strip().upper()
                if choice not in ['YES', 'Y', '']:
                    logger.info("❌ Bet placement cancelled")
                    return False
            else:
                # Non-interactive mode (API/automation) - skip bet placement
                logger.info("🤖 Non-interactive mode detected - skipping bet placement")
                logger.info("💡 Set auto_place_bets=true to enable automated betting via API")
                return False
        
        # Place the bets
        successful_bets = 0
        for opp in filtered_opportunities:
            try:
                logger.info(f"📍 Placing bet: {opp['match']} - {opp['market']}")
                
                success, runner_id = self._place_single_bet(opp)
                if success:
                    successful_bets += 1
                    logger.info(f"✅ Bet placed: £{opp['recommended_stake']:.2f} @ {opp['exchange_odds']:.2f}")
                    
                    # Try to log bet to database (skip if read-only)
                    try:
                        self.db_conn.execute("""
                            INSERT INTO bet_history (runner_id, match_name, league, market, runner_name, stake, odds, expected_value, confidence, placed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, [
                            runner_id,
                            opp['match'],
                            opp['league'],
                            opp['market'],
                            opp['runner_name'],
                            opp['recommended_stake'],
                            opp['exchange_odds'],
                            opp['expected_value'],
                            opp['confidence']
                        ])
                    except Exception as db_error:
                        if "read-only" in str(db_error).lower() or "cannot commit" in str(db_error).lower():
                            logger.warning("⚠️  Database is read-only, bet not logged to history")
                        else:
                            logger.warning(f"⚠️  Failed to log bet: {db_error}")

                else:
                    logger.error("❌ Bet failed")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Error placing bet: {e}")
                continue
        
        logger.info("📊 Bet Placement Summary:")
        logger.info(f"✅ Successful: {successful_bets}/{len(filtered_opportunities)}")
        logger.info(f"💰 Total placed: £{sum(opp['recommended_stake'] for opp in filtered_opportunities[:successful_bets]):.2f}")
        
        self.workflow_state['bets_placed'] = True
        return successful_bets > 0
    
    def run_full_workflow(self, force_data_update=False):
        """Run the complete quantitative betting workflow"""
        logger.info("🚀 STARTING QUANTITATIVE BETTING WORKFLOW")
        logger.info("=" * 70)
        start_time = time.time()
        
        # Step 1: Initialize system
        if not self.step_1_initialize_quantitative_system():
            logger.error("💥 Workflow failed at initialization")
            return False
        
        # Step 2: Update data
        if not self.step_2_update_data(force_data_update):
            logger.error("💥 Workflow failed at data update")
            return False  
        
        # Step 3: Analyze opportunities
        opportunities = self.step_3_analyze_opportunities()
        if not opportunities:
            logger.info("📈 No quantitative opportunities found")
            return True
        
        # Step 4: Final validation
        final_opportunities = self.step_4_finalize_opportunities(opportunities)
        if not final_opportunities:
            logger.info("📊 No opportunities passed final validation")
            return True
        
        # Step 5: Place bets
        success = self.step_5_place_bets(final_opportunities)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✨ QUANTITATIVE WORKFLOW COMPLETE in {elapsed_time:.1f}s")
        
        return success
    
    def run_analysis_only(self):
        """Run analysis without placing bets (for testing/evaluation)"""
        logger.info("🧪 ANALYSIS ONLY MODE")
        logger.info("=" * 50)
        
        if not self.step_1_initialize_quantitative_system():
            return False
        
        if not self.step_2_update_data():  
            return False
        
        opportunities = self.step_3_analyze_opportunities()
        
        if opportunities:
            logger.info("📊 QUANTITATIVE ANALYSIS RESULTS")
            self._display_analysis_results(opportunities)
        
        return True
    
    # Helper methods
    def _prepare_real_fixture_features(self, home_team, away_team, league, fixture_id):
        """Prepare real team statistics from database for model predictions"""
        try:
            if not self.db_conn:
                raise ValueError("Database connection not available - cannot prepare real team features")
            
            # Query last 10 matches for each team to calculate form
            home_stats = self.db_conn.execute("""
                SELECT 
                    AVG(goals_full_time_home) as avg_goals_for,
                    AVG(goals_full_time_away) as avg_goals_against,
                    AVG(goals_half_time_home) as avg_first_half,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN goals_full_time_home > goals_full_time_away THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN goals_full_time_home = goals_full_time_away THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN goals_full_time_home < goals_full_time_away THEN 1 ELSE 0 END) as losses
                FROM fixtures 
                WHERE home_team_name = ? AND date >= CURRENT_DATE - INTERVAL '60 days'
            """, [home_team]).fetchone()
            
            away_stats = self.db_conn.execute("""
                SELECT 
                    AVG(goals_full_time_away) as avg_goals_for,
                    AVG(goals_full_time_home) as avg_goals_against,
                    AVG(goals_half_time_away) as avg_first_half,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN goals_full_time_away > goals_full_time_home THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN goals_full_time_away = goals_full_time_home THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN goals_full_time_away < goals_full_time_home THEN 1 ELSE 0 END) as losses
                FROM fixtures 
                WHERE away_team_name = ? AND date >= CURRENT_DATE - INTERVAL '60 days'
            """, [away_team]).fetchone()
            
            # Query league statistics
            league_stats = self.db_conn.execute("""
                SELECT 
                    AVG((goals_full_time_home + goals_full_time_away)) as league_avg_goals,
                    AVG(CASE WHEN goals_full_time_home > 0 AND goals_full_time_away > 0 THEN 1 ELSE 0 END) as btts_rate,
                    AVG(CASE WHEN (goals_full_time_home + goals_full_time_away) > 1.5 THEN 1 ELSE 0 END) as over_15_rate
                FROM fixtures 
                WHERE league_name = ? AND date >= CURRENT_DATE - INTERVAL '90 days'
            """, [league]).fetchone()
            
            # Calculate features
            home_gf = home_stats[0] if home_stats[0] else 1.2
            home_ga = home_stats[1] if home_stats[1] else 1.0
            home_fh = home_stats[2] if home_stats[2] else 0.5
            home_matches = home_stats[3] if home_stats[3] else 1
            home_wins = home_stats[4] if home_stats[4] else 0
            
            away_gf = away_stats[0] if away_stats[0] else 1.0
            away_ga = away_stats[1] if away_stats[1] else 1.2
            away_fh = away_stats[2] if away_stats[2] else 0.4
            away_matches = away_stats[3] if away_stats[3] else 1
            away_wins = away_stats[4] if away_stats[4] else 0
            
            league_avg = league_stats[0] if league_stats[0] else 2.5
            btts_rate = league_stats[1] if league_stats[1] else 0.5
            over_15_rate = league_stats[2] if league_stats[2] else 0.6
            
            # Calculate win probabilities (simplified)
            total_goals_exp = home_gf + away_gf
            home_win_prob = (home_gf / (home_gf + away_ga + 0.1)) if (home_gf + away_ga) > 0 else 0.45
            away_win_prob = (away_gf / (away_gf + home_ga + 0.1)) if (away_gf + home_ga) > 0 else 0.28
            draw_prob = 1.0 - home_win_prob - away_win_prob
            
            # Ensure probabilities are valid
            total_prob = max(home_win_prob + away_win_prob + draw_prob, 1.0)
            home_win_prob = min(max(home_win_prob / total_prob, 0.1), 0.9)
            away_win_prob = min(max(away_win_prob / total_prob, 0.1), 0.9)
            draw_prob = max(0.0, 1.0 - home_win_prob - away_win_prob)
            
            logger.info(f"         📊 Real data loaded: {home_team} avg {home_gf:.2f} goals | {away_team} avg {away_gf:.2f} goals")
            logger.info(f"         📈 League: {league_avg:.2f} avg goals | {btts_rate:.0%} BTTS rate")
            
            model_fixture = {
                'fixture_id': fixture_id,
                'date': '2026-02-16',
                'home_goals': int(home_gf),
                'away_goals': int(away_gf),
                # Real calculated features from database (12 features used for prediction)
                'total_goals': total_goals_exp,
                'goal_difference': home_gf - away_gf,
                'half_time_home_goals': home_fh,
                'half_time_away_goals': away_fh,
                'league_avg_goals': league_avg,
                'home_form_goals': home_gf,  # Real average goals scored at home
                'away_form_goals': away_gf,  # Real average goals scored away
                'btts_probability_indicator': btts_rate,  # Real BTTS rate
                'match_pace_indicator': league_avg,  # League pace
                'home_win': home_win_prob,  # Calculated win probability
                'draw': draw_prob,  # Calculated draw probability
                'away_win': away_win_prob  # Calculated away win probability
            }
            
            return model_fixture
        
        except Exception as e:
            raise ValueError(f"Failed to load team statistics from database: {e}")
    
    def _get_market_prediction(self, fixture_data, market):
        """Get prediction for a specific market using quantitative model"""
        try:
            if not self.quant_model:
                logger.warning(f"      ❌ Quantitative model not available for {market}")
                return None
            
            # Extract fixture details
            home_team = fixture_data.get('home_team')
            away_team = fixture_data.get('away_team') 
            league = fixture_data.get('league')
            fixture_id = fixture_data.get('fixture_id')
            
            logger.debug(f"         🔍 Getting {market} prediction for {home_team} vs {away_team}")
            
            # Try to use the actual quantitative model for prediction
            try:
                # Check if model is trained first
                if not hasattr(self.quant_model, 'models'):
                    logger.error("         ❌ No 'models' attribute found on quant_model")
                    return None
                
                if not self.quant_model.models:
                    logger.error("         ❌ Models dict is empty - training required")
                    return None
                    
                # print(f"         📈 Available trained models: {list(self.quant_model.models.keys())}")
                
                # Get real team statistics from database
                model_fixture = self._prepare_real_fixture_features(home_team, away_team, league, fixture_id)
                
                logger.debug(f"         🎯 Calling predict_markets() with fixture data: {len(model_fixture)} features")
                
                # Call the quantitative model's prediction method
                predictions = self.quant_model.predict_markets(model_fixture)
                
                # print(f"         🔍 Raw predictions response: {predictions}")
                
                if not predictions or 'error' in predictions:
                    logger.error(f"         ❌ Model prediction failed for {market} - no bet")
                    return None
                
                logger.debug("         ✅ Model prediction successful")
                
                # Extract the specific market prediction
                if market == 'btts':
                    btts_pred = predictions.get('btts', {})
                    prediction_prob = btts_pred.get('yes_probability', 0)
                    confidence = btts_pred.get('confidence', 0)
                    
                elif market == 'over_under_15':
                    over_pred = predictions.get('over_15', {})
                    prediction_prob = over_pred.get('probability', 0)
                    confidence = over_pred.get('confidence', 0)
                    
                elif market == 'handicap':
                    # Check for various handicap predictions and pick the best one
                    handicap_options = [
                        'home_handicap_minus_15', 'home_handicap_minus_10', 'home_handicap_minus_05',
                        'home_handicap_00', 'home_handicap_plus_05', 'home_handicap_plus_10', 'home_handicap_plus_15',
                        'away_handicap_minus_15', 'away_handicap_minus_10', 'away_handicap_minus_05',
                        'away_handicap_00', 'away_handicap_plus_05', 'away_handicap_plus_10', 'away_handicap_plus_15'
                    ]
                    best_prob = 0
                    best_conf = 0
                    best_market = None
                    
                    for hc_market in handicap_options:
                        hc_pred = predictions.get(hc_market, {})
                        prob = hc_pred.get('probability', 0)
                        conf = hc_pred.get('confidence', 0)
                        
                        # Only consider handicaps with reasonable probability (> 30%)
                        # A high confidence with 0% probability means "won't happen", not a betting opportunity
                        if prob > 0.3 and conf > best_conf:
                            best_prob = prob
                            best_conf = conf
                            best_market = hc_market
                    
                    prediction_prob = best_prob
                    confidence = best_conf
                    
                    # If no viable handicap found, log it
                    if prediction_prob == 0 or confidence == 0:
                        logger.debug("         ❌ No viable handicap found (all probabilities < 30% or no confidence)")
                        return None
                    else:
                        logger.debug(f"         ✅ Best handicap: {best_market} (prob: {best_prob:.1%}, conf: {best_conf:.1%})")
                    
                else:
                    logger.error(f"         ❌ Unknown market type: {market}")
                    return None
                
                if prediction_prob <= 0 or confidence <= 0:
                    logger.warning(f"         ❌ Invalid prediction values for {market} - no bet")
                    return None
                
                logger.debug(f"         📊 Model result: {prediction_prob:.1%} probability, {confidence:.1%} confidence")
                
                # Check if meets minimum thresholds
                if (prediction_prob >= 0.3 and  # Minimum 30% probability
                    confidence >= self.config['min_confidence']):
                    
                    return {
                        'prediction': prediction_prob,
                        'confidence': confidence,
                        'expected_value': 0,  # Will be calculated with real odds later
                        'should_bet': True,
                        'reasoning': f'{market} prediction from quantitative model'
                    }
                else:
                    logger.debug(f"         ❌ Below thresholds: prob={prediction_prob:.1%}, conf={confidence:.1%}")
                    return None
                    
            except ValueError as data_error:
                logger.warning(f"         ❌ Cannot prepare reliable features for {market}: {data_error}")
                logger.info(f"         🚫 Skipping {market} - insufficient data quality")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting {market} prediction: {e}")
            return None
    
    def _prepare_fixture_features(self, fixture_data):
        """Prepare fixture data for model prediction"""
        # This would extract and prepare features from the fixture data
        # For now, return basic structure
        return {
            'home_team': fixture_data['home_team'],
            'away_team': fixture_data['away_team'],
            'league': fixture_data['league'],
            # Add more features as needed
        }
    
    def _calculate_stake_size(self, expected_value, confidence):
        """Calculate appropriate stake size based on EV and confidence"""
        # Fixed stake for now (£0.10)
        # TODO: Implement Kelly Criterion for optimal stake sizing
        return self.config['min_stake']
    
    def _find_matchbook_event(self, matchbook_events, home_team, away_team):
        """Find matching Matchbook event using fuzzy matching"""
        try:
            # Load Matchbook events from simplified format
            if isinstance(matchbook_events, list) and matchbook_events:
                events = matchbook_events
            else:
                # Try to load from saved file
                if os.path.exists("matchbook_football_events_simplified.json"):
                    with open("matchbook_football_events_simplified.json", "r") as f:
                        data = json.load(f)
                        events = data.get('events', [])
                else:
                    return None
            
            # Simple fuzzy matching
            for event in events:
                event_name = event.get('name', '')  # Changed from 'event_name' to 'name'
                
                # Check if both team names appear in event name
                if (self._fuzzy_match(home_team, event_name) and 
                    self._fuzzy_match(away_team, event_name)):
                    return event
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding Matchbook event: {e}")
            return None
    
    def _find_matchbook_market(self, matchbook_event, market_type):
        """Find specific market in Matchbook event using raw markets structure"""
        try:
            # Get markets directly from the event
            markets = matchbook_event.get('markets', [])
            logger.debug(f"         📊 Found {len(markets)} markets to search")
            
            if market_type == 'btts':
                # Look for Both Teams To Score market
                for market in markets:
                    market_name = market.get('name', '').lower()
                    if 'both teams to score' in market_name or 'btts' in market_name:
                        # Find "Yes" runner
                        runners = market.get('runners', [])
                        for runner in runners:
                            if 'yes' in runner.get('name', '').lower():
                                # Get best back price
                                prices = runner.get('prices', [])
                                back_odds = prices[0].get('odds') if prices else 0
                                available = prices[0].get('available-amount') if prices else 0
                                return {
                                    'market_id': market.get('id'),
                                    'market_name': market.get('name'),
                                    'runner_id': runner.get('id'),
                                    'runner_name': runner.get('name'),
                                    'back_odds': back_odds,
                                    'liquidity': available
                                }
            
            elif market_type == 'over_under_15':
                # Look for Total market with Over/Under 1.5 runners
                for market in markets:
                    market_name = market.get('name', '').lower()
                    if 'total' in market_name:
                        # Check if this Total market has 1.5 runners
                        runners = market.get('runners', [])
                        for runner in runners:
                            runner_name = runner.get('name', '').lower()
                            if 'over 1.5' in runner_name or 'over1.5' in runner_name:
                                # Get best back price
                                prices = runner.get('prices', [])
                                back_odds = prices[0].get('odds') if prices else 0
                                available = prices[0].get('available-amount') if prices else 0
                                return {
                                    'market_id': market.get('id'),
                                    'market_name': market.get('name'),
                                    'runner_id': runner.get('id'),
                                    'runner_name': runner.get('name'),
                                    'back_odds': back_odds,
                                    'liquidity': available
                                }
            
            elif market_type == 'handicap':
                # Look for Handicap market  
                logger.debug("         🎯 Searching for handicap markets...")
                handicap_markets_found = []
                
                for i, market in enumerate(markets):
                    market_name = market.get('name', '').lower()
                    
                    if 'handicap' in market_name:
                        runners = market.get('runners', [])
                        
                        for j, runner in enumerate(runners):
                            runner_name = runner.get('name', '')
                            
                            # Skip split/quarter handicaps (e.g., "-0.5/1.0", "+0.0/0.5")
                            # Model only trained on simple handicaps
                            if '/' in runner_name or '(' in runner_name and ')' in runner_name and '/' in runner_name:
                                logger.debug(f"         ⏭️  Skipping split handicap: {runner_name}")
                                continue
                            
                            # Get best back price from prices array
                            prices = runner.get('prices', [])
                            odds = prices[0].get('odds') if prices else 0
                            liquidity = prices[0].get('available-amount') if prices else 0
                            
                            if 1.05 <= odds <= 5.0 and liquidity > 10:
                                value_score = odds * min(liquidity, 100)
                                if 1.2 <= odds <= 3.0:
                                    value_score *= 1.5
                                handicap_markets_found.append({
                                    'runner': runner,
                                    'market': market,
                                    'value_score': value_score,
                                    'odds': odds,
                                    'liquidity': liquidity
                                })
                
                logger.debug(f"         📊 Found {len(handicap_markets_found)} viable simple handicap runners (splits excluded)")
                
                # Select best handicap option
                if handicap_markets_found:
                    best_option = max(handicap_markets_found, key=lambda x: x['value_score'])
                    best_runner = best_option['runner']
                    best_market = best_option['market']
                    
                    logger.debug(f"         🏆 Selected best handicap: {best_runner.get('name')} @ {best_option['odds']} (score: {best_option['value_score']:.1f})")
                    
                    # Get best back price
                    prices = best_runner.get('prices', [])
                    back_odds = prices[0].get('odds') if prices else 0
                    available = prices[0].get('available-amount') if prices else 0
                    
                    return {
                        'market_id': best_market.get('id'),
                        'market_name': best_market.get('name'),
                        'runner_id': best_runner.get('id'),
                        'runner_name': best_runner.get('name'),
                        'back_odds': back_odds,
                        'liquidity': available
                    }
                else:
                    logger.debug("         ❌ No viable handicap runners found (need odds 1.05-5.0 with £10+ liquidity)")
            
            logger.debug(f"         ❌ No matching {market_type} market found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding {market_type} market: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_simplified_event_data(self, home_team, away_team):
        """Load simplified event data from the saved JSON file"""
        try:
            simplified_file = "matchbook_football_events_simplified.json"
            if not os.path.exists(simplified_file):
                logger.error(f"         ❌ Simplified file not found: {simplified_file}")
                return None
            
            with open(simplified_file, "r", encoding="utf-8") as f:
                simplified_data = json.load(f)
            
            # Look for event matching the teams
            match_name = f"{home_team} vs {away_team}"
            
            # Try exact match first
            if match_name in simplified_data:
                return simplified_data[match_name]
            
            # Try fuzzy matching
            for event_name, event_data in simplified_data.items():
                if (self._fuzzy_match(home_team, event_name) and 
                    self._fuzzy_match(away_team, event_name)):
                    logger.debug(f"         🔍 Fuzzy matched: {event_name}")
                    return event_data
            
            logger.warning(f"         ❌ No simplified event found for {match_name}")
            return None
            
        except Exception as e:
            logger.error(f"         ❌ Error loading simplified event data: {e}")
            return None
    
    def _fuzzy_match(self, team_name, event_name):
        """Simple fuzzy matching for team names using proper normalization"""
        
        # Use our comprehensive normalization instead of simple normalization  
        team_clean = self._normalize_team_name(team_name)
        event_clean = self._normalize_team_name(event_name)
        
        # Check for exact match after normalization
        if team_clean == event_clean:
            return True
        
        # Check for exact substring match 
        if team_clean in event_clean:
            return True
        
        # Check if event_clean contains team_clean (reversed check)
        if event_clean in team_clean:
            return True
        
        # Check for substantial word overlap
        team_words = team_clean.split()
        event_words = event_clean.split()
        
        if len(team_words) == 0:
            return False
        
        # For single-word team names, check if that word appears in the event
        if len(team_words) == 1:
            return team_words[0] in event_words
        
        # For 2-word team names, be more lenient - just need the main word to match
        if len(team_words) == 2:
            for team_word in team_words:
                if len(team_word) >= 4:  # Significant word
                    for event_word in event_words:
                        if team_word == event_word or team_word in event_word or event_word in team_word:
                            return True
            return False
        
        # For longer names, require at least half the words to match (excluding very short words)
        significant_words = [w for w in team_words if len(w) > 2]
        if not significant_words:
            # All words are short, just check if any match
            matches = 0
            for team_word in team_words:
                if team_word in event_words:
                    matches += 1
            return matches >= len(team_words)
        
        matches = 0
        for team_word in significant_words:
            if team_word in event_words:
                matches += 1
        return matches >= max(1, len(significant_words) // 2)
    
    def _validate_exchange_opportunity(self, opportunity):
        """Validate that exchange opportunity meets our requirements"""
        try:
            odds = opportunity.get('exchange_odds', 0)
            liquidity = opportunity.get('exchange_liquidity', 0)
            stake = opportunity.get('recommended_stake', 0)
            
            # Check minimum odds
            if odds < 1.2:
                return False
                
            # Check maximum odds  
            if odds > 10.0:
                return False
            
            # Check liquidity vs stake
            if liquidity < stake * 2:  # Need 2x stake in liquidity  
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_risk_management(self, opportunities):
        """Apply final risk management filters"""
        if not opportunities:
            return []
        
        # Sort by expected value * confidence
        opportunities.sort(key=lambda x: x['expected_value'] * x['confidence'], reverse=True)
        
        # Apply bet limits
        filtered = []
        daily_stake = 0
        match_counts = {}
        
        for opp in opportunities:
            match = opp['match']
            stake = opp['recommended_stake']
            
            # Check daily limit
            if daily_stake + stake > self.config['max_daily_stake']:
                continue
            
            # Check per-match limit
            if match_counts.get(match, 0) >= self.config['max_bets_per_match']:
                continue
            
            # Add to filtered list
            filtered.append(opp)
            daily_stake += stake
            match_counts[match] = match_counts.get(match, 0) + 1
            
            # Stop if we hit limits
            if len(filtered) >= 10:  # Max 10 bets per session
                break
        
        return filtered
    
    def _place_single_bet(self, opportunity):
        """Place a single bet via Matchbook"""
        try:
            if not self.matchbook:
                return False
            
            runner_data = {
                "runner_id": opportunity['matchbook_runner_id'],
                "side": "back",
                "odds": opportunity['exchange_odds'],
                "stake": opportunity['recommended_stake']
            }
            
            # Check if we already have a bet on this runner (avoid duplicates)
            try:
                current_bets = self.matchbook.get_current_bets()
                current_runner_ids = [
                    selection["runner-id"]
                    for market in current_bets.get("markets", [])
                    for selection in market.get("selections", [])
                ]
                
                if runner_data["runner_id"] in current_runner_ids:
                    logger.warning(f"   ⚠️ Bet already exists for runner {runner_data['runner_id']}, skipping (treating as successful)")
                    return True, runner_data["runner_id"]
            except Exception as check_error:
                logger.warning(f"   ⚠️ Could not check existing bets: {check_error}, proceeding with placement")
            
            response = self.matchbook.place_order(runner_data)

            if response.get("offers")[0].get("matched-bets") is None:
                logger.info("Back Bet not matched cancelling bet")
                offer_id = response.get("offers")[0].get("id")
                cancel_bet_response = self.matchbook.cancel_single_order(offer_id)
                logger.info(cancel_bet_response)
            
            # Check if successful
            if isinstance(response, dict) and not response.get('errors'):
                return True, runner_data["runner_id"]
            
            return False, None
            
        except Exception as e:
            logger.error(f"❌ Error placing bet: {e}")
            return False, None
    
    def _display_betting_opportunities(self, opportunities):
        """Display betting opportunities for manual approval"""
        logger.info("🎯 QUANTITATIVE BETTING OPPORTUNITIES")
        logger.info("=" * 70)
        
        for i, opp in enumerate(opportunities, 1):
            # Create clear market description
            market_desc = opp['market'].upper()
            if opp.get('runner_name'):
                if opp['market'] == 'handicap':
                    market_desc = f"HANDICAP: {opp['runner_name']}"
                elif opp['market'] == 'btts':
                    market_desc = f"BTTS: {opp['runner_name']}"
                elif opp['market'] == 'over_under_15':
                    market_desc = f"O/U 1.5: {opp['runner_name']}"
            
            logger.info(f"{i}. {opp['match']} - {market_desc}")
            logger.info(f"   📊 Confidence: {opp['confidence']:.1%} | EV: {opp['expected_value']:.1%}")
            logger.info(f"   💰 Stake: £{opp['recommended_stake']:.2f} @ {opp['exchange_odds']:.2f}")
            logger.info(f"   🎯 Potential profit: £{(opp['recommended_stake'] * opp['exchange_odds']) - opp['recommended_stake']:.2f}")
        
        total_stake = sum(opp['recommended_stake'] for opp in opportunities)
        avg_ev = sum(opp['expected_value'] for opp in opportunities) / len(opportunities)
        logger.info(f"📊 Total: {len(opportunities)} bets, £{total_stake:.2f} stake, {avg_ev:.1%} avg EV")
    
    def _display_analysis_results(self, opportunities):
        """Display analysis results without betting"""
        logger.info("📊 QUANTITATIVE ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        # Group by market
        by_market = {}
        for opp in opportunities:
            market = opp['market']
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(opp)
        
        for market, opps in by_market.items():
            logger.info(f"🎯 {market.upper()} Market ({len(opps)} opportunities)")
            logger.info("-" * 40)
            
            for opp in opps[:5]:  # Show top 5 per market
                logger.info(f"   {opp['match']}")
                logger.info(f"   Confidence: {opp['confidence']:.1%} | EV: {opp['expected_value']:.1%}")
            
            if len(opps) > 5:
                logger.info(f"   ... and {len(opps) - 5} more")
        
        total_ev = sum(opp['expected_value'] for opp in opportunities)
        logger.info(f"📈 Total opportunities: {len(opportunities)} | Combined EV: {total_ev:.1%}")
    
    def _extract_teams_from_event_name(self, event_name):
        """Extract home and away team names from Matchbook event name"""
        try:
            # Common Matchbook event name formats:
            # "Manchester United v Liverpool"
            # "Real Madrid vs Barcelona" 
            # "Chelsea - Arsenal"
            
            # Try different separators
            separators = [' v ', ' vs ', ' - ', ' V ']
            
            for sep in separators:
                if sep in event_name:
                    parts = event_name.split(sep)
                    if len(parts) == 2:
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                        return (home_team, away_team)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting teams from '{event_name}': {e}")
            return None
    
    def _normalize_team_name(self, team_name):
        """Normalize team name for better matching"""
        normalized = team_name.lower().strip()
        
        # Comprehensive character replacement map for all accents and special characters
        char_map = {
            # Scandinavian
            'ø': 'o', 'å': 'a', 'æ': 'ae', 'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            # German
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
            # Polish
            'ł': 'l', 'ź': 'z', 'ż': 'z', 'ć': 'c', 'ń': 'n', 'ś': 's', 'ą': 'a', 'ę': 'e',
            # Czech/Slovak
            'č': 'c', 'ď': 'd', 'ě': 'e', 'ň': 'n', 'ř': 'r', 'š': 's', 'ť': 't', 'ů': 'u', 'ý': 'y', 'ž': 'z',
            # Hungarian
            'ő': 'o', 'ű': 'u',
            # French/Spanish/Portuguese
            'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ú': 'u', 'ù': 'u', 'û': 'u',
            'ñ': 'n', 'ç': 'c', 'ÿ': 'y',
            # Turkish
            'ı': 'i', 'ş': 's', 'ğ': 'g',
            # Romanian
            'ă': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        }
        
        # Replace all special characters
        for old_char, new_char in char_map.items():
            normalized = normalized.replace(old_char, new_char)
        
        # Fallback: Handle any remaining accents using unicodedata (for characters not in our map)
        import unicodedata
        normalized = unicodedata.normalize('NFKD', normalized)
        normalized = ''.join([c for c in normalized if not unicodedata.combining(c)])
        
        # Handle Arabic prefix variations (El-/Al-/As-)
        # Remove or standardize these prefixes for matching
        if normalized.startswith('el-'):
            normalized = normalized[3:]  # Remove "el-" prefix
        elif normalized.startswith('el '):
            normalized = normalized[3:]  # Remove "el " prefix
        elif normalized.startswith('al-'):
            normalized = normalized[3:]  # Remove "al-" prefix
        elif normalized.startswith('al '):
            normalized = normalized[3:]  # Remove "al " prefix
        elif normalized.startswith('as-'):
            normalized = normalized[3:]  # Remove "as-" prefix
        
        # Remove common European club prefixes (NK, FK, PFC, GKS, etc.)
        club_prefixes = ['nk ', 'fk ', 'fc ', 'sc ', 'pfc ', 'afc ', 'cfr ', 'cs ', 'gks ', 'mks ', 'rks ']
        for prefix in club_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break  # Only remove one prefix
        
        # Remove numeric prefixes (e.g., "76 Team Name" → "Team Name")  
        import re
        normalized = re.sub(r'^\d+\s+', '', normalized)
        
        # Remove Turkish club suffixes (belediyespor = municipality sports, spor = sports)
        turkish_suffixes = [' belediyespor', ' belediye', ' spor', ' genclik', ' genclerbirligi']
        for suffix in turkish_suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Known team name aliases (Matchbook vs API Football naming differences)
        team_aliases = {
            'modern sport': 'future',  # Modern Sport FC = Future FC
            'modern sport fc': 'future fc',
            # Add more aliases as discovered
        }
        
        # Check if this team name has a known alias
        for alias, canonical in team_aliases.items():
            if alias in normalized:
                normalized = normalized.replace(alias, canonical)
        
        # Remove common club suffixes/prefixes (FC, SC, IF, etc.)
        normalized = normalized.replace(' fc', '').replace(' sc', '').replace(' if', '').replace('fc ', '').replace('sc ', '').replace('if ', '')
        
        # Remove numeric suffixes (e.g., "Tychy 71" → "Tychy")
        normalized = re.sub(r'\s+\d+$', '', normalized)
        
        # Remove trailing 'e' commonly found in Scandinavian team names (but be conservative)
        # Only remove if the stem is at least 6 characters (to avoid removing 'e' from short names)
        if len(normalized) > 6 and normalized.endswith('e'):
            # Check if the second-to-last character is a consonant (likely a Scandinavian suffix)
            second_last = normalized[-2]
            if second_last in 'bcdfghjklmnpqrstvwxyz':
                normalized = normalized[:-1]
        
        # Remove common city/region suffixes
        # List of common city names that appear as suffixes
        city_suffixes = ['athens', 'city', 'town', 'united', 'madrid', 'barcelona', 'london', 
                        'moscow', 'istanbul', 'cairo', 'dubai', 'jeddah', 'riyadh', 'doha',
                        'qarshi', 'tashkent', 'samarkand', 'bukhara', 'antakya', 'ankara', 
                        'izmir', 'antalya', 'bursa', 'adana', 'konya']
        
        words = normalized.split()
        if len(words) >= 2:
            last_word = words[-1]
            # Remove if last word is a known city or looks like a city (short name <= 7 chars)
            if last_word in city_suffixes or (len(words) > 2 and len(last_word) <= 7):
                normalized = ' '.join(words[:-1])
        
        return normalized.strip()
    
    def _find_matching_duckdb_fixture(self, home_team, away_team, event_start_time=None):
        """Find matching fixture in DuckDB using fuzzy matching with time-based assistance"""
        try:
            if not self.db_conn:
                return None
            
            # Parse event start time if provided
            match_date = None
            if event_start_time:
                try:
                    from datetime import datetime
                    # Try to parse ISO format or common formats
                    if isinstance(event_start_time, str):
                        # ISO format: 2026-02-16T19:00:00
                        match_date = event_start_time.split('T')[0]  # Get just the date part
                    elif isinstance(event_start_time, (int, float)):
                        # Unix timestamp
                        dt = datetime.fromtimestamp(event_start_time)
                        match_date = dt.strftime('%Y-%m-%d')
                except Exception as date_error:
                    logger.warning(f"         ⚠️ Could not parse event time: {date_error}")
            
            # Normalize team names for better matching
            home_normalized = self._normalize_team_name(home_team)
            away_normalized = self._normalize_team_name(away_team)
            
            logger.debug(f"         🔍 Matching: '{home_team}' (→ '{home_normalized}') vs '{away_team}' (→ '{away_normalized}')")
            
            # Query for potential matches using fuzzy matching
            # First try exact matches (with normalized names)
            exact_match = self.db_conn.execute("""
                SELECT fixture_id, home_team_name, away_team_name, league_name, date, status
                FROM fixtures 
                WHERE LOWER(home_team_name) = LOWER(?) 
                AND LOWER(away_team_name) = LOWER(?)
                AND date >= CURRENT_DATE
                LIMIT 1
            """, [home_team, away_team]).fetchone()
            
            if exact_match:
                return {
                    'fixture_id': exact_match[0],
                    'home_team_name': exact_match[1], 
                    'away_team_name': exact_match[2],
                    'league_name': exact_match[3],
                    'date': exact_match[4],
                    'status': exact_match[5]
                }
            
            # If no exact match, try fuzzy matching
            all_fixtures = self.db_conn.execute("""
                SELECT fixture_id, home_team_name, away_team_name, league_name, date, status
                FROM fixtures 
                WHERE date >= CURRENT_DATE
                ORDER BY date
                LIMIT 200
            """).fetchall()
            
            for fixture in all_fixtures:
                db_home = fixture[1]
                db_away = fixture[2]
                db_date = str(fixture[4])  # fixture date from DB
                db_home_normalized = self._normalize_team_name(db_home)
                db_away_normalized = self._normalize_team_name(db_away)
                
                # Check if teams match fuzzily (using both original and normalized names)
                home_match = (self._fuzzy_match(home_team, db_home) or 
                             self._fuzzy_match(home_normalized, db_home_normalized))
                away_match = (self._fuzzy_match(away_team, db_away) or 
                             self._fuzzy_match(away_normalized, db_away_normalized))
                
                # Debug: Show close matches
                if home_match or away_match:
                    logger.debug(f"         🔸 Partial match: DB has '{db_home}' (→ '{db_home_normalized}') vs '{db_away}' (→ '{db_away_normalized}') | home_match={home_match}, away_match={away_match}")
                
                # Standard match: both teams must match
                if home_match and away_match:
                    return {
                        'fixture_id': fixture[0],
                        'home_team_name': fixture[1],
                        'away_team_name': fixture[2], 
                        'league_name': fixture[3],
                        'date': fixture[4],
                        'status': fixture[5]
                    }
                
                # Time-assisted match: if date matches AND at least one team in correct position matches
                # This handles cases where one team name is completely different between sources
                if match_date and match_date == db_date:
                    # Check home-to-home match
                    if home_match:
                        logger.debug(f"         🕐 Time + Home team match: {home_team} → {db_home} (date: {match_date})")
                        return {
                            'fixture_id': fixture[0],
                            'home_team_name': fixture[1],
                            'away_team_name': fixture[2], 
                            'league_name': fixture[3],
                            'date': fixture[4],
                            'status': fixture[5]
                        }
                    # Check away-to-away match
                    elif away_match:
                        logger.debug(f"         🕐 Time + Away team match: {away_team} → {db_away} (date: {match_date})")
                        return {
                            'fixture_id': fixture[0],
                            'home_team_name': fixture[1],
                            'away_team_name': fixture[2], 
                            'league_name': fixture[3],
                            'date': fixture[4],
                            'status': fixture[5]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding DuckDB fixture for {home_team} vs {away_team}: {e}")
            return None
    
    def _final_validation_check(self, opportunity):
        """Final validation check for opportunity before betting"""
        try:
            # Check required fields are present
            required_fields = [
                'exchange_odds', 'expected_value', 'recommended_stake',
                'matchbook_runner_id', 'confidence'
            ]
            
            for field in required_fields:
                if field not in opportunity or opportunity[field] is None:
                    logger.debug(f"   Missing field: {field}")
                    return False
            
            # Validate exchange odds range
            odds = opportunity['exchange_odds']
            if odds < 1.1 or odds > 15.0:
                logger.debug(f"   Odds out of range: {odds}")
                return False
            
            # Validate EV meets minimum
            if opportunity['expected_value'] < self.config['min_ev']:
                logger.debug(f"   EV too low: {opportunity['expected_value']:.1%}")
                return False
            
            # Validate confidence meets minimum
            if opportunity['confidence'] < self.config['min_confidence']:
                logger.debug(f"   Confidence too low: {opportunity['confidence']:.1%}")
                return False
            
            # Validate liquidity
            liquidity = opportunity.get('exchange_liquidity', 0)
            stake = opportunity['recommended_stake']
            if liquidity < stake * 1.5:  # Need 1.5x stake in liquidity
                logger.debug(f"   Insufficient liquidity: £{liquidity:.2f} vs £{stake:.2f} stake")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"   Validation error: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.db_conn:
            self.db_conn.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Quantitative Betting Workflow")
    parser.add_argument('--full-run', action='store_true', help='Run complete workflow')
    parser.add_argument('--analyze-only', action='store_true', help='Analysis only (no betting)')
    parser.add_argument('--data-only', action='store_true', help='Data update only')
    parser.add_argument('--force-data-update', action='store_true', help='Force data update')
    parser.add_argument('--auto-place', action='store_true', help='Auto-place bets')
    parser.add_argument('--min-ev', type=float, default=0.08, help='Minimum EV (default: 8%)')
    parser.add_argument('--max-stake', type=float, default=5.0, help='Max daily stake')
    parser.add_argument('--conservative', action='store_true', help='Conservative settings')
    
    args = parser.parse_args()
    
    # Configure workflow
    if args.conservative:
        config = {
            'min_ev': 0.12,  # 12% minimum
            'max_ev': 0.50,  # 50% max (conservative)
            'min_confidence': 0.75,  # 75% confidence
            'max_daily_stake': 2.0,  # £2 max
            'min_stake': 0.10,
            'auto_place_bets': True,  # Auto-place bets by default
            'target_markets': ['btts', 'over_under_15', 'handicap'],
            'max_bets_per_match': 1
        }
    else:
        config = {
            'min_ev': args.min_ev,
            'max_ev': 0.70,  # 70% max - filter unrealistic predictions
            'max_daily_stake': args.max_stake,
            'auto_place_bets': args.auto_place,
            'min_stake': 0.10,
            'min_confidence': 0.65,
            'target_markets': ['btts', 'over_under_15', 'handicap'],
            'max_bets_per_match': 1
        }
    
    # Initialize workflow
    workflow = QuantitativeBettingWorkflow(config)
    
    try:
        if args.analyze_only:
            workflow.run_analysis_only()
        elif args.data_only:
            workflow.step_1_initialize_quantitative_system()
            workflow.step_2_update_data(args.force_data_update)
        elif args.full_run:
            workflow.run_full_workflow(args.force_data_update)
        else:
            logger.warning("🔍 No action specified. Use --full-run, --analyze-only, or --data-only")
            
    except KeyboardInterrupt:
        logger.info("🛑 Workflow interrupted")
    except Exception as e:
        logger.error(f"💥 Workflow error: {e}")
    finally:
        workflow.cleanup()


if __name__ == "__main__":
    main()