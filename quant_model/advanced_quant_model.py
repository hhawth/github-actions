#!/usr/bin/env python3
"""
Advanced Quantitative Betting Model - DuckDB Edition
====================================================

Next-generation model targeting profitable markets based on empirical evidence:
- Asian Handicap markets (new focus)
- Both Teams To Score (proven profitable: 66.7% WR)
- Over/Under 1.5 Goals (replacement for failing 2.5 line)

Data Source: DuckDB for high-performance analytics
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import pickle
from datetime import datetime
from typing import Dict, List
import logging
import warnings
warnings.filterwarnings('ignore')

class AdvancedQuantModel:
    """
    Advanced quantitative model for profitable market prediction
    Based on empirical findings from previous betting analysis
    """
    
    def __init__(self, db_path: str = "football_data.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # EVIDENCE-BASED MARKET FOCUS
        self.target_markets = {
            'btts': {
                'name': 'Both Teams To Score',
                'historical_wr': 66.7,
                'historical_units': 0.30,
                'status': 'PROFITABLE'
            },
            'handicap': {
                'name': 'Asian Handicap',
                'historical_wr': None,  # New market to explore
                'historical_units': None,
                'status': 'TESTING'
            },
            'over_under_15': {
                'name': 'Over/Under 1.5 Goals',
                'historical_wr': None,  # Replacement for failing 2.5 line
                'historical_units': None,
                'status': 'REPLACEMENT'
            }
        }
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.feature_importance = {}
        self.feature_columns = []  # Store expected feature names and order
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Advanced Quant Model Initialized")
        self.logger.info(f"📊 Target Markets: {list(self.target_markets.keys())}")
    
    def setup_database_schema(self):
        """
        Setup optimized DuckDB schema for quant analysis
        """
        print("📊 Setting up DuckDB schema...")
        
        # Create optimized tables for each market type
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY,
                date DATE,
                home_team VARCHAR,
                away_team VARCHAR,
                league_id INTEGER,
                season VARCHAR,
                home_goals INTEGER,
                away_goals INTEGER,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS market_odds (
                id INTEGER PRIMARY KEY,
                fixture_id INTEGER,
                market_type VARCHAR,
                selection VARCHAR,
                odds DECIMAL(10,3),
                bookmaker VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY,
                team_name VARCHAR,
                league_id INTEGER,
                season VARCHAR,
                
                -- Attack metrics
                goals_scored_home DECIMAL(5,2),
                goals_scored_away DECIMAL(5,2),
                shots_per_game DECIMAL(5,2),
                shots_on_target_pct DECIMAL(5,2),
                
                -- Defense metrics  
                goals_conceded_home DECIMAL(5,2),
                goals_conceded_away DECIMAL(5,2),
                clean_sheets_pct DECIMAL(5,2),
                
                -- BTTS specific metrics
                btts_frequency_home DECIMAL(5,2),
                btts_frequency_away DECIMAL(5,2),
                avg_match_goals DECIMAL(5,2),
                
                -- Handicap specific metrics
                handicap_performance DECIMAL(5,2),
                margin_of_victory DECIMAL(5,2),
                close_game_frequency DECIMAL(5,2),
                
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY,
                fixture_id INTEGER,
                market_type VARCHAR,
                selection VARCHAR,
                raw_probability DECIMAL(5,4),
                calibrated_probability DECIMAL(5,4),
                expected_value DECIMAL(8,4),
                confidence_score DECIMAL(5,4),
                model_version VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS bet_results (
                id INTEGER PRIMARY KEY,
                fixture_id INTEGER,
                market_type VARCHAR,
                selection VARCHAR,
                stake DECIMAL(8,2),
                odds DECIMAL(10,3),
                result VARCHAR, -- 'won', 'lost', 'void'
                profit_loss DECIMAL(8,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(id)
            )
            """
        ]
        
        for query in schema_queries:
            self.conn.execute(query)
        
        # Create performance indexes
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)",
            "CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(home_team, away_team)",
            "CREATE INDEX IF NOT EXISTS idx_market_odds_fixture ON market_odds(fixture_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON model_predictions(fixture_id)",
            "CREATE INDEX IF NOT EXISTS idx_bet_results_market ON bet_results(market_type)",
        ]
        
        for query in index_queries:
            self.conn.execute(query)
        
        print("✅ Database schema setup complete")
    
    def load_historical_data(self) -> pd.DataFrame:
        """
        Load and prepare historical data from DuckDB with advanced feature engineering
        """
        print("📈 Loading historical data from DuckDB...")
        
        query = """
        SELECT 
            f.id as fixture_id,
            f.date,
            f.home_team,
            f.away_team,
            f.league_id,
            f.home_goals,
            f.away_goals,
            
            -- Basic match features
            (f.home_goals + f.away_goals) as total_goals,
            CASE WHEN f.home_goals > 0 AND f.away_goals > 0 THEN 1 ELSE 0 END as btts_result,
            CASE WHEN f.home_goals + f.away_goals > 1.5 THEN 1 ELSE 0 END as over_15_result,
            CASE WHEN f.home_goals + f.away_goals < 1.5 THEN 1 ELSE 0 END as under_15_result,
            
            -- Handicap results (various lines)
            CASE WHEN f.home_goals - f.away_goals > 0.5 THEN 1 ELSE 0 END as home_handicap_minus_05,
            CASE WHEN f.home_goals - f.away_goals > 1.5 THEN 1 ELSE 0 END as home_handicap_minus_15,
            CASE WHEN f.away_goals - f.home_goals > 0.5 THEN 1 ELSE 0 END as away_handicap_minus_05,
            
            -- Team stats (if available)
            hs.goals_scored_home as home_goals_scored_avg,
            hs.goals_conceded_home as home_goals_conceded_avg,
            hs.btts_frequency_home as home_btts_freq,
            hs.handicap_performance as home_handicap_perf,
            
            as_.goals_scored_away as away_goals_scored_avg,
            as_.goals_conceded_away as away_goals_conceded_avg,
            as_.btts_frequency_away as away_btts_freq,
            as_.handicap_performance as away_handicap_perf
            
        FROM fixtures f
        LEFT JOIN team_stats hs ON f.home_team = hs.team_name AND f.league_id = hs.league_id
        LEFT JOIN team_stats as_ ON f.away_team = as_.team_name AND f.league_id = as_.league_id
        WHERE f.status = 'FT' 
        AND f.home_goals IS NOT NULL 
        AND f.away_goals IS NOT NULL
        AND f.date >= '2025-01-01'  -- Focus on recent data
        ORDER BY f.date DESC
        """
        
        df = self.conn.execute(query).fetchdf()
        
        if df.empty:
            self.logger.warning("⚠️ No historical data found in DuckDB")
            return df
        
        print(f"✅ Loaded {len(df)} completed matches")
        
        # Advanced feature engineering
        df = self._engineer_advanced_features(df)
        
        return df
    
    def _engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features based on empirical findings
        """
        print("🔧 Engineering advanced features...")
        
        # League-specific features
        league_stats = df.groupby('league_id').agg({
            'total_goals': ['mean', 'std'],
            'btts_result': 'mean',
            'over_15_result': 'mean'
        }).round(3)
        
        league_stats.columns = ['league_avg_goals', 'league_std_goals', 'league_btts_freq', 'league_over15_freq']
        league_stats = league_stats.reset_index()
        
        df = df.merge(league_stats, on='league_id', how='left')
        
        # Team form features (last 5 games)
        df = df.sort_values(['home_team', 'date'])
        df['home_form_goals_scored'] = df.groupby('home_team')['home_goals'].rolling(5, min_periods=2).mean().reset_index(0, drop=True)
        df['home_form_goals_conceded'] = df.groupby('home_team')['away_goals'].rolling(5, min_periods=2).mean().reset_index(0, drop=True)
        
        df = df.sort_values(['away_team', 'date'])
        df['away_form_goals_scored'] = df.groupby('away_team')['away_goals'].rolling(5, min_periods=2).mean().reset_index(0, drop=True)
        df['away_form_goals_conceded'] = df.groupby('away_team')['home_goals'].rolling(5, min_periods=2).mean().reset_index(0, drop=True)
        
        # Interaction features for BTTS prediction (based on 66.7% WR success)
        df['btts_probability_indicator'] = (
            (df['home_form_goals_scored'].fillna(1.2) > 1.0) & 
            (df['away_form_goals_scored'].fillna(1.2) > 1.0)
        ).astype(int)
        
        # Handicap-specific features
        df['goal_difference_expectation'] = (
            df['home_form_goals_scored'].fillna(1.2) - df['home_form_goals_conceded'].fillna(1.2) +
            df['away_form_goals_conceded'].fillna(1.2) - df['away_form_goals_scored'].fillna(1.2)
        )
        
        # Over/Under 1.5 features (replacement for failing 2.5 line)
        df['match_pace_indicator'] = (
            df['home_form_goals_scored'].fillna(1.2) + df['away_form_goals_scored'].fillna(1.2) +
            df['home_form_goals_conceded'].fillna(1.2) + df['away_form_goals_conceded'].fillna(1.2)
        ) / 4
        
        # Day of week and month effects
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Fill remaining NaN values
        df = df.fillna(df.median())
        
        print(f"✅ Feature engineering complete: {df.shape[1]} features")
        
        return df
    
    def train_market_models(self, df: pd.DataFrame):
        """
        Train specialized models for each target market
        """
        if df.empty:
            self.logger.error("❌ No data available for training")
            return
        
        # Define feature columns (exclude target and metadata columns)
        target_and_meta_cols = [
            'fixture_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
            'btts_result', 'over_15_result', 'under_15_result',
            # All 16 handicap target columns
            'home_handicap_minus_15', 'home_handicap_minus_10', 'home_handicap_minus_05',
            'home_handicap_00', 'home_handicap_plus_05', 'home_handicap_plus_10', 'home_handicap_plus_15',
            'away_handicap_minus_15', 'away_handicap_minus_10', 'away_handicap_minus_05',
            'away_handicap_00', 'away_handicap_plus_05', 'away_handicap_plus_10', 'away_handicap_plus_15'
        ]
        
        feature_cols = [col for col in df.columns if col not in target_and_meta_cols]
        
        # 🔑 STORE FEATURE COLUMN NAMES AND ORDER for prediction consistency
        self.feature_columns = feature_cols.copy()
        print(f"💾 Stored {len(self.feature_columns)} feature columns: {self.feature_columns[:5]}...")
        
        X = df[feature_cols]
        
        print(f"🎯 Training models with {len(feature_cols)} features on {len(df)} samples")
        
        # Train BTTS model (proven profitable market)
        self._train_btts_model(X, df['btts_result'])
        
        # Train Over/Under 1.5 model (replacement for failing 2.5 line)
        self._train_over_under_15_model(X, df[['over_15_result', 'under_15_result']])
        
        # Train Handicap models (new market exploration) - all 16 handicap lines
        handicap_columns = [
            'home_handicap_minus_15', 'home_handicap_minus_10', 'home_handicap_minus_05',
            'home_handicap_00', 'home_handicap_plus_05', 'home_handicap_plus_10', 'home_handicap_plus_15',
            'away_handicap_minus_15', 'away_handicap_minus_10', 'away_handicap_minus_05',
            'away_handicap_00', 'away_handicap_plus_05', 'away_handicap_plus_10', 'away_handicap_plus_15'
        ]
        self._train_handicap_models(X, df[handicap_columns])
    
    def _train_btts_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train BTTS model - focus on profitable market (66.7% WR historically)
        """
        print("🎯 Training BTTS model (profitable market)...")
        
        # Use ensemble approach for stability
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }
        
        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            avg_score = scores.mean()
            
            print(f"   {name.upper()} CV Score: {avg_score:.3f} (+/- {scores.std() * 2:.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model on full data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model with scaled data
        best_model.fit(X_scaled, y)
        
        # Fit isotonic calibrator
        calibrator = IsotonicRegression(out_of_bounds='clip')
        probas = best_model.predict_proba(X_scaled)[:, 1]
        calibrator.fit(probas, y)
        
        self.models['btts'] = best_model
        self.scalers['btts'] = scaler
        self.calibrators['btts'] = calibrator
        
        # Store feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance['btts'] = importance_df
            
            print("🔝 Top BTTS Features:")
            for _, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print(f"✅ BTTS model trained successfully (CV Score: {best_score:.3f})")
    
    def _train_over_under_15_model(self, X: pd.DataFrame, y_df: pd.DataFrame):
        """
        Train Over/Under 1.5 model - replacement for failing 2.5 line
        """
        print("🎯 Training Over/Under 1.5 model (2.5 line replacement)...")
        
        # Train separate models for Over and Under
        for target in ['over_15_result', 'under_15_result']:
            y = y_df[target]
            
            # Gradient boosting works well for goal prediction
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=10,
                subsample=0.8,
                random_state=42
            )
            
            # Cross validation (using unscaled data for CV)
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            print(f"   {target} CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            # Setup scaler and train on scaled data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model with scaled data
            model.fit(X_scaled, y)
            
            # Setup calibration
            calibrator = IsotonicRegression(out_of_bounds='clip')
            probas = model.predict_proba(X_scaled)[:, 1]
            calibrator.fit(probas, y)
            
            # Store model components
            model_key = target.replace('_result', '')
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.calibrators[model_key] = calibrator
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_key] = importance_df
        
        print("✅ Over/Under 1.5 models trained successfully")
    
    def _train_handicap_models(self, X: pd.DataFrame, y_df: pd.DataFrame):
        """
        Train Asian Handicap models - new market exploration
        """
        print("🎯 Training Handicap models (new market exploration)...")
        
        handicap_targets = {
            'home_handicap_minus_15': 'Home -1.5',
            'home_handicap_minus_10': 'Home -1.0', 
            'home_handicap_minus_05': 'Home -0.5',
            'home_handicap_00': 'Home 0.0',
            'home_handicap_plus_05': 'Home +0.5',
            'home_handicap_plus_10': 'Home +1.0',
            'home_handicap_plus_15': 'Home +1.5',
            'away_handicap_minus_15': 'Away -1.5',
            'away_handicap_minus_10': 'Away -1.0',
            'away_handicap_minus_05': 'Away -0.5',
            'away_handicap_00': 'Away 0.0',
            'away_handicap_plus_05': 'Away +0.5',
            'away_handicap_plus_10': 'Away +1.0',
            'away_handicap_plus_15': 'Away +1.5'
        }
        
        for target, description in handicap_targets.items():
            y = y_df[target]
            
            # Random Forest for handicap prediction
            model = RandomForestClassifier(
                n_estimators=250,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42
            )
            
            # Cross validation (using unscaled data for CV)
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            print(f"   {description} CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            # Setup scaler and train on scaled data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model with scaled data
            model.fit(X_scaled, y)
            
            # Setup calibration
            calibrator = IsotonicRegression(out_of_bounds='clip')
            probas = model.predict_proba(X_scaled)[:, 1]
            calibrator.fit(probas, y)
            
            # Store model components
            self.models[target] = model
            self.scalers[target] = scaler  
            self.calibrators[target] = calibrator
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[target] = importance_df
        
        print("✅ Handicap models trained successfully")
    
    def predict_markets(self, fixture_data: Dict) -> Dict:
        """
        Generate predictions for all target markets
        """
        if not self.models:
            raise ValueError("❌ No models trained. Run train_market_models() first.")
        
        print(f"         🔍 Input fixture_data keys: {list(fixture_data.keys())}")
        print(f"         📊 Sample values: total_goals={fixture_data.get('total_goals')}, home_win={fixture_data.get('home_win')}")
        
        # Convert input to DataFrame format
        feature_df = self._prepare_prediction_features(fixture_data)
        
        print(f"         📊 Feature DataFrame created: shape={feature_df.shape}, empty={feature_df.empty}")
        if not feature_df.empty:
            print(f"         📊 Sample feature values: {dict(list(feature_df.iloc[0].items())[:3])}")
        
        predictions = {}
        
        # BTTS predictions
        if 'btts' in self.models:
            btts_prob = self._predict_single_market('btts', feature_df)
            predictions['btts'] = {
                'yes_probability': btts_prob,
                'no_probability': 1 - btts_prob,
                'confidence': self._calculate_confidence(btts_prob)
            }
        
        # Over/Under 1.5 predictions
        for market in ['over_15', 'under_15']:
            if market in self.models:
                prob = self._predict_single_market(market, feature_df)
                market_name = market.replace('_', ' ').title()
                predictions[market] = {
                    'probability': prob,
                    'confidence': self._calculate_confidence(prob)
                }
        
        # Handicap predictions - all 16 handicap lines
        handicap_markets = [
            'home_handicap_minus_15', 'home_handicap_minus_10', 'home_handicap_minus_05',
            'home_handicap_00', 'home_handicap_plus_05', 'home_handicap_plus_10', 'home_handicap_plus_15',
            'away_handicap_minus_15', 'away_handicap_minus_10', 'away_handicap_minus_05',
            'away_handicap_00', 'away_handicap_plus_05', 'away_handicap_plus_10', 'away_handicap_plus_15'
        ]
        for market in handicap_markets:
            if market in self.models:
                prob = self._predict_single_market(market, feature_df)
                predictions[market] = {
                    'probability': prob,
                    'confidence': self._calculate_confidence(prob)
                }
        
        return predictions
    
    def _predict_single_market(self, market: str, feature_df: pd.DataFrame) -> float:
        """
        Generate calibrated probability for a single market
        Fails explicitly if prediction cannot be made reliably
        """
        # Check if DataFrame is valid - FAIL if not
        if feature_df.empty:
            raise ValueError(f"Empty DataFrame for {market} - cannot make reliable prediction")
        
        model = self.models[market]
        scaler = self.scalers.get(market)
        calibrator = self.calibrators[market]
        
        print(f"         🔍 {market} DataFrame shape: {feature_df.shape}, columns: {list(feature_df.columns)[:3]}...")
        
        # 🔍 Debug feature alignment - FAIL if mismatched
        if hasattr(self, 'feature_columns') and len(self.feature_columns) > 0:
            expected_features = self.feature_columns
            provided_features = list(feature_df.columns)
            if expected_features != provided_features:
                error_msg = f"Feature mismatch for {market}: expected {len(expected_features)} features, got {len(provided_features)}"
                raise ValueError(error_msg)
            
            # Check for extreme feature values that might cause overconfidence
            extreme_features = []
            for col in feature_df.columns:
                value = feature_df[col].iloc[0]
                if abs(value) > 10 or value < -5:  # Flag potentially extreme values
                    extreme_features.append(f"{col}={value:.2f}")
            
            if extreme_features:
                print(f"         ⚠️  Extreme features detected for {market}: {', '.join(extreme_features[:3])}")
            
            # Scale features if scaler available
            if scaler:
                features = scaler.transform(feature_df)
            else:
                features = feature_df.values
        else:
            raise ValueError(f"No stored feature columns available for {market} - model not properly trained")
        
        # Ensure features array is valid - FAIL if not
        if features.size == 0:
            raise ValueError(f"Empty features array for {market} after transformation")
        
        # Get raw probability
        raw_prob = model.predict_proba(features)[0, 1]
        
        # 🛡️ PRE-CALIBRATION SMOOTHING: Reduce extreme raw probabilities
        # Sports betting models should rarely predict < 5% or > 90% 
        smoothed_prob = raw_prob
        if raw_prob < 0.02:  # Very low probabilities
            smoothed_prob = 0.02 + (raw_prob * 1.5)  # Pull up from 0%
        elif raw_prob > 0.98:  # Very high probabilities  
            smoothed_prob = 0.85 + ((raw_prob - 0.85) * 0.3)  # Pull down from 100%
        elif raw_prob > 0.90:  # High probabilities
            smoothed_prob = 0.80 + ((raw_prob - 0.80) * 0.5)  # Moderate reduction
            
        if smoothed_prob != raw_prob:
            print(f"         🔧 Pre-calibration smoothing for {market}: {raw_prob:.1%} → {smoothed_prob:.1%}")
        
        # Debug: Check if probability is extreme
        is_extreme = raw_prob > 0.95 or raw_prob < 0.05
        if is_extreme:
            print(f"         🚨 Raw model probability is extreme: {raw_prob:.1%} for {market}")
            
            # 🔍 DETAILED FEATURE ANALYSIS for debugging extreme predictions
            print(f"         🧪 Analyzing {market} extreme prediction ({raw_prob:.1%}):")
            
            # Show key features that might cause extremes
            key_features = ['total_goals', 'goal_difference', 'home_form_goals', 'away_form_goals', 
                          'league_avg_goals', 'btts_probability_indicator', 'home_win', 'away_win']
            
            for col in feature_df.columns:
                if col in key_features:
                    value = feature_df[col].iloc[0]
                    print(f"             {col}: {value:.3f}")
                    
            # Check scaled feature values
            if scaler:
                for i, col in enumerate(feature_df.columns):
                    if col in key_features:
                        scaled_val = features[0, i]
                        if abs(scaled_val) > 2:
                            print(f"             {col} scaled: {scaled_val:.2f}σ (extreme!)")
        
        # Apply calibration to smoothed probability
        calibrated_prob = calibrator.predict([smoothed_prob])[0]
        
        # 🏈 MARKET-SPECIFIC PROBABILITY CONSTRAINTS for realistic betting
        if 'handicap' in market:
            # Handicap markets should be more conservative (closer to 50/50)
            if calibrated_prob > 0.85:
                calibrated_prob = 0.75 + ((calibrated_prob - 0.75) * 0.4)  # Pull down high handicap confidence
            elif calibrated_prob < 0.15:
                calibrated_prob = 0.25 - ((0.25 - calibrated_prob) * 0.4)  # Pull up low handicap confidence
                
        elif market == 'btts':
            # BTTS typically ranges 30-70% in real markets
            if calibrated_prob > 0.80:
                calibrated_prob = 0.70 + ((calibrated_prob - 0.70) * 0.3)
            elif calibrated_prob < 0.20:
                calibrated_prob = 0.30 - ((0.30 - calibrated_prob) * 0.3)
                
        elif 'over_' in market or 'under_' in market:
            # Goal markets should be moderate confidence
            if calibrated_prob > 0.85:
                calibrated_prob = 0.75 + ((calibrated_prob - 0.75) * 0.4)
        
        # Debug: Check calibration impact
        if abs(calibrated_prob - smoothed_prob) > 0.08:
            print(f"         🔧 Calibration changed {market}: {smoothed_prob:.1%} → {calibrated_prob:.1%}")
        
        # 🚨 FINAL SAFETY: Enforce absolute bounds for sports betting
        # No sports bet should ever be > 90% or < 10% confidence
        # 🚨 FINAL SAFETY: Enforce absolute bounds for sports betting
        # No sports bet should ever be > 90% or < 10% confidence
        original_prob = calibrated_prob
        calibrated_prob = max(0.10, min(0.90, calibrated_prob))  # Stricter cap: 10-90%
        
        if original_prob != calibrated_prob:
            print(f"         🛡️  Probability capped for {market}: {original_prob:.1%} → {calibrated_prob:.1%}")
        
        return float(calibrated_prob)

    def _prepare_prediction_features(self, fixture_data: Dict) -> pd.DataFrame:
        """
        Prepare features for prediction with bounds checking to prevent extreme values
        Must match EXACT feature names and order from training
        """
        print(f"         🔍 Preparing features from fixture_data with {len(fixture_data)} keys")
        
        # Ensure all values are numeric and valid
        raw_features = {}
        for key, default in [
            ('total_goals', 3), ('goal_difference', 1), ('half_time_home_goals', 1),
            ('half_time_away_goals', 0), ('league_avg_goals', 2.5), ('home_form_goals', 1.2),
            ('away_form_goals', 1.0), ('btts_probability_indicator', 1), ('match_pace_indicator', 2.5),
            ('home_win', 1), ('draw', 0), ('away_win', 0)
        ]:
            raw_value = fixture_data.get(key, default)
            # Ensure numeric value
            try:
                raw_features[key] = float(raw_value) if raw_value is not None else default
            except (ValueError, TypeError):
                print(f"         ⚠️  Invalid value for {key}: {raw_value}, using default {default}")
                raw_features[key] = default
        
        # 🛡️ FEATURE NORMALIZATION to prevent extreme model predictions
        normalized_features = {}
        
        # Ensure win probabilities are realistic and sum to 1
        home_win = max(0.05, min(0.85, raw_features['home_win']))
        away_win = max(0.05, min(0.85, raw_features['away_win']))  
        draw = max(0.05, min(0.85, raw_features['draw']))
        
        # Normalize probabilities to sum to 1
        total_prob = home_win + draw + away_win
        normalized_features['home_win'] = home_win / total_prob
        normalized_features['draw'] = draw / total_prob
        normalized_features['away_win'] = away_win / total_prob
        
        # Apply realistic bounds to key features
        normalized_features['goal_difference'] = max(-2.5, min(2.5, raw_features['goal_difference']))
        normalized_features['home_form_goals'] = max(0.4, min(3.5, raw_features['home_form_goals']))
        normalized_features['away_form_goals'] = max(0.4, min(3.5, raw_features['away_form_goals']))
        normalized_features['btts_probability_indicator'] = max(0.15, min(0.85, raw_features['btts_probability_indicator']))
        normalized_features['league_avg_goals'] = max(1.8, min(3.5, raw_features['league_avg_goals']))
        
        # Apply bounds to other features  
        normalized_features['total_goals'] = max(0.5, min(5.5, raw_features['total_goals']))
        normalized_features['half_time_home_goals'] = max(0, min(3, raw_features['half_time_home_goals']))
        normalized_features['half_time_away_goals'] = max(0, min(3, raw_features['half_time_away_goals']))
        normalized_features['match_pace_indicator'] = max(1.8, min(3.5, raw_features['match_pace_indicator']))
        
        print(f"         ✅ Feature normalization complete: {len(normalized_features)} features prepared")
        
        # Log significant normalizations
        major_changes = []
        for key in ['home_win', 'away_win', 'home_form_goals', 'away_form_goals', 'goal_difference']:
            if key in raw_features:
                raw_val = raw_features[key]
                norm_val = normalized_features[key]
                if abs(norm_val - raw_val) > 0.05:
                    major_changes.append(f"{key}: {raw_val:.2f}→{norm_val:.2f}")
        
        if major_changes:
            print(f"         🔧 Feature normalization: {', '.join(major_changes[:2])}")
        
        # 🔑 CRITICAL: Use exact feature column order from training
        if hasattr(self, 'feature_columns') and len(self.feature_columns) > 0:
            print(f"         🔑 Using stored feature columns: {len(self.feature_columns)} features")
            # Create DataFrame with exact column order from training
            ordered_features = {}
            missing_features = []
            
            for col in self.feature_columns:
                if col in normalized_features:
                    ordered_features[col] = normalized_features[col]
                else:
                    missing_features.append(col)
            
            # FAIL if critical features are missing
            if missing_features:
                raise ValueError(f"Missing critical features for prediction: {missing_features}")
            
            return pd.DataFrame([ordered_features], columns=self.feature_columns)
        else:
            # FAIL if no feature schema available
            raise ValueError("No stored feature columns available - model not properly trained or loaded")
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate confidence score based on how far probability is from 0.5
        Now with more realistic confidence scaling for sports betting
        """
        # Distance from 50% (neutral)
        distance_from_neutral = abs(probability - 0.5)
        
        # Scale to 0-1 but cap maximum confidence at 85% for sports realism
        raw_confidence = distance_from_neutral * 2
        
        # Apply realistic confidence ceiling for sports betting
        realistic_confidence = min(0.85, raw_confidence)  # Max 85% confidence
        
        return realistic_confidence
    
    def save_models(self, version: str = None):
        """
        Save trained models with versioning
        """
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'calibrators': self.calibrators,
            'feature_importance': self.feature_importance,
            'feature_columns': getattr(self, 'feature_columns', []),  # Save feature order
            'target_markets': self.target_markets,
            'version': version,
            'created': datetime.now().isoformat()
        }
        
        filename = f"advanced_quant_model_{version}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Models saved as {filename}")
        return filename
    
    def load_models(self, filename: str):
        """
        Load pre-trained models
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']  
        self.calibrators = model_data['calibrators']
        self.feature_importance = model_data['feature_importance']
        
        # 🔑 Load feature columns for prediction consistency
        self.feature_columns = model_data.get('feature_columns', [])
        
        # Fallback for older models without stored feature columns
        if not self.feature_columns and self.models:
            raise ValueError("Model loaded without feature columns - retrain model to store feature schema")
        
        print(f"✅ Models loaded from {filename}")
        print(f"📊 Version: {model_data['version']}")
        print(f"🎯 Markets: {list(self.models.keys())}")
        if self.feature_columns:
            print(f"🔑 Features: {len(self.feature_columns)} columns stored")
    
    def evaluate_model_performance(self, test_df: pd.DataFrame) -> Dict:
        """
        Comprehensive model evaluation
        """
        if test_df.empty or not self.models:
            return {}
        
        results = {}
        
        # Feature preparation
        feature_cols = [col for col in test_df.columns if col not in [
            'fixture_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
            'btts_result', 'over_15_result', 'under_15_result',
            'home_handicap_minus_05', 'home_handicap_minus_15', 'away_handicap_minus_05'
        ]]
        
        X_test = test_df[feature_cols]
        
        # Evaluate each market
        market_targets = {
            'btts': 'btts_result',
            'over_15': 'over_15_result', 
            'under_15': 'under_15_result',
            'home_handicap_minus_05': 'home_handicap_minus_05',
            'home_handicap_minus_15': 'home_handicap_minus_15',
            'away_handicap_minus_05': 'away_handicap_minus_05'
        }
        
        for market, target_col in market_targets.items():
            if market in self.models and target_col in test_df.columns:
                y_true = test_df[target_col]
                y_prob = []
                
                for idx, row in X_test.iterrows():
                    prob = self._predict_single_market(market, pd.DataFrame([row]))
                    y_prob.append(prob)
                
                y_prob = np.array(y_prob)
                y_pred = (y_prob > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                brier = brier_score_loss(y_true, y_prob)
                log_loss_score = log_loss(y_true, y_prob)
                
                results[market] = {
                    'accuracy': accuracy,
                    'brier_score': brier,
                    'log_loss': log_loss_score,
                    'samples': len(y_true)
                }
                
                print(f"📊 {market.upper()} Performance:")
                print(f"   Accuracy: {accuracy:.3f}")
                print(f"   Brier Score: {brier:.3f}")
                print(f"   Log Loss: {log_loss_score:.3f}")
        
        return results
    
    def generate_betting_recommendations(self, fixture_data: Dict, odds_data: Dict) -> List[Dict]:
        """
        Generate betting recommendations with expected value calculations
        """
        predictions = self.predict_markets(fixture_data)
        recommendations = []
        
        for market, pred_data in predictions.items():
            if market in odds_data:
                market_odds = odds_data[market]
                
                if market == 'btts':
                    # BTTS Yes recommendation
                    yes_prob = pred_data['yes_probability']
                    yes_odds = market_odds.get('yes', 0)
                    
                    if yes_odds > 0:
                        ev = (yes_prob * yes_odds) - 1
                        if ev > 0.03:  # 3% minimum edge for profitable BTTS
                            recommendations.append({
                                'market': 'BTTS Yes',
                                'probability': yes_prob,
                                'odds': yes_odds,
                                'expected_value': ev,
                                'confidence': pred_data['confidence'],
                                'stake_recommendation': self._calculate_stake(ev, yes_prob),
                                'reasoning': "Profitable BTTS market (historical 66.7% WR)"
                            })
                
                else:
                    # Single outcome markets
                    prob = pred_data['probability']
                    odds_value = market_odds.get('odds', 0)
                    
                    if odds_value > 0:
                        ev = (prob * odds_value) - 1
                        
                        # Different thresholds for different markets
                        min_edge = 0.02 if '1.5' in market else 0.035  # Lower threshold for 1.5 goals
                        
                        if ev > min_edge:
                            recommendations.append({
                                'market': market,
                                'probability': prob,
                                'odds': odds_value,
                                'expected_value': ev,
                                'confidence': pred_data['confidence'],
                                'stake_recommendation': self._calculate_stake(ev, prob),
                                'reasoning': self._get_market_reasoning(market)
                            })
        
        # Sort by expected value
        recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return recommendations
    
    def _calculate_stake(self, expected_value: float, probability: float, bankroll: float = 100) -> float:
        """
        Calculate stake using Kelly Criterion with safety factor
        """
        if expected_value <= 0:
            return 0
        
        # Kelly fraction = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = (1 / probability) - 1  # Fair odds - 1
        kelly_fraction = expected_value / b
        
        # Apply safety factor (25% of Kelly)
        safe_fraction = kelly_fraction * 0.25
        
        # Cap at maximum 5% of bankroll
        return min(safe_fraction * bankroll, bankroll * 0.05)
    
    def _get_market_reasoning(self, market: str) -> str:
        """
        Get reasoning for market recommendation
        """
        reasoning_map = {
            'btts': "Historically profitable market (66.7% WR, +0.30 units)",
            'over_15': "Replacement for failing Over 2.5 line (43.3% WR)",
            'under_15': "Replacement for failing Under 2.5 line (43.3% WR)", 
            'home_handicap_minus_05': "Asian handicap exploration - new market",
            'home_handicap_minus_15': "Asian handicap exploration - new market",
            'away_handicap_minus_05': "Asian handicap exploration - new market"
        }
        
        return reasoning_map.get(market, "Quantitative edge identified")
    
    def close_connection(self):
        """
        Close DuckDB connection
        """
        if self.conn:
            self.conn.close()
            print("📊 DuckDB connection closed")


def main():
    """
    Example usage of the Advanced Quant Model
    """
    print("🚀 ADVANCED QUANTITATIVE BETTING MODEL")
    print("=" * 60)
    print("Target Markets: Handicap, BTTS, Over/Under 1.5")
    print("Data Source: DuckDB")
    print("=" * 60)
    
    # Initialize model
    model = AdvancedQuantModel()
    
    try:
        # Setup database schema
        model.setup_database_schema()
        
        # Load historical data
        historical_data = model.load_historical_data()
        
        if not historical_data.empty:
            # Train models
            model.train_market_models(historical_data)
            
            # Save models
            model_file = model.save_models()
            
            # Example prediction
            fixture_example = {
                'league_avg_goals': 2.6,
                'home_form_goals': 1.4,
                'away_form_goals': 1.1,
                'is_weekend': 1
            }
            
            predictions = model.predict_markets(fixture_example)
            print("\n🎯 Example Predictions:")
            for market, pred in predictions.items():
                print(f"   {market}: {pred}")
            
            # Example betting recommendations
            odds_example = {
                'btts': {'yes': 2.10, 'no': 1.75},
                'over_15': {'odds': 1.45},
                'home_handicap_minus_05': {'odds': 1.85}
            }
            
            recommendations = model.generate_betting_recommendations(fixture_example, odds_example)
            
            print(f"\n💰 Betting Recommendations ({len(recommendations)} found):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['market']}: {rec['expected_value']:.1%} EV @ {rec['odds']:.2f}")
        
        else:
            print("⚠️ No historical data available for training")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        model.close_connection()


if __name__ == "__main__":
    main()