#!/usr/bin/env python3
"""
Quant Model Configuration System
==============================

Configuration management for the Advanced Quantitative Betting Model
Based on empirical findings and evidence-based market selection
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class QuantConfig:
    """
    Configuration manager for quantitative betting models
    """
    
    def __init__(self, config_file: str = "quant_config.json"):
        self.config_file = config_file
        self._config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load evidence-based default configuration
        """
        return {
            "model_settings": {
                "target_markets": {
                    "btts": {
                        "enabled": True,
                        "historical_wr": 66.7,
                        "historical_units": 0.30,
                        "status": "PROFITABLE",
                        "min_edge": 0.03,  # 3% minimum edge for BTTS
                        "max_stake_pct": 5.0,
                        "confidence_threshold": 0.65
                    },
                    "over_under_15": {
                        "enabled": True,
                        "historical_wr": None,
                        "historical_units": None,
                        "status": "REPLACEMENT_FOR_2_5",
                        "min_edge": 0.02,  # Lower threshold for testing 1.5 goals
                        "max_stake_pct": 3.0,
                        "confidence_threshold": 0.70
                    },
                    "asian_handicap": {
                        "enabled": True,
                        "historical_wr": None,
                        "historical_units": None,
                        "status": "EXPLORATION",
                        "min_edge": 0.035,  # Higher threshold for new market
                        "max_stake_pct": 2.0,
                        "confidence_threshold": 0.75
                    },
                    "over_under_25": {
                        "enabled": False,
                        "historical_wr": 43.3,
                        "historical_units": -1.15,
                        "status": "BANNED",
                        "ban_reason": "Poor performance - 43.3% WR, -1.15 units"
                    },
                    "double_chance": {
                        "enabled": False,
                        "historical_wr": 58.5,
                        "historical_units": -1.18,
                        "status": "UNPROFITABLE",
                        "ban_reason": "Breakeven at best - poor risk/reward"
                    }
                },
                
                "model_parameters": {
                    "training_window_days": 365,
                    "min_training_samples": 500,
                    "cv_splits": 5,
                    "random_state": 42,
                    "calibration_method": "isotonic"
                },
                
                "feature_engineering": {
                    "form_window": 5,  # Last 5 games for form
                    "min_games_for_stats": 10,
                    "include_league_context": True,
                    "include_calendar_effects": True,
                    "interaction_features": True
                }
            },
            
            "risk_management": {
                "bankroll_management": {
                    "kelly_safety_factor": 0.25,  # 25% of Kelly criterion
                    "max_stake_per_bet": 5.0,  # % of bankroll
                    "max_exposure_per_day": 15.0,  # % of bankroll
                    "max_exposure_per_market": 10.0  # % of bankroll
                },
                
                "bet_filtering": {
                    "min_probability": 0.55,  # Minimum confidence
                    "max_probability": 0.90,  # Maximum confidence (avoid overconfidence)
                    "min_odds": 1.20,  # Minimum odds value
                    "max_odds": 5.00,  # Maximum odds value
                    "min_stake_amount": 0.10  # Minimum bet amount
                },
                
                "stop_loss": {
                    "daily_loss_limit": -50.0,  # Stop betting if daily loss exceeds
                    "weekly_loss_limit": -200.0,
                    "drawdown_limit": -1000.0,  # Total drawdown limit
                    "consecutive_loss_limit": 10  # Stop after X consecutive losses
                }
            },
            
            "data_sources": {
                "primary_db": "football_data.duckdb",
                "backup_json_dir": "historical_data",
                "model_storage_dir": "models",
                "results_storage_dir": "results"
            },
            
            "monitoring": {
                "model_performance": {
                    "retrain_threshold": 0.05,  # Retrain if accuracy drops 5%
                    "calibration_check_frequency": 50,  # Check every 50 predictions
                    "max_staleness_days": 7  # Retrain if model >7 days old
                },
                
                "logging": {
                    "log_level": "INFO",
                    "log_file": "quant_model.log",
                    "enable_performance_logging": True,
                    "enable_prediction_logging": True
                }
            }
        }
    
    def _load_config(self):
        """
        Load configuration from file if exists
        """
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults (file config takes precedence)
                self._merge_config(self._config, file_config)
                print(f"✅ Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}, using defaults")
        else:
            print("📝 No config file found, using defaults")
    
    def _merge_config(self, base: Dict, override: Dict):
        """
        Recursively merge configuration dictionaries
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def save_config(self):
        """
        Save current configuration to file
        """
        try:
            # Add metadata
            self._config['_metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '1.0.0',
                'description': 'Advanced Quantitative Betting Model Configuration'
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation (e.g., 'model_settings.target_markets.btts')
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """
        Set configuration value using dot notation
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the correct nested location
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_market_config(self, market: str) -> Dict:
        """
        Get configuration for a specific market
        """
        return self.get(f'model_settings.target_markets.{market}', {})
    
    def is_market_enabled(self, market: str) -> bool:
        """
        Check if a market is enabled for betting
        """
        market_config = self.get_market_config(market)
        return market_config.get('enabled', False)
    
    def get_enabled_markets(self) -> List[str]:
        """
        Get list of enabled markets
        """
        markets = self.get('model_settings.target_markets', {})
        return [market for market, config in markets.items() if config.get('enabled', False)]
    
    def get_banned_markets(self) -> List[str]:
        """
        Get list of banned markets with reasons
        """
        markets = self.get('model_settings.target_markets', {})
        banned = {}
        
        for market, config in markets.items():
            if not config.get('enabled', True):
                ban_reason = config.get('ban_reason', 'Disabled in config')
                banned[market] = ban_reason
        
        return banned
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues
        """
        issues = []
        
        # Check enabled markets
        enabled_markets = self.get_enabled_markets()
        if not enabled_markets:
            issues.append("No markets enabled for betting")
        
        # Check risk management settings
        max_stake = self.get('risk_management.bankroll_management.max_stake_per_bet', 0)
        if max_stake > 10:
            issues.append(f"Max stake per bet too high: {max_stake}%")
        
        # Check minimum probability threshold
        min_prob = self.get('risk_management.bet_filtering.min_probability', 0)
        if min_prob < 0.5:
            issues.append(f"Minimum probability too low: {min_prob}")
        
        # Check data sources
        db_path = self.get('data_sources.primary_db')
        if not db_path:
            issues.append("Primary database path not configured")
        
        return issues
    
    def print_summary(self):
        """
        Print configuration summary
        """
        print("🎯 QUANT MODEL CONFIGURATION SUMMARY")
        print("=" * 50)
        
        # Enabled markets
        enabled = self.get_enabled_markets()
        print(f"📈 ENABLED MARKETS ({len(enabled)}):")
        for market in enabled:
            config = self.get_market_config(market)
            status = config.get('status', 'UNKNOWN')
            min_edge = config.get('min_edge', 0) * 100
            print(f"   {market:<20} | {status:<15} | Min Edge: {min_edge:.1f}%")
        
        # Banned markets
        banned = self.get_banned_markets()
        if banned:
            print(f"\n🚫 BANNED MARKETS ({len(banned)}):")
            for market, reason in banned.items():
                print(f"   {market:<20} | {reason}")
        
        # Risk management
        print("\n⚖️ RISK MANAGEMENT:")
        max_stake = self.get('risk_management.bankroll_management.max_stake_per_bet')
        daily_limit = self.get('risk_management.bankroll_management.max_exposure_per_day')
        print(f"   Max Stake: {max_stake}% | Daily Exposure: {daily_limit}%")
        
        # Validation
        issues = self.validate_config()
        if issues:
            print(f"\n⚠️  CONFIGURATION ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ Configuration validated successfully")


def main():
    """
    Configuration setup and management
    """
    print("⚙️  QUANT MODEL CONFIGURATION")
    print("=" * 40)
    
    # Initialize configuration
    config = QuantConfig()
    
    # Print summary
    config.print_summary()
    
    # Save default configuration
    config.save_config()
    
    print(f"\n📁 Configuration saved as: {config.config_file}")
    print("💡 Edit the JSON file to customize settings")


if __name__ == "__main__":
    main()