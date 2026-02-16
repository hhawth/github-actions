#!/usr/bin/env python3
"""
Advanced Quantitative Model Setup Script
=======================================

Complete setup and initialization for the evidence-based quantitative betting model

Based on empirical findings:
✅ BTTS: 66.7% WR, +0.30 units (PROFITABLE)  
✅ Over/Under 1.5: Replacement for failing 2.5 line
✅ Asian Handicap: New market exploration
❌ Over/Under 2.5: 43.3% WR, -1.15 units (BANNED)
"""

import subprocess
import sys
from pathlib import Path

class QuantModelSetup:
    """
    Setup manager for the Advanced Quantitative Betting Model
    """
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.required_files = [
            'advanced_quant_model.py',
            'duckdb_data_pipeline.py', 
            'quant_config.py',
            'requirements_quant.txt'
        ]
        
        print("🚀 ADVANCED QUANTITATIVE MODEL SETUP")
        print("=" * 60)
        print("Evidence-Based Markets: BTTS, Over/Under 1.5, Asian Handicap")
        print("Data Source: DuckDB High-Performance Analytics")
        print("=" * 60)
    
    def check_python_version(self) -> bool:
        """
        Check if Python version is compatible
        """
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
            return True
        else:
            print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
            return False
    
    def check_required_files(self) -> bool:
        """
        Check if all required files are present
        """
        print("📁 Checking required files...")
        
        all_present = True
        for file in self.required_files:
            file_path = self.project_dir / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (missing)")
                all_present = False
        
        return all_present
    
    def install_dependencies(self) -> bool:
        """
        Install required Python packages
        """
        print("📦 Installing dependencies...")
        
        requirements_file = self.project_dir / 'requirements_quant.txt'
        if not requirements_file.exists():
            print("   ❌ requirements_quant.txt not found")
            return False
        
        try:
            # Install packages
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ])
            print("   ✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error installing dependencies: {e}")
            return False
    
    def setup_configuration(self) -> bool:
        """
        Initialize configuration system
        """
        print("⚙️  Setting up configuration...")
        
        try:
            # Import and run configuration setup
            from quant_config import QuantConfig
            
            config = QuantConfig()
            config.save_config()
            
            print("   ✅ Configuration initialized")
            return True
        except Exception as e:
            print(f"   ❌ Error setting up configuration: {e}")
            return False
    
    def setup_database_schema(self) -> bool:
        """
        Initialize DuckDB database schema
        """
        print("📊 Setting up DuckDB schema...")
        
        try:
            from advanced_quant_model import AdvancedQuantModel
            
            # Initialize model and setup schema
            model = AdvancedQuantModel()
            model.setup_database_schema()
            model.close_connection()
            
            print("   ✅ Database schema created")
            return True
        except Exception as e:
            print(f"   ❌ Error setting up database: {e}")
            return False
    
    def migrate_existing_data(self) -> bool:
        """
        Migrate existing JSON data to DuckDB
        """
        print("🔄 Checking for existing data to migrate...")
        
        # Look for JSON data files
        json_files = list(self.project_dir.glob("api_football_*.json"))
        
        if not json_files:
            print("   ℹ️  No JSON data files found to migrate")
            return True
        
        print(f"   📁 Found {len(json_files)} data files")
        
        try:
            from duckdb_data_pipeline import DuckDBDataPipeline
            
            pipeline = DuckDBDataPipeline()
            pipeline.migrate_json_data(str(self.project_dir))
            
            # Get summary
            summary = pipeline.get_data_summary()
            pipeline.close_connection()
            
            fixtures_total = summary.get('fixtures', {}).get('total', 0)
            print(f"   ✅ Migrated {fixtures_total} fixtures to DuckDB")
            
            return True
        except Exception as e:
            print(f"   ❌ Error migrating data: {e}")
            return False
    
    def run_initial_training(self) -> bool:
        """
        Train initial models if data is available
        """
        print("🎯 Training initial quantitative models...")
        
        try:
            from advanced_quant_model import AdvancedQuantModel
            
            model = AdvancedQuantModel()
            
            # Load historical data
            historical_data = model.load_historical_data()
            
            if historical_data.empty:
                print("   ℹ️  No training data available - skipping model training")
                print("   💡 Load historical data first using the data pipeline")
                return True
            
            print(f"   📊 Training on {len(historical_data)} historical matches")
            
            # Train models
            model.train_market_models(historical_data)
            
            # Save models
            model_file = model.save_models()
            model.close_connection()
            
            print(f"   ✅ Models trained and saved as {model_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error training models: {e}")
            return False
    
    def create_usage_examples(self):
        """
        Create example usage scripts
        """
        print("📝 Creating usage examples...")
        
        # Example 1: Basic model usage
        example1_content = '''#!/usr/bin/env python3
"""
Example 1: Basic Quantitative Model Usage
========================================

Example of how to use the Advanced Quantitative Betting Model
for prediction and betting recommendations.
"""

from advanced_quant_model import AdvancedQuantModel
from quant_config import QuantConfig

def main():
    print("🎯 QUANTITATIVE MODEL EXAMPLE")
    print("=" * 40)
    
    # Load configuration
    config = QuantConfig()
    print("Enabled markets:", config.get_enabled_markets())
    
    # Initialize model
    model = AdvancedQuantModel()
    
    try:
        # Example fixture data
        fixture_data = {
            'league_avg_goals': 2.6,
            'home_form_goals': 1.4,
            'away_form_goals': 1.1,
            'is_weekend': 1,
            'home_team': 'Team A',
            'away_team': 'Team B'
        }
        
        # Example odds data
        odds_data = {
            'btts': {'yes': 2.10, 'no': 1.75},
            'over_15': {'odds': 1.45},
            'under_15': {'odds': 2.80},
            'home_handicap_minus_05': {'odds': 1.85}
        }
        
        # Generate predictions
        predictions = model.predict_markets(fixture_data)
        print("\\n🔮 PREDICTIONS:")
        for market, pred in predictions.items():
            print(f"  {market}: {pred}")
        
        # Generate betting recommendations
        recommendations = model.generate_betting_recommendations(fixture_data, odds_data)
        
        print(f"\\n💰 BETTING RECOMMENDATIONS ({len(recommendations)} found):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['market']}")
            print(f"     Expected Value: {rec['expected_value']:.1%}")
            print(f"     Confidence: {rec['confidence']:.1%}")
            print(f"     Stake: {rec['stake_recommendation']:.2f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to train models first or load historical data")
    
    finally:
        model.close_connection()

if __name__ == "__main__":
    main()
'''
        
        # Example 2: Data pipeline usage
        example2_content = '''#!/usr/bin/env python3
"""
Example 2: Data Pipeline Usage
=============================

Example of how to use the DuckDB data pipeline
for migrating and analyzing betting data.
"""

from duckdb_data_pipeline import DuckDBDataPipeline

def main():
    print("📊 DATA PIPELINE EXAMPLE")
    print("=" * 40)
    
    pipeline = DuckDBDataPipeline()
    
    try:
        # Migrate JSON data
        pipeline.migrate_json_data()
        
        # Get data summary
        summary = pipeline.get_data_summary()
        
        print("\\n📈 DATA SUMMARY:")
        fixtures = summary.get('fixtures', {})
        print(f"Total Fixtures: {fixtures.get('total', 0)}")
        print(f"Date Range: {fixtures.get('date_range', 'N/A')}")
        print(f"Average Goals: {fixtures.get('avg_total_goals', 0)}")
        print(f"BTTS Frequency: {fixtures.get('btts_frequency', 0):.1%}")
        
        print("\\n📊 MARKET COVERAGE:")
        for market_info in summary.get('odds', []):
            print(f"  {market_info['market']}: {market_info['records']} records")
        
        # Export training data
        training_file = pipeline.export_training_data()
        if training_file:
            print(f"\\n✅ Training data exported to: {training_file}")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    
    finally:
        pipeline.close_connection()

if __name__ == "__main__":
    main()
'''
        
        # Save examples
        try:
            with open('example_basic_usage.py', 'w') as f:
                f.write(example1_content)
            
            with open('example_data_pipeline.py', 'w') as f:
                f.write(example2_content)
            
            print("   ✅ Created example_basic_usage.py")
            print("   ✅ Created example_data_pipeline.py")
        except Exception as e:
            print(f"   ❌ Error creating examples: {e}")
    
    def display_next_steps(self):
        """
        Display next steps for the user
        """
        print("\n🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print("📋 NEXT STEPS:")
        print("   1. Review configuration in 'quant_config.json'")
        print("   2. Run 'python example_data_pipeline.py' to migrate data")
        print("   3. Run 'python example_basic_usage.py' to test predictions")
        print("   4. Integrate with your existing betting system")
        
        print("\n🎯 TARGET MARKETS (Evidence-Based):")
        print("   ✅ BTTS: 66.7% WR, +0.30 units (PROFITABLE)")
        print("   ✅ Over/Under 1.5: Replacement for failing 2.5 line")
        print("   ✅ Asian Handicap: New market exploration")
        print("   ❌ Over/Under 2.5: BANNED (43.3% WR, -1.15 units)")
        
        print("\n📊 KEY FILES:")
        print("   • advanced_quant_model.py - Main quantitative model")
        print("   • duckdb_data_pipeline.py - High-performance data pipeline")
        print("   • quant_config.py - Configuration management")
        print("   • betting_data.duckdb - DuckDB database (created)")
        print("   • quant_config.json - Configuration file (created)")
        
        print("\n💡 USAGE TIPS:")
        print("   • Start with BTTS predictions (proven profitable)")
        print("   • Test Over/Under 1.5 goals as 2.5 replacement")
        print("   • Use conservative stakes for Asian Handicap exploration")
        print("   • Monitor model performance and retrain regularly")
    
    def run_setup(self) -> bool:
        """
        Run complete setup process
        """
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Required Files", self.check_required_files),
            ("Dependencies", self.install_dependencies),
            ("Configuration", self.setup_configuration),
            ("Database Schema", self.setup_database_schema),
            ("Data Migration", self.migrate_existing_data),
            ("Model Training", self.run_initial_training)
        ]
        
        all_successful = True
        
        for step_name, step_func in setup_steps:
            print(f"\n🔄 {step_name}...")
            if not step_func():
                all_successful = False
                print(f"   ⚠️  {step_name} failed - continuing with remaining steps")
        
        # Always create examples
        print("\n🔄 Usage Examples...")
        self.create_usage_examples()
        
        return all_successful


def main():
    """
    Main setup execution
    """
    setup = QuantModelSetup()
    
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues")
    
    setup.display_next_steps()


if __name__ == "__main__":
    main()