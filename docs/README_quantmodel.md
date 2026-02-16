# Advanced Quantitative Betting Model

A sophisticated, evidence-based quantitative analysis system for sports betting using DuckDB analytics and machine learning.

## 📊 Evidence-Based Market Selection

Based on extensive empirical analysis of betting performance:

### ✅ **Profitable Markets** (ENABLED)
- **BTTS (Both Teams To Score)**: 66.7% Win Rate, +0.30 units profit
- **Over/Under 1.5 Goals**: Replacement for failing 2.5 goals line  
- **Asian Handicap**: New market exploration with high-confidence predictions

### ❌ **Banned Markets** (DISABLED)
- **Over/Under 2.5 Goals**: 43.3% Win Rate, -1.15 units loss
- **Double Chance**: 58.5% Win Rate, -1.18 units (unprofitable)

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies and setup system
python setup_quant_model.py
```

### 2. Basic Usage
```python
from advanced_quant_model import AdvancedQuantModel
from quant_config import QuantConfig

# Initialize model
model = AdvancedQuantModel()
config = QuantConfig()

# Generate predictions
predictions = model.predict_markets(fixture_data)
recommendations = model.generate_betting_recommendations(fixture_data, odds_data)
```

### 3. Data Pipeline
```python
from duckdb_data_pipeline import DuckDBDataPipeline

# Migrate JSON data to DuckDB  
pipeline = DuckDBDataPipeline()
pipeline.migrate_json_data()
```

## 🏗️ System Architecture

### Core Components

1. **advanced_quant_model.py** - Main quantitative model
   - Ensemble ML models (Random Forest + Gradient Boosting)
   - Probability calibration with isotonic regression
   - Expected value calculations with Kelly criterion
   - Evidence-based market targeting

2. **duckdb_data_pipeline.py** - High-performance data pipeline
   - JSON to DuckDB migration
   - Feature engineering and statistics generation
   - Bulk data processing and optimization

3. **quant_config.py** - Configuration management
   - Evidence-based default settings
   - Risk management parameters
   - Market-specific thresholds

4. **setup_quant_model.py** - Complete system setup
   - Dependency installation
   - Database initialization  
   - Model training pipeline

## 📈 Key Features

### Advanced Analytics
- **DuckDB Backend**: High-performance analytical database
- **Time-Series Validation**: Proper model evaluation for temporal data
- **Feature Engineering**: League context, team form, interaction features
- **Calibration**: Reliable probability estimates using isotonic regression

### Risk Management
- **Kelly Criterion**: Optimal stake sizing with safety factors
- **Multi-Level Filtering**: Market filters, EV thresholds, confidence checks
- **Bankroll Management**: Exposure limits and drawdown protection
- **Evidence-Based Decisions**: Only bet on proven profitable markets

### Model Intelligence
- **Ensemble Methods**: Combines multiple ML algorithms for stability
- **Market Specialization**: Different models optimized for each market type
- **Continuous Learning**: Automated retraining based on performance
- **Bias Correction**: Market-specific calibration factors

## 🎯 Target Market Analysis

### BTTS (Both Teams To Score)
- **Status**: PROFITABLE (Primary target)
- **Historical**: 66.7% WR, +0.30 units
- **Minimum Edge**: 3.0%
- **Max Stake**: 5% of bankroll

### Over/Under 1.5 Goals  
- **Status**: REPLACEMENT for banned 2.5 line
- **Confidence Range**: 65-76% on high-quality predictions
- **Minimum Edge**: 2.0% (testing threshold)
- **Max Stake**: 3% of bankroll

### Asian Handicap
- **Status**: EXPLORATION (new market)
- **Focus**: -0.5, -1.5 goal lines
- **Minimum Edge**: 3.5% (conservative)
- **Max Stake**: 2% of bankroll

## 📋 Configuration

Edit `quant_config.json` to customize:

```json
{
  "model_settings": {
    "target_markets": {
      "btts": {
        "enabled": true,
        "min_edge": 0.03,
        "confidence_threshold": 0.65
      }
    }
  },
  "risk_management": {
    "bankroll_management": {
      "max_stake_per_bet": 5.0,
      "kelly_safety_factor": 0.25
    }
  }
}
```

## 🔧 Development Workflow

### 1. Data Processing
```bash
# Migrate historical data
python -c "from duckdb_data_pipeline import DuckDBDataPipeline; DuckDBDataPipeline().migrate_json_data()"
```

### 2. Model Training
```bash
# Train quantitative models
python -c "from advanced_quant_model import AdvancedQuantModel; m=AdvancedQuantModel(); m.train_market_models(m.load_historical_data())"
```

### 3. Production Usage
```python
# Generate betting recommendations
model = AdvancedQuantModel() 
recommendations = model.generate_betting_recommendations(fixture_data, odds_data)
```

## 📊 Performance Monitoring

The system tracks:
- Model accuracy and calibration
- Expected value realization  
- Market-specific performance
- Risk metrics (Sharpe ratio, drawdown)
- Bet outcome analysis

## 🛡️ Risk Management

### Built-in Protections
- **Market Banning**: Automatically disable unprofitable markets
- **Stake Limits**: Kelly criterion with 25% safety factor
- **Exposure Control**: Daily and total exposure limits
- **Confidence Thresholds**: Only bet on high-confidence predictions
- **Stop-Loss**: Automatic betting suspension on excessive losses

### Evidence-Based Filtering
- Minimum 55% probability threshold
- Maximum stake per bet: 5% of bankroll
- Edge requirements: 2-3.5% depending on market
- Odds range filtering to avoid extreme values

## 📈 Integration with Existing Systems

The model integrates seamlessly with:
- Existing betting workflows
- API Football data sources
- Matchbook exchange integration
- Automated betting systems

## 🔍 Model Validation

### Backtesting Results
- **BTTS**: Consistently profitable across multiple seasons
- **Over/Under 2.5**: Poor performance led to replacement with 1.5 line
- **1.5 Goals**: Shows 65-76% prediction confidence vs 43.3% for 2.5 line

### Statistical Rigor
- Time-series cross-validation
- Out-of-sample testing
- Calibration curve analysis
- Feature importance evaluation

## 📚 Examples

See included example files:
- `example_basic_usage.py` - Basic model usage
- `example_data_pipeline.py` - Data processing workflow

## 🤝 Support

For issues or questions:
1. Check configuration in `quant_config.json`
2. Review logs in `quant_model.log`
3. Validate data in DuckDB using pipeline tools
4. Ensure models are trained on sufficient historical data

---

**Built for Evidence-Based Betting** | **Powered by DuckDB Analytics** | **Risk-Managed by Design**