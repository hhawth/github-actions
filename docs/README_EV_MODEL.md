# Improved EV Betting Model 🎯

A complete rewrite of the football betting expected value model that addresses all previous issues with a scientific, conservative approach.

## 🔧 What's Fixed

### Previous Problems ❌
- **34% accuracy** (worse than random)
- **Hardcoded placeholders** instead of real features
- **Extreme market betting** (OVER 4.5 goals, etc.)  
- **False EV claims** (126%, 82% returns)
- **No proper validation** (random splits on time-series data)
- **Overconfidence** leading to systematic losses

### New Improvements ✅
- **Proper feature engineering** (no hardcoded values)
- **Time-series cross-validation** (respects temporal nature)
- **Conservative EV calculations** (2-8% realistic range)
- **Market filtering** (avoid extreme/lottery markets)
- **Probability calibration** (fix overconfidence)
- **Risk management** (position sizing, limits)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_ev_model.txt
```

### 2. Test Installation
```bash
python demo_ev_model.py
```

### 3. Train Model
```bash
python ev_betting_runner.py --train
```

### 4. Find Value Bets
```bash
python ev_betting_runner.py --analyze
```

## 📊 Data Requirements

The model expects JSON files with the pattern: `api_football_merged_*.json`

Each fixture should contain:
- `fixture`: Match details (id, date, status)
- `league`: League information
- `teams`: Home and away team details  
- `goals`: Actual match scores (for completed matches)
- `odds`: Pre-match betting odds
- `predictions`: Team statistics and form data

## 🧠 Model Architecture

### Features (50+ engineered features)
- **Team Form**: Last 5 games performance, goals, attack/defense ratings
- **Season Stats**: Wins, draws, losses, goals for/against
- **Derived Metrics**: Win rates, goal differences, form momentum
- **Market Intelligence**: Betting odds, implied probabilities
- **Matchup Analysis**: Attack vs defense confrontations

### Models
- **Match Result**: 3-way classifier (Home/Draw/Away)
- **BTTS**: Binary classifier (Both Teams To Score)  
- **Goals**: Regression model for total goals

### Validation
- **Time-Series CV**: 5-fold cross-validation respecting temporal order
- **Probability Calibration**: Isotonic regression to fix overconfidence
- **Conservative Thresholds**: Reject models below 40% accuracy

## 💎 Expected Value Calculation

### Conservative Approach
```python
EV = (prediction_prob * (odds - 1)) - (1 - prediction_prob)

# Filters applied:
# - Min EV: 2% (realistic edge)
# - Max EV: 8% (reject suspicious high values)
# - Min probability: 15% (avoid longshots)
# - Market filtering: Mainstream markets only
```

### Risk Management
- **Maximum stake**: 2% of bankroll
- **Odds limits**: 1.2 to 10.0 (avoid extremes)
- **Position limits**: 1 bet per match maximum
- **Daily limits**: Conservative exposure caps

## 📈 Usage Examples

### Training
```bash
# Train new model
python ev_betting_runner.py --train

# This will:
# 1. Load all api_football_merged_*.json files
# 2. Extract features from completed matches
# 3. Train models with time-series CV
# 4. Save model with timestamp
```

### Analysis
```bash
# Analyze latest fixtures
python ev_betting_runner.py --analyze

# Use specific fixtures file
python ev_betting_runner.py --analyze --fixtures api_football_merged_2026-02-10.json

# Use existing model
python ev_betting_runner.py --analyze --model ev_model_20260212_0726.pkl
```

### Validation
```bash
# Check model performance
python ev_betting_runner.py --validate --model ev_model_20260212_0726.pkl
```

## 🛡️ Safety Features

### Model Validation Checks
- **Minimum accuracy thresholds** (40% match, 52% BTTS)
- **Performance warnings** for poor models
- **Conservative recommendations** always

### Betting Safeguards
- **Small stake recommendations** (£0.10-£0.50 maximum)
- **Market filtering** (no extreme markets)
- **EV caps** (reject >8% EV as suspicious)
- **Position limits** (1 bet per match)

### Output Example
```
💎 VALUE BET #1
   📍 Match: Arsenal vs Chelsea  
   🎰 Market: Home Win
   📊 Our Probability: 42.3%
   💰 Odds: 2.45
   💎 Expected Value: 0.034 (3.4% ROI)
   💸 Stake: Low (£0.25 maximum)
```

## 📊 Performance Expectations

### Realistic Targets
- **Match accuracy**: 40-45% (significantly above random 33%)
- **BTTS accuracy**: 55-60% (above random 50%)
- **Expected value range**: 2-8% per bet
- **Win rate**: 40-45% of bets profitable
- **Long-term ROI**: 3-6% if model is accurate

### Warning Signs
- **Match accuracy <35%**: Don't bet, model is broken
- **BTTS accuracy <52%**: Avoid BTTS markets
- **High EV claims >10%**: Likely false positives
- **Consistent losses**: Stop and retrain

## 🔍 Model Interpretation

### Feature Importance
The model automatically calculates which features are most predictive:
- Recent form (last 5 games)
- Home/away performance splits  
- Goal scoring and conceding rates
- Head-to-head attack vs defense
- Market odds as baseline probability

### Probability Calibration
Raw model probabilities are calibrated using isotonic regression to ensure:
- 40% predicted probability = 40% actual win rate
- No systematic overconfidence
- Conservative estimates for betting

## 🚨 Important Disclaimers

1. **Past performance doesn't guarantee future results**
2. **Sports betting is inherently risky** - never bet more than you can afford to lose
3. **Start with paper trading** to validate model performance  
4. **Use tiny stakes initially** (£0.10-£0.50 maximum)
5. **Track all bets** to measure real-world performance
6. **Stop betting if losing** consistently

## 🆚 Comparison to Previous Model

| Aspect | Old Model | New Model |
|--------|-----------|-----------|
| Accuracy | 34% (failing) | 40-45% target |
| Features | Hardcoded placeholders | Real calculated metrics |
| Validation | Random splits | Time-series CV |
| EV Range | 100%+ (unrealistic) | 2-8% (conservative) |
| Markets | Extreme (OVER 4.5) | Mainstream only |
| Risk Mgmt | None | Comprehensive limits |
| Calibration | Overconfident | Probability calibrated |

## 🔧 Customization

### Config Options
```python
config = {
    'min_ev': 0.03,           # Require 3% minimum edge
    'max_ev': 0.06,           # Cap at 6% (more conservative)
    'min_probability': 0.20,  # 20% minimum win chance
    'max_stake_pct': 0.01,    # 1% max position size
    'allowed_markets': ['Match Winner'],  # Only main market
}

model = ImprovedEVBettingModel(config)
```

## 📞 Support

This is a complete rewrite designed to be:
- **Conservative** by default
- **Transparent** about performance  
- **Safety-first** in all recommendations
- **Realistic** about expected returns

The goal is capital preservation while finding small edges, not get-rich-quick schemes.

**Remember: The house edge exists for a reason. Bet responsibly.**