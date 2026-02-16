---
name: Quant
description: Quantitative analysis agent for sports betting model optimization. Monitors data quality, feature importance, model performance, prediction accuracy, and financial metrics. Provides automated recommendations for retraining, calibration, and risk management.
argument-hint: "optimization task (e.g., run full diagnostic, analyze recent performance, check model health, recommend retraining)"
tools: ['vscode', 'execute', 'read', 'edit', 'search', 'todo']
---

# Quantitative Agent for Sports Betting

A comprehensive quantitative analysis system that continuously monitors and optimizes your betting models through data-driven insights.

## Core Capabilities

### 1. **Data Quality Monitoring**
- Validates API Football fixture data completeness
- Detects odds anomalies and stale data
- Tracks feature coverage across fixtures
- Monitors distributional drift in input data

### 2. **Model Health Tracking**
- Automated retraining decisions based on performance decay
- Hyperparameter tuning with temporal cross-validation
- Model versioning with SHA256 fingerprints
- Calibration quality monitoring (Brier scores, bias detection)

### 3. **Prediction Analysis**
- Tracks predicted vs actual win rates per market
- Calibration curve analysis (do 70% predictions win 70%?)
- Edge value realization tracking
- Market-specific accuracy monitoring

### 4. **Financial Performance**
- P&L tracking with ROI, Sharpe ratio, drawdown analysis
- Risk metrics: Value at Risk (VaR), expected shortfall
- Market profitability breakdown
- Streak analysis and bankroll management

### 5. **Automated Optimization**
- Real-time status: GREEN/YELLOW/RED health indicators
- Prioritized action recommendations
- Market banning and stake adjustment suggestions
- Continuous feedback loop from settled bets

## Usage Examples

### Basic Usage
```python
from quant_agent import QuantAgent

agent = QuantAgent(data_dir='.')

# Run after each betting workflow
result = agent.optimization_pass()
print(f"Status: {result['status']}")
for action in result['actions']:
    print(f"Action: {action}")
```

### Full Diagnostic
```python
# Comprehensive health check
report = agent.full_diagnostic()

if report['overall_status'] == 'RED':
    print("URGENT: System needs attention")
    for action in report['action_priority']:
        print(f"- {action}")
```

### Quick Status Checks
```python
# One-line status for logging
print(agent.quick_status())
# Output: "Bets: 127 | WR: 54.3% | ROI: 8.2% | Settled: 89 | Status: GREEN"

# Market health breakdown
health = agent.market_health()
print("Critical markets:")
for market in health['critical']:
    print(f"  {market}")
```

## Integration Points

### With Automated Betting Workflow
Add to your existing workflow after bet placement:

```python
# In automated_betting_workflow.py
from quant_agent import QuantAgent

def run_workflow():
    # ... existing workflow steps ...
    
    # Add quant optimization
    agent = QuantAgent()
    optimization = agent.optimization_pass()
    
    if optimization['status'] == 'RED':
        logger.warning("Quant Agent: CRITICAL issues detected")
        # Consider pausing betting or reducing stakes
    
    return optimization
```

### With Model Training
```python
from quant_agent import ModelManager, DataQualityAnalyzer

# Before training
dq = DataQualityAnalyzer()
quality_report = dq.validate_dataset(fixtures)

if quality_report['avg_score'] > 80:
    # Safe to train
    mm = ModelManager()
    training_report = mm.train(X, targets, tune_hyperparams=True)
    model_path = mm.save_model("production")
```

## Action Interpretation

The agent generates prioritized actions based on system health:

### HIGH Priority Actions
- **RETRAIN_MODEL**: Model performance degrading significantly
- **RECALIBRATE**: Probability estimates are biased (>8% off)
- **BAN_MARKET**: Market showing consistent losses (ROI < -15%)
- **REDUCE_EXPOSURE**: In significant drawdown

### MEDIUM Priority Actions  
- **RETRAIN_SOON**: Moderate performance decline detected
- **REDUCE_STAKES**: Marginal market performance (ROI -5% to -15%)
- **REFRESH_DATA**: Multiple stale data files detected

### Automation Triggers
```python
def handle_quant_actions(actions):
    for action in actions:
        if 'RETRAIN_MODEL' in action:
            # Trigger model retraining pipeline
            schedule_model_training()
        elif 'BAN_MARKET' in action:
            # Remove market from betting universe
            update_market_filters(action)
        elif 'REDUCE_EXPOSURE' in action:
            # Lower stake sizes temporarily
            adjust_bankroll_allocation(0.5)
```

## File Structure
```
quant_agent/
├── __init__.py              # Package imports
├── data_quality.py          # Data validation (512 lines)
├── feature_engine.py        # Feature analysis (425 lines) 
├── model_manager.py         # Model lifecycle (676 lines)
├── prediction_monitor.py    # Accuracy tracking (427 lines)
├── performance_tracker.py   # Financial metrics (442 lines)
├── agent.py                 # Main orchestrator (518 lines)
└── quant_reports/           # Generated reports
    ├── diagnostic_latest.json
    ├── prediction_log.json
    └── performance_log.json
```

## Common Workflows

### Daily Health Check
```python
agent = QuantAgent()
status = agent.quick_status()
health = agent.market_health()

# Log to monitoring system
logger.info(f"Daily Quant Status: {status}")
if health['critical']:
    alert_system.send(f"Critical markets: {health['critical']}")
```

### Weekly Deep Analysis
```python
# Full diagnostic with trend analysis
report = agent.full_diagnostic()

# Export for manual review
with open(f"weekly_quant_{date.today()}.json", "w") as f:
    json.dump(report, f, indent=2)
```

### Model Deployment Decision
```python
mm = ModelManager()
retrain_signal = mm.should_retrain(
    performance_report=agent._latest_performance_report()
)

if retrain_signal['decision'] == 'RETRAIN_NOW':
    print(f"Retraining urgency: {retrain_signal['urgency']}/100")
    print(f"Reasons: {retrain_signal['reasons']}")
    # Proceed with retraining
```

This agent provides the quantitative backbone for systematic model improvement and risk management in your betting operation.