# Football Prediction & Betting Agents

A multi-agent Streamlit app providing global fixtures, enhanced predictions, market intelligence, risk management, and accumulator strategies.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Features
- Predictions, analysis, betting suggestions
- Value bets via Market Intelligence Agent
- Kelly staking via Risk Management Agent
- News impact via BBC Sport scraping
- Accumulator variants (Ultra Safe → Maximum Risk)

## Docker
```bash
docker build -t football-betting-app .
docker run -p 8501:8501 football-betting-app
```
