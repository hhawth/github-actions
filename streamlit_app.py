# Multi-Agent Football Prediction System
# Install required packages: pip install streamlit openai langchain pandas plotly

import streamlit as st
import plotly.express as px
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
import requests
import feedparser
from bs4 import BeautifulSoup

# Import your existing modules
from get_fixtures import get_fixtures
from stat_getter import (
    get_form, get_relative_performance, 
    get_top_scorers, get_official_team_names,
    get_todays_fixtures, predict_match_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BettingAgent:
    def __init__(self):
        self.name = "💰 Betting Agent"
        self.min_confidence_single = 0.65  # 65% confidence for single bets
        self.min_confidence_double = 0.55  # 55% confidence for double chance
        self.min_confidence_acc = 0.50     # 50% confidence for accumulator inclusion
        self.max_odds_estimate = 10.0      # Maximum estimated odds for safety
    
    def estimate_odds(self, probability):
        """Convert probability to estimated odds"""
        if probability <= 0.01:
            return self.max_odds_estimate
        return min(1 / probability, self.max_odds_estimate)
    
    def analyze_match_for_betting(self, prediction, home_team, away_team, news_agent=None):
        """Analyze a single match for betting opportunities with news impact"""
        confidence = prediction['confidence']
        home_prob = prediction['home_win_prob']
        draw_prob = prediction['draw_prob']
        away_prob = prediction['away_win_prob']
        
        # Get news impact if agent is available
        news_impact = None
        if news_agent:
            try:
                news_impact = news_agent.analyze_team_news(home_team, away_team)
            except Exception as e:
                logger.warning(f"Error getting news impact: {e}")
        
        # Adjust probabilities based on news impact
        if news_impact:
            home_news_impact = news_impact['home_team']['impact_score']
            away_news_impact = news_impact['away_team']['impact_score']
            
            # Apply news impact (small adjustments)
            home_prob_adjusted = min(0.95, max(0.05, home_prob + (home_news_impact * 0.1)))
            away_prob_adjusted = min(0.95, max(0.05, away_prob + (away_news_impact * 0.1)))
            
            # Renormalize probabilities
            total_prob = home_prob_adjusted + away_prob_adjusted + draw_prob
            home_prob = home_prob_adjusted / total_prob
            away_prob = away_prob_adjusted / total_prob
            draw_prob = draw_prob / total_prob
        
        suggestions = []
        
        # Always include basic single outcome bets (with lower threshold for display)
        min_display_threshold = 0.30  # 30% minimum to show any betting option
        
        # Home win bet
        if home_prob >= min_display_threshold:
            bet_type = 'High Confidence Win' if home_prob >= self.min_confidence_single else 'Standard Win'
            suggestions.append({
                'type': bet_type,
                'bet': f'{home_team} to Win',
                'confidence': home_prob,
                'estimated_odds': self.estimate_odds(home_prob),
                'reasoning': f'Home win probability: {home_prob:.1%}'
            })
        
        # Away win bet
        if away_prob >= min_display_threshold:
            bet_type = 'High Confidence Win' if away_prob >= self.min_confidence_single else 'Standard Win'
            suggestions.append({
                'type': bet_type,
                'bet': f'{away_team} to Win',
                'confidence': away_prob,
                'estimated_odds': self.estimate_odds(away_prob),
                'reasoning': f'Away win probability: {away_prob:.1%}'
            })
        
        # Draw bet
        if draw_prob >= min_display_threshold:
            bet_type = 'High Confidence Draw' if draw_prob >= self.min_confidence_single else 'Standard Draw'
            suggestions.append({
                'type': bet_type,
                'bet': 'Match to end in Draw',
                'confidence': draw_prob,
                'estimated_odds': self.estimate_odds(draw_prob),
                'reasoning': f'Draw probability: {draw_prob:.1%}'
            })
        
        # Double chance bets for safer options
        home_or_draw = home_prob + draw_prob
        away_or_draw = away_prob + draw_prob
        home_or_away = home_prob + away_prob
        
        # Always show double chance if individual outcomes are uncertain
        if home_or_draw >= self.min_confidence_double:
            suggestions.append({
                'type': 'Double Chance',
                'bet': f'{home_team} Win or Draw',
                'confidence': home_or_draw,
                'estimated_odds': self.estimate_odds(home_or_draw),
                'reasoning': f'Safe bet covering home win or draw ({home_or_draw:.1%})'
            })
        
        if away_or_draw >= self.min_confidence_double:
            suggestions.append({
                'type': 'Double Chance',
                'bet': f'{away_team} Win or Draw',
                'confidence': away_or_draw,
                'estimated_odds': self.estimate_odds(away_or_draw),
                'reasoning': f'Safe bet covering away win or draw ({away_or_draw:.1%})'
            })
        
        # Home or Away (no draw) - useful for matches unlikely to draw
        if home_or_away >= self.min_confidence_double and draw_prob < 0.25:
            suggestions.append({
                'type': 'Double Chance',
                'bet': f'{home_team} or {away_team} Win (No Draw)',
                'confidence': home_or_away,
                'estimated_odds': self.estimate_odds(home_or_away),
                'reasoning': f'Bet against draw - match likely to have a winner ({home_or_away:.1%})'
            })
        
        return suggestions
    
    def get_accumulator_selections(self, all_predictions, fold_count):
        """Get best selections for accumulator bets"""
        acc_candidates = []
        
        for match_data in all_predictions:
            prediction = match_data['prediction']
            home_team = match_data['home_team']
            away_team = match_data['away_team']
            
            home_prob = prediction['home_win_prob']
            draw_prob = prediction['draw_prob']
            away_prob = prediction['away_win_prob']
            
            # Calculate double chance probabilities
            home_or_draw = home_prob + draw_prob
            away_or_draw = away_prob + draw_prob
            home_or_away = home_prob + away_prob
            
            # Find best single outcome
            best_single_prob = max(home_prob, draw_prob, away_prob)
            best_double_prob = max(home_or_draw, away_or_draw, home_or_away)
            
            # Choose between single and double chance based on which has higher probability
            if best_double_prob > best_single_prob and best_double_prob >= 0.60:  # 60% threshold for double chance
                # Use double chance option
                if best_double_prob == home_or_draw:
                    selection = f'{home_team} Win or Draw'
                    probability = home_or_draw
                elif best_double_prob == away_or_draw:
                    selection = f'{away_team} Win or Draw'
                    probability = away_or_draw
                else:  # home_or_away
                    selection = f'{home_team} or {away_team} Win'
                    probability = home_or_away
                
                acc_candidates.append({
                    'match': f'{home_team} vs {away_team}',
                    'selection': selection,
                    'probability': probability,
                    'estimated_odds': self.estimate_odds(probability),
                    'bet_type': 'Double Chance'
                })
            
            # Also consider single outcomes with lower threshold
            elif best_single_prob >= 0.40:  # 40% threshold for single outcomes
                if best_single_prob == home_prob:
                    selection = f'{home_team} Win'
                elif best_single_prob == away_prob:
                    selection = f'{away_team} Win'
                else:
                    selection = 'Draw'
                
                acc_candidates.append({
                    'match': f'{home_team} vs {away_team}',
                    'selection': selection,
                    'probability': best_single_prob,
                    'estimated_odds': self.estimate_odds(best_single_prob),
                    'bet_type': 'Single Outcome'
                })
        
        # Sort by probability (highest first)
        acc_candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        # Always try to create accumulator if we have any matches
        if len(acc_candidates) >= fold_count:
            selected = acc_candidates[:fold_count]
            
            total_odds = 1.0
            total_probability = 1.0
            
            for sel in selected:
                total_odds *= sel['estimated_odds']
                total_probability *= sel['probability']
            
            return {
                'selections': selected,
                'total_estimated_odds': total_odds,
                'combined_probability': total_probability * 100,  # Convert to percentage
                'fold_count': len(selected)
            }
        
        # If we don't have enough for full accumulator, create smaller one
        elif len(acc_candidates) >= 3:  # Minimum 3 selections
            selected = acc_candidates[:min(len(acc_candidates), fold_count)]
            
            total_odds = 1.0
            total_probability = 1.0
            
            for sel in selected:
                total_odds *= sel['estimated_odds']
                total_probability *= sel['probability']
            
            return {
                'selections': selected,
                'total_estimated_odds': total_odds,
                'combined_probability': total_probability * 100,  # Convert to percentage
                'fold_count': len(selected),
                'warning': f'Only {len(selected)} selections available (requested {fold_count})'
            }
        
        # Return a debug message if no candidates found
        return {
            'debug_info': {
                'total_predictions': len(all_predictions),
                'candidates_found': len(acc_candidates),
                'threshold_used': 0.45
            }
        }
    
    def get_accumulator_variants(self, all_predictions):
        """Generate multiple accumulator variants with different risk/success levels"""
        variants = []
        
        # 1. Ultra Safe (80%+ confidence, mostly double chance)
        ultra_safe = self.get_accumulator_by_criteria(all_predictions, 0.80, 'double_chance_preferred', 6, "Ultra Safe")
        if ultra_safe:
            variants.append(('🛡️ Ultra Safe (6-fold)', ultra_safe, '🛡️'))
        
        # 2. Conservative (70%+ confidence, mixed)
        conservative = self.get_accumulator_by_criteria(all_predictions, 0.70, 'mixed', 8, "Conservative")
        if conservative:
            variants.append(('🟢 Conservative (8-fold)', conservative, '🟢'))
        
        # 3. Balanced (60%+ confidence)
        balanced = self.get_accumulator_by_criteria(all_predictions, 0.60, 'mixed', 10, "Balanced")
        if balanced:
            variants.append(('⚖️ Balanced (10-fold)', balanced, '⚖️'))
        
        # 4. Aggressive (50%+ confidence)
        aggressive = self.get_accumulator_by_criteria(all_predictions, 0.50, 'single_preferred', 12, "Aggressive")
        if aggressive:
            variants.append(('🟡 Aggressive (12-fold)', aggressive, '🟡'))
        
        # 5. High Risk (40%+ confidence)
        high_risk = self.get_accumulator_by_criteria(all_predictions, 0.40, 'single_preferred', 15, "High Risk")
        if high_risk:
            variants.append(('🔴 High Risk (15-fold)', high_risk, '🔴'))
        
        # 6. Maximum Risk (35%+ confidence)
        max_risk = self.get_accumulator_by_criteria(all_predictions, 0.35, 'single_only', 20, "Maximum Risk")
        if max_risk:
            variants.append(('💀 Maximum Risk (20-fold)', max_risk, '💀'))
        
        return variants
    
    def get_accumulator_by_criteria(self, all_predictions, min_prob, strategy, target_count, risk_level):
        """Get accumulator selections based on specific criteria"""
        candidates = []
        
        for match_data in all_predictions:
            prediction = match_data['prediction']
            home_team = match_data['home_team']
            away_team = match_data['away_team']
            
            home_prob = prediction['home_win_prob']
            draw_prob = prediction['draw_prob']
            away_prob = prediction['away_win_prob']
            
            home_or_draw = home_prob + draw_prob
            away_or_draw = away_prob + draw_prob
            home_or_away = home_prob + away_prob
            
            best_single_prob = max(home_prob, draw_prob, away_prob)
            best_double_prob = max(home_or_draw, away_or_draw, home_or_away)
            
            # Apply strategy-specific selection
            if strategy == 'double_chance_preferred' and best_double_prob >= min_prob:
                # Prefer double chance for safety
                if best_double_prob == home_or_draw:
                    selection = f'{home_team} Win or Draw'
                    prob = home_or_draw
                elif best_double_prob == away_or_draw:
                    selection = f'{away_team} Win or Draw'
                    prob = away_or_draw
                else:
                    selection = f'{home_team} or {away_team} Win'
                    prob = home_or_away
                bet_type = 'Double Chance'
                
            elif strategy == 'single_preferred' and best_single_prob >= min_prob:
                # Prefer single outcomes for higher odds
                if best_single_prob == home_prob:
                    selection = f'{home_team} Win'
                elif best_single_prob == away_prob:
                    selection = f'{away_team} Win'
                else:
                    selection = 'Draw'
                prob = best_single_prob
                bet_type = 'Single Outcome'
                
            elif strategy == 'single_only' and best_single_prob >= min_prob:
                # Only single outcomes
                if best_single_prob == home_prob:
                    selection = f'{home_team} Win'
                elif best_single_prob == away_prob:
                    selection = f'{away_team} Win'
                else:
                    selection = 'Draw'
                prob = best_single_prob
                bet_type = 'Single Outcome'
                
            elif strategy == 'mixed':
                # Use whichever is best and meets threshold
                if best_double_prob >= min_prob and best_double_prob > best_single_prob:
                    if best_double_prob == home_or_draw:
                        selection = f'{home_team} Win or Draw'
                        prob = home_or_draw
                    elif best_double_prob == away_or_draw:
                        selection = f'{away_team} Win or Draw'
                        prob = away_or_draw
                    else:
                        selection = f'{home_team} or {away_team} Win'
                        prob = home_or_away
                    bet_type = 'Double Chance'
                elif best_single_prob >= min_prob:
                    if best_single_prob == home_prob:
                        selection = f'{home_team} Win'
                    elif best_single_prob == away_prob:
                        selection = f'{away_team} Win'
                    else:
                        selection = 'Draw'
                    prob = best_single_prob
                    bet_type = 'Single Outcome'
                else:
                    continue
            else:
                continue
                
            candidates.append({
                'match': f'{home_team} vs {away_team}',
                'selection': selection,
                'probability': prob,
                'estimated_odds': self.estimate_odds(prob),
                'bet_type': bet_type
            })
        
        # Sort by probability (highest first)
        candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        # Take best selections up to target count
        if len(candidates) >= min(target_count, 3):  # Need at least 3
            selected = candidates[:min(len(candidates), target_count)]
            
            total_odds = 1.0
            total_probability = 1.0
            
            for sel in selected:
                total_odds *= sel['estimated_odds']
                total_probability *= sel['probability']
            
            return {
                'selections': selected,
                'total_estimated_odds': total_odds,
                'combined_probability': total_probability * 100,
                'fold_count': len(selected),
                'risk_level': risk_level,
                'strategy': strategy
            }
        
        return None
    
    def generate_betting_report(self, fixtures_df, news_agent=None):
        """Generate comprehensive betting suggestions"""
        all_predictions = []
        single_bets = []
        
        # Analyze more matches for better accumulator options
        # Use more matches for accumulators but limit single bet analysis for performance
        acc_sample_size = min(50, len(fixtures_df))  # Increased from 25 to 50 for accumulators
        single_bet_sample_size = min(25, len(fixtures_df))  # Keep 25 for single bet analysis
        
        for idx in range(acc_sample_size):
            row = fixtures_df.iloc[idx]
            home_team = row['Home']
            away_team = row['Away']
            
            try:
                prediction = predict_match_score(home_team, away_team, fixtures_df)
                
                # Only analyze single bets for first 25 matches (performance)
                if idx < single_bet_sample_size:
                    match_bets = self.analyze_match_for_betting(
                        prediction, home_team, away_team, news_agent=news_agent
                    )
                    single_bets.extend(match_bets)
                
                # But include all matches for accumulator consideration
                all_predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'prediction': prediction,
                    'country': row.get('Country', 'Unknown'),
                    'time': row.get('Time', 'TBD')
                })
                
            except Exception:
                continue
        
        # Generate accumulator bet variants (from safest to riskiest)
        accumulator_variants = self.get_accumulator_variants(all_predictions)
        
        return {
            'single_bets': sorted(single_bets, key=lambda x: x['confidence'], reverse=True),
            'accumulator_variants': accumulator_variants,
            'total_matches_analyzed': len(all_predictions)
        }

@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    confidence: float
    likely_outcome: str
    insights: List[str]

@dataclass
class TeamInsight:
    team: str
    form: float
    recent_performance: float
    key_players: List[str]
    weaknesses: List[str]
    strengths: List[str]

class PredictionAgent:
    """Agent responsible for match predictions using existing functions"""
    
    def __init__(self):
        self.name = "🔮 Prediction Agent"
        
    @st.cache_data(ttl=3600)
    def get_all_predictions(_self) -> Dict[str, MatchPrediction]:
        """Get predictions for all available fixtures"""
        try:
            fixtures_data = get_fixtures()
            predictions = {}
            
            for fixture_key, fixture_info in fixtures_data.items():
                if fixture_info.get('home_goals') != 'N/A' and fixture_info.get('away_goals') != 'N/A':
                    prediction = MatchPrediction(
                        home_team=fixture_info['home_team'],
                        away_team=fixture_info['away_team'],
                        home_score=fixture_info['home_goals'],
                        away_score=fixture_info['away_goals'],
                        home_win_prob=fixture_info.get('broker_home_win_percentage', 0),
                        away_win_prob=fixture_info.get('broker_away_win_percentage', 0),
                        draw_prob=fixture_info.get('broker_draw_percentage', 0),
                        confidence=_self._calculate_confidence(fixture_info),
                        likely_outcome=fixture_info.get('likely_outcome', 'N/A'),
                        insights=_self._generate_insights(fixture_info)
                    )
                    predictions[fixture_key] = prediction
                    
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {}
    
    def _calculate_confidence(self, fixture_info: Dict) -> float:
        """Calculate confidence based on available data quality"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have good data
        if fixture_info.get('home_goals') != 'N/A':
            confidence += 0.2
        if fixture_info.get('likely_home_scorers'):
            confidence += 0.1
        if fixture_info.get('avg_goals_home') and fixture_info.get('avg_goals_away'):
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def _generate_insights(self, fixture_info: Dict) -> List[str]:
        """Generate insights for a specific match"""
        insights = []
        
        try:
            home_avg = fixture_info.get('avg_goals_home', 0)
            away_avg = fixture_info.get('avg_goals_away', 0)
            
            if home_avg > 2.0:
                insights.append(f"🔥 {fixture_info['home_team']} has strong home scoring record ({home_avg:.1f} goals/game)")
            
            if away_avg > 1.5:
                insights.append(f"⚡ {fixture_info['away_team']} scores well away from home ({away_avg:.1f} goals/game)")
                
            if fixture_info.get('home_goals', 0) > fixture_info.get('away_goals', 0):
                insights.append(f"🏠 Home advantage favors {fixture_info['home_team']}")
            elif fixture_info.get('away_goals', 0) > fixture_info.get('home_goals', 0):
                insights.append(f"🛣️ {fixture_info['away_team']} predicted to win away")
            else:
                insights.append("⚖️ Evenly matched teams - potential for a draw")
                
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
            
        return insights

class AnalysisAgent:
    """Agent for deeper analysis and insights"""
    
    def __init__(self):
        self.name = "📊 Analysis Agent"
        
    @st.cache_data(ttl=3600)
    def get_team_insights(_self) -> Dict[str, TeamInsight]:
        """Get comprehensive team analysis"""
        try:
            form_data = get_form()
            performance_data = get_relative_performance()
            top_scorers = get_top_scorers()
            
            insights = {}
            
            for team in get_official_team_names():
                team_scorers = top_scorers[top_scorers['team_name'] == team]['second_name'].head(3).tolist()
                
                insight = TeamInsight(
                    team=team,
                    form=form_data.get(team, 0),
                    recent_performance=performance_data.get(team, 0),
                    key_players=team_scorers,
                    strengths=_self._analyze_strengths(team, form_data.get(team, 0), performance_data.get(team, 0)),
                    weaknesses=_self._analyze_weaknesses(team, form_data.get(team, 0), performance_data.get(team, 0))
                )
                insights[team] = insight
                
            return insights
        except Exception as e:
            logger.error(f"Error getting team insights: {e}")
            return {}
    
    def _analyze_strengths(self, team: str, form: float, performance: float) -> List[str]:
        """Analyze team strengths"""
        strengths = []
        
        if form > 2.0:
            strengths.append("Excellent recent form")
        if performance > 0.6:
            strengths.append("Performing above expectations")
        if form > 1.5 and performance > 0.5:
            strengths.append("Consistent team performance")
            
        return strengths or ["Steady performance"]
    
    def _analyze_weaknesses(self, team: str, form: float, performance: float) -> List[str]:
        """Analyze team weaknesses"""
        weaknesses = []
        
        if form < 1.0:
            weaknesses.append("Poor recent form")
        if performance < 0.3:
            weaknesses.append("Underperforming expectations")
        if form < 1.2 and performance < 0.4:
            weaknesses.append("Struggling for consistency")
            
        return weaknesses or ["No major weaknesses identified"]
    
    @st.cache_data(ttl=3600)
    def get_league_analysis(_self) -> Dict[str, Any]:
        """Get overall league analysis"""
        try:
            form_data = get_form()
            performance_data = get_relative_performance()
            
            # Sort teams by form
            sorted_by_form = sorted(form_data.items(), key=lambda x: x[1], reverse=True)
            
            analysis = {
                "top_form_teams": sorted_by_form[:5],
                "bottom_form_teams": sorted_by_form[-5:],
                "overperformers": [(team, perf) for team, perf in performance_data.items() if perf > 0.7],
                "underperformers": [(team, perf) for team, perf in performance_data.items() if perf < 0.3],
                "average_goals": sum(form_data.values()) / len(form_data)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error getting league analysis: {e}")
            return {}
    
    def get_head_to_head_analysis(self, home_team, away_team):
        """Analyze historical head-to-head record between two teams"""
        try:
            h2h_data = {
                'total_matches': 0,
                'home_team_wins': 0,
                'away_team_wins': 0,
                'draws': 0,
                'avg_goals_home': 0.0,
                'avg_goals_away': 0.0,
                'recent_trend': 'Neutral',
                'psychological_edge': 'None'
            }
            
            # In real implementation, would fetch historical data
            # For now, simulate based on team strength
            home_form = self.get_team_form(home_team)
            away_form = self.get_team_form(away_team)
            
            if home_form and away_form:
                home_strength = home_form.get('attack', 0) + home_form.get('defense', 0)
                away_strength = away_form.get('attack', 0) + away_form.get('defense', 0)
                
                if home_strength > away_strength * 1.2:
                    h2h_data['psychological_edge'] = home_team
                elif away_strength > home_strength * 1.2:
                    h2h_data['psychological_edge'] = away_team
            
            return h2h_data
        except Exception as e:
            logger.error(f"Error getting head-to-head analysis: {e}")
            return {}

class MarketIntelligenceAgent:
    """Agent for tracking betting market odds and finding value bets"""
    
    def __init__(self):
        self.name = "🏪 Market Intelligence Agent"
        self.odds_cache = {}
    
    def get_market_value_bets(self, predictions_data):
        """Find value bets by comparing our predictions with market odds"""
        value_bets = []
        
        for match_data in predictions_data:
            prediction = match_data['prediction']
            home_team = match_data['home_team']
            away_team = match_data['away_team']
            
            # Simulate market odds (in real implementation, scrape from betting sites)
            market_odds = self._estimate_market_odds(prediction)
            our_probabilities = {
                'home': prediction['home_win_prob'],
                'draw': prediction['draw_prob'], 
                'away': prediction['away_win_prob']
            }
            
            # Find value opportunities
            for outcome, our_prob in our_probabilities.items():
                market_implied_prob = 1 / market_odds[outcome] if market_odds[outcome] > 0 else 0
                
                # Value bet if our probability is significantly higher than market
                if our_prob > market_implied_prob * 1.15:  # 15% edge threshold
                    value_percentage = ((our_prob / market_implied_prob) - 1) * 100
                    value_bets.append({
                        'match': f"{home_team} vs {away_team}",
                        'outcome': outcome.title(),
                        'market_odds': market_odds[outcome],
                        'our_probability': our_prob,
                        'market_probability': market_implied_prob,
                        'value_percentage': value_percentage,
                        'confidence': 'High' if value_percentage > 25 else 'Medium'
                    })
        
        return sorted(value_bets, key=lambda x: x['value_percentage'], reverse=True)
    
    def _estimate_market_odds(self, prediction):
        """Estimate market odds based on probabilities (placeholder for real odds API)"""
        margin = 0.05  # Bookmaker margin
        
        home_prob = max(prediction['home_win_prob'] - margin, 0.01)
        draw_prob = max(prediction['draw_prob'] - margin, 0.01) 
        away_prob = max(prediction['away_win_prob'] - margin, 0.01)
        
        return {
            'home': round(1 / home_prob, 2),
            'draw': round(1 / draw_prob, 2),
            'away': round(1 / away_prob, 2)
        }

class RiskManagementAgent:
    """Agent for bankroll management and risk assessment"""
    
    def __init__(self):
        self.name = "⚠️ Risk Management Agent"
    
    def calculate_kelly_stake(self, probability, odds, bankroll):
        """Calculate optimal stake using Kelly Criterion"""
        if probability <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = probability, q = 1-probability
        b = odds - 1
        p = probability
        q = 1 - probability
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 10% of bankroll for safety
        return min(max(kelly_fraction, 0), 0.10) * bankroll
    
    def assess_portfolio_risk(self, betting_selections):
        """Assess overall risk of betting portfolio"""
        total_stake = sum(bet.get('suggested_stake', 0) for bet in betting_selections)
        total_probability = 1
        
        for bet in betting_selections:
            total_probability *= bet.get('probability', 0.5)
        
        risk_metrics = {
            'portfolio_success_probability': total_probability,
            'total_exposure': total_stake,
            'risk_level': self._categorize_risk(total_probability, total_stake),
            'diversification_score': len(betting_selections),
            'max_loss': total_stake,
            'expected_return': self._calculate_expected_return(betting_selections)
        }
        
        return risk_metrics
    
    def _categorize_risk(self, probability, stake):
        """Categorize risk level based on probability and stake"""
        if probability > 0.7 and stake < 100:
            return "Low Risk"
        elif probability > 0.5 and stake < 200:
            return "Medium Risk" 
        elif probability > 0.3:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _calculate_expected_return(self, betting_selections):
        """Calculate expected return of portfolio"""
        total_expected = 0
        for bet in betting_selections:
            probability = bet.get('probability', 0.5)
            odds = bet.get('odds', 2.0)
            stake = bet.get('suggested_stake', 0)
            expected_return = (probability * odds * stake) - stake
            total_expected += expected_return
        return total_expected

class NewsAnalysisAgent:
    """Agent for analyzing team news, injuries, and external factors from BBC Sport"""
    
    def __init__(self):
        self.name = "📰 News Analysis Agent"
        self.base_url = "https://www.bbc.co.uk/sport/football"
        self.rss_url = "https://feeds.bbci.co.uk/sport/football/rss.xml"
        self.news_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        self.impact_keywords = {
            'high_negative': ['injured', 'suspended', 'banned', 'out for season', 'red card', 'ruled out', 'major injury', 'surgery', 'torn', 'broken'],
            'medium_negative': ['doubtful', 'fitness test', 'strain', 'knock', 'minor injury', 'precautionary', 'assessment'], 
            'positive': ['return', 'fit again', 'back in training', 'available', 'recovered', 'cleared', 'ready'],
            'team_changes': ['new signing', 'manager', 'coach', 'formation', 'tactics', 'strategy']
        }
    
    @st.cache_data(ttl=3600)
    def fetch_bbc_football_news(_self):
        """Fetch latest football news from BBC Sport (RSS first, HTML fallback)"""
        # Try RSS feed (most reliable)
        try:
            feed = feedparser.parse(_self.rss_url)
            articles = []
            for entry in feed.entries[:25]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                if title and len(title) > 10:
                    articles.append({
                        'title': title,
                        'summary': BeautifulSoup(summary, 'html.parser').get_text(strip=True),
                        'link': link,
                        'impact_score': _self._calculate_news_impact(title, summary),
                        'teams_mentioned': _self._extract_team_names(title, summary)
                    })
            if articles:
                return articles
        except Exception as e:
            logger.warning(f"RSS fetch failed, falling back to HTML: {e}")

        # HTML fallback if RSS fails or empty
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(_self.base_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            news_articles = []
            articles = soup.find_all(['article', 'div'], class_=['qa-story-cta', 'gel-layout__item', 'gs-c-promo', 'gs-c-promo-body'])
            for article in articles[:20]:
                try:
                    title_elem = article.find(['h3', 'h2', 'h4']) or article.find('a')
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    link_elem = article.find('a')
                    link = link_elem.get('href', '') if link_elem else ''
                    if link and not link.startswith('http'):
                        link = f"https://www.bbc.co.uk{link}"
                    summary_elem = article.find(['p', 'span'], class_=['qa-story-body', 'gs-c-promo-summary'])
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    if title and len(title) > 10:
                        news_articles.append({
                            'title': title,
                            'summary': summary,
                            'link': link,
                            'impact_score': _self._calculate_news_impact(title, summary),
                            'teams_mentioned': _self._extract_team_names(title, summary)
                        })
                except Exception:
                    continue
            return news_articles
        except Exception as e:
            logger.error(f"Error fetching BBC Sport news (HTML fallback): {e}")
            return []
    
    def _calculate_news_impact(self, title, summary):
        """Calculate impact score for news article"""
        text = (title + ' ' + summary).lower()
        impact_score = 0.0
        
        # Check for negative impact keywords
        for keyword in self.impact_keywords['high_negative']:
            if keyword in text:
                impact_score -= 0.8
        
        for keyword in self.impact_keywords['medium_negative']:
            if keyword in text:
                impact_score -= 0.4
        
        # Check for positive impact keywords
        for keyword in self.impact_keywords['positive']:
            if keyword in text:
                impact_score += 0.6
        
        # Normalize score
        return max(-1.0, min(1.0, impact_score))
    
    def _extract_team_names(self, title, summary):
        """Extract team names mentioned in the news (uses official names if available)"""
        text = (title + ' ' + summary).lower()
        teams_found = []
        try:
            official_names = get_official_team_names()
            # Flatten mapping to a set of canonical names and known aliases
            all_names = set()
            for league, names in official_names.items():
                for name in names:
                    all_names.add(name.lower())
        except Exception:
            # Fallback: basic PL names
            all_names = set([
                'arsenal','manchester city','liverpool','chelsea','tottenham','spurs',
                'manchester united','brighton','aston villa','villa','newcastle',
                'fulham','brentford','crystal palace','palace','bournemouth','wolves',
                'everton','nottingham forest','forest','west ham','leicester','leeds',
                'southampton','burnley','sheffield united','luton'
            ])

        for team in all_names:
            if team in text:
                teams_found.append(team.title())
        return teams_found
    
    def analyze_team_news(self, home_team, away_team):
        """Analyze team news impact on match using real BBC Sport news"""
        try:
            news_articles = self.fetch_bbc_football_news()
            
            home_news = []
            away_news = []
            home_impact = 0.0
            away_impact = 0.0
            
            # Filter news relevant to the teams
            for article in news_articles:
                teams_mentioned = article['teams_mentioned']
                
                # Check if home team is mentioned
                for team in teams_mentioned:
                    if self._team_name_match(team, home_team):
                        home_news.append(article)
                        home_impact += article['impact_score']
                        break
                
                # Check if away team is mentioned  
                for team in teams_mentioned:
                    if self._team_name_match(team, away_team):
                        away_news.append(article)
                        away_impact += article['impact_score']
                        break
            
            return {
                'home_team': {
                    'impact_score': max(-1.0, min(1.0, home_impact)),
                    'key_news': home_news[:3],  # Top 3 relevant articles
                    'confidence': 0.8 if home_news else 0.2
                },
                'away_team': {
                    'impact_score': max(-1.0, min(1.0, away_impact)), 
                    'key_news': away_news[:3],
                    'confidence': 0.8 if away_news else 0.2
                },
                'total_articles_analyzed': len(news_articles)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing team news: {e}")
            return {
                'home_team': {'impact_score': 0.0, 'key_news': [], 'confidence': 0.1},
                'away_team': {'impact_score': 0.0, 'key_news': [], 'confidence': 0.1},
                'total_articles_analyzed': 0
            }
    
    def _team_name_match(self, news_team, match_team):
        """Check if team names match (fuzzy matching)"""
        news_team_lower = news_team.lower()
        match_team_lower = match_team.lower()
        
        # Direct match
        if news_team_lower == match_team_lower:
            return True
        
        # Check if one contains the other
        if news_team_lower in match_team_lower or match_team_lower in news_team_lower:
            return True
        
        # Common abbreviations
        abbreviations = {
            'man city': 'manchester city',
            'man utd': 'manchester united', 
            'man united': 'manchester united',
            'spurs': 'tottenham',
            'villa': 'aston villa'
        }
        
        for abbrev, full_name in abbreviations.items():
            if (abbrev in news_team_lower and full_name in match_team_lower) or \
               (abbrev in match_team_lower and full_name in news_team_lower):
                return True
        
        return False
    
    def get_injury_report(self, team_name):
        """Get injury report for team from news analysis"""
        news_analysis = self.analyze_team_news(team_name, "dummy")
        team_data = news_analysis.get('home_team', {})
        
        injured_players = []
        suspended_players = []
        returning_players = []
        
        for article in team_data.get('key_news', []):
            text = (article['title'] + ' ' + article['summary']).lower()
            
            # Extract player names and status (simplified)
            if any(word in text for word in ['injured', 'injury', 'strain', 'knock']):
                injured_players.append('Player mentioned in news')
            elif any(word in text for word in ['suspended', 'banned', 'red card']):
                suspended_players.append('Player mentioned in news')
            elif any(word in text for word in ['return', 'fit again', 'recovered']):
                returning_players.append('Player mentioned in news')
        
        return {
            'injured_players': list(set(injured_players)),
            'suspended_players': list(set(suspended_players)),
            'doubtful_players': [],
            'returning_players': list(set(returning_players)),
            'impact_rating': team_data.get('impact_score', 0.0),
            'confidence': team_data.get('confidence', 0.5)
        }

class QueryAgent:
    """Agent for handling natural language queries"""
    
    def __init__(self, prediction_agent: PredictionAgent, analysis_agent: AnalysisAgent):
        self.name = "🤖 Query Agent"
        self.prediction_agent = prediction_agent
        self.analysis_agent = analysis_agent
        
    def process_query(self, query: str) -> str:
        """Process natural language queries"""
        query_lower = query.lower()
        
        try:
            if "likely outcome" in query_lower and "all matches" in query_lower:
                return self._get_all_outcomes()
            elif "top teams" in query_lower or "best form" in query_lower:
                return self._get_top_teams()
            elif "worst teams" in query_lower or "poor form" in query_lower:
                return self._get_bottom_teams()
            elif "predictions" in query_lower:
                return self._get_prediction_summary()
            elif "analysis" in query_lower:
                return self._get_analysis_summary()
            else:
                return self._handle_general_query(query)
        except Exception as e:
            return f"Sorry, I encountered an error processing your query: {str(e)}"
    
    def _get_all_outcomes(self) -> str:
        """Get likely outcomes for all matches"""
        predictions = self.prediction_agent.get_all_predictions()
        
        if not predictions:
            return "No fixture predictions available at the moment."
        
        result = "🔮 **Likely Outcomes for All Matches:**\n\n"
        
        for fixture_key, pred in predictions.items():
            if pred.home_score > pred.away_score:
                outcome = f"🏠 {pred.home_team} to win"
            elif pred.away_score > pred.home_score:
                outcome = f"🛣️ {pred.away_team} to win"
            else:
                outcome = "⚖️ Draw predicted"
                
            result += f"• **{pred.home_team} vs {pred.away_team}**: {outcome} ({pred.home_score}-{pred.away_score}) - Confidence: {pred.confidence:.1%}\n"
            
        return result
    
    def _get_top_teams(self) -> str:
        """Get top performing teams"""
        analysis = self.analysis_agent.get_league_analysis()
        
        result = "🏆 **Top Form Teams:**\n\n"
        for team, form in analysis.get("top_form_teams", []):
            result += f"• **{team}**: {form:.2f} points per game\n"
            
        return result
    
    def _get_bottom_teams(self) -> str:
        """Get bottom performing teams"""
        analysis = self.analysis_agent.get_league_analysis()
        
        result = "📉 **Teams Struggling for Form:**\n\n"
        for team, form in analysis.get("bottom_form_teams", []):
            result += f"• **{team}**: {form:.2f} points per game\n"
            
        return result
    
    def _get_prediction_summary(self) -> str:
        """Get prediction summary"""
        predictions = self.prediction_agent.get_all_predictions()
        
        if not predictions:
            return "No predictions available."
        
        total_matches = len(predictions)
        home_wins = sum(1 for p in predictions.values() if p.home_score > p.away_score)
        away_wins = sum(1 for p in predictions.values() if p.away_score > p.home_score)
        draws = total_matches - home_wins - away_wins
        
        result = "📈 **Prediction Summary:**\n\n"
        result += f"• Total Matches: {total_matches}\n"
        result += f"• Home Wins: {home_wins} ({home_wins/total_matches:.1%})\n"
        result += f"• Away Wins: {away_wins} ({away_wins/total_matches:.1%})\n"
        result += f"• Draws: {draws} ({draws/total_matches:.1%})\n"
        
        return result
    
    def _get_analysis_summary(self) -> str:
        """Get analysis summary"""
        analysis = self.analysis_agent.get_league_analysis()
        
        result = "📊 **League Analysis Summary:**\n\n"
        result += f"**Top Form Team:** {analysis.get('top_form_teams', [('N/A', 0)])[0][0]}\n"
        result += f"**Average Goals per Game:** {analysis.get('average_goals', 0):.2f}\n"
        
        overperformers = analysis.get('overperformers', [])
        if overperformers:
            result += f"**Overperformers:** {', '.join([team for team, _ in overperformers[:3]])}\n"
            
        return result
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        return f"I understand you're asking about: '{query}'. Try asking about:\n" + \
               "• 'Show me the likely outcome for all matches'\n" + \
               "• 'Which teams have the best form?'\n" + \
               "• 'Give me a prediction summary'\n" + \
               "• 'Show me the league analysis'"

def main():
    st.set_page_config(
        page_title="⚽ Football Prediction Agents",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚽ Multi-Agent Football Prediction System")
    st.markdown("*Intelligent agents for match predictions, analysis, and insights*")
    
    # Initialize agents
    prediction_agent = PredictionAgent()
    analysis_agent = AnalysisAgent()
    betting_agent = BettingAgent()
    market_agent = MarketIntelligenceAgent()
    risk_agent = RiskManagementAgent()
    news_agent = NewsAnalysisAgent()
    
    # Sidebar
    st.sidebar.title("🤖 Agent Dashboard")
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔮 Predictions", 
        "📊 Enhanced Fixtures & Predictions", 
        "📈 Analysis", 
        "💰 Betting Suggestions",
        "🏪 Market Intelligence",
        "⚠️ Risk Management",
        "📰 News & Impact"
    ])
    
    with tab1:
        st.header("🔮 Match Predictions & Insights")
        
        with st.spinner("Getting predictions..."):
            predictions = prediction_agent.get_all_predictions()
        
        if predictions:
            for fixture_key, pred in predictions.items():
                with st.expander(f"⚽ {pred.home_team} vs {pred.away_team}", expanded=True):
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.metric("🏠 Home Team", pred.home_team)
                        st.metric("Predicted Score", pred.home_score)
                        if pred.home_win_prob:
                            st.metric("Win Probability", f"{pred.home_win_prob:.1f}%")
                    
                    with col2:
                        st.markdown("<h3 style='text-align: center'>VS</h3>", unsafe_allow_html=True)
                        st.metric("Confidence", f"{pred.confidence:.1%}")
                    
                    with col3:
                        st.metric("🛣️ Away Team", pred.away_team)
                        st.metric("Predicted Score", pred.away_score)
                        if pred.away_win_prob:
                            st.metric("Win Probability", f"{pred.away_win_prob:.1f}%")
                    
                    if pred.insights:
                        st.markdown("**🧠 AI Insights:**")
                        for insight in pred.insights:
                            st.markdown(f"• {insight}")
        else:
            st.warning("No predictions available at the moment.")
    
    with tab2:
        st.header("📊 Enhanced Fixtures & Predictions")
        
        try:
            with st.spinner("Loading enhanced fixtures with predictions..."):
                fixtures_df = get_todays_fixtures()
            
            if not fixtures_df.empty:
                # Show fixtures by country with enhanced info
                st.subheader(f"🌍 {len(fixtures_df)} Fixtures from {fixtures_df['Country'].nunique()} Countries/Leagues")
                
                # Display country breakdown
                country_counts = fixtures_df['Country'].value_counts()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Leagues:** {', '.join([f'{country} ({count})' for country, count in country_counts.head(8).items()])}...")
                
                with col2:
                    # Time range display
                    if 'Time' in fixtures_df.columns:
                        st.metric("⏰ Time Range", f"{fixtures_df['Time'].iloc[0]} → {fixtures_df['Time'].iloc[-1]}")
                
                # Add filtering options
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    selected_countries = st.multiselect(
                        "Filter by League:", 
                        options=fixtures_df['Country'].unique(),
                        default=fixtures_df['Country'].unique()[:5]  # Show first 5 by default
                    )
                
                with col2:
                    prediction_mode = st.selectbox(
                        "Prediction Detail:", 
                        ["Basic View", "Enhanced Predictions", "Full Analysis"],
                        index=1
                    )
                
                with col3:
                    max_fixtures = st.selectbox(
                        "Show fixtures:",
                        [20, 50, 100, "All"],
                        index=0
                    )
                
                with col4:
                    fixtures_per_timeslot = st.selectbox(
                        "Per time slot:",
                        [5, 10, 15, 25, "All"],
                        index=2,  # Default to 15
                        help="How many fixtures to show per time slot in Enhanced Predictions"
                    )
                
                # Filter data
                if selected_countries:
                    filtered_df = fixtures_df[fixtures_df['Country'].isin(selected_countries)]
                else:
                    filtered_df = fixtures_df
                
                # Limit display
                if max_fixtures != "All":
                    filtered_df = filtered_df.head(max_fixtures)
                
                if prediction_mode == "Basic View":
                    # Simple table view
                    display_columns = ['Country', 'Time', 'Home', 'Away', 'Home Win', 'Draw', 'Away Win']
                    available_columns = [col for col in display_columns if col in filtered_df.columns]
                    st.dataframe(filtered_df[available_columns], width='stretch')
                
                elif prediction_mode == "Enhanced Predictions":
                    # Show fixtures with enhanced predictions
                    st.write(f"📈 **Enhanced Predictions for {len(filtered_df)} fixtures:**")
                    
                    # Group by time for better organization
                    if 'Time' in filtered_df.columns:
                        for time_slot, time_group in filtered_df.groupby('Time'):
                            if len(time_group) > 0:
                                st.markdown(f"### 🕐 {time_slot} Kickoffs ({len(time_group)} matches)")
                                
                                # Use the user-selected limit per time slot
                                if fixtures_per_timeslot == "All":
                                    display_group = time_group
                                else:
                                    display_group = time_group.head(fixtures_per_timeslot)
                                
                                for idx, row in display_group.iterrows():
                                    home_team = row['Home']
                                    away_team = row['Away']
                                    country = row.get('Country', 'Unknown')
                                    
                                    with st.expander(f"⚽ {home_team} vs {away_team} ({country})", expanded=False):
                                        # Get enhanced prediction
                                        try:
                                            prediction = predict_match_score(home_team, away_team, fixtures_df)
                                            
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric("🏠 " + home_team[:15], f"Score: {prediction['home_goals']}")
                                                st.progress(prediction['home_win_prob'])
                                                st.caption(f"Win: {prediction['home_win_prob']:.1%}")
                                            
                                            with col2:
                                                st.metric("🎯 Prediction", f"{prediction['home_goals']}-{prediction['away_goals']}")
                                                st.metric("📊 Confidence", f"{prediction['confidence']:.1%}")
                                                st.metric("⚖️ Draw", f"{prediction['draw_prob']:.1%}")
                                            
                                            with col3:
                                                st.metric("🛣️ " + away_team[:15], f"Score: {prediction['away_goals']}")
                                                st.progress(prediction['away_win_prob'])
                                                st.caption(f"Win: {prediction['away_win_prob']:.1%}")
                                            
                                            if prediction['reasoning']:
                                                st.markdown("**🧠 AI Reasoning:**")
                                                for reason in prediction['reasoning'][:3]:
                                                    st.caption(f"• {reason}")
                                                    
                                        except Exception as e:
                                            st.error(f"Error generating prediction: {e}")
                    else:
                        # Fallback if no time column
                        for idx, row in filtered_df.head(10).iterrows():
                            home_team = row['Home']
                            away_team = row['Away']
                            
                            with st.expander(f"⚽ {home_team} vs {away_team}", expanded=False):
                                prediction = predict_match_score(home_team, away_team, fixtures_df)
                                st.write(f"**Prediction:** {prediction['home_goals']}-{prediction['away_goals']} (Confidence: {prediction['confidence']:.1%})")
                
                else:  # Full Analysis
                    # Detailed analysis view
                    st.write("🔍 **Full Analysis Mode** - Detailed predictions with statistics")
                    
                    analysis_sample = filtered_df.head(10)  # Increased from 5 to 10 for more analysis
                    for idx, row in analysis_sample.iterrows():
                        home_team = row['Home']
                        away_team = row['Away']
                        
                        with st.expander(f"📈 FULL ANALYSIS: {home_team} vs {away_team}", expanded=True):
                            prediction = predict_match_score(home_team, away_team, fixtures_df)
                            
                            # Main prediction display
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 Predicted Score", f"{prediction['home_goals']}-{prediction['away_goals']}")
                            with col2:
                                st.metric("🏠 Home Win", f"{prediction['home_win_prob']:.1%}")
                            with col3:
                                st.metric("⚖️ Draw", f"{prediction['draw_prob']:.1%}")
                            with col4:
                                st.metric("🛣️ Away Win", f"{prediction['away_win_prob']:.1%}")
                            
                            # Detailed breakdown
                            st.markdown("**📊 Analysis Breakdown:**")
                            for reason in prediction['reasoning']:
                                st.write(f"• {reason}")
                            
                            # Basic stats if available
                            if 'Home Win' in row and row['Home Win']:
                                st.markdown(f"**📈 SoccerStats Data:** Home {row['Home Win']:.1%} | Draw {row.get('Draw', 0):.1%} | Away {row.get('Away Win', 0):.1%}")
                
                # Summary visualization
                if len(filtered_df) > 1:
                    st.markdown("---")
                    st.subheader("📊 League Summary")
                    
                    fig = px.bar(
                        filtered_df.head(15), 
                        x=['Home Win', 'Draw', 'Away Win'], 
                        y='Home',
                        title=f"Match Probabilities - Top {min(15, len(filtered_df))} Fixtures",
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("No fixtures data available.")
                
        except Exception as e:
            st.error(f"Error loading enhanced fixtures: {e}")
            st.exception(e)
    
    with tab3:
        st.header("⚽ Team Analysis & League Stats")
        
        with st.spinner("Analyzing league data..."):
            team_insights = analysis_agent.get_team_insights()
            league_analysis = analysis_agent.get_league_analysis()
        
        if league_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Form Teams")
                for team, form in league_analysis.get("top_form_teams", [])[:5]:
                    st.metric(team, f"{form:.2f} PPG")
            
            with col2:
                st.subheader("📉 Bottom Form Teams")
                for team, form in league_analysis.get("bottom_form_teams", [])[:5]:
                    st.metric(team, f"{form:.2f} PPG")
            
            # Team insights
            if team_insights:
                st.subheader("🔍 Team Insights")
                selected_team = st.selectbox("Select a team for detailed analysis:", 
                                           list(team_insights.keys()))
                
                if selected_team:
                    insight = team_insights[selected_team]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Form", f"{insight.form:.2f}")
                    with col2:
                        st.metric("Performance", f"{insight.recent_performance:.1%}")
                    with col3:
                        st.metric("Key Players", len(insight.key_players))
                    
                    if insight.strengths:
                        st.markdown("**💪 Strengths:**")
                        for strength in insight.strengths:
                            st.markdown(f"• {strength}")
                    
                    if insight.weaknesses:
                        st.markdown("**⚠️ Areas to Improve:**")
                        for weakness in insight.weaknesses:
                            st.markdown(f"• {weakness}")
        else:
            st.info("💡 Team analysis will be available once league data is processed.")
            
            # Show some basic team stats from fixtures
            try:
                fixtures_df = get_todays_fixtures()
                if not fixtures_df.empty:
                    st.subheader("📈 Quick Team Stats from Today's Fixtures")
                    
                    # Get unique teams
                    home_teams = fixtures_df['Home'].unique()
                    away_teams = fixtures_df['Away'].unique()
                    all_teams = list(set(list(home_teams) + list(away_teams)))
                    
                    selected_teams = st.multiselect(
                        "Select teams to compare:",
                        options=all_teams[:20],  # Limit to first 20 for performance
                        default=all_teams[:3] if len(all_teams) >= 3 else all_teams
                    )
                    
                    if selected_teams:
                        # Simple comparison
                        for team in selected_teams:
                            home_matches = fixtures_df[fixtures_df['Home'] == team]
                            away_matches = fixtures_df[fixtures_df['Away'] == team]
                            
                            with st.expander(f"📊 {team} Analysis", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Home Games Today", len(home_matches))
                                    if len(home_matches) > 0:
                                        avg_home_win = home_matches['Home Win'].mean()
                                        st.metric("Avg Home Win Prob", f"{avg_home_win:.1%}")
                                
                                with col2:
                                    st.metric("Away Games Today", len(away_matches))
                                    if len(away_matches) > 0:
                                        avg_away_win = away_matches['Away Win'].mean()
                                        st.metric("Avg Away Win Prob", f"{avg_away_win:.1%}")
                                
                                with col3:
                                    total_games = len(home_matches) + len(away_matches)
                                    st.metric("Total Games Today", total_games)
                                    
                                    if total_games > 0:
                                        # Calculate overall win probability
                                        total_win_prob = (home_matches['Home Win'].sum() + away_matches['Away Win'].sum()) / total_games
                                        st.metric("Overall Win Prob", f"{total_win_prob:.1%}")
                                
                                # Show the matches
                                if total_games > 0:
                                    st.markdown("**📅 Today's Fixtures:**")
                                    for _, match in home_matches.iterrows():
                                        st.write(f"🏠 **{match['Home']}** vs {match['Away']} ({match.get('Time', 'TBD')})")
                                    for _, match in away_matches.iterrows():
                                        st.write(f"🛣️ {match['Home']} vs **{match['Away']}** ({match.get('Time', 'TBD')})")
            except Exception as e:
                st.warning(f"Could not load basic team stats: {e}")
    
    with tab4:
        st.header("💰 Betting Agent Suggestions")
        st.markdown("🎯 **AI-Powered Betting Analysis** - Smart suggestions based on prediction confidence")
        
        try:
            with st.spinner("🔍 Analyzing fixtures for betting opportunities..."):
                fixtures_df = get_todays_fixtures()
                betting_report = betting_agent.generate_betting_report(fixtures_df)
            
            if betting_report['total_matches_analyzed'] > 0:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Matches Analyzed", betting_report['total_matches_analyzed'])
                
                with col2:
                    st.metric("🎯 Single Bet Tips", len(betting_report['single_bets']))
                
                with col3:
                    acc_count = len(betting_report['accumulator_variants']) if betting_report['accumulator_variants'] else 0
                    st.metric("📈 Accumulator Bets", acc_count)
                
                with col4:
                    if betting_report['single_bets']:
                        avg_confidence = sum(bet['confidence'] for bet in betting_report['single_bets']) / len(betting_report['single_bets'])
                        st.metric("🎪 Avg Confidence", f"{avg_confidence:.1%}")
                
                # Betting suggestions tabs
                bet_tab1, bet_tab2 = st.tabs(["🎯 Single Bets", "📈 Accumulator Bets"])
                
                with bet_tab1:
                    st.subheader("🎯 Recommended Single Bets")
                    
                    if betting_report['single_bets']:
                        # Filter options
                        bet_types = list(set([bet['type'] for bet in betting_report['single_bets']]))
                        selected_types = st.multiselect(
                            "Filter by bet type:",
                            options=bet_types,
                            default=bet_types
                        )
                        
                        min_confidence = st.slider(
                            "Minimum confidence level:",
                            min_value=0.5,
                            max_value=1.0,
                            value=0.6,
                            step=0.05
                        )
                        
                        # Filter and display bets
                        filtered_bets = [
                            bet for bet in betting_report['single_bets']
                            if bet['type'] in selected_types and bet['confidence'] >= min_confidence
                        ]
                        
                        if filtered_bets:
                            for i, bet in enumerate(filtered_bets[:15], 1):
                                with st.expander(f"💰 #{i}: {bet['bet']} (Est. odds: {bet['estimated_odds']:.2f})", expanded=False):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Bet Type", bet['type'])
                                        conf_text = f"{bet['confidence']:.1f}%"
                                        st.metric("Confidence", conf_text)
                                    
                                    with col2:
                                        st.metric("Estimated Odds", f"{bet['estimated_odds']:.2f}")
                                        roi = (bet['estimated_odds'] * bet['confidence']) - 1
                                        roi_text = f"{roi:.1f}%"
                                        st.metric("Expected ROI", roi_text)
                                    
                                    with col3:
                                        # Risk assessment
                                        if bet['confidence'] >= 0.75:
                                            risk_level = "🟢 Low Risk"
                                        elif bet['confidence'] >= 0.65:
                                            risk_level = "🟡 Medium Risk"
                                        else:
                                            risk_level = "🔴 Higher Risk"
                                        
                                        st.metric("Risk Level", risk_level)
                                    
                                    # Stake suggestion
                                    if bet['confidence'] >= 0.8:
                                        stake_suggestion = "💪 Strong confidence - consider higher stake"
                                    elif bet['confidence'] >= 0.7:
                                        stake_suggestion = "👍 Good confidence - moderate stake"
                                    else:
                                        stake_suggestion = "⚠️ Lower confidence - small stake only"
                                    
                                    reasoning_text = str(bet['reasoning']).replace('%', 'percent')
                                    st.markdown(f"**💡 Reasoning:** {reasoning_text}")
                                    st.caption(stake_suggestion)
                        else:
                            st.info("No bets match your current filters. Try adjusting the confidence level or bet types.")
                    else:
                        st.warning("No single betting opportunities found with current confidence thresholds.")
                
                with bet_tab2:
                    st.subheader("📈 Accumulator Bet Suggestions")
                    st.markdown("💡 **Risk Range:** From ultra-safe to maximum risk - choose your comfort level!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if betting_report['accumulator_variants']:
                            total_variants = len(betting_report['accumulator_variants'])
                            st.metric("🎯 Available Strategies", total_variants)
                    
                    with col2:
                        # Show success rate range
                        if betting_report['accumulator_variants']:
                            success_rates = [variant[1]['combined_probability'] for variant in betting_report['accumulator_variants']]
                            min_rate = min(success_rates)
                            max_rate = max(success_rates)
                            st.metric("📊 Success Rate Range", f"{min_rate:.1f}% - {max_rate:.1f}%")
                    
                    # Display accumulator variants from safest to riskiest
                    if betting_report['accumulator_variants']:
                        st.markdown("**🎪 Accumulator Strategies (Ordered by Success Rate):**")
                        
                        for variant_name, variant_data, emoji in betting_report['accumulator_variants']:
                            if variant_data and 'selections' in variant_data:
                                success_rate = variant_data['combined_probability']
                                risk_level = variant_data['risk_level']
                                strategy = variant_data['strategy']
                                
                                with st.expander(f"{emoji} {variant_name} - Odds: {variant_data['total_estimated_odds']:.2f} | Success: {success_rate:.2f}%", expanded=False):
                                    # Strategy info
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**🎯 Strategy:** {risk_level} ({strategy.replace('_', ' ').title()})")
                                        st.markdown("**📋 Selections:**")
                                        
                                        for j, selection in enumerate(variant_data['selections'], 1):
                                            confidence_emoji = "🟢" if selection['probability'] >= 0.8 else "🟡" if selection['probability'] >= 0.6 else "🔴"
                                            prob_text = f"{selection['probability']:.1%}"
                                            bet_type_emoji = "🛡️" if selection.get('bet_type') == 'Double Chance' else "🎯"
                                            st.write(f"{j}. {confidence_emoji}{bet_type_emoji} **{selection['match']}** - {selection['selection']} ({prob_text})")
                                    
                                    with col2:
                                        st.metric("Total Odds", f"{variant_data['total_estimated_odds']:.2f}")
                                        st.metric("Success Rate", f"{success_rate:.2f}%")
                                        st.metric("Selections", variant_data['fold_count'])
                                        
                                        # Risk indicator
                                        if success_rate >= 25:
                                            st.success("🟢 Good Odds")
                                        elif success_rate >= 15:
                                            st.warning("🟡 Moderate Risk")
                                        elif success_rate >= 5:
                                            st.warning("🔴 High Risk")
                                        else:
                                            st.error("💀 Extreme Risk")
                                    
                                    # Potential returns
                                    st.markdown("**💰 Potential Returns:**")
                                    stakes = [1, 5, 10, 20, 50]
                                    returns_data = []
                                    for stake in stakes:
                                        return_amount = stake * variant_data['total_estimated_odds']
                                        returns_data.append(f"£{stake} → £{return_amount:.0f}")
                                    returns_text = " | ".join(returns_data)
                                    st.caption(returns_text)
                                    
                                    # Strategy advice
                                    if success_rate >= 20:
                                        st.success("💡 **Recommended:** Good balance of risk and reward")
                                    elif success_rate >= 10:
                                        st.info("💡 **Caution:** Moderate risk - consider smaller stakes")
                                    else:
                                        st.warning("💡 **High Risk:** Only bet what you can afford to lose completely")
                    
                    else:
                        st.info("❌ No accumulator variants available - not enough suitable predictions found")
                    
                    # Overall accumulator tips
                    st.markdown("---")
                    st.markdown("**🎓 Accumulator Strategy Tips:**")
                    st.markdown("""
                    - 🛡️ **Ultra Safe:** Maximum safety with double chance focus
                    - 🟢 **Conservative:** Good balance with mixed strategies  
                    - ⚖️ **Balanced:** Standard risk level for regular bettors
                    - 🟡 **Aggressive:** Higher risk for experienced bettors
                    - 🔴 **High Risk:** Significant risk, substantial rewards
                    - 💀 **Maximum Risk:** Extreme speculation only
                    - 💰 **Golden Rule:** Never bet more than you can afford to lose
                    """)
            
            else:
                st.warning("No fixtures available for betting analysis.")
                
        except Exception as e:
            st.error(f"Error generating betting suggestions: {e}")
            st.exception(e)
    
    with tab5:
        st.header("🏪 Market Intelligence")
        st.markdown("*Find value bets by comparing our predictions with market odds*")
        
        try:
            with st.spinner("Analyzing market opportunities..."):
                fixtures_df = get_todays_fixtures()
                
            if not fixtures_df.empty:
                # Get sample predictions for market analysis
                all_predictions = []
                sample_size = min(20, len(fixtures_df))
                
                for idx in range(sample_size):
                    row = fixtures_df.iloc[idx]
                    try:
                        prediction = predict_match_score(row['Home'], row['Away'], fixtures_df)
                        all_predictions.append({
                            'home_team': row['Home'],
                            'away_team': row['Away'], 
                            'prediction': prediction
                        })
                    except Exception:
                        continue
                
                if all_predictions:
                    value_bets = market_agent.get_market_value_bets(all_predictions)
                    
                    st.subheader("💎 Value Betting Opportunities")
                    
                    if value_bets:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Value Bets Found", len(value_bets))
                        with col2:
                            avg_value = sum(bet['value_percentage'] for bet in value_bets) / len(value_bets)
                            st.metric("📊 Avg Value Edge", f"{avg_value:.1f}%")
                        
                        st.markdown("**🔍 Top Value Opportunities:**")
                        
                        for i, bet in enumerate(value_bets[:10], 1):
                            with st.expander(f"💰 #{i}: {bet['match']} - {bet['outcome']} ({bet['value_percentage']:.1f}% edge)", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Market Odds", f"{bet['market_odds']:.2f}")
                                    st.metric("Our Probability", f"{bet['our_probability']:.1%}")
                                
                                with col2:
                                    st.metric("Market Probability", f"{bet['market_probability']:.1%}")
                                    st.metric("Value Edge", f"{bet['value_percentage']:.1f}%")
                                
                                with col3:
                                    confidence_color = "🟢" if bet['confidence'] == 'High' else "🟡"
                                    st.metric("Confidence", f"{confidence_color} {bet['confidence']}")
                                    
                                    if bet['value_percentage'] > 30:
                                        st.success("🚀 Excellent Value")
                                    elif bet['value_percentage'] > 20:
                                        st.info("💎 Good Value") 
                                    else:
                                        st.warning("⚡ Moderate Value")
                    else:
                        st.info("📊 No significant value opportunities found in current market")
                else:
                    st.warning("Unable to load prediction data for market analysis")
            else:
                st.warning("No fixtures available for market analysis")
                
        except Exception as e:
            st.error(f"Error in market intelligence: {e}")
    
    with tab6:
        st.header("⚠️ Risk Management")
        st.markdown("*Bankroll management and portfolio risk assessment*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Bankroll Calculator")
            
            bankroll = st.number_input("Your Bankroll (£)", min_value=10.0, value=100.0, step=10.0)
            probability = st.slider("Bet Probability", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
            odds = st.number_input("Betting Odds", min_value=1.1, value=2.0, step=0.1)
            
            if st.button("Calculate Optimal Stake"):
                optimal_stake = risk_agent.calculate_kelly_stake(probability, odds, bankroll)
                
                st.metric("📊 Kelly Criterion Stake", f"£{optimal_stake:.2f}")
                st.metric("📈 Stake Percentage", f"{(optimal_stake/bankroll)*100:.1f}% of bankroll")
                
                if optimal_stake > 0:
                    potential_profit = (odds - 1) * optimal_stake
                    potential_loss = optimal_stake
                    
                    st.metric("💚 Potential Profit", f"£{potential_profit:.2f}")
                    st.metric("🔴 Potential Loss", f"£{potential_loss:.2f}")
                    
                    # Risk assessment
                    if optimal_stake < bankroll * 0.02:
                        st.success("✅ Low risk bet - suitable for conservative betting")
                    elif optimal_stake < bankroll * 0.05:
                        st.info("ℹ️ Moderate risk - standard bet sizing")
                    else:
                        st.warning("⚠️ Higher risk - consider reducing stake")
                else:
                    st.error("❌ No betting value detected - avoid this bet")
        
        with col2:
            st.subheader("📊 Risk Guidelines")
            
            st.markdown("""
            **🎯 Kelly Criterion Guidelines:**
            - **0-2%**: Very safe, conservative betting
            - **2-5%**: Standard risk level for most bets  
            - **5-10%**: Higher risk, only for high confidence bets
            - **10%+**: Maximum recommended (system caps at 10%)
            
            **💡 Bankroll Management Tips:**
            - Never bet more than 10% on any single bet
            - Keep 50-70% of bankroll in reserve
            - Track all bets and calculate ROI regularly
            - Set stop-loss limits (e.g., -20% of bankroll)
            - Take profits when you're up 50-100%
            
            **🚨 Risk Warnings:**
            - Past performance doesn't guarantee future results
            - Sports betting involves significant risk
            - Only bet money you can afford to lose completely
            - Consider gambling addiction resources if needed
            """)
            
            st.markdown("---")
            st.info("💡 **Pro Tip:** The Kelly Criterion maximizes long-term growth but can be aggressive. Many professionals use 25-50% of the Kelly stake for safety.")

    with tab7:
        st.header("📰 News & Impact")
        st.markdown("*Latest BBC Sport football news and impact on teams*")

        try:
            with st.spinner("Fetching BBC Sport football news..."):
                news_articles = news_agent.fetch_bbc_football_news()
                fixtures_df = get_todays_fixtures()

            teams_in_fixtures = []
            if not fixtures_df.empty:
                teams_in_fixtures = sorted(list(set(fixtures_df['Home'].tolist() + fixtures_df['Away'].tolist())))

            st.subheader("🗞️ Latest Headlines")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Articles Fetched", len(news_articles))
            with col2:
                st.metric("Leagues Today", fixtures_df['Country'].nunique() if not fixtures_df.empty else 0)

            selected_teams = st.multiselect(
                "Filter news by teams (optional)", options=teams_in_fixtures, default=[]
            )

            def article_matches_filter(article):
                if not selected_teams:
                    return True
                mentioned = [t.lower() for t in article.get('teams_mentioned', [])]
                for team in selected_teams:
                    if team.lower() in mentioned:
                        return True
                return False

            displayed = 0
            for i, article in enumerate(news_articles, 1):
                if not article_matches_filter(article):
                    continue
                displayed += 1
                impact = article.get('impact_score', 0.0)
                impact_label = "Neutral"
                if impact > 0.4:
                    impact_label = "Positive"
                elif impact < -0.4:
                    impact_label = "Negative"

                with st.expander(f"#{i} {article['title']} ({impact_label} impact)", expanded=False):
                    if article.get('summary'):
                        st.write(article['summary'])
                    if article.get('link'):
                        st.markdown(f"[Read more]({article['link']})")
                    if article.get('teams_mentioned'):
                        st.caption(f"Teams mentioned: {', '.join(article['teams_mentioned'])}")

            if displayed == 0:
                st.info("No matching articles for selected filters.")

        except Exception as e:
            st.error(f"Error fetching or displaying news: {e}")

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 Active Agents:**")
    st.sidebar.markdown(f"• {prediction_agent.name}")
    st.sidebar.markdown(f"• {analysis_agent.name}")
    st.sidebar.markdown(f"• {betting_agent.name}")
    st.sidebar.markdown(f"• {market_agent.name}")
    st.sidebar.markdown(f"• {risk_agent.name}")
    st.sidebar.markdown(f"• {news_agent.name}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Data Sources:**")
    st.sidebar.markdown("• FPL API")
    st.sidebar.markdown("• SoccerStats")
    st.sidebar.markdown("• ClubElo")
    st.sidebar.markdown("• Market Odds (simulated)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 New Features:**")
    st.sidebar.markdown("• 🏪 Market Intelligence")
    st.sidebar.markdown("• ⚠️ Risk Management") 
    st.sidebar.markdown("• 📰 News Impact Analysis")
    st.sidebar.markdown("• 💰 Kelly Criterion Staking")

if __name__ == "__main__":
    main()