"""
Prediction Monitor
Tracks prediction quality, calibration drift, and market-specific accuracy.
Closes the feedback loop between predictions and actual outcomes.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PredictionMonitor:
    """
    Monitors prediction quality in real-time and historically.
    
    Responsibilities:
    - Record every prediction with its outcome once settled
    - Track calibration quality per market
    - Detect when model accuracy is degrading
    - Signal when recalibration or retraining is needed
    - Provide rolling accuracy windows
    """

    def __init__(self, storage_path: str = 'quant_agent/prediction_log.json'):
        self.storage_path = storage_path
        self.predictions: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # 1. Record Predictions & Outcomes
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        fixture_id: str,
        market: str,
        predicted_prob: float,
        odds: float,
        ev: float,
        selection: str,
        stake: float = 0.0,
        metadata: Optional[Dict] = None,
    ):
        """Record a prediction at the time it's made."""
        entry = {
            'fixture_id': str(fixture_id),
            'market': market,
            'predicted_prob': round(predicted_prob, 4),
            'implied_prob': round(1.0 / odds, 4) if odds > 0 else 0,
            'odds': round(odds, 3),
            'ev': round(ev, 4),
            'selection': selection,
            'stake': round(stake, 4),
            'prediction_time': datetime.utcnow().isoformat(),
            'outcome': None,  # filled later
            'settled': False,
            'metadata': metadata or {},
        }
        self.predictions.append(entry)
        self._save()

    def record_outcome(self, fixture_id: str, market: str, won: bool):
        """Record the outcome of a settled prediction."""
        updated = 0
        for pred in self.predictions:
            if (pred['fixture_id'] == str(fixture_id) and
                pred['market'] == market and
                not pred['settled']):
                pred['outcome'] = 1 if won else 0
                pred['settled'] = True
                pred['settlement_time'] = datetime.utcnow().isoformat()
                updated += 1
        
        if updated > 0:
            self._save()
        return updated

    def bulk_record_outcomes(self, settled_bets: List[Dict]):
        """
        Process a list of settled bets from settled_bets.json format.
        Expected keys: fixture_id or event, market, profit (positive=won).
        """
        for bet in settled_bets:
            fixture_id = str(bet.get('fixture_id', bet.get('event', '')))
            market = bet.get('market', '')
            won = bet.get('profit', 0) > 0
            self.record_outcome(fixture_id, market, won)

    # ------------------------------------------------------------------
    # 2. Calibration Analysis
    # ------------------------------------------------------------------

    def calibration_report(self, n_bins: int = 10) -> Dict:
        """
        Analyze calibration quality across all markets and per-market.
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn required'}

        settled = [p for p in self.predictions if p['settled']]
        if len(settled) < 20:
            return {'status': 'insufficient_data', 'n_settled': len(settled)}

        # Overall calibration
        probs = np.array([p['predicted_prob'] for p in settled])
        outcomes = np.array([p['outcome'] for p in settled])
        
        overall = self._compute_calibration(probs, outcomes, n_bins, 'overall')
        
        # Per-market calibration
        market_reports = {}
        markets = set(self._normalize_market(p['market']) for p in settled)
        
        for market in markets:
            market_preds = [p for p in settled if self._normalize_market(p['market']) == market]
            if len(market_preds) < 10:
                continue
            
            m_probs = np.array([p['predicted_prob'] for p in market_preds])
            m_outcomes = np.array([p['outcome'] for p in market_preds])
            market_reports[market] = self._compute_calibration(m_probs, m_outcomes, n_bins, market)
        
        return {
            'overall': overall,
            'per_market': market_reports,
            'n_settled': len(settled),
            'n_unsettled': len(self.predictions) - len(settled),
        }

    def _compute_calibration(self, probs, outcomes, n_bins, label) -> Dict:
        """Compute calibration metrics for a set of predictions."""
        try:
            brier = brier_score_loss(outcomes, probs)
        except Exception:
            brier = None
        
        try:
            ll = log_loss(outcomes, probs)
        except Exception:
            ll = None
        
        try:
            auc = roc_auc_score(outcomes, probs)
        except Exception:
            auc = None
        
        # Calibration curve
        try:
            fraction_pos, mean_predicted = calibration_curve(
                outcomes, probs, n_bins=min(n_bins, len(probs) // 5), strategy='uniform'
            )
            cal_error = np.mean(np.abs(fraction_pos - mean_predicted))
            calibration = {
                'mean_calibration_error': round(float(cal_error), 4),
                'bins': [
                    {'predicted': round(float(mp), 3), 'actual': round(float(fp), 3)}
                    for mp, fp in zip(mean_predicted, fraction_pos)
                ],
            }
        except Exception:
            calibration = {'mean_calibration_error': None, 'bins': []}
        
        win_rate = float(np.mean(outcomes))
        avg_prob = float(np.mean(probs))
        
        # Bias: positive = overconfident, negative = underconfident
        bias = avg_prob - win_rate
        
        return {
            'label': label,
            'n': len(probs),
            'win_rate': round(win_rate, 4),
            'avg_predicted_prob': round(avg_prob, 4),
            'bias': round(bias, 4),
            'brier_score': round(brier, 5) if brier is not None else None,
            'log_loss': round(ll, 5) if ll is not None else None,
            'auc_roc': round(auc, 4) if auc is not None else None,
            'calibration': calibration,
            'verdict': self._calibration_verdict(bias, brier, auc),
        }

    def _calibration_verdict(self, bias, brier, auc) -> str:
        """Human-readable calibration verdict."""
        issues = []
        if abs(bias) > 0.10:
            direction = 'overconfident' if bias > 0 else 'underconfident'
            issues.append(f'significantly {direction} (bias={bias:.3f})')
        elif abs(bias) > 0.05:
            direction = 'overconfident' if bias > 0 else 'underconfident'
            issues.append(f'slightly {direction} (bias={bias:.3f})')
        
        if brier is not None and brier > 0.25:
            issues.append(f'poor Brier score ({brier:.4f})')
        
        if auc is not None and auc < 0.55:
            issues.append(f'weak discrimination (AUC={auc:.3f})')
        
        if not issues:
            return 'WELL_CALIBRATED'
        return 'NEEDS_ATTENTION: ' + '; '.join(issues)

    # ------------------------------------------------------------------
    # 3. Accuracy Degradation Detection
    # ------------------------------------------------------------------

    def detect_degradation(self, window_size: int = 50, threshold: float = 0.05) -> Dict:
        """
        Compare recent accuracy to historical baseline.
        Uses a rolling window to detect performance drop-offs.
        """
        settled = [p for p in self.predictions if p['settled']]
        if len(settled) < window_size * 2:
            return {'status': 'insufficient_data', 'n_settled': len(settled)}
        
        outcomes = [p['outcome'] for p in settled]
        probs = [p['predicted_prob'] for p in settled]
        
        # Full history baseline
        baseline_wr = np.mean(outcomes)
        baseline_bias = np.mean(probs) - baseline_wr
        
        # Recent window
        recent_outcomes = outcomes[-window_size:]
        recent_probs = probs[-window_size:]
        recent_wr = np.mean(recent_outcomes)
        recent_bias = np.mean(recent_probs) - recent_wr
        
        # Older window (just before recent)
        older_outcomes = outcomes[-2*window_size:-window_size]
        older_wr = np.mean(older_outcomes)
        
        wr_change = recent_wr - older_wr
        bias_change = recent_bias - baseline_bias
        
        degraded = (wr_change < -threshold) or (abs(recent_bias) > 0.10)
        
        # Rolling accuracy over time
        rolling = []
        for i in range(0, len(outcomes) - window_size + 1, max(1, window_size // 5)):
            w = outcomes[i:i + window_size]
            rolling.append(round(np.mean(w), 3))
        
        return {
            'status': 'DEGRADED' if degraded else 'STABLE',
            'baseline_win_rate': round(baseline_wr, 4),
            'recent_win_rate': round(recent_wr, 4),
            'older_win_rate': round(older_wr, 4),
            'win_rate_change': round(wr_change, 4),
            'baseline_bias': round(baseline_bias, 4),
            'recent_bias': round(recent_bias, 4),
            'bias_change': round(bias_change, 4),
            'rolling_accuracy': rolling,
            'recommendation': self._degradation_recommendation(degraded, wr_change, recent_bias),
        }

    def _degradation_recommendation(self, degraded, wr_change, bias) -> str:
        if not degraded:
            return 'No action needed. Model performance is stable.'
        
        actions = []
        if wr_change < -0.10:
            actions.append('RETRAIN: Significant accuracy drop detected')
        elif wr_change < -0.05:
            actions.append('RECALIBRATE: Moderate accuracy decline')
        
        if abs(bias) > 0.10:
            direction = 'overconfident' if bias > 0 else 'underconfident'
            actions.append(f'ADJUST_BIAS: Model is {direction} by {abs(bias):.1%}')
        
        return ' | '.join(actions) if actions else 'Monitor closely'

    # ------------------------------------------------------------------
    # 4. Market-Specific Performance
    # ------------------------------------------------------------------

    def market_performance(self) -> Dict:
        """Break down prediction quality by market type."""
        settled = [p for p in self.predictions if p['settled']]
        if not settled:
            return {'status': 'no_settled_predictions'}
        
        markets = defaultdict(list)
        for p in settled:
            market = self._normalize_market(p['market'])
            markets[market].append(p)
        
        report = {}
        for market, preds in markets.items():
            outcomes = [p['outcome'] for p in preds]
            probs = [p['predicted_prob'] for p in preds]
            evs = [p['ev'] for p in preds]
            stakes = [p['stake'] for p in preds]
            
            wins = sum(outcomes)
            total = len(outcomes)
            
            # Compute actual P&L from odds
            pnl = sum(
                p['stake'] * (p['odds'] - 1) if p['outcome'] == 1 else -p['stake']
                for p in preds
            )
            
            report[market] = {
                'total_bets': total,
                'wins': wins,
                'win_rate': round(wins / total, 3),
                'avg_predicted_prob': round(np.mean(probs), 3),
                'avg_ev': round(np.mean(evs), 3),
                'avg_odds': round(np.mean([p['odds'] for p in preds]), 2),
                'total_staked': round(sum(stakes), 2),
                'total_pnl': round(pnl, 2),
                'roi': round(pnl / max(sum(stakes), 0.01) * 100, 1),
                'bias': round(np.mean(probs) - wins/total, 4),
            }
        
        # Rank markets by ROI
        ranked = sorted(report.items(), key=lambda x: -x[1]['roi'])
        
        return {
            'markets': dict(ranked),
            'best_market': ranked[0][0] if ranked else None,
            'worst_market': ranked[-1][0] if ranked else None,
        }

    # ------------------------------------------------------------------
    # 5. EV Accuracy Tracking
    # ------------------------------------------------------------------

    def ev_accuracy(self, ev_bins: List[float] = None) -> Dict:
        """
        Check if predicted EV actually materializes in results.
        Groups bets by EV range and checks if win rates match predictions.
        """
        if ev_bins is None:
            ev_bins = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
        
        settled = [p for p in self.predictions if p['settled']]
        if len(settled) < 20:
            return {'status': 'insufficient_data'}
        
        results = {}
        for i in range(len(ev_bins)):
            low = ev_bins[i - 1] if i > 0 else 0
            high = ev_bins[i]
            
            bucket = [p for p in settled if low <= p['ev'] < high]
            if not bucket:
                continue
            
            outcomes = [p['outcome'] for p in bucket]
            probs = [p['predicted_prob'] for p in bucket]
            
            label = f"{low:.0%}-{high:.0%}"
            results[label] = {
                'n_bets': len(bucket),
                'win_rate': round(np.mean(outcomes), 3),
                'avg_predicted_prob': round(np.mean(probs), 3),
                'avg_ev': round(np.mean([p['ev'] for p in bucket]), 3),
                'actual_edge': round(
                    np.mean(outcomes) - np.mean([p['implied_prob'] for p in bucket]), 3
                ),
            }
        
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_market(self, market: str) -> str:
        """Normalize market names for grouping."""
        m = market.lower().strip()
        if 'btts' in m or 'both teams' in m:
            return 'btts'
        elif 'double chance' in m or 'double_chance' in m:
            return 'double_chance'
        elif '1.5' in m and ('1st half' in m or 'first half' in m):
            return 'first_half_1.5'
        elif '1.5' in m:
            return 'total_1.5'
        elif '2.5' in m:
            return 'total_2.5'
        elif 'draw no bet' in m or 'dnb' in m:
            return 'draw_no_bet'
        return m

    def _save(self):
        """Persist predictions to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.predictions, f, indent=2, default=str)
        except Exception as e:
            print(f"[PredictionMonitor] Save error: {e}")

    def _load(self):
        """Load predictions from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.predictions = json.load(f)
        except Exception:
            self.predictions = []

    def summary(self) -> Dict:
        """Quick summary of prediction tracking state."""
        settled = [p for p in self.predictions if p['settled']]
        unsettled = [p for p in self.predictions if not p['settled']]
        
        return {
            'total_predictions': len(self.predictions),
            'settled': len(settled),
            'unsettled': len(unsettled),
            'win_rate': round(np.mean([p['outcome'] for p in settled]), 3) if settled else None,
            'markets_tracked': list(set(self._normalize_market(p['market']) for p in self.predictions)),
        }
