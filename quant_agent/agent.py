"""
Quant Agent - Main Orchestrator
Ties together all quant modules into a single diagnostic and optimization engine.
"""

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .data_quality import DataQualityAnalyzer
from .feature_engine import FeatureEngine
from .prediction_monitor import PredictionMonitor
from .model_manager import ModelManager
from .performance_tracker import PerformanceTracker

logger = logging.getLogger("quant_agent")


class QuantAgent:
    """
    Central orchestrator for quantitative model management.
    
    Runs diagnostics across all modules and produces:
    - Health status (overall system readiness)
    - Actionable recommendations (retrain, recalibrate, ban markets, etc.)
    - Performance dashboards (Sharpe, drawdown, ROI per market)
    - Continuous optimization decisions
    
    Usage:
        agent = QuantAgent(data_dir='.')
        report = agent.full_diagnostic()
        print(report['overall_status'])
        print(report['actions'])
    """

    def __init__(
        self,
        data_dir: str = '.',
        model_dir: str = '.',
        report_dir: str = 'quant_reports',
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.report_dir = Path(data_dir) / report_dir
        self.report_dir.mkdir(exist_ok=True)

        # Initialise sub-modules
        self.data_quality = DataQualityAnalyzer()
        self.features = FeatureEngine()
        self.predictions = PredictionMonitor(
            storage_path=str(self.data_dir / 'quant_agent' / 'prediction_log.json')
        )
        self.models = ModelManager(model_dir=str(self.model_dir))
        self.performance = PerformanceTracker(
            storage_path=str(self.data_dir / 'quant_agent' / 'performance_log.json')
        )

    # ------------------------------------------------------------------
    # 1. Full Diagnostic
    # ------------------------------------------------------------------

    def full_diagnostic(self) -> Dict:
        """
        Run every check and return a comprehensive health report.
        This is the primary entry point — call after each workflow run.
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'modules': {},
            'overall_status': 'UNKNOWN',
            'actions': [],
        }

        # A. Data Quality
        try:
            data_report = self._check_data_quality()
            report['modules']['data_quality'] = data_report
        except Exception as e:
            report['modules']['data_quality'] = {'error': str(e)}

        # B. Prediction Quality
        try:
            pred_report = self._check_predictions()
            report['modules']['predictions'] = pred_report
        except Exception as e:
            report['modules']['predictions'] = {'error': str(e)}

        # C. Model Health
        try:
            model_report = self._check_model_health()
            report['modules']['model'] = model_report
        except Exception as e:
            report['modules']['model'] = {'error': str(e)}

        # D. Financial Performance
        try:
            perf_report = self._check_performance()
            report['modules']['performance'] = perf_report
        except Exception as e:
            report['modules']['performance'] = {'error': str(e)}

        # E. Synthesise
        report['overall_status'] = self._determine_status(report['modules'])
        report['actions'] = self._generate_actions(report['modules'])
        report['action_priority'] = self._prioritize_actions(report['actions'])

        # Persist
        self._save_report(report)

        return report

    # ------------------------------------------------------------------
    # 2. Sub-Checks
    # ------------------------------------------------------------------

    def _check_data_quality(self) -> Dict:
        """Run data quality checks on available fixture files."""
        result = {
            'freshness': self.data_quality.check_data_freshness(str(self.data_dir)),
        }

        # Try to find and validate merged JSON data
        merged_files = sorted(self.data_dir.glob('api_football_merged_*.json'))
        if merged_files:
            latest = merged_files[-1]
            try:
                with open(latest) as f:
                    fixtures = json.load(f)
                if isinstance(fixtures, list):
                    grades = []
                    for fix in fixtures[:50]:  # sample
                        validation = self.data_quality.validate_fixture(fix)
                        grades.append(validation.get('score', 0))
                    result['avg_fixture_score'] = round(sum(grades) / max(len(grades), 1), 1)
                    result['fixtures_sampled'] = len(grades)
                    result['latest_file'] = latest.name
            except Exception as e:
                result['fixture_validation_error'] = str(e)

        return result

    def _check_predictions(self) -> Dict:
        """Assess prediction quality and calibration."""
        summary = self.predictions.summary()
        
        result = {
            'summary': summary,
        }
        
        if summary.get('settled', 0) >= 20:
            result['calibration'] = self.predictions.calibration_report()
            result['degradation'] = self.predictions.detect_degradation()
            result['market_performance'] = self.predictions.market_performance()
            result['ev_accuracy'] = self.predictions.ev_accuracy()
        
        return result

    def _check_model_health(self) -> Dict:
        """Check model status and retraining needs."""
        retrain = self.models.should_retrain(
            performance_report=self._latest_performance_report()
        )
        
        model_info = self.models.get_model_info()
        
        return {
            'retrain_decision': retrain,
            'model_info': model_info,
        }

    def _latest_performance_report(self) -> Dict:
        """Build a performance report for the retraining decision engine."""
        summary = self.predictions.summary()
        if not summary or summary.get('settled', 0) < 10:
            return {}
        
        cal = self.predictions.calibration_report()
        overall = cal.get('overall', {}) if isinstance(cal, dict) else {}
        
        return {
            'trend': 'declining' if overall.get('bias', 0) > 0.08 else 'stable',
            'recent_brier_score': overall.get('brier_score'),
        }

    def _check_performance(self) -> Dict:
        """Financial performance dashboard."""
        summary = self.performance.summary()
        
        result = {
            'summary': summary,
        }
        
        if summary.get('total_bets', 0) >= 10:
            result['risk'] = self.performance.risk_report()
            result['markets'] = self.performance.market_breakdown()
            result['drawdown'] = self.performance.drawdown_analysis()
            result['streaks'] = self.performance.streak_analysis()
        
        return result

    # ------------------------------------------------------------------
    # 3. Status Determination
    # ------------------------------------------------------------------

    def _determine_status(self, modules: Dict) -> str:
        """
        Overall system status based on all module outputs.
        GREEN / YELLOW / RED
        """
        issues = 0
        critical = 0

        # Data quality
        dq = modules.get('data_quality', {})
        avg_score = dq.get('avg_fixture_score', 100)
        if avg_score < 50:
            critical += 1
        elif avg_score < 70:
            issues += 1

        # Predictions
        pred = modules.get('predictions', {})
        degradation = pred.get('degradation', {})
        if degradation.get('status') == 'DEGRADED':
            critical += 1
        cal = pred.get('calibration', {})
        overall_cal = cal.get('overall', {}) if isinstance(cal, dict) else {}
        if overall_cal.get('brier_score') and overall_cal['brier_score'] > 0.28:
            issues += 1

        # Model
        model = modules.get('model', {})
        retrain = model.get('retrain_decision', {})
        if retrain.get('decision') == 'RETRAIN_NOW':
            critical += 1
        elif retrain.get('decision') == 'RETRAIN_SOON':
            issues += 1

        # Performance
        perf = modules.get('performance', {})
        summary = perf.get('summary', {})
        if isinstance(summary, dict):
            roi = summary.get('roi', 0)
            if roi < -20:
                critical += 1
            elif roi < 0:
                issues += 1

        if critical > 0:
            return 'RED'
        elif issues > 0:
            return 'YELLOW'
        return 'GREEN'

    # ------------------------------------------------------------------
    # 4. Action Generation
    # ------------------------------------------------------------------

    def _generate_actions(self, modules: Dict) -> List[Dict]:
        """Generate specific actions based on diagnostic results."""
        actions = []

        # Model retraining
        model = modules.get('model', {})
        retrain = model.get('retrain_decision', {})
        if retrain.get('decision') in ('RETRAIN_NOW', 'RETRAIN_SOON'):
            actions.append({
                'action': 'RETRAIN_MODEL',
                'priority': 'HIGH' if retrain['decision'] == 'RETRAIN_NOW' else 'MEDIUM',
                'reason': '; '.join(retrain.get('reasons', [])),
                'urgency': retrain.get('urgency', 0),
            })

        # Calibration fix
        pred = modules.get('predictions', {})
        cal = pred.get('calibration', {})
        if isinstance(cal, dict):
            overall = cal.get('overall', {})
            if abs(overall.get('bias', 0)) > 0.08:
                direction = 'up' if overall['bias'] < 0 else 'down'
                actions.append({
                    'action': 'RECALIBRATE',
                    'priority': 'HIGH',
                    'reason': f"Bias={overall['bias']:.3f}, adjust probabilities {direction}",
                    'suggested_factor': round(1.0 - overall['bias'], 3),
                })

        # Market banning / adjusting
        perf = modules.get('performance', {})
        markets = perf.get('markets', {})
        for market, stats in markets.items():
            if isinstance(stats, dict):
                if stats.get('total_bets', 0) >= 10 and stats.get('roi', 0) < -15:
                    actions.append({
                        'action': 'BAN_MARKET',
                        'priority': 'HIGH',
                        'market': market,
                        'reason': f"ROI={stats['roi']:.1f}% over {stats['total_bets']} bets",
                    })
                elif stats.get('total_bets', 0) >= 10 and stats.get('roi', 0) < -5:
                    actions.append({
                        'action': 'REDUCE_STAKES',
                        'priority': 'MEDIUM',
                        'market': market,
                        'reason': f"Marginal ROI={stats['roi']:.1f}%",
                    })

        # Drawdown warning
        drawdown = perf.get('drawdown', {})
        if isinstance(drawdown, dict) and drawdown.get('in_drawdown'):
            dd = drawdown.get('current_drawdown', 0)
            if dd < -1.0:
                actions.append({
                    'action': 'REDUCE_EXPOSURE',
                    'priority': 'HIGH',
                    'reason': f"In drawdown: £{dd:.2f}",
                })

        # Data freshness
        dq = modules.get('data_quality', {})
        freshness = dq.get('freshness', {})
        if isinstance(freshness, dict):
            stale = freshness.get('stale_files', [])
            if len(stale) > 3:
                actions.append({
                    'action': 'REFRESH_DATA',
                    'priority': 'MEDIUM',
                    'reason': f"{len(stale)} stale data files",
                })

        return actions

    def _prioritize_actions(self, actions: List[Dict]) -> List[str]:
        """Return actions sorted by priority as a simple ordered list."""
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_actions = sorted(
            actions,
            key=lambda a: priority_order.get(a.get('priority', 'LOW'), 3)
        )
        return [
            f"[{a['priority']}] {a['action']}: {a.get('reason', '')}"
            for a in sorted_actions
        ]

    # ------------------------------------------------------------------
    # 5. Quick Reports
    # ------------------------------------------------------------------

    def quick_status(self) -> str:
        """One-line status for logging or alerts."""
        pred_summary = self.predictions.summary()
        perf_summary = self.performance.summary()

        settled = pred_summary.get('settled', 0)
        wr = pred_summary.get('win_rate')
        roi = perf_summary.get('roi', 0) if isinstance(perf_summary, dict) else 0
        total_bets = perf_summary.get('total_bets', 0) if isinstance(perf_summary, dict) else 0

        wr_str = f"{wr:.1%}" if wr is not None else 'N/A'
        return (
            f"Bets: {total_bets} | WR: {wr_str} | ROI: {roi:.1f}% | "
            f"Settled: {settled} | Status: checking..."
        )

    def market_health(self) -> Dict:
        """Quick market-level health check."""
        markets = self.performance.market_breakdown()
        
        healthy = []
        warning = []
        critical = []
        
        for market, stats in markets.items():
            if not isinstance(stats, dict):
                continue
            roi = stats.get('roi', 0)
            bets = stats.get('total_bets', 0)
            
            if bets < 5:
                continue
            
            if roi >= 5:
                healthy.append(f"{market}: ROI={roi:.1f}% ({bets} bets)")
            elif roi >= -5:
                warning.append(f"{market}: ROI={roi:.1f}% ({bets} bets)")
            else:
                critical.append(f"{market}: ROI={roi:.1f}% ({bets} bets)")
        
        return {
            'healthy': healthy,
            'warning': warning,
            'critical': critical,
        }

    # ------------------------------------------------------------------
    # 6. Feedback Loop: Ingest Settled Bets
    # ------------------------------------------------------------------

    def ingest_settled_bets(self, filepath: str = None):
        """
        Read settled bets and feed them into both PredictionMonitor and PerformanceTracker.
        This closes the feedback loop.
        """
        if filepath is None:
            filepath = str(self.data_dir / 'settled_bets.json')
        
        if not os.path.exists(filepath):
            logger.warning("No settled bets file found at %s", filepath)
            return {'status': 'no_file', 'path': filepath}
        
        with open(filepath) as f:
            settled = json.load(f)
        
        if not isinstance(settled, list):
            return {'status': 'invalid_format'}
        
        # Track what was already imported
        existing_ids = set()
        for b in self.performance.bets:
            existing_ids.add(f"{b['fixture_id']}_{b['market']}")
        
        new_count = 0
        for bet in settled:
            bet_id = f"{bet.get('fixture_id', bet.get('event', ''))}_{bet.get('market', '')}"
            if bet_id not in existing_ids:
                won = bet.get('profit', 0) > 0
                
                # Feed to PerformanceTracker
                self.performance.record_bet(
                    fixture_id=str(bet.get('fixture_id', bet.get('event', ''))),
                    market=bet.get('market', ''),
                    selection=bet.get('selection', ''),
                    stake=abs(bet.get('stake', bet.get('amount', 0))),
                    odds=bet.get('odds', 0),
                    won=won,
                    placed_at=bet.get('placed_at', bet.get('timestamp', '')),
                    settled_at=bet.get('settled_at', ''),
                )
                
                # Feed to PredictionMonitor
                self.predictions.record_outcome(
                    fixture_id=str(bet.get('fixture_id', bet.get('event', ''))),
                    market=bet.get('market', ''),
                    won=won,
                )
                
                new_count += 1
        
        return {
            'status': 'imported',
            'new_bets': new_count,
            'total_bets': len(self.performance.bets),
            'file': filepath,
        }

    # ------------------------------------------------------------------
    # 7. Automated Optimization Loop
    # ------------------------------------------------------------------

    def optimization_pass(self) -> Dict:
        """
        Single optimization pass — call this after each betting workflow run.
        
        1. Ingest new settled results
        2. Run diagnostics
        3. Return actions to take
        
        The caller (automated_betting_workflow.py) should act on the returned actions.
        """
        # Step 1: Ingest
        ingest_result = self.ingest_settled_bets()
        
        # Step 2: Diagnose
        diagnostic = self.full_diagnostic()
        
        # Step 3: Return optimizations
        return {
            'ingest': ingest_result,
            'status': diagnostic['overall_status'],
            'actions': diagnostic['action_priority'],
            'raw_diagnostic': diagnostic,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: Dict):
        """Save diagnostic report to disk."""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            path = self.report_dir / f'diagnostic_{timestamp}.json'
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also save as "latest"
            latest_path = self.report_dir / 'diagnostic_latest.json'
            with open(latest_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Keep only last 30 reports
            reports = sorted(self.report_dir.glob('diagnostic_2*.json'))
            while len(reports) > 30:
                reports.pop(0).unlink()
        except Exception as e:
            logger.error("Failed to save report: %s", e)

    def load_latest_report(self) -> Optional[Dict]:
        """Load the most recent diagnostic report."""
        path = self.report_dir / 'diagnostic_latest.json'
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
