"""
Performance Tracker
Financial metrics: P&L, ROI, Sharpe, drawdown, rolling windows, market breakdown.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class PerformanceTracker:
    """
    Tracks financial performance of the betting portfolio.
    
    Responsibilities:
    - Rolling P&L with daily / weekly / monthly windows
    - Sharpe ratio & risk-adjusted returns
    - Max drawdown and drawdown duration
    - Market-specific ROI breakdown
    - Bankroll management metrics
    - Streak tracking (win/loss runs)
    """

    def __init__(self, storage_path: str = 'quant_agent/performance_log.json'):
        self.storage_path = storage_path
        self.bets: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # 1. Record Bets
    # ------------------------------------------------------------------

    def record_bet(
        self,
        fixture_id: str,
        market: str,
        selection: str,
        stake: float,
        odds: float,
        won: bool,
        placed_at: str = None,
        settled_at: str = None,
        metadata: Optional[Dict] = None,
    ):
        """Record a settled bet result."""
        profit = stake * (odds - 1) if won else -stake
        
        entry = {
            'fixture_id': str(fixture_id),
            'market': market,
            'selection': selection,
            'stake': round(stake, 4),
            'odds': round(odds, 3),
            'won': won,
            'profit': round(profit, 4),
            'placed_at': placed_at or datetime.utcnow().isoformat(),
            'settled_at': settled_at or datetime.utcnow().isoformat(),
            'metadata': metadata or {},
        }
        self.bets.append(entry)
        self._save()

    def bulk_import(self, settled_bets: List[Dict]):
        """Import a batch of settled bets (e.g., from settled_bets.json)."""
        for bet in settled_bets:
            won = bet.get('profit', 0) > 0
            self.record_bet(
                fixture_id=str(bet.get('fixture_id', bet.get('event', ''))),
                market=bet.get('market', ''),
                selection=bet.get('selection', ''),
                stake=abs(bet.get('stake', bet.get('amount', 0))),
                odds=bet.get('odds', 0),
                won=won,
                placed_at=bet.get('placed_at', bet.get('timestamp', '')),
                settled_at=bet.get('settled_at', ''),
            )

    # ------------------------------------------------------------------
    # 2. Portfolio Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Overall portfolio performance summary."""
        if not self.bets:
            return {'status': 'no_bets'}
        
        profits = [b['profit'] for b in self.bets]
        stakes = [b['stake'] for b in self.bets]
        wins = sum(1 for b in self.bets if b['won'])
        
        total_staked = sum(stakes)
        total_profit = sum(profits)
        
        return {
            'total_bets': len(self.bets),
            'wins': wins,
            'losses': len(self.bets) - wins,
            'win_rate': round(wins / len(self.bets), 4),
            'total_staked': round(total_staked, 2),
            'total_profit': round(total_profit, 2),
            'roi': round(total_profit / max(total_staked, 0.01) * 100, 2),
            'avg_stake': round(np.mean(stakes), 4),
            'avg_odds': round(np.mean([b['odds'] for b in self.bets]), 2),
            'avg_profit_per_bet': round(np.mean(profits), 4),
            'best_bet': round(max(profits), 2),
            'worst_bet': round(min(profits), 2),
            'sharpe': self._sharpe_ratio(profits),
            'max_drawdown': self._max_drawdown(profits),
            'current_streak': self._current_streak(),
        }

    # ------------------------------------------------------------------
    # 3. Sharpe Ratio
    # ------------------------------------------------------------------

    def _sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.0) -> Optional[float]:
        """
        Sharpe ratio of bet returns.
        Uses per-bet returns (profit/stake) rather than raw profits.
        """
        if len(profits) < 10:
            return None
        
        returns = [
            b['profit'] / max(b['stake'], 0.01) for b in self.bets
        ]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return None
        
        # Annualize: assume ~1 bet per day on average
        total_days = self._total_days() or len(self.bets)
        bets_per_year = len(self.bets) / max(total_days, 1) * 365
        
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(bets_per_year)
        return round(sharpe, 3)

    # ------------------------------------------------------------------
    # 4. Drawdown Analysis
    # ------------------------------------------------------------------

    def _max_drawdown(self, profits: List[float]) -> Dict:
        """Calculate maximum drawdown and its duration."""
        if len(profits) < 2:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'duration_bets': 0}
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        
        max_dd = float(np.min(drawdowns))
        max_dd_idx = int(np.argmin(drawdowns))
        
        # Find drawdown duration
        peak_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if cumulative[i] == running_max[i]:
                peak_idx = i
                break
        
        # Find recovery point
        recovery_idx = len(profits) - 1
        for i in range(max_dd_idx, len(profits)):
            if cumulative[i] >= running_max[max_dd_idx]:
                recovery_idx = i
                break
        
        peak_value = running_max[max_dd_idx] if running_max[max_dd_idx] != 0 else 1.0
        
        return {
            'max_drawdown': round(max_dd, 4),
            'max_drawdown_pct': round(max_dd / max(abs(peak_value), 0.01) * 100, 2),
            'peak_bet': peak_idx,
            'trough_bet': max_dd_idx,
            'recovery_bet': recovery_idx,
            'duration_bets': max_dd_idx - peak_idx,
        }

    def drawdown_analysis(self) -> Dict:
        """Full drawdown analysis including current drawdown."""
        if not self.bets:
            return {'status': 'no_bets'}
        
        profits = [b['profit'] for b in self.bets]
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        
        current_dd = float(cumulative[-1] - running_max[-1])
        in_drawdown = current_dd < -0.001
        
        return {
            'max_drawdown_info': self._max_drawdown(profits),
            'current_drawdown': round(current_dd, 4),
            'in_drawdown': in_drawdown,
            'cumulative_pnl': round(float(cumulative[-1]), 4),
            'all_time_high': round(float(running_max[-1]), 4),
        }

    # ------------------------------------------------------------------
    # 5. Rolling Window Metrics
    # ------------------------------------------------------------------

    def rolling_performance(self, window: int = 50) -> List[Dict]:
        """Calculate rolling performance metrics over a sliding window."""
        if len(self.bets) < window:
            return []
        
        results = []
        for i in range(window, len(self.bets) + 1):
            window_bets = self.bets[i - window:i]
            profits = [b['profit'] for b in window_bets]
            wins = sum(1 for b in window_bets if b['won'])
            total_staked = sum(b['stake'] for b in window_bets)
            
            results.append({
                'end_bet': i,
                'win_rate': round(wins / window, 3),
                'roi': round(sum(profits) / max(total_staked, 0.01) * 100, 2),
                'total_profit': round(sum(profits), 4),
                'avg_odds': round(np.mean([b['odds'] for b in window_bets]), 2),
            })
        
        return results

    def daily_pnl(self) -> Dict:
        """Break down P&L by calendar day."""
        daily = defaultdict(lambda: {'bets': 0, 'profit': 0, 'stake': 0, 'wins': 0})
        
        for bet in self.bets:
            date_str = bet.get('settled_at', bet.get('placed_at', ''))[:10]
            if not date_str:
                continue
            daily[date_str]['bets'] += 1
            daily[date_str]['profit'] += bet['profit']
            daily[date_str]['stake'] += bet['stake']
            daily[date_str]['wins'] += int(bet['won'])
        
        # Sort by date
        sorted_daily = dict(sorted(daily.items()))
        
        # Compute cumulative
        cumulative = 0
        for date_str, data in sorted_daily.items():
            cumulative += data['profit']
            data['cumulative_pnl'] = round(cumulative, 4)
            data['profit'] = round(data['profit'], 4)
            data['roi'] = round(data['profit'] / max(data['stake'], 0.01) * 100, 2)
        
        return sorted_daily

    # ------------------------------------------------------------------
    # 6. Market Breakdown
    # ------------------------------------------------------------------

    def market_breakdown(self) -> Dict:
        """P&L and ROI per market type."""
        markets = defaultdict(list)
        
        for bet in self.bets:
            market = self._normalize_market(bet['market'])
            markets[market].append(bet)
        
        report = {}
        for market, bets in markets.items():
            profits = [b['profit'] for b in bets]
            stakes = [b['stake'] for b in bets]
            wins = sum(1 for b in bets if b['won'])
            
            report[market] = {
                'total_bets': len(bets),
                'wins': wins,
                'win_rate': round(wins / len(bets), 3),
                'total_staked': round(sum(stakes), 2),
                'total_profit': round(sum(profits), 2),
                'roi': round(sum(profits) / max(sum(stakes), 0.01) * 100, 2),
                'avg_odds': round(np.mean([b['odds'] for b in bets]), 2),
                'sharpe': self._sharpe_ratio(profits),
            }
        
        # Rank by ROI
        return dict(sorted(report.items(), key=lambda x: -x[1]['roi']))

    # ------------------------------------------------------------------
    # 7. Streak Analysis
    # ------------------------------------------------------------------

    def _current_streak(self) -> Dict:
        """Current win/loss streak."""
        if not self.bets:
            return {'type': None, 'length': 0}
        
        last = self.bets[-1]['won']
        streak_type = 'win' if last else 'loss'
        length = 0
        
        for bet in reversed(self.bets):
            if bet['won'] == last:
                length += 1
            else:
                break
        
        return {'type': streak_type, 'length': length}

    def streak_analysis(self) -> Dict:
        """Full streak history: longest win, longest loss, average lengths."""
        if not self.bets:
            return {'status': 'no_bets'}
        
        streaks = []
        current = {'type': 'win' if self.bets[0]['won'] else 'loss', 'length': 1}
        
        for bet in self.bets[1:]:
            bet_type = 'win' if bet['won'] else 'loss'
            if bet_type == current['type']:
                current['length'] += 1
            else:
                streaks.append(current)
                current = {'type': bet_type, 'length': 1}
        streaks.append(current)
        
        win_streaks = [s['length'] for s in streaks if s['type'] == 'win']
        loss_streaks = [s['length'] for s in streaks if s['type'] == 'loss']
        
        return {
            'longest_win_streak': max(win_streaks) if win_streaks else 0,
            'longest_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': round(np.mean(win_streaks), 1) if win_streaks else 0,
            'avg_loss_streak': round(np.mean(loss_streaks), 1) if loss_streaks else 0,
            'current': self._current_streak(),
            'total_streaks': len(streaks),
        }

    # ------------------------------------------------------------------
    # 8. Risk Metrics
    # ------------------------------------------------------------------

    def risk_report(self) -> Dict:
        """Portfolio-level risk assessment."""
        if len(self.bets) < 10:
            return {'status': 'insufficient_data'}
        
        profits = [b['profit'] for b in self.bets]
        returns = [b['profit'] / max(b['stake'], 0.01) for b in self.bets]
        
        # Value at Risk (95th percentile)
        var_95 = float(np.percentile(profits, 5))
        
        # Expected Shortfall (average of worst 5%)
        worst_5pct = sorted(profits)[:max(1, len(profits) // 20)]
        es = float(np.mean(worst_5pct))
        
        # Profit factor (gross wins / gross losses)
        gross_wins = sum(p for p in profits if p > 0) or 0.01
        gross_losses = abs(sum(p for p in profits if p < 0)) or 0.01
        profit_factor = gross_wins / gross_losses
        
        # Win/loss ratio
        avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
        avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0.01
        
        return {
            'sharpe_ratio': self._sharpe_ratio(profits),
            'var_95': round(var_95, 4),
            'expected_shortfall': round(es, 4),
            'profit_factor': round(profit_factor, 3),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'win_loss_ratio': round(avg_win / max(avg_loss, 0.01), 3),
            'return_volatility': round(float(np.std(returns)), 4),
            'skewness': round(float(self._skewness(profits)), 3),
            'max_drawdown': self._max_drawdown(profits),
        }

    def _skewness(self, data: List[float]) -> float:
        """Calculate skewness of returns."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(n / ((n-1)*(n-2)) * np.sum(((np.array(data) - mean) / std) ** 3))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_days(self) -> int:
        """Total span of betting activity in days."""
        if not self.bets:
            return 0
        dates = []
        for b in self.bets:
            date_str = b.get('placed_at', '')[:10]
            if date_str:
                try:
                    dates.append(datetime.fromisoformat(date_str))
                except Exception:
                    pass
        if len(dates) < 2:
            return 1
        return max(1, (max(dates) - min(dates)).days)

    def _normalize_market(self, market: str) -> str:
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
        try:
            os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.bets, f, indent=2, default=str)
        except Exception as e:
            print(f"[PerformanceTracker] Save error: {e}")

    def _load(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.bets = json.load(f)
        except Exception:
            self.bets = []
