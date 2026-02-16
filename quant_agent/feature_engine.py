"""
Feature Engine
Discovers, ranks, and manages features for the betting model.
Handles feature importance, selection, and engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeatureEngine:
    """
    Quantitative feature analysis and management.
    
    Responsibilities:
    - Rank features by predictive power per target
    - Detect useless or harmful features
    - Track feature importance drift over time
    - Recommend feature engineering opportunities
    - Identify feature interactions
    """

    def __init__(self):
        self.importance_history: List[Dict] = []
        self.feature_rankings: Dict[str, List] = {}  # target -> [(feature, score)]
        self._last_analysis: Optional[Dict] = None

    # ------------------------------------------------------------------
    # 1. Feature Importance Analysis
    # ------------------------------------------------------------------

    def analyze_importance(
        self,
        X: pd.DataFrame,
        targets: Dict[str, pd.Series],
        method: str = 'all'
    ) -> Dict:
        """
        Comprehensive feature importance analysis across all targets.
        
        Methods:
        - 'xgb': XGBoost built-in feature importance (gain)
        - 'mutual_info': Mutual information scores
        - 'permutation': Permutation importance (most reliable but slowest)
        - 'all': Combine all methods into a consensus ranking
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'error': 'Requires xgboost and sklearn'}

        results = {}
        feature_names = list(X.columns)
        
        for target_name, y in targets.items():
            if y is None or len(y) == 0:
                continue
                
            target_results = {}
            
            # Determine if classification or regression
            is_classification = y.nunique() <= 10
            
            # Clean data
            mask = y.notna() & X.notna().all(axis=1)
            X_clean = X[mask].copy()
            y_clean = y[mask].copy()
            
            if len(X_clean) < 50:
                results[target_name] = {'error': f'Insufficient samples: {len(X_clean)}'}
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_clean),
                columns=feature_names,
                index=X_clean.index
            )
            
            # Method 1: XGBoost importance (gain-based)
            if method in ('xgb', 'all'):
                target_results['xgb'] = self._xgb_importance(
                    X_scaled, y_clean, feature_names, is_classification
                )
            
            # Method 2: Mutual information
            if method in ('mutual_info', 'all'):
                target_results['mutual_info'] = self._mutual_info_importance(
                    X_scaled, y_clean, feature_names, is_classification
                )
            
            # Method 3: Permutation importance
            if method in ('permutation', 'all'):
                target_results['permutation'] = self._permutation_importance(
                    X_scaled, y_clean, feature_names, is_classification
                )
            
            # Consensus ranking (if multiple methods)
            if len(target_results) > 1:
                target_results['consensus'] = self._consensus_ranking(
                    target_results, feature_names
                )
            
            results[target_name] = target_results

        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_features': len(feature_names),
            'n_samples': len(X),
            'targets_analyzed': list(results.keys()),
            'results': results,
            'recommendations': self._generate_feature_recommendations(results, feature_names),
        }
        
        self._last_analysis = analysis
        self.importance_history.append({
            'timestamp': analysis['timestamp'],
            'n_features': analysis['n_features'],
            'top_features': self._extract_top_features(results, n=20),
        })
        
        return analysis

    # ------------------------------------------------------------------
    # 2. Feature Selection
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: pd.DataFrame,
        targets: Dict[str, pd.Series],
        min_importance: float = 0.001,
        max_features: int = 100,
    ) -> Dict:
        """
        Select the best features based on importance analysis.
        Returns features to keep and features to drop.
        """
        analysis = self.analyze_importance(X, targets, method='all')
        
        if 'error' in analysis:
            return analysis
        
        # Aggregate importance across all targets
        feature_scores = defaultdict(list)
        
        for target_name, target_results in analysis['results'].items():
            if isinstance(target_results, dict) and 'consensus' in target_results:
                consensus = target_results['consensus']
                for feat, score in consensus:
                    feature_scores[feat].append(score)
            elif isinstance(target_results, dict):
                # Use first available method
                for method_name, method_scores in target_results.items():
                    if isinstance(method_scores, list):
                        for feat, score in method_scores:
                            feature_scores[feat].append(score)
                        break
        
        # Average scores across targets
        avg_scores = {
            feat: np.mean(scores) for feat, scores in feature_scores.items()
        }
        
        # Sort by importance
        ranked = sorted(avg_scores.items(), key=lambda x: -x[1])
        
        # Split into keep/drop
        features_to_keep = []
        features_to_drop = []
        
        for feat, score in ranked:
            if score >= min_importance and len(features_to_keep) < max_features:
                features_to_keep.append((feat, round(score, 5)))
            else:
                features_to_drop.append((feat, round(score, 5)))
        
        return {
            'features_to_keep': features_to_keep,
            'features_to_drop': features_to_drop,
            'n_kept': len(features_to_keep),
            'n_dropped': len(features_to_drop),
            'importance_threshold': min_importance,
        }

    # ------------------------------------------------------------------
    # 3. Feature Importance Drift
    # ------------------------------------------------------------------

    def detect_importance_drift(self) -> Dict:
        """
        Compare current feature importance to historical baselines.
        Detects when features become more or less important.
        """
        if len(self.importance_history) < 2:
            return {'status': 'insufficient_history', 'message': 'Need ≥2 analyses'}
        
        latest = self.importance_history[-1]['top_features']
        previous = self.importance_history[-2]['top_features']
        
        latest_set = set(f[0] for f in latest)
        previous_set = set(f[0] for f in previous)
        
        new_important = latest_set - previous_set
        lost_importance = previous_set - latest_set
        stable = latest_set & previous_set
        
        # For stable features, compute rank changes
        latest_ranks = {f: i for i, (f, _) in enumerate(latest)}
        previous_ranks = {f: i for i, (f, _) in enumerate(previous)}
        
        rank_changes = []
        for feat in stable:
            change = previous_ranks.get(feat, 0) - latest_ranks.get(feat, 0)
            rank_changes.append((feat, change))
        
        rank_changes.sort(key=lambda x: -abs(x[1]))
        
        return {
            'new_important_features': list(new_important),
            'lost_importance_features': list(lost_importance),
            'biggest_rank_changes': rank_changes[:10],
            'stability_pct': round(len(stable) / max(len(latest_set), 1) * 100, 1),
            'recommendation': 'RETRAIN' if len(new_important) > 5 else 'STABLE',
        }

    # ------------------------------------------------------------------
    # 4. Feature Correlation Analysis
    # ------------------------------------------------------------------

    def analyze_correlations(self, X: pd.DataFrame, threshold: float = 0.90) -> Dict:
        """Find highly correlated feature pairs (candidates for removal)."""
        numeric_X = X.select_dtypes(include=[np.number])
        corr_matrix = numeric_X.corr().abs()
        
        # Find pairs above threshold
        high_corr_pairs = []
        seen = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if corr >= threshold and (col_i, col_j) not in seen:
                    high_corr_pairs.append((col_i, col_j, round(corr, 3)))
                    seen.add((col_i, col_j))
        
        high_corr_pairs.sort(key=lambda x: -x[2])
        
        # Suggest which of each pair to drop (fewer correlations overall)
        drop_candidates = defaultdict(int)
        for f1, f2, _ in high_corr_pairs:
            drop_candidates[f1] += 1
            drop_candidates[f2] += 1
        
        suggested_drops = []
        for f1, f2, corr in high_corr_pairs:
            # Drop the one with more correlations (more redundant)
            drop = f1 if drop_candidates[f1] >= drop_candidates[f2] else f2
            if drop not in [d[0] for d in suggested_drops]:
                suggested_drops.append((drop, corr))
        
        return {
            'high_correlation_pairs': high_corr_pairs[:30],
            'suggested_drops': suggested_drops,
            'total_pairs_above_threshold': len(high_corr_pairs),
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _xgb_importance(self, X, y, feature_names, is_classification):
        """XGBoost gain-based importance."""
        try:
            if is_classification:
                model = XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss'
                )
            else:
                model = XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42, verbosity=0
                )
            
            model.fit(X, y)
            importances = model.feature_importances_
            
            ranked = sorted(
                zip(feature_names, importances),
                key=lambda x: -x[1]
            )
            return ranked
        except Exception as e:
            return [('error', str(e))]

    def _mutual_info_importance(self, X, y, feature_names, is_classification):
        """Mutual information scores."""
        try:
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            ranked = sorted(
                zip(feature_names, mi_scores),
                key=lambda x: -x[1]
            )
            return ranked
        except Exception as e:
            return [('error', str(e))]

    def _permutation_importance(self, X, y, feature_names, is_classification):
        """Permutation-based importance (gold standard)."""
        try:
            if is_classification:
                model = XGBClassifier(
                    n_estimators=50, max_depth=3, random_state=42,
                    verbosity=0, use_label_encoder=False, eval_metric='logloss'
                )
            else:
                model = XGBRegressor(
                    n_estimators=50, max_depth=3, random_state=42, verbosity=0
                )
            
            model.fit(X, y)
            
            result = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            
            ranked = sorted(
                zip(feature_names, result.importances_mean),
                key=lambda x: -x[1]
            )
            return ranked
        except Exception as e:
            return [('error', str(e))]

    def _consensus_ranking(self, method_results, feature_names):
        """Combine rankings from multiple methods using Borda count."""
        n = len(feature_names)
        borda_scores = defaultdict(float)
        n_methods = 0
        
        for method_name, rankings in method_results.items():
            if not isinstance(rankings, list) or not rankings:
                continue
            if rankings[0][0] == 'error':
                continue
            
            n_methods += 1
            for rank, (feat, _) in enumerate(rankings):
                borda_scores[feat] += (n - rank) / n
        
        if n_methods > 0:
            borda_scores = {f: s / n_methods for f, s in borda_scores.items()}
        
        return sorted(borda_scores.items(), key=lambda x: -x[1])

    def _extract_top_features(self, results, n=20):
        """Extract top N features across all targets."""
        all_scores = defaultdict(list)
        
        for target_name, target_results in results.items():
            if isinstance(target_results, dict):
                consensus = target_results.get('consensus', [])
                if consensus:
                    for feat, score in consensus[:n*2]:
                        all_scores[feat].append(score)
        
        avg_scores = [(f, np.mean(s)) for f, s in all_scores.items()]
        return sorted(avg_scores, key=lambda x: -x[1])[:n]

    def _generate_feature_recommendations(self, results, feature_names) -> List[str]:
        """Generate actionable recommendations based on feature analysis."""
        recommendations = []
        
        # Check for commonly useless features
        zero_importance = set()
        for target_name, target_results in results.items():
            if isinstance(target_results, dict):
                for method_name, rankings in target_results.items():
                    if isinstance(rankings, list) and rankings and rankings[0][0] != 'error':
                        bottom = [f for f, s in rankings if s <= 0.0001]
                        zero_importance.update(bottom)
        
        if len(zero_importance) > 10:
            recommendations.append(
                f"DROP {len(zero_importance)} features with near-zero importance: "
                f"reduces noise and speeds up training"
            )
        
        if len(feature_names) > 100:
            recommendations.append(
                f"Consider feature selection: {len(feature_names)} features is high. "
                f"Top 60-80 features often perform as well or better."
            )
        
        recommendations.append(
            "Run analyze_correlations() to find redundant features that can be removed."
        )
        
        return recommendations
