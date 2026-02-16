"""
Model Manager
=============
Handles the full model lifecycle: training, hyperparameter tuning,
versioning, evaluation, and automated retraining decisions.
"""

import json
import logging
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, log_loss,
        f1_score, mean_absolute_error, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("quant_agent.model_manager")


class ModelManager:
    """
    Manages model training, evaluation, versioning, and retraining decisions.

    Core principles:
    - Temporal cross-validation only (no future data leakage)
    - Calibration on held-out sets (not training data)
    - Hyperparameter tuning with proper validation
    - Model versioning with performance metadata
    - Automatic retraining triggers based on drift and decay
    """

    # Default hyperparameter search space
    PARAM_GRID = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1.0, 1.5, 2.0],
    }

    # Targets and their types
    TARGETS = {
        "match_result": {"type": "classification", "classes": 3},
        "double_chance_1x": {"type": "binary", "classes": 2},
        "double_chance_x2": {"type": "binary", "classes": 2},
        "double_chance_12": {"type": "binary", "classes": 2},
        "btts": {"type": "binary", "classes": 2},
        "goals": {"type": "regression", "classes": None},
    }

    def __init__(self, model_dir: str = ".", max_versions: int = 10):
        self.model_dir = Path(model_dir)
        self.max_versions = max_versions
        self.models: Dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.calibrators: Dict = {}
        self.training_metadata: Dict = {}
        self.version_history: List[Dict] = []
        self._load_version_history()

    # ------------------------------------------------------------------
    # 1. Training with Temporal CV
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        targets: Dict[str, pd.Series],
        n_splits: int = 5,
        tune_hyperparams: bool = False,
        calibrate: bool = True,
    ) -> Dict:
        """
        Train all models with proper temporal cross-validation.

        Args:
            X: Feature DataFrame (must be sorted by date)
            targets: Dict of target name -> Series
            n_splits: Number of CV splits
            tune_hyperparams: Whether to run hyperparameter search
            calibrate: Whether to calibrate probability outputs

        Returns:
            Training report with per-model metrics
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return {"error": "Requires xgboost and sklearn"}

        start_time = time.time()
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "models": {},
        }

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        # Encode match result labels
        if "match_result" in targets and targets["match_result"] is not None:
            self.label_encoder = LabelEncoder()
            targets = dict(targets)  # copy
            targets["match_result"] = pd.Series(
                self.label_encoder.fit_transform(targets["match_result"]),
                index=targets["match_result"].index,
            )

        # Train each model
        for target_name, target_config in self.TARGETS.items():
            if target_name not in targets or targets[target_name] is None:
                continue

            y = targets[target_name]
            mask = y.notna() & X_scaled.notna().all(axis=1)
            X_train = X_scaled[mask]
            y_train = y[mask]

            if len(X_train) < 50:
                logger.warning(
                    "Skipping %s: only %d samples", target_name, len(X_train)
                )
                report["models"][target_name] = {"skipped": True, "reason": "insufficient_data"}
                continue

            logger.info(
                "Training %s (%s) on %d samples",
                target_name, target_config["type"], len(X_train),
            )

            # Get hyperparameters
            if tune_hyperparams:
                best_params = self._tune_hyperparams(
                    X_train, y_train, target_config["type"], n_splits
                )
            else:
                best_params = self._default_params(target_config["type"])

            # Cross-validate
            cv_results = self._temporal_cv(
                X_train, y_train, target_config, best_params, n_splits
            )

            # Train final model on all data
            model = self._create_model(target_config["type"], best_params)
            model.fit(X_train, y_train)
            self.models[target_name] = model

            # Calibrate probability outputs (on last CV fold's test set)
            if calibrate and target_config["type"] in ("binary", "classification"):
                self._calibrate_model(
                    X_train, y_train, target_name, target_config, n_splits
                )

            report["models"][target_name] = {
                "params": best_params,
                "cv_results": cv_results,
                "n_samples": len(X_train),
                "tuned": tune_hyperparams,
                "calibrated": target_name in self.calibrators,
            }

        report["training_time_seconds"] = round(time.time() - start_time, 1)
        report["feature_columns"] = list(X.columns)
        self.training_metadata = report

        return report

    def _temporal_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_config: Dict,
        params: Dict,
        n_splits: int,
    ) -> Dict:
        """Run TimeSeriesSplit cross-validation and return metrics per fold."""
        tss = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(tss.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model = self._create_model(target_config["type"], params)
            model.fit(X_tr, y_tr)

            metrics = self._evaluate_fold(
                model, X_te, y_te, target_config["type"]
            )
            metrics["fold"] = fold_idx
            metrics["train_size"] = len(X_tr)
            metrics["test_size"] = len(X_te)
            fold_metrics.append(metrics)

        # Aggregate
        metric_keys = [k for k in fold_metrics[0] if k not in ("fold", "train_size", "test_size")]
        aggregated = {}
        for key in metric_keys:
            values = [f[key] for f in fold_metrics if f[key] is not None]
            if values:
                aggregated[f"{key}_mean"] = round(np.mean(values), 4)
                aggregated[f"{key}_std"] = round(np.std(values), 4)

        return {
            "folds": fold_metrics,
            "aggregated": aggregated,
            "n_splits": n_splits,
        }

    def _evaluate_fold(self, model, X_test, y_test, task_type: str) -> Dict:
        """Evaluate a model on a single fold."""
        metrics = {}
        y_pred = model.predict(X_test)

        if task_type == "regression":
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 4)
            # How often we're within 0.5 goals
            metrics["within_0_5"] = round(float(np.mean(np.abs(y_test - y_pred) < 0.5)), 4)
            metrics["within_1"] = round(float(np.mean(np.abs(y_test - y_pred) < 1.0)), 4)
        else:
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                try:
                    if task_type == "binary":
                        metrics["brier"] = round(
                            float(brier_score_loss(y_test, y_proba[:, 1])), 4
                        )
                        metrics["auc"] = round(
                            float(roc_auc_score(y_test, y_proba[:, 1])), 4
                        )
                        metrics["log_loss"] = round(
                            float(log_loss(y_test, y_proba[:, 1])), 4
                        )
                    else:
                        metrics["log_loss"] = round(
                            float(log_loss(y_test, y_proba)), 4
                        )
                except Exception:
                    pass

            try:
                metrics["f1_weighted"] = round(
                    float(f1_score(y_test, y_pred, average="weighted")), 4
                )
            except Exception:
                pass

        return metrics

    def _calibrate_model(
        self, X, y, target_name, target_config, n_splits
    ):
        """Calibrate on last fold's test set to avoid data leakage."""
        tss = TimeSeriesSplit(n_splits=max(n_splits, 3))
        splits = list(tss.split(X))
        # Use last split: train on first part, calibrate on second
        train_idx, cal_idx = splits[-1]

        X_tr, X_cal = X.iloc[train_idx], X.iloc[cal_idx]
        y_tr, y_cal = y.iloc[train_idx], y.iloc[cal_idx]

        model = self._create_model(target_config["type"], self._default_params(target_config["type"]))
        model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_cal)
            if target_config["type"] == "binary" and proba.shape[1] == 2:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(proba[:, 1], y_cal)
                self.calibrators[target_name] = calibrator
                logger.info("Calibrated %s on %d samples", target_name, len(X_cal))

    # ------------------------------------------------------------------
    # 2. Hyperparameter Tuning
    # ------------------------------------------------------------------

    def _tune_hyperparams(
        self, X, y, task_type: str, n_splits: int, n_trials: int = 30
    ) -> Dict:
        """
        Random search over hyperparameter grid with temporal CV.
        Returns the best parameter set.
        """
        import random

        best_score = -float("inf")
        best_params = self._default_params(task_type)

        for trial in range(n_trials):
            params = {
                k: random.choice(v) for k, v in self.PARAM_GRID.items()
            }
            params["verbosity"] = 0
            params["random_state"] = 42

            try:
                tss = TimeSeriesSplit(n_splits=min(n_splits, 3))
                scores = []

                for train_idx, test_idx in tss.split(X):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                    model = self._create_model(task_type, params)
                    model.fit(X_tr, y_tr)

                    if task_type == "regression":
                        pred = model.predict(X_te)
                        score = -mean_absolute_error(y_te, pred)
                    else:
                        score = accuracy_score(y_te, model.predict(X_te))
                        if hasattr(model, "predict_proba"):
                            try:
                                proba = model.predict_proba(X_te)
                                if task_type == "binary":
                                    score = -brier_score_loss(y_te, proba[:, 1])
                                else:
                                    score = -log_loss(y_te, proba)
                            except Exception:
                                pass

                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    logger.info(
                        "Trial %d/%d: new best score %.4f",
                        trial + 1, n_trials, avg_score,
                    )
            except Exception as e:
                logger.debug("Trial %d failed: %s", trial + 1, e)

        return best_params

    # ------------------------------------------------------------------
    # 3. Model Versioning
    # ------------------------------------------------------------------

    def save_model(self, tag: str = None) -> str:
        """
        Save current model state with version metadata.
        Returns the filepath of the saved model.
        """
        if not JOBLIB_AVAILABLE:
            logger.error("joblib required for model serialization")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        tag = tag or "auto"
        filename = f"quant_model_{tag}_{timestamp}.pkl"
        filepath = self.model_dir / filename

        state = {
            "models": self.models,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "calibrators": self.calibrators,
            "training_metadata": self.training_metadata,
            "feature_columns": self.training_metadata.get("feature_columns", []),
            "saved_at": datetime.now().isoformat(),
            "tag": tag,
        }

        joblib.dump(state, filepath)

        # Create version record
        version_record = {
            "filename": filename,
            "tag": tag,
            "saved_at": state["saved_at"],
            "n_samples": self.training_metadata.get("n_samples", 0),
            "n_features": self.training_metadata.get("n_features", 0),
            "cv_summary": self._extract_cv_summary(),
            "fingerprint": self._compute_fingerprint(filepath),
        }

        self.version_history.append(version_record)
        self._save_version_history()
        self._cleanup_old_versions()

        logger.info("Model saved: %s", filename)
        return str(filepath)

    def load_model(self, filepath: str = None) -> bool:
        """
        Load a saved model. If no path given, loads the latest version.
        """
        if not JOBLIB_AVAILABLE:
            return False

        if filepath is None:
            filepath = self._find_latest_model()
            if filepath is None:
                logger.warning("No saved models found")
                return False

        filepath = Path(filepath)
        if not filepath.exists():
            logger.error("Model file not found: %s", filepath)
            return False

        try:
            state = joblib.load(filepath)
            self.models = state["models"]
            self.scaler = state["scaler"]
            self.label_encoder = state.get("label_encoder")
            self.calibrators = state.get("calibrators", {})
            self.training_metadata = state.get("training_metadata", {})
            logger.info("Loaded model: %s", filepath.name)
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def _find_latest_model(self) -> Optional[Path]:
        """Find the most recently saved quant model."""
        models = sorted(self.model_dir.glob("quant_model_*.pkl"))
        return models[-1] if models else None

    def _cleanup_old_versions(self):
        """Remove oldest model files beyond max_versions."""
        models = sorted(self.model_dir.glob("quant_model_*.pkl"))
        while len(models) > self.max_versions:
            oldest = models.pop(0)
            oldest.unlink()
            logger.info("Cleaned up old model: %s", oldest.name)

    def _compute_fingerprint(self, filepath) -> str:
        """SHA256 fingerprint of model file for integrity checks."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _extract_cv_summary(self) -> Dict:
        """Extract key CV metrics from training metadata."""
        summary = {}
        models_meta = self.training_metadata.get("models", {})
        for name, meta in models_meta.items():
            if isinstance(meta, dict) and "cv_results" in meta:
                agg = meta["cv_results"].get("aggregated", {})
                summary[name] = {
                    k: v for k, v in agg.items() if "_mean" in k
                }
        return summary

    def _load_version_history(self):
        """Load version history from disk."""
        path = self.model_dir / "quant_reports" / "model_versions.json"
        if path.exists():
            with open(path, "r") as f:
                self.version_history = json.load(f)

    def _save_version_history(self):
        """Persist version history."""
        report_dir = self.model_dir / "quant_reports"
        report_dir.mkdir(exist_ok=True)
        path = report_dir / "model_versions.json"
        with open(path, "w") as f:
            json.dump(self.version_history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # 4. Retraining Decisions
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        drift_report: Dict = None,
        performance_report: Dict = None,
        days_since_training: int = None,
    ) -> Dict:
        """
        Decide whether models should be retrained based on multiple signals.

        Signals checked:
        1. Feature drift (from DataQualityAnalyzer)
        2. Prediction performance decay (from PredictionMonitor)
        3. Time since last training
        4. New data volume

        Returns:
            Dict with decision, confidence, and reasons.
        """
        reasons = []
        urgency = 0  # 0-100

        # Signal 1: Feature drift
        if drift_report:
            drift_status = drift_report.get("status", "stable")
            if drift_status == "significant_drift":
                reasons.append("Significant feature drift detected")
                urgency += 40
            elif drift_status == "some_drift":
                reasons.append("Some feature drift detected")
                urgency += 15
            rec = drift_report.get("recommendation", "")
            if rec == "RETRAIN_IMMEDIATELY":
                urgency += 30

        # Signal 2: Performance decay
        if performance_report:
            trend = performance_report.get("trend", "stable")
            if trend == "declining":
                reasons.append("Model performance is declining")
                urgency += 35
            recent_brier = performance_report.get("recent_brier_score")
            if recent_brier and recent_brier > 0.3:
                reasons.append(f"High Brier score: {recent_brier:.3f}")
                urgency += 20

        # Signal 3: Staleness
        if days_since_training is not None:
            if days_since_training > 14:
                reasons.append(f"Model is {days_since_training} days old")
                urgency += 25
            elif days_since_training > 7:
                reasons.append(f"Model is {days_since_training} days old")
                urgency += 10

        # Decision
        if urgency >= 50:
            decision = "RETRAIN_NOW"
        elif urgency >= 25:
            decision = "RETRAIN_SOON"
        elif urgency > 0:
            decision = "MONITOR"
        else:
            decision = "NO_ACTION"

        return {
            "decision": decision,
            "urgency": min(urgency, 100),
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
        }

    def compare_versions(self, version_a: int = -1, version_b: int = -2) -> Dict:
        """
        Compare two model versions by their CV metrics.
        Defaults to comparing latest vs previous.
        """
        if len(self.version_history) < 2:
            return {"error": "Need at least 2 versions to compare"}

        a = self.version_history[version_a]
        b = self.version_history[version_b]

        comparison = {
            "version_a": a.get("filename"),
            "version_b": b.get("filename"),
            "improvements": [],
            "regressions": [],
        }

        cv_a = a.get("cv_summary", {})
        cv_b = b.get("cv_summary", {})

        for model_name in set(list(cv_a.keys()) + list(cv_b.keys())):
            metrics_a = cv_a.get(model_name, {})
            metrics_b = cv_b.get(model_name, {})

            for metric in set(list(metrics_a.keys()) + list(metrics_b.keys())):
                val_a = metrics_a.get(metric)
                val_b = metrics_b.get(metric)
                if val_a is not None and val_b is not None:
                    diff = val_a - val_b
                    is_better = diff > 0
                    # For metrics where lower is better
                    if any(neg in metric for neg in ("loss", "brier", "mae", "rmse")):
                        is_better = diff < 0

                    entry = {
                        "model": model_name,
                        "metric": metric,
                        "version_a": val_a,
                        "version_b": val_b,
                        "diff": round(diff, 4),
                    }
                    if is_better:
                        comparison["improvements"].append(entry)
                    elif abs(diff) > 0.001:
                        comparison["regressions"].append(entry)

        return comparison

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_model(self, task_type: str, params: Dict):
        """Create an XGBoost model with given params."""
        clean_params = {k: v for k, v in params.items() if k in self.PARAM_GRID or k in ("verbosity", "random_state", "use_label_encoder", "eval_metric")}
        clean_params.setdefault("verbosity", 0)
        clean_params.setdefault("random_state", 42)

        if task_type == "regression":
            return XGBRegressor(**clean_params)
        else:
            clean_params.setdefault("use_label_encoder", False)
            clean_params.setdefault("eval_metric", "logloss")
            return XGBClassifier(**clean_params)

    def _default_params(self, task_type: str) -> Dict:
        return {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "verbosity": 0,
            "random_state": 42,
        }

    def get_model_info(self) -> Dict:
        """Get a summary of the currently loaded models."""
        return {
            "loaded_models": list(self.models.keys()),
            "has_scaler": self.scaler is not None,
            "has_label_encoder": self.label_encoder is not None,
            "calibrated_models": list(self.calibrators.keys()),
            "training_metadata": {
                k: v for k, v in self.training_metadata.items()
                if k not in ("feature_columns",)
            },
            "total_versions": len(self.version_history),
            "latest_version": self.version_history[-1] if self.version_history else None,
        }
