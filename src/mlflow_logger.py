"""
MLflow Logger Module

Centralized experiment tracking for all phases.
Logs metrics, parameters, artifacts, and models to MLflow.

Usage:
    from src.mlflow_logger import MLflowLogger

    logger = MLflowLogger(experiment_name="Phase 1: Imbalance Ablation")
    logger.log_params({"fp_cost": 75, "fn_cost_model": "dynamic"})
    logger.log_metrics({"cost_total": 57988, "threshold": 0.09})
    logger.end_run()
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from pathlib import Path


class MLflowLogger:
    """
    Wrapper around MLflow for consistent experiment logging.

    Attributes:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the current run
        tracking_uri: MLflow tracking server URI (local file system by default)
    """

    def __init__(
            self,
            experiment_name: str,
            run_name: Optional[str] = None,
            tracking_uri: str = "mlruns",  # Local directory
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the experiment (e.g., "Phase 1: Imbalance Ablation")
            run_name: Optional name for this specific run
            tracking_uri: Where to store MLflow data (default: local ./mlruns directory)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name

        # Set tracking URI (local file system or remote server)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        self.experiment_id = experiment_id
        mlflow.set_experiment(experiment_name)

        # Start a new run
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters (hyperparameters, configuration).

        Args:
            params: Dictionary of parameter name -> value

        Example:
            logger.log_params({
                "model": "LogisticRegression",
                "fp_cost": 75,
                "threshold": 0.09,
            })
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics (evaluation results).

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (for tracking metrics over time)

        Example:
            logger.log_metrics({
                "cost_total": 57988,
                "auc": 0.82,
                "brier": 0.1386,
            })
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact (file, model, plot).

        Args:
            local_path: Path to file to log
            artifact_path: Optional subdirectory in MLflow artifacts

        Example:
            logger.log_artifact("assets/churn_analysis.html", artifact_path="plots")
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """
        Log a sklearn model.

        Args:
            model: Fitted sklearn model or pipeline
            artifact_path: Where to store the model in MLflow

        Example:
            logger.log_model(final_pipeline, artifact_path="sklearn_model")
        """
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)

    def log_dict(self, data: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            data: Dictionary to log
            artifact_path: File name (without .json extension)

        Example:
            logger.log_dict({
                "segment_lifetime": {"Month-to-month": 36.3, "One year": 66.4},
                "offer_success_rate": 0.30,
            }, artifact_path="phase3_config")
        """
        import json

        import tempfile

        artifact_dir = Path(tempfile.gettempdir()) / "mlflow_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        file_path = artifact_dir / f"{artifact_path}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        mlflow.log_artifact(str(file_path), artifact_path="config")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, SCHEDULED, etc.)

        Example:
            logger.end_run()
        """
        mlflow.end_run(status=status)

    def get_run_uri(self) -> str:
        """
        Get the URI of the current run (useful for linking in reports).

        Returns:
            MLflow run URI
        """
        return f"mlruns/{self.experiment_id}/{self.run_id}"


# ===== PHASE 1: IMBALANCE ABLATION =====

def log_phase1_ablation_results():
    """
    Retroactively log Phase 1 imbalance ablation results.

    This is a one-time function to populate MLflow with historical results.
    Run once to populate experiment history.
    """
    logger = MLflowLogger(
        experiment_name="Phase 1: Imbalance Ablation Study",
        run_name="Historical - All Results"
    )

    # Phase 1 parameters (constant across all methods)
    logger.log_params({
        "dataset": "IBM Telco Churn",
        "test_size": 0.2,
        "random_state": 42,
        "fp_cost": 75,
        "fn_cost_model": "monthly_charges",
        "model": "LogisticRegression",
    })

    # Ablation results (from Phase 1 notebook)
    methods = {
        "No Treatment": {"cost": 95123, "threshold": None},
        "SMOTE": {"cost": 83453, "threshold": None},
        "SMOTENC": {"cost": 105309, "threshold": None},
        "class_weight_balanced": {"cost": 80803, "threshold": None},
        "threshold_moving": {"cost": 57988, "threshold": 0.09},  # WINNER
    }

    for method, results in methods.items():
        logger.log_metrics({
            f"{method}_cost": results["cost"],
        })

    logger.log_params({
        "best_method": "threshold_moving"
    })

    # Log summary metrics
    logger.log_metrics({
        "best_cost": 57988.0,
        "best_threshold": 0.09,
        "mcnemar_chi2": 126.49,
        "mcnemar_pvalue": 1.37e-29,
    })

    logger.end_run()
    print(f"✅ Logged Phase 1 ablation results: {logger.get_run_uri()}")


# ===== PHASE 2: ADVANCED MODELING =====

def log_phase2_results():
    """
    Log Phase 2 (Conformal Prediction, Survival, Counterfactuals, Fairness) results.
    """
    logger = MLflowLogger(
        experiment_name="Phase 2: Advanced Modeling",
        run_name="Conformal + Survival + Fairness"
    )

    logger.log_params({
        "confidence_level": 0.90,
        "conformal_method": "SplitConformalClassifier",
        "survival_method": "KaplanMeierFitter",
        "counterfactual_method": "DiCE (genetic)",
        "fairness_library": "fairlearn",
    })

    logger.log_metrics({
        # Conformal Prediction
        "conformal_coverage": 0.908,
        "conformal_single_label_rate": 0.731,
        # Fairness (Gender)
        "gender_demographic_parity_diff": 0.054,
        "gender_equalized_odds_diff": 0.063,
        # Fairness (Senior Citizen)
        "senior_demographic_parity_diff": 0.283,
        "senior_equalized_odds_diff": 0.312,
        "senior_actual_churn_rate": 0.4168,
        "nonsensior_actual_churn_rate": 0.2365,
    })

    logger.log_dict({
        "segment_lifetime": {
            "Month-to-month": 36.3,
            "One year": 66.4,
            "Two year": 71.5,
        },
        "conformal_coverage_target": 0.90,
    }, artifact_path="phase2_config")

    logger.end_run()
    print(f"✅ Logged Phase 2 results: {logger.get_run_uri()}")


# ===== PHASE 3: EXPECTED VALUE ENGINE =====

def log_phase3_ev_engine():
    """
    Log Phase 3 Expected Value engine configuration and results.
    """
    logger = MLflowLogger(
        experiment_name="Phase 3: Expected Value Engine",
        run_name="EV-Based Retention Strategy"
    )

    logger.log_params({
        "offer_success_rate": 0.30,
        "cost_of_offer": 75,
        "ev_high_value_threshold": 200,
        "ev_standard_threshold": 0,
        "horizon_months": 72,
    })

    logger.log_metrics({
        "high_value_tier_pct": 0.175,
        "standard_tier_pct": 0.207,
        "no_action_tier_pct": 0.618,
        "sanity_check_outlier_ev": 75.79,  # tenure=60, month-to-month
    })

    # Sensitivity analysis: offer success rate
    sensitivity_results = {
        "offer_sr_15pct": {"high_val": 0.009, "standard": 0.062, "no_action": 0.929},
        "offer_sr_20pct": {"high_val": 0.031, "standard": 0.119, "no_action": 0.850},
        "offer_sr_25pct": {"high_val": 0.074, "standard": 0.167, "no_action": 0.759},
        "offer_sr_30pct": {"high_val": 0.175, "standard": 0.207, "no_action": 0.618},  # baseline
        "offer_sr_35pct": {"high_val": 0.283, "standard": 0.232, "no_action": 0.485},
        "offer_sr_40pct": {"high_val": 0.383, "standard": 0.244, "no_action": 0.373},
    }

    logger.log_dict(sensitivity_results, artifact_path="phase3_sensitivity")

    logger.end_run()
    print(f"✅ Logged Phase 3 EV engine: {logger.get_run_uri()}")


# ===== PHASE 3.5: STRATEGY DEPLOYMENT =====

def log_phase3_5_deployment():
    """
    Log Phase 3.5 app deployment with new EV-based strategy.
    """
    logger = MLflowLogger(
        experiment_name="Phase 3.5: Strategy Deployment",
        run_name="EV-Based Streamlit App"
    )

    logger.log_params({
        "app_framework": "streamlit",
        "strategy_version": "v2_ev_based",
        "senior_citizen_targeting": True,
        "segment_lifetime_source": "kaplan_meier",
    })

    logger.log_metrics({
        "app_response_time_ms": 850,  # Estimate
        "senior_churn_rate": 0.4168,
        "nonsensior_churn_rate": 0.2365,
    })

    logger.end_run()
    print(f"✅ Logged Phase 3.5 deployment: {logger.get_run_uri()}")


# ===== PHASE 4: PRODUCTION READINESS =====

def log_phase4_start():
    """
    Initialize Phase 4 MLflow tracking.
    """
    logger = MLflowLogger(
        experiment_name="Phase 4: Production Readiness",
        run_name="MLflow + pytest + FastAPI + Docker"
    )

    logger.log_params({
        "mlflow_enabled": True,
        "pytest_coverage_target": 0.80,
        "fastapi_enabled": True,
        "docker_enabled": True,
        "ci_cd_enabled": True,
    })

    return logger


# ===== CONVENIENCE FUNCTIONS =====

def init_all_experiments():
    """
    Run all phase logging functions to populate MLflow with full history.
    """
    print("Initializing MLflow experiments...")
    log_phase1_ablation_results()
    log_phase2_results()
    log_phase3_ev_engine()
    log_phase3_5_deployment()
    print("\n✅ All experiments logged to mlruns/")
    print("   Run 'mlflow ui' to view results")


if __name__ == "__main__":
    # Populate MLflow with all phase results
    init_all_experiments()