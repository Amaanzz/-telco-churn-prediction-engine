"""
FastAPI REST API for Telco Churn Prediction

Endpoints:
- POST /predict — Get churn probability + EV tier + retention strategy + SHAP explainability
- GET /health — Health check + model info
- GET /metrics — Model performance metrics

Usage:
    from src.api import app
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

Then:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{...customer_data...}'
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from src.preprocess import preprocess_input
from src.predict import predict_churn
from src.strategy import generate_retention_strategy_v2
from src.explainability import explain_customer

# ===== SETUP =====
app = FastAPI(
    title="Telco Churn Prediction API",
    description="AI-powered customer retention intelligence system",
    version="1.0.0",
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== MODELS (Pydantic) =====

class CustomerInput(BaseModel):
    """Customer data for churn prediction."""

    gender: str = Field(..., description="Gender: Male or Female")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior citizen: 0 or 1")
    partner: str = Field(..., description="Has partner: Yes or No")
    dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure: int = Field(..., ge=0, le=72, description="Tenure in months (0-72)")
    phone_service: str = Field(..., description="Phone service: Yes or No")
    multiple_lines: str = Field(..., description="Multiple lines: Yes, No, or No phone service")
    internet_service: str = Field(..., description="Internet service: DSL, Fiber optic, or No")
    online_security: str = Field(..., description="Online security: Yes, No, or No internet service")
    online_backup: str = Field(..., description="Online backup: Yes, No, or No internet service")
    device_protection: str = Field(..., description="Device protection: Yes, No, or No internet service")
    tech_support: str = Field(..., description="Tech support: Yes, No, or No internet service")
    streaming_tv: str = Field(..., description="Streaming TV: Yes, No, or No internet service")
    streaming_movies: str = Field(..., description="Streaming movies: Yes, No, or No internet service")
    contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    paperless_billing: str = Field(..., description="Paperless billing: Yes or No")
    payment_method: str = Field(...,
                                description="Payment method: Electronic check, Mailed check, Bank transfer, or Credit card")
    monthly_charges: float = Field(..., ge=15.0, le=120.0, description="Monthly charges in dollars")
    total_charges: float = Field(..., ge=0.0, le=9000.0, description="Total charges in dollars")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 24,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "One year",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 65.0,
                "total_charges": 1560.0,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with the PascalCase keys preprocess_input() expects."""
        return {
            'gender': self.gender,
            'SeniorCitizen': self.senior_citizen,
            'Partner': self.partner,
            'Dependents': self.dependents,
            'tenure': self.tenure,
            'PhoneService': self.phone_service,
            'MultipleLines': self.multiple_lines,
            'InternetService': self.internet_service,
            'OnlineSecurity': self.online_security,
            'OnlineBackup': self.online_backup,
            'DeviceProtection': self.device_protection,
            'TechSupport': self.tech_support,
            'StreamingTV': self.streaming_tv,
            'StreamingMovies': self.streaming_movies,
            'Contract': self.contract,
            'PaperlessBilling': self.paperless_billing,
            'PaymentMethod': self.payment_method,
            'MonthlyCharges': self.monthly_charges,
            'TotalCharges': self.total_charges,
        }


class ExplainabilityResponse(BaseModel):
    """SHAP explainability payload for a single prediction."""

    base_value: float = Field(..., description="SHAP base value (expected model output)")
    feature_names: List[str] = Field(..., description="Feature names, aligned with shap_values")
    shap_values: List[float] = Field(..., description="Per-feature SHAP contribution to this prediction")


class PredictionResponse(BaseModel):
    """Prediction response with churn risk, retention strategy, and explainability."""

    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_percentage: str = Field(..., description="Churn probability as percentage")
    ev_tier: str = Field(..., description="EV tier: high_value, standard, or no_action")
    expected_value: float = Field(..., description="Expected value of intervention in dollars")
    retention_action: str = Field(..., description="Recommended retention action")
    senior_citizen_alert: bool = Field(..., description="Whether customer is senior citizen")
    customer_value: str = Field(..., description="Customer value: High or Standard")
    explainability: Optional[ExplainabilityResponse] = Field(
        None, description="SHAP feature contributions for this prediction, if available"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status: ok or error")
    model_version: str = Field(..., description="Model version identifier")
    pipeline_loaded: bool = Field(..., description="Whether pipeline.pkl is loaded")
    timestamp: str = Field(..., description="Current timestamp")


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    auc_roc: float = Field(..., description="AUC-ROC score on test set")
    brier_score: float = Field(..., description="Brier score (calibration)")
    threshold_optimal: float = Field(..., description="Cost-optimal decision threshold")
    conformal_coverage: float = Field(..., description="Conformal prediction coverage")
    test_set_size: int = Field(..., description="Test set sample count")


# ===== ROUTES =====

@app.get("/", tags=["Info"])
def root():
    """API root endpoint with documentation."""
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        pipeline_path = Path(__file__).parent.parent / "models" / "pipeline.pkl"
        pipeline_loaded = pipeline_path.exists()

        if not pipeline_loaded:
            logger.warning("Pipeline file not found")

        from datetime import datetime

        return HealthResponse(
            status="ok",
            model_version="Phase 3.5 (EV-based strategy)",
            pipeline_loaded=pipeline_loaded,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
def model_metrics() -> MetricsResponse:
    """Get model performance metrics."""
    return MetricsResponse(
        auc_roc=0.822,
        brier_score=0.1386,
        threshold_optimal=0.09,
        conformal_coverage=0.908,
        test_set_size=1407,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput) -> PredictionResponse:
    """
    Predict churn, generate retention strategy, and compute SHAP explainability.

    Returns:
        PredictionResponse with churn probability, EV tier, action, and
        per-feature SHAP contributions (when explainability succeeds).
    """
    try:
        # Convert to dict (PascalCase) and preprocess
        customer_dict = customer.to_dict()
        engineered_df = preprocess_input(customer_dict)

        # Predict churn probability
        probability = predict_churn(engineered_df)

        # Generate EV-based retention strategy
        strategy = generate_retention_strategy_v2(
            probability=probability,
            monthly_charges=customer.monthly_charges,
            contract_type=customer.contract,
            tenure=customer.tenure,
            is_senior_citizen=(customer.senior_citizen == 1),
        )

        # Determine customer value
        customer_value = "High" if customer.monthly_charges > 70.35 else "Standard"

        # SHAP explainability — wrapped so a SHAP failure never breaks the
        # core prediction response, it just omits the explainability field
        explainability = None
        try:
            shap_values, base_value, feature_names = explain_customer(engineered_df)
            explainability = ExplainabilityResponse(
                base_value=float(base_value),
                feature_names=feature_names,
                shap_values=[float(v) for v in shap_values],
            )
        except Exception as shap_error:
            logger.warning(f"SHAP explainability failed: {str(shap_error)}")

        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_percentage=f"{probability * 100:.1f}%",
            ev_tier=strategy['tier'],
            expected_value=strategy['expected_value'],
            retention_action=strategy['action'],
            senior_citizen_alert=strategy['senior_warning'],
            customer_value=customer_value,
            explainability=explainability,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline error: {str(e)}"
        )


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(customers: List[CustomerInput]) -> List[Dict[str, Any]]:
    """Predict churn for multiple customers (batch)."""
    results = []
    for customer in customers:
        try:
            result = predict(customer)
            results.append(result.dict())
        except HTTPException as e:
            results.append({
                "error": True,
                "detail": str(e.detail),
                "status_code": e.status_code,
            })

    return results


# ===== ERROR HANDLERS =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "error": True,
        "status_code": exc.status_code,
        "message": exc.detail,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": True,
        "status_code": 500,
        "message": "Internal server error",
    }


# ===== MAIN =====

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Telco Churn Prediction API...")
    logger.info("API Documentation: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )