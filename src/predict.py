import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / 'models' / 'pipeline.pkl'

# Load the full pipeline (encoder + scaler + classifier) once, at import time
pipeline = joblib.load(PIPELINE_PATH)


def predict_churn(engineered_df: pd.DataFrame) -> float:
    """
    Takes the engineered (raw, unencoded) single-row DataFrame produced by
    preprocess_input() and returns the probability of churn (class 1).

    Must be a DataFrame with the original column names, not a numpy array —
    the pipeline's ColumnTransformer selects columns by name, not position.
    """
    probability = pipeline.predict_proba(engineered_df)[0][1]
    return float(probability)
