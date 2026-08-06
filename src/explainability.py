import joblib
import shap
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / 'models' / 'pipeline.pkl'
BACKGROUND_PATH = BASE_DIR / 'models' / 'shap_background.pkl'

pipeline = joblib.load(PIPELINE_PATH)
_preprocessor = pipeline.named_steps['preprocess']
_clf = pipeline.named_steps['clf']

# LinearExplainer needs a representative background sample to compute expected
# values from — the app doesn't have access to X_train, so a small sample is
# exported once from the notebook and loaded here. See notebook cell to add.
_background = joblib.load(BACKGROUND_PATH)
_explainer = shap.LinearExplainer(_clf, _background)


def explain_customer(engineered_df: pd.DataFrame):
    """
    Returns (shap_values, base_value, feature_names) for one engineered
    customer row — real SHAP, computed the same way as the notebook's
    per-customer waterfall. Use this to build the "Why this prediction?"
    chart instead of whatever currently generates it.
    """
    transformed = _preprocessor.transform(engineered_df)
    feature_names = _preprocessor.get_feature_names_out().tolist()
    shap_values = _explainer.shap_values(transformed)[0]
    return shap_values, _explainer.expected_value, feature_names