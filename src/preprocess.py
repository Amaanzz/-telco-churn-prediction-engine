import pandas as pd
from typing import Dict, Any


def preprocess_input(user_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms raw user input into a single-row DataFrame carrying the same
    engineered features the training notebook produces. Encoding and scaling
    are no longer done here — pipeline.pkl does both internally, so this
    function's only job now is feature engineering.
    """
    df = pd.DataFrame([user_data])

    # Base data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # Feature engineering — must match the notebook exactly.
    # IMPORTANT: no custom `labels=` here. The notebook creates tenure_group as
    # pd.cut(df['tenure'], bins=[0,12,24,48,60,72]).astype(str), which produces
    # strings like "(0, 12]", not "0-12". The pipeline's OneHotEncoder was fit
    # on those interval strings — passing "0-12" gets silently treated as an
    # unknown category (handle_unknown='ignore') and encoded as all zeros,
    # quietly discarding tenure_group from every prediction.
    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 60, 72]
    ).astype(str)

    df['yearly_charges'] = df['MonthlyCharges'] * 12
    df['value_ratio'] = df['TotalCharges'] / (df['tenure'] + 1)

    # 70.35 is the training-set median MonthlyCharges, hardcoded so inference
    # doesn't need the training set just to compute one threshold. Re-verify
    # this against df['MonthlyCharges'].median() next time the notebook re-runs.
    df['high_value'] = (df['MonthlyCharges'] > 70.35).astype(int)

    return df
