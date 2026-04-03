import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any

# Suppress harmless sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'


def preprocess_input(user_data: Dict[str, Any]) -> np.ndarray:
    """
    Transforms raw user input into a fully engineered, encoded, and scaled numpy array
    using the exact memory of the trained scaler to guarantee alignment.
    """
    # 1. Load ONLY the scaler
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 2. EXTRACT TRUTH FROM SCALER MEMORY
    # This guarantees we never get a column mismatch error again
    model_columns = scaler.feature_names_in_

    # 3. Convert user dictionary to a single-row DataFrame
    df = pd.DataFrame([user_data])

    # 4. Base Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # 5. Replicate Feature Engineering
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-12', '13-24', '25-48', '49-60', '61-72']
    ).astype(str)

    df['yearly_charges'] = df['MonthlyCharges'] * 12
    df['value_ratio'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['high_value'] = (df['MonthlyCharges'] > 70.35).astype(int)

    # 6. One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 7. Feature Alignment
    # We force the new data to perfectly match the scaler's memory
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)

    # 8. Scale as a pure NumPy Array
    # .to_numpy() strips the pandas metadata, bypassing the strict sklearn naming error
    scaled_array = scaler.transform(df_aligned.to_numpy())

    return scaled_array