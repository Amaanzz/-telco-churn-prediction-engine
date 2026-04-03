import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'model.pkl'

# Load model globally into memory upon application start
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def predict_churn(processed_data: np.ndarray) -> float:
    """
    Takes a preprocessed and scaled numpy array, returning the probability
    of the customer churning (Class 1).
    """
    # predict_proba returns an array of shape (n_samples, n_classes).
    # [0][1] extracts the probability for class 1 (Churn)
    probability = model.predict_proba(processed_data)[0][1]
    return float(probability)