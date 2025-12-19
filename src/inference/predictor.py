from joblib import load
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "saved_models"

# Load trained models
intent_model = load(MODEL_DIR / "intent_model.joblib")
team_model = load(MODEL_DIR / "team_model.joblib")

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def is_unclear(probabilities, gap_threshold=0.15):
    """
    Check if prediction is unclear based on probability gap.
    """
    sorted_probs = np.sort(probabilities)[::-1]
    return (sorted_probs[0] - sorted_probs[1]) < gap_threshold


def predict_query(user_query: str):
    """
    Predict intent and team with uncertainty handling
    """

    embedding = sbert_model.encode([user_query])

    # ---------- Intent Prediction ----------
    intent_probs = intent_model.predict_proba(embedding)[0]
    intent_idx = np.argmax(intent_probs)
    intent = intent_model.classes_[intent_idx]

    if is_unclear(intent_probs):
        intent = "Unclear"

    # ---------- Team Prediction ----------
    team_probs = team_model.predict_proba(embedding)[0]
    team_idx = np.argmax(team_probs)
    team = team_model.classes_[team_idx]

    if is_unclear(team_probs):
        team = "Support Team"

    return intent, team
