from joblib import load
from sentence_transformers import SentenceTransformer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "saved_models"

intent_model = load(MODEL_DIR / "intent_model.joblib")
team_model = load(MODEL_DIR / "team_model.joblib")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def predict_query(user_query: str):
    embedding = sbert_model.encode([user_query])

    intent = intent_model.predict(embedding)[0]
    team = team_model.predict(embedding)[0]

    return intent, team
