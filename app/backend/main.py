import time
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Create app FIRST ──
app = FastAPI(
    title="Song Popularity Prediction API",
    description="Predicts song popularity from Spotify audio features.",
    version="1.0.0",
)

# ── CORS middleware IMMEDIATELY after app creation ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model and pipeline ──
MODEL_PATH = "models/best_model.pkl"
PIPELINE_PATH = "data/processed/preprocessing_pipeline.pkl"

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)


class SongFeatures(BaseModel):
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    key: int
    liveness: float
    loudness: float
    audio_mode: int
    speechiness: float
    tempo: float
    time_signature: int
    audio_valence: float
    song_duration_ms: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(f: SongFeatures):
    t0 = time.time()
    arr = np.array([[
        f.acousticness, f.danceability, f.energy, f.instrumentalness,
        f.key, f.liveness, f.loudness, f.audio_mode,
        f.speechiness, f.tempo, f.time_signature,
        f.audio_valence, f.song_duration_ms,
    ]])
    processed = pipeline.transform(arr)
    pred = model.predict(processed)[0]
    return {
        "predicted_popularity": round(float(pred), 2),
        "model_version": "local-best-model-v1",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }