"""Model registry: fetch the pinned model artifact, verify it, and describe it.

The deployed model is identified by the git commit of the training repository and
the SHA-256 of the artifact. The service refuses to start if the downloaded file
does not match the expected hash, so what runs in production is exactly what was
evaluated.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import requests

MODEL_REPO = "anascs12/pakistan-smog-forecast"
MODEL_COMMIT = os.environ.get("MODEL_COMMIT", "5d490e669fb42dd32c3ec1a183065d7c4ef6948f")
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "5e48d24d4e1806a004be669e7dd8914c375d69e7278f220bebfa63a57a498865")
CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "model_cache"))


def raw_url(path: str, commit: str = MODEL_COMMIT) -> str:
    return f"https://raw.githubusercontent.com/{MODEL_REPO}/{commit}/{path}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(path: str, expected_sha256: str | None = None, commit: str = MODEL_COMMIT) -> bytes:
    """Download a file from the training repo at the pinned commit, with an on-disk cache."""
    cached = CACHE_DIR / commit[:12] / path
    if cached.exists():
        data = cached.read_bytes()
    else:
        response = requests.get(raw_url(path, commit), timeout=120)
        response.raise_for_status()
        data = response.content
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    if expected_sha256 and sha256(data) != expected_sha256:
        cached.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: SHA-256 mismatch; refusing to load an unverified model")
    return data


@dataclass
class ModelCard:
    version: str
    commit: str
    sha256: str
    trained_through: str
    horizons: list[int]
    backtest_mae: dict[int, float]
    persistence_mae: dict[int, float]


@dataclass
class LoadedModel:
    models: dict
    card: ModelCard


def load(local_path: Path | None = None) -> LoadedModel:
    """Load the verified model bundle and its backtest metrics."""
    if local_path is not None:  # used by tests and offline development
        data = Path(local_path).read_bytes()
        if sha256(data) != MODEL_SHA256:
            raise RuntimeError("local model does not match MODEL_SHA256")
    else:
        data = fetch("models/forecaster.joblib", MODEL_SHA256)
    bundle = joblib.load(io.BytesIO(data))
    scores = pd.read_csv(io.BytesIO(fetch("data/processed/backtest_scores.csv")))
    mae = scores.pivot_table(index="horizon", columns="method", values="MAE")
    card = ModelCard(
        version=f"smog-gbr@{MODEL_COMMIT[:7]}",
        commit=MODEL_COMMIT,
        sha256=MODEL_SHA256,
        trained_through=bundle["trained_through"],
        horizons=sorted(int(h) for h in bundle["models"]),
        backtest_mae={int(h): round(float(v), 2) for h, v in mae["Gradient boosting"].items()},
        persistence_mae={int(h): round(float(v), 2) for h, v in mae["Persistence"].items()},
    )
    return LoadedModel(models=bundle["models"], card=card)
