"""FastAPI service for the Pakistan smog forecaster.

    uv run uvicorn smog_service.app:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.responses import RedirectResponse

from . import predictor, registry
from .schemas import City, PredictRequest, PredictResponse
from .store import PredictionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("smog_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    local = os.environ.get("MODEL_LOCAL_PATH")
    app.state.model = registry.load(Path(local) if local else None)
    app.state.store = PredictionStore()
    app.state.cache = predictor.LiveCache()
    app.state.started = time.time()
    log.info("loaded %s (trained through %s)", app.state.model.card.version, app.state.model.card.trained_through)
    yield


app = FastAPI(
    title="Pakistan Smog Forecast API",
    version="1.0.0",
    description="1–3 day daily-mean PM2.5 forecasts for Lahore, Peshawar, Islamabad and Karachi. "
                "PM2.5 inputs are CAMS model estimates (via Open-Meteo), which underestimate extreme peaks. "
                "Not an official health advisory.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    response.headers["x-latency-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["ops"])
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model.card.version,
            "uptime_s": round(time.time() - request.app.state.started)}


@app.get("/model", tags=["ops"])
def model_card(request: Request):
    """Which model is deployed, how it was verified, and how accurate it was in backtesting."""
    return request.app.state.model.card


@app.get("/metrics", tags=["ops"])
def metrics(request: Request):
    return request.app.state.store.stats()


def _serve(request: Request, req: PredictRequest, endpoint: str) -> PredictResponse:
    start = time.perf_counter()
    try:
        response, _ = predictor.predict(request.app.state.model, req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    latency = (time.perf_counter() - start) * 1000
    request.app.state.store.log(request.state.request_id, endpoint, response, latency)
    log.info("%s %s %s -> %s", endpoint, req.city, req.issue_day, [f.pm25 for f in response.forecasts])
    return response


@app.post("/predict", response_model=PredictResponse, tags=["forecast"])
def predict(request: Request, body: PredictRequest):
    """Forecast from your own inputs: 7 observed days of PM2.5 plus weather for each target day."""
    return _serve(request, body, "predict")


@app.get("/forecast/{city}", response_model=PredictResponse, tags=["forecast"])
def forecast(request: Request, city: Annotated[City, PathParam(description="One of the four supported cities")]):
    """Live forecast for today, tomorrow and the day after, from the latest Open-Meteo data."""
    cached = request.app.state.cache.get(city)
    if cached:
        return cached
    try:
        req = predictor.live_request(city)
    except (requests.RequestException, RuntimeError) as exc:
        log.warning("upstream data unavailable for %s: %s", city, exc)
        raise HTTPException(
            status_code=503, detail="Upstream weather/air-quality data is unavailable; try again later."
        ) from exc
    except ValueError as exc:  # incomplete upstream data fails schema validation
        raise HTTPException(status_code=503, detail=f"Upstream data incomplete: {exc}") from exc
    response = _serve(request, req, "forecast")
    request.app.state.cache.put(city, response)
    return response
