"""Turn a validated request into forecasts, and build requests from live data."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from smog import data as smog_data
from smog import model as smog_model

from .registry import LoadedModel
from .schemas import DayInput, Forecast, PredictRequest, PredictResponse

TZ = ZoneInfo(smog_data.TIMEZONE)
LIVE_TTL_SECONDS = 3600


def request_frame(request: PredictRequest) -> pd.DataFrame:
    frame = pd.DataFrame([d.model_dump() for d in request.days])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["pm25"] = frame["pm25"].astype(float)
    return frame.assign(city=request.city)


def predict(loaded: LoadedModel, request: PredictRequest) -> tuple[PredictResponse, pd.DataFrame]:
    """Return the response plus the exact feature rows used (for monitoring)."""
    daily = request_frame(request)
    # The model one-hot encodes cities seen in training; every request must expose all four.
    cities = pd.DataFrame({"city": ["Lahore", "Peshawar", "Islamabad", "Karachi"]})
    forecasts, feature_rows = [], []
    for h in request.horizons:
        entry = loaded.models[h]
        frame = smog_model.make_features(daily, h)
        for city in cities["city"]:
            frame[f"is_{city.lower()}"] = int(city == request.city)
        row = frame[frame["date"] == pd.Timestamp(request.issue_day)]
        X = row[entry["columns"]].astype(float)
        if X.isna().any(axis=None):
            missing = list(X.columns[X.isna().iloc[0]])
            raise ValueError(f"cannot build features for horizon {h}; missing: {missing}")
        value = float(entry["model"].predict(X)[0])
        forecasts.append(Forecast(
            horizon=h, date=request.issue_day + timedelta(days=h), pm25=round(value, 1),
            category=str(smog_model.aqi_category([value]).iloc[0]),
        ))
        feature_rows.append(X.assign(horizon=h))
    response = PredictResponse(model_version=loaded.card.version, city=request.city,
                               issue_day=request.issue_day, forecasts=forecasts)
    return response, pd.concat(feature_rows, ignore_index=True)


def live_request(city: str, today: date | None = None) -> PredictRequest:
    """Build a request from Open-Meteo: observed PM2.5 up to yesterday, weather forecast ahead."""
    today = today or datetime.now(TZ).date()
    daily = smog_data.to_daily(smog_data.hourly_recent(city, past_days=10, forecast_days=4))
    # CAMS also returns its own forecast for today onward; never treat it as an observation.
    daily.loc[daily.index >= pd.Timestamp(today), "pm25"] = np.nan
    daily = daily.dropna(subset=["temp_mean", "temp_range", "rh_mean", "wind_mean", "blh_mean",
                                 "blh_night", "precip_sum"])
    days = [DayInput(date=d.date(), pm25=None if pd.isna(r["pm25"]) else float(r["pm25"]),
                     **{k: float(r[k]) for k in ["temp_mean", "temp_range", "rh_mean", "wind_mean",
                                                 "blh_mean", "blh_night", "precip_sum"]})
            for d, r in daily.iterrows()]
    return PredictRequest(city=city, issue_day=today - timedelta(days=1), days=days)


class LiveCache:
    """Live forecasts change at most hourly; cache them to protect the free upstream API."""

    def __init__(self, ttl: int = LIVE_TTL_SECONDS):
        self.ttl, self.items = ttl, {}

    def get(self, key):
        hit = self.items.get(key)
        return hit[1] if hit and time.monotonic() - hit[0] < self.ttl else None

    def put(self, key, value):
        self.items[key] = (time.monotonic(), value)
