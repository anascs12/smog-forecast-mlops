import sqlite3

import pandas as pd
import pytest
import requests
from smog import model as smog_model

from smog_service import predictor
from smog_service.schemas import PredictRequest

from .conftest import make_days


def test_health_and_model_card(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["version"].startswith("smog-gbr@")
    assert card["horizons"] == [1, 2, 3]
    assert card["backtest_mae"]["1"] < card["persistence_mae"]["1"]


def test_predict_matches_the_training_code_exactly(client, history):
    """The service must reproduce what the training repo's own feature code predicts."""
    body = {"city": "Lahore", "issue_day": "2026-01-10", "days": make_days(history, "Lahore", "2026-01-10")}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    served = {f["horizon"]: f["pm25"] for f in response.json()["forecasts"]}

    model = client.app.state.model
    for h, entry in model.models.items():
        frame = smog_model.make_features(history, h)
        row = frame[(frame["city"] == "Lahore") & (frame["date"] == pd.Timestamp("2026-01-10"))]
        expected = float(entry["model"].predict(row[entry["columns"]].astype(float))[0])
        assert served[h] == pytest.approx(round(expected, 1), abs=0.05)


def test_predictions_are_logged(client, history):
    body = {"city": "Karachi", "issue_day": "2026-03-01", "horizons": [1, 2],
            "days": make_days(history, "Karachi", "2026-03-01")}
    rid = client.post("/predict", json=body, headers={"x-request-id": "test-123"}).headers["x-request-id"]
    assert rid == "test-123"
    with sqlite3.connect(client.app.state.store.path) as db:
        rows = db.execute("SELECT horizon, model_version FROM predictions WHERE request_id='test-123'").fetchall()
    assert sorted(r[0] for r in rows) == [1, 2]
    assert client.get("/metrics").json()["requests_logged"] >= 1


@pytest.mark.parametrize("mutate, message", [
    (lambda b: b["days"].pop(3), "consecutive"),
    (lambda b: b["days"][-4].update(pm25=None), "pm25 is required"),
    (lambda b: b["days"][0].update(rh_mean=140), "less than or equal to 100"),
    (lambda b: b.update(horizons=[4]), "Input should be 1, 2 or 3"),
    (lambda b: b.update(city="Multan"), "Input should be"),
    (lambda b: b["days"].append(dict(b["days"][0])), "duplicate"),
])
def test_invalid_requests_are_rejected(client, history, mutate, message):
    body = {"city": "Lahore", "issue_day": "2026-01-10", "days": make_days(history, "Lahore", "2026-01-10")}
    mutate(body)
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert message in response.text


def test_live_forecast_uses_cache(client, history, monkeypatch):
    calls = []
    def fake_live(city):
        calls.append(city)
        return PredictRequest(city=city, issue_day="2026-01-10", days=make_days(history, city, "2026-01-10"))
    monkeypatch.setattr(predictor, "live_request", fake_live)
    first = client.get("/forecast/Peshawar").json()
    second = client.get("/forecast/Peshawar").json()
    assert first == second and calls == ["Peshawar"]


def test_upstream_outage_returns_503(client, monkeypatch):
    def broken(city):
        raise requests.ConnectionError("open-meteo down")
    monkeypatch.setattr(predictor, "live_request", broken)
    response = client.get("/forecast/Islamabad")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
