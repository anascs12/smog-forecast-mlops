import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smog_service import registry


@pytest.fixture(scope="session")
def history() -> pd.DataFrame:
    """Real daily data from the training repo at the pinned commit (cached after first download)."""
    return pd.read_csv(io.BytesIO(registry.fetch("data/processed/daily.csv")), parse_dates=["date"])


def make_days(history: pd.DataFrame, city: str, issue_day: str, ahead: int = 3) -> list[dict]:
    issue = pd.Timestamp(issue_day)
    g = history[(history["city"] == city)
                & history["date"].between(issue - pd.Timedelta(days=6), issue + pd.Timedelta(days=ahead))]
    days = []
    for _, r in g.iterrows():
        day = {k: float(r[k]) for k in ["temp_mean", "temp_range", "rh_mean", "wind_mean", "blh_mean",
                                        "blh_night", "precip_sum"]}
        day["date"] = str(r["date"].date())
        day["pm25"] = float(r["pm25"]) if r["date"] <= issue else None
        days.append(day)
    return days


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB", str(tmp_path / "test.db"))
    import importlib

    from smog_service import app as app_module
    from smog_service import store
    importlib.reload(store)
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c
