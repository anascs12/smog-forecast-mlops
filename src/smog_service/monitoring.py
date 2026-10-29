"""Daily production monitoring: log forecasts, score them when the truth arrives,
and check the inputs for drift.

    python -m smog_service.monitoring      # run by .github/workflows/monitor.yml every morning

State lives in monitoring/*.csv (committed to the repo), so the full history of
what the model predicted and how it performed is auditable.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from smog import data as smog_data

from . import predictor, registry

ROOT = Path(".")
MON = ROOT / "monitoring"
CITIES = list(smog_data.CITIES)
DRIFT_FEATURES = ["pm25_lag0", "blh_night_target", "wind_mean_target", "rh_mean_target", "temp_mean_target"]

WARN_RATIO, ALERT_RATIO = 1.3, 1.6  # live MAE relative to backtest MAE
MIN_SCORED = 20                     # forecasts needed before accuracy alerts can fire
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25
SEASON_WINDOW_DAYS = 30             # drift reference: same time of year in training data
MIN_DRIFT_ROWS = 28                 # a week across four cities; PSI is meaningless on a handful of points


# ----------------------------------------------------------------------------- drift
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, min_current: int = MIN_DRIFT_ROWS) -> float:
    """Population Stability Index of `current` against `reference`, using reference deciles.

    Returns NaN when either sample is too small: with only a few current values most
    bins are empty and PSI explodes, which would raise false drift alarms.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) < bins or len(current) < min_current:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def seasonal_reference(training_features: pd.DataFrame, today: date, window: int = SEASON_WINDOW_DAYS) -> pd.DataFrame:
    """Training rows whose target date falls within ±window days of today's day of year."""
    doy = training_features["target_date"].dt.dayofyear
    distance = np.abs(((doy - pd.Timestamp(today).dayofyear + 183) % 366) - 183)
    return training_features[distance <= window]


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    rows = []
    for feature in DRIFT_FEATURES:
        value = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        level = ("insufficient data" if np.isnan(value) else "significant" if value > PSI_SIGNIFICANT
                 else "moderate" if value > PSI_MODERATE else "stable")
        rows.append({"feature": feature, "psi": None if np.isnan(value) else round(value, 3), "level": level,
                     "reference_median": round(float(reference[feature].median()), 1),
                     "current_median": round(float(current[feature].median()), 1) if len(current) else None})
    return rows


# -------------------------------------------------------------------------- accuracy
def score(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts with observed values; also score persistence for the same issues."""
    obs = actuals.rename(columns={"date": "target_date", "pm25": "actual"})
    scored = predictions.merge(obs, on=["city", "target_date"], how="inner")
    persist = actuals.rename(columns={"date": "issue_day", "pm25": "persistence"})
    scored = scored.merge(persist, on=["city", "issue_day"], how="left")
    scored["abs_error"] = (scored["pm25_pred"] - scored["actual"]).abs()
    scored["persistence_abs_error"] = (scored["persistence"] - scored["actual"]).abs()
    return scored


@dataclass
class HorizonStatus:
    horizon: int
    scored: int
    live_mae_30d: float | None
    persistence_mae_30d: float | None
    backtest_mae: float
    ratio: float | None
    status: str


def accuracy_status(scored: pd.DataFrame, backtest_mae: dict[int, float], today: date) -> list[HorizonStatus]:
    recent = scored[pd.to_datetime(scored["target_date"]) >= pd.Timestamp(today - timedelta(days=30))]
    out = []
    for h, expected in sorted(backtest_mae.items()):
        g = recent[recent["horizon"] == h]
        n = len(g)
        mae = float(g["abs_error"].mean()) if n else None
        pmae = float(g["persistence_abs_error"].mean()) if n and g["persistence_abs_error"].notna().any() else None
        ratio = mae / expected if mae is not None else None
        if n < MIN_SCORED:
            status = "collecting"
        elif ratio > ALERT_RATIO or (pmae is not None and mae > pmae):
            status = "alert"
        elif ratio > WARN_RATIO:
            status = "warn"
        else:
            status = "ok"
        out.append(HorizonStatus(h, n, None if mae is None else round(mae, 2),
                                 None if pmae is None else round(pmae, 2), expected,
                                 None if ratio is None else round(ratio, 2), status))
    return out


# ----------------------------------------------------------------------- network
RETRY_WAITS = (15, 45, 120)  # seconds between attempts


def _retry(fn, what: str, waits=RETRY_WAITS):
    """Call fn(), retrying when the network or Open-Meteo fails.

    smog.data retries HTTP errors (429/5xx) itself, but a timeout or dropped connection
    raises straight out of it; one slow Open-Meteo response once failed a whole daily run.
    """
    for attempt in range(len(waits) + 1):
        try:
            return fn()
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == len(waits):
                raise
            print(f"{what}: {type(exc).__name__}: {exc}; retrying in {waits[attempt]} s")
            time.sleep(waits[attempt])


# --------------------------------------------------------------------------- storage
def _load(name: str, parse=("issue_day", "target_date", "date")) -> pd.DataFrame:
    path = MON / name
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for col in parse:
        if col in frame:
            frame[col] = pd.to_datetime(frame[col]).dt.strftime("%Y-%m-%d")
    return frame


def _upsert(name: str, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    old = _load(name)
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    merged = merged.sort_values(keys).reset_index(drop=True)
    MON.mkdir(exist_ok=True)
    merged.to_csv(MON / name, index=False)
    return merged


def observed_actuals(today: date) -> pd.DataFrame:
    rows = []
    for city in CITIES:
        hourly = _retry(lambda city=city: smog_data.hourly_recent(city, past_days=14, forecast_days=1),
                        f"observations for {city}")
        daily = smog_data.to_daily(hourly)
        daily = daily[daily.index < pd.Timestamp(today)].dropna(subset=["pm25"])
        rows += [{"city": city, "date": d.strftime("%Y-%m-%d"), "pm25": round(float(v), 2)}
                 for d, v in daily["pm25"].items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- run
def run(today: date | None = None) -> dict:
    loaded = registry.load()
    today = today or datetime.now(predictor.TZ).date()
    issued_at = datetime.now(predictor.TZ).isoformat(timespec="minutes")

    pred_rows, input_rows = [], []
    for city in CITIES:
        response, features = _retry(lambda city=city: predictor.predict(loaded, predictor.live_request(city, today)),
                                    f"forecast for {city}")
        for f, (_, x) in zip(response.forecasts, features.iterrows(), strict=True):
            pred_rows.append({"issue_day": str(response.issue_day), "city": city, "horizon": f.horizon,
                              "target_date": str(f.date), "pm25_pred": f.pm25,
                              "model_version": response.model_version, "issued_at": issued_at})
            input_rows.append({"issue_day": str(response.issue_day), "city": city, "horizon": f.horizon,
                               "target_date": str(f.date), **{k: round(float(x[k]), 3) for k in DRIFT_FEATURES}})
    predictions = _upsert("predictions.csv", pd.DataFrame(pred_rows), ["issue_day", "city", "horizon"])
    inputs = _upsert("inputs.csv", pd.DataFrame(input_rows), ["issue_day", "city", "horizon"])
    actuals = _upsert("actuals.csv", observed_actuals(today), ["city", "date"])

    scored = score(predictions, actuals)
    accuracy = accuracy_status(scored, loaded.card.backtest_mae, today)

    training = pd.read_csv(io.BytesIO(_retry(lambda: registry.fetch("data/processed/daily.csv"), "training data")),
                           parse_dates=["date"])
    from smog import model as smog_model
    reference = seasonal_reference(smog_model.make_features(training, 1).dropna(subset=DRIFT_FEATURES), today)
    current = inputs[(inputs["horizon"] == 1)
                     & (pd.to_datetime(inputs["issue_day"]) >= pd.Timestamp(today - timedelta(days=14)))]
    drift = drift_report(reference, current)

    levels = [a.status for a in accuracy] + ["alert" if d["level"] == "significant" else "ok" for d in drift]
    overall = "alert" if "alert" in levels else "warn" if "warn" in levels else "ok"
    status = {
        "generated_at": issued_at, "today": str(today), "overall": overall,
        "model": asdict(loaded.card),
        "monitoring_since": predictions["issue_day"].min(),
        "forecasts_logged": len(predictions), "forecasts_scored": len(scored),
        "accuracy": [asdict(a) for a in accuracy], "drift": drift,
        "drift_reference": f"training days within ±{SEASON_WINDOW_DAYS} days of {today:%d %b} (n={len(reference)})",
    }
    (MON / "status.json").write_text(json.dumps(status, indent=1, default=str))
    from . import report
    report.write(status, predictions, actuals, scored, ROOT / "docs" / "index.html")
    print(f"overall={overall} logged={len(predictions)} scored={len(scored)}")
    return status


if __name__ == "__main__":
    run()
