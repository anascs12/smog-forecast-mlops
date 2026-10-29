from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from smog_service.monitoring import _retry, accuracy_status, psi, score, seasonal_reference


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(100, 20, 5000)
    assert psi(reference, rng.normal(100, 20, 2000)) < 0.02
    assert psi(reference, rng.normal(160, 20, 2000)) > 0.25


def test_psi_handles_tiny_or_empty_samples():
    assert np.isnan(psi(np.arange(5.0), np.arange(3.0)))
    assert np.isnan(psi(np.arange(100.0), np.array([])))


def test_psi_does_not_raise_false_alarms_on_a_few_points():
    """Regression test: one day of live data (4 cities) once produced PSI ≈ 5.8."""
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, 1000)
    assert np.isnan(psi(reference, rng.normal(50, 10, 4)))
    assert psi(reference, rng.normal(50, 10, 56)) < 0.25


def test_seasonal_reference_wraps_around_the_year():
    dates = pd.to_datetime(["2024-12-20", "2025-01-05", "2025-06-01"])
    frame = pd.DataFrame({"target_date": dates, "x": [1, 2, 3]})
    picked = seasonal_reference(frame, date(2026, 1, 1), window=15)
    assert picked["x"].tolist() == [1, 2]


def scored_frame(errors, persistence_errors, horizon=1, n=None):
    n = n or len(errors)
    days = pd.date_range("2026-10-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"horizon": horizon, "target_date": days, "abs_error": errors,
                         "persistence_abs_error": persistence_errors})


@pytest.mark.parametrize("live, persistence, expected", [
    (6.0, 8.0, "ok"),
    (9.0, 12.0, "warn"),     # 1.5× the backtest error
    (11.0, 14.0, "alert"),   # 1.83× the backtest error
    (6.5, 6.0, "alert"),     # worse than simply repeating today
])
def test_accuracy_status_thresholds(live, persistence, expected):
    scored = scored_frame([live] * 25, [persistence] * 25)
    [status] = accuracy_status(scored, {1: 6.0}, date(2026, 10, 26))
    assert status.status == expected


def test_no_alerts_until_enough_forecasts_are_scored():
    scored = scored_frame([50.0] * 5, [1.0] * 5)
    [status] = accuracy_status(scored, {1: 6.0}, date(2026, 10, 6))
    assert status.status == "collecting"


def test_score_joins_forecasts_with_truth_and_persistence():
    predictions = pd.DataFrame({"issue_day": ["2026-10-10"], "city": ["Lahore"], "horizon": [2],
                                "target_date": ["2026-10-12"], "pm25_pred": [50.0]})
    actuals = pd.DataFrame({"city": ["Lahore", "Lahore"], "date": ["2026-10-10", "2026-10-12"], "pm25": [40.0, 55.0]})
    row = score(predictions, actuals).iloc[0]
    assert row["abs_error"] == 5.0 and row["persistence_abs_error"] == 15.0


def test_retry_survives_a_timeout_then_succeeds():
    """Regression test: an Open-Meteo read timeout once failed the whole daily run."""
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.ReadTimeout("Read timed out. (read timeout=120)")
        return "ok"

    assert _retry(flaky, "test", waits=(0, 0, 0)) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_after_the_last_attempt():
    def down():
        raise requests.ConnectionError("unreachable")

    with pytest.raises(requests.ConnectionError):
        _retry(down, "test", waits=(0, 0))
