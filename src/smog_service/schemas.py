"""Request and response schemas. Validation rejects physically impossible inputs
before they ever reach the model."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator

City = Literal["Lahore", "Peshawar", "Islamabad", "Karachi"]
OBSERVED_DAYS = 7  # lags up to 6 days plus a 7-day mean


class DayInput(BaseModel):
    date: date
    pm25: float | None = Field(None, ge=0, le=1500, description="Daily mean PM2.5 (µg/m³); null for future days")
    temp_mean: float = Field(ge=-40, le=60, description="°C")
    temp_range: float = Field(ge=0, le=40, description="Daily max − min temperature, °C")
    rh_mean: float = Field(ge=0, le=100, description="Relative humidity, %")
    wind_mean: float = Field(ge=0, le=200, description="10 m wind speed, km/h")
    blh_mean: float = Field(ge=0, le=6000, description="Boundary layer height, m")
    blh_night: float = Field(ge=0, le=6000, description="Mean boundary layer height 00–06 h, m")
    precip_sum: float = Field(ge=0, le=500, description="mm")


class PredictRequest(BaseModel):
    city: City
    issue_day: date = Field(description="Last day with an observed PM2.5 value")
    horizons: list[Literal[1, 2, 3]] = Field(default=[1, 2, 3], min_length=1)
    days: list[DayInput] = Field(min_length=OBSERVED_DAYS + 1, max_length=60)

    @model_validator(mode="after")
    def check_days(self) -> PredictRequest:
        by_date = {d.date: d for d in self.days}
        if len(by_date) != len(self.days):
            raise ValueError("days contain duplicate dates")
        ordered = sorted(by_date)
        if any((b - a).days != 1 for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("days must be consecutive")
        history = [self.issue_day - timedelta(days=i) for i in range(OBSERVED_DAYS)]
        missing = [str(d) for d in history if d not in by_date or by_date[d].pm25 is None]
        if missing:
            raise ValueError(f"pm25 is required for the {OBSERVED_DAYS} days ending at issue_day; missing: {missing}")
        targets = [self.issue_day + timedelta(days=h) for h in self.horizons]
        absent = [str(d) for d in targets if d not in by_date]
        if absent:
            raise ValueError(f"weather is required for every target day; missing: {absent}")
        return self


class Forecast(BaseModel):
    horizon: int
    date: date
    pm25: float
    category: str


class PredictResponse(BaseModel):
    model_version: str
    city: City
    issue_day: date
    forecasts: list[Forecast]
