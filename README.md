# 🛰️ Smog Forecast MLOps: Serving and Monitoring a Model in Production

**Taking the [Pakistan Smog Forecast](https://github.com/anascs12/pakistan-smog-forecast) model from a notebook to a deployed, tested and monitored service: a FastAPI API in Docker, with daily monitoring of live accuracy and data drift that opens a GitHub issue when something goes wrong.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-24%20passing-2EA44F)
[![CI](https://github.com/anascs12/smog-forecast-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/anascs12/smog-forecast-mlops/actions/workflows/ci.yml)
[![Daily monitoring](https://github.com/anascs12/smog-forecast-mlops/actions/workflows/monitor.yml/badge.svg)](https://github.com/anascs12/smog-forecast-mlops/actions/workflows/monitor.yml)

> 📈 **[Live monitoring dashboard](https://anascs12.github.io/smog-forecast-mlops/)** · 🌫️ **[Public forecast](https://anascs12.github.io/pakistan-smog-forecast/)**

## Why this project
A model that works in a notebook isn't yet useful. This repository adds what production needs: a **reproducible, verified model artifact**, an **API with input validation**, a **container**, **CI**, and **monitoring that measures real-world accuracy**. Smog forecasts are ideal for this, because the true answer arrives a day later, so live accuracy is measured rather than assumed.

## Architecture

```
pakistan-smog-forecast (training repo)            smog-forecast-mlops (this repo)
───────────────────────────────────────           ─────────────────────────────────────────────────
models/forecaster.joblib  @ commit 5d490e6  ──▶  registry: download from the pinned commit,
data/processed/backtest_scores.csv              verify SHA-256, refuse to load on mismatch
                                                         │
                                                         ▼
                              FastAPI service (Docker, non-root, health check)
                              /health  /model  /metrics  /forecast/{city}  /predict
                              pydantic validation · request IDs · prediction log (SQLite)
                                                         │
          Open-Meteo (CAMS PM2.5 + weather) ◀───────────┤
                                                         ▼
                 GitHub Action, daily 07:00 PKT: issue forecasts → log them → fetch what actually
                 happened → score → drift check (PSI vs. same season) → dashboard → issue on alert
```

## What's inside

| Concern | How it's handled |
|---|---|
| **Reproducibility** | The model package and the artifact are pinned to the exact training commit; the artifact's **SHA-256 is verified** before loading, both at image build time and at startup. |
| **Training–serving parity** | The service reuses the training repo's own feature code, and a test checks that served predictions equal the training code's predictions. Live forecasts matched the published forecast **to the decimal**. |
| **Input validation** | Pydantic schemas reject impossible inputs (humidity above 100%, missing PM2.5 history, non-consecutive days, unknown cities) with a clear `422`. |
| **Resilience** | An upstream outage returns a `503` with a clear message instead of a crash; live forecasts are cached for an hour to protect the free upstream API. |
| **Observability** | Every response carries `x-request-id` and `x-latency-ms`; every prediction is logged with its model version; `/metrics` reports p50/p95 latency. |
| **Container** | Multi-stage build with the verified model baked in, a slim runtime, a **non-root user** and a `HEALTHCHECK`. |
| **CI** | Lint (ruff), 24 tests, Docker build, then a **smoke test of the running container** (`/health`, `/model`, non-root check). |
| **Live accuracy monitoring** | Forecasts are logged daily and scored once the real values arrive; 30-day live MAE is compared with the **backtest MAE** and with **persistence**. Warn above 1.3×, alert above 1.6× or if the model loses to persistence. Alerts start after 20 scored forecasts. |
| **Drift monitoring** | PSI on key inputs (PM2.5, night-time mixing height, wind, humidity, temperature), compared with training data from the **same time of year**, so the start of smog season isn't mistaken for drift. At least a week of inputs is needed before drift is judged. |
| **Alerting** | An alert automatically opens a **GitHub issue** with the details (without duplicating an open one). |

## API

```bash
uv run uvicorn smog_service.app:app        # http://127.0.0.1:8000/docs
```

```bash
curl localhost:8000/forecast/Lahore
```
```json
{"model_version": "smog-gbr@5d490e6", "city": "Lahore", "issue_day": "2026-10-10",
 "forecasts": [{"horizon": 1, "date": "2026-10-11", "pm25": 38.9, "category": "Unhealthy for sensitive groups"},
               {"horizon": 2, "date": "2026-10-12", "pm25": 45.9, "category": "Unhealthy for sensitive groups"},
               {"horizon": 3, "date": "2026-10-13", "pm25": 57.1, "category": "Unhealthy"}]}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, with the deployed model version |
| `GET /model` | Model card: commit, SHA-256, training cut-off, backtest and persistence MAE |
| `GET /metrics` | Logged requests and p50/p95 prediction latency |
| `GET /forecast/{city}` | Live 1–3 day forecast from the latest data (cached for 1 h) |
| `POST /predict` | Forecast from your own inputs: 7 observed days plus the target days' weather |

## Run it

```bash
uv sync
uv run pytest                                   # 24 tests
uv run ruff check .
uv run python -m smog_service.monitoring        # one monitoring cycle → monitoring/, docs/index.html
docker build -t smog-forecast-api . && docker run -p 7860:7860 smog-forecast-api
```

## Things I fixed along the way
- **A false drift alarm on day one:** with a single day of live data (4 rows), PSI came out around 5.8 on every feature, because most bins were empty. Drift now needs at least a week of inputs, and a regression test guards it.
- **Seasonal false positives:** comparing October inputs with the whole year would flag drift every autumn, so the reference is the same calendar window in the training years.

## Limitations
- PM2.5 "actuals" are CAMS model estimates (via Open-Meteo), not ground sensors.
- Monitoring state is stored as CSV files in the repository, which is simple and auditable but not suited to high traffic.
- The SQLite prediction log inside a container is ephemeral; durable monitoring comes from the daily job.
- The free upstream API has rate limits; the service caches live forecasts but cannot guarantee availability.

Code: MIT licence. Built by **[Anas Khan](https://www.linkedin.com/in/anas-khan-ds/)**, Data Science student at IMSciences Peshawar.
