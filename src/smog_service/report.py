"""Render the public monitoring dashboard (docs/index.html)."""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

API_URL = os.environ.get("PUBLIC_API_URL", "")
COLORS = {"ok": "#16A34A", "warn": "#D97706", "alert": "#DC2626", "collecting": "#6B7280",
          "stable": "#16A34A", "moderate": "#D97706", "significant": "#DC2626", "insufficient data": "#6B7280"}
CITY_COLORS = {"Lahore": "#E4572E", "Peshawar": "#4C7AF2", "Islamabad": "#29A36A", "Karachi": "#8E5BD9"}


def pill(text: str) -> str:
    return f'<span class="pill" style="background:{COLORS.get(text, "#6B7280")}">{html.escape(text)}</span>'


def forecast_chart(predictions: pd.DataFrame, actuals: pd.DataFrame) -> str:
    fig = go.Figure()
    for city, color in CITY_COLORS.items():
        a = actuals[actuals["city"] == city].sort_values("date").tail(30)
        fig.add_trace(go.Scatter(x=a["date"], y=a["pm25"], name=f"{city} actual", mode="lines",
                                 line=dict(color=color, width=2.4), legendgroup=city))
        p = predictions[(predictions["city"] == city) & (predictions["horizon"] == 1)].sort_values("target_date")
        if not p.empty:
            fig.add_trace(go.Scatter(x=p["target_date"], y=p["pm25_pred"], name=f"{city} 1-day forecast",
                                     mode="markers", marker=dict(color=color, size=9, symbol="diamond-open",
                                                                 line=dict(width=2)), legendgroup=city))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=40, r=20, t=50, b=40),
                      title="Observed PM2.5 (lines) and the deployed model's 1-day forecasts (diamonds)",
                      yaxis_title="µg/m³", legend=dict(orientation="h", y=-0.2, font=dict(size=11)))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def error_chart(scored: pd.DataFrame, backtest: dict) -> str:
    if scored.empty:
        return ('<p class="note">No forecast has reached its target date yet. Live accuracy appears here as '
                'forecasts are scored (from the day after monitoring starts), and alerts unlock after '
                '20 scored forecasts per horizon.</p>')
    fig = go.Figure()
    palette = {1: "#1D4ED8", 2: "#7C3AED", 3: "#DB2777"}
    for h, g in scored.groupby("horizon"):
        daily = g.groupby("target_date")["abs_error"].mean().sort_index()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.rolling(7, min_periods=1).mean(), mode="lines+markers",
                                 name=f"{h}-day live MAE (7-day rolling)", line=dict(color=palette[h], width=2.2)))
        fig.add_hline(y=backtest[str(h)] if str(h) in backtest else backtest[h], line_dash="dot",
                      line_color=palette[h], annotation_text=f"{h}-day backtest", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=20, t=50, b=40),
                      title="Live error vs. the error expected from backtesting", yaxis_title="MAE (µg/m³)",
                      legend=dict(orientation="h", y=-0.2))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def write(status: dict, predictions: pd.DataFrame, actuals: pd.DataFrame, scored: pd.DataFrame, path: Path) -> Path:
    card = status["model"]
    acc_rows = "".join(
        f"<tr><td>{a['horizon']} day{'s' if a['horizon'] > 1 else ''}</td><td>{pill(a['status'])}</td>"
        f"<td>{a['scored']}</td><td>{'–' if a['live_mae_30d'] is None else a['live_mae_30d']}</td>"
        f"<td>{a['backtest_mae']}</td><td>{'–' if a['persistence_mae_30d'] is None else a['persistence_mae_30d']}</td></tr>"
        for a in status["accuracy"])
    drift_rows = "".join(
        f"<tr><td><code>{d['feature']}</code></td><td>{pill(d['level'])}</td><td>{'–' if d['psi'] is None else d['psi']}</td>"
        f"<td>{d['reference_median']}</td><td>{'–' if d['current_median'] is None else d['current_median']}</td></tr>"
        for d in status["drift"])
    api = (f'<a href="{API_URL}/docs">Live API (interactive docs)</a> · ' if API_URL else "")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smog Forecast · Model Monitoring</title>
<style>
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:#F4F6FB; color:#1C2333; }}
  header {{ background:linear-gradient(120deg,#0F172A,#134E4A); color:#fff; padding:30px 20px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; }} header p {{ margin:4px 0; opacity:.9; }}
  header a {{ color:#99F6E4; }}
  main {{ max-width:1160px; margin:0 auto; padding:22px 16px 48px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
  .card, .panel {{ background:#fff; border-radius:14px; padding:16px; box-shadow:0 2px 10px rgba(15,23,42,.08); }}
  .card .k {{ font-size:12px; color:#5A6478; text-transform:uppercase; letter-spacing:.04em; }}
  .card .v {{ font-size:20px; font-weight:700; margin-top:6px; word-break:break-all; }}
  .panel {{ margin-top:18px; overflow-x:auto; }}
  .pill {{ color:#fff; border-radius:999px; padding:3px 10px; font-size:12.5px; font-weight:600; white-space:nowrap; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid #E5E7EB; text-align:left; }}
  .note {{ color:#5A6478; font-size:13.5px; line-height:1.55; }}
  code {{ background:#F1F5F9; padding:1px 5px; border-radius:4px; }}
</style></head>
<body>
<header>
  <h1>🛰️ Smog Forecast · Model Monitoring {pill(status['overall'])}</h1>
  <p>Production monitoring for the deployed PM2.5 forecaster: every forecast is logged, scored when the real value arrives, and its inputs are checked for drift.</p>
  <p style="font-size:13px">Updated {html.escape(status['generated_at'])} · {api}<a href="https://github.com/anascs12/smog-forecast-mlops">Code</a> · <a href="https://anascs12.github.io/pakistan-smog-forecast/">Public forecast</a></p>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="k">Deployed model</div><div class="v">{html.escape(card['version'])}</div></div>
    <div class="card"><div class="k">Artifact SHA-256</div><div class="v" style="font-size:13px">{card['sha256'][:16]}…</div></div>
    <div class="card"><div class="k">Trained through</div><div class="v">{card['trained_through']}</div></div>
    <div class="card"><div class="k">Monitoring since</div><div class="v">{status['monitoring_since']}</div></div>
    <div class="card"><div class="k">Forecasts logged / scored</div><div class="v">{status['forecasts_logged']} / {status['forecasts_scored']}</div></div>
  </div>
  <div class="panel">
    <h3 style="margin-top:0">Live accuracy (last 30 days)</h3>
    <table><thead><tr><th>Horizon</th><th>Status</th><th>Scored</th><th>Live MAE</th><th>Backtest MAE</th><th>Persistence MAE (live)</th></tr></thead>
    <tbody>{acc_rows}</tbody></table>
    <p class="note">Warn when live error exceeds 1.3× the backtest error; alert above 1.6×, or if the model does worse than simply repeating today's value. An alert automatically opens a GitHub issue.</p>
    {error_chart(scored, card['backtest_mae'])}
  </div>
  <div class="panel">{forecast_chart(predictions, actuals)}</div>
  <div class="panel">
    <h3 style="margin-top:0">Input drift (Population Stability Index)</h3>
    <table><thead><tr><th>Feature</th><th>Level</th><th>PSI</th><th>Reference median</th><th>Current median</th></tr></thead>
    <tbody>{drift_rows}</tbody></table>
    <p class="note">Current = the last 14 days of live inputs across all cities. Reference = {html.escape(status['drift_reference'])}, so normal seasonal change (such as the start of smog season) is not mistaken for drift. PSI below 0.10 is stable, 0.10–0.25 moderate, and above 0.25 significant.</p>
  </div>
  <div class="panel note">
    <b>About the data.</b> PM2.5 values are CAMS model estimates served by Open-Meteo, not ground sensors; "actual" means the CAMS value observed after the fact. Not an official health advisory.
  </div>
</main>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
