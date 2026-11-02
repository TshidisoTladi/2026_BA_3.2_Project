"""
routing.py - reusable route-ranking logic (used by route_score.py and chatbot.py)

Combines the REGRESSION model (predicted volume) and the CLASSIFICATION model
(traffic state) with hazard penalties into one "adjusted travel time" per route.

The four routes are SIMULATED. In the real system, distance and speed would come
from the Google Maps / Mapbox API and the hazard flags from the accident, weather
and closure feeds.
"""
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

from features import (HISTORY_FEATURES, LIVE_COUNT_FEATURES, build_dataset,
                      chronological_split)

ROOT = Path(__file__).resolve().parent.parent

# ---- tunable weights (design choices - justify them in your report) -------------
MAX_VOLUME = 280          # highest 15-minute volume seen in the data
CONGESTION_WEIGHT = 0.5   # +50% travel time when the road is at maximum volume
ACCIDENT_PENALTY_MIN = 15
WEATHER_PENALTY_MIN = 5   # per weather-severity level (0-3)

# ---- SIMULATED candidate routes ----------------------------------------------------
# traffic_pct: how busy this segment is right now (0.9 = among the busiest 10% of
# readings, 0.3 = fairly quiet). Its traffic history is borrowed from REAL test rows.
ROUTES = [
    {"key": "A", "name": "Direct highway", "km": 42, "speed": 80, "traffic_pct": 0.90},
    {"key": "B", "name": "Ring road",      "km": 48, "speed": 80, "traffic_pct": 0.35},
    {"key": "C", "name": "Local roads",    "km": 39, "speed": 50, "traffic_pct": 0.50},
    {"key": "D", "name": "Arterial route", "km": 45, "speed": 70, "traffic_pct": 0.70},
]

_reg = joblib.load(ROOT / "models" / "regression_model.joblib")
_clf = joblib.load(ROOT / "models" / "classification_model.joblib")
_, _test = chronological_split(build_dataset())
_pool = _test.sort_values("total")


def scenario_row(hour: int) -> pd.Series:
    """A typical weekday reading at the requested hour (from the unseen test days)."""
    rows = _test[(_test["hour"] == hour) & (_test["day_of_week"] < 5)]
    return rows.iloc[0]


def forecast_hour(hour: int) -> dict:
    """Predicted traffic volume + traffic state for a typical weekday hour."""
    frame = pd.DataFrame([scenario_row(hour)])
    volume = float(_reg["model"].predict(frame[_reg["features"]])[0])
    label = _clf["labels"][int(_clf["model"].predict(frame[_clf["features"]])[0])]
    return {"hour": hour, "volume": round(volume), "state": label,
            "rush_hour": bool(frame["is_rush_hour"].iloc[0])}


def rank_routes(hour: int = 17, conditions: Optional[dict] = None) -> pd.DataFrame:
    """
    Rank all routes at the given hour.
    conditions: {"accident": {"A": 1}, "closed": {"C": 1}, "weather": 2}
    """
    cond = conditions or {}
    accidents = cond.get("accident", {})
    closed = cond.get("closed", {})
    weather = int(cond.get("weather", 0))

    scenario = scenario_row(hour)
    rows = []
    for r in ROUTES:
        borrowed = _pool.iloc[int(r["traffic_pct"] * (len(_pool) - 1))]
        x = scenario.copy()
        for col in HISTORY_FEATURES + LIVE_COUNT_FEATURES:
            x[col] = borrowed[col]
        acc, clo = int(accidents.get(r["key"], 0)), int(closed.get(r["key"], 0))
        x["accidents"], x["road_closed"], x["weather_severity"] = acc, clo, weather
        frame = pd.DataFrame([x])

        volume = float(_reg["model"].predict(frame[_reg["features"]])[0])
        label = _clf["labels"][int(_clf["model"].predict(frame[_clf["features"]])[0])]

        base_time = r["km"] / r["speed"] * 60
        congestion = base_time * CONGESTION_WEIGHT * volume / MAX_VOLUME
        penalty = ACCIDENT_PENALTY_MIN * acc + WEATHER_PENALTY_MIN * weather
        rows.append({
            "route": f"{r['key']}  {r['name']}", "key": r["key"], "km": r["km"],
            "free_flow_min": round(base_time), "pred_volume": round(volume),
            "traffic_state": label, "accident": acc, "closed": clo, "weather": weather,
            "congestion_min": round(congestion), "penalty_min": penalty,
            "adjusted_min": None if clo else round(base_time + congestion + penalty),
        })
    return pd.DataFrame(rows)


def best_route(table: pd.DataFrame):
    """Return (recommended row, shortest-open-by-distance row) or (None, None)."""
    open_routes = table[table["closed"] == 0]
    if open_routes.empty:
        return None, None
    best = open_routes.loc[open_routes["adjusted_min"].idxmin()]
    shortest = open_routes.loc[open_routes["km"].idxmin()]
    return best, shortest
