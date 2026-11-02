"""
features.py - shared data loading + feature engineering (Step 3: Designing the Model)

Both models (regression + classification) use the SAME feature table so that
their outputs can be combined later when ranking routes.

Design rules followed here:
  REGRESSION  (forecast the traffic volume "Total"):
     - the live vehicle counts at time t are NOT used, because they add up to Total
       (that would be data leakage);
     - only lagged history is used (t-1, t-2, t-4, same time yesterday);
     - conditions a live system already knows for that interval (time, weather
       forecast, accident alert, road-closure alert) are allowed.
  CLASSIFICATION (label the current road state low/normal/high/heavy):
     - the live vehicle counts ARE allowed, because a real system streams them
       from the traffic API; this is "mapping" the current state of a segment.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Dataset.csv"

STEPS_PER_DAY = 96  # the data is recorded every 15 minutes -> 96 rows per day

# Ordinal order of the traffic classes, sorted by their average traffic volume
SITUATION_ORDER = {"low": 0, "normal": 1, "high": 2, "heavy": 3}
SITUATION_LABELS = ["low", "normal", "high", "heavy"]

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

REGRESSION_TARGET = "total"
CLASSIFICATION_TARGET = "situation_code"

TIME_FEATURES = ["tod_sin", "tod_cos", "hour", "day_of_week", "is_weekend", "is_rush_hour"]
HAZARD_FEATURES = ["weather_severity", "accidents", "road_closed"]
HISTORY_FEATURES = ["total_lag1", "total_lag2", "total_lag4", "total_lag96", "total_roll4"]
LIVE_COUNT_FEATURES = ["carcount", "bikecount", "buscount", "truckcount", "total"]

# Regression = forecast, so it only sees context + past traffic
REGRESSION_FEATURES = TIME_FEATURES + HAZARD_FEATURES + HISTORY_FEATURES
# Classification = label the current state, so it also sees the live counts
CLASSIFICATION_FEATURES = TIME_FEATURES + HAZARD_FEATURES + LIVE_COUNT_FEATURES


def weather_severity(text: str) -> int:
    """Convert the free-text weather description into a 0-3 safety score."""
    t = str(text).lower()
    if any(k in t for k in ["blizzard", "heavy", "freezing", "ice pellets",
                            "thunder", "blowing snow"]):
        score = 3
    elif any(k in t for k in ["moderate", "snow", "sleet", "fog", "shower"]):
        score = 2
    elif any(k in t for k in ["rain", "drizzle", "mist"]):
        score = 1
    else:  # Sunny, Clear, Partly cloudy, Cloudy, Overcast
        score = 0
    if "possible" in t:  # "Patchy rain possible" is less serious than actual rain
        score = min(score, 1)
    return score


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and tidy the column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # -> time, date, day_of_the_week, carcount, bikecount, buscount, truckcount,
    #    total, traffic_situation, weather, accidents, road_closed
    return df


def build_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return a model-ready table (features + both targets), ordered by time."""
    df = load_raw(path)

    # ---- time features -------------------------------------------------
    t = pd.to_datetime(df["time"].str.strip(), format="%I:%M:%S %p")
    df["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    df["hour"] = t.dt.hour
    df["tod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["day_of_week"] = df["day_of_the_week"].map(DAY_MAP)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([6, 7, 8, 16, 17, 18]).astype(int)

    # The "date" column restarts after 31, so build a running day counter instead.
    df["day_index"] = (df["date"] != df["date"].shift()).cumsum() - 1

    # ---- hazard features -----------------------------------------------
    df["weather_severity"] = df["weather"].apply(weather_severity)
    df["road_closed"] = (df["road_closed"].str.strip().str.lower() == "yes").astype(int)
    # 'accidents' is already 0/1

    # ---- targets ---------------------------------------------------------
    df["situation_code"] = df["traffic_situation"].str.strip().str.lower().map(SITUATION_ORDER)

    # ---- time-series (lag) features -------------------------------------
    df["total_lag1"] = df["total"].shift(1)
    df["total_lag2"] = df["total"].shift(2)
    df["total_lag4"] = df["total"].shift(4)              # one hour ago
    df["total_lag96"] = df["total"].shift(STEPS_PER_DAY)  # same time yesterday
    df["total_roll4"] = df["total"].shift(1).rolling(4).mean()  # avg of the last hour

    # The first day has no "yesterday", so drop the rows with missing lags
    df = df.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split by DAY (never shuffle time-series data!): the last days become the test set."""
    last_train_day = int(df["day_index"].max() * (1 - test_fraction))
    train = df[df["day_index"] <= last_train_day]
    test = df[df["day_index"] > last_train_day]
    return train, test


if __name__ == "__main__":
    data = build_dataset()
    train, test = chronological_split(data)
    print("Model-ready table :", data.shape)
    print("Train rows / days :", len(train), "/", train["day_index"].nunique())
    print("Test rows / days  :", len(test), "/", test["day_index"].nunique())
    print("\nFeature preview:")
    print(data[REGRESSION_FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]].head())
