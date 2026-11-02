# Logistics Route AI - model design and training

Predicts traffic volume (regression) and labels traffic state (classification),
then combines them with hazard penalties to recommend the fastest + safest route.

## Setup
    python -m venv .venv
    .venv\Scripts\Activate.ps1        (Mac/Linux: source .venv/bin/activate)
    pip install -r requirements.txt

Copy Dataset.csv into data/ and run from the project folder:

    python src/run_all.py

## Files (src/)
| Step | File | Purpose |
|---|---|---|
| 3 | features.py | Shared data loading + feature engineering |
| 3 | explore_data.py | Data understanding and quality checks |
| 3 | explore_time_series.py | Time-series analysis (daily pattern, rolling mean, autocorrelation) |
| 4 | train_regression.py | Train and compare regression models |
| 4 | train_classification.py | Train and compare classification models |
| 4 | tune_models.py | Hyper-parameter tuning with time-series cross-validation |
| 4 | evaluate_models.py | In-depth evaluation on unseen days |
| 4 | test_models.py | Smoke test for the saved models |
| - | routing.py | Shared route-ranking logic (both models + hazard penalties) |
| - | route_score.py | Demo: rank simulated routes using both models |
| - | chatbot.py | NLP route-advisor chatbot (intent classifier + optional speech). Run `python src/chatbot.py` to chat, `--demo`, `--evaluate`, `--speak` |
| - | run_all.py | Runs everything in order |

## Optional: speech synthesis
    pip install pyttsx3
    python src/chatbot.py --speak
