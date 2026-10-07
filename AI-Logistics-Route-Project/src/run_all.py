"""
run_all.py - runs the whole pipeline in the right order (about 1-2 minutes).
Usage (from the project folder):   python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "explore_data.py",          # Step 3: data understanding + quality checks
    "explore_time_series.py",   # Step 3: time-series analysis
    "train_regression.py",      # Step 4: train + compare regression models
    "train_classification.py",  # Step 4: train + compare classification models
    "tune_models.py",           # Step 4: hyper-parameter tuning
    "evaluate_models.py",       # Step 4: in-depth evaluation on unseen days
    "test_models.py",           # Step 4: check the saved models load and predict
    "route_score.py",           # Demo: combine both models to rank routes
    ("chatbot.py", "--demo"),   # NLP chatbot: scripted demo conversation
    ("chatbot.py", "--evaluate"),  # NLP chatbot: intent-classifier accuracy
]

for step in STEPS:
    name, *args = (step,) if isinstance(step, str) else step
    print("\n" + "#" * 70 + f"\n# {name} {' '.join(args)}\n" + "#" * 70)
    result = subprocess.run([sys.executable, str(SRC / name), *args])
    if result.returncode != 0:
        sys.exit(f"\nStopped: {name} failed. Fix the error above and run again.")
print("\nAll steps finished. See the outputs/ and models/ folders.")
