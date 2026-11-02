"""
route_score.py - Step 3.6 in action: combine BOTH models into a route recommendation

For every candidate route we:
  1. predict the traffic volume            (regression model)
  2. label the traffic state               (classification model)
  3. add penalties for accidents + weather (risk factors)
  4. exclude the route if the road is closed
  5. rank by "adjusted travel time"        -> the recommended route is the lowest

The routes are SIMULATED (see routing.py). In the real system, distance/speed come
from the Google Maps / Mapbox API and the hazard flags from the live data feeds.
"""
from pathlib import Path

from routing import best_route, rank_routes

ROOT = Path(__file__).resolve().parent.parent

# Demo scenario: weekday 17:00, accident on A, route C closed, moderate weather
DEMO_HOUR = 17
DEMO_CONDITIONS = {"accident": {"A": 1}, "closed": {"C": 1}, "weather": 2}


def evaluate_routes():
    """Return (table, best_row, shortest_open_row) for the demo scenario.
    chatbot.py imports this function."""
    table = rank_routes(hour=DEMO_HOUR, conditions=DEMO_CONDITIONS)
    best, shortest = best_route(table)
    return table, best, shortest


if __name__ == "__main__":
    table, best, shortest = evaluate_routes()
    print("Scenario: weekday rush hour (17:00), accident on A, road C closed, weather severity 2\n")
    show = table[["route", "km", "free_flow_min", "pred_volume", "traffic_state",
                  "accident", "closed", "adjusted_min"]]
    print(show.to_string(index=False))
    print(f"\nShortest OPEN route by distance only : {shortest['route']} "
          f"({shortest['km']} km, adjusted time {shortest['adjusted_min']:.0f} min)")
    print(f"RECOMMENDED (fastest + safest)       : {best['route']} "
          f"(adjusted time {best['adjusted_min']:.0f} min)")
    if best["route"] != shortest["route"]:
        saved = shortest["adjusted_min"] - best["adjusted_min"]
        print(f"-> Avoids the accident/congestion and saves about {saved:.0f} minutes.")
    table.to_csv(ROOT / "outputs" / "route_recommendation_demo.csv", index=False)
