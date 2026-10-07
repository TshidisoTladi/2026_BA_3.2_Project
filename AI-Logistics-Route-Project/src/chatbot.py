"""
chatbot.py - NLP route-advisor chatbot ("Softbot") for drivers and dispatchers

How it works (the NLP pipeline):
  1. Text -> TF-IDF features (words and word pairs)
  2. Intent classifier (Logistic Regression) -> what does the user want?
  3. Entity extraction (regex + keyword synonyms) -> which route do they mean?
  4. Answer generator -> fills a reply template with numbers produced by the
     regression + classification models (via route_score.evaluate_routes)
  5. Optional speech synthesis (--speak) using the pyttsx3 library

Usage (from the project folder):
    python src/chatbot.py              interactive chat (type 'quit' to exit)
    python src/chatbot.py --demo       scripted demo conversation
    python src/chatbot.py --evaluate   accuracy of the intent classifier
    python src/chatbot.py --speak      read the replies aloud (needs: pip install pyttsx3)

The routes are the SIMULATED routes from route_score.py; in the real system the same
answers would come from live API data.
"""
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from route_score import evaluate_routes

# ------------------------------------------------------------------ training data
INTENTS = {
    "greeting": [
        "hello", "hi", "hey there", "good morning", "good afternoon", "hi bot",
        "hello assistant", "hey", "good evening", "howzit",
    ],
    "goodbye": [
        "bye", "goodbye", "see you later", "thanks bye", "that is all", "quit",
        "exit", "thank you goodbye", "i am done", "talk later",
    ],
    "help": [
        "what can you do", "help", "how do you work", "what can i ask you",
        "help me", "what are your features", "how can you help me",
        "show me the options", "what do you know", "i need help using this",
    ],
    "recommend_route": [
        "which route should i take", "what is the fastest route",
        "recommend the best route", "what is the safest and fastest way",
        "best route for my delivery", "which way is quickest right now",
        "suggest a route", "what route do you recommend", "where should i drive",
        "give me the best route", "which road is best", "shortest and safest route please",
        "how should i get there",
    ],
    "check_accident": [
        "is there an accident on route a", "any crashes ahead",
        "were there accidents on the highway", "accident on the ring road",
        "any collisions reported", "has there been a crash on route b",
        "is there an accident", "any accidents on my way", "accident report",
        "did an accident happen on the arterial", "are there crashes on the local roads",
    ],
    "check_traffic": [
        "how is the traffic on route b", "is the ring road congested",
        "how busy is the highway", "traffic volume on route d",
        "is there a traffic jam", "how heavy is the traffic",
        "is route a busy", "what are the congestion levels",
        "how many cars are on the road", "is the road full of traffic",
        "traffic status please",
    ],
    "check_closure": [
        "is route c closed", "any road closures", "is the local road open",
        "are any roads blocked", "is the highway closed", "can i use the local roads",
        "which roads are shut", "is the ring road open", "road closed on route a",
        "any blocked roads ahead", "is the arterial open",
    ],
    "check_weather": [
        "what is the weather like on the route", "is it raining", "any bad weather ahead",
        "weather conditions on route b", "is the weather safe for driving",
        "any storm warnings", "how is the weather on the highway",
        "is it foggy", "will the weather affect my trip", "weather report please",
    ],
    "eta": [
        "how long will route b take", "what is the travel time",
        "how many minutes to the destination", "estimated time of arrival on the highway",
        "how long does route d take", "eta please", "how long is the trip",
        "when will i arrive", "time to destination on the ring road",
        "how long will the drive take", "travel time for the local roads",
    ],
    "explain": [
        "why did you recommend that route", "explain your recommendation",
        "why not route a", "how did you decide", "what is the reason for that choice",
        "why is that the best route", "why not the highway",
        "how do you choose a route", "justify your recommendation",
        "what factors do you consider",
    ],
    "compare": [
        "compare all routes", "show me all the routes", "list the options",
        "give me a summary of all routes", "route comparison", "show every route",
        "how do the routes compare", "give me all route details",
        "show me the alternatives", "list all routes with times",
    ],
}

# Phrases the classifier never sees while training - used for an honest accuracy check
TEST_PHRASES = {
    "greeting": ["hello there", "good day"],
    "goodbye": ["ok bye now", "thank you that is all"],
    "help": ["what can i ask", "help please"],
    "recommend_route": ["what is the best way to go", "recommend a route for me"],
    "check_accident": ["any accident on the ring road", "did a crash happen on route c"],
    "check_traffic": ["how bad is traffic on the highway", "is route d congested"],
    "check_closure": ["is the arterial closed", "any closed roads"],
    "check_weather": ["how is the weather", "is it raining on route a"],
    "eta": ["how long to get there", "eta for route a"],
    "explain": ["why did you pick that", "explain why not route d"],
    "compare": ["compare the routes", "show me every option"],
}

# ------------------------------------------------------------------ entity extraction
ROUTE_KEYWORDS = {
    "A": ["direct", "highway", "route a"],
    "B": ["ring", "route b"],
    "C": ["local", "route c"],
    "D": ["arterial", "route d"],
}


def find_route(text: str):
    """Return the route letter mentioned in the text ('A'-'D') or None."""
    t = text.lower()
    for letter, words in ROUTE_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return letter
    return None


# ------------------------------------------------------------------ the classifier
def build_intent_model():
    texts = [p for phrases in INTENTS.values() for p in phrases]
    labels = [intent for intent, phrases in INTENTS.items() for _ in phrases]
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(C=10, max_iter=2000),
    )
    model.fit(texts, labels)
    return model


CONFIDENCE_THRESHOLD = 0.30  # below this the bot says it did not understand


class RouteBot:
    def __init__(self):
        self.intent_model = build_intent_model()
        self.table, self.best, self.shortest = evaluate_routes()
        self.rows = {r["route"][0]: r for _, r in self.table.iterrows()}
        self.last_route = None  # remembers the route we talked about last

    # -------------------------------------------------------------- helpers
    def _name(self, letter):
        return self.rows[letter]["route"].split(None, 1)[1]

    def _resolve(self, text):
        """Which route does the user mean? (explicit mention > last route > None)"""
        letter = find_route(text)
        if letter:
            self.last_route = letter
        return letter or self.last_route

    # -------------------------------------------------------------- main entry
    def reply(self, text: str) -> str:
        probs = self.intent_model.predict_proba([text])[0]
        intent = self.intent_model.classes_[probs.argmax()]
        if probs.max() < CONFIDENCE_THRESHOLD:
            return ("Sorry, I did not understand that. You can ask about the best route, "
                    "accidents, traffic, road closures, weather, travel time, or say 'help'.")
        return getattr(self, f"_on_{intent}")(text)

    # -------------------------------------------------------------- intent handlers
    def _on_greeting(self, text):
        return "Hello! I am your route assistant. Ask me for the fastest and safest route."

    def _on_goodbye(self, text):
        return "Goodbye, drive safely!"

    def _on_help(self, text):
        return ("I can: recommend the fastest + safest route, report accidents, traffic, "
                "road closures and weather per route, give travel times, compare all routes "
                "and explain my recommendation. Example: 'Is there an accident on route A?'")

    def _on_recommend_route(self, text):
        b = self.best
        self.last_route = b["route"][0]
        return (f"I recommend route {b['route']}: expected travel time about "
                f"{b['adjusted_min']:.0f} minutes, traffic is {b['traffic_state']} and there "
                f"are no accidents or closures on it.")

    def _on_check_accident(self, text):
        letter = self._resolve(text)
        if letter:
            r = self.rows[letter]
            if r["accident"]:
                return f"Yes - an accident has been reported on route {r['route']}. I advise avoiding it."
            return f"No accidents are reported on route {r['route']}."
        hit = [r["route"] for r in self.rows.values() if r["accident"]]
        return ("Accident reported on: " + ", ".join(hit) + ". All other routes are clear."
                if hit else "No accidents are reported on any route.")

    def _on_check_traffic(self, text):
        letter = self._resolve(text)
        if letter:
            r = self.rows[letter]
            return (f"Traffic on route {r['route']} is {r['traffic_state']} "
                    f"(about {r['pred_volume']} vehicles expected per 15 minutes).")
        parts = [f"{r['route']}: {r['traffic_state']}" for r in self.rows.values()]
        return "Traffic right now - " + "; ".join(parts) + "."

    def _on_check_closure(self, text):
        letter = self._resolve(text)
        if letter:
            r = self.rows[letter]
            return (f"Route {r['route']} is CLOSED - do not use it." if r["closed"]
                    else f"Route {r['route']} is open.")
        closed = [r["route"] for r in self.rows.values() if r["closed"]]
        return ("Closed roads: " + ", ".join(closed) + ". All other routes are open."
                if closed else "All routes are open.")

    def _on_check_weather(self, text):
        names = {0: "clear", 1: "light rain or mist", 2: "moderate rain, snow or fog",
                 3: "severe weather"}
        letter = self._resolve(text)
        r = self.rows[letter] if letter else next(iter(self.rows.values()))
        level = int(r["weather"])
        advice = " Drive carefully and allow extra time." if level >= 2 else ""
        return f"Weather conditions: {names[level]} (severity {level} out of 3).{advice}"

    def _on_eta(self, text):
        letter = self._resolve(text) or self.best["route"][0]
        r = self.rows[letter]
        if r["closed"]:
            return f"Route {r['route']} is closed, so I cannot give a travel time."
        return (f"Route {r['route']}: about {r['adjusted_min']:.0f} minutes "
                f"({r['km']} km, {r['free_flow_min']} min without traffic).")

    def _on_explain(self, text):
        letter = find_route(text)
        b, s = self.best, self.shortest
        if letter and letter != b["route"][0]:
            r = self.rows[letter]
            reason = ("it is closed" if r["closed"] else
                      f"it has {r['traffic_state']} traffic"
                      + (" and an accident" if r["accident"] else "")
                      + f", giving about {r['adjusted_min']:.0f} minutes")
            return f"I did not choose route {r['route']} because {reason}."
        why = (f"I scored every open route on predicted traffic volume, accidents and weather "
               f"on top of the distance. Route {b['route']} has the lowest adjusted time "
               f"({b['adjusted_min']:.0f} min).")
        if b["route"] != s["route"]:
            why += (f" The shortest route, {s['route']}, would take about "
                    f"{s['adjusted_min']:.0f} min because of {s['traffic_state']} traffic"
                    + (" and an accident." if s["accident"] else "."))
        return why

    def _on_compare(self, text):
        lines = ["Route comparison:"]
        for r in self.rows.values():
            status = ("CLOSED" if r["closed"] else
                      f"{r['adjusted_min']:.0f} min, {r['traffic_state']} traffic"
                      + (", accident" if r["accident"] else ""))
            lines.append(f"  {r['route']} ({r['km']} km): {status}")
        return "\n".join(lines)


# ------------------------------------------------------------------ speech synthesis
def make_speaker():
    """Return a function that reads text aloud, or None if pyttsx3 is not installed."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)

        def speak(text):
            engine.say(text)
            engine.runAndWait()
        return speak
    except Exception:
        print("(Speech synthesis unavailable: run 'pip install pyttsx3' to enable --speak)")
        return None


# ------------------------------------------------------------------ entry points
def evaluate():
    model = build_intent_model()
    texts = [p for ps in TEST_PHRASES.values() for p in ps]
    truth = [i for i, ps in TEST_PHRASES.items() for _ in ps]
    pred = model.predict(texts)
    print(f"Intent classifier accuracy on {len(texts)} unseen phrases: "
          f"{accuracy_score(truth, pred) * 100:.1f}%")
    for t, y, p in zip(texts, truth, pred):
        if y != p:
            print(f"  wrong: '{t}'  expected {y}, got {p}")
    print(f"Training phrases: {sum(len(v) for v in INTENTS.values())} across {len(INTENTS)} intents")


DEMO_CONVERSATION = [
    "Hi", "Which route should I take?", "Why not the highway?",
    "Is there an accident on route A?", "How is the traffic on the ring road?",
    "Is the local road open?", "How long will route B take?",
    "What is the weather like?", "Compare all routes", "asdf qwerty", "Thanks, bye",
]


def main():
    args = sys.argv[1:]
    if "--evaluate" in args:
        return evaluate()
    bot = RouteBot()
    speak = make_speaker() if "--speak" in args else None

    def say(text):
        print(f"Bot: {text}")
        if speak:
            speak(text)

    if "--demo" in args:
        for line in DEMO_CONVERSATION:
            print(f"\nYou: {line}")
            say(bot.reply(line))
        return

    print("Route assistant (demo data). Type 'quit' to exit.\n")
    say(bot._on_greeting(""))
    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            say(bot._on_goodbye(text))
            break
        say(bot.reply(text))


if __name__ == "__main__":
    main()
