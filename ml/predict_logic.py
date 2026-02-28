import joblib
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "models", "tfidf.pkl")

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

def predict_news(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    label = "FAKE 🟥" if pred == 1 else "REAL 🟩"

    return label, prob[1], prob[0]
