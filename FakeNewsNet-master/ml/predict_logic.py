import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

def predict_news(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    label = "FAKE 🟥" if pred == 1 else "REAL 🟩"

    return label, prob[1], prob[0]
