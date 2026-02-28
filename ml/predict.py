import joblib

# load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

print("Paste the news article text below (press ENTER twice to finish):\n")

# read multi-line input
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

news_text = " ".join(lines)

if len(news_text) < 50:
    print("❌ Text too short to analyze.")
    exit()

# vectorize
X = vectorizer.transform([news_text])

# predict
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0]

label = "FAKE 🟥" if prediction == 1 else "REAL 🟩"

print("\nPrediction:", label)
print("Confidence:")
print(f"  Real : {probability[0]:.2f}")
print(f"  Fake : {probability[1]:.2f}")
