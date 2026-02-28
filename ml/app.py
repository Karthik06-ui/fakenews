from flask import Flask, render_template, request
from ml.predict_logic import predict_news
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    fake_prob = real_prob = None
    text = ""

    if request.method == "POST":
        text = request.form["news"]

        if len(text) > 50:
            result, fake_prob, real_prob = predict_news(text)

    return render_template(
        "index.html",
        result=result,
        fake_prob=fake_prob,
        real_prob=real_prob,
        text=text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
