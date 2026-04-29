from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import csv
import random

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset rows for auto-fill (first 11 columns = features, 12th = quality)
with open("winequality.csv", "r") as f:
    reader = csv.reader(f)
    dataset_rows = [row for row in reader]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sample")
def sample():
    """Return a random sample row from the dataset for auto-fill."""
    row = random.choice(dataset_rows)
    # First 11 values are features
    features = row[:11]
    return jsonify({
        "f1": features[0],
        "f2": features[1],
        "f3": features[2],
        "f4": features[3],
        "f5": features[4],
        "f6": features[5],
        "f7": features[6],
        "f8": features[7],
        "f9": features[8],
        "f10": features[9],
        "f11": features[10],
        "quality": row[11] if len(row) > 11 else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    return render_template("index.html", prediction_text=f"Wine Quality: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)