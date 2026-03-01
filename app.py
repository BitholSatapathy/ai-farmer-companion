from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/crop_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        return render_template("index.html", prediction_text=f"Recommended Crop: {prediction[0]}")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)