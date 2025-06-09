from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the trained XGBoost Model
model = joblib.load("xgb_ProvantageNGX-AI_model.pkl")

# Define feature names (must match training order)
FEATURES = [
    'Open', 'High', 'Low', 'Vol.', 
    'SMA_7', 'SMA_14', 'EMA_10', 
    'Close_Lag_1', 'Close_Lag_2'
]

# Initialize Flask app
app = Flask(__name__)

# 🏠 Home Route – Renders the HTML Form
@app.route('/')
def home():
    return render_template('form.html', features=FEATURES)

# 🎯 Form Submission Route – Handles Form Input
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Collect input from HTML form
        values = [float(request.form[feature]) for feature in FEATURES]
        input_df = pd.DataFrame([values], columns=FEATURES)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('form.html', features=FEATURES, prediction=prediction)

    except Exception as e:
        return render_template('form.html', features=FEATURES, prediction=f"Error: {e}")

# 🧪 Optional API Endpoint for Programmatic Access
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        return jsonify({
            "predicted_close_price": round(prediction, 2),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

# 🚀 Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
