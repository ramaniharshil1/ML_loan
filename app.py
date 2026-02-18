from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Load ML Models ---
# Ideally, these paths should be dynamic or absolute to avoid errors
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

# Load the trained model and scaler
# We use try-except to handle cases where files might be missing during development
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found. Please run train_model.py first.")
    model = None
    scaler = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return render_template("result.html", prediction="Error: Model not loaded.")

    try:
        # 1. Collect Input from Form
        income = float(request.form.get("income"))
        loan_amount = float(request.form.get("loan_amount"))
        credit_score = float(request.form.get("credit_score"))
        age = int(request.form.get("age"))

        # 2. Prepare Data (2D Array)
        features = np.array([[income, loan_amount, credit_score, age]])

        # 3. Scale Features
        scaled_features = scaler.transform(features)

        # 4. Make Prediction
        # Prediction: 0 = No Default, 1 = Default
        prediction = model.predict(scaled_features)[0]
        
        # Get Probability (Confidence Score)
        # model.predict_proba returns [[prob_class_0, prob_class_1]]
        probability = model.predict_proba(scaled_features)[0][1] 
        risk_score = round(probability * 100, 1)

        # 5. Determine Result Message
        if prediction == 1:
            result_message = "High Risk of Default"
            is_defaulter = True
        else:
            result_message = "Low Risk of Default"
            is_defaulter = False

        # 6. Render Result Page
        return render_template("result.html", 
                             prediction=result_message, 
                             is_defaulter=is_defaulter,
                             risk_score=risk_score,
                             income=income,
                             loan_amount=loan_amount,
                             credit_score=credit_score,
                             age=age)

    except Exception as e:
        # Handle errors gracefully
        return render_template("result.html", prediction=f"Error: {str(e)}", is_defaulter=None)

if __name__ == "__main__":
    # Use os.environ.get to configure the port for flexible deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
