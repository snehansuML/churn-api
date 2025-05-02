from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

model = joblib.load("model/churn_model.pkl")
encoders = joblib.load("model/label_encoders.pkl")

@app.route("/api/predict_churn", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received:", data)

        # Encode categorical values
        contract = data["contract_type"]
        if contract not in encoders["contract_type"].classes_:
            return jsonify({"error": f"Invalid value '{contract}' for 'contract_type'"}), 400
        contract_encoded = encoders["contract_type"].transform([contract])[0]

        payment = data["payment_method"]
        if payment not in encoders["payment_method"].classes_:
            return jsonify({"error": f"Invalid value '{payment}' for 'payment_method'"}), 400
        payment_encoded = encoders["payment_method"].transform([payment])[0]

        input_data = [
            float(data["tenure_months"]),
            float(data["monthly_charges"]),
            float(data["total_charges"]),
            float(data["complaints"]),
            float(contract_encoded),
            float(payment_encoded)
        ]

        input_array = np.array(input_data).reshape(1, -1)
        proba = model.predict_proba(input_array)[0][1]
        prediction = "Yes" if proba >= 0.5 else "No"

        return jsonify({
            "churn_prediction": prediction,
            "churn_probability": round(proba, 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/allowed_values", methods=["GET"])
def allowed_values():
    return jsonify({key: list(encoders[key].classes_) for key in encoders})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
