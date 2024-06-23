from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from scipy.stats import norm

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Prepare input for prediction
    features = [
        float(data['potential']),
        float(data['value_eur']),
        float(data['wage_eur']),
        float(data['passing']),
        float(data['dribbling']),
        float(data['physic']),
        float(data['movement_reactions']),
        float(data['mentality_composure'])
    ]

    # Convert to NumPy array and reshape
    features_array = np.array(features).reshape(1, -1)

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features_array)

    # Predict using the model
    prediction = model.predict(features_scaled)[0]

    # Calculate confidence
    if hasattr(model, 'estimators_'):
        predictions = np.array([est.predict(features_scaled)[0] for est in model.estimators_])
        std_pred = np.std(predictions)
        mean_pred = np.mean(predictions)
        n = len(predictions)

        confidence_level = 0.90

        z_score = norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * (std_pred / np.sqrt(n))
        percentage_margin_of_error = (margin_of_error / mean_pred) * 100
        absolute_margin_of_error = (percentage_margin_of_error / 100) * prediction
        confidence_interval = (prediction - absolute_margin_of_error, prediction + absolute_margin_of_error)
    else:
        confidence_interval = None  # Handle the case where confidence cannot be estimated
        
    # Prepare response
    response = {
        'prediction': prediction,
        'confidence_interval': confidence_interval,
        'confidence_level': confidence_level
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)