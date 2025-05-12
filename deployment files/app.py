from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[feature]) for feature in [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return render_template('index.html', prediction_text=f'Predicted House Value: ${prediction * 100000:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
