from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load models
model_path = "model.pkl"
models = {}

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
else:
    print(f"Missing model: {model_path}")

# Feature labels for UI
feature_labels = {
    'air_temperature_': 'Air Temperature [K]',
    'process_temperature_': 'Process Temperature [K]',
    'rotational_speed_': 'Rotational Speed [rpm]',
    'torque_': 'Torque [Nm]',
    'tool_wear_': 'Tool Wear [min]'
}

features = list(feature_labels.keys())

@app.route('/')
def home():
    return render_template('index.html', features=feature_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values and preserve them
        input_data = {}
        input_values = []
        for key in features:
            val = float(request.form[key])
            input_data[key] = val
            input_values.append(val)

        df = pd.DataFrame([input_values], columns=features)

        # Focus strictly on machine failure prediction
        machine_failure_model = models.get('machine_failure')
        
        prediction_result = None
        if machine_failure_model:
            prediction = machine_failure_model.predict(df)[0]
            prediction_result = "FAIL" if int(prediction) == 1 else "PASS"

        return render_template('index.html', features=feature_labels, input_data=input_data, prediction=prediction_result)

    except Exception as e:
        return render_template('index.html', features=feature_labels, error_msg=str(e))
if __name__ == '__main__':
    app.run(port=8080, debug=True)