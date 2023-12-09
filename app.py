# app.py

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
model_path = 'random_forest_model.joblib'
loaded_model = joblib.load(model_path)

def preprocess_input(data):
    # You might need to preprocess the input data based on how your model was trained
    # For example, convert strings to numeric values, scale features, etc.
    # Here, I assume all inputs are numeric, but you might need additional preprocessing.
   return np.array([float(data[key]) for key in data.keys()]).reshape(1, -1)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Preprocess the input data
    input_data = preprocess_input(data)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Assuming a binary classifier, you can return the result
    return render_template('result.html', prediction=int(predictions[0]))

if __name__ == '__main__':
    app.run(debug=True)
