# Set up the Flask app dependencies
from flask import Flask, render_template, request
import joblib
import numpy as np
import random
import string

app = Flask(__name__)

# Load the logistic regression model
model_path = 'model/random_forest_model.joblib'
loaded_model = joblib.load(model_path)

# Define a function to preprocess the input data
def preprocess_input(data):
   return np.array([float(data[key]) for key in data.keys()]).reshape(1, -1)


# Define the home page route
@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction result page route with random code generation
def generate_random_code():
    characters = string.hexdigits
    return ''.join(random.choice(characters) for _ in range(5))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Preprocess the input data
    input_data = preprocess_input(data)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Generate random code based on the prediction
    prediction = int(predictions[0])
    if prediction == 1:
        random_code = generate_random_code()
    else:
        random_code = None 

    # Render the `result.html` template with the prediction results
    return render_template('result.html', prediction=prediction, random_code=random_code)

if __name__ == '__main__':
    app.run(debug=True)