from flask import Flask, request, render_template
import joblib
import numpy as np

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

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Preprocess the input data
    input_data = preprocess_input(data)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Render the `result.html` template with the prediction results
    return render_template('result.html', prediction=int(predictions[0]))

if __name__ == '__main__':
    app.run(debug=True)
