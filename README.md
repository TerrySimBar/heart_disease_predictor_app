# heart_disease_predictor
A supervised machine learning project which uses Python (Pandas, NumPy, Scikit-learn, Flask, Joblib) HTML, CSS, and JavaScript technologies to build a binary classifier, which performs with at least 90% accuracy, to make predictions about the risk of heart disease.
<br>
<br>
The project features a full stack-front and back end development-to create a functional application that satisfies modern industry standards.  
<br>

<img src="static/images/home.png" width="1600">   
<br>
<br>

## Table of Contents
### Overview
### Features
### Requirements & Dependencies
### Project Structure
### Usage
### Contributing
### License
<br>

### Overview
Heart Disease is among the most prevalent chronic diseases in the United States. It claims roughly 647,000 lives each year.

NuCare Health Insurance Company is concerned about 2 phenomena which occurred in 2015:
- 80% of their cardiology specialist referrals were for cases that had low risk 
for heart disease.
<br>

- The expenditure for cardiology specialist care was 40% more than the previous year. 

The National Health Insurance Equity Board is looking to lower the number of “No Referral Needed from Primary Care” appointments to make more specialist care spots available to patients who need them the most. 

Our organization is contracted to develop an application which can be used as a baseline for primary care doctors to use to generate codes to make cardiology referrals. 
<br>

The application features a navigation bar with buttons to navigate to external resources such as a manual and code book. Users can also navigate to webpages for making referrals or requesting authorization for referrals that are not recommended but believed to be necessary. 
<br>

<br>

### Features
- ETL: Use Python and Pandas to clean and format a CSV dataset to train the machine learning model.

- Build a binary classifier using Python Scikit-learn

- Use Flask to build routes for the application's backend

- Use HTML, CSS, Bootstrap, and JavaScript to develop the front end of the application

<br>

### Requirements & Dependencies
jQuery is included directly in the HTML files using `<script>` a tag. This library is hosted via a Content Delivery Network (CDN). 
<br>
<br>

You will also need to install the following dependencies:   

Python (version 3.10.9)

Pandas (version: 2.0.3)

Flask (version: 2.2.2)

Sci-Kit Learn (version: 1.3.0)



<br>

### Project Structure
<br>

#### ETL

The ETL process for this project is performed in a Jupyter Notebook (`etl.ipynb`) using Python, Pandas, and Sci-Kit Learn. The notebook contains the following steps:   

1. **Set Up Dependencies:** Import the necessary Python software.

2. **Read the CSV File:** Read the CSV file into a Pandas DataFrame using `pd.read_csv`. The dataset is downloaded from the source and contains over half a million rows.   

3. **Clean the Data:** The dataset is cleaned and formatted using Pandas. The following steps are included:
   - Drop unnecessary columns
   - Drop rows with missing/invalid values  
   - Rename columns
   - Replace data values to match the expected inputs from the HTML form
 
<br>

#### Build the Machine Learning Model
An appropriate binary classifier is built using the following steps:    

1. **Separate the Data into Target and Features** The target variable is the `heart_disease` column, which contains binary values. The features are the remaining columns.

2. **Create a Logistic Regression Model:** Instantiate a Logistic Regression model using `LogisticRegression()` from the Sci-Kit Learn library. 

3. **Fit the Model and Make Predictions:** Use`fit()` to fit the model to the training data and `predict()` to make predictions on the test data.

4. **Evaluate the Model:** Print the balanced accuracy score, confusion matrix, and classification report to evaluate the model's performance.

5. **Save the Model:** Save the model using `joblib.dump()` from the Joblib library. 

6. **Repeat the Process for Random Forest Classifier:** Repeat the process for  Random Forest Classifier, save the model, and compare the performance of the two models.  

#### Flask   
The Flask API is created in a Python file (`app.py`). Here are the key steps in the API creation process:   

1. **Set Up Dependencies:** Import Flask, render_template, joblib, and the other necessary dependencies. 

2. **Create the Flask App:** Create a Flask app instance using `Flask(__name__)`.   

3. **Define the Home Route:** Use `render_template` to ensure that the home route renders the `index.html` template. This is important to run the dashboard on the local server. You will need to ensure that the `index.html` file is in the `templates` folder, and you will have to restructure the project folder and necessary scripts if you wish to deploy the application on a web server. 
- The home route displays the navigation bar and the form for making referrals.  

4. **Define the Prediction Route:** Use `render_template` to ensure that the prediction route renders the `result.html` template. The `result.html` file should also be placed in the `templates` folder. The prediction route is defined with steps to process the form data and predict the risk of heart disease, and then render the `result.html` template with the prediction results and specified messages.

<br>

#### HTML Setup
The HTML files (`index.html`and `result.html`) in this project is structured as follows:

1. **Document Type Declaration:** `<!DOCTYPE html>` defines the document type and version of HTML being used.    

2. **Head Section** `<head>` contains meta information about the document, such as character encoding, viewport settings, and the page title.   

3. **Bootstrap Styling** `<link rel="stylesheet">` is a popular CSS framework that is linked in the `<head>` section to provide styling for the webpage, including the navbar and buttons.  

4. **Body Section** `<body>` contains the main content of the webpage, including the form for taking in user input as well as displaying the results of the prediction.  

<br>

#### CSS Styling
The CSS file (`index_style.css` and `result_style.css`) is used to style the HTML elements. The following are the key steps in the styling process:

navbar: with title and functioning buttons 


body: background picture with no repeat  


formWizard: size, position, padding, linear gradient color, shadow
1. **Set Up the Body** The `body` element is styled with a background image set to `no-repeat` and `center`. 
2. **Set Up the Navbar** The `navbar` element is styled with gradient colors and functioning buttons that link to external 
resources.
3. **Set Up the Form/Result Box** The `form` element is styled with a linear gradient color, padding, and shadow.

<br>

### Usage
1. Ensure that you have all the necessary dependencies and files/links/scripts. 
2. Load the project files in an appropriate code editor such as Jupyter Notebook. 
3. Open the terminal at the folder containing the `app.py` file, ensure any virtual environments are activated, and run  appropriate command such as `flask run` to start the Flask server.
4. Follow the `http` link in the terminal to open the dashboard in a browser. 

### Contributions
Contributions to this project are highly encouraged! If you wish to contribute, please follow these guidelines:

- Fork the heart_disease_predictor_app repository and clone it locally.
- Create a new branch for your feature or bug fix.
- Commit your changes with descriptive commit messages.
- Push your branch to your forked repository.
- Submit a pull request to the original repository.
- Please ensure that your code adheres to the project's coding style and conventions.


If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository.

### License
These projects are licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license. 

