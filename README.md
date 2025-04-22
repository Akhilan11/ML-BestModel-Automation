# ML-BestModel-Automation

**ML-BestModel-Automation** is a Streamlit-based web application that automates the process of building and evaluating machine learning models. This tool allows users to upload a dataset, choose target features, apply preprocessing, and automatically select the best-performing model — all through a simple and interactive interface.

## Features

- Upload CSV datasets directly in the app  
- Select target column for prediction  
- Choose preprocessing options (e.g., StandardScaler, MinMaxScaler)  
- Automatically train multiple ML models, **evaluate them**, and highlight the one with the **highest accuracy**  
- No coding required!

Usage

1.Open the web app.
2.Upload a CSV dataset.
3.Select the target variable for prediction.
4.Choose a scaling method.
5.Let the app process the data and train models.
6.View the results including accuracy scores and the best model.

Project structure 
ML-BestModel-Automation/
├── main.py                # Streamlit app logic
├── ml_functions.py        # ML model training & utility functions
├── study.py               # Optional model evaluation or experimentation
├── requirements.txt       # Required Python packages
├── setup.sh               # Setup script (for deployment)
├── Procfile               # For Heroku deployment
└── README.md              # Project documentation


