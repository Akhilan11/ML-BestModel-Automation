import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import pickle

# Step 2 -> Preprocessing the data
def preprocessing(df, target_variable, scalar_type):
    
    # Split the features and target 
    x = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Convert the target variable to a 1D array (important for certain models)
    y = np.asarray(y).squeeze()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Identify numerical and categorical columns
    numerical_columns = x.select_dtypes(include="number").columns
    categorical_columns = x.select_dtypes(include=["object", "category"]).columns

    ### Numerical columns processing ###
    if not numerical_columns.empty:
        # Impute missing values with the mean strategy
        num_imputer = SimpleImputer(strategy='mean')
        x_train[numerical_columns] = num_imputer.fit_transform(x_train[numerical_columns])
        x_test[numerical_columns] = num_imputer.transform(x_test[numerical_columns])

        # Select the appropriate scaler
        if scalar_type == 'standard':
            scaler = StandardScaler()
        elif scalar_type == 'minmax':
            scaler = MinMaxScaler()

        # Apply scaling to the numerical columns
        x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])
        x_test[numerical_columns] = scaler.transform(x_test[numerical_columns])

    ### Categorical columns processing ###
    if not categorical_columns.empty:
        # Impute missing values with the most frequent strategy for categorical columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        x_train[categorical_columns] = cat_imputer.fit_transform(x_train[categorical_columns])
        x_test[categorical_columns] = cat_imputer.transform(x_test[categorical_columns])

        # OneHotEncode categorical columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        x_train_encoded = encoder.fit_transform(x_train[categorical_columns])
        x_test_encoded = encoder.transform(x_test[categorical_columns])

        # Convert encoded columns back to DataFrames
        x_train_encoded = pd.DataFrame(x_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
        x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))

        # Concatenate the encoded columns back with the numerical columns
        x_train = pd.concat([x_train.drop(columns=categorical_columns), x_train_encoded], axis=1)
        x_test = pd.concat([x_test.drop(columns=categorical_columns), x_test_encoded], axis=1)

    return x_train, x_test, y_train, y_test

# Step 3 -> Train the model
def model_training(x_train, y_train, x_test, y_test, export_name, model_type):
    
    if model_type == "classification":
        # Initialize models for classification
        models = {
            'Logistic Regression': LogisticRegression(max_iter=200),
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier()
        }

        # Fit models and calculate accuracy scores
        model_results = []  # List to store model results
        best_score = 0
        best_model_name = None
        best_model = None

        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Append model name and accuracy to the results list
            model_results.append({'Model': name, 'Accuracy': accuracy})
            
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                best_model = model

        result_df = pd.DataFrame(model_results)

        # Print accuracy scores 
        st.write("### Model Accuracy Scores")
        col1, col2 = st.columns(2)
        with col1: 
            st.dataframe(result_df)

        with col2: 
            # Display the best model and its accuracy in the Streamlit app
            st.success(f"### Best model: {best_model_name} with accuracy: {best_score:.4f}", icon='😍')

        # Save the best model
        with open(f'{export_name}.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        return best_model

    if model_type == "regression":
        # Initialize models for regression
        models = {
            'Linear Regression': LinearRegression(),
            'Support Vector Regressor (SVR)': SVR(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor(),
            'AdaBoost Regressor': AdaBoostRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor(),
            'XGBoost Regressor': XGBRegressor()
        }

        # Fit models and calculate R² scores
        model_results = []  # List to store model results
        best_score = float('-inf')  # For R², higher is better
        best_model_name = None
        best_model = None

        # Check if there are still any NaNs in x_train or x_test
        if np.any(np.isnan(x_train)):
            print("Warning: NaNs detected in training data.")
        
        elif np.any(np.isnan(x_test)):
            print("Warning: NaNs detected in testing data.")
        else:
            print("No NaNs detected. Proceeding with model.")
            
        for name, model in models.items():
            model.fit(x_train, y_train)


            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)

            # Append model name and R² to the results list
            model_results.append({'Model': name, 'R2 Score': r2})

            if r2 > best_score:
                best_score = r2
                best_model_name = name
                best_model = model

        result_df = pd.DataFrame(model_results)

        # Print R² scores 
        st.write("### Model R² Scores")
        col1, col2 = st.columns(2)
        with col1: 
            st.dataframe(result_df)

        with col2: 
            # Display the best model and its R² in the Streamlit app
            st.success(f"### Best model: {best_model_name} with R²: {best_score:.4f}", icon='😍')

        # Save the best model
        with open(f'{export_name}.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        return best_model



