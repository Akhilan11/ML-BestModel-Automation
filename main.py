import streamlit as st
import pandas as pd
import ml_functions

st.title("ML Automation")

# Step 1 -> Upload the file
# File uploader (accepts single file)
uploaded_file = st.file_uploader("Choose a CSV file")

submitted = False
model_type = None

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the content of the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file,on_bad_lines='skip')

    # Display filename and contents
    st.write("Filename:", uploaded_file.name)
    st.write(df)

    # Step 2 -> Data Preprocessing
    with st.form("ml_form"): 

        col1, col2, col3 = st.columns(3)

        with col1: 
            # Select target variables
            option = st.selectbox("Select the target column:", df.columns)

            # Show the selected option
            st.caption(f"You selected: {option}")

        with col2: 
            # Select scaling method
            scalar = st.selectbox("Select Scaling method",("standard","minmax"))
        
        with col3: 
            # Select model name
            title = st.text_input("Model name")
            st.caption(f"The current model title is {title}")

        model_type = st.selectbox("Select your Model type", ("regression","classification"))

        # Add a submit button
        submitted = st.form_submit_button("Submit")

else:
    st.write("Please upload a CSV file.")


if submitted:
    # Step 3 -> Preprocess and Train the model
    x_train, x_test, y_train, y_test = ml_functions.preprocessing(df, option, scalar)
    ml_functions.model_training(x_train, y_train, x_test, y_test, title, model_type)

