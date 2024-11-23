import math
import numpy as np
import pickle
import streamlit as st

# Load the saved model using pickle
filename = "Admission_model.sav"
with open(filename, "rb") as file:  # Open the file in binary read mode
    model = pickle.load(file)

# Define Streamlit app title
st.title("M&M Student Admission Prediction")

# Collect user input for each feature
maths = st.number_input("Maths Score", min_value=0.0, max_value=100.0, value=36.0)
english_studies = st.number_input("English Score", min_value=0.0, max_value=100.0, value=30.0)
general_paper = st.number_input("General Paper Score", min_value=0.0, max_value=100.0, value=80.0)
total = st.number_input("Total", min_value=0.0, max_value=300.0, value=146.0)
weight_average = st.number_input("Weighted average", min_value=0.0, max_value=100.0, value=49.0)

# Create input array from the user input
input_data = [	maths,	english_studies, general_paper,	total,	weight_average]
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prediction button
if st.button("Predict Admission"):
    prediction = model.predict(input_data_reshaped)

    # Display result based on the prediction
    if prediction[0] == 0:
        st.write("Result: Not admitted")
    elif prediction[0] == 1:
        st.write("Result: Advised for intensive class")
    elif prediction[0] == 2:
        st.write("Result: Admitted")
