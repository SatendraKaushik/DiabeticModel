import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Diabetes Prediction App")

st.write("""
    Please enter the following details to predict if the person is diabetic or not.
""")

# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=85)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=66)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=29)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=26.6)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.351)
age = st.number_input('Age', min_value=20, max_value=100, value=31)

# Create a button for prediction
if st.button('Predict'):
    # Prepare input data
    input_data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    }
    
    # Convert the input data to a Pandas DataFrame
    input_data_df = pd.DataFrame(input_data)
    
    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_df)
    
    # Make the prediction
    prediction = classifier.predict(std_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The person is Diabetic")
    else:
        st.success("The person is Non-Diabetic")
