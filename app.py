import streamlit as st
import pandas as pd
import joblib
import sklearn  # Add this import to ensure scikit-learn is loaded
import numpy as np  # Add this import
from scipy import stats  # Add this import

# Load the trained model
@st.cache_resource  # Use cache to load the model only once
def load_model():
    return joblib.load("mortality_rate_model.pkl")

model = load_model()

# Define the input features
features = [
    "Prevalence Rate (%)", "Incidence Rate (%)", "Age Group", "Gender",
    "Healthcare Access (%)", "Doctors per 1000", "Hospital Beds per 1000",
    "Treatment Type", "Average Treatment Cost (USD)", "Availability of Vaccines/Treatment",
    "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
    "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)",
]

st.title("Mortality Rate Prediction")

# Create input fields for each feature
input_data = {}

for feature in features:
    if feature in ["Age Group", "Gender", "Treatment Type", "Availability of Vaccines/Treatment"]:
        # For categorical features, use selectbox
        options = ["Option 1", "Option 2", "Option 3"]  # Replace with actual options
        input_data[feature] = st.selectbox(f"Select {feature}", options)
    else:
        # For numerical features, use number_input
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, step=0.1)

# Create a button to make predictions
if st.button("Predict Mortality Rate"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display the prediction
    st.success(f"The predicted Mortality Rate is: {prediction[0]:.2f}%")

# Add some information about the model
st.info("This model predicts mortality rates based on various health and socioeconomic factors.")
