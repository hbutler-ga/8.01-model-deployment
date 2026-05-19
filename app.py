import pandas as pd
import joblib
import streamlit as st

# Load the trained model
model = joblib.load("./models/model.joblib")

# Define the Streamlit app
# We can start running the app with the command: streamlit run app.py

st.title("Housing Price Predictor")

# Create input fields for the features
sqft = st.number_input("Square Feet", min_value=0, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

# convert our inputs into a dataframe
input_data = pd.DataFrame({
    "sqft": [sqft],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms]
})

prediction = model.predict(input_data)

st.subheader(f"Predicted Price: ${prediction[0]:,.2f}")