# -*- coding: utf-8 -*-
"""G100SuperApp.ipynb

"""

import joblib
import streamlit as st
#!pip install scikit-learn

# Load your trained model and vectorizer

model = joblib.load('C:/Users/USER/Downloads/G2024model.pkl')
vectorizer = joblib.load('C:/Users/USER/Downloads/G2024vectorizer.pkl')

def predict(email_text):
    processed_text = vectorizer.transform([email_text])
    prediction = model.predict(processed_text)
    return prediction[0]

from PIL import Image

# Set the title of the app
st.title("Welcome to Super Email Spam Detector App (MIS542)")

# Add an image at the top of the app
image_url = "https://miro.medium.com/v2/resize:fit:693/0*u_3GNniqZ6e7DSFK.png"  # My image URL
st.image(image_url, caption='Email Spam Prediction', use_column_width=True)

# Input box for the email
email_input = st.text_input("Let's Validate Your Email:")

# Add a button for prediction
if st.button("Predict"):
    prediction = predict(email_input)
    st.write(f"Prediction: {prediction}")

# Add a footer image or any other image
footer_image_url = "https://miro.medium.com/v2/resize:fit:1400/1*_igArwmR7Pj_Mu_KUGD1SQ.png"  # My footer image URL
st.image(footer_image_url, caption='Footer Image', use_column_width=True)

