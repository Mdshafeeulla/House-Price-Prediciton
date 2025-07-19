import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

st.title('House Price Prediction')

# Load the trained model
model = load_model('house_price_model.h5')

# Load the preprocessed data to get the columns for the UI
df_encoded = pd.read_csv('preprocessed_data.csv')

# Streamlit UI
st.sidebar.header('User Input Features')

def user_input_features():
    Carpet_Area = st.sidebar.number_input('Carpet Area (sqft)', 0, 5000, 1000)
    Flat_floor = st.sidebar.number_input('Flat Floor', 0, 50, 2)
    Total_floors = st.sidebar.number_input('Total Floors', 0, 50, 10)
    Bathrooms = st.sidebar.number_input('Bathrooms', 1, 10, 2)
    Balcony = st.sidebar.number_input('Balcony', 0, 10, 1)

    Location = st.sidebar.selectbox('Location', df_encoded.filter(regex='Location_').columns)
    Transaction = st.sidebar.selectbox('Transaction', df_encoded.filter(regex='Transaction_').columns)
    Furnishing = st.sidebar.selectbox('Furnishing', df_encoded.filter(regex='Furnishing_').columns)
    Facing = st.sidebar.selectbox('Facing', df_encoded.filter(regex='Facing_').columns)
    overlooking = st.sidebar.selectbox('Overlooking', df_encoded.filter(regex='overlooking_').columns)
    Ownership = st.sidebar.selectbox('Ownership', df_encoded.filter(regex='Ownership_').columns)
    Parking_type = st.sidebar.selectbox('Parking Type', df_encoded.filter(regex='Parking_type_').columns)

    data = {'Carpet_Area': Carpet_Area,
            'Flat_floor': Flat_floor,
            'Total_floors': Total_floors,
            'Bathrooms': Bathrooms,
            'Balcony': Balcony}

    for col in df_encoded.filter(regex='Location_').columns:
        data[col] = 1 if col == Location else 0
    for col in df_encoded.filter(regex='Transaction_').columns:
        data[col] = 1 if col == Transaction else 0
    for col in df_encoded.filter(regex='Furnishing_').columns:
        data[col] = 1 if col == Furnishing else 0
    for col in df_encoded.filter(regex='Facing_').columns:
        data[col] = 1 if col == Facing else 0
    for col in df_encoded.filter(regex='overlooking_').columns:
        data[col] = 1 if col == overlooking else 0
    for col in df_encoded.filter(regex='Ownership_').columns:
        data[col] = 1 if col == Ownership else 0
    for col in df_encoded.filter(regex='Parking_type_').columns:
        data[col] = 1 if col == Parking_type else 0

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader('User Input parameters')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f"The predicted house price is {prediction[0][0]:,.2f} rupees.")
