import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title('House Price Prediction')

# Data cleaning and preprocessing functions from the notebook
def parse_amount(amount_str):
    amount_str = str(amount_str).strip().lower().replace(" ", "") # Clean string

    if 'lac' in amount_str:
        num_part = amount_str.replace('lac', '')
        try:
            return float(num_part) * 100000
        except ValueError:
            return np.nan # Or handle specific error, e.g., print a warning
    elif 'cr' in amount_str:
        num_part = amount_str.replace('cr', '')
        try:
            return float(num_part) * 10000000 # 1 Crore = 100 Lakhs = 10,000,000
        except ValueError:
            return np.nan
    else:
        # Assume it's already a plain number
        try:
            return float(amount_str)
        except ValueError:
            return np.nan # Handle cases where it's not a recognizable number

CONVERSION_FACTORS = {
    'sqft': 1,
    'sqyrd': 9,       # 1 Square Yard = 9 Square Feet
    'sqm': 10.764,    # 1 Square Meter = 10.764 Square Feet (approx)
    'kanal': 4500,    # Common in parts of Punjab, Haryana, HP (India)
                      # Alternative: 5445 sqft (common in Pakistan/some older Indian contexts)
    'acre': 43560,    # 1 Acre = 43560 Square Feet
    'marla': 272.25,  # Standard Marla, approx. 1/160th of an Acre
                      # Alternative: 225 sqft (common in some urban areas like Lahore)
    'hectare': 107639.104 # 1 Hectare = 107639.104 Square Feet (approx)
}

def convert_area_to_sqft(area_str):
    if pd.isna(area_str) or not isinstance(area_str, str) or not area_str.strip():
        return np.nan # Return NaN for NaN, non-string, or empty strings

    area_str_clean = area_str.strip().lower() # Clean and standardize case

    # Regular expression to extract number and unit
    # This regex looks for:
    # (\\d+\\.?\\d*): one or more digits, optionally followed by a dot and more digits (for decimals)
    # \\s*: zero or more whitespace characters
    # ([a-z.]+): one or more letters, allowing for periods (e.g., 'sq.m', 'sq.yd')
    match = re.match(r'(\\d+\\.?\\d*)\\s*([a-z.]+)', area_str_clean)

    if match:
        value = float(match.group(1))
        unit = match.group(2).replace('.', '') # Remove periods from unit (e.g., 'sq.m' -> 'sqm')

        if unit in CONVERSION_FACTORS:
            return value * CONVERSION_FACTORS[unit]
        else:
            # print(f"Warning: Unknown unit '{unit}' found for value '{area_str}'. Returning NaN.")
            return np.nan # Return NaN for unknown units
    else:
        # If no unit is found, try to convert the string directly to float.
        # This handles cases where only a number (assumed to be sqft) is present.
        try:
            return float(area_str_clean)
        except ValueError:
            # print(f"Warning: Could not parse '{area_str}'. Returning NaN.")
            return np.nan # Return NaN for strings that are not numbers or recognized units

def parse_parking_details(parking_str):
    parking_type = 'Not Available' # Default for NaN/unparseable

    # Handle NaN, non-string, or empty strings
    if pd.isna(parking_str) or not isinstance(parking_str, str) or not parking_str.strip():
        return parking_type # Return default 'Not Available'

    parking_str = parking_str.strip().lower().replace(',', '') # Clean and standardize

    # Check for 'not available' first
    if parking_str == 'not available':
        return 'Not Available'

    # Regex to find number and type (e.g., "10 covered", "5 open")
    # We are still using the regex to identify 'covered' or 'open' types,
    # even if we don't extract the count.
    match = re.search(r'(\\d+)\\s*(covered|open)', parking_str)
    if match:
        type_str = match.group(2)     # Extract the type ('covered' or 'open')
        return type_str.capitalize() # Capitalize the type (e.g., 'Covered')

    # If no specific pattern matched, it's an unhandled case.
    # In this scenario, we return the default 'Not Available' type.
    return 'Not Available'

def parse_floor_info(floor_str):
    flat_floor = np.nan
    total_floors = np.nan

    if pd.isna(floor_str) or not isinstance(floor_str, str) or not floor_str.strip():
        return flat_floor, total_floors # Return NaNs for invalid/missing strings

    floor_str = floor_str.strip().lower()

    # --- 1. Handle "X out of Y" patterns (most common in your data) ---
    # Matches "number out of number"
    match_num_out_of_num = re.match(r'(\\d+)\\s*out of\\s*(\\d+)', floor_str)
    if match_num_out_of_num:
        flat_floor = float(match_num_out_of_num.group(1))
        total_floors = float(match_num_out_of_num.group(2))
        return flat_floor, total_floors

    # Matches "ground out of number"
    match_ground_out_of_num = re.match(r'ground\\s*out of\\s*(\\d+)', floor_str)
    if match_ground_out_of_num:
        flat_floor = 0.0 # Ground floor
        total_floors = float(match_ground_out_of_num.group(1))
        return flat_floor, total_floors

    # Matches "upper basement out of number"
    match_ub_out_of_num = re.match(r'upper basement\\s*out of\\s*(\\d+)', floor_str)
    if match_ub_out_of_num:
        flat_floor = -1.0 # Upper basement
        total_floors = float(match_ub_out_of_num.group(1))
        return flat_floor, total_floors

    # Matches "lower basement out of number"
    match_lb_out_of_num = re.match(r'lower basement\\s*out of\\s*(\\d+)', floor_str)
    if match_lb_out_of_num:
        flat_floor = -2.0 # Lower basement
        total_floors = float(match_lb_out_of_num.group(1))
        return flat_floor, total_floors

    # --- 2. Handle standalone textual descriptions (without "out of Y") ---
    if floor_str in ['ground floor', 'ground', 'g']:
        flat_floor = 0.0
        # total_floors remains NaN as it's not specified
        return flat_floor, total_floors
    elif floor_str in ['first floor', 'first', '1st']:
        flat_floor = 1.0
        return flat_floor, total_floors
    elif floor_str in ['second floor', 'second', '2nd']:
        flat_floor = 2.0
        return flat_floor, total_floors
    elif floor_str in ['basement', 'b', 'upper basement', 'lower basement']:
        # If no "out of Y" given, default basement to -1.0
        # You could refine this to -1.0 for UB, -2.0 for LB if they appear standalone.
        flat_floor = -1.0
        return flat_floor, total_floors

    # --- 3. Handle 'Top Floor', 'Penthouse', 'Multiple Floors' (ambiguous for numbers) ---
    if floor_str in ['top floor', 'penthouse', 'multiple floors', 'duplex', 'triplex']:
        # Both remain NaN as it's hard to assign specific numbers reliably
        return flat_floor, total_floors

    # --- 4. Handle simple numbers (e.g., '2', '10') ---
    try:
        flat_floor = float(floor_str)
        # total_floors remains NaN
        return flat_floor, total_floors
    except ValueError:
        pass # Not a simple number, continue to regex attempts

    # --- 5. Handle ranges like "3-5" (less common but good to keep) ---
    if '-' in floor_str:
        parts = floor_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            flat_floor = (float(parts[0]) + float(parts[1])) / 2.0
            # total_floors remains NaN
            return flat_floor, total_floors

    # --- 6. Final fallback: If nothing matched, return NaN for both ---
    # print(f"Warning: Could not parse '{floor_str}'. Returning NaN for both floor numbers.") # Uncomment for debugging
    return np.nan, np.nan

def clean_Bathroom_count(Bathroom_str):
    if pd.isna(Bathroom_str) or not isinstance(Bathroom_str, str):
        return np.nan

    Bathroom_str_lower = Bathroom_str.strip().lower()

    if Bathroom_str_lower == 'not available':
        return np.nan
    elif Bathroom_str_lower == '> 10': # If this value exists, handle it
        return 11.0 # return 11.0
    else:
        try:
            return float(Bathroom_str_lower)
        except ValueError:
            # Handle other potential non-numeric strings if they appear
            return np.nan

# Load and preprocess data
def load_data():
    df = pd.read_csv('house price prediction/house_prices_small.csv')
    df = df.rename(columns = {'Amount(in rupees)':'Amount','Price (in rupees)':'Price'})
    df = df.drop(['Index','Dimensions','Plot Area','Society','Title','Description','Price','Super Area'],axis = 1)
    df = df.dropna(subset=['Amount','Status','Floor','Furnishing','Bathroom','Transaction','Carpet Area'])
    df['facing'] = df['facing'].fillna('Unknown')
    df['overlooking'] = df['overlooking'].fillna('Not Available')
    df['Ownership'] = df['Ownership'].fillna('Not Available')
    df['Car Parking'] = df['Car Parking'].fillna('Not Available')
    df['Balcony'] = df['Balcony'].fillna(0000)
    df['Amount_Cleaned'] = df['Amount'].apply(parse_amount)
    df['Amount'] = df['Amount_Cleaned']
    df = df.drop(columns=['Amount_Cleaned'])
    column_to_convert = ['Carpet Area']
    for col in column_to_convert:
        df[f'{col}_sqft'] = df[col].apply(convert_area_to_sqft)
    df = df.drop(columns=column_to_convert)
    df['parking_type'] = df['Car Parking'].apply(parse_parking_details)
    df = df.drop('Car Parking', axis=1)
    df[['flat_floor', 'total_floors']] = df['Floor'].apply(lambda x: pd.Series(parse_floor_info(x)))
    df = df.drop('Floor',axis=1)
    df['overlooking'] = df['overlooking'].replace({'Pool, Garden/Park, Main Road':'Garden/Park, Pool, Main Road',
                                              'Main Road, Garden/Park':'Garden/Park, Main Road',
                                              'Main Road, Garden/Park, Pool':'Garden/Park, Pool, Main Road',
                                              'Pool, Garden/Park':'Garden/Park, Pool',
                                              'Garden/Park, Main Road, Pool':'Garden/Park, Pool, Main Road',
                                              'Main Road, Pool, Garden/Park':'Garden/Park, Pool, Main Road',
                                              'Pool, Main Road, Garden/Park':'Garden/Park, Pool, Main Road',
                                              'Main Road, Pool':'Pool, Main Road'})
    df['Bathroom_numeric'] = df['Bathroom'].apply(clean_Bathroom_count)
    df = df.drop('Bathroom',axis=1)
    df['Balcony_numeric'] = df['Balcony'].apply(clean_balcony_count)
    df = df.drop('Balcony',axis=1)
    df = df.dropna(subset = ['Amount','Balcony_numeric','total_floors','Carpet Area_sqft'])
    df = df.rename(columns={'Bathroom_numeric':'Bathrooms','Balcony_numeric':'Balcony','flat_floor':'Flat_floor','total_floors':'Total_floors',
                       'location':'Location','facing':'Facing','Carpet Area_sqft':'Carpet_Area','parking_type':'Parking_type'})

    columns_to_check = ['Carpet_Area', 'Flat_floor', 'Total_floors', 'Bathrooms', 'Balcony']

    non_outlier_mask = pd.Series(True, index=df.index)

    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        current_col_outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        non_outlier_mask = non_outlier_mask & (~current_col_outliers_mask)

    df = df[non_outlier_mask].copy()
    return df

df_encoded = load_data()
df_encoded = pd.get_dummies(df_encoded, columns=[
    'Location',
    'Transaction',
    'Furnishing',
    'Facing',
    'overlooking',
    'Ownership',
    'Parking_type'
], drop_first=False)
df_encoded = df_encoded.drop('Status', axis=1)

# Train the model
X = df_encoded.drop(columns='Amount')
y = df_encoded['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

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
st.write(f"The predicted house price is {prediction[0]:,.2f} rupees.")

def clean_balcony_count(balcony_str):
    if pd.isna(balcony_str) or not isinstance(balcony_str, str):
        return np.nan

    balcony_str_lower = balcony_str.strip().lower()

    if balcony_str_lower == 'not available':
        return np.nan
    elif balcony_str_lower == '> 10': # If this value exists, handle it
        return 11.0 # Let KNN impute, or return 11.0
    else:
        try:
            return float(balcony_str_lower)
        except ValueError:
            # Handle other potential non-numeric strings if they appear
            return np.nan
