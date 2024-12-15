# #################### Files Required For Deployment ################################
#
# # UI : implemented by python streamlit library
# # Trained Model Files: Saved pkl files
# # Logic Code connecting UI & Pkl files
#
# ################# UI ##########################

import streamlit as st
import pandas as pd
import pickle

st.header("Traffic Situation Prediction")

# Load dataset
df = pd.read_csv("Traffic_1.csv")

# Display dataset
st.write("Dataset Overview")
st.dataframe(df.head())

# Taking X column values from user
col1, col2 = st.columns(2)
with col1:
    # Use selectbox for categorical 'Time' field
    choice = st.selectbox('Enter Time:', df['Time'].unique())  # Ensure correct column name

with col2:
    # Use number input for Date (ensure it's a numerical value)
    n = st.number_input(f"Enter Date Min {df.Date.min()} to Max {df.Date.max()}")

col3, col4 = st.columns(2)
with col3:
    # Use selectbox for 'Day of the week'
    choice1 = st.selectbox("Enter Day of the Week:", df['Day of the week'].unique())

with col4:
    # Use number input for 'CarCount'
    p = st.number_input(f"Enter CarCount Min {df.CarCount.min()} to Max {df.CarCount.max()}")

col5, col6 = st.columns(2)
with col5:
    # Use number input for 'BikeCount'
    k = st.number_input(f"Enter BikeCount Min {df.BikeCount.min()} to Max {df.BikeCount.max()}")

with col6:
    # Use number input for 'BusCount'
    t = st.number_input(f"Enter BusCount Min {df.BusCount.min()} to Max {df.BusCount.max()}")

col7, col8 = st.columns(2)
with col7:
    # Use number input for 'TruckCount'
    h = st.number_input(f"Enter TruckCount Min {df.TruckCount.min()} to Max {df.TruckCount.max()}")

with col8:
    # Use number input for 'Total'
    ph = st.number_input(f"Enter Total Min {df.Total.min()} to Max {df.Total.max()}")

# Manually encode categorical values for 'Time' and 'Day of the Week'
time_mapping = {time: idx for idx, time in enumerate(df['Time'].unique())}
day_mapping = {day: idx for idx, day in enumerate(df['Day of the week'].unique())}

# Convert 'choice' and 'choice1' to numeric using the mappings
choice_encoded = time_mapping[choice]
choice1_encoded = day_mapping[choice1]

# Create the xdata list with the encoded values
xdata_encoded = [choice_encoded, n, choice1_encoded, p, k, t, h, ph]

# Load the trained model (xgboost model)
with open('xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Create DataFrame to match the model's expected input format
x = pd.DataFrame([xdata_encoded], columns=['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total'])

# Rename 'Time' to 'Time Category' to match the model's expected feature names
x.rename(columns={'Time': 'Time Category'}, inplace=True)

# Ensure the feature order matches the model's training data
x = x[['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Time Category']]

# Display the input data for verification
st.write("Given Input:")
st.dataframe(x)

# Predict with the model
if st.button("Predict"):
    prediction = model.predict(x)

    # Based on the model's output, display the traffic situation
    if prediction[0] == 0:
        st.write('High Traffic')
    elif prediction[0] == 1:
        st.write('Low Traffic')
    else:
        st.write('Normal Traffic')
