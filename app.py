import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# load the model
xgb_model = joblib.load("xgb_model.pkl")


# function to make predictions
def predict_bike_rentals(model, input_data):
    prediction = model.predict(input_data)
    return prediction


# streamlit
st.title("Bike Rental Prediction")

# user inputs
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=60)
windspeed = st.number_input("Wind Speed (m/s)", value=1.0)
hour = st.slider("Hour of the day", 0, 23, 12)
season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
if season == "Spring":
    seasons_spring = 1
elif season == "Summer":
    seasons_summer = 1
elif season == "Winter":
    seasons_winter = 1
holiday = st.radio("Is it a holiday?", ["Yes", "No"])
holiday_no_holiday = 1 if holiday == "No" else 0

# default values
visibility = 2000
dew_point = 10.0
solar_radiation = 0.5
rainfall = 0.0
snowfall = 0.0
day_of_week = datetime.now().weekday()
month = datetime.now().month
is_weekend = 1 if day_of_week >= 5 else 0
seasons_spring = 0
seasons_summer = 0
seasons_winter = 0
functioning_day_yes = 1

# data preparation
input_data = pd.DataFrame(
    {
        "Hour": [hour],
        "Temperature(°C)": [temperature],
        "Humidity(%)": [humidity],
        "Wind speed (m/s)": [windspeed],
        "Visibility (10m)": [visibility],
        "Dew point temperature(°C)": [dew_point],
        "Solar Radiation (MJ/m2)": [solar_radiation],
        "Rainfall(mm)": [rainfall],
        "Snowfall (cm)": [snowfall],
        "Day of Week": [day_of_week],
        "Month": [month],
        "Is Weekend": [is_weekend],
        "Seasons_Spring": [seasons_spring],
        "Seasons_Summer": [seasons_summer],
        "Seasons_Winter": [seasons_winter],
        "Holiday_No Holiday": [holiday_no_holiday],
        "Functioning Day_Yes": [functioning_day_yes],
    }
)

if st.button("Predict"):
    prediction = predict_bike_rentals(xgb_model, input_data)
    st.write(f"Predicted Bike Rentals: {int(prediction[0])}")
