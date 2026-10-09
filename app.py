import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# LOAD MODELS
# =========================================================

rf_model = joblib.load("model/aqi_random_forest.pkl")

nn_model = load_model("model/aqi_neural_network.keras")

nn_scaler = joblib.load("model/nn_scaler.pkl")


# =========================================================
# TITLE
# =========================================================

st.title("🌿 Air Quality Index Prediction")

st.write(
    "Enter the average pollutant values to predict the Air Quality Index (AQI)."
)


# =========================================================
# INPUT FEATURES
# =========================================================

st.subheader("Enter Pollutant Values")


col1, col2 = st.columns(2)

with col1:

    pm25 = st.number_input(
        "PM2.5 Average",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10 Average",
        min_value=0.0,
        value=80.0
    )

    no2 = st.number_input(
        "NO2 Average",
        min_value=0.0,
        value=30.0
    )

    nh3 = st.number_input(
        "NH3 Average",
        min_value=0.0,
        value=20.0
    )


with col2:

    so2 = st.number_input(
        "SO2 Average",
        min_value=0.0,
        value=20.0
    )

    co = st.number_input(
        "CO Average",
        min_value=0.0,
        value=1.0
    )

    ozone = st.number_input(
        "OZONE Average",
        min_value=0.0,
        value=30.0
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict AQI"):

    # -----------------------------------------------------
    # Create input DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [[
            pm25,
            pm10,
            no2,
            nh3,
            so2,
            co,
            ozone
        ]],
        columns=[
            "PM2.5_Avg",
            "PM10_Avg",
            "NO2_Avg",
            "NH3_Avg",
            "SO2_Avg",
            "CO_Avg",
            "OZONE_Avg"
        ]
    )


    # =====================================================
    # RANDOM FOREST PREDICTION
    # =====================================================

    rf_prediction = rf_model.predict(input_data)[0]


    # =====================================================
    # NEURAL NETWORK PREDICTION
    # =====================================================

    input_scaled = nn_scaler.transform(input_data)

    nn_prediction = nn_model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Random Forest AQI",
            f"{rf_prediction:.2f}"
        )


    with result_col2:

        st.metric(
            "Neural Network AQI",
            f"{nn_prediction:.2f}"
        )


    # =====================================================
    # BEST MODEL
    # =====================================================

    st.success(
        "🏆 Best Performing Model: Random Forest"
    )

    st.write(
        "Random Forest was selected because it achieved "
        "lower MAE and RMSE and higher R² than the Neural Network "
        "on the test dataset."
    )


    # =====================================================
    # AQI CATEGORY
    # =====================================================

    st.subheader("AQI Category")


    if rf_prediction <= 50:
        category = "Good"

    elif rf_prediction <= 100:
        category = "Satisfactory"

    elif rf_prediction <= 200:
        category = "Moderate"

    elif rf_prediction <= 300:
        category = "Poor"

    elif rf_prediction <= 400:
        category = "Very Poor"

    else:
        category = "Severe"


    st.info(
        f"Predicted AQI: {rf_prediction:.2f} — {category}"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AQI Prediction using Random Forest and Neural Network"
)