import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model(
    "artifacts/housing_model.keras",
    compile=False
)

preprocessor = joblib.load("artifacts/preprocessor.pkl")

df = pd.read_csv("data/Housing_Hamilton_Compressed.csv.gz")

land_use_options = sorted(df["LAND_USE_CODE_DESC"].dropna().astype(str).unique())
neighborhood_options = sorted(df["NEIGHBORHOOD_CODE_DESC"].dropna().astype(str).unique())
zoning_options = sorted(df["ZONING_DESC"].dropna().astype(str).unique())
property_type_options = sorted(df["PROPERTY_TYPE_CODE_DESC"].dropna().astype(str).unique())

st.title("Housing Value Prediction")

calc_acres = st.number_input(
    "CALC_ACRES",
    min_value=0.01,
    max_value=100.0,
    value=0.25
)

land_use = st.selectbox(
    "LAND_USE_CODE_DESC",
    land_use_options
)

neighborhood = st.selectbox(
    "NEIGHBORHOOD_CODE_DESC",
    neighborhood_options
)

zoning = st.selectbox(
    "ZONING_DESC",
    zoning_options
)

property_type = st.selectbox(
    "PROPERTY_TYPE_CODE_DESC",
    property_type_options
)

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "CALC_ACRES": calc_acres,
        "LAND_USE_CODE_DESC": land_use,
        "NEIGHBORHOOD_CODE_DESC": neighborhood,
        "ZONING_DESC": zoning,
        "PROPERTY_TYPE_CODE_DESC": property_type
    }])

    X = preprocessor.transform(input_df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    pred = model.predict(X)[0][0]

    st.success(f"Predicted value: ${pred:,.2f}")