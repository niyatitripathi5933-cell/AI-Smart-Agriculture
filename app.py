import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Smart Agriculture", layout="wide")

st.title("🌾 AI Powered Smart Agriculture System")
st.write("Crop Recommendation | Yield Prediction | Disease Detection")

# -------------------------------
# Sidebar Menu
# -------------------------------
menu = st.sidebar.selectbox("Select Module", 
                             ["Crop Recommendation", 
                              "Yield Prediction", 
                              "Disease Detection"])

# ================================
# 1️⃣ CROP RECOMMENDATION
# ================================
if menu == "Crop Recommendation":

    st.header("🌱 Crop Recommendation")

    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("Soil pH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

    if st.button("Predict Crop"):

        model = joblib.load("models/crop_model.pkl")

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.success(f"🌾 Recommended Crop: {prediction[0]}")
        st.info(f"Confidence Score: {np.max(probability)*100:.2f}%")

# ================================
# 2️⃣ YIELD PREDICTION
# ================================
elif menu == "Yield Prediction":

    st.header("📈 Crop Yield Prediction")

    area = st.number_input("Area (Hectares)", 0.0, 1000.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0)

    if st.button("Predict Yield"):

        model = joblib.load("models/yield_model.pkl")

        input_data = np.array([[area, rainfall, temperature]])
        prediction = model.predict(input_data)

        st.success(f"🌾 Estimated Yield: {prediction[0]:.2f} tons")

# ================================
# 3️⃣ DISEASE DETECTION
# ================================
elif menu == "Disease Detection":

    st.header("🌿 Crop Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((128,128))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = tf.keras.models.load_model("models/disease_model.h5")
        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.error("⚠ Diseased Leaf Detected")
            st.write("💊 Suggested Treatment: Use organic fungicide spray.")
        else:
            st.success("✅ Healthy Leaf")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("Developed by Niyati Tripathi 💚")
