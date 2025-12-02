import streamlit as st
import pandas as pd
import joblib

def user_input_features():
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    suhu = st.number_input("Suhu (°C)", min_value=0.0, max_value=100.0, value=20.0)
    rasa = st.radio("Rasa", (0, 1), captions=["Buruk", "Baik"])
    bau = st.radio("Bau", (0, 1), captions=["Buruk", "Baik"])
    lemak = st.radio("Lemak", (0, 1), captions=["Rendah", "Tinggi"])
    kekeruhan = st.radio("Kekeruhan", (0, 1), captions=["Keruh", "Jernih"])
    warna = st.number_input("Warna", min_value=240, max_value=255, value=240)

    data = {
        "pH": pH,
        "Temprature": suhu,
        "Taste": rasa,
        "Odor": bau,
        "Fat ": lemak,
        "Turbidity": kekeruhan,
        "Colour": warna,
    }

    features = pd.DataFrame(data, index=[0])
    return features


def predict(features):
    model = joblib.load("model_milk_quality.pkl")
    hasil = model.predict(features)

    return hasil


def run():
    st.title("Milk Quality Prediction")
    st.write("Enter the following parameters to predict milk quality:")

    features = user_input_features()
    result = predict(features)

    if st.button("Prediksi"):
        st.subheader("Hasil Prediksi:")
        st.write(f"Kualitas Susu: {result[0]}")


if __name__ == "__main__":
    run()
