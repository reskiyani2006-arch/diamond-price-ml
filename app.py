import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_xgboost.pkl")

st.title("Prediksi Harga Diamond")

carat = st.number_input("Carat")
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")
cut = st.number_input("Cut")
color = st.number_input("Color")
clarity = st.number_input("Clarity")

if st.button("Prediksi"):

    data = np.array([[carat,depth,table,x,y,z,cut,color,clarity]])

    hasil = model.predict(data)

    st.success(f"Hasil Prediksi: {hasil[0]}")