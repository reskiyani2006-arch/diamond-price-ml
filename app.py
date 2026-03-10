import streamlit as st
import joblib
import numpy as np

# load model dan scaler
model = joblib.load("model_xgboost.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Harga Diamond")

st.write("Masukkan spesifikasi diamond untuk memprediksi harga")

# input numerik
carat = st.number_input("Carat")
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")

# input kategori (huruf)
cut = st.selectbox(
    "Cut",
    ["Fair","Good","Very Good","Premium","Ideal"]
)

color = st.selectbox(
    "Color",
    ["D","E","F","G","H","I","J"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
)

# mapping huruf ke angka
cut_map = {
    "Fair":0,
    "Good":1,
    "Very Good":2,
    "Premium":3,
    "Ideal":4
}

color_map = {
    "D":0,
    "E":1,
    "F":2,
    "G":3,
    "H":4,
    "I":5,
    "J":6
}

clarity_map = {
    "I1":0,
    "SI2":1,
    "SI1":2,
    "VS2":3,
    "VS1":4,
    "VVS2":5,
    "VVS1":6,
    "IF":7
}

# ubah huruf menjadi angka
cut = cut_map[cut]
color = color_map[color]
clarity = clarity_map[clarity]

# prediksi
if st.button("Prediksi Harga"):

    data = np.array([[carat, depth, table, x, y, z, cut, color, clarity]])

    data_scaled = scaler.transform(data)

    hasil = model.predict(data_scaled)

    st.success(f"Prediksi Harga Diamond: ${hasil[0]:,.2f}")
