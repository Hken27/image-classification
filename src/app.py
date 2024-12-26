import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

model_cnn = load_model(r"D:\KULIAH\Semester VII\Machine Learning\Praktikum_UAP\src\cnn_model.h5")
model_mobilenet = load_model(r"D:\KULIAH\Semester VII\Machine Learning\Praktikum_UAP\src\MobileNet.h5")

# Kelas pembungkus biskuit (ganti dengan kelas yang sesuai)
class_names = [
    "Americana",
    "Amul Chocolate",
    "Amul Elaichi Rusk",
    "Bhagwati Choco Vanilla Puff Biscuits",
    "Bhagwati Lemony Puff Biscuits",
    "Bisk Farm Sugar Free Biscuits",
    "Bonn Jeera Bite Biscuits",
    "Britannia 50-50 Maska Chaska",
    "Britannia 50-50 Maska Chaska Salted Biscuits",
    "Britannia 50-50 Potazos - Masti Masala",
    "Britannia 50-50 Sweet and Salty Biscuits",
    "Britannia 50-50 Timepass Classic Salted Biscuit",
    "Britannia Biscafe Coffee Cracker",
    "Britannia Bourbon",
    "Britannia Bourbon The Original Cream Biscuits",
    "Britannia Chocolush - Pure Magic",
    "Britannia Good Day - Chocochip Cookies",
    "Britannia Good Day Cashew Almond Cookies",
    "Britannia Good Day Harmony Biscuit",
    "Britannia Good Day Pista Badam Cookies",
    "Britannia Little Hearts",
    "Britannia Marie Gold Biscuit",
    "Britannia Milk Bikis Milk Biscuits",
    "Britannia Nice Time - Coconut Biscuits",
    "Britannia Nutri Choice Oats Cookies - Chocolate and Almonds",
    "Britannia Nutri Choice Oats Cookies - Orange With Almonds",
    "Britannia Nutri Choice Seed Biscuits",
    "Britannia Nutri Choice Sugar Free Cream Cracker Biscuits",
    "Britannia Nutrichoice Herbs Biscuits",
    "Britannia Tiger Glucose Biscuit",
    "Britannia Tiger Kreemz - Chocolate Cream Biscuits",
    "Britannia Tiger Kreemz - Elaichi Cream Biscuits",
    "Britannia Tiger Kreemz - Orange Cream Biscuits",
    "Britannia Tiger Krunch Chocochips Biscuit",
    "Britannia Treat Chocolate Cream Biscuits",
    "Britannia Treat Crazy Pineapple Cream Biscuit",
    "Britannia Treat Jim Jam Cream Biscuit",
    "Britannia Treat Osom Orange Cream Biscuit",
    "Britannia Vita Marie Gold Biscuits",
    "Cadbury Bournvita Biscuits",
    "Cadbury Chocobakes Choc Filled Cookies",
    "Cadbury Oreo Chocolate Flavour Biscuit Cream Sandwich",
    "Cadbury Oreo Strawberry Flavour Creme Sandwich Biscuit",
    "Canberra Big Orange Cream Biscuits",
    "CookieMan Hand Pound Chocolate Cookies",
    "Cremica Coconut Cookies",
    "Cremica Elaichi Sandwich Biscuits",
    "Cremica Jeera Lite",
    "Cremica Non-Stop Thin Potato Crackers - Baked, Crunchy Masala",
    "Cremica Orange Sandwich Biscuits",
    "Krown Black Magic Cream Biscuits",
    "MARIO Coconut Crunchy Biscuits",
    "McVities Bourbon Cream Biscuits",
    "McVities Dark Cookie Cream",
    "McVities Marie Biscuit",
    "Parle 20-20 Cashew Cookies",
    "Parle 20-20 Nice Biscuits",
    "Parle Happy Happy Choco-Chip Cookies",
    "Parle Hide and Seek",
    "Parle Hide and Seek - Black Bourbon Choco",
    "Parle Hide and Seek - Milano Choco Chip Cookies",
    "Parle Hide and Seek Caffe Mocha Cookies",
    "Parle Hide and Seek Chocolate and Almonds",
    "Parle Krack Jack Original Sweet and Salty Cracker Biscuit",
    "Parle Krackjack Biscuits",
    "Parle Magix Sandwich Biscuits - Chocolate",
    "Parle Milk Shakti Biscuits",
    "Parle Monaco Biscuit - Classic Regular",
    "Parle Monaco Piri Piri",
    "Parle Platina Hide and Seek Creme Sandwich - Vanilla",
    "Parle-G Gold Gluco Biscuits",
    "Parle-G Original Gluco Biscuits",
    "Patanjali Doodh Biscuit",
    "Priyagold Butter Delite Biscuits",
    "Priyagold Cheese Chacker Biscuits",
    "Priyagold CNC Biscuits",
    "Priyagold Snacks Zig Zag Biscuits",
    "Richlite Rich Butter Cookies",
    "RiteBite Max Protein 7 Grain Breakfast Cookies - Cashew Delite",
    "Sagar Coconut Munch Biscuits",
    "Sri Sri Tattva Cashew Nut Cookies",
    "Sri Sri Tattva Choco Hazelnut Cookies",
    "Sri Sri Tattva Coconut Cookies",
    "Sri Sri Tattva Digestive Cookies",
    "Sunfeast All Rounder - Cream and Herb",
    "Sunfeast All Rounder - Thin, Light and Crunchy Potato Biscuit",
    "Sunfeast Bounce Creme Biscuits",
    "Sunfeast Bounce Creme Biscuits - Elaichi",
    "Sunfeast Bounce Creme Biscuits - Pineapple Zing",
    "Sunfeast Dark Fantasy - Choco Creme",
    "Sunfeast Dark Fantasy Bourbon Biscuits",
    "Sunfeast Dark Fantasy Choco Fills",
    "Sunfeast Glucose Biscuits",
    "Sunfeast Moms Magic - Fruit and Milk Cookies",
    "Sunfeast Moms Magic - Rich Butter Cookies",
    "Sunfeast Moms Magic - Rich Cashew and Almond Cookies",
    "Tasties Chocochip Cookies",
    "Tasties Coconut Cookies",
    "UNIBIC Choco Chip Cookies",
    "UNIBIC Pista Badam Cookies",
    "UNIBIC Snappers Potato Crackers"
]

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Ukuran input model (sesuaikan dengan ukuran input model)
    image_array = np.array(image) / 255.0  # Normalisasi gambar
    image_array = np.expand_dims(image_array, axis=0)  # Menambahkan dimensi batch
    return image_array

# Fungsi untuk memprediksi gambar menggunakan model yang dipilih
def predict_image(image, model):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    return prediction

# Fungsi untuk mengambil gambar dari URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# UI Streamlit
st.title("Aplikasi Prediksi Pembungkus Biskuit")

st.sidebar.header("Pilih Input Gambar")
image_source = st.sidebar.selectbox("Pilih sumber gambar", ["Unggah Gambar", "Masukkan URL"])

if image_source == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)  # Perbarui di sini
        
        # Melakukan prediksi dengan MobileNet
        if st.button("Prediksi dengan MobileNet"):
            prediction_mobilenet = predict_image(image, model_mobilenet)
            predicted_class_mobilenet = np.argmax(prediction_mobilenet, axis=1)
            predicted_prob_mobilenet = np.max(prediction_mobilenet)  # Nilai probabilitas tertinggi
            st.write(f'Prediksi MobileNet: {class_names[predicted_class_mobilenet[0]]}')
            st.write(f'Akurasi MobileNet: {predicted_prob_mobilenet * 100:.2f}%')

        # Melakukan prediksi dengan CNN
        if st.button("Prediksi dengan CNN"):
            prediction_cnn = predict_image(image, model_cnn)
            predicted_class_cnn = np.argmax(prediction_cnn, axis=1)
            predicted_prob_cnn = np.max(prediction_cnn)  # Nilai probabilitas tertinggi
            st.write(f'Prediksi CNN: {class_names[predicted_class_cnn[0]]}')
            st.write(f'Akurasi CNN: {predicted_prob_cnn * 100:.2f}%')
    
elif image_source == "Masukkan URL":
    image_url = st.text_input("Masukkan URL Gambar")
    if image_url:
        try:
            image = load_image_from_url(image_url)
            st.image(image, caption="Gambar dari URL", use_container_width=True)  # Perbarui di sini
            
            # Melakukan prediksi dengan MobileNet
            if st.button("Prediksi dengan MobileNet"):
                prediction_mobilenet = predict_image(image, model_mobilenet)
                predicted_class_mobilenet = np.argmax(prediction_mobilenet, axis=1)
                predicted_prob_mobilenet = np.max(prediction_mobilenet)  # Nilai probabilitas tertinggi
                st.write(f'Prediksi MobileNet: {class_names[predicted_class_mobilenet[0]]}')
                st.write(f'Akurasi MobileNet: {predicted_prob_mobilenet * 100:.2f}%')

            # Melakukan prediksi dengan CNN
            if st.button("Prediksi dengan CNN"):
                prediction_cnn = predict_image(image, model_cnn)
                predicted_class_cnn = np.argmax(prediction_cnn, axis=1)
                predicted_prob_cnn = np.max(prediction_cnn)  # Nilai probabilitas tertinggi
                st.write(f'Prediksi CNN: {class_names[predicted_class_cnn[0]]}')
                st.write(f'Akurasi CNN: {predicted_prob_cnn * 100:.2f}%')
        except Exception as e:
            st.error(f"Gagal memuat gambar dari URL: {e}")
