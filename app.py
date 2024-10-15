import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from clustering import process_image, initialize_centroids, kmeans_manual, predict_clustering, visualize_clusters, show_image_with_legend

# Judul Aplikasi
st.title("K-Means Clustering pada Gambar")

# Input untuk mengunggah gambar pelatihan
st.subheader("Unggah Gambar untuk Training")
train_images = st.file_uploader("Pilih gambar untuk folder Training", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Input untuk mengunggah gambar pengujian
st.subheader("Unggah Gambar untuk Testing")
test_images = st.file_uploader("Pilih gambar untuk folder Testing", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Input untuk jumlah cluster
k = st.number_input("Masukkan jumlah cluster (k)", min_value=1, value=3)

# Tombol untuk melatih model
if st.button("Train"):
    all_pixels = []
    for file in train_images:
        img = process_image(file)
        if img is not None:
            all_pixels.append(img)
    all_pixels = np.vstack(all_pixels)
    st.session_state.pixels = all_pixels
    st.session_state.stcentroids = initialize_centroids(all_pixels, k)
    st.session_state.centroids = kmeans_manual(st.session_state.pixels, k, st.session_state.stcentroids)
    st.success("Model berhasil dilatih!")

# Tombol untuk memprediksi
if st.button("Predict"):
    if 'pixels' not in st.session_state or 'centroids' not in st.session_state:
        st.warning("Silakan latih model terlebih dahulu.")
    else:
        all_images = []
        for file in test_images:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image.reshape(-1, 3)
            labels = predict_clustering(pixels, st.session_state.centroids)
            clustered_image = visualize_clusters(image, labels, k, st.session_state.centroids)

            # Tampilkan gambar dengan legenda
            show_image_with_legend(clustered_image, st.session_state.centroids, k, "Hasil Klustering")

            all_images.append(clustered_image)

        # Tampilkan semua gambar tercluster
        for img in all_images:
            st.image(img, caption="Hasil Klustering", use_column_width=True)
