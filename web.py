import streamlit as st
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Fungsi untuk membaca dan meresample gambar
def preprocess_image(image):
    img = cv2.resize(image, (100, 100))
    return img

# Fungsi untuk mengekstrak fitur HSV dan histogram
def extract_features(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Fungsi clustering KMeans
def kmeans_clustering(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_
    return labels, centroids

# Visualisasi histogram
def extract_and_show_histogram(hsv_img):
    plt.figure(figsize=(12, 4))

    # Plot histogram untuk Hue
    plt.subplot(1, 3, 1)
    plt.title('Hue Channel')
    plt.hist(hsv_img[:, :, 0].ravel(), bins=180, range=[0, 180], color='r')
    plt.xlabel('Hue')
    plt.ylabel('Frequency')

    # Plot histogram untuk Saturation
    plt.subplot(1, 3, 2)
    plt.title('Saturation Channel')
    plt.hist(hsv_img[:, :, 1].ravel(), bins=256, range=[0, 256], color='g')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')

    # Plot histogram untuk Value
    plt.subplot(1, 3, 3)
    plt.title('Value Channel')
    plt.hist(hsv_img[:, :, 2].ravel(), bins=256, range=[0, 256], color='b')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(plt)

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Image Clustering Application")
    
    # Upload gambar
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    # Parameter clustering
    num_clusters = st.slider('Number of Clusters (K)', min_value=2, max_value=10, value=4)
    
    if uploaded_files:
        features = []
        label_images = []
        for file in uploaded_files:
            # Baca dan tampilkan gambar
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, caption=file.name, use_column_width=True)
            
            # Preprocessing dan ekstraksi fitur
            processed_img = preprocess_image(image)
            feature = extract_features(processed_img)
            features.append(feature)
            label_images.append(file.name)
        
        # Konversi fitur menjadi array numpy
        features = np.array(features)
        
        # Clustering KMeans
        labels, centroids = kmeans_clustering(features, num_clusters)
        
        # Visualisasi hasil clustering
        st.subheader(f"Clustering Results for {num_clusters} Clusters")
        for i in range(num_clusters):
            st.write(f"Cluster {i+1}:")
            cluster_images = [label_images[j] for j in range(len(label_images)) if labels[j] == i]
            st.write(cluster_images)
        
        # Perhitungan silhouette score
        sil_score = silhouette_score(features, labels)
        st.write(f"Silhouette Score: {sil_score:.3f}")
        
        # Plot clustering result
        plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
        plt.title(f'Clustering Results (Silhouette Score: {sil_score:.3f})')
        st.pyplot(plt)

if __name__ == '__main__':
    main()
