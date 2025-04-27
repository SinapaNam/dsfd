# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 21:05:08 2025

@author: Useru
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Page settings
st.set_page_config(page_title="K-Means Clustering App with Iris", layout="wide")

# App title
st.markdown("<h1 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset by SINAPA NAMPANMUEANG</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Load Iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# K-Means model
model = KMeans(n_clusters=k, random_state=42)
y_kmeans = model.fit_predict(X)


# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(model.cluster_centers_)
# Dimensionality reduction for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])


# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='tab10', s=50)


# Add legend with cluster labels
handles, labels = scatter.legend_elements()
labels = [f"Cluster {i}" for i in range(len(handles))]
ax.legend(handles, labels, title="Clusters")

# Show plot in Streamlit
st.pyplot(fig)
st.dataframe(reduced_df.head(10))
