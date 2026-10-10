import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def visualize_results():
    # Load clustered data and artifacts
    df = pd.read_csv('clustered_customers.csv')
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')

    rfm_scaled = scaler.transform(df[['Recency', 'Frequency', 'Monetary']])

    # Calculate Elbow and Silhouette metrics for plotting
    sse = []
    silhouette_scores = []
    K_range = range(2, 10)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        sse.append(km.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, km.labels_))

    # Create 2x2 Plot Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Means Customer Segmentation Dashboard', fontsize=16, fontweight='bold')

    # 1. Elbow Method Plot
    axes[0, 0].plot(K_range, sse, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_title('Elbow Method (Inertia vs K)')
    axes[0, 0].set_xlabel('Number of Clusters (K)')
    axes[0, 0].set_ylabel('Sum of Squared Errors (SSE)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Silhouette Score Plot
    axes[0, 1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Silhouette Score vs K')
    axes[0, 1].set_xlabel('Number of Clusters (K)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. 2D Cluster Scatter: Recency vs Monetary
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    
    sns.scatterplot(
        data=df, x='Recency', y='Monetary', hue='Cluster',
        palette='tab10', ax=axes[1, 0], alpha=0.7, s=60
    )
    axes[1, 0].scatter(
        centroids_unscaled[:, 0], centroids_unscaled[:, 2],
        color='black', marker='X', s=200, label='Centroids'
    )
    axes[1, 0].set_title('Clusters: Recency vs Monetary Spend')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. 2D Cluster Scatter: Frequency vs Monetary
    sns.scatterplot(
        data=df, x='Frequency', y='Monetary', hue='Cluster',
        palette='tab10', ax=axes[1, 1], alpha=0.7, s=60
    )
    axes[1, 1].scatter(
        centroids_unscaled[:, 1], centroids_unscaled[:, 2],
        color='black', marker='X', s=200, label='Centroids'
    )
    axes[1, 1].set_title('Clusters: Frequency vs Monetary Spend')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_results()