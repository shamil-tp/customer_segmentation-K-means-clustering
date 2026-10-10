import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def train_kmeans_model(data_path="customer_rfm_data.csv", optimal_k=4):
    df = pd.read_csv(data_path)

    # Feature Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[["Recency", "Frequency", "Monetary"]])

    # Determine Metrics for K range (Elbow & Silhouette)
    sse = []
    silhouette_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

    # Fit Final K-Means Model
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = final_kmeans.fit_predict(rfm_scaled)

    # Save output dataset with cluster tags
    df.to_csv("clustered_customers.csv", index=False)

    # Save fitted model and scaler artifacts
    joblib.dump(final_kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Display Cluster Summary
    cluster_summary = (
        df.groupby("Cluster")
        .agg(
            {
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "CustomerID": "count",
            }
        )
        .rename(columns={"CustomerID": "CustomerCount"})
        .round(2)
    )

    print("--- Cluster Averages (Unscaled) ---")
    print(cluster_summary)

    return df, final_kmeans, scaler


if __name__ == "__main__":
    train_kmeans_model()
