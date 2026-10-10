import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Create simple customer data
data = pd.DataFrame(
    {"Age": [19, 21, 20, 55, 60, 58], "AnnualSpend": [15, 18, 12, 85, 90, 88]}
)

# 2. Scale the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(data_scaled)

# 4. Find Centroids (In the original unscaled values)
# To get centroids back to original Age/Spend scale, we invert the scaler.
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# 5. Visualize with Matplotlib
plt.figure(figsize=(8, 6))

# Plot actual customer data points, colored by cluster assignment
scatter = plt.scatter(
    data["Age"],
    data["AnnualSpend"],
    c=data["Cluster"],
    cmap="coolwarm",
    s=150,  # point size
    edgecolor="k",
    label="Customers",
)

# Plot the calculated Centroids (Red 'X' markers)
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    color="red",
    marker="X",
    s=300,  # marker size
    edgecolor="black",
    label="Centroids (Cluster Centers)",
)

# Customizing the plot
plt.title("K-Means Clustering: Age vs Annual Spend", fontsize=14, fontweight="bold")
plt.xlabel("Age (Years)", fontsize=12)
plt.ylabel("Annual Spend ($1,000s)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# Add Legend
plt.legend(loc="upper left", fontsize=11)

# Show the Plot
plt.show()
