# 📊 E-Commerce Customer Segmentation using K-Means Clustering

An end-to-end Machine Learning pipeline that uses **RFM (Recency, Frequency, Monetary)** analysis and **K-Means Clustering** to segment e-commerce customers into actionable personas for targeted marketing.

---

## 📌 Project Architecture & Features

This project is built in a clean, modular structure split into three main components:

1. **`data_generator.py`**: Generates a synthetic dataset of 500 e-commerce customers containing realistic shopping behaviors (Recency, Frequency, and Monetary spend).
2. **`model_creator.py`**: Handles feature scaling using `StandardScaler`, evaluates optimal cluster counts ($K$), fits the final K-Means model, exports the tagged customer dataset, and saves serialized model artifacts (`kmeans_model.pkl` and `scaler.pkl`).
3. **`visualize_clusters.py`**: Loads trained artifacts and creates a 4-panel Matplotlib/Seaborn visualization dashboard including:
   * **Elbow Method Curve** (Inertia vs. $K$)
   * **Silhouette Score Evaluation**
   * **2D Scatter Plot**: Recency vs. Monetary Spend (with Centroids)
   * **2D Scatter Plot**: Frequency vs. Monetary Spend (with Centroids)

---

## 🛠️ RFM Metrics Explained

* **Recency ($R$)**: Days since the customer's last purchase.
* **Frequency ($F$)**: Total number of purchases made by the customer.
* **Monetary ($M$)**: Total monetary value spent by the customer.

---

## 📁 Directory Structure

```text
customer-segmentation-kmeans/
├── data_generator.py         # Step 1: Generates synthetic customer data
├── model_creator.py          # Step 2: Trains model & exports artifacts
├── visualize_clusters.py     # Step 3: Plots evaluation & cluster metrics
├── customer_rfm_data.csv     # Generated raw RFM dataset
├── clustered_customers.csv   # Processed dataset with Cluster tags
├── kmeans_model.pkl          # Saved K-Means model artifact
├── scaler.pkl                # Saved StandardScaler artifact
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Execution Steps

Run the modular scripts sequentially:

#### 1. Generate the Dataset
```bash
python data_generator.py
```

#### 2. Train the Model
```bash
python model_creator.py
```

#### 3. Visualize Clusters and Model Diagnostics
```bash
python visualize_clusters.py
```

---

## 📊 Business Segment Personas

Based on $K=4$, the segments map directly to actionable marketing strategies:

| Cluster | Recency | Frequency | Monetary | Target Persona | Recommended Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | Low | High | High | **Champions / VIPs** | Loyalty rewards, early access to new releases |
| **Cluster 1** | Low | Low–Med | Low | **New / Recent Shoppers** | Welcome email series, product recommendations |
| **Cluster 2** | High | Low | Low | **At Risk / Hibernating** | Win-back discount coupons, re-engagement emails |
| **Cluster 3** | High | High | High | **Can't Lose Them** | High-value retention offers, direct support outreach |

---

## 📄 License

This repository is available under the [MIT License](LICENSE).