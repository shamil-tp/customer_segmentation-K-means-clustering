import pandas as pd
import numpy as np

def generate_customer_data(n_customers=500, filename='customer_rfm_data.csv'):
    # Set seed for reproducibility
    np.random.seed(42)

    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
    recency = np.random.randint(1, 365, size=n_customers)          # Days since last purchase
    frequency = np.random.poisson(lam=5, size=n_customers) + 1     # Total number of purchases
    monetary = frequency * np.random.uniform(20, 150, size=n_customers) # Total spend ($)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })

    df.to_csv(filename, index=False)
    print(f"Dataset with {n_customers} samples saved successfully to '{filename}'.")
    return df

if __name__ == '__main__':
    generate_customer_data()