import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Custom DBSCAN Algorithm
class DBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    def fit(self, X):
        n = len(X)
        self.labels = [-1] * n  # -1 = unclassified
        cluster_id = 0
        visited = [False] * n

        def region_query(p_idx):
            neighbors = []
            for i in range(n):
                if np.linalg.norm(X[i] - X[p_idx]) <= self.eps:
                    neighbors.append(i)
            return neighbors

        def expand_cluster(p_idx, neighbors, cluster_id):
            self.labels[p_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                q_idx = neighbors[i]
                if not visited[q_idx]:
                    visited[q_idx] = True
                    q_neighbors = region_query(q_idx)
                    if len(q_neighbors) >= self.min_pts:
                        neighbors += q_neighbors
                if self.labels[q_idx] == -1:
                    self.labels[q_idx] = cluster_id
                i += 1

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) < self.min_pts:
                self.labels[i] = -2  # noise
            else:
                expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels

# Load and preprocess data
def load_data(csv_file, temp_column='DAYTON_MW'):
    df = pd.read_csv(csv_file)
    
    # Check for required columns
    if not all(col in df.columns for col in ['Datetime', temp_column]):
        raise ValueError(f"Dataset must contain 'Datetime' and '{temp_column}' columns")

    # Drop rows with missing temperature or Datetime
    df = df.dropna(subset=[temp_column, 'Datetime'])
    if df.empty:
        raise ValueError("No valid temperature data found after removing missing values")

    # Convert Datetime to datetime object
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])

    # Create DateOrdinal for DBSCAN
    df['DateOrdinal'] = df['Datetime'].apply(lambda x: x.timestamp() / 3600)  # Convert to hours since epoch

    # Aggregate duplicate datetimes by taking the mean for temperature
    df = df.groupby('Datetime').agg({
        temp_column: 'mean',
        'DateOrdinal': 'first'
    }).reset_index()

    # Check data size
    print(f"Data points: {len(df)}, Datetime range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    if len(df) < 24 * 7:  # Less than a week of hourly data
        print("Warning: Dataset may be too short for robust deseasoning")
    elif len(df) > 24 * 365 * 2:  # More than 2 years of hourly data
        print("Warning: Large dataset may slow down DBSCAN")

    # Compute z-scores by hour of day for deseasoning (using temperature in Celsius)
    df['Hour'] = df['Datetime'].dt.hour
    hourly_stats = df.groupby('Hour')[temp_column].agg(['mean', 'std']).reset_index()
    hourly_stats.columns = ['Hour', 'MeanTemp', 'StdTemp']
    df = df.merge(hourly_stats, on='Hour', how='left')
    df['DeseasonedTemperature'] = (df[temp_column] - df['MeanTemp']) / df['StdTemp']
    df['DeseasonedTemperature'] = df['DeseasonedTemperature'].fillna(0)

    # Ensure data is sorted by datetime and set Datetime as index for reindexing
    df = df.sort_values('Datetime').set_index('Datetime')

    # Handle missing datetimes by reindexing to hourly frequency and interpolating
    df = df.asfreq('h', method='ffill').reset_index()  # Updated to 'h'

    # Recreate DateOrdinal after reindexing
    df['DateOrdinal'] = df['Datetime'].apply(lambda x: x.timestamp() / 3600)

    X = df[['DateOrdinal', 'DeseasonedTemperature']].values
    if len(X) == 0:
        raise ValueError("No valid data points to process")

    return df.reset_index(drop=True), X

# Run DBSCAN for temperature anomaly detection
def find_temperature_anomalies(csv_file, eps=0.3, min_pts=5, temp_column='DAYTON_MW'):
    df, X = load_data(csv_file, temp_column)
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    model = DBSCAN(eps=0.15, min_pts=5)
    labels = model.fit(X_scaled)

    df['Cluster'] = labels
    df['Anomaly'] = df['Cluster'] == -2

    # Plot 1: Deseasoned temperature (z-score) vs datetime
    plt.figure(figsize=(14, 6))
    plt.plot(df['Datetime'], df['DeseasonedTemperature'], label='Deseasoned Temperature (Z-Score)')
    plt.scatter(df[df['Anomaly']]['Datetime'], df[df['Anomaly']]['DeseasonedTemperature'], 
                color='red', label='Anomaly')
    plt.xlabel('Datetime')
    plt.ylabel('Deseasoned Temperature (Z-Score)')
    plt.title('Temperature Anomalies (DBSCAN on Z-Score Deseasoned Data)')
    plt.legend()
    plt.savefig('anomalies_deseasoned.png')
    plt.close()

    # Plot 2: Deseasoned temperature with clusters
    plt.figure(figsize=(14, 6))
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -2 in unique_labels else 0)
    colors = plt.get_cmap('tab10', num_clusters + 1)

    for label in unique_labels:
        cluster_data = df[df['Cluster'] == label]
        if label == -2:
            plt.scatter(cluster_data['Datetime'], cluster_data['DeseasonedTemperature'],
                        color='red', label='Anomaly', marker='o')
        else:
            plt.scatter(cluster_data['Datetime'], cluster_data['DeseasonedTemperature'],
                        color=colors(label), label=f'Cluster {label}', s=10)

    plt.xlabel('Datetime')
    plt.ylabel('Deseasoned Temperature (Z-Score)')
    plt.title('Temperature Clusters and Anomalies (DBSCAN)')
    plt.legend()
    plt.savefig('clusters_deseasoned.png')
    plt.close()

    # Plot 3: Original temperature (Celsius) vs datetime
    plt.figure(figsize=(14, 6))
    plt.plot(df['Datetime'], df[temp_column], label='Temperature (°C)')
    plt.scatter(df[df['Anomaly']]['Datetime'], df[df['Anomaly']][temp_column], 
                color='red', label='Anomaly')
    plt.xlabel('Datetime')
    plt.ylabel('Temperature (°C)')
    plt.title('Original Temperature and Anomalies (DBSCAN)')
    plt.legend()
    plt.savefig('anomalies_temperature_celsius.png')
    plt.close()

    # Plot 4: Original temperature (Celsius) with clusters
    plt.figure(figsize=(14, 6))
    for label in unique_labels:
        cluster_data = df[df['Cluster'] == label]
        if label == -2:
            plt.scatter(cluster_data['Datetime'], cluster_data[temp_column],
                        color='red', label='Anomaly', marker='o')
        else:
            plt.scatter(cluster_data['Datetime'], cluster_data[temp_column],
                        color=colors(label), label=f'Cluster {label}', s=10)

    plt.xlabel('Datetime')
    plt.ylabel('Temperature (°C)')
    plt.title('Original Temperature Clusters and Anomalies (DBSCAN)')
    plt.legend()
    plt.savefig('clusters_temperature_celsius.png')
    plt.close()

    return df

# Example usage
if __name__ == "__main__":
    file_path = "MLTempDataset.csv"  # Your dataset file
    result_df = find_temperature_anomalies(file_path, eps=0.5, min_pts=5, temp_column='DAYTON_MW')
    result_df.to_csv("anomalies_temperature.csv", index=False)