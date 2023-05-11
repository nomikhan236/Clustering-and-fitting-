import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(data_file):
    # Read the data into a DataFrame
    data = pd.read_csv(data_file)

    # Select relevant columns for clustering
    columns_to_cluster = ['1960', '1961', '1962', '1963', '1964', '1965', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    df = data[columns_to_cluster].copy()

    # Clean and preprocess the data
    df = df.replace('NaN', np.nan)
    df = df.replace(',', '', regex=True).astype(float)
    df = df.dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform clustering using K-means
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_scaled)
    cluster_labels = kmeans.labels_
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Add cluster labels to the DataFrame
    df['Cluster'] = cluster_labels

    # Plot the cluster membership and cluster centers
    plt.figure(figsize=(12, 6))
    plt.scatter(df.index, df['2020'], c=df['Cluster'], cmap='viridis', alpha=0.7)
    plt.scatter(range(len(cluster_centers)), cluster_centers[:, -1], c='red', marker='X', s=100, label='Cluster Centers')
    plt.xlabel('Country')
    plt.ylabel('CO2 emissions (kt) in 2020')
    plt.title('Clustering of Countries based on CO2 emissions')
    plt.legend()
    plt.show()

# Example usage:
data_file = 'Complete DS 2.csv'
perform_clustering(data_file)
