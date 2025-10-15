import numpy as np
import pandas as pd

def initialize_centroids(data, k):
    # Randomly select k data points as initial centroids
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # Assign each data point to the nearest centroid
    distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)

def update_centroids(data, clusters, k):
    # Update centroids based on the mean of points in each cluster
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iterations=100):
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):
        # Assign data points to clusters
        clusters = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, clusters, k)
        
        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids