import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def initialize_pheromone(n, tau0=0.001):
    return np.full((n, n), tau0)

def initialize_local_heuristic(X):
    n = X.shape[0]
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                eta[i, j] = 1 / np.linalg.norm(X[i] - X[j])
    return eta

def initialize_probabilities(tau, eta, alpha, beta):
    numerator = np.power(tau, alpha) * np.power(eta, beta)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator / denominator

def update_pheromone(tau, delta_tau, rho):
    return (1 - rho) * tau + rho * delta_tau

def k_means_local_search(X,k=3):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X)
    return kmeans.labels_

def calculate_squared_error(X, partitions):
    distortion = 0.0
    for i, cluster_points in enumerate(partitions):
        if len(cluster_points) > 0:
            centroid = np.mean(X[cluster_points], axis=0)
            distortion += np.sum(np.linalg.norm(X[cluster_points] - centroid, axis=1) ** 2)
    return distortion

def calculate_B_values(X, partition):
    B_values = []
    for cluster in np.unique(partition):
        cluster_points = X[partition == cluster]
        centroid = np.mean(cluster_points, axis=0)
        B = np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
        B_values.append(B)
    return np.array(B_values)

def ant_based_clustering(X, M, tau0, tmax, alpha, beta, rho, S,k):
    n = X.shape[0]
    tau = initialize_pheromone(n, tau0)
    eta = initialize_local_heuristic(X)
    
    best_partition = None
    best_distortion = float('inf')

    for t in range(tmax):
        delta_tau = np.zeros((n, n))
        probabilities = initialize_probabilities(tau, eta, alpha, beta)

        for m in range(M):
            for _ in range(S):
                i = np.random.randint(n)
                j = np.random.choice(n, p=probabilities[i])

            current_partition = k_means_local_search(X,k)
            current_distortion = calculate_squared_error(X, [np.where(current_partition == c)[0] for c in np.unique(current_partition)])

            if current_distortion < best_distortion:
                best_distortion = current_distortion
                best_partition = current_partition

            B_values = calculate_B_values(X, current_partition)
            for i in range(n):
                for j in range(n):
                    if i != j and current_partition[i] == current_partition[j]:  
                        delta_tau[i, j] += B_values[current_partition[i]] / np.linalg.norm(X[i] - X[j])

        for i in range(n):
            for j in range(n):
                if i != j:
                    tau[i, j] = update_pheromone(tau[i, j], delta_tau[i, j], rho)

    return best_partition

def visualize_clusters(X, partition):
    unique_labels = np.unique(partition)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        cluster_points = X[partition == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {label}')

    plt.title('Ant-Based Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    cluster1_num_samples = 20
    cluster1_x1_start = 0
    cluster1_x1_end = 5
    cluster1_x2_start = 2
    cluster1_x2_end = 6
    cluster1_x1 = np.random.random(size=(cluster1_num_samples))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x2 = np.random.random(size=(cluster1_num_samples))
    cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

    cluster2_num_samples = 20
    cluster2_x1_start = 4
    cluster2_x1_end = 12
    cluster2_x2_start = 14
    cluster2_x2_end = 18
    cluster2_x1 = np.random.random(size=(cluster2_num_samples))
    cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
    cluster2_x2 = np.random.random(size=(cluster2_num_samples))
    cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

    cluster3_num_samples = 20
    cluster3_x1_start = 12
    cluster3_x1_end = 18
    cluster3_x2_start = 8
    cluster3_x2_end = 11
    cluster3_x1 = np.random.random(size=(cluster3_num_samples))
    cluster3_x1 = cluster3_x1 * (cluster3_x1_end - cluster3_x1_start) + cluster3_x1_start
    cluster3_x2 = np.random.random(size=(cluster3_num_samples))
    cluster3_x2 = cluster3_x2 * (cluster3_x2_end - cluster3_x2_start) + cluster3_x2_start

    c1 = np.array([cluster1_x1, cluster1_x2]).T
    c2 = np.array([cluster2_x1, cluster2_x2]).T
    c3 = np.array([cluster3_x1, cluster3_x2]).T

    data = np.concatenate((c1, c2, c3), axis=0)

    df = pd.DataFrame(data, columns=['X1', 'X2'])
    X=df[['X1','X2']].values
    M = 20
    tau0 = 0.001
    tmax = 20
    alpha = 1.0
    beta = 0.8
    rho = 0.3
    S = 2 * df.shape[0]

    best_partition = ant_based_clustering(X, M, tau0, tmax, alpha, beta, rho, S,3)

    visualize_clusters(X, best_partition)

