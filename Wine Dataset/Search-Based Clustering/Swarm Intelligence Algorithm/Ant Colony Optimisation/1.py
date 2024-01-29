import numpy as np

from aco import ant_based_clustering,visualize_clusters
num_points = 100

# Generate data for five clusters
cluster1 = np.random.normal(loc=[2, 2], scale=1, size=(num_points, 2))
cluster2 = np.random.normal(loc=[8, 2], scale=1, size=(num_points, 2))
cluster3 = np.random.normal(loc=[5, 5], scale=1, size=(num_points, 2))
cluster4 = np.random.normal(loc=[2, 8], scale=1, size=(num_points, 2))
cluster5 = np.random.normal(loc=[8, 8], scale=1, size=(num_points, 2))

# Combine data from all clusters
data = np.vstack([cluster1, cluster2, cluster3, cluster4, cluster5])
M = 20
tau0 = 0.001
tmax = 20
alpha = 1.0
beta = 0.8
rho = 0.3
S = 2 * data.shape[0]

# Run the ant-based clustering algorithm
best_partition = ant_based_clustering(data, M, tau0, tmax, alpha, beta, rho, S,5)

# Visualize clusters
visualize_clusters(data, best_partition)

