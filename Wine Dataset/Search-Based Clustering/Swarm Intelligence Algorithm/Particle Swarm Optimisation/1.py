import numpy as np

from pso import particle_swarm_clustering,plot_clusters
num_points = 100

# Generate data for five clusters
cluster1 = np.random.normal(loc=[2, 2], scale=1, size=(num_points, 2))
cluster2 = np.random.normal(loc=[8, 2], scale=1, size=(num_points, 2))
cluster3 = np.random.normal(loc=[5, 5], scale=1, size=(num_points, 2))
cluster4 = np.random.normal(loc=[2, 8], scale=1, size=(num_points, 2))
cluster5 = np.random.normal(loc=[8, 8], scale=1, size=(num_points, 2))

# Combine data from all clusters
data = np.vstack([cluster1, cluster2, cluster3, cluster4, cluster5])
num_clusters = 5
num_particles = 10
max_iterations = 100
min_fitness = 1e-6
max_kmeans_iterations = 100
min_centroid_change = 1e-4
w, c1, c2 = 0.72, 1.49, 1.49

best_position, best_labels = particle_swarm_clustering(
    data, num_clusters, num_particles, max_iterations, min_fitness, max_kmeans_iterations, min_centroid_change,w,c1,c2
)

# Visualize clusters
plot_clusters(data, best_position,best_labels)

