import numpy as np
from cso import optimize_crows,visualize_clusters
num_points = 100

# Generate data for five clusters
cluster1 = np.random.normal(loc=[2, 2], scale=1, size=(num_points, 2))
cluster2 = np.random.normal(loc=[8, 2], scale=1, size=(num_points, 2))
cluster3 = np.random.normal(loc=[5, 5], scale=1, size=(num_points, 2))
cluster4 = np.random.normal(loc=[2, 8], scale=1, size=(num_points, 2))
cluster5 = np.random.normal(loc=[8, 8], scale=1, size=(num_points, 2))
data = np.vstack([cluster1, cluster2, cluster3, cluster4, cluster5])

num_clusters = 5
num_crows = 100
num_dimensions = 2
max_range = 15
max_iterations = 1000
awareness_probability = np.random.rand()

best_crow=optimize_crows(max_iterations,awareness_probability,max_range,data,num_crows,num_clusters,num_dimensions)
                
assignments = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - best_crow, axis=2), axis=1)
visualize_clusters(data, best_crow, assignments)