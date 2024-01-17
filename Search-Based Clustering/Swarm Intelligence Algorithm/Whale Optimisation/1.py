import numpy as np

from wao import whale_optimization_algorithm,plot_final_clusters
num_points = 100

# Generate data for five clusters
cluster1 = np.random.normal(loc=[2, 2], scale=1, size=(num_points, 2))
cluster2 = np.random.normal(loc=[8, 2], scale=1, size=(num_points, 2))
cluster3 = np.random.normal(loc=[5, 5], scale=1, size=(num_points, 2))
cluster4 = np.random.normal(loc=[2, 8], scale=1, size=(num_points, 2))
cluster5 = np.random.normal(loc=[8, 8], scale=1, size=(num_points, 2))
data = np.vstack([cluster1, cluster2, cluster3, cluster4, cluster5])

best_agent=whale_optimization_algorithm(data,5,10,100)
plot_final_clusters(data,best_agent)