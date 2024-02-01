import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def quantization_error(data, centroids, labels):
    total_error = 0
    for i in range(len(data)):
        total_error += np.linalg.norm(data[i] - centroids[labels[i]]) / len(data[labels[i]])
    return total_error / len(data)

def update_position_velocity(position, velocity, pbest, gbest, w, c1, c2):
    new_velocity = w * velocity + c1 * np.random.rand() * (pbest - position) + c2 * np.random.rand() * (gbest - position)
    new_position = position + new_velocity
    return new_position, new_velocity

def particle_swarm_clustering(data, num_clusters, num_particles, max_iterations, min_fitness, max_kmeans_iterations, min_centroid_change, w, c1, c2):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_kmeans_iterations, tol=min_centroid_change, n_init='auto')
    kmeans.fit(data)
    best_kmeans_solution = kmeans.cluster_centers_

    particles = [np.random.rand(num_clusters, data.shape[1]) for _ in range(num_particles - 1)]
    particles.append(best_kmeans_solution)

    velocities = [np.zeros_like(particle) for particle in particles]

    pbest_positions = particles.copy()
    pbest_labels = [np.argmin(np.linalg.norm(data[:, np.newaxis, :] - particles[i], axis=2), axis=1) for i in range(num_particles)]
    pbest_fitness = [quantization_error(data, particles[i], pbest_labels[i]) for i in range(num_particles)]

    gbest_index = np.argmin(pbest_fitness)
    gbest_position = particles[gbest_index].copy()
    gbest_labels = pbest_labels[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]

    for _ in range(max_iterations):
        for i in range(num_particles):
            # Update position and velocity
            particles[i], velocities[i] = update_position_velocity(
                particles[i], velocities[i], pbest_positions[i], gbest_position, w, c1, c2
            )

            # Calculate fitness based on the particle's centroids
            labels = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - particles[i], axis=2), axis=1)
            fitness = quantization_error(data, particles[i], labels)

            if fitness < pbest_fitness[i]:
                pbest_positions[i] = particles[i].copy()
                pbest_labels[i] = labels
                pbest_fitness[i] = fitness

            if fitness < gbest_fitness:
                gbest_position = particles[i].copy()
                gbest_labels = labels
                gbest_fitness = fitness

    return gbest_position, gbest_labels

def plot_clusters(data, centroids, labels, title="Clusters"):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage

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

    num_clusters = 3
    num_particles = 10
    max_iterations = 100
    min_fitness = 1e-6
    max_kmeans_iterations = 100
    min_centroid_change = 1e-4
    w, c1, c2 = 0.72, 1.49, 1.49

    best_position, best_labels = particle_swarm_clustering(
        X, num_clusters, num_particles, max_iterations, min_fitness, max_kmeans_iterations, min_centroid_change,w,c1,c2
    )

    plot_clusters(X, best_position, best_labels, title="PSO-K-Means Clustering")

if __name__ == '__main__':
    main()