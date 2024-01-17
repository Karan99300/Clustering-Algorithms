import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def initialize_crows_positions(num_crows, num_clusters, num_dimensions, max_ranges):
    positions = np.random.uniform(low=0, high=1, size=(num_crows, num_clusters, num_dimensions))

    for dim in range(num_dimensions):
        positions[:, :, dim] *= max_ranges[dim]

    return positions

def calculate_fitness(data, cluster_centers):
    distances = np.zeros((len(data), len(cluster_centers)))
    for i, point in enumerate(data):
        distances[i, :] = np.linalg.norm(point - cluster_centers, axis=1)
    min_distances = np.min(distances, axis=1)
    fitness_value = np.sum(min_distances)
    return fitness_value

def evaluate_fitness(data, crow_positions):
    fitness = []
    for crow_position in crow_positions:
        fitness_val = calculate_fitness(data, crow_position)
        fitness.append(fitness_val)

    return np.array(fitness)

def update_positions(crows_positions, memory_positions, awareness_probability, flight_length, max_ranges):
    num_crows, num_clusters, num_dimensions = crows_positions.shape

    for v in range(num_crows):
        mu = np.random.randint(num_crows)
        if np.random.rand() >= awareness_probability:
            crows_positions[v] = crows_positions[v] + np.random.rand() * flight_length * (memory_positions[mu] - crows_positions[v])
            
            # Ensure the updated positions are within the specified ranges
            for dim in range(num_dimensions):
                crows_positions[v, :, dim] = np.clip(crows_positions[v, :, dim], 0, max_ranges[dim])
        else:
            crows_positions[v] = np.random.uniform(low=0, high=1, size=(num_clusters, num_dimensions))
            
            # Scale the randomly initialized positions
            for dim in range(num_dimensions):
                crows_positions[v, :, dim] *= max_ranges[dim]

    return crows_positions

def optimize_crows(max_iterations, awareness_probability, max_ranges, data, num_crows, num_clusters, num_dimensions):
    crows_positions = initialize_crows_positions(num_crows, num_clusters, num_dimensions, max_ranges)
    memory_positions = np.copy(crows_positions)
    memory_fitness = evaluate_fitness(data, crows_positions)

    for iteration in range(max_iterations):
        flight_length = 2 * np.exp(-((iteration / max_iterations) ** 2) * np.pi)
        crows_positions = update_positions(crows_positions, memory_positions, awareness_probability, flight_length, max_ranges)
        current_fitness = evaluate_fitness(data, crows_positions)

        for i, val in enumerate(current_fitness):
            if current_fitness[i] < memory_fitness[i]:
                memory_fitness[i] = current_fitness[i]
                memory_positions[i] = crows_positions[i]

    best_fitness_index = np.argmin(memory_fitness)
    best_crow = memory_positions[best_fitness_index]

    return best_crow

def visualize_clusters(data, cluster_centers, assignments):
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', marker='o', edgecolors='k')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Crow Search Algorithm - Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

def main():
    num_clusters = 3
    num_crows = 10
    num_dimensions = 2
    max_range = 20  
    max_iterations = 100
    awareness_probability = np.random.rand()

    cluster1_num_samples = 20
    cluster1_x1_start, cluster1_x1_end = 0, 5
    cluster1_x2_start, cluster1_x2_end = 2, 6
    cluster1_x1 = np.random.random(size=(cluster1_num_samples))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x2 = np.random.random(size=(cluster1_num_samples))
    cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

    cluster2_num_samples = 20
    cluster2_x1_start, cluster2_x1_end = 4, 12
    cluster2_x2_start, cluster2_x2_end = 14, 18
    cluster2_x1 = np.random.random(size=(cluster2_num_samples))
    cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
    cluster2_x2 = np.random.random(size=(cluster2_num_samples))
    cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

    cluster3_num_samples = 20
    cluster3_x1_start, cluster3_x1_end = 12, 18
    cluster3_x2_start, cluster3_x2_end = 8, 11
    cluster3_x1 = np.random.random(size=(cluster3_num_samples))
    cluster3_x1 = cluster3_x1 * (cluster3_x1_end - cluster3_x1_start) + cluster3_x1_start
    cluster3_x2 = np.random.random(size=(cluster3_num_samples))
    cluster3_x2 = cluster3_x2 * (cluster3_x2_end - cluster3_x2_start) + cluster3_x2_start

    c1 = np.array([cluster1_x1, cluster1_x2]).T
    c2 = np.array([cluster2_x1, cluster2_x2]).T
    c3 = np.array([cluster3_x1, cluster3_x2]).T

    data = np.concatenate((c1, c2, c3), axis=0)
    df = pd.DataFrame(data, columns=['X1', 'X2'])
    max_ranges=df.max().values
    
    best_crow=optimize_crows(max_iterations,awareness_probability,max_ranges,df.values,num_crows,num_clusters,num_dimensions)
                
    assignments = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - best_crow, axis=2), axis=1)
    visualize_clusters(data, best_crow, assignments)
if __name__ == "__main__":
    main()
