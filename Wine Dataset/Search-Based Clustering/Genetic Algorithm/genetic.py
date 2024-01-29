import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad
def euclidean_distance(X, Y):
    return np.sqrt(np.sum(np.power(X - Y, 2), axis=1))

def cluster_data(solution, solution_idx,num_clusters,feature_vector_length,df):
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length * clust_idx:feature_vector_length * (clust_idx + 1)])
        cluster_center_dists = euclidean_distance(df.values, cluster_centers[clust_idx])
        all_clusters_dists.append(np.array(cluster_center_dists))

    cluster_centers = np.array(cluster_centers)
    all_clusters_dists = np.array(all_clusters_dists)

    cluster_indices = np.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_clusters):
        clusters.append(np.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = np.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist

def fitness_func(ga_instance,solution, solution_idx):
    _, _, _, _, clusters_sum_dist = cluster_data(solution, solution_idx,num_clusters,feature_vector_length,df)

    fitness = 1.0 / (np.sum(clusters_sum_dist) + 0.00000001)

    return fitness

def plot_clusters(df, cluster_labels):
    plt.scatter(df['X1'], df['X2'], c=cluster_labels, cmap='viridis', edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def main():
    global df,num_clusters,feature_vector_length
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
    df= pd.DataFrame(data, columns=['X1', 'X2'])
    num_clusters = 3
    feature_vector_length = df.shape[1]
    num_genes = num_clusters * feature_vector_length

    ga_instance = pygad.GA(num_generations=100,
                        sol_per_pop=10,
                        init_range_low=0,
                        init_range_high=20,
                        num_parents_mating=5,
                        keep_parents=2,
                        num_genes=num_genes,
                        fitness_func=fitness_func,
                        suppress_warnings=True)

    ga_instance.run()

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Best solution is {bs}".format(bs=best_solution))
    print("Fitness of the best solution is {bsf}".format(bsf=best_solution_fitness))
    print("Best solution found after {gen} generations".format(gen=ga_instance.best_solution_generation))

    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx,num_clusters,feature_vector_length,df)

    # Display the final clusters
    plot_clusters(df, cluster_indices)

if __name__ == "__main__":
    main()