from pso import particle_swarm_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
from tqdm import tqdm
from itertools import product
import numpy as np

def particle_based_tuning(
    X, 
    k,
    num_particles = [5, 10, 20, 25],
    max_iterations = [100, 200],
    min_fitness = [1e-6, 1e-5],
    max_kmeans_iterations = [100, 200],
    min_centroid_change = [1e-4, 1e-3],
    w = [0.7, 0.8], 
    c1 = [1.5, 1.6, 1.7], 
    c2 = [1.5, 1.6, 1.7]
):
    best_metrics = []

    for num_particle, max_iteration, min_fitness_val, max_kmeans_iteration, min_centroid_change_val, w_val, c1_val, c2_val in tqdm(
        product(num_particles, max_iterations, min_fitness, max_kmeans_iterations, min_centroid_change, w, c1, c2),
        total=len(num_particles) * len(max_iterations) * len(min_fitness) * len(max_kmeans_iterations) * len(min_centroid_change) * len(w) * len(c1) * len(c2)
    ):
        try: 
            best_position, clusters = particle_swarm_clustering(
                X, k, num_particle, max_iteration, min_fitness_val, max_kmeans_iteration, min_centroid_change_val,
                w_val, c1_val, c2_val
            )
            print(clusters)

            silhouette_avg = silhouette_score(X, clusters)
            db_index = davies_bouldin_score(X, clusters)
            ch_index = calinski_harabasz_score(X, clusters)
            best_metrics.append({
                'silhouette': silhouette_avg,
                'db_index': db_index,
                'ch_index': ch_index,
                'params': {
                    'num_particles': num_particle,
                    'max_iterations': max_iteration,
                    'min_fitness': min_fitness_val,
                    'max_kmeans_iteration': max_kmeans_iteration,
                    'min_centroid_change': min_centroid_change_val,
                    'w': w_val,
                    'c1': c1_val,
                    'c2': c2_val
                }
            })

        except Exception:
            continue 
    return best_metrics

def main():
    folder_path = "/mnt/private/Clustering-Algorithms/csvs"  
    k_values = [2, 6, 2, 3, 3]  
   
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    if len(k_values) != len(csv_files):
        print("Error: The number of k values must match the number of CSV files.")
        return

    results = []

    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # scaler = StandardScaler()
        # df_scaled = scaler.fit_transform(df)
        # df_normalized = normalize(df_scaled)

        # df_normalized = pd.DataFrame(df_normalized)
        # df_normalized.columns = df.columns

        print(f"\nProcessing file: {file} with k = {k_values[i]}")

        best_metrics = particle_based_tuning(df.values, k_values[i])
        print(best_metrics)

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/pso_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()