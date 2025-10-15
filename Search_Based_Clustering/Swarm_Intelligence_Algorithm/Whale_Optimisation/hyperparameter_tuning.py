from wao import whale_optimization_algorithm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

def whale_based_tuning(
    X, 
    k,
    num_agents = [5, 10, 25, 50], 
    num_iterations = [50, 100, 500]
):
    best_metrics = []

    for num_agent, num_iteration in tqdm(
        product(num_agents, num_iterations),
        total=len(num_agents)*len(num_iterations)
    ):
            try:
                best_agent = whale_optimization_algorithm(X, k, num_agent, num_iteration)
                print(best_agent)
                clusters = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - best_agent['cluster_centers'], axis=2), axis=1)
                print(clusters)

                silhouette_avg = silhouette_score(X, clusters)
                db_index = davies_bouldin_score(X, clusters)
                ch_index = calinski_harabasz_score(X, clusters)

                best_metrics.append({
                    'silhouette': silhouette_avg,
                    'db_index': db_index,
                    'ch_index': ch_index,
                    'params': {
                        'num_agents': num_agent,
                        'num_iterations': num_iteration
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

        best_metrics = whale_based_tuning(df.values, k_values[i])
        print(best_metrics)

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/whale_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()