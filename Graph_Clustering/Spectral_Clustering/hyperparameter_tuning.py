from clustering import spectral_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
import numpy as np

def spectral_based_tuning(
    X, 
    k,
    n_neighbors = [2, 3, 5, 7, 8, 10]
):
    best_metrics = []

    for n_neighbor in n_neighbors:
        try:
            clusters = spectral_clustering(X, n_neighbor, k)
            silhouette_avg = silhouette_score(X, clusters)
            db_index = davies_bouldin_score(X, clusters)
            ch_index = calinski_harabasz_score(X, clusters)

            best_metrics.append({
                'silhouette': silhouette_avg,
                'db_index': db_index,
                'ch_index': ch_index,
                'params': {
                    'num_neighbors': n_neighbor
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

        best_metrics = spectral_based_tuning(df.values, k_values[i])

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/spectral_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()