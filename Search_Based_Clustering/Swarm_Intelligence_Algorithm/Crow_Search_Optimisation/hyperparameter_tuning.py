from cso import optimize_crows
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

def crow_based_tuning(
    df,
    k, 
    num_cols,
    max_iterations = [100, 200, 500],
    awareness_probability = [np.random.rand()],
    num_crows = [5,10, 15, 20, 25]
):
    max_ranges = []
    max_ranges.append(df.max().values)
    X = df.values
    best_metrics = []

    for max_iteration, awareness_prob, max_range, num_crow in tqdm(
        product(max_iterations, awareness_probability, max_ranges, num_crows), 
        total=len(max_iterations)*len(awareness_probability)*len(max_ranges)*len(num_crows)
    ):
        try:
            best_crow = optimize_crows(X, max_iteration, awareness_prob, max_range, num_crow, k, num_cols)
            clusters = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - best_crow, axis=2), axis=1)

            silhouette_avg = silhouette_score(X, clusters)
            db_index = davies_bouldin_score(X, clusters)
            ch_index = calinski_harabasz_score(X, clusters)

            best_metrics.append({
                'silhouette': silhouette_avg,
                'db_index': db_index,
                'ch_index': ch_index,
                'params': {
                    'max_iterations': max_iteration,
                    'awareness_probability': awareness_prob,
                    'max_ranges': max_range,
                    'num_crows': num_crow
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

        best_metrics = crow_based_tuning(df, k_values[i], df.shape[1])
        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/crow_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()