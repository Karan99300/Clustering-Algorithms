import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from OPTICS import optics

def optics_based_tuning(
    X,
    k,
    minPts = [2, 5, 10, 15, 25, 50],
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
):
    best_metrics = []

    for minPt in minPts:
        for ep in eps:
            try:
                clusters = optics(X, minPt, ep)

                silhouette_avg = silhouette_score(X, clusters)
                db_index = davies_bouldin_score(X, clusters)
                ch_index = calinski_harabasz_score(X, clusters)

                best_metrics.append({
                    'silhouette': silhouette_avg,
                    'db_index': db_index,
                    'ch_index': ch_index,
                    'params': {
                        'eps': ep,
                        'minPts': minPt
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

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        df_normalized = normalize(df_scaled)

        df_normalized = pd.DataFrame(df_normalized)
        df_normalized.columns = df.columns

        print(f"\nProcessing file: {file} with k = {k_values[i]}")

        best_metrics = optics_based_tuning(df_normalized.values, k_values[i])

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/optics_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()