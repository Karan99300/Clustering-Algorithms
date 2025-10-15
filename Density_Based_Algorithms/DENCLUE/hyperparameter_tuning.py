import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from DENCLUE_method import DENCLUE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def denclue_based_tuning(
    X,
    k,
    h = [0.1, 0.2, 0.3, 0.4, 0.5],
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    min_density = [0.0, 0.5, 1.0, 2.0]
):
    best_metrics = []

    for h_val in h:
        for ep in eps:
            for min_density_val in min_density:
                try:
                    denclue = DENCLUE(
                        h_val, ep, min_density_val
                    )

                    denclue.fit(X)

                    clusters = denclue.labels_

                    silhouette_avg = silhouette_score(X, clusters)
                    db_index = davies_bouldin_score(X, clusters)
                    ch_index = calinski_harabasz_score(X, clusters)

                    best_metrics.append({
                        'silhouette': silhouette_avg,
                        'db_index': db_index,
                        'ch_index': ch_index,
                        'params': {
                            'h': h_val,
                            'eps': ep,
                            'min_density': min_density_val
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

        best_metrics = denclue_based_tuning(df_normalized.values, k_values[i])

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/denclue_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()