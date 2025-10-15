from KMEANS import kmeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

def kmeans_tuning(X, n_clusters, max_iterations_list=[50, 100, 200, 500]):
    best_metrics = []

    for max_iter in max_iterations_list:
        # Run KMeans
        clusters, centroids = kmeans(X, n_clusters, max_iterations=max_iter)

        # Compute clustering metrics
        try:
            silhouette_avg = silhouette_score(X, clusters)
            db_index = davies_bouldin_score(X, clusters)
            ch_index = calinski_harabasz_score(X, clusters)
        except:
            continue  # skip degenerate cases

        best_metrics.append({
            'silhouette': silhouette_avg,
            'db_index': db_index,
            'ch_index': ch_index,
            'params': {'max_iterations': max_iter}
        })

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

        best_metrics = kmeans_tuning(df_normalized.values, k_values[i])

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })


    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/kmeans_tuning_results.csv", index=False)
    print("\nAll results saved to 'kmeans_tuning_results.csv'.")


if __name__ == "__main__":
    main()