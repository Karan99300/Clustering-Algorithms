from aco import ant_based_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from tqdm import tqdm
from itertools import product

def ant_based_tuning(
    X,
    k,
    M_list=[10,20],
    tau0_list=[0.0005,0.001],
    tmax_list=[10, 20],
    alpha_list=[0.5, 1.0],
    beta_list=[0.5, 1.0],
    rho_list=[0.1, 0.3],
    S=None
):
    if S is None:
        S = 2 * X.shape[0]

    best_metrics = []

    for M, tau0, tmax, alpha, beta, rho in tqdm(
        product(M_list, tau0_list, tmax_list, alpha_list, beta_list, rho_list),
        total=len(M_list) * len(tau0_list) * len(tmax_list) * len(alpha_list) * len(beta_list) * len(rho_list)
    ):
        try:
            clusters = ant_based_clustering(
                X, M, tau0, tmax, alpha, beta, rho, S, k
            )
            print(clusters)

            # Evaluate clustering metrics
            silhouette_avg = silhouette_score(X, clusters)
            db_index = davies_bouldin_score(X, clusters)
            ch_index = calinski_harabasz_score(X, clusters)

            best_metrics.append({
                'silhouette': silhouette_avg,
                'db_index': db_index,
                'ch_index': ch_index,
                'params': {
                    'M': M,
                    'tau0': tau0,
                    'tmax': tmax,
                    'alpha': alpha,
                    'beta': beta,
                    'rho': rho,
                    'S': S
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

        best_metrics = ant_based_tuning(df.values, k_values[i])
        print(best_metrics)

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/ant_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()