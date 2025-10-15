import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from autoencoder import AutoEncoder
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd 
import os

def autoencoder_tuning(X_train, X_test, n_clusters, device='cpu', lr_list=[1e-3, 1e-4], bottleneck_list=[5, 10, 20], num_epochs=100, batch_size=64):
    
    tensor_train = torch.FloatTensor(X_train).to(device)
    tensor_test=torch.FloatTensor(X_test).to(device)
    inputs_dim = X_train.shape[1]
    
    best_metrics = []
    
    for n_bottleneck in bottleneck_list:
        for lr in lr_list:
            model = AutoEncoder(inputs_dim, n_bottleneck).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            train_loader = DataLoader(TensorDataset(tensor_train, tensor_train), batch_size=64, shuffle=True)
            # Training loop
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_x, _ in train_loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Encode data
            model.eval()
            with torch.no_grad():
                encoded = model.encoder(tensor_test).cpu().numpy()
            
            # UMAP reduction
            n_umap = max(2, n_bottleneck // 2)  # Ensure at least 2 components
            umap_data = umap.UMAP(n_components=n_umap, metric='euclidean', n_neighbors=50, min_dist=0.0, random_state=13).fit_transform(encoded)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(umap_data)
            
            # Compute metrics
            silhouette_avg = silhouette_score(umap_data, labels)
            db_index = davies_bouldin_score(umap_data, labels)
            ch_index = calinski_harabasz_score(umap_data, labels)

            best_metrics.append({
                'silhouette': silhouette_avg,
                'db_index': db_index,
                'ch_index': ch_index,
                'bottleneck': n_bottleneck,
                'lr': lr
            })
    
    return best_metrics['silhouette'], best_metrics['db_index'], best_metrics['ch_index']

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

        best_metrics = autoencoder_tuning(df.values, k_values[i])

        for metrics in best_metrics:
            results.append({
                'file': file,
                'num_clusters': k_values[i],
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("/mnt/private/Clustering-Algorithms/hyperparameter_tuning/autoencoder_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()