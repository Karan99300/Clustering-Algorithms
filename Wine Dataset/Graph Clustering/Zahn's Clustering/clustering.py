import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score

# Function to calculate Euclidean distance
def euclid_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to compute Euclidean Minimum Spanning Tree
def euclid_MST(X):
    G = nx.Graph()
    n = len(X)

    for i in tqdm(range(n), desc="Computing EMST"):
        for j in range(i + 1, n):
            dist = euclid_distance(X.iloc[i], X.iloc[j])
            G.add_edge(i, j, weight=dist)

    return nx.minimum_spanning_tree(G)

# Function to sort edges in descending order of weight
def sort_edges(G):
    return sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

# Function to delete edges from the MST
def del_edges(G, k):
    edges_to_del = sort_edges(G)[:k - 1]
    G.remove_edges_from(edges_to_del)

# Function to assign clusters based on connected components
def clusters(G, df):
    cluster_assignments = np.zeros(len(df), dtype=np.int32)
    for cluster_number, cluster in enumerate(nx.connected_components(G)):
        cluster_indices = np.array(list(cluster), dtype=np.int32)
        cluster_assignments[cluster_indices] = cluster_number

    return cluster_assignments

# Zahn's Clustering Algorithm
def Zahns_Clustering(df, k):
    emst = euclid_MST(X=df)
    del_edges(G=emst, k=k)
    return clusters(G=emst, df=df)

def main():
    cluster1_num_samples = 20
    cluster1_x1_start = 0
    cluster1_x1_end = 5
    cluster1_x2_start = 2
    cluster1_x2_end = 6
    cluster1_x1 = np.random.random(size=(cluster1_num_samples))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x2 = np.random.random(size=(cluster1_num_samples))
    cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

    # Cluster 2
    cluster2_num_samples = 20
    cluster2_x1_start = 4
    cluster2_x1_end = 12
    cluster2_x2_start = 14
    cluster2_x2_end = 18
    cluster2_x1 = np.random.random(size=(cluster2_num_samples))
    cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
    cluster2_x2 = np.random.random(size=(cluster2_num_samples))
    cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

    # Cluster 3
    cluster3_num_samples = 20
    cluster3_x1_start = 12
    cluster3_x1_end = 18
    cluster3_x2_start = 8
    cluster3_x2_end = 11
    cluster3_x1 = np.random.random(size=(cluster3_num_samples))
    cluster3_x1 = cluster3_x1 * (cluster3_x1_end - cluster3_x1_start) + cluster3_x1_start
    cluster3_x2 = np.random.random(size=(cluster3_num_samples))
    cluster3_x2 = cluster3_x2 * (cluster3_x2_end - cluster3_x2_start) + cluster3_x2_start

    # Combine clusters into a single dataset
    c1 = np.array([cluster1_x1, cluster1_x2]).T
    c2 = np.array([cluster2_x1, cluster2_x2]).T
    c3 = np.array([cluster3_x1, cluster3_x2]).T

    data = np.concatenate((c1, c2, c3), axis=0)
    df = pd.DataFrame(data, columns=['X1', 'X2'])
    # Example usage
    k_value = 3  # You can adjust the value of k as needed
    df['cluster'] = Zahns_Clustering(df, k_value)
    plot_clusters(df, df['cluster'])
    silhouette_avg = silhouette_score(df[['X1', 'X2']], df['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")

# Visualize clusters
def plot_clusters(df, cluster_labels):
    plt.scatter(df['X1'], df['X2'], c=cluster_labels, cmap='viridis', edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Zahn\'s Clustering')
    plt.show()


